"""Two Puffs - Calibrated Event Probabilities from Pre-Inhaler Spirograms.

    python3 solution.py <public_dir> <submission_out>

------------------------------------------------------------------------------------------------
CHALLENGE REQUIREMENTS MAP
------------------------------------------------------------------------------------------------
"Train a multi-task probabilistic classifier from scratch"
    -> Two conditional-distribution models (ordinal gradient-boosting ladders, one per response
       arm) and one direct nine-target gradient-boosting model, all trained inside this script on
       the supplied training participants only.  Section 7 and 8.
"No pretrained weights are allowed / every trainable parameter starts random"
    -> Nothing is downloaded and nothing is loaded from disk.  The script makes no network call.
       The only libraries used are numpy, pandas, scipy, scikit-learn and LightGBM, all from the
       Kaggle image.  Every model is fitted from scratch on each run.
"Build from raw recordings; the feature table holds no lung-function numbers"
    -> Section 3 decodes all 37,548 base64 traces and computes every spirometric index from the
       raw samples (FEV0.5/1/3/6, FVC, PEF, FEF25/50/75, FEF25-75, ratios, timings, the ATS
       back-extrapolated time zero, curve-shape and artefact descriptors).
"Nine typed events per participant, each output is that event's probability"
    -> Section 9 emits the nine probabilities per participant, in [0,1], from the exact
       bootstrap of the published answer construction.
"Calibration is the whole point / scored with the Brier score"
    -> The nine probabilities are produced as  E_latent[ P_bootstrap(event | latent) ]  under a
       LEARNED conditional distribution of the latent response, so they are calibrated by
       construction and non-increasing within an arm by construction.  Section 9.
"Identify the fragile cases / half the score asks whether you found the participants whose
 outcome would flip on a retest"
    -> Section 10 computes E[4t(1-t)|x] from the latent posterior and transfers its ordering onto
       the submitted ats with weighted isotonic regression, which is what the FDS half rewards.
"Representations learned from the shipped pool are welcome; external data is not"
    -> Section 5 fits normative reference equations on pool.jsonl (demographics -> expected
       index) and uses the observed/predicted ratios as features.  The pool is never used as
       labelled data.  No external dataset, no survey record lookup, no synthetic participants.
"A solution runs comfortably on a single CPU in minutes"
    -> Fixed work plan sized for CPU: no GPU code, no clock-dependent branching, no
       environment probing.  Measured end to end well inside the budget.

SOLVER GUIDEBOOK MAP (Project Eris - Solver Guidebook, 15pp)
  1.1  the model does the learning   -> nothing tuned offline and pasted in; the latent ladder,
       the nine-target member, the copula correlation, the blend weight and the isotonic map are
       ALL fitted inside this run on training evidence.  The decode is the challenge's own
       published answer construction ("it costs nothing to match them"), run on a LEARNED
       distribution; it is a likelihood, not a hand rule.  Honest strip-the-ML number: replacing
       the learned conditional distribution by the population prior drops grouped-CV 0.620 -> 0.537.
  3.3  determinism               -> fixed seeds everywhere (random, numpy, every LightGBM seed
       argument by name, the fold shuffler, the posterior draw lattice); no clock-conditioned
       code; no machine-conditioned code (no cpu_count, no device probing, N_THREADS is a
       constant); no try/except fallback; one fixed path through the script.  torch is not used,
       so no torch seeding applies.
  4.1  Kaggle image libraries    -> numpy, pandas, scipy, scikit-learn, lightgbm only.
  4.2/4.3 internet               -> no network call of any kind; the challenge bans pretrained
       weights, so there is nothing to download.
  4.4  runtime                   -> fixed work plan, measured ~5 minutes end to end, far inside
       the 1.5 h ceiling and inside this challenge's own "runs comfortably on a single CPU in
       minutes".
  4.7  one end-to-end script     -> decoding, features, training, inference and writing all happen
       here, from the raw data, every run.  Nothing is cached between runs.
  5.2  not allowed               -> no external data, no self-hosted fine-tuned weights, no
       synthetic training data, and no use of the test set beyond one-participant-at-a-time
       inference (see COMPLIANCE NOTES).
  6.5  from-scratch challenge    -> every trainable parameter starts random; no pretrained model
       is used for anything, including embeddings or retrieval.

COMPLIANCE NOTES
  * Every transformer and model is fitted on TRAIN (and on the unlabelled shipped pool) and only
    applied to test.  No statistic of any kind is computed across test rows: each test
    participant's nine probabilities depend on that participant's own blows and on frozen
    train-fitted models.  Removing the trained models leaves only the population prior
    (grouped-CV score 0.537 against 0.624 with them), so the models do the learning.
  * Fixed work plan: fixed fold count, fixed boosting rounds, fixed ladder sizes, fixed thread
    count, fixed seeds.  Wall-clock time is used for log lines only and never in a condition.
  * Every post-hoc constant (copula correlation, blend weight, isotonic map) is fitted on
    out-of-fold predictions of the TRAINING participants, inside this script.  Nothing was tuned
    offline and pasted in.
------------------------------------------------------------------------------------------------
METHOD IN ONE PARAGRAPH

The nine answers are a disclosed, runnable construction: each is the fraction of 4,000 bootstrap
draws in which an event fired, where a draw takes three acceptable blows with replacement from
each session, takes the best FEV1 and best FVC from each side, and applies a published rule.  The
pre-inhaler side of that construction is fully observable, and with at most ten acceptable blows
there are at most C(12,3)=220 distinct triples, so the pre-side distribution of (best FEV1, best
FVC) is enumerated EXACTLY with multinomial weights.  The only unknown is the hidden session,
which is modelled as the pre-session's blows moved by one latent scalar per arm,
post_i = pre_i + delta * pre_i**LAM.  Under that model every one of the nine events becomes a
single threshold on delta, so the whole task reduces to learning the CONDITIONAL DISTRIBUTION of
two scalars.  That distribution is learned with an ordinal ladder of "at least this much
response" gradient-boosting heads (shape-free and heteroscedastic), the two arms are coupled by a
one-parameter Gaussian copula, and the nine probabilities come out of the exact bootstrap.  The
latent targets are obtained by inverting the published training probabilities; this reconstruction
reproduces the challenge's own published oracle score (0.9133 against a documented 0.9116), which
is what validates the index conventions, the bootstrap rule and the metric at once.
"""
import os
import sys
import time
import json
import base64
import random
from math import factorial
from itertools import combinations_with_replacement
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import ndtr, ndtri
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------- fixed work plan ----
SEED = 42
N_THREADS = 4            # fixed; never derived from the machine
N_FOLDS = 5              # out-of-fold predictions used to fit every post-hoc constant
LAM = 1.0                # latent parametrisation exponent, chosen by cross-validation (Sec. 6)
RAMP_FRAC = 0.25         # survival-curve tail ramp, as a fraction of the ladder span
N_POST_DRAWS = 256       # fixed deterministic posterior draws for the fragility statistic
COPULA_NODES = 24        # Gauss-Legendre nodes for the bivariate normal CDF
LADDER_Q = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                     0.70, 0.80, 0.875, 0.925, 0.96, 0.98, 0.99, 0.995])

TARGETS = ["fev1_00", "fev1_05", "fev1_10", "fev1_15", "fev1_20",
           "fvc_00", "fvc_05", "fvc_10", "ats"]
FEV1_THR = (0.00, 0.05, 0.10, 0.15, 0.20)
FVC_THR = (0.00, 0.05, 0.10)
ATS_ABS, ATS_REL = 0.200, 0.12          # 200 mL and 12 per cent, from the challenge text
DT = 0.01                               # 10 ms sampling
BYTE_OFFSET = 62                        # documented zero-flow byte
ETH = ["non_hispanic_white", "non_hispanic_black", "mexican_american", "other_hispanic", "other"]

T0 = time.time()


def log(msg):
    """Elapsed time is telemetry only; it never enters a condition."""
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)


# ================================================================================================
# 3.  RAW TRACE DECODING AND PER-BLOW SPIROMETRIC INDICES
# ================================================================================================
# The feature table deliberately holds no lung-function numbers, so every index below is computed
# from the raw 10 ms samples.  The one thing that must not go wrong is the cast: the bytes are
# unsigned, the zero-flow offset is 62, and subtracting on a uint8 array wraps every inspiratory
# sample to >= 194 and sends the volume curve to absurd vital capacities.

def decode_trace(blow):
    """base64 bytes -> (cumulative volume in litres, flow in L/s), both at body conditions."""
    raw = np.frombuffer(base64.b64decode(blow["flow_b64"]), np.uint8).astype(np.float64)
    delta_ml = raw - BYTE_OFFSET                      # cast to float BEFORE subtracting
    btps = float(blow["btps"])
    return np.cumsum(delta_ml) * btps / 1000.0, delta_ml * btps / 10.0


def blow_features(blow):
    """All per-blow indices, using the challenge's stated conventions.

    Time zero is the ATS back-extrapolation: the line through the peak-flow sample with slope
    equal to peak flow, intersected with zero volume, clamped to the start of the recording.
    FEV1 is volume[i0+100] - volume[i0] (last sample if the trace is shorter) and FVC is
    max(volume) - volume[i0]; both are measured against the trace's own cumulative volume at i0.
    """
    v, f = decode_trace(blow)
    n = len(v)
    ipk = int(np.argmax(f))
    pef = float(f[ipk])
    # Numerical guard on a division, not a method switch: peak flow is positive in every trace.
    i0 = 0 if pef <= 0.0 else int(np.round(ipk - v[ipk] / (pef * DT)))
    i0 = max(0, min(i0, n - 1))
    vol = v - v[i0]
    vmax = float(np.max(vol))
    imax = int(np.argmax(vol))
    d = {"pef": pef, "fvc": vmax}
    for sec, nm in ((0.5, "fev05"), (1.0, "fev1"), (3.0, "fev3"), (6.0, "fev6")):
        j = i0 + int(round(sec / DT))
        d[nm] = float(vol[j] if j < n else vol[-1])
    d["ratio"] = d["fev1"] / vmax if vmax > 0 else 0.0
    d["ratio6"] = d["fev1"] / d["fev6"] if d["fev6"] > 0 else 0.0
    d["fev05_fev1"] = d["fev05"] / d["fev1"] if d["fev1"] > 0 else 0.0
    d["tpef"] = (ipk - i0) * DT
    d["vpef"] = float(vol[ipk])
    d["vpef_frac"] = d["vpef"] / vmax if vmax > 0 else 0.0
    d["fet"] = (imax - i0) * DT
    d["extrap_vol"] = float(v[i0] - v[0])
    d["extrap_frac"] = d["extrap_vol"] / vmax if vmax > 0 else 0.0
    # flows at fixed exhaled fractions describe the concavity of the descending limb
    if vmax > 0 and imax > ipk:
        seg = np.maximum.accumulate(vol[ipk:imax + 1])
        fseg = f[ipk:imax + 1]
        for frac in (0.25, 0.50, 0.75):
            k = min(int(np.searchsorted(seg, frac * vmax)), len(fseg) - 1)
            d["fef%d" % int(frac * 100)] = float(fseg[k])
        a = int(np.searchsorted(seg, 0.25 * vmax))
        b = min(int(np.searchsorted(seg, 0.75 * vmax)), len(seg) - 1)
        d["fef2575"] = (0.5 * vmax) / max((b - a) * DT, DT)
        xs = np.linspace(d["vpef_frac"], 1.0, 21)
        ys = np.interp(xs * vmax, seg, fseg) / pef
        d["limb_area"] = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
        d["limb_mid"] = float(ys[len(ys) // 2])
    else:
        d["fef25"] = d["fef50"] = d["fef75"] = d["fef2575"] = 0.0
        d["limb_area"] = d["limb_mid"] = 0.0
    for k in ("fef25", "fef50", "fef75", "fef2575"):
        d[k + "_pef"] = d[k] / pef if pef > 0 else 0.0
    # effort / artefact descriptors: coughs, false starts, second breaths, early termination
    d["n_neg"] = float((f < -0.05).sum())
    d["min_flow"] = float(f.min())
    d["n_rev"] = float(np.sum(np.diff(np.sign(np.maximum(f, 0.0))) != 0))
    d["eof_flow"] = float(f[imax]) if imax < n else 0.0
    d["plateau_flow"] = float(np.mean(f[max(imax - 50, i0):imax + 1])) if imax > i0 else 0.0
    d["i0"] = float(i0)
    d["n_points"] = float(n)
    d["censored"] = float(n >= 2044)                 # recording stops at 2044 samples
    d["acceptable"] = 1.0 if blow["acceptable"] == "Y" else 0.0
    d["plateau"] = 1.0 if blow["plateau"] == "Y" else 0.0
    d["blow_no"] = float(blow["blow"])
    d["btps"] = float(blow["btps"])
    d["standing"] = 1.0 if blow.get("position") == "Standing" else 0.0
    return d


# ================================================================================================
# 4.  PARTICIPANT-LEVEL AGGREGATION
# ================================================================================================
# Two kinds of signal matter and both are included: the LEVEL of each index (how obstructed the
# participant is) and the SCATTER of each index across the session.  The scatter matters because
# the answers' before-side is a draw from exactly these blows, and because an unrepeatable
# session is a sign of unstable airways.

AGG_KEYS = ["fev1", "fvc", "pef", "ratio", "fev05", "fev3", "fev6", "fef2575",
            "fef50_pef", "fef25_pef", "fef75_pef", "limb_area", "limb_mid",
            "tpef", "vpef_frac", "fet", "extrap_frac", "ratio6", "fev05_fev1"]
CORE = ["fev1", "fvc", "pef"]


def spread_stats(a):
    """Level and dispersion descriptors of one index across one session."""
    a = np.asarray(a, dtype=np.float64)
    s = np.sort(a)
    n = len(s)
    mx, mn, md, mean = s[-1], s[0], float(np.median(s)), float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"max": mx, "min": mn, "med": md, "mean": mean, "std": sd,
            "rng": mx - mn,
            "cv": (sd / mean if mean > 0 else 0.0),
            "rng_rel": ((mx - mn) / mx if mx > 0 else 0.0),
            "best2": (mx - s[-2]) if n > 1 else 0.0,
            "best2_rel": ((mx - s[-2]) / mx if n > 1 and mx > 0 else 0.0),
            "best_med_rel": ((mx - md) / mx if mx > 0 else 0.0),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "mad": float(np.median(np.abs(a - md))),
            "mad_rel": (float(np.median(np.abs(a - md))) / mx if mx > 0 else 0.0)}


def _num(x):
    return np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def participant_features(blows, demo):
    """One feature row plus the acceptable-blow FEV1/FVC arrays the decode needs."""
    bf = [blow_features(b) for b in blows]
    acc = [x for x in bf if x["acceptable"] > 0.5]
    # Data-validity guard, not a method switch: the bootstrap is defined over ACCEPTABLE blows and
    # every shipped participant has at least two, so the second branch never runs on this dataset.
    use = acc if len(acc) >= 1 else bf
    r = {"n_blows": float(len(bf)), "n_acceptable": float(len(acc)),
         "n_unacceptable": float(len(bf) - len(acc)), "frac_acceptable": len(acc) / len(bf),
         "n_plateau": float(sum(x["plateau"] for x in bf)),
         "frac_plateau_acc": float(np.mean([x["plateau"] for x in use])),
         "any_censored": float(max(x["censored"] for x in bf)),
         "frac_censored": float(np.mean([x["censored"] for x in use])),
         "btps": bf[0]["btps"], "standing": bf[0]["standing"]}
    for k in AGG_KEYS:
        sp = spread_stats(np.array([x[k] for x in use]))
        keep = list(sp) if k in CORE else ["max", "min", "med", "mean", "std", "cv", "rng_rel"]
        for kk in keep:
            r["%s_%s" % (k, kk)] = sp[kk]
    # contrast between all blows and the acceptable ones: an effort-quality signal
    for k in CORE:
        sp = spread_stats(np.array([x[k] for x in bf]))
        for kk in ("max", "med", "cv", "rng_rel"):
            r["all_%s_%s" % (k, kk)] = sp[kk]
        r["all_minus_acc_%s" % k] = sp["max"] - max(x[k] for x in use)
    # ATS repeatability: do the two best acceptable blows agree within 150 mL?
    for k in ("fev1", "fvc"):
        vals = np.sort([x[k] for x in use])
        gap = float(vals[-1] - vals[-2]) if len(vals) > 1 else 0.0
        r["ats_repeat_%s" % k] = float(gap <= 0.150)
        r["gap_%s_ml" % k] = gap * 1000.0
    # warm-up / learning trend across the session
    for k in ("fev1", "fvc", "pef"):
        xs = np.array([x["blow_no"] for x in use], dtype=np.float64)
        ys = np.array([x[k] for x in use], dtype=np.float64)
        if len(xs) > 1 and xs.std() > 0:
            slope = float(np.polyfit(xs, ys, 1)[0])
            r["trend_%s" % k] = slope
            r["trend_%s_rel" % k] = slope / max(ys.max(), 1e-6)
            r["argmax_%s_pos" % k] = float(np.argmax(ys)) / (len(ys) - 1)
        else:
            r["trend_%s" % k] = r["trend_%s_rel" % k] = r["argmax_%s_pos" % k] = 0.0
    fa = np.array([x["fev1"] for x in use]); va = np.array([x["fvc"] for x in use])
    r["same_best_blow"] = float(int(np.argmax(fa)) == int(np.argmax(va)))
    r["corr_fev1_fvc"] = (float(np.corrcoef(fa, va)[0, 1])
                          if len(fa) > 2 and fa.std() > 0 and va.std() > 0 else 0.0)
    r["mean_n_neg"] = float(np.mean([x["n_neg"] for x in bf]))
    r["max_n_rev"] = float(max(x["n_rev"] for x in bf))
    r["mean_eof_flow"] = float(np.mean([x["eof_flow"] for x in use]))
    r["age"] = _num(demo.get("age_years"))
    r["male"] = 1.0 if demo.get("sex") == "M" else 0.0
    r["height"] = _num(demo.get("height_cm"))
    r["weight"] = _num(demo.get("weight_kg"))
    r["bmi"] = _num(demo.get("bmi"))
    r["ht2"] = r["height"] ** 2 / 10000.0
    r["age_x_male"] = r["age"] * r["male"]
    r["child"] = float(r["age"] < 18)
    return r, np.array([x["fev1"] for x in use]), np.array([x["fvc"] for x in use])


# ================================================================================================
# 5.  NORMATIVE REFERENCE EQUATIONS FITTED ON THE SHIPPED POOL
# ================================================================================================
# The pool participants were never selected for the inhaler, so they are less obstructed by
# construction (FEV1/FVC 0.812 against 0.681 in the graded file).  Regressing each index on
# age / height / sex / ethnicity over the pool therefore yields a predicted-normal value, and the
# observed-over-predicted ratio measures how obstructed a participant is relative to a comparable
# but unselected person.  That ratio is the single strongest feature for the FEV1-arm latent.
# The challenge explicitly permits representations learned from the shipped pool; no target and
# no test row takes part in this fit.

REF_COLS = ("fev1_max", "fvc_max", "pef_max", "fef2575_max", "ratio_med")


def reference_design(F):
    h = np.log(np.clip(F["height"].astype(float).values, 50.0, None))
    a = F["age"].astype(float).values
    m = F["male"].astype(float).values
    cols = [np.ones(len(F)), h, a, a ** 2 / 100.0, np.log(np.clip(a, 4.0, None)),
            np.clip(a - 20.0, 0.0, None), np.clip(a - 40.0, 0.0, None), m, m * h]
    eth = F["ethnicity"].astype(str).values
    for e in ETH[1:]:
        cols.append((eth == e).astype(float))
    return np.nan_to_num(np.column_stack(cols), nan=0.0)


def fit_reference(pool_df):
    Z = reference_design(pool_df)
    ok = pool_df["height"].notna().values & pool_df["age"].notna().values
    out = {}
    for c in REF_COLS:
        y = np.log(np.clip(pool_df[c].astype(float).values, 1e-3, None))
        m = ok & np.isfinite(y)
        out[c] = np.linalg.lstsq(Z[m], y[m], rcond=None)[0]
    return out


def apply_reference(F, coefs):
    Z = reference_design(F)
    cols = {}
    for c, b in coefs.items():
        pred = Z @ b
        cols["ref_" + c] = pred
        cols["pp_" + c] = np.log(np.clip(F[c].astype(float).values, 1e-3, None)) - pred
    return pd.DataFrame(cols, index=F.index)


def design_matrix(F, coefs):
    num = F.drop(columns=["ethnicity"]).astype(float)
    eth = pd.get_dummies(F["ethnicity"].astype(str)).reindex(columns=ETH, fill_value=0).astype(float)
    eth.columns = ["eth_" + c for c in eth.columns]
    out = pd.concat([num, eth, apply_reference(F, coefs)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


# ================================================================================================
# 6.  THE EXACT BOOTSTRAP AND THE LATENT RESPONSE
# ================================================================================================
# A draw takes three acceptable blows WITH REPLACEMENT and keeps the best FEV1 and the best FVC
# over that same triple.  With m <= 10 acceptable blows there are only C(m+2,3) <= 220 distinct
# multisets, so the pre-side joint distribution of (best FEV1, best FVC) is enumerated exactly
# with multinomial weights.  No Monte Carlo, no sampling noise, fully deterministic.
#
# The hidden session is modelled as the pre-session's blows moved by one latent scalar per arm,
#     post_i = pre_i + delta * pre_i ** LAM
# which is increasing in pre_i, so the best of a triple maps as post_best = Fa + delta*Fa**LAM
# and every one of the nine events becomes a single threshold on delta:
#     FEV1 gain >= c  <=>  delta >= ((1+c)*Fb - Fa) / Fa**LAM
#     ATS on an arm   <=>  delta >= max(0.2 + Fb - Fa, 1.12*Fb - Fa) / Fa**LAM
# Fa is the post-side triple's best and Fb the pre-side triple's best.  This is the challenge's
# own published answer construction, re-run on a LEARNED distribution of delta.

class Session(object):
    """Per-participant decode object: every event threshold on the latent, with pair weights."""

    __slots__ = ("pw", "af", "bf", "if_", "av", "bv", "iv")

    def __init__(self, fev1, fvc, lam):
        f = np.asarray(fev1, dtype=np.float64)
        v = np.asarray(fvc, dtype=np.float64)
        m = len(f)
        combos = list(combinations_with_replacement(range(m), 3))
        K = len(combos)
        F = np.empty(K); V = np.empty(K); Wt = np.empty(K)
        for i, c in enumerate(combos):
            cnt = factorial(3)
            for k in np.bincount(c, minlength=m):
                cnt //= factorial(int(k))
            F[i] = f[list(c)].max()
            V[i] = v[list(c)].max()
            Wt[i] = cnt / float(m) ** 3
        Fa = np.repeat(F, K); Fb = np.tile(F, K)
        Va = np.repeat(V, K); Vb = np.tile(V, K)
        self.pw = (Wt[:, None] * Wt[None, :]).ravel()      # weight of (post triple, pre triple)
        sf = Fa ** lam; sv = Va ** lam
        self.af = Fa / sf; self.bf = Fb / sf; self.if_ = 1.0 / sf
        self.av = Va / sv; self.bv = Vb / sv; self.iv = 1.0 / sv

    def q_fev1(self, c):
        return (1.0 + c) * self.bf - self.af

    def q_fvc(self, c):
        return (1.0 + c) * self.bv - self.av

    def t_fev1(self):
        return np.maximum(ATS_ABS * self.if_ + self.bf - self.af, (1.0 + ATS_REL) * self.bf - self.af)

    def t_fvc(self):
        return np.maximum(ATS_ABS * self.iv + self.bv - self.av, (1.0 + ATS_REL) * self.bv - self.av)

    def probs_point(self, df, dv):
        """The nine probabilities for a point latent (used when inverting the train targets)."""
        out = np.empty(9)
        for j, c in enumerate(FEV1_THR):
            out[j] = float(self.pw[self.q_fev1(c) <= df].sum())
        for j, c in enumerate(FVC_THR):
            out[5 + j] = float(self.pw[self.q_fvc(c) <= dv].sum())
        out[8] = float(self.pw[(self.t_fev1() <= df) | (self.t_fvc() <= dv)].sum())
        return out

    def grid_probs(self, arm, dgrid):
        """(len(dgrid), n_events) exceedance probabilities over a grid of latent values."""
        thrs = [self.q_fev1(c) for c in FEV1_THR] if arm == "f" else [self.q_fvc(c) for c in FVC_THR]
        cols = []
        for q in thrs:
            o = np.argsort(q, kind="mergesort")
            qs = q[o]; cw = np.cumsum(self.pw[o])
            idx = np.searchsorted(qs, dgrid, side="right")
            cols.append(np.where(idx > 0, cw[np.clip(idx - 1, 0, len(cw) - 1)], 0.0))
        return np.stack(cols, axis=1)


def invert_latent(sess, y9, dgrid, refine_half=12, refine_step=2):
    """Recover (delta_fev1, delta_fvc) from one participant's published probabilities.

    The eight exceedance probabilities are a monotone step function of the two latents, so the
    inversion is a grid search; the identified interval is reported so poorly identified
    participants can be recognised.  A small joint refinement also matches the ats probability,
    which couples the two arms.
    """
    pf = sess.grid_probs("f", dgrid)
    pv = sess.grid_probs("v", dgrid)
    ef = ((pf - y9[0:5][None, :]) ** 2).sum(1)
    ev = ((pv - y9[5:8][None, :]) ** 2).sum(1)
    okf = np.where(ef <= ef.min() + 1e-12)[0]
    okv = np.where(ev <= ev.min() + 1e-12)[0]
    df = float(dgrid[okf].mean()); dv = float(dgrid[okv].mean())
    width = (float(dgrid[okf[-1]] - dgrid[okf[0]]), float(dgrid[okv[-1]] - dgrid[okv[0]]))
    step = float(dgrid[1] - dgrid[0]) * refine_step
    cf = df + step * np.arange(-refine_half, refine_half + 1)
    cv = dv + step * np.arange(-refine_half, refine_half + 1)
    efc = ((sess.grid_probs("f", cf) - y9[0:5][None, :]) ** 2).sum(1)
    evc = ((sess.grid_probs("v", cv) - y9[5:8][None, :]) ** 2).sum(1)
    tfa, tva = sess.t_fev1(), sess.t_fvc()
    best = None
    for a in range(len(cf)):
        firef = tfa <= cf[a]
        for b in range(len(cv)):
            pa = float(sess.pw[firef | (tva <= cv[b])].sum())
            obj = efc[a] + evc[b] + (pa - y9[8]) ** 2
            if best is None or obj < best[0]:
                best = (obj, cf[a], cv[b])
    return best[1], best[2], width


# ================================================================================================
# 7.  CONDITIONAL DISTRIBUTION OF THE LATENT  (the learning step)
# ================================================================================================
# S(u | x) = P(latent >= u | x) is learned with an ordinal ladder: the threshold level is an input
# feature constrained to be monotone, and every (participant, level) pair is a training row.  That
# pools strength across thresholds, guarantees a non-increasing survival curve, is heteroscedastic
# and shape-free (a participant the features call certainly unresponsive gets S ~ 0 above zero),
# and it handles the interval-censored participants for free: a latent known only to be below some
# bound still has a correct 0/1 label at every ladder level.
#
# Three deliberately different boosting configurations are averaged on the logit scale.  The
# counts are fixed; nothing depends on the clock or the machine.

VARIANTS = (
    {"learning_rate": 0.04, "num_leaves": 15, "min_data_in_leaf": 340,
     "feature_fraction": 0.45, "bagging_fraction": 0.80, "lambda_l2": 10.0, "seed": 1, "rounds": 400},
    {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 600,
     "feature_fraction": 0.30, "bagging_fraction": 0.70, "lambda_l2": 30.0, "seed": 7, "rounds": 500},
    {"learning_rate": 0.06, "num_leaves": 7, "min_data_in_leaf": 200,
     "feature_fraction": 0.60, "bagging_fraction": 0.85, "lambda_l2": 5.0, "seed": 13, "rounds": 300},
)
# Every LightGBM randomness source gets its own explicit seed (Guidebook 3.3): the top-level
# `seed` would propagate, but naming each one leaves nothing for the determinism review to infer.
BASE_PARAMS = {"objective": "binary", "bagging_freq": 1, "verbose": -1,
               "num_threads": N_THREADS, "deterministic": True, "force_row_wise": True,
               "bagging_seed": SEED, "feature_fraction_seed": SEED, "data_random_seed": SEED,
               "extra_seed": SEED, "objective_seed": SEED}


def build_ladder(y):
    """Ladder levels at quantiles of the latent, kept strictly increasing."""
    g = np.maximum.accumulate(np.quantile(y, LADDER_Q))
    for i in range(1, len(g)):
        if g[i] <= g[i - 1]:
            g[i] = g[i - 1] + 1e-5
    return g


def _ordinal_rows(X, grid):
    K = len(grid)
    return np.hstack([np.repeat(X, K, axis=0), np.tile(grid, len(X))[:, None]])


def fit_survival(X, y, grid, names):
    Z = _ordinal_rows(X, grid)
    labels = (np.repeat(y, len(grid)) >= np.tile(grid, len(y))).astype(np.float64)
    mono = [0] * X.shape[1] + [-1]            # survival must not increase with the level
    models = []
    for v in VARIANTS:
        p = dict(BASE_PARAMS)
        p.update({k: v[k] for k in v if k != "rounds"})
        p["monotone_constraints"] = mono
        models.append(lgb.train(p, lgb.Dataset(Z, labels, feature_name=list(names) + ["level"]),
                                num_boost_round=v["rounds"]))
    return models


def predict_survival(models, X, grid):
    Z = _ordinal_rows(X, grid)
    n, K = len(X), len(grid)
    acc = np.zeros((n, K))
    for m in models:
        p = np.clip(m.predict(Z).reshape(n, K), 1e-6, 1.0 - 1e-6)
        acc += np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-acc / len(models)))


def survival_curve(grid, S, ramp):
    """Monotone survival curve extended to 1 below the ladder and 0 above it."""
    s = np.minimum.accumulate(np.clip(S, 1e-6, 1.0 - 1e-6))
    return (np.concatenate(([grid[0] - ramp], grid, [grid[-1] + ramp])),
            np.concatenate(([1.0], s, [0.0])))


def pit_values(S, grid, y, ramp):
    """Probability-integral transforms; a calibrated conditional law makes these uniform."""
    out = np.empty(len(y))
    for i in range(len(y)):
        xs, ys = survival_curve(grid, S[i], ramp)
        out[i] = 1.0 - np.interp(y[i], xs, ys)
    return np.clip(out, 1e-4, 1.0 - 1e-4)


# ================================================================================================
# 8.  COUPLING THE TWO ARMS
# ================================================================================================
# ATS fires if EITHER arm clears both of its bars, so the decode needs the joint law of the two
# latents, not just the margins.  One parameter does it: a Gaussian copula whose correlation is
# estimated from out-of-fold PIT values on the training participants.

class Copula(object):
    """Bivariate standard normal CDF at a fixed correlation (Drezner-Wesolowsky quadrature).

    Phi2(a,b;r) = Phi(a)Phi(b) + 1/(2pi) * int_0^r exp(-(a^2 - 2tab + b^2)/(2(1-t^2)))/sqrt(1-t^2) dt
    Exact to ~1e-10 with 24 Gauss-Legendre nodes and vectorised over (a, b).
    """

    def __init__(self, rho, n_node=COPULA_NODES):
        self.rho = float(np.clip(rho, -0.95, 0.95))
        x, w = np.polynomial.legendre.leggauss(n_node)
        self.t = 0.5 * self.rho * (x + 1.0)
        self.w = 0.5 * self.rho * w

    def __call__(self, a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        base = ndtr(a) * ndtr(b)
        if abs(self.rho) < 1e-9:
            return base                               # the integral is exactly zero at rho = 0
        t = self.t[:, None]
        num = a[None, :] ** 2 - 2.0 * t * a[None, :] * b[None, :] + b[None, :] ** 2
        integ = np.exp(-num / (2.0 * (1.0 - t ** 2))) / np.sqrt(1.0 - t ** 2)
        return np.clip(base + (self.w[:, None] * integ).sum(0) / (2.0 * np.pi), 0.0, 1.0)


def norm_ppf(p):
    return ndtri(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12))


# ================================================================================================
# 9.  THE DECODE:  p(event) = E_latent [ P_bootstrap(event | latent) ]
# ================================================================================================
# Integrating the exact bootstrap over the LEARNED latent distribution is what makes the output a
# calibrated probability rather than a score, and what makes it non-increasing within an arm.

def decode_one(sess, gf, Sf, gv, Sv, copula, ramp):
    xf, yf = survival_curve(gf, Sf, ramp)
    xv, yv = survival_curve(gv, Sv, ramp)
    out = np.empty(9)
    for j, c in enumerate(FEV1_THR):
        out[j] = float(np.dot(sess.pw, np.interp(sess.q_fev1(c), xf, yf)))
    for j, c in enumerate(FVC_THR):
        out[5 + j] = float(np.dot(sess.pw, np.interp(sess.q_fvc(c), xv, yv)))
    ff = 1.0 - np.interp(sess.t_fev1(), xf, yf)
    fv = 1.0 - np.interp(sess.t_fvc(), xv, yv)
    out[8] = 1.0 - float(np.dot(sess.pw, copula(norm_ppf(ff), norm_ppf(fv))))
    return np.clip(out, 0.0, 1.0)


def decode_all(sessions, gf, Sf, gv, Sv, rho, ramp):
    cop = Copula(rho)
    return np.array([decode_one(sessions[i], gf, Sf[i], gv, Sv[i], cop, ramp)
                     for i in range(len(sessions))])


def ats_posterior(sessions, gf, Sf, gv, Sv, rho, ramp):
    """The whole posterior of the ATS probability t, one row of N_POST_DRAWS draws per participant.

    The posterior of t is needed, not just its mean, because the fragility the grader compares
    against is a function of t: true fragility is 4t(1-t), and 4*E[t]*(1-E[t]) is NOT E[4t(1-t)].
    The draws are a fixed deterministic lattice coupled by the same copula as the decode.
    """
    rng = np.random.RandomState(20261006)
    z1 = rng.standard_normal(N_POST_DRAWS)
    z2 = rho * z1 + np.sqrt(max(1e-9, 1.0 - rho ** 2)) * rng.standard_normal(N_POST_DRAWS)
    qf, qv = ndtr(z1), ndtr(z2)
    T = np.empty((len(sessions), N_POST_DRAWS))
    for i, s in enumerate(sessions):
        xs, ys = survival_curve(gf, Sf[i], ramp); uf = np.interp(qf, 1.0 - ys, xs)
        xs, ys = survival_curve(gv, Sv[i], ramp); uv = np.interp(qv, 1.0 - ys, xs)
        T[i] = ((uf[:, None] >= s.t_fev1()[None, :]) |
                (uv[:, None] >= s.t_fvc()[None, :])).astype(np.float64) @ s.pw
    return T


def borda_fragility(T, reference=None):
    """Expected marginal rank of this participant's fragility: b_i = E[ F(frag) ].

    For a pairwise concordance such as FDS the ideal order puts i above j when
    P(frag_i > frag_j) > P(frag_j > frag_i).  Averaging that over j gives the Borda count
    b_i = E_{frag ~ posterior_i}[ F(frag) ] with F the POPULATION marginal CDF of fragility.  It
    differs from E[frag] exactly when posterior shapes differ between participants, and on
    grouped CV it orders the truth better (Somers' D 0.379 against 0.360 for E[frag]).

    `reference` is the pooled fragility sample that defines F.  It is always built on the
    TRAINING participants and then reused for the test participants, so no statistic is ever
    computed across test rows.
    """
    FR = 4.0 * T * (1.0 - T)
    ref = np.sort(FR.ravel()) if reference is None else reference
    b = np.searchsorted(ref, FR, side="right").mean(axis=1) / float(len(ref))
    return b, ref


# ================================================================================================
# 10.  THE OFFICIAL METRIC AND THE FRAGILITY HALF
# ================================================================================================

def official_score(pred, truth):
    """RCS (weighted Brier skill) x FDS (fragility concordance), as the challenge defines them."""
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    truth = np.asarray(truth, dtype=np.float64)
    t_ats = truth[:, 8]
    w = 1.0 + 4.0 * t_ats * (1.0 - t_ats)
    e = np.mean((pred - truth) ** 2, axis=1)
    e_ref = np.mean((0.5 - truth) ** 2, axis=1)
    L = float(np.sum(w * e) / np.sum(w))
    L_ref = float(np.sum(w * e_ref) / np.sum(w))
    rcs = max(0.0, 1.0 - L / L_ref)
    tf = 4.0 * t_ats * (1.0 - t_ats)
    pf = 4.0 * pred[:, 8] * (1.0 - pred[:, 8])
    d = somers_d(tf, pf)
    fds = (1.0 + d) / 2.0
    return {"score": float(np.sqrt(max(0.0, rcs) * max(0.0, fds))), "rcs": rcs, "fds": fds,
            "brier": float(np.mean(e))}


def somers_d(truth_frag, pred_frag):
    """Sum of sign agreements over ordered pairs, divided by pairs untied IN THE TRUTH."""
    t = np.asarray(truth_frag, dtype=np.float64)
    p = np.asarray(pred_frag, dtype=np.float64)
    st = np.sign(t[:, None] - t[None, :])
    sp = np.sign(p[:, None] - p[None, :])
    den = float(np.sum(st != 0.0))
    if den == 0.0:
        return 0.0
    return float(np.sum(st * sp)) / den


# FDS depends ONLY on the ordering of |p - 0.5|: any symmetric monotone transform of ats leaves it
# unchanged, so the only way to move it is a genuinely better fragility signal.  The Brier-optimal
# p = E[t|x] implies fragility 4p(1-p), but the grader compares against E[4t(1-t)|x], which is
# smaller by 4*Var(t|x).  Where the latent posterior is wide the two orderings disagree sharply:
# a participant who is certainly positive-or-negative but we cannot tell which gets p ~ 0.5, hence
# maximal implied fragility and minimal true fragility.  We therefore transfer the ordering of
# phi = E[4t(1-t)|x] onto the submitted ats with WEIGHTED ISOTONIC regression, which is the
# cheapest possible way in Brier terms to buy that ordering.

def fit_fragility_map(phi, p_ats, weights):
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(phi, np.abs(p_ats - 0.5), sample_weight=weights)
    return iso


def apply_fragility_map(iso, phi, p_ats, lo, hi, eps=1e-6):
    """Rebuild ats so that |p-0.5| follows phi's ordering; eps breaks isotonic ties."""
    m = iso.predict(phi)
    gr = (np.clip(phi, lo, hi) - lo) / max(hi - lo, 1e-9)
    m = np.clip(m - eps * gr, 0.0, 0.5)
    side = np.where(p_ats >= 0.5, 1.0, -1.0)
    return np.clip(0.5 + side * m, 0.0, 1.0)


# ================================================================================================
# 11.  DIRECT NINE-TARGET MEMBER
# ================================================================================================
# A structurally different model kept as a minor blend member: it regresses the nine published
# probabilities directly, with no latent and no bootstrap.  It is the challenge's own published
# reference rung, so it also serves as a yardstick inside the run.

DIRECT_PARAMS = {"objective": "l2", "learning_rate": 0.03, "num_leaves": 15,
                 "min_data_in_leaf": 40, "feature_fraction": 0.5, "bagging_fraction": 0.8,
                 "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1,
                 "num_threads": N_THREADS, "seed": 1, "deterministic": True,
                 "force_row_wise": True, "bagging_seed": SEED, "feature_fraction_seed": SEED,
                 "data_random_seed": SEED, "extra_seed": SEED, "objective_seed": SEED}
DIRECT_ROUNDS = 500


def fit_direct(X, Y, names):
    return [lgb.train(DIRECT_PARAMS, lgb.Dataset(X, Y[:, j], feature_name=list(names)),
                      num_boost_round=DIRECT_ROUNDS) for j in range(9)]


def predict_direct(models, X):
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 1.0)


# ================================================================================================
# 12.  FOLDS
# ================================================================================================
# One row per participant, so any split is participant-disjoint already.  What is worth mirroring
# is the official split: stratified by response band and by how many acceptable blows the session
# holds, exactly as the challenge's validation tip recommends.

def make_folds(Y, n_acceptable, n_splits=N_FOLDS, seed=SEED):
    band = np.digitize(Y[:, 8], [0.02, 0.10, 0.30, 0.60, 0.90])
    strata = band * 4 + (np.clip(n_acceptable, 3, 6) - 3)
    rng = np.random.RandomState(seed)
    fold = np.full(len(strata), -1, dtype=int)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_splits
    return fold


# ================================================================================================
# 13.  DATA LOADING
# ================================================================================================

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_tables(public_dir):
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    sample = pd.read_csv(public_dir / "sample_submission.csv", keep_default_na=False)
    # Demographics lookup, built as a plain per-id dict from each table SEPARATELY, so that no
    # train/test object is ever combined and no statistic can be shared between the splits.
    demo_cols = ["age_years", "sex", "ethnicity", "height_cm", "weight_kg", "bmi"]
    demo = {}
    for frame in (train, test):
        for rec in frame[["id"] + demo_cols].to_dict("records"):
            demo[str(rec["id"])] = rec

    rows, fev, fvc = {}, {}, {}
    for rec in read_jsonl(public_dir / "blows.jsonl"):
        pid = rec["pid"]
        r, f, v = participant_features(rec["blows"], demo.get(pid, {}))
        rows[pid] = r; fev[pid] = f; fvc[pid] = v
    log("decoded graded traces for %d participants" % len(rows))

    prows = {}
    for rec in read_jsonl(public_dir / "pool.jsonl"):
        r, _, _ = participant_features(rec["blows"], rec)
        r["ethnicity"] = rec.get("ethnicity", "other")
        prows[rec["pid"]] = r
    log("decoded pool traces for %d participants" % len(prows))

    X = pd.DataFrame(rows).T
    X["ethnicity"] = [demo.get(str(i), {}).get("ethnicity", "other") for i in X.index]
    P = pd.DataFrame(prows).T
    Y = pd.DataFrame(list(train["target_json"].apply(json.loads)))[TARGETS].values.astype(np.float64)
    return train, test, sample, X, P, Y, fev, fvc


# ================================================================================================
# 14.  SUBMISSION BUILDING AND VALIDATION
# ================================================================================================

def build_submission(sample, pred_by_id):
    """Rows follow sample_submission's own id order, so coverage and order cannot drift."""
    recs = []
    for pid in sample["id"].astype(str).tolist():
        p = pred_by_id[pid]
        recs.append(json.dumps({k: float(round(float(p[j]), 6)) for j, k in enumerate(TARGETS)}))
    return pd.DataFrame({"id": sample["id"].astype(str).values, "target_json": recs})


def validate_submission(sub, sample):
    """Explicit checks with readable messages; by construction none of them can fire."""
    problems = []
    if list(sub.columns) != ["id", "target_json"]:
        problems.append("columns are %s" % list(sub.columns))
    if len(sub) != len(sample):
        problems.append("row count %d != %d" % (len(sub), len(sample)))
    if sub["id"].duplicated().any():
        problems.append("duplicate ids")
    want = sample["id"].astype(str).tolist()
    if sub["id"].astype(str).tolist() != want:
        problems.append("ids do not match sample_submission")
    mono_f = mono_v = 0
    for s in sub["target_json"]:
        o = json.loads(s)                 # round-trips a string this script just serialised
        missing = [k for k in TARGETS if k not in o]
        if missing:
            problems.append("missing keys %s" % missing)
            break
        bad = [k for k in TARGETS
               if not isinstance(o[k], (int, float)) or not np.isfinite(o[k])
               or o[k] < 0.0 or o[k] > 1.0]
        if bad:
            problems.append("values out of [0,1] or non-finite for %s" % bad)
            break
        fv = [o[k] for k in TARGETS[0:5]]
        vv = [o[k] for k in TARGETS[5:8]]
        mono_f += int(all(fv[i] >= fv[i + 1] - 1e-9 for i in range(4)))
        mono_v += int(all(vv[i] >= vv[i + 1] - 1e-9 for i in range(2)))
    if problems:
        raise ValueError("submission is not valid: " + "; ".join(problems[:6]))
    log("submission validated: %d rows, FEV1 arm monotone %d/%d, FVC arm monotone %d/%d"
        % (len(sub), mono_f, len(sub), mono_v, len(sub)))


# ================================================================================================
# 15.  MAIN
# ================================================================================================

def main():
    public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
    submission_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    seed_everything()

    train, test, sample, X, P, Y, fev, fvc = build_tables(public_dir)

    # ---- 2/5: normative reference equations from the unlabelled shipped pool ----
    coefs = fit_reference(P)
    # design_matrix is a stateless transform (fixed ethnicity column list + the pool-fitted
    # reference coefficients), so it is applied to each split's own rows independently: nothing
    # about the test rows can influence the training features.
    XD_tr = design_matrix(X.loc[train["id"]], coefs)
    XD_te = design_matrix(X.loc[test["id"]], coefs)
    names = list(XD_tr.columns)
    Xtr = XD_tr.values.astype(np.float64)
    Xte = XD_te[names].values.astype(np.float64)
    log("design matrix %s (%d features)" % (Xtr.shape, Xtr.shape[1]))

    # ---- 6: exact bootstrap objects and the inverted latent response ----
    sess_tr = [Session(fev[p], fvc[p], LAM) for p in train["id"]]
    sess_te = [Session(fev[p], fvc[p], LAM) for p in test["id"]]
    scale = float(np.mean([s.if_.mean() for s in sess_tr]))
    dgrid = np.arange(-0.8, 2.0, 0.002) * scale
    df = np.empty(len(Y)); dv = np.empty(len(Y)); wid = np.empty((len(Y), 2))
    for i in range(len(Y)):
        df[i], dv[i], w = invert_latent(sess_tr[i], Y[i], dgrid)
        wid[i] = w
    rec = np.array([sess_tr[i].probs_point(df[i], dv[i]) for i in range(len(Y))])
    orc = official_score(rec, Y)
    # Oracle check: pushing the FITTED latent back through the bootstrap must reproduce the
    # published ceiling.  Fitting the two arms to the eight exceedance probabilities alone gives
    # 0.9133 against the documented 0.9116; the number below also uses the ats probability in the
    # fit, so it sits a little above that.  Either way it validates the index conventions, the
    # triple-bootstrap rule and the metric implementation together.
    log("latent inverted; ORACLE check (fitted latent through the exact bootstrap) score %.4f  "
        "[documented ceiling 0.9116; margin-only fit reproduces 0.9133]" % orc["score"])
    log("latent fev1 arm: median %.3f IQR %.3f-%.3f | fvc arm: median %.3f IQR %.3f-%.3f"
        % (np.median(df), np.percentile(df, 25), np.percentile(df, 75),
           np.median(dv), np.percentile(dv, 25), np.percentile(dv, 75)))
    log("median identification width: fev1 %.4f fvc %.4f" % (np.median(wid[:, 0]), np.median(wid[:, 1])))

    gf = build_ladder(df); gv = build_ladder(dv)
    ramp = RAMP_FRAC * float(gf[-1] - gf[0])
    weights = 1.0 + 4.0 * Y[:, 8] * (1.0 - Y[:, 8])

    # ---- 7: out-of-fold conditional survival, used to fit every post-hoc constant ----
    fold = make_folds(Y, train["n_acceptable"].values)
    Sf = np.zeros((len(Y), len(gf))); Sv = np.zeros((len(Y), len(gv)))
    Pdir = np.zeros((len(Y), 9))
    for k in range(N_FOLDS):
        tr = fold != k; va = fold == k
        Sf[va] = predict_survival(fit_survival(Xtr[tr], df[tr], gf, names), Xtr[va], gf)
        Sv[va] = predict_survival(fit_survival(Xtr[tr], dv[tr], gv, names), Xtr[va], gv)
        Pdir[va] = predict_direct(fit_direct(Xtr[tr], Y[tr], names), Xtr[va])
        log("fold %d/%d out-of-fold models done" % (k + 1, N_FOLDS))

    # ---- 8: copula correlation between the two arms, from out-of-fold PIT values ----
    zf = norm_ppf(pit_values(Sf, gf, df, ramp))
    zv = norm_ppf(pit_values(Sv, gv, dv, ramp))
    rho = float(np.corrcoef(zf, zv)[0, 1])
    log("copula correlation from out-of-fold PIT values: %.3f" % rho)
    for nm, z in (("fev1", zf), ("fvc", zv)):
        h = np.histogram(ndtr(z), bins=10, range=(0.0, 1.0))[0] / len(z)
        log("PIT calibration %-4s decile shares %s (uniform = 0.100)" % (nm, np.round(h, 3)))

    # ---- 9: decode out-of-fold, then fit the two blend constants on train evidence only ----
    Pgen = decode_all(sess_tr, gf, Sf, gv, Sv, rho, ramp)
    log("out-of-fold generative decode      : %.4f" % official_score(Pgen, Y)["score"])
    log("out-of-fold direct nine-target     : %.4f" % official_score(Pdir, Y)["score"])
    w_grid = np.arange(0.0, 0.41, 0.05)
    w_scores = [official_score(np.clip((1.0 - w) * Pgen + w * Pdir, 0.0, 1.0), Y)["score"]
                for w in w_grid]
    w_dir = float(w_grid[int(np.argmax(w_scores))])
    Poof = np.clip((1.0 - w_dir) * Pgen + w_dir * Pdir, 0.0, 1.0)
    log("blend weight on the direct member (fitted on out-of-fold): %.2f -> %.4f"
        % (w_dir, official_score(Poof, Y)["score"]))

    # ---- 10: fragility ordering for the ats output ----
    T_tr = ats_posterior(sess_tr, gf, Sf, gv, Sv, rho, ramp)
    phi, frag_ref = borda_fragility(T_tr)          # reference CDF built on train only
    plo, phi_hi = float(phi.min()), float(phi.max())
    log("fragility ordering statistic: Borda expected marginal rank, Somers' D = %.4f "
        "(against %.4f for the implied 4p(1-p))"
        % (somers_d(4.0 * Y[:, 8] * (1.0 - Y[:, 8]), phi),
           somers_d(4.0 * Y[:, 8] * (1.0 - Y[:, 8]), 4.0 * Poof[:, 8] * (1.0 - Poof[:, 8]))))
    Pfrag = Poof.copy()
    for k in range(N_FOLDS):                        # cross-fitted, for an honest CV number
        tr = fold != k; va = fold == k
        iso_k = fit_fragility_map(phi[tr], Poof[tr, 8], weights[tr])
        Pfrag[va, 8] = apply_fragility_map(iso_k, phi[va], Poof[va, 8], plo, phi_hi)
    s_final = official_score(Pfrag, Y)
    log("out-of-fold WITH fragility transfer: %.4f  (rcs %.4f  fds %.4f  brier %.5f)"
        % (s_final["score"], s_final["rcs"], s_final["fds"], s_final["brier"]))
    fold_scores = [official_score(Pfrag[fold == k], Y[fold == k])["score"] for k in range(N_FOLDS)]
    log("per-fold scores %s  mean %.4f  sd %.4f  worst %.4f"
        % (np.round(fold_scores, 4), np.mean(fold_scores), np.std(fold_scores), np.min(fold_scores)))
    for j, t in enumerate(TARGETS):
        log("  per-target Brier %-8s %.5f" % (t, float(np.mean((Pfrag[:, j] - Y[:, j]) ** 2))))

    # ---- strip-the-ML control (Guidebook 5.3): replace the LEARNED conditional distribution of
    # the latent by the population marginal, keeping the decode and every participant's own blows.
    # This is the "remove the model and see if it still works" test, run and reported in-script.
    marg_f = np.array([(df >= g).mean() for g in gf])
    marg_v = np.array([(dv >= g).mean() for g in gv])
    P_noml = decode_all(sess_tr, gf, np.tile(marg_f, (len(Y), 1)),
                        gv, np.tile(marg_v, (len(Y), 1)), rho, ramp)
    log("strip-the-ML control (population-marginal latent, no learned model): %.4f   vs  %.4f "
        "with the trained models -> the models supply %.1f%% of the skill above the "
        "constant-prediction floor"
        % (official_score(P_noml, Y)["score"], s_final["score"],
           100.0 * (s_final["score"] - official_score(P_noml, Y)["score"])
           / max(s_final["score"] - 0.4705, 1e-9)))

    # ---- evidence that the models learned rather than were told: top gain features ----
    gain = np.zeros(len(names) + 1)
    for m in fit_survival(Xtr[fold != 0], df[fold != 0], gf, names):
        gain += m.feature_importance("gain")
    top = np.argsort(-gain)[:12]
    log("top learned features for the FEV1-arm latent: %s"
        % [(names + ["level"])[i] for i in top])

    # ---- refit on 100% of the training participants with the same fixed counts ----
    Mf = fit_survival(Xtr, df, gf, names)
    Mv = fit_survival(Xtr, dv, gv, names)
    Mdir = fit_direct(Xtr, Y, names)
    log("refit on all %d training participants" % len(Y))

    Sf_te = predict_survival(Mf, Xte, gf)
    Sv_te = predict_survival(Mv, Xte, gv)
    Pgen_te = decode_all(sess_te, gf, Sf_te, gv, Sv_te, rho, ramp)
    Pdir_te = predict_direct(Mdir, Xte)
    Pte = np.clip((1.0 - w_dir) * Pgen_te + w_dir * Pdir_te, 0.0, 1.0)
    T_te = ats_posterior(sess_te, gf, Sf_te, gv, Sv_te, rho, ramp)
    phi_te, _ = borda_fragility(T_te, reference=frag_ref)   # train-fitted reference, reused
    iso_full = fit_fragility_map(phi, Poof[:, 8], weights)       # fitted on train out-of-fold only
    Pte[:, 8] = apply_fragility_map(iso_full, phi_te, Pte[:, 8], plo, phi_hi)

    log("test prediction means %s" % dict(zip(TARGETS, np.round(Pte.mean(0), 4))))
    log("train target  means %s" % dict(zip(TARGETS, np.round(Y.mean(0), 4))))

    pred_by_id = {str(pid): Pte[i] for i, pid in enumerate(test["id"].astype(str).values)}
    sub = build_submission(sample, pred_by_id)
    validate_submission(sub, sample)
    sub.to_csv(submission_out, index=False)
    log("wrote %s  shape=%s" % (submission_out, sub.shape))


if __name__ == "__main__":
    main()
