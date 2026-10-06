"""Per-blow spirometric indices (rich) and participant-level aggregation."""
import base64
import numpy as np

DT = 0.01
BYTE_OFFSET = 62


def decode(blow):
    buf = base64.b64decode(blow["flow_b64"])
    raw = np.frombuffer(buf, np.uint8).astype(np.float64)
    delta_ml = raw - BYTE_OFFSET                      # cast first: uint8 arithmetic wraps
    btps = float(blow["btps"])
    return np.cumsum(delta_ml) * btps / 1000.0, delta_ml * btps / 10.0


def blow_feats(blow):
    """One raw trace -> a dict of standard spirometric indices plus shape/quality descriptors."""
    v, f = decode(blow)
    n = len(v)
    ipk = int(np.argmax(f))
    pef = float(f[ipk])
    # ATS back-extrapolation: line through the peak-flow sample with slope PEF, intersect V=0
    i0 = 0 if pef <= 0 else int(np.round(ipk - v[ipk] / (pef * DT)))
    i0 = max(0, min(i0, n - 1))
    v0 = float(v[i0])
    vol = v - v0                                      # volume measured from back-extrapolated zero
    vmax = float(np.max(vol))
    imax = int(np.argmax(vol))
    d = {}
    d["pef"] = pef
    d["fvc"] = vmax
    for sec, nm in ((0.5, "fev05"), (1.0, "fev1"), (3.0, "fev3"), (6.0, "fev6")):
        j = i0 + int(round(sec / DT))
        d[nm] = float(vol[j] if j < n else vol[-1])
    d["ratio"] = d["fev1"] / vmax if vmax > 0 else 0.0
    d["ratio6"] = d["fev1"] / d["fev6"] if d["fev6"] > 0 else 0.0
    d["fev05_fev1"] = d["fev05"] / d["fev1"] if d["fev1"] > 0 else 0.0
    d["tpef"] = (ipk - i0) * DT                       # time to peak flow
    d["vpef"] = float(vol[ipk])                       # volume exhaled at peak flow
    d["vpef_frac"] = d["vpef"] / vmax if vmax > 0 else 0.0
    d["fet"] = (imax - i0) * DT                       # forced expiratory time to max volume
    d["extrap_vol"] = v0 - float(v[0])                # back-extrapolated volume (quality index)
    d["extrap_frac"] = d["extrap_vol"] / vmax if vmax > 0 else 0.0
    # flows at fixed exhaled fractions -> concavity of the descending limb
    if vmax > 0 and imax > ipk:
        seg = vol[ipk:imax + 1]
        fseg = f[ipk:imax + 1]
        for frac in (0.25, 0.50, 0.75):
            tgt = frac * vmax
            k = int(np.searchsorted(np.maximum.accumulate(seg), tgt))
            k = min(k, len(fseg) - 1)
            d["fef%d" % int(frac * 100)] = float(fseg[k])
        # mean mid-expiratory flow FEF25-75
        a = int(np.searchsorted(np.maximum.accumulate(seg), 0.25 * vmax))
        b = int(np.searchsorted(np.maximum.accumulate(seg), 0.75 * vmax))
        b = min(b, len(seg) - 1)
        d["fef2575"] = (0.5 * vmax) / max((b - a) * DT, DT)
    else:
        d["fef25"] = d["fef50"] = d["fef75"] = d["fef2575"] = 0.0
    for k in ("fef25", "fef50", "fef75", "fef2575"):
        d[k + "_pef"] = d[k] / pef if pef > 0 else 0.0
    # curve-shape index: area under the normalised descending limb (1 = square, <1 = concave)
    if vmax > 0 and imax > ipk and pef > 0:
        seg = np.maximum.accumulate(vol[ipk:imax + 1])
        fseg = f[ipk:imax + 1]
        xs = np.linspace(d["vpef_frac"], 1.0, 21)
        ys = np.interp(xs * vmax, seg, fseg) / pef
        d["limb_area"] = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
        d["limb_mid"] = float(ys[len(ys) // 2])
    else:
        d["limb_area"] = d["limb_mid"] = 0.0
    # artefact / effort descriptors
    neg = f < -0.05
    d["n_neg"] = float(neg.sum())
    d["min_flow"] = float(f.min())
    d["n_rev"] = float(np.sum(np.diff(np.sign(np.maximum(f, 0))) != 0))  # flow interruptions
    d["eof_flow"] = float(f[imax]) if imax < n else 0.0                  # flow at end of test
    d["plateau_flow"] = float(np.mean(f[max(imax - 50, i0):imax + 1])) if imax > i0 else 0.0
    d["i0"] = float(i0)
    d["n_points"] = float(n)
    d["censored"] = float(n >= 2044)
    d["acceptable"] = 1.0 if blow["acceptable"] == "Y" else 0.0
    d["plateau"] = 1.0 if blow["plateau"] == "Y" else 0.0
    d["blow_no"] = float(blow["blow"])
    d["btps"] = float(blow["btps"])
    d["standing"] = 1.0 if blow.get("position") == "Standing" else 0.0
    return d


# ------------------------------------------------------------------ aggregation ----
AGG_KEYS = ["fev1", "fvc", "pef", "ratio", "fev05", "fev3", "fev6", "fef2575",
            "fef50_pef", "fef25_pef", "fef75_pef", "limb_area", "limb_mid",
            "tpef", "vpef_frac", "fet", "extrap_frac", "ratio6", "fev05_fev1"]
CORE = ["fev1", "fvc", "pef"]


def _spread(a):
    """Dispersion descriptors of one index across the blows of a session."""
    a = np.asarray(a, dtype=np.float64)
    s = np.sort(a)
    n = len(s)
    mx, mn, md = s[-1], s[0], float(np.median(s))
    mean = float(a.mean())
    out = {"max": mx, "min": mn, "med": md, "mean": mean,
           "std": float(a.std(ddof=1)) if n > 1 else 0.0,
           "rng": mx - mn,
           "cv": (float(a.std(ddof=1)) / mean if n > 1 and mean > 0 else 0.0),
           "rng_rel": (mx - mn) / mx if mx > 0 else 0.0,
           "best2": (mx - s[-2]) if n > 1 else 0.0,
           "best2_rel": ((mx - s[-2]) / mx if n > 1 and mx > 0 else 0.0),
           "best_med_rel": ((mx - md) / mx if mx > 0 else 0.0),
           "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
           "mad": float(np.median(np.abs(a - md))),
           "mad_rel": float(np.median(np.abs(a - md)) / mx) if mx > 0 else 0.0}
    return out


def participant_feats(blows, demo):
    """Aggregate a participant's blows into one feature row."""
    bf = [blow_feats(b) for b in blows]
    acc = [x for x in bf if x["acceptable"] > 0.5]
    use = acc if len(acc) >= 1 else bf
    r = {}
    r["n_blows"] = float(len(bf))
    r["n_acceptable"] = float(len(acc))
    r["n_unacceptable"] = float(len(bf) - len(acc))
    r["frac_acceptable"] = len(acc) / len(bf)
    r["n_plateau"] = float(sum(x["plateau"] for x in bf))
    r["frac_plateau_acc"] = float(np.mean([x["plateau"] for x in use]))
    r["any_censored"] = float(max(x["censored"] for x in bf))
    r["frac_censored"] = float(np.mean([x["censored"] for x in use]))
    r["btps"] = bf[0]["btps"]
    r["standing"] = bf[0]["standing"]
    for k in AGG_KEYS:
        vals = np.array([x[k] for x in use])
        sp = _spread(vals)
        keep = ["max", "min", "med", "mean", "std", "cv", "rng_rel"] if k not in CORE else list(sp)
        for kk in keep:
            r["%s_%s" % (k, kk)] = sp[kk]
    # all-blow (including unacceptable) core spread: effort-quality signal
    for k in CORE:
        vals = np.array([x[k] for x in bf])
        sp = _spread(vals)
        for kk in ("max", "med", "cv", "rng_rel"):
            r["all_%s_%s" % (k, kk)] = sp[kk]
        accmax = max(x[k] for x in use)
        r["all_minus_acc_%s" % k] = sp["max"] - accmax
    # ATS repeatability: do the two best acceptable blows agree within 150 mL?
    for k in ("fev1", "fvc"):
        vals = np.sort([x[k] for x in use])
        gap = (vals[-1] - vals[-2]) if len(vals) > 1 else 0.0
        r["ats_repeat_%s" % k] = float(gap <= 0.150)
        r["gap_%s_ml" % k] = gap * 1000.0
    # order / learning trend across the session (blow number vs index)
    for k in ("fev1", "fvc", "pef"):
        xs = np.array([x["blow_no"] for x in use], dtype=np.float64)
        ys = np.array([x[k] for x in use], dtype=np.float64)
        if len(xs) > 1 and xs.std() > 0:
            r["trend_%s" % k] = float(np.polyfit(xs, ys, 1)[0])
            r["trend_%s_rel" % k] = r["trend_%s" % k] / max(ys.max(), 1e-6)
            r["argmax_%s_pos" % k] = float(np.argmax(ys)) / (len(ys) - 1)
        else:
            r["trend_%s" % k] = r["trend_%s_rel" % k] = 0.0
            r["argmax_%s_pos" % k] = 0.0
    # best blow: is the best FEV1 blow also the best FVC blow?
    f_arr = np.array([x["fev1"] for x in use]); v_arr = np.array([x["fvc"] for x in use])
    r["same_best_blow"] = float(int(np.argmax(f_arr)) == int(np.argmax(v_arr)))
    r["corr_fev1_fvc"] = float(np.corrcoef(f_arr, v_arr)[0, 1]) if len(f_arr) > 2 and f_arr.std() > 0 and v_arr.std() > 0 else 0.0
    # artefact counts over all blows
    r["mean_n_neg"] = float(np.mean([x["n_neg"] for x in bf]))
    r["max_n_rev"] = float(max(x["n_rev"] for x in bf))
    r["mean_eof_flow"] = float(np.mean([x["eof_flow"] for x in use]))
    # demographics and simple interactions
    def _num(x):
        return np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    r["age"] = _num(demo["age_years"])
    r["male"] = 1.0 if demo["sex"] == "M" else 0.0
    r["height"] = _num(demo["height_cm"])
    r["weight"] = _num(demo["weight_kg"])
    r["bmi"] = _num(demo["bmi"])
    r["ht2"] = r["height"] ** 2 / 10000.0
    r["age_x_male"] = r["age"] * r["male"]
    r["child"] = float(r["age"] < 18)
    return r, bf
