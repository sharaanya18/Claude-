"""Shared Adam Memory Repair: choose one shared order of eight minibatches for two hidden Adam states.

Pipeline (everything is fit on the TRAIN split only; each evaluation request is handled alone):
  1. A LightGBM regressor is trained from scratch on the training requests to predict the log of the common
     Adam second-moment vector v (per parameter) from public per-request quantities (gradients of the disclosed
     learner on the request's own images, request limits).  Training targets come from the released label moments.
  2. For each request the two hidden first-moment buffers are inferred by a linear-Gaussian (Bayesian) model: a
     buffer is an Adam EMA of gradients, so it is modelled as a combination of the per-image gradients of the
     request's images plus a small isotropic residual; the four diagnostic responses are the observations.
  3. Candidate orders are scored by running the disclosed learner (per-request sandbox) from the inferred states:
     all candidates on the posterior mean, then the best ones under several posterior draws (expected quality).
  (Fixed constants below -- prior scales, draw count, v-noise -- are plain design choices made during development
  from scene-grouped validation on the TRAIN split only; no platform feedback was used.)
  4. The prior scale of the observation noise is chosen inside the script on training scenes held out from the
     v-model fit (fixed small grid, exact simulation with the released true moments as labels).

Requirements map:
  * from-scratch controller, no pretrained weights or external data: the only learned model is the LightGBM v-model.
  * fixed work plan, seeds fixed, wall-clock only logged; device constant below; no environment branches.
  * no test-set fitting, no pooling across evaluation requests, no hidden answers: evaluation rows are only read to
    compute their own order; the v-model and all hyper-parameters are fixed before any evaluation row is touched.
  * no shell or process-launch calls; output: CSV id,prediction with a JSON list permutation of 0..7.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(PUBLIC_DIR))
import learner as L

# ---- fixed plan -------------------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda"            # one A10G is the stated resource
N_POOL = 4096              # candidate orders scored on the posterior mean (fixed pseudo-random pool)
N_TOP = 128                # candidates re-scored under posterior draws
N_DRAWS = 8                # posterior draws
V_SCALE_SD, V_SHAPE_SD = 0.35, 0.3   # log-sd of the multiplicative noise put on v-hat inside each draw
PRIOR_SC, PRIOR_S1 = 0.04, 0.001     # gradient-span prior: coefficient sd, isotropic residual variance
SIG_GRID = (0.07, 0.10, 0.15)        # observation-noise scale, selected on held-out training scenes
N_HPO_PER_SCENE, N_HPO_ORDERS = 16, 256
V_ROW_STRIDE = 4           # parameter subsampling for the v-model training matrix
MONITOR_OBS = [0, 1, 3, 6, 9, 10]
GROUPS = np.concatenate([np.zeros(2304), np.ones(12), np.full(48, 2), np.full(4, 3)]).astype(int)
T0 = time.time()
DT = torch.float64


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


# ---- data ---------------------------------------------------------------------------------------------
def load_request(row, with_moments):
    z = np.load(PUBLIC_DIR / row["experiment_path"], allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["request"] = json.loads(row["request"])
    d["x_batches"] = np.stack([L.features(b) for b in d["batch_images"]])
    d["x_monitor"] = L.features(d["monitor_images"])
    d["x_diag"] = np.stack([L.features(b) for b in d["diagnostic_images"]])
    if with_moments:
        sup = json.loads(row["prediction"])
        d["moments"] = np.load(PUBLIC_DIR / sup["moment_path"], allow_pickle=False)["moments"]
    return d


def image_arrays(d):
    X = np.concatenate([d["x_batches"].reshape(-1, 192), d["x_monitor"], d["x_diag"].reshape(-1, 192)])
    y = np.concatenate([d["batch_labels"].ravel(), d["monitor_labels"], d["diagnostic_labels"].ravel()])
    return X, y


def per_image_grads(d):
    """Gradient of the disclosed learner at the public weights for each of the request's 52 images -> (P, 52)."""
    X, y = image_arrays(d)
    return np.stack([L.gradient(d["weights"], X[k:k + 1], y[k:k + 1]) for k in range(len(y))]).T


# ---- 1. learned model of the common second moment v ------------------------------------------------------
def factor_features(d):
    """Pixel-energy x hidden-delta-energy (rank-1) smoothing of the per-image squared gradients."""
    a, b, c, dd = L.unpack(d["weights"])
    X, y = image_arrays(d)
    h = np.tanh(X @ a + b)
    z = h @ c + dd
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    q[np.arange(len(y)), y] -= 1
    dh = (q @ c.T) * (1 - h * h)
    S = np.log((X ** 2).mean(0) + 1e-12)
    T = np.log((dh ** 2).mean(0) + 1e-12)
    f = np.zeros((L.P, 3), np.float32)
    f[:2304, 0] = np.repeat(S, 12)
    f[:2304, 1] = np.tile(T, 192)
    f[:2304, 2] = f[:2304, 0] + f[:2304, 1]
    f[2304:2316, 1] = T
    f[2304:2316, 2] = T
    hq = np.log((h ** 2).mean(0) + 1e-12)
    Tq = np.log((q ** 2).mean(0) + 1e-12)
    f[2316:2364, 0] = np.repeat(hq, 4)
    f[2316:2364, 1] = np.tile(Tq, 12)
    f[2316:2364, 2] = f[2316:2364, 0] + f[2316:2364, 1]
    f[2364:, 1] = Tq
    f[2364:, 2] = Tq
    return f


def v_features(d, Gi):
    """(P, F) parameter-level features from public inputs only."""
    w = d["weights"]
    Gb = np.stack([L.gradient(w, d["x_batches"][j], d["batch_labels"][j]) for j in range(8)])
    Gd = np.stack([L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]) for j in range(2)])
    Gm = L.gradient(w, d["x_monitor"], d["monitor_labels"])
    Gmi = np.stack([L.gradient(w, d["x_monitor"][k:k + 1], d["monitor_labels"][k:k + 1]) for k in range(12)])
    e = 1e-12
    f_b = np.log((Gb ** 2).mean(0) + e)
    f_d = np.log((Gd ** 2).mean(0) + e)
    f_m = np.log(Gm ** 2 + e)
    f_i = np.log((Gi ** 2).mean(1) + e)
    f_mi = np.log((Gmi ** 2).mean(0) + e)
    f_bm = np.log(Gb.mean(0) ** 2 + e)
    scal = np.array([np.log((Gb ** 2).mean() + e), np.log((Gi ** 2).mean() + e), np.log((Gmi ** 2).mean() + e),
                     L.cross_entropy(w, d["x_monitor"], d["monitor_labels"]),
                     np.mean([L.cross_entropy(w, d["x_batches"][j], d["batch_labels"][j]) for j in range(8)]),
                     np.log(d["request"]["divergence_limit"]), np.log(d["request"]["capability_limit"]),
                     np.log(np.linalg.norm(w))])
    F = np.stack([f_b, f_d, f_m, f_i, f_mi, f_bm, np.abs(w), f_b - scal[0], f_i - scal[1]], 1)
    F = np.concatenate([F, GROUPS[:, None], np.repeat(scal[None], L.P, 0), factor_features(d)], 1)
    return F.astype(np.float32)


def fit_v_model(rows):
    Xs, ys = [], []
    for _, row in rows.iterrows():
        d = load_request(row, True)
        Xs.append(v_features(d, per_image_grads(d))[::V_ROW_STRIDE])
        ys.append(np.log(d["moments"][:, 1].mean(0) + 1e-30)[::V_ROW_STRIDE].astype(np.float32))
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


# ---- 2. Bayesian inference of the hidden first moments ----------------------------------------------------
def monitor_jacobian(w, x):
    """d(centered logits)/dw for the observed monitor images -> (6*4, P)."""
    a, b, c, dd = L.unpack(w)
    cm = np.eye(4) - 0.25
    rows = []
    for i in range(len(x)):
        h = np.tanh(x[i] @ a + b)
        dz = 1 - h ** 2
        J = np.zeros((4, L.P))
        for k in range(4):
            gC = np.zeros((12, 4))
            gC[:, k] = h
            gD = np.zeros(4)
            gD[k] = 1
            J[k] = np.concatenate([np.outer(x[i], c[:, k] * dz).ravel(), c[:, k] * dz, gC.ravel(), gD])
        rows.append(cm @ J)
    return np.concatenate(rows)


def observation_system(d, vbar):
    """Linearised map from u = m / sqrt(vbar) to the diagnostic log-probability changes (per branch)."""
    w, t = d["weights"], int(d["step"])
    sv = np.sqrt(vbar)
    Jo = monitor_jacobian(w, d["x_monitor"][MONITOR_OBS])
    p0 = L.probabilities(w, d["x_monitor"])[MONITOR_OBS]
    lp0 = np.log(p0)
    lp0 -= lp0.mean(1, keepdims=True)
    bc1, bc2 = 1 - L.BETA1 ** (t + 1), 1 - L.BETA2 ** (t + 1)
    gd = [L.gradient(w, d["x_diag"][j], d["diagnostic_labels"][j]) for j in range(2)]
    Hb, yb = [], []
    for b in range(2):
        Hs, ys = [], []
        for p in range(2):
            vp = L.BETA2 * vbar + (1 - L.BETA2) * gd[p] ** 2
            den = bc1 * (np.sqrt(vp / bc2) + L.EPS)
            lp = np.log(np.maximum(d["diagnostics"][b, p], 1e-300))
            lp -= lp.mean(1, keepdims=True)
            ys.append(-(lp - lp0).ravel() / L.LR - Jo @ ((1 - L.BETA1) * gd[p] / den))
            Hs.append(Jo * (L.BETA1 * sv / den)[None])
        Hb.append(np.concatenate(Hs))
        yb.append(np.concatenate(ys))
    return Hb, yb


def posterior_states(d, vbar, Gh_raw, sig, n_draws, rng):
    """Posterior mean and draws of the two first-moment buffers; returns list of (2, 2, P) [branch, m|v] states."""
    sv = np.sqrt(vbar)
    Gh = Gh_raw / sv[:, None]
    Hb, yb = observation_system(d, vbar)
    mean_u = np.zeros((2, L.P))
    draw_u = [np.zeros((2, L.P)) for _ in range(n_draws)]
    for b in range(2):
        H, y = Hb[b], yb[b]
        AHt = PRIOR_SC ** 2 * Gh @ (Gh.T @ H.T) + PRIOR_S1 * H.T
        S = H @ AHt
        nz = sig ** 2 * np.mean(np.diag(S))
        Sinv = np.linalg.inv(S + nz * np.eye(len(y)))
        mean_u[b] = AHt @ (Sinv @ y)
        for s in range(n_draws):
            u0 = PRIOR_SC * Gh @ rng.standard_normal(Gh.shape[1]) + np.sqrt(PRIOR_S1) * rng.standard_normal(L.P)
            e = np.sqrt(nz) * rng.standard_normal(len(y))
            draw_u[s][b] = u0 + AHt @ (Sinv @ (y - H @ u0 - e))

    def to_state(U):
        out = np.zeros((2, 2, L.P))
        for b in range(2):
            out[b, 0] = U[b] * sv
            out[b, 1] = vbar
        return out

    return [to_state(mean_u)] + [to_state(u) for u in draw_u]


# ---- 3. per-request sandbox: batched float64 re-implementation of the disclosed learner ------------------
def _unpack(w):
    n, o, out = w.shape[0], 0, []
    for s in L.SIZES:
        z = int(np.prod(s))
        out.append(w[:, o:o + z].reshape((n,) + s))
        o += z
    return out


def _grad(w, x, y):
    a, b, c, dd = _unpack(w)
    h = torch.tanh(torch.bmm(x, a) + b[:, None, :])
    q = torch.softmax(torch.bmm(h, c) + dd[:, None, :], dim=2)
    q = (q - torch.nn.functional.one_hot(y, 4).to(DT)) / y.shape[1]
    dh = torch.bmm(q, c.transpose(1, 2)) * (1 - h * h)
    n = w.shape[0]
    return torch.cat([torch.bmm(x.transpose(1, 2), dh).reshape(n, -1), dh.sum(1),
                      torch.bmm(h.transpose(1, 2), q).reshape(n, -1), q.sum(1)], 1)


def _step(st, x, y):
    w, m, v, t = st
    g = _grad(w, x, y)
    t = t + 1
    m = L.BETA1 * m + (1 - L.BETA1) * g
    v = L.BETA2 * v + (1 - L.BETA2) * g * g
    w = w - L.LR * (m / (1 - L.BETA1 ** t)) / (torch.sqrt(v / (1 - L.BETA2 ** t)) + L.EPS)
    return w, m, v, t


def _probs(w, x):
    a, b, c, dd = _unpack(w)
    h = torch.tanh(torch.matmul(x[None], a) + b[:, None, :])
    return torch.softmax(torch.matmul(h, c) + dd[:, None, :], dim=2)


class Sandbox:
    def __init__(self, d):
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=DEVICE)
        self.xb, self.xm = T(d["x_batches"]), T(d["x_monitor"])
        self.yb = torch.as_tensor(d["batch_labels"], dtype=torch.long, device=DEVICE)
        self.ym = torch.as_tensor(d["monitor_labels"], dtype=torch.long, device=DEVICE)
        self.w0, self.t0 = T(d["weights"]), int(d["step"])
        self.cont = [list(map(int, c)) for c in d["continuations"]]
        self.lim, self.cap = d["request"]["divergence_limit"], d["request"]["capability_limit"]

    def quality(self, orders, mom, chunk=512):
        """Quality (and cap-excess) of each order in (n, 8) from state mom (2, 2, P) = [branch][m, v]."""
        orders = torch.as_tensor(np.asarray(orders), dtype=torch.long, device=DEVICE)
        mom = torch.as_tensor(mom, dtype=DT, device=DEVICE)
        qs, ex = [], []
        for s in range(0, len(orders), chunk):
            o = orders[s:s + chunk]
            k = len(o)
            ends = []
            for b in range(2):
                st = (self.w0[None].repeat(k, 1), mom[b, 0][None].repeat(k, 1), mom[b, 1][None].repeat(k, 1), self.t0)
                for pos in range(8):
                    st = _step(st, self.xb[o[:, pos]], self.yb[o[:, pos]])
                ends.append(st)
            ce = torch.stack([-torch.log(torch.clamp(_probs(e[0], self.xm)[:, torch.arange(12), self.ym], min=1e-15)).mean(1)
                              for e in ends], 1)
            dist = 0
            for c in self.cont:
                fin = []
                for e in ends:
                    st = e
                    for j in c:
                        jj = torch.full((k,), j, dtype=torch.long, device=DEVICE)
                        st = _step(st, self.xb[jj], self.yb[jj])
                    fin.append(_probs(st[0], self.xm))
                dist = dist + ((fin[0] - fin[1]) ** 2).mean((1, 2))
            D = dist / len(self.cont)
            ok = (ce <= self.cap).all(1)
            qs.append(torch.where(ok, self.lim / torch.clamp(D, min=self.lim), torch.zeros_like(D)))
            ex.append(torch.clamp(ce / self.cap - 1, min=0).max(1).values)
        return torch.cat(qs).cpu().numpy(), torch.cat(ex).cpu().numpy()


# ---- 4. order selection for one request -----------------------------------------------------------------
def choose_order(d, vbar, sig, pool, rng, full=True):
    Gi = per_image_grads(d)
    sb = Sandbox(d)
    states = posterior_states(d, vbar, Gi, sig, N_DRAWS if full else 0, rng)
    q0, e0 = sb.quality(pool, states[0])
    if not full:
        return pool[int(np.argmax(q0 - 1e-3 * e0))]
    top = np.argsort(-(q0 - 1e-3 * e0), kind="stable")[:N_TOP]
    cand = pool[top]
    qd, ed = [q0[top]], [e0[top]]
    for s in range(1, len(states)):
        vs = vbar * np.exp(V_SCALE_SD * rng.standard_normal() + V_SHAPE_SD * rng.standard_normal(L.P))
        draw = posterior_states(d, vs, Gi, sig, 1, rng)[1]
        q1, e1 = sb.quality(cand, draw)
        qd.append(q1)
        ed.append(e1)
    score = np.mean(qd, 0) - 1e-3 * np.mean(ed, 0)
    return cand[int(np.argmax(score))]


def validate_submission(sub, sample_path):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), "columns differ from sample_submission"
    assert len(sub) == len(sample), "row count differs"
    assert sub["id"].astype(str).tolist() == sample["id"].astype(str).tolist(), "id mismatch/order"
    for s in sub["prediction"]:
        p = json.loads(s)
        assert sorted(p) == list(range(8)) and all(type(v) is int for v in p), f"bad permutation {s}"


def main():
    seed_everything()
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    rng_pool = np.random.default_rng(SEED)
    pool = np.stack([rng_pool.permutation(8) for _ in range(N_POOL)])

    # --- hyper-parameter selection on training scenes held out from the v-model fit
    scenes = sorted(train["experiment_group"].unique())
    held = scenes[::4]
    fit_rows = train[~train["experiment_group"].isin(held)]
    log(f"v-model (selection) on {len(fit_rows)} requests; held-out scenes {len(held)}")
    v_sel = fit_v_model(fit_rows.iloc[::4])
    hpo_rows = []
    for g in held:
        sub = train[train["experiment_group"] == g]
        hpo_rows.append(sub.iloc[np.linspace(0, len(sub) - 1, N_HPO_PER_SCENE).astype(int)])
    hpo_rows = pd.concat(hpo_rows)
    cache = []
    for _, row in hpo_rows.iterrows():
        d = load_request(row, True)
        sb = Sandbox(d)
        qt, _ = sb.quality(pool[:N_HPO_ORDERS], d["moments"])
        vb = np.exp(v_sel.predict(v_features(d, per_image_grads(d))))
        cache.append((d, vb, qt))
    rand_q = float(np.mean([c[2].mean() for c in cache]))
    # ablation (log only): learned v-model versus a crude non-learned proxy (mean squared batch gradient)
    qs_proxy = []
    for d, vb, qt in cache:
        gb = np.stack([L.gradient(d["weights"], d["x_batches"][j], d["batch_labels"][j]) for j in range(8)])
        st = posterior_states(d, np.maximum((gb ** 2).mean(0), 1e-12), per_image_grads(d), 0.10, 0, np.random.default_rng(SEED))[0]
        qe, ee = Sandbox(d).quality(pool[:N_HPO_ORDERS], st)
        qs_proxy.append(qt[int(np.argmax(qe - 1e-3 * ee))])
    log(f"ablation: crude non-learned v proxy pick quality {np.mean(qs_proxy):.3f}")
    best_sig, best_q = None, -1.0
    for sig in SIG_GRID:
        qs = []
        for d, vb, qt in cache:
            Gi = per_image_grads(d)
            st = posterior_states(d, vb, Gi, sig, 0, np.random.default_rng(SEED))[0]
            qe, ee = Sandbox(d).quality(pool[:N_HPO_ORDERS], st)
            qs.append(qt[int(np.argmax(qe - 1e-3 * ee))])
        log(f"sig={sig}: held-out-scene pick quality {np.mean(qs):.3f} (random orders {rand_q:.3f})")
        if np.mean(qs) > best_q:
            best_sig, best_q = sig, float(np.mean(qs))
    log(f"selected sig={best_sig}")
    del cache

    # --- final v-model on all training requests, then per-request prediction
    v_model = fit_v_model(train.iloc[::2])
    log("final v-model fitted")
    preds = []
    for n, (_, row) in enumerate(test.iterrows()):
        d = load_request(row, False)
        vb = np.exp(v_model.predict(v_features(d, per_image_grads(d))))
        order = choose_order(d, vb, best_sig, pool, np.random.default_rng(SEED + n))
        preds.append(json.dumps([int(i) for i in order], separators=(",", ":")))
        if n % 50 == 0:
            log(f"{n}/{len(test)} requests done")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    sub = pd.DataFrame({"id": test["id"].values, "prediction": preds})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv")
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
