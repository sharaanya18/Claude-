"""Shared library: trace decoding, spirometric indices, bootstrap target machinery, official metric.

Kept separate from the dataset directory on purpose (untrusted archive hygiene).
"""
import base64
import json
import numpy as np

DT = 0.01            # 10 ms sampling
BYTE_OFFSET = 62     # documented zero-flow byte
FEV1_SAMPLES = 100   # 1.00 s = 100 samples

TARGETS = ["fev1_00", "fev1_05", "fev1_10", "fev1_15", "fev1_20",
           "fvc_00", "fvc_05", "fvc_10", "ats"]
FEV1_THR = [0.00, 0.05, 0.10, 0.15, 0.20]
FVC_THR = [0.00, 0.05, 0.10]


def decode_blow(blow):
    """bytes -> (volume_l, flow_ls). Cast to float BEFORE subtracting (uint8 wraps)."""
    buf = base64.b64decode(blow["flow_b64"])
    raw = np.frombuffer(buf, np.uint8).astype(np.float64)
    delta_ml = raw - BYTE_OFFSET
    btps = float(blow["btps"])
    volume_l = np.cumsum(delta_ml) * btps / 1000.0
    flow_ls = delta_ml * btps / 10.0
    return volume_l, flow_ls


def back_extrap_i0(volume_l, flow_ls):
    """ATS back-extrapolation: line through the peak-flow sample with slope PEF, hit V=0."""
    ipk = int(np.argmax(flow_ls))
    pef = float(flow_ls[ipk])
    if pef <= 0:
        return 0, ipk, pef
    # slope per sample is pef * DT litres; solve volume[ipk] - slope*(ipk - i0) = 0
    i0 = ipk - volume_l[ipk] / (pef * DT)
    i0 = int(np.round(i0))
    i0 = max(0, min(i0, len(volume_l) - 1))
    return i0, ipk, pef


def blow_indices(blow):
    """Full per-blow index dictionary from one raw trace."""
    v, f = decode_blow(blow)
    n = len(v)
    i0, ipk, pef = back_extrap_i0(v, f)
    v0 = v[i0]
    j1 = i0 + FEV1_SAMPLES
    fev1 = (v[j1] if j1 < n else v[-1]) - v0
    vmax = float(np.max(v))
    imax = int(np.argmax(v))
    fvc = vmax - v0
    out = {
        "blow": int(blow["blow"]),
        "acceptable": 1 if blow["acceptable"] == "Y" else 0,
        "plateau": 1 if blow["plateau"] == "Y" else 0,
        "btps": float(blow["btps"]),
        "n_points": int(blow["n_points"]),
        "standing": 1 if blow.get("position") == "Standing" else 0,
        "i0": i0, "ipk": ipk, "n": n,
        "pef": pef,
        "fev1": float(fev1),
        "fvc": float(fvc),
        "vmax": vmax, "imax": imax,
        "v0": float(v0),
    }
    out["_v"] = v
    out["_f"] = f
    return out


def load_jsonl(path, with_demo=False):
    recs = []
    with open(path) as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------- metric ----
def brier_terms(pred, truth):
    """pred, truth: (N,9) arrays. Returns per-participant mean squared error over 9 events."""
    return np.mean((pred - truth) ** 2, axis=1)


def rcs(pred, truth):
    """Reversibility-agreement Brier half. ats is column 8 of TARGETS."""
    t_ats = truth[:, 8]
    w = 1.0 + 4.0 * t_ats * (1.0 - t_ats)
    e = brier_terms(pred, truth)
    L = np.sum(w * e) / np.sum(w)
    e_ref = brier_terms(np.full_like(truth, 0.5), truth)
    L_ref = np.sum(w * e_ref) / np.sum(w)
    return max(0.0, 1.0 - L / L_ref), L, L_ref


def somers_d(true_frag, pred_frag):
    """D = sum over ordered pairs sign(p_i-p_j)*sign(t_i-t_j) / number untied in TRUTH."""
    t = np.asarray(true_frag, dtype=np.float64)
    p = np.asarray(pred_frag, dtype=np.float64)
    st = np.sign(t[:, None] - t[None, :])
    sp = np.sign(p[:, None] - p[None, :])
    num = float(np.sum(st * sp))
    den = float(np.sum(st != 0))
    if den == 0:
        return 0.0
    return num / den


def fds(pred_ats, true_ats):
    pf = 4.0 * pred_ats * (1.0 - pred_ats)
    tf = 4.0 * true_ats * (1.0 - true_ats)
    return (1.0 + somers_d(tf, pf)) / 2.0


def official_score(pred, truth):
    """pred, truth: (N,9) in TARGETS order. Returns dict."""
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    truth = np.asarray(truth, dtype=np.float64)
    r, L, L_ref = rcs(pred, truth)
    f = fds(pred[:, 8], truth[:, 8])
    return {"score": float(np.sqrt(max(0.0, r) * max(0.0, f))),
            "rcs": float(r), "fds": float(f),
            "brier": float(np.mean(brier_terms(pred, truth))),
            "L": float(L), "L_ref": float(L_ref)}
