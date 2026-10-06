from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path


PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn


SEED = 42
DEVICE = "cpu"
T = 256
N_MARKERS = 22
FS = 150.0
DT = 1.0 / FS
G = 9.80665

WIDTH = 72
BLOCKS = 6
KERNEL = 5
DROPOUT = 0.05
EPOCHS = 28
BATCH = 32
LR = 3e-3
WEIGHT_DECAY = 1e-4
HUBER = 0.02
N_SEEDS = 3
N_PEOPLE = 32
HOLDOUT_PEOPLE = 7

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


COLS = ["sample_id", "force_x", "force_y", "force_z"]

REQUIRED = ["train.csv", "test.csv", "train_targets.csv", "sample_submission.csv"]


def resolve_public_dir(p: Path) -> Path:
    candidates = [p, p / "public", p.parent, Path("./dataset/public"), Path(".")]
    for c in candidates:
        try:
            if all((c / f).exists() for f in REQUIRED):
                return c
        except OSError:
            continue
    for c in candidates:
        try:
            if (c / "train.csv").exists() and (c / "test.csv").exists():
                return c
        except OSError:
            continue
    raise FileNotFoundError(
        f"none of {[str(c) for c in candidates]} contains {REQUIRED}")


def load_path(root: Path, rel: str) -> Path:
    rel = str(rel)
    for cand in (root / rel, root / Path(rel).name,
                 root / Path(rel).parent.name / Path(rel).name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"motion file not found for {rel!r} under {root}")


def coerce_motion(a: np.ndarray, rel: str) -> np.ndarray:
    if a.ndim == 3:
        if a.shape == (N_MARKERS, 3, T):
            return np.ascontiguousarray(a.transpose(2, 0, 1))
        if a.shape == (3, N_MARKERS, T):
            return np.ascontiguousarray(a.transpose(2, 1, 0))
        if a.shape[1:] == (N_MARKERS, 3):
            out = np.zeros((T, N_MARKERS, 3), dtype=np.float32)
            n = min(T, a.shape[0])
            out[:n] = a[:n]
            if n < T:
                out[n:] = a[n - 1]
            return out
    raise ValueError(f"{rel}: cannot interpret motion array of shape {a.shape}")


def parse_waveform(cell) -> np.ndarray:
    v = json.loads(cell) if isinstance(cell, str) else list(cell)
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == T:
        return v
    out = np.zeros(T, dtype=np.float64)
    n = min(T, v.size)
    out[:n] = v[:n]
    return out


MARKERS = [
    "R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "L.Iliac.Crest", "R.Iliac.Crest",
    "R.GTR", "R.Knee", "R.HF", "R.TT", "R.Ankle", "R.Heel", "R.MT1", "R.MT5",
    "L.GTR", "L.Knee", "L.HF", "L.TT", "L.Ankle", "L.Heel", "L.MT1", "L.MT5",
]
MI = {m: i for i, m in enumerate(MARKERS)}

MIRROR_IDX = [MI[("L." + m[2:]) if m.startswith("R.") else ("R." + m[2:])] for m in MARKERS]
PELVIS4 = [MI["R.ASIS"], MI["L.ASIS"], MI["R.PSIS"], MI["L.PSIS"]]


SEG_COM = [
    (0.497, PELVIS4),
    (0.100, [MI["R.GTR"], MI["R.Knee"]]),
    (0.100, [MI["L.GTR"], MI["L.Knee"]]),
    (0.0465, [MI["R.Knee"], MI["R.Ankle"]]),
    (0.0465, [MI["L.Knee"], MI["L.Ankle"]]),
    (0.0145, [MI["R.Heel"], MI["R.MT1"], MI["R.MT5"]]),
    (0.0145, [MI["L.Heel"], MI["L.MT1"], MI["L.MT5"]]),
]


def savgol_coeffs(window: int, order: int, deriv: int) -> np.ndarray:
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(t, order + 1, increasing=True)
    return np.linalg.pinv(A)[deriv] * math.factorial(deriv)


def sg(x: np.ndarray, window: int, deriv: int, order: int = 3) -> np.ndarray:
    c = savgol_coeffs(window, order, deriv) / DT ** deriv
    half = window // 2
    xp = np.concatenate([np.repeat(x[:, :1], half, axis=1), x,
                         np.repeat(x[:, -1:], half, axis=1)], axis=1)
    out = np.zeros(x.shape, dtype=np.float64)
    for k in range(window):
        out += c[k] * xp[:, k:k + x.shape[1]]
    return out


W_POS, W_VEL, W_ACC = 9, 15, 31
PRIOR_CH = [3, 4, 5]


def com_proxy(X: np.ndarray) -> np.ndarray:
    tot = sum(w for w, _ in SEG_COM)
    com = np.zeros((X.shape[0], T, 3), dtype=np.float64)
    for w, idx in SEG_COM:
        com += (w / tot) * X[:, :, idx, :].mean(axis=2)
    return com


def mirror_motion(X: np.ndarray) -> np.ndarray:
    Xm = X[:, :, MIRROR_IDX, :].copy()
    Xm[..., 2] *= -1.0
    return Xm


def mirror_force(Y: np.ndarray) -> np.ndarray:
    Ym = Y.copy()
    Ym[:, 2] *= -1.0
    return Ym


KEY = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.GTR", "L.GTR",
       "R.Knee", "L.Knee", "R.Ankle", "L.Ankle", "R.Heel", "L.Heel",
       "R.MT1", "L.MT1", "R.MT5", "L.MT5", "R.TT", "L.TT",
       "R.Iliac.Crest", "L.Iliac.Crest", "R.HF", "L.HF"]
FOOT = ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5",
        "R.Ankle", "L.Ankle", "R.Knee", "L.Knee", "R.TT", "L.TT"]


def frame_features(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    Xd = X.astype(np.float64)
    pel = Xd[:, :, PELVIS4, :].mean(axis=2)
    rel = Xd - pel[:, :, None, :]

    com = com_proxy(Xd)
    com_v = sg(com, W_VEL, 1)
    com_a15, com_a31 = sg(com, W_VEL, 2), sg(com, W_ACC, 2)
    pel_v = sg(pel, W_VEL, 1)
    pel_a15, pel_a31 = sg(pel, W_VEL, 2), sg(pel, W_ACC, 2)

    f = []


    for a in (com_a15, com_a31, pel_a15, pel_a31):
        f += [a[:, :, 0] / G, a[:, :, 1] / G + 1.0, a[:, :, 2] / G]

    f += [com_v[:, :, k] for k in range(3)]
    f += [pel_v[:, :, k] for k in range(3)]
    f += [com[:, :, 1], pel[:, :, 1],
          np.linalg.norm(com_v[:, :, [0, 2]], axis=-1)]

    relp = sg(rel.reshape(N, T, -1), W_POS, 0).reshape(N, T, N_MARKERS, 3)
    for m in (MI[k] for k in KEY):
        f += [relp[:, :, m, k] for k in range(3)]


    mv = sg(Xd.reshape(N, T, -1), W_VEL, 1).reshape(N, T, N_MARKERS, 3)
    ma = sg(Xd.reshape(N, T, -1), W_VEL, 2).reshape(N, T, N_MARKERS, 3)
    for m in (MI[k] for k in FOOT):
        for k in range(3):
            f += [mv[:, :, m, k], ma[:, :, m, k] / G]


    for nm in ["R.Heel", "L.Heel", "R.MT1", "L.MT1", "R.MT5", "L.MT5"]:
        m = MI[nm]
        h = Xd[:, :, m, 1]
        f += [h - h.min(axis=1, keepdims=True),
              np.linalg.norm(mv[:, :, m, :], axis=-1),
              rel[:, :, m, 0]]
    f += [Xd[:, :, MI["R.Heel"], 0] - Xd[:, :, MI["L.Heel"], 0],
          Xd[:, :, MI["R.MT1"], 0] - Xd[:, :, MI["L.MT1"], 0],
          Xd[:, :, MI["R.Heel"], 1] - Xd[:, :, MI["L.Heel"], 1],
          Xd[:, :, MI["R.Ankle"], 2] - Xd[:, :, MI["L.Ankle"], 2]]


    def ang(a, b, c):
        u = Xd[:, :, MI[a], :] - Xd[:, :, MI[b], :]
        v = Xd[:, :, MI[c], :] - Xd[:, :, MI[b], :]
        return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)

    for trip in [("R.ASIS", "R.GTR", "R.Knee"), ("L.ASIS", "L.GTR", "L.Knee"),
                 ("R.GTR", "R.Knee", "R.Ankle"), ("L.GTR", "L.Knee", "L.Ankle"),
                 ("R.Knee", "R.Ankle", "R.MT1"), ("L.Knee", "L.Ankle", "L.MT1")]:
        f.append(ang(*trip))
    return np.stack(f, axis=1).astype(np.float32)


RIGID = [("R.ASIS", "L.ASIS"), ("R.ASIS", "R.PSIS"), ("L.ASIS", "L.PSIS"),
         ("R.Iliac.Crest", "L.Iliac.Crest"), ("R.GTR", "L.GTR"),
         ("R.GTR", "R.Knee"), ("L.GTR", "L.Knee"), ("R.Knee", "R.Ankle"),
         ("L.Knee", "L.Ankle"), ("R.TT", "R.Ankle"), ("L.TT", "L.Ankle"),
         ("R.Heel", "R.MT1"), ("L.Heel", "L.MT1"), ("R.Heel", "R.MT5"),
         ("L.Heel", "L.MT5"), ("R.MT1", "R.MT5"), ("L.MT1", "L.MT5"),
         ("R.ASIS", "R.GTR"), ("L.ASIS", "L.GTR"), ("R.Knee", "R.HF"),
         ("L.Knee", "L.HF"), ("R.ASIS", "R.Knee"), ("L.ASIS", "L.Knee"),
         ("R.PSIS", "R.Iliac.Crest"), ("L.PSIS", "L.Iliac.Crest")]


def segment_lengths(X: np.ndarray) -> np.ndarray:
    Xd = X.astype(np.float64)
    out = np.empty((len(X), len(RIGID)), dtype=np.float64)
    for j, (m1, m2) in enumerate(RIGID):
        d = np.linalg.norm(Xd[:, :, MI[m1], :] - Xd[:, :, MI[m2], :], axis=-1)
        out[:, j] = np.median(d, axis=1)
    return out


def trial_features(X: np.ndarray) -> np.ndarray:
    Xd = X.astype(np.float64)
    S = segment_lengths(X)

    def s(m1, m2):
        return S[:, RIGID.index((m1, m2))]

    thigh = 0.5 * (s("R.GTR", "R.Knee") + s("L.GTR", "L.Knee"))
    shank = 0.5 * (s("R.Knee", "R.Ankle") + s("L.Knee", "L.Ankle"))
    foot = 0.5 * (s("R.Heel", "R.MT1") + s("L.Heel", "L.MT1"))
    pelw, gtrw = s("R.ASIS", "L.ASIS"), s("R.GTR", "L.GTR")
    crestw = s("R.Iliac.Crest", "L.Iliac.Crest")
    pdep = 0.5 * (s("R.ASIS", "R.PSIS") + s("L.ASIS", "L.PSIS"))
    leg = thigh + shank

    pel = Xd[:, :, PELVIS4, :].mean(axis=2)
    footy = np.minimum(Xd[:, :, MI["R.Heel"], 1], Xd[:, :, MI["L.Heel"], 1])
    pelh = np.median(pel[:, :, 1] - footy, axis=1)

    com = com_proxy(Xd)
    cv, ca = sg(com, W_VEL, 1), sg(com, W_VEL, 2)
    speed = np.linalg.norm(cv[:, :, [0, 2]], axis=-1).mean(axis=1)


    v = pel[:, :, 1] - pel[:, :, 1].mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(v * np.hanning(T)[None, :], axis=1))
    frq = np.fft.rfftfreq(T, DT)
    band = (frq > 0.6) & (frq < 4.0)
    cad = frq[band][np.argmax(spec[:, band], axis=1)]

    cols = [thigh, shank, foot, pelw, gtrw, crestw, pdep, leg, pelh,
            thigh / leg, shank / leg, foot / leg, pelw / leg, gtrw / leg, pelh / leg,
            speed, cad, speed / np.maximum(leg, 1e-6), speed / np.maximum(cad, 1e-6),
            np.abs(ca).mean(axis=(1, 2)) / G,
            ca[:, :, 1].std(axis=1) / G, ca[:, :, 0].std(axis=1) / G,
            ca[:, :, 2].std(axis=1) / G,
            np.ptp(pel[:, :, 1], axis=1),
            np.ptp(Xd[:, :, MI["R.Heel"], 1], axis=1),
            np.ptp(Xd[:, :, MI["L.Heel"], 1], axis=1),
            Xd[:, -1, :, 0].mean(axis=1) - Xd[:, 0, :, 0].mean(axis=1)]
    return np.stack(cols, axis=1).astype(np.float32)


def derive_person_groups(S: np.ndarray, n_people: int = N_PEOPLE) -> np.ndarray:
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    n = len(Z)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)

    members = {i: [i] for i in range(n)}
    Dm = D.copy()
    while len(members) > n_people:
        keys = sorted(members)
        sub = Dm[np.ix_(keys, keys)]
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        ka, kb = keys[a], keys[b]
        na, nb = len(members[ka]), len(members[kb])

        newd = (na * Dm[ka, :] + nb * Dm[kb, :]) / (na + nb)
        members[ka] = members[ka] + members[kb]
        del members[kb]
        Dm[ka, :] = newd
        Dm[:, ka] = newd
        Dm[ka, ka] = np.inf
        Dm[kb, :] = np.inf
        Dm[:, kb] = np.inf
    lab = np.empty(n, dtype=np.int64)
    for c, (_, idx) in enumerate(sorted(members.items())):
        lab[idx] = c
    return lab


class ResBlock(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        pad = dil * (KERNEL - 1) // 2
        self.c1 = nn.Conv1d(c, c, KERNEL, padding=pad, dilation=dil)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, KERNEL, padding=pad, dilation=dil)
        self.n2 = nn.GroupNorm(8, c)
        self.do = nn.Dropout(DROPOUT)

    def forward(self, x):
        h = Fn.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(self.do(h)))
        return Fn.gelu(x + h)


class ForceTCN(nn.Module):

    def __init__(self, c_in: int, n_scalar: int):
        super().__init__()
        self.stem = nn.Conv1d(c_in + n_scalar, WIDTH, KERNEL, padding=KERNEL // 2)
        self.sn = nn.GroupNorm(8, WIDTH)
        self.blocks = nn.ModuleList([ResBlock(WIDTH, min(2 ** i, 32)) for i in range(BLOCKS)])
        self.head = nn.Sequential(nn.Conv1d(WIDTH, WIDTH, 1), nn.GELU(),
                                  nn.Conv1d(WIDTH, 3, 1))


        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, xf, xs, prior):
        x = torch.cat([xf, xs[:, :, None].expand(-1, -1, xf.shape[2])], dim=1)
        h = Fn.gelu(self.sn(self.stem(x)))
        for b in self.blocks:
            h = b(h)
        return self.head(h) + prior


def metric_loss(pred, truth, den, huber: float = 0.0):
    r = pred - truth
    if huber > 0:
        a = r.abs()
        r = torch.where(a < huber, 0.5 * r * r / huber + 0.5 * huber, a)
    else:
        r = r.abs()
    return (r.sum(dim=2) / den.clamp_min(1e-6)).mean()


def train_fold(FFtr, SStr, Ytr, PRtr, seed: int, tag: str):
    seed_everything(seed)
    model = ForceTCN(FFtr.shape[1], SStr.shape[1]).to(DEVICE)
    Xf = torch.from_numpy(FFtr)
    Xs = torch.from_numpy(SStr)
    Yt = torch.from_numpy(Ytr.astype(np.float32))
    Pr = torch.from_numpy(PRtr.astype(np.float32))
    den = Yt.abs().sum(dim=2)

    n = Xf.shape[0]
    steps = ((n + BATCH - 1) // BATCH) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.15,
        div_factor=10.0, final_div_factor=100.0)
    gen = torch.Generator().manual_seed(seed)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for st in range(0, n, BATCH):
            b = perm[st:st + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = metric_loss(model(Xf[b], Xs[b], Pr[b]), Yt[b], den[b], HUBER)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        if ep % 7 == 0 or ep == EPOCHS - 1:
            log(f"    {tag} epoch {ep:3d}/{EPOCHS}  train normalised-L1 {tot / n:.4f}")
    model.eval()
    return model


def predict(model, FF, SS, PR, batch: int = 64) -> np.ndarray:
    out = np.empty((len(FF), 3, T), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(FF), batch):
            e = min(s + batch, len(FF))
            out[s:e] = model(torch.from_numpy(FF[s:e]),
                             torch.from_numpy(SS[s:e]),
                             torch.from_numpy(PR[s:e].astype(np.float32))).numpy()
    return out


def axis_skill(pred, truth) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    baseline = np.abs(truth).sum()
    err = np.abs(pred - truth).sum()
    return float(err == 0) if baseline == 0 else max(0.0, 1.0 - err / baseline)


def official_score(pred, truth):
    per = np.empty((len(pred), 3))
    for i in range(len(pred)):
        for a in range(3):
            per[i, a] = axis_skill(pred[i, a], truth[i, a])
    return per.mean(), per.mean(axis=0), per


def check_submission(sub: pd.DataFrame, n_expected: int) -> None:
    if list(sub.columns) != COLS:
        raise ValueError(f"columns {list(sub.columns)} != {COLS}")
    if len(sub) != n_expected:
        raise ValueError(f"rows {len(sub)} != test rows {n_expected}")
    if not sub["sample_id"].is_unique:
        raise ValueError("duplicate sample_id")
    for col in COLS[1:]:
        for i, cell in enumerate(sub[col].tolist()):
            if not isinstance(cell, str) or not cell:
                raise ValueError(f"empty cell in {col} row {i}")
            arr = json.loads(cell)
            if not isinstance(arr, list) or len(arr) != T:
                raise ValueError(f"{col} row {i}: length {len(arr)} != {T}")
            if not np.isfinite(np.asarray(arr, dtype=float)).all():
                raise ValueError(f"{col} row {i}: non-finite value")
    log(f"submission checked: {len(sub)} rows x 3 axes x {T} finite values")


def to_json_row(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main() -> None:
    seed_everything()
    log(f"public_dir={PUBLIC_DIR}  submission_out={SUBMISSION_OUT}  device={DEVICE}")


    root = resolve_public_dir(PUBLIC_DIR)
    log(f"resolved data root: {root}")

    tr_man = pd.read_csv(root / "train.csv")
    te_man = pd.read_csv(root / "test.csv")
    tg = pd.read_csv(root / "train_targets.csv")
    sample = pd.read_csv(root / "sample_submission.csv", keep_default_na=False)

    tg = tg.drop_duplicates(subset="sample_id").set_index("sample_id")
    log(f"train trials {len(tr_man)}  test trials {len(te_man)}  "
        f"sample rows {len(sample)}")


    def load_split(man):
        A = np.zeros((len(man), T, N_MARKERS, 3), dtype=np.float32)
        for k, rel in enumerate(man["motion_file"]):
            a = np.asarray(np.load(load_path(root, rel)), dtype=np.float32)


            if a.shape != (T, N_MARKERS, 3):
                a = coerce_motion(a, rel)
            A[k] = a
        n_bad = int((~np.isfinite(A)).sum())
        if n_bad:


            log(f"  note: {n_bad} non-finite marker values replaced with 0")
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        return A

    Xtr, Xte = load_split(tr_man), load_split(te_man)

    Ytr = np.zeros((len(tr_man), 3, T), dtype=np.float64)
    missing = 0
    for k, sid in enumerate(tr_man["sample_id"]):
        if sid not in tg.index:
            missing += 1
            continue
        for a, col in enumerate(["force_x", "force_y", "force_z"]):
            Ytr[k, a] = parse_waveform(tg.at[sid, col])
    if missing:
        log(f"  note: {missing} training ids had no target row and are zero-filled")
    n_bad = int((~np.isfinite(Ytr)).sum())
    if n_bad:
        log(f"  note: {n_bad} non-finite target values replaced with 0")
        Ytr = np.nan_to_num(Ytr, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"loaded motion {Xtr.shape} / {Xte.shape} and targets {Ytr.shape}")


    FFtr, SStr = frame_features(Xtr), trial_features(Xtr)
    FFte, SSte = frame_features(Xte), trial_features(Xte)


    Xm = mirror_motion(Xtr)
    FFmr, SSmr = frame_features(Xm), trial_features(Xm)
    Ymr = mirror_force(Ytr)
    log(f"features: frame {FFtr.shape[1]} channels, trial scalars {SStr.shape[1]}")

    PRtr = np.ascontiguousarray(FFtr[:, PRIOR_CH, :])
    PRte = np.ascontiguousarray(FFte[:, PRIOR_CH, :])
    PRmr = np.ascontiguousarray(FFmr[:, PRIOR_CH, :])


    n_people = int(min(N_PEOPLE, max(2, len(Xtr) // 8)))
    lab = derive_person_groups(segment_lengths(Xtr), n_people)
    sizes = np.bincount(lab)
    log(f"derived {len(sizes)} person groups, sizes min {sizes.min()} "
        f"max {sizes.max()} median {int(np.median(sizes))}")
    order = np.argsort(-sizes)
    n_hold = int(min(HOLDOUT_PEOPLE, max(1, len(sizes) // 5)))
    hold_people = set(order[:n_hold].tolist())
    va = np.array([i for i in range(len(lab)) if lab[i] in hold_people], dtype=int)
    trn = np.array([i for i in range(len(lab)) if lab[i] not in hold_people], dtype=int)


    run_holdout = len(trn) >= 32 and len(va) >= 4
    log(f"validation sets aside {len(hold_people)} whole persons "
        f"({len(va)} trials); training on {len(trn)}"
        f"{'' if run_holdout else '  [too small -- diagnostic skipped]'}")


    def fit_std(FF, SS, idx):
        fm = FF[idx].mean(axis=(0, 2), keepdims=True).astype(np.float32)
        fs = (FF[idx].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        sm = SS[idx].mean(axis=0, keepdims=True).astype(np.float32)
        ss = (SS[idx].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        return fm, fs, sm, ss

    if run_holdout:
        fm, fs, sm, ss = fit_std(FFtr, SStr, trn)
        mdl = train_fold(
            np.concatenate([(FFtr[trn] - fm) / fs, (FFmr[trn] - fm) / fs]),
            np.concatenate([(SStr[trn] - sm) / ss, (SSmr[trn] - sm) / ss]),
            np.concatenate([Ytr[trn], Ymr[trn]]),
            np.concatenate([PRtr[trn], PRmr[trn]]),
            seed=SEED, tag="holdout")
        vp = predict(mdl, (FFtr[va] - fm) / fs, (SStr[va] - sm) / ss, PRtr[va])
        vtot, vax, vper = official_score(vp, Ytr[va])
        log(f"person-disjoint TRAIN validation, official metric: TOTAL {vtot:.4f}  "
            f"x {vax[0]:.4f}  y {vax[1]:.4f}  z {vax[2]:.4f}")
        pp = sorted(float(vper[lab[va] == c].mean()) for c in hold_people)
        log(f"  per-unseen-person scores: {' '.join(f'{v:.3f}' for v in pp)}")


        med = np.median(Ytr[trn], axis=0)
        mtot, _, _ = official_score(np.repeat(med[None], len(va), axis=0), Ytr[va])
        log(f"  reference: best constant (median) waveform scores {mtot:.4f} "
            f"-> model adds {vtot - mtot:+.4f}")
        if vtot <= mtot:


            log("  WARNING: the model did not beat the constant-waveform baseline")


    fm, fs, sm, ss = fit_std(FFtr, SStr, np.arange(len(FFtr)))
    FFa = np.concatenate([(FFtr - fm) / fs, (FFmr - fm) / fs])
    SSa = np.concatenate([(SStr - sm) / ss, (SSmr - sm) / ss])
    Ya = np.concatenate([Ytr, Ymr])
    PRa = np.concatenate([PRtr, PRmr])
    FFteS, SSteS = (FFte - fm) / fs, (SSte - sm) / ss

    preds = np.zeros((len(FFte), 3, T), dtype=np.float64)
    for si in range(N_SEEDS):
        log(f"  full-data refit, seed {si + 1}/{N_SEEDS}")
        m = train_fold(FFa, SSa, Ya, PRa, seed=SEED + 101 * si, tag=f"full{si}")
        preds += predict(m, FFteS, SSteS, PRte) / N_SEEDS
    log(f"averaged {N_SEEDS} full-data models")


    log(f"  prediction stats   force_x mean {preds[:,0].mean():+.4f} sd {preds[:,0].std():.4f}"
        f" | train mean {Ytr[:,0].mean():+.4f} sd {Ytr[:,0].std():.4f}")
    log(f"                     force_y mean {preds[:,1].mean():+.4f} sd {preds[:,1].std():.4f}"
        f" | train mean {Ytr[:,1].mean():+.4f} sd {Ytr[:,1].std():.4f}")
    log(f"                     force_z mean {preds[:,2].mean():+.4f} sd {preds[:,2].std():.4f}"
        f" | train mean {Ytr[:,2].mean():+.4f} sd {Ytr[:,2].std():.4f}")


    n_bad = int((~np.isfinite(preds)).sum())
    if n_bad:
        log(f"  WARNING: {n_bad} non-finite predicted values replaced with 0")
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    if preds.std() <= 1e-6:
        log("  WARNING: predictions are nearly constant")


    rows = {"sample_id": te_man["sample_id"].astype(str).tolist()}
    for a, col in enumerate(COLS[1:]):
        rows[col] = [to_json_row(preds[i, a]) for i in range(len(preds))]
    sub = pd.DataFrame(rows)[COLS]
    if list(sample.columns) == COLS and \
            set(sample["sample_id"].astype(str)) == set(sub["sample_id"]):
        sub = sub.set_index("sample_id").loc[
            sample["sample_id"].astype(str)].reset_index()
    else:
        log("  note: sample_submission ids/columns differ from test.csv; "
            "writing test.csv order (the grader ignores row order)")

    check_submission(sub, len(te_man))
    tmp = SUBMISSION_OUT.with_suffix(".csv.tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, SUBMISSION_OUT)
    back = pd.read_csv(SUBMISSION_OUT, keep_default_na=False)
    check_submission(back, len(te_man))
    log(f"wrote {SUBMISSION_OUT} shape={back.shape}")


if __name__ == "__main__":
    main()
