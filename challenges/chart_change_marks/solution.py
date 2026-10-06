"""Hospital data-quality charts: consensus change marks + calibrated weak/split/firm odds.

Usage: python3 solution.py <public_dir> <submission_out>

Pipeline (all learning happens in this script, from the raw PNGs + train.csv, every run):
  1. Read each binary scatter PNG and turn it into per-column ink profiles (a 1-D "image" along time).
  2. Train a 1-D convolutional U-Net with a BiGRU bottleneck (from scratch, no pretrained weights) that
     outputs, for every pixel column, (a) a heat-map of "a panel mark is here" and (b) the weak/split/firm
     bracket logits. Chart metadata (n_points, aggunit, aggfn) conditions the network.
  3. 5-fold cross-validation over the supplied folds.csv (each fold = whole dataset x fieldtype cells, i.e. the
     same cross-cell transfer as the hidden test). Out-of-fold heat-maps are used to choose ONE decision threshold
     and the bracket-calibration constants for the exact challenge metric; the five fold models are averaged
     (per chart, one chart at a time) to predict the test charts.

Requirements map (CPU only, 10 cores, 90 min; no external data; train.csv only; no pseudo-labels;
no pooling across test charts; test rows are only ever passed through the trained networks one chart at a time):
  - training: _train_fold() ; thresholds/calibration: fit on OOF only ; fixed epochs/seeds/threads, no clock logic.
"""
import os, sys, json, random, time
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")

# ----------------------------- fixed work plan -----------------------------
SEED = 1234
NUM_THREADS = 4           # fixed; set to the real core count of the platform before submitting
N_FOLDS = 5
EPOCHS = 24
BATCH = 16
LR = 2e-3
WD = 1e-2
WMAX = 928                # widest chart is 920 px
H0, H1, BIN = 18, 282, 4  # plot rows 18..281 -> 66 bins of 4 px
NB = (H1 - H0) // BIN
SIGMA = 2.5               # gaussian width (px) of the heat-map target
WIDTH_LEFT = 38           # plot area starts at column 38 and ends at W-39 (rendering is identical for all charts)
AGGFNS = ['subcat_perc', 'subcat_n', 'n', 'missing_perc', 'missing_n', 'distinct', 'min', 'max', 'midnight_n',
          'midnight_perc', 'nonconformant_n', 'nonconformant_perc', 'meanlength', 'minlength', 'maxlength', 'mean',
          'median', 'nonzero_perc', 'sum']
AGGUNITS = ['day', 'week', 'month']
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(NUM_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------- features -----------------------------
def load_chart(path):
    """Binary PNG -> (NB+2, W) float32 per-column features: ink fraction per 4-px row bin, log ink count, plot mask."""
    a = np.array(Image.open(path).convert("L")) < 128           # True = ink
    W = a.shape[1]
    ink = a[H0:H1].reshape(NB, BIN, W).mean(1)                   # (NB, W) fraction of ink in each row bin
    cnt = np.log1p(a[H0:H1].sum(0))[None] / 4.0                  # (1, W)
    mask = np.zeros((1, W)); mask[0, WIDTH_LEFT:W - WIDTH_LEFT] = 1.0
    return np.concatenate([ink, cnt, mask], 0).astype(np.float32), W


def build_meta(df, widths):
    out = np.zeros((len(df), 2 + len(AGGUNITS) + len(AGGFNS)), dtype=np.float32)
    for i, (r, w) in enumerate(zip(df.itertuples(), widths)):
        out[i, 0] = np.log(r.n_points) / 8.0
        out[i, 1] = np.log(r.n_points / (w - 2 * WIDTH_LEFT)) / 4.0   # timepoints per pixel
        out[i, 2 + AGGUNITS.index(r.aggunit)] = 1.0
        out[i, 2 + len(AGGUNITS) + AGGFNS.index(r.aggfn)] = 1.0
    return out


def load_split(df, img_dir):
    feats, widths = [], []
    for name in df["image"]:
        f, w = load_chart(Path(img_dir) / name)
        feats.append(f); widths.append(w)
    X = np.zeros((len(df), NB + 2, WMAX), dtype=np.float32)
    for i, f in enumerate(feats):
        X[i, :, :f.shape[1]] = f
    return X, np.array(widths), build_meta(df, widths)


def parse_marks(s):
    ms = json.loads(s)
    return [(float(m["x"]), int(np.argmax(m["p"]))) for m in ms]


# ----------------------------- model -----------------------------
class Res(nn.Module):
    def __init__(self, c, k=5, d=1):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d)
        self.c2 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d)
        self.n1 = nn.GroupNorm(8, c); self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class Net(nn.Module):
    """1-D U-Net over the time (pixel-column) axis. The stem halves the resolution (stride 2); the heads are
    upsampled back to one output per pixel column. A BiGRU at 1/16 resolution gives a chart-wide context."""

    def __init__(self, cin, nmeta, ch=(48, 64, 96, 128), hid=96):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch[0], 6, stride=2, padding=2)
        self.e0 = Res(ch[0])
        self.d1 = nn.Conv1d(ch[0], ch[1], 4, stride=2, padding=1); self.e1 = Res(ch[1])
        self.d2 = nn.Conv1d(ch[1], ch[2], 4, stride=2, padding=1); self.e2 = Res(ch[2])
        self.d3 = nn.Conv1d(ch[2], ch[3], 4, stride=2, padding=1); self.e3 = Res(ch[3])
        self.meta = nn.Linear(nmeta, ch[3])
        self.gru = nn.GRU(ch[3], hid, batch_first=True, bidirectional=True)
        self.gp = nn.Linear(2 * hid, ch[3])
        self.u2 = nn.ConvTranspose1d(ch[3], ch[2], 4, stride=2, padding=1); self.f2 = Res(ch[2])
        self.u1 = nn.ConvTranspose1d(ch[2], ch[1], 4, stride=2, padding=1); self.f1 = Res(ch[1])
        self.u0 = nn.ConvTranspose1d(ch[1], ch[0], 4, stride=2, padding=1); self.f0 = Res(ch[0])
        self.up = nn.ConvTranspose1d(ch[0], 24, 4, stride=2, padding=1)
        self.head = nn.Sequential(nn.GELU(), nn.Conv1d(24, 4, 5, padding=2))

    def forward(self, x, m):
        x0 = self.e0(self.inp(x))
        x1 = self.e1(self.d1(x0))
        x2 = self.e2(self.d2(x1))
        x3 = self.e3(self.d3(x2))
        x3 = x3 + self.meta(m)[:, :, None]
        g, _ = self.gru(x3.transpose(1, 2))
        x3 = x3 + self.gp(g).transpose(1, 2)
        y = self.f2(self.u2(x3) + x2)
        y = self.f1(self.u1(y) + x1)
        y = self.f0(self.u0(y) + x0)
        return self.head(self.up(y))      # (B, 4, W): heat logit, 3 bracket logits


# ----------------------------- batching / targets -----------------------------
def make_targets(marks_list, widths, flips):
    B = len(marks_list)
    heat = np.zeros((B, WMAX), np.float32)
    brk = np.zeros((B, 3, WMAX), np.float32)
    bw = np.zeros((B, WMAX), np.float32)
    cols = np.arange(WMAX, dtype=np.float32)
    for i, (ms, w, fl) in enumerate(zip(marks_list, widths, flips)):
        for x, b in ms:
            if fl:
                x = (w - 1) - x
            g = np.exp(-0.5 * ((cols - x) / SIGMA) ** 2)
            heat[i] = np.maximum(heat[i], g)
            near = np.exp(-0.5 * ((cols - x) / 1.5) ** 2)
            upd = near > bw[i]
            bw[i][upd] = near[upd]
            brk[i, :, upd] = 0
            brk[i, b, upd] = 1
    return heat, brk, bw


def augment_batch(X, widths, flips, vflips):
    X = X.copy()
    for i, (w, fl, vf) in enumerate(zip(widths, flips, vflips)):
        if fl:
            X[i, :, :w] = X[i, :, :w][:, ::-1]
        if vf:
            X[i, :NB] = X[i, :NB][::-1]
    return X


def loss_fn(out, heat, brk, bw, valid):
    hl = out[:, 0]
    # focal-ish BCE on the heat-map, only over the plot area
    p = torch.sigmoid(hl)
    bce = F.binary_cross_entropy_with_logits(hl, heat, reduction="none")
    wt = 1.0 + 4.0 * heat
    lh = (bce * wt * valid).sum() / valid.sum()
    # bracket: cross-entropy + RPS around marks
    lp = F.log_softmax(out[:, 1:4], 1)
    ce = -(brk * lp).sum(1)
    pr = lp.exp()
    cp = torch.cumsum(pr, 1)[:, :2]; cr = torch.cumsum(brk, 1)[:, :2]
    rps = ((cp - cr) ** 2).sum(1) / 2
    lb = ((ce + rps) * bw).sum() / bw.sum().clamp(min=1.0)
    return lh + 0.5 * lb


def _train_fold(X, Wd, M, marks, tr_idx, epochs=EPOCHS, seed=SEED, vflip_aug=True, hflip_aug=True, verbose=True):
    seed_everything(seed)
    rng = np.random.RandomState(seed)
    net = Net(NB + 2, M.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = epochs * int(np.ceil(len(tr_idx) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            fl = (rng.rand(len(idx)) < 0.5) & hflip_aug
            vf = (rng.rand(len(idx)) < 0.5) & vflip_aug
            xb = augment_batch(X[idx], Wd[idx], fl, vf)
            heat, brk, bw = make_targets([marks[i] for i in idx], Wd[idx], fl)
            valid = np.zeros((len(idx), WMAX), np.float32)
            for j, i in enumerate(idx):
                valid[j, :Wd[i]] = 1
            out = net(torch.from_numpy(xb), torch.from_numpy(M[idx]))
            loss = loss_fn(out, torch.from_numpy(heat), torch.from_numpy(brk), torch.from_numpy(bw),
                           torch.from_numpy(valid))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if verbose:
            log(f"  ep {ep + 1}/{epochs} loss {tot / len(tr_idx):.4f}")
    return net


@torch.no_grad()
def predict_net(net, X, Wd, M, idx, tta=True):
    """Return per-chart (heat prob (W,), bracket probs (3,W)) arrays; each chart is processed independently."""
    net.eval()
    heats, brks = [], []
    for s in range(0, len(idx), 32):
        ii = idx[s:s + 32]
        xb = X[ii]; mb = torch.from_numpy(M[ii])
        o = net(torch.from_numpy(xb), mb)
        h = torch.sigmoid(o[:, 0]).numpy(); b = F.softmax(o[:, 1:4], 1).numpy()
        if tta:
            fl = np.ones(len(ii), bool)
            xf = augment_batch(xb, Wd[ii], fl, np.zeros(len(ii), bool))
            of = net(torch.from_numpy(xf), mb)
            hf = torch.sigmoid(of[:, 0]).numpy(); bf = F.softmax(of[:, 1:4], 1).numpy()
            for j, w in enumerate(Wd[ii]):
                hf[j, :w] = hf[j, :w][::-1]; bf[j, :, :w] = bf[j, :, :w][:, ::-1]
            h = (h + hf) / 2; b = (b + bf) / 2
        for j, w in enumerate(Wd[ii]):
            heats.append(h[j, :w].copy()); brks.append(b[j, :, :w].copy())
    return heats, brks


# ----------------------------- decoding -----------------------------
def decode_chart(h, b, thr, win=5, nms=6):
    """Peak-pick the heat-map. Score of a peak = mass of heat within +-win px (capped), sub-pixel centroid."""
    W = len(h)
    cand = []
    hs = h.copy()
    used = np.zeros(W, bool)
    for _ in range(40):
        c = int(np.argmax(np.where(used, -1, hs)))
        if hs[c] < thr * 0.5 or used[c]:
            break
        lo, hi = max(0, c - 3), min(W, c + 4)
        wts = h[lo:hi]
        x = float((np.arange(lo, hi) * wts).sum() / wts.sum())
        score = float(h[c])
        pb = b[:, max(0, c - 1):c + 2].mean(1)
        cand.append((x, score, pb / pb.sum()))
        used[max(0, c - nms):c + nms + 1] = True
    return [(x, s, p) for x, s, p in cand if s >= thr]


def decode_all(heats, brks, thr):
    out = []
    for h, b in zip(heats, brks):
        out.append([{"x": round(x, 2), "p": [float(v) for v in p]} for x, s, p in decode_chart(h, b, thr)])
    return out


def run_cv(df, X, Wd, M, marks, folds, epochs=EPOCHS, **kw):
    oof_h = [None] * len(df); oof_b = [None] * len(df)
    for f in range(N_FOLDS):
        te_idx = np.where(folds == f)[0]; tr_idx = np.where(folds != f)[0]
        log(f"fold {f}: train {len(tr_idx)} held-out {len(te_idx)}")
        net = _train_fold(X, Wd, M, marks, tr_idx, epochs=epochs, seed=SEED + f, **kw)
        hh, bb = predict_net(net, X, Wd, M, te_idx)
        for k, i in enumerate(te_idx):
            oof_h[i] = hh[k]; oof_b[i] = bb[k]
    return oof_h, oof_b


# ----------------------------- metric (exact re-implementation of the grader, used for OOF threshold choice) -----------------------------
TOL = 5.0


def chart_score(pred, truth):
    """Greedy closest-pair matching within 5 px; credit = 1 - RPS; chart = 2*sum(credit)/(n_pred + n_true)."""
    n_p, n_t = len(pred), len(truth)
    if n_p == 0 and n_t == 0:
        return 1.0
    pairs = sorted((abs(a["x"] - b["x"]), i, j) for i, a in enumerate(pred) for j, b in enumerate(truth)
                   if abs(a["x"] - b["x"]) <= TOL)
    ui, uj, credit = set(), set(), 0.0
    for d, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j)
        P = np.cumsum(pred[i]["p"])[:2]; R = np.cumsum(truth[j]["p"])[:2]
        credit += 1.0 - float(((P - R) ** 2).sum()) / 2.0
    return 2.0 * credit / (n_p + n_t)


MIN_CELL = 30   # cells smaller than this are too noisy to steer the decision threshold


def cv_skill(cells, truth, preds):
    """Mean over (dataset, fieldtype) cells of skill above the empty submission, as in the challenge score."""
    ch = np.array([chart_score(p, t) for p, t in zip(preds, truth)])
    emp = np.array([chart_score([], t) for t in truth])
    out = []
    for c in np.unique(cells):
        m = cells == c
        if m.sum() < MIN_CELL or emp[m].mean() >= 1.0:
            continue
        out.append((ch[m].mean() - emp[m].mean()) / (1 - emp[m].mean()))
    return float(np.mean(out))


# ----------------------------- bracket calibration -----------------------------
def calibrate_p(p, temp, shrink, prior):
    """Temperature on the log-probabilities, then shrink toward the train bracket mix."""
    q = np.asarray(p, dtype=np.float64) ** (1.0 / temp)
    q = q / q.sum()
    q = (1 - shrink) * q + shrink * prior
    return q / q.sum()


def apply_calib(marks, temp, shrink, prior):
    return [{"x": m["x"], "p": [float(v) for v in calibrate_p(m["p"], temp, shrink, prior)]} for m in marks]


def validate_submission(sub, sample):
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub["id"].astype(str).tolist() == sample["id"].astype(str).tolist(), "id mismatch/order"
    assert not sub["id"].duplicated().any(), "duplicate ids"
    for cell in sub["marks"]:
        ms = json.loads(cell)
        assert isinstance(ms, list)
        for m in ms:
            assert set(m.keys()) == {"x", "p"} and np.isfinite(m["x"]) and len(m["p"]) == 3
            assert min(m["p"]) >= 0 and abs(sum(m["p"]) - 1.0) < 1e-6, m


def main():
    seed_everything()
    tr = pd.read_csv(PUBLIC_DIR / "train.csv").merge(pd.read_csv(PUBLIC_DIR / "folds.csv"), on="id")
    te = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    X, Wd, M = load_split(tr, PUBLIC_DIR / "train_images")
    Xt, Wt, Mt = load_split(te, PUBLIC_DIR / "test_images")
    log(f"features built: train {X.shape} test {Xt.shape}")
    marks = [parse_marks(s) for s in tr["marks"]]
    truth = [json.loads(s) for s in tr["marks"]]
    cells = (tr["dataset_name"] + "|" + tr["fieldtype"]).values
    folds = tr["fold"].values
    prior = np.zeros(3)
    for ms in marks:
        for _, b in ms:
            prior[b] += 1
    prior /= prior.sum()

    # 1) cross-cell CV: every fold model never sees the held-out dataset x fieldtype cells; test = mean of fold models
    oof_h = [None] * len(tr); oof_b = [None] * len(tr)
    test_h = [None] * len(te); test_b = [None] * len(te)
    for f in range(N_FOLDS):
        te_idx = np.where(folds == f)[0]; tr_idx = np.where(folds != f)[0]
        log(f"fold {f}: train {len(tr_idx)} held-out {len(te_idx)}")
        net = _train_fold(X, Wd, M, marks, tr_idx, epochs=EPOCHS, seed=SEED + f)
        hh, bb = predict_net(net, X, Wd, M, te_idx)
        for k, i in enumerate(te_idx):
            oof_h[i] = hh[k]; oof_b[i] = bb[k]
        th, tb = predict_net(net, Xt, Wt, Mt, np.arange(len(te)))   # each test chart is scored on its own
        for i in range(len(te)):
            if test_h[i] is None:
                test_h[i] = th[i] / N_FOLDS; test_b[i] = tb[i] / N_FOLDS
            else:
                test_h[i] = test_h[i] + th[i] / N_FOLDS; test_b[i] = test_b[i] + tb[i] / N_FOLDS

    # 2) decision threshold and bracket calibration, chosen on out-of-fold predictions only (two numbers + one temperature)
    grid = [round(t, 2) for t in np.arange(0.30, 0.86, 0.05)]
    skills = []
    for thr in grid:
        skills.append(cv_skill(cells, truth, decode_all(oof_h, oof_b, thr)))
    sm = np.convolve(np.pad(skills, 1, mode="edge"), np.ones(3) / 3, mode="valid")   # prefer a plateau over a spike
    thr = grid[int(np.argmax(sm))]
    log("threshold grid " + " ".join(f"{g}:{s:.3f}" for g, s in zip(grid, skills)) + f" -> {thr}")
    base = decode_all(oof_h, oof_b, thr)
    best = (cv_skill(cells, truth, base), 1.0, 0.0)
    for temp in (0.8, 1.0, 1.25, 1.5):
        for shrink in (0.0, 0.1, 0.2):
            sc = cv_skill(cells, truth, [apply_calib(m, temp, shrink, prior) for m in base])
            if sc > best[0] + 1e-4:
                best = (sc, temp, shrink)
    _, temp, shrink = best
    log(f"OOF skill (mean over cells >= {MIN_CELL} charts) at thr {thr}: {best[0]:.4f}  bracket temp {temp} shrink {shrink}")

    # 3) test predictions
    pred = [apply_calib(m, temp, shrink, prior) for m in decode_all(test_h, test_b, thr)]
    sub = sample.copy()
    sub["marks"] = [json.dumps(p) for p in pred]
    sub["id"] = te["id"].values
    validate_submission(sub, sample)
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUBMISSION_OUT.with_suffix(".tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, SUBMISSION_OUT)
    n_marks = sum(len(p) for p in pred)
    log(f"wrote {SUBMISSION_OUT} rows={len(sub)} marks={n_marks} empty charts={sum(len(p) == 0 for p in pred)}")


if __name__ == "__main__":
    main()
