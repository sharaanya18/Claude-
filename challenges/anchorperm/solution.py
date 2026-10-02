"""Solver for AnchorPerm: few-shot scientific correspondence completion (row-local bipartite matching + confidence).

Requirements map (challenge rules -> where satisfied)
  * Public-data model training only: a pair-scoring neural network is trained in this script on train.csv (random anchor
    subsets re-drawn every epoch = training-only robustness augmentation; per-row code jitter / re-quantisation augmentation).
  * Row-local anchor conditioning: the revealed anchors of a row enter the scorer as structure features (how similar a
    question is to the anchored questions vs. how similar an answer is to the anchored answers).
  * Coherent one-to-one decoding: Hungarian assignment on the symmetric log-probability score of the hidden block.
  * Confidence: a gradient-boosted regressor maps row diagnostics (assignment margins, probabilities, ensemble
    disagreement, set size, anchor count, the row's own code spread) to expected row accuracy; it is trained on
    out-of-fold predictions of train rows (clean and augmented copies) only.
  * No test-set fitting: every transformer/normaliser is row-local; test rows are only predicted one row at a time.
    No example_id / row-order channel, no lookup tables, no external data or weights.
  * Fixed work plan (folds, seeds, epochs are constants); elapsed time is only logged; CPU is a fixed constant device.
"""
import os
import sys
import random
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 5
N_SEEDS = 5                 # models per fold / final ensemble members
EPOCHS = 40
BATCH = 64
LR = 2e-3
NOISE_MAX = 0.6             # training augmentation: per-row jitter
GAIN_MAX = 1.6              # training augmentation: per-row re-quantisation gain
ROT_PROB = 0.0              # fraction of training rows whose Q / A coordinate frames are randomly rotated (mapping-shift robustness)
STRESS_COPIES = 3           # augmented copies of every OOF row used to fit the confidence calibrator
T0 = time.time()            # LOGGING ONLY


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


N_MAX = 12
K_MAX = 5
MAP = {"a": -2, "b": -1, "c": 0, "d": 1, "e": 2}


def parse_items(s):
    keys, vecs = [], []
    for t in s.split():
        k, v = t.split("=")
        keys.append(k)
        vecs.append([MAP[c] for c in v])
    return keys, np.array(vecs, np.float32)


def load_rows(df):
    rows = []
    for r in df.itertuples():
        qk, Q = parse_items(r.q_items)
        ak, A = parse_items(r.a_items)
        an = dict(t.split("=") for t in r.anchors.split())
        d = dict(id=r.example_id, n=len(qk), k=r.anchor_count, qk=qk, ak=ak, Q=Q, A=A,
                 anc=[(qk.index(a), ak.index(b)) for a, b in an.items()])
        d["hq"] = [i for i in range(len(qk)) if qk[i] not in an]
        anchored_a = {b for b in an.values()}
        d["ha"] = [j for j in range(len(ak)) if ak[j] not in anchored_a]
        if hasattr(r, "target_sequence"):
            d["y"] = [ak.index(t) for t in r.target_sequence.split()]
            pi = dict(d["anc"])
            pi.update(zip(d["hq"], d["y"]))
            d["pi"] = np.array([pi[i] for i in range(len(qk))])        # full true matching question -> answer
        rows.append(d)
    return rows


def pad_rows(rows):
    B = len(rows)
    Q = np.zeros((B, N_MAX, 32), np.float32); A = np.zeros_like(Q)
    nm = np.zeros((B, N_MAX), bool); pi = np.zeros((B, N_MAX), np.int64)
    for b, r in enumerate(rows):
        n = r["n"]; Q[b, :n] = r["Q"]; A[b, :n] = r["A"]; nm[b, :n] = True
        pi[b] = torch.arange(N_MAX).numpy()          # padded slots map to themselves so pi stays a permutation
        if "pi" in r: pi[b, :n] = r["pi"]
    return torch.tensor(Q), torch.tensor(A), torch.tensor(nm), torch.tensor(pi)


def normalise(X, nm):
    """Row-local, role-local normalisation: centre every coordinate over the row's items, divide by the row's overall spread."""
    m = nm[..., None].float()
    cnt = m.sum(1, keepdim=True)
    mu = (X * m).sum(1, keepdim=True) / cnt
    Xc = (X - mu) * m
    sd = ((Xc ** 2).sum((1, 2), keepdim=True) / (cnt * X.shape[-1])).sqrt().clamp_min(1e-6)
    return Xc / sd


def cos_matrix(X):
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return Xn @ Xn.transpose(1, 2)


def zscore_sim(S, nm):
    N = S.shape[1]
    valid = nm[:, :, None] & nm[:, None, :] & ~torch.eye(N, dtype=torch.bool)[None]
    v = valid.float(); cnt = v.sum((1, 2), keepdim=True).clamp_min(1)
    mu = (S * v).sum((1, 2), keepdim=True) / cnt
    sd = (((S - mu) ** 2 * v).sum((1, 2), keepdim=True) / cnt).sqrt().clamp_min(1e-6)
    return (S - mu) / sd


class PairScorer(nn.Module):
    def __init__(self, d=96, hid=64, use_tower=True, ctx_layers=0, drop=0.0, rich=0):
        super().__init__()
        self.rich = rich
        if rich: self.rh = nn.Sequential(nn.Linear(2 * d, rich), nn.GELU(), nn.Linear(rich, 1))
        self.use_tower = use_tower
        self.ctx_layers = ctx_layers
        if ctx_layers:
            mk = lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True, norm_first=True), ctx_layers)
            self.cq, self.ca = mk(), mk()
        self.fq = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, d))
        self.fa = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, d))
        self.d = d
        self.head = nn.Sequential(nn.Linear(1 + 4 + 2 + 1, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, Q, A, nm, am, pi):
        """Q,A raw codes (B,N,32); nm valid mask (B,N); am anchored-question mask (B,N); pi true matching (B,N) (used only
        to read the anchored pairs, which are inputs). Returns logits (B,N,N) for [question i, answer j]."""
        B, N, _ = Q.shape
        Qn, An = normalise(Q, nm), normalise(A, nm)
        eq, ea = self.fq(Qn), self.fa(An)
        if self.ctx_layers:
            eq = self.cq(eq, src_key_padding_mask=~nm); ea = self.ca(ea, src_key_padding_mask=~nm)
        s1 = torch.einsum("bid,bjd->bij", eq, ea) / self.d ** 0.5
        if self.rich:
            pf = torch.cat([eq[:, :, None, :] * ea[:, None, :, :], (eq[:, :, None, :] - ea[:, None, :, :]).abs()], -1)
            s1 = s1 + self.rh(pf)[..., 0]
        if not self.use_tower: s1 = s1 * 0
        Zq, Za = zscore_sim(cos_matrix(Qn), nm), zscore_sim(cos_matrix(An), nm)
        # anchors: up to K_MAX anchored questions per row (am) and their answers pi[am]
        idx = torch.arange(N)[None].expand(B, N)
        key = torch.where(am, idx, torch.full_like(idx, N + 1))
        aq = key.sort(1).values[:, :K_MAX]                          # anchored question indices, padded with N+1
        km = aq <= N - 1                                             # (B,K)
        aq_c = aq.clamp(max=N - 1)
        aa = pi.gather(1, aq_c)                                      # their answers
        zq = Zq.gather(2, aq_c[:, None, :].expand(B, N, K_MAX))      # (B,N,K): similarity of every question to anchored q_k
        za = Za.gather(2, aa[:, None, :].expand(B, N, K_MAX))        # (B,N,K): similarity of every answer to anchored a_k
        kf = km.float()[:, None, :]                                  # (B,1,K)
        kc = kf.sum(-1, keepdim=True).clamp_min(1)                   # (B,1,1)
        prod = torch.einsum("bik,bjk->bij", zq * kf, za) / kc
        diff = (zq[:, :, None, :] - za[:, None, :, :])               # (B,N,N,K)
        absd = (diff.abs() * kf[:, :, None, :]).sum(-1) / kc
        sq = (diff ** 2 * kf[:, :, None, :]).sum(-1) / kc
        mx = (diff.abs() * kf[:, :, None, :] + (1 - kf[:, :, None, :]) * -1).amax(-1)
        # reliability of the cross-role tower on THIS row: how well it ranks the row's own anchors (anchored pair z-score in its row)
        s1m = s1.masked_fill(~nm[:, None, :], 0.0)
        cntA = nm.float().sum(1)[:, None, None].clamp_min(1)
        mu = s1m.sum(2, keepdim=True) / cntA
        sd = ((((s1 - mu) ** 2) * nm[:, None, :].float()).sum(2, keepdim=True).div(cntA) + 1e-6).sqrt()
        zrow = ((s1 - mu) / sd)                                              # (B,N,N) z-score of each answer within its question's row
        ztrue = zrow.gather(2, pi[:, :, None])[..., 0]                         # (B,N) z-score of the true answer for each question
        zanch = (ztrue * am.float()).sum(1) / am.float().sum(1).clamp_min(1)   # mean over anchored questions
        rel = zanch[:, None, None].expand(B, N, N)
        ctx = torch.stack([nm.float().sum(1) / 12.0, am.float().sum(1) / 5.0], -1)[:, None, None, :].expand(B, N, N, 2)
        f = torch.cat([s1[..., None], prod[..., None], absd[..., None], sq[..., None], mx[..., None], ctx, rel[..., None]], -1)
        return self.head(f)[..., 0]


def listwise_loss(logits, nm, am, pi):
    """CE of the true answer among the hidden answers for each hidden question, plus the column direction."""
    B, N, _ = logits.shape
    inv = torch.zeros_like(pi); inv.scatter_(1, pi, torch.arange(N)[None].expand(B, N))
    aa_mask = torch.zeros(B, N).scatter_add_(1, pi, am.float()) > 0   # answers consumed by anchors
    hq = nm & ~am
    ha = nm & ~aa_mask
    big = -1e9
    lr = logits.masked_fill(~ha[:, None, :], big)
    lc = logits.masked_fill(~hq[:, :, None], big)
    ce_r = F.cross_entropy(lr.reshape(B * N, N), pi.reshape(-1), reduction="none").view(B, N)
    ce_c = F.cross_entropy(lc.transpose(1, 2).reshape(B * N, N), inv.reshape(-1), reduction="none").view(B, N)
    # column j is a hidden answer; its true question is inv[j]
    loss_r = (ce_r * hq).sum() / hq.sum()
    loss_c = (ce_c * ha).sum() / ha.sum()
    return 0.5 * (loss_r + loss_c)


def sample_anchors(rows_n, gen):
    """Random anchor subset size k in 2..min(5,n-5) per row."""
    B = len(rows_n)
    am = torch.zeros(B, N_MAX, dtype=torch.bool)
    for b, n in enumerate(rows_n):
        k = int(torch.randint(2, min(5, n - 5) + 1, (1,), generator=gen))
        am[b, torch.randperm(n, generator=gen)[:k]] = True
    return am


def augment(Q, A, nm, gen, noise_max=0.6, gain_max=1.0, noise_min=0.0, requant=False, rot_prob=0.0):
    """Training-only robustness augmentation: per-row Gaussian jitter and optional re-quantisation with a random gain
    (simulates rows whose codes are noisier / spread wider, as in unseen representation regimes)."""
    B = Q.shape[0]
    s = noise_min + torch.rand(B, 1, 1, generator=gen) * (noise_max - noise_min)
    Q = Q + torch.randn(Q.shape, generator=gen) * s * nm[..., None]
    A = A + torch.randn(A.shape, generator=gen) * s * nm[..., None]
    if requant and gain_max == 1.0:
        Q = torch.clamp(torch.round(Q), -2, 2) * nm[..., None]; A = torch.clamp(torch.round(A), -2, 2) * nm[..., None]
    if gain_max > 1.0:
        g = 1.0 + torch.rand(B, 1, 1, generator=gen) * (gain_max - 1.0)
        Q = torch.clamp(torch.round(Q * g), -2, 2) * nm[..., None]
        A = torch.clamp(torch.round(A * g), -2, 2) * nm[..., None]
    if rot_prob > 0:
        for X, tag in ((Q, 0), (A, 1)):
            R = torch.linalg.qr(torch.randn(B, 32, 32, generator=gen))[0]
            use = (torch.rand(B, 1, 1, generator=gen) < rot_prob).float()
            Rm = use * R + (1 - use) * torch.eye(32)[None]
            if tag == 0: Q = torch.bmm(Q, Rm)
            else: A = torch.bmm(A, Rm)
    return Q, A


def anchor_mask_from_rows(rows):
    am = torch.zeros(len(rows), N_MAX, dtype=torch.bool)
    for b, r in enumerate(rows):
        for q, a in r["anc"]: am[b, q] = True
    return am


def anchored_pi(rows, pi_true=None):
    """pi tensor where only the anchored questions carry their true answer (others arbitrary): used for test rows."""
    pi = torch.zeros(len(rows), N_MAX, dtype=torch.long)
    for b, r in enumerate(rows):
        for q, a in r["anc"]: pi[b, q] = a
    return pi


@torch.no_grad()
def predict_logits(model, rows, bs=256):
    model.eval(); outs = []
    for s in range(0, len(rows), bs):
        ch = rows[s:s + bs]
        Q, A, nm, _ = pad_rows(ch); am = anchor_mask_from_rows(ch); pi = anchored_pi(ch)
        outs.append(model(Q, A, nm, am, pi))
    return torch.cat(outs)


def decode_row(logit, r):
    """Hungarian on the symmetric log-prob score over the hidden block. Returns (answer indices in hidden-question order, score matrix)."""
    hq, ha = r["hq"], r["ha"]
    L = logit[np.ix_(hq, ha)]
    lr = L - np.logaddexp.reduce(L, axis=1, keepdims=True)
    lc = L - np.logaddexp.reduce(L, axis=0, keepdims=True)
    S = 0.5 * (lr + lc)
    ri, ci = linear_sum_assignment(-S)
    return [ha[c] for c in ci], S




def train_model(rows, seed, epochs=EPOCHS, bs=BATCH, lr=LR):
    """Train one pair scorer. Every epoch re-draws a random anchor subset (2..min(5, n-5)) per row and jitters / re-quantises
    the codes, so the network cannot memorise a fixed anchor pattern or one code scale."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = PairScorer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * ((len(rows) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Q0, A0, nm0, pi0 = pad_rows(rows)
    ns = [r["n"] for r in rows]
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(rows), generator=gen)
        for s in range(0, len(rows), bs):
            ix = order[s:s + bs]
            Q, A = augment(Q0[ix], A0[ix], nm0[ix], gen, NOISE_MAX, GAIN_MAX, 0.0, False, ROT_PROB)
            am = sample_anchors([ns[i] for i in ix.tolist()], gen)
            loss = listwise_loss(model(Q, A, nm0[ix], am, pi0[ix]), nm0[ix], am, pi0[ix])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model


def stress_rows(rows, rng, kind):
    """Augmented copies of held-out TRAIN rows. The confidence calibrator is fitted on these so that prediction diagnostics
    (margins, anchor agreement, ensemble disagreement, code spread) are linked to realised accuracy under other regimes:
    kind 0 = random gain + noise re-quantised, kind 1 = additive code noise, kind 2 = partially rotated coordinate frames."""
    from scipy.linalg import expm
    out = []
    for r in rows:
        r2 = dict(r)
        if kind == 2:
            theta = rng.uniform(0.4, 1.2)
            for key in ("Q", "A"):
                G = rng.normal(size=(32, 32)) / 32 ** 0.5
                R = expm((G - G.T) / 2 ** 0.5 * theta)
                r2[key] = np.clip(np.round(r[key] @ R), -2, 2).astype(np.float32)
        else:
            g, sd = (rng.uniform(1.0, 1.8), rng.uniform(0.0, 0.5)) if kind == 0 else (1.0, rng.uniform(0.4, 0.9))
            for key in ("Q", "A"):
                x = r[key]
                r2[key] = np.clip(np.round(x * g + rng.normal(0, sd, x.shape)), -2, 2).astype(np.float32)
        out.append(r2)
    return out


def best_total(S):
    ri, ci = linear_sum_assignment(-S)
    return S[ri, ci].sum(), ci


def row_features(member_logits, r):
    """Diagnostics of one row's ensemble prediction. Returns (answer indices, feature vector)."""
    hq, ha = r["hq"], r["ha"]
    mean_logit = np.mean(member_logits, axis=0)
    pred, S = decode_row(mean_logit, r)
    m = len(hq)
    L = mean_logit[np.ix_(hq, ha)]
    pr = np.exp(L - np.logaddexp.reduce(L, axis=1, keepdims=True))
    pc = np.exp(L - np.logaddexp.reduce(L, axis=0, keepdims=True))
    ci = [ha.index(a) for a in pred]
    p_row, p_col = pr[np.arange(m), ci], pc[np.arange(m), ci]
    tot, _ = best_total(S)
    drops = []
    for i in range(m):                      # how much worse is the best assignment that forbids this pair?
        S2 = S.copy(); S2[i, ci[i]] = -1e6
        drops.append(tot - best_total(S2)[0])
    ent = float(-(pr * np.log(pr + 1e-9)).sum(1).mean())
    agree = []
    for ml in member_logits:
        pm, _ = decode_row(ml, r)
        agree.append(np.mean([a == b for a, b in zip(pm, pred)]))
    feats = [r["n"], r["k"], m, p_row.mean(), p_row.min(), p_col.mean(), p_col.min(), S[np.arange(m), ci].mean(),
             float(np.min(drops)), float(np.mean(drops)), float(np.median(drops)), ent, float(np.mean(agree)), float(np.min(agree)),
             r["Q"].std(), r["A"].std(), float((np.abs(r["Q"]) == 2).mean()), float((np.abs(r["A"]) == 2).mean())]
    return pred, np.array(feats, np.float64)


def ensemble_logits(models, rows):
    return [predict_logits(m, rows).numpy() for m in models]


def predict_rows(models, rows):
    per_model = ensemble_logits(models, rows)
    preds, feats = [], []
    for i, r in enumerate(rows):
        p, f = row_features([pm[i] for pm in per_model], r)
        preds.append(p); feats.append(f)
    return preds, np.array(feats)


def row_loss(pred, truth, conf):
    m = len(truth)
    if len(pred) != m or len(set(pred)) != m or not (0.0 <= conf <= 1.0):
        return 1.0
    l_seq = float(np.mean([p != t for p, t in zip(pred, truth)]))
    return 0.85 * l_seq + 0.15 * (conf - (1.0 - l_seq)) ** 2


def new_calibrator():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=SEED)


def validate_submission(sub, sample, test_rows):
    assert list(sub.columns) == list(sample.columns), "columns"
    assert len(sub) == len(sample) and sub.example_id.tolist() == sample.example_id.tolist(), "ids"
    assert sub.example_id.is_unique
    byid = {r["id"]: r for r in test_rows}
    for ex, seq, c in zip(sub.example_id, sub.target_sequence, sub.confidence):
        r = byid[ex]; toks = seq.split()
        avail = {r["ak"][j] for j in r["ha"]}
        assert len(toks) == r["n"] - r["k"] and len(set(toks)) == len(toks) and set(toks) == avail, f"bad sequence for {ex}"
        assert np.isfinite(c) and 0.0 <= c <= 1.0, f"bad confidence for {ex}"


def main():
    seed_everything()
    train = load_rows(pd.read_csv(PUBLIC_DIR / "train.csv", keep_default_na=False))
    test = load_rows(pd.read_csv(PUBLIC_DIR / "test.csv", keep_default_na=False))
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    log(f"train rows {len(train)}, test rows {len(test)}")
    rng = np.random.default_rng(SEED)
    # ---- out-of-fold predictions on clean and stressed copies of held-out train rows (for the confidence calibrator)
    feats, targets, meta = [], [], []
    for fold, (fi, vi) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train)):
        fit = [train[i] for i in fi]
        models = [train_model(fit, SEED + 100 * fold + s) for s in range(N_SEEDS)]
        val = [train[i] for i in vi]
        variants = [(0, val)] + [(1 + c, stress_rows(val, rng, c)) for c in range(STRESS_COPIES)]
        for vid, rows in variants:
            preds, f = predict_rows(models, rows)
            for j, (p, r) in enumerate(zip(preds, rows)):
                acc = float(np.mean([a == b for a, b in zip(p, train[vi[j]]["y"])]))
                feats.append(f[j]); targets.append(acc); meta.append((int(vi[j]), vid))
            log(f"fold {fold} variant {vid}: accuracy {np.mean(targets[-len(rows):]):.4f}")
    feats, targets = np.array(feats), np.array(targets)
    meta = np.array(meta)
    # ---- calibrator evaluated out-of-fold over rows, then the exact metric on the calibrated confidences
    conf = np.zeros(len(targets))
    for tri, tei in KFold(5, shuffle=True, random_state=SEED + 1).split(np.arange(len(train))):
        trm = np.isin(meta[:, 0], tri)
        cal = new_calibrator().fit(feats[trm], targets[trm])
        conf[~trm] = np.clip(cal.predict(feats[~trm]), 0.0, 1.0)
    loss = np.array([1.0] * len(targets))
    for i in range(len(targets)):
        l_seq = 1.0 - targets[i]
        loss[i] = 0.85 * l_seq + 0.15 * (conf[i] - targets[i]) ** 2
    sparse = feats[:, 1] <= 3
    for name, sel in [("clean", meta[:, 1] == 0), ("gain+noise", meta[:, 1] == 1), ("noise", meta[:, 1] == 2), ("part-rotated", meta[:, 1] == 3)]:
        strata = [loss[sel & sparse].mean(), loss[sel & ~sparse].mean()]
        log(f"OOF {name}: accuracy {targets[sel].mean():.4f} mean row loss {loss[sel].mean():.4f} "
            f"sparse {strata[0]:.4f} rich {strata[1]:.4f} | 0.75*mean+0.25*worst = {0.75 * loss[sel].mean() + 0.25 * max(strata):.4f}")
    log(f"calibration check: mean conf {conf.mean():.4f} vs mean accuracy {targets.mean():.4f}")
    calibrator = new_calibrator().fit(feats, targets)
    # ---- final ensemble on all training rows, then predict the test rows one row at a time
    models = [train_model(train, SEED + 1000 + s) for s in range(N_SEEDS)]
    preds, f = predict_rows(models, test)
    confidence = np.clip(calibrator.predict(f), 0.0, 1.0)
    seqs = [" ".join(r["ak"][j] for j in p) for r, p in zip(test, preds)]
    by_id = {r["id"]: (s, float(c)) for r, s, c in zip(test, seqs, confidence)}
    sub = pd.DataFrame({"example_id": sample.example_id,
                        "target_sequence": [by_id[i][0] for i in sample.example_id],
                        "confidence": [round(by_id[i][1], 6) for i in sample.example_id]})
    validate_submission(sub, sample, test)
    tmp = SUBMISSION_OUT.with_suffix(".tmp")
    sub.to_csv(tmp, index=False)
    validate_submission(pd.read_csv(tmp, keep_default_na=False), sample, test)
    os.replace(tmp, SUBMISSION_OUT)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}; confidence mean {confidence.mean():.3f}")


if __name__ == "__main__":
    main()
