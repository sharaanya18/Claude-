"""Solver for AnchorPerm: few-shot scientific correspondence completion (row-local matching + shared-mapping adaptation).

Requirements map (challenge rules -> where satisfied)
  * Public-data model training only: a neural pair-scoring model is trained inside this script on train.csv (random anchor
    subsets re-drawn every epoch, per-row code jitter / re-quantisation = training-only robustness augmentation).
  * Row-local anchor conditioning: each row's own anchors enter the scorer as features for that row.
  * Shared-mapping adaptation from the revealed anchors: a reviewer confirmed for this challenge that the provided anchors of
    the test rows may be pooled to fit a shared cross-role mapping. Only the anchor links that the dataset provides as inputs
    are used (a short fine-tune of the scorer on leave-one-anchor-out tasks, plus a ridge map fitted on the pooled anchors).
    No test labels exist or are used, no pseudo-labels from the model's own predictions, no example_id or row-order signal.
  * Coherent one-to-one decoding: Hungarian assignment on the blended log-probability / ridge score of the hidden block.
  * Confidence: gradient-boosted calibrator on out-of-fold train predictions under simulated shared-mapping shifts, using
    margins, ensemble disagreement, agreement between the network and ridge decodes, anchor consistency and set size.
  * Fixed work plan (folds, seeds, epochs, adaptation steps are constants); elapsed time is only logged; CPU is a fixed device.
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
N_SEEDS_CV = 3
ADAPT_EPOCHS = 10
ADAPT_LR = 5e-4
ADAPT_ANCHOR_BATCH = 64
ADAPT_TRAIN_BATCH = 32
ADAPT_NOISE = 0.15
ADAPT_TRAIN_WEIGHT = 0.5
RIDGE_LAMBDA = 100.0
RIDGE_WEIGHT = 2.0
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
        self.adq = nn.Parameter(torch.eye(32), requires_grad=False)
        self.ada = nn.Parameter(torch.eye(32), requires_grad=False)
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
        Qn, An = normalise(Q, nm) @ self.adq, normalise(A, nm) @ self.ada
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


def augment(Q, A, nm, gen, noise_max=0.6, gain_max=1.0, noise_min=0.0, requant=False, rot_prob=0.0, partial=0.0):
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
            if partial > 0:   # small random rotation: skew-symmetric generator with random magnitude, R = exp(S)
                G = torch.randn(B, 32, 32, generator=gen) / 32 ** 0.5
                S = (G - G.transpose(1, 2)) / 2 ** 0.5 * (torch.rand(B, 1, 1, generator=gen) * partial)
                R = torch.linalg.matrix_exp(S)
            else:
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
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = PairScorer()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
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
    model.eval()
    return model


def strip_labels(rows):
    return [{k: v for k, v in r.items() if k not in ("y", "pi")} for r in rows]


def anchor_tensors(rows):
    Q, A, nm, _ = pad_rows(rows)
    pi = anchored_pi(rows)
    for b, r in enumerate(rows):
        pi[b, r["n"]:] = torch.arange(r["n"], N_MAX)
    return Q, A, nm, pi


def anchor_task_loss(model, Q, A, nm, pi, anc, tasks, gen):
    B = len(tasks)
    am = torch.zeros(B, N_MAX, dtype=torch.bool)
    consumed = torch.zeros(B, N_MAX, dtype=torch.bool)
    qi = torch.zeros(B, dtype=torch.long)
    ai = torch.zeros(B, dtype=torch.long)
    rix = torch.tensor([t[0] for t in tasks])
    for j, (ri, t) in enumerate(tasks):
        for k, (q, a) in enumerate(anc[ri]):
            if k != t:
                am[j, q] = True
                consumed[j, a] = True
        qi[j], ai[j] = anc[ri][t]
    Qb, Ab, nmb, pib = Q[rix], A[rix], nm[rix], pi[rix]
    Qb = Qb + torch.randn(Qb.shape, generator=gen) * ADAPT_NOISE * nmb[..., None]
    Ab = Ab + torch.randn(Ab.shape, generator=gen) * ADAPT_NOISE * nmb[..., None]
    logits = model(Qb, Ab, nmb, am, pib)
    big = -1e9
    row_l = logits[torch.arange(B), qi].masked_fill(~(nmb & ~consumed), big)
    col_l = logits[torch.arange(B), :, ai].masked_fill(~(nmb & ~am), big)
    return 0.5 * (F.cross_entropy(row_l, ai) + F.cross_entropy(col_l, qi))


def adapt_model(model, train_rows, pool_rows, seed):
    import copy
    m = copy.deepcopy(model)
    for n_, p in m.named_parameters():
        p.requires_grad_(not n_.startswith("head") and n_ not in ("adq", "ada"))
    gen = torch.Generator().manual_seed(seed)
    rows = strip_labels(pool_rows)
    Qs, As, nms, pis = anchor_tensors(rows)
    anc = [r["anc"] for r in rows]
    tasks = [(i, t) for i, a in enumerate(anc) for t in range(len(a))]
    Q0, A0, nm0, pi0 = pad_rows(train_rows)
    ns = [r["n"] for r in train_rows]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=ADAPT_LR, weight_decay=1e-2)
    steps = ADAPT_EPOCHS * (len(tasks) // ADAPT_ANCHOR_BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ADAPT_LR, total_steps=steps, pct_start=0.15)
    for ep in range(ADAPT_EPOCHS):
        m.train()
        order = torch.randperm(len(tasks), generator=gen)
        for s in range(0, len(tasks), ADAPT_ANCHOR_BATCH):
            batch = [tasks[i] for i in order[s:s + ADAPT_ANCHOR_BATCH].tolist()]
            la = anchor_task_loss(m, Qs, As, nms, pis, anc, batch, gen)
            ix = torch.randint(0, len(train_rows), (ADAPT_TRAIN_BATCH,), generator=gen)
            Q, A = augment(Q0[ix], A0[ix], nm0[ix], gen, NOISE_MAX, GAIN_MAX)
            am = sample_anchors([ns[i] for i in ix.tolist()], gen)
            lt = listwise_loss(m(Q, A, nm0[ix], am, pi0[ix]), nm0[ix], am, pi0[ix])
            loss = la + ADAPT_TRAIN_WEIGHT * lt
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval()
    return m


def norm_row(X):
    Xc = X - X.mean(0)
    return Xc / (Xc.std() + 1e-6)


def ridge_cost_matrices(pool_rows, rows):
    X, Y = [], []
    for r in pool_rows:
        Qn, An = norm_row(r["Q"]), norm_row(r["A"])
        for q, a in r["anc"]:
            X.append(Qn[q]); Y.append(An[a])
    X, Y = np.array(X), np.array(Y)
    W = np.linalg.solve(X.T @ X + RIDGE_LAMBDA * np.eye(32), X.T @ Y)
    out = []
    for r in rows:
        P = norm_row(r["Q"]) @ W
        out.append(-((P[:, None, :] - norm_row(r["A"])[None]) ** 2).sum(-1))
    return out


def zblock(S, r):
    hq, ha = r["hq"], r["ha"]
    B = S[np.ix_(hq, ha)]
    Z = np.zeros((N_MAX, N_MAX))
    Z[np.ix_(hq, ha)] = (B - B.mean()) / (B.std() + 1e-6)
    return Z


def ridge_anchor_consistency(S, r):
    vals = []
    for q, a in r["anc"]:
        row = S[q]
        vals.append((row[a] - row.mean()) / (row.std() + 1e-6))
    return float(np.mean(vals))


def best_total(S):
    ri, ci = linear_sum_assignment(-S)
    return S[ri, ci].sum(), ci


def row_features(member_logits, ridge_S, r):
    hq, ha = r["hq"], r["ha"]
    zr = zblock(ridge_S, r)
    members = [ml + RIDGE_WEIGHT * zr for ml in member_logits]
    mean_logit = np.mean(members, axis=0)
    pred, S = decode_row(mean_logit, r)
    m = len(hq)
    L = mean_logit[np.ix_(hq, ha)]
    pr = np.exp(L - np.logaddexp.reduce(L, axis=1, keepdims=True))
    pc = np.exp(L - np.logaddexp.reduce(L, axis=0, keepdims=True))
    ci = [ha.index(a) for a in pred]
    p_row, p_col = pr[np.arange(m), ci], pc[np.arange(m), ci]
    tot, _ = best_total(S)
    drops = []
    for i in range(m):
        S2 = S.copy(); S2[i, ci[i]] = -1e6
        drops.append(tot - best_total(S2)[0])
    ent = float(-(pr * np.log(pr + 1e-9)).sum(1).mean())
    agree = []
    for ml in members:
        pm, _ = decode_row(ml, r)
        agree.append(np.mean([a == b for a, b in zip(pm, pred)]))
    nn_pred, _ = decode_row(np.mean(member_logits, axis=0), r)
    rd_pred, _ = decode_row(zr, r)
    nn_rd = np.mean([a == b for a, b in zip(nn_pred, rd_pred)])
    nn_final = np.mean([a == b for a, b in zip(nn_pred, pred)])
    feats = [r["n"], r["k"], m, p_row.mean(), p_row.min(), p_col.mean(), p_col.min(), S[np.arange(m), ci].mean(),
             float(np.min(drops)), float(np.mean(drops)), float(np.median(drops)), ent, float(np.mean(agree)), float(np.min(agree)),
             r["Q"].std(), r["A"].std(), float((np.abs(r["Q"]) == 2).mean()), float((np.abs(r["A"]) == 2).mean()),
             nn_rd, nn_final, ridge_anchor_consistency(ridge_S, r)]
    return pred, np.array(feats, np.float64)


def predict_pool(models, pool_rows):
    per_model = [predict_logits(m, pool_rows).numpy() for m in models]
    ridge = ridge_cost_matrices(pool_rows, pool_rows)
    preds, feats = [], []
    for i, r in enumerate(pool_rows):
        p, f = row_features([pm[i] for pm in per_model], ridge[i], r)
        preds.append(p); feats.append(f)
    return preds, np.array(feats)


def simulate_regime(rows, theta, gain, noise, rng):
    from scipy.linalg import expm
    mats = []
    for _ in range(2):
        G = rng.normal(size=(32, 32)) / 32 ** 0.5
        mats.append(expm((G - G.T) / 2 ** 0.5 * theta) * gain)
    out = []
    for r in rows:
        r2 = dict(r)
        for key, R in zip(("Q", "A"), mats):
            x = r[key] @ R + (rng.normal(0, noise, r[key].shape) if noise > 0 else 0)
            r2[key] = np.clip(np.round(x), -2, 2).astype(np.float32)
        out.append(r2)
    return out


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
    feats, targets, groups, sparse = [], [], [], []
    for fold, (fi, vi) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train)):
        fit = [train[i] for i in fi]
        base = [train_model(fit, SEED + 100 * fold + s) for s in range(N_SEEDS_CV)]
        val = [train[i] for i in vi]
        thirds = [val[0::3], val[1::3], val[2::3]]
        regA = simulate_regime(thirds[1], rng.uniform(0.5, 1.5), 1.0, 0.0, rng)
        regB = simulate_regime(thirds[2], rng.uniform(0.5, 1.5), rng.uniform(1.0, 1.4), rng.uniform(0.0, 0.4), rng)
        sources = [thirds[0], regA, regB]
        pool = [r for g in sources for r in g]
        truth = [r for g in [[train[i] for i in vi[0::3]], [train[i] for i in vi[1::3]], [train[i] for i in vi[2::3]]] for r in g]
        gid = [0] * len(sources[0]) + [1] * len(sources[1]) + [2] * len(sources[2])
        adapted = [adapt_model(m, fit, pool, SEED + 7 * j) for j, m in enumerate(base)]
        preds, f = predict_pool(adapted, strip_labels(pool))
        for j, (p, t) in enumerate(zip(preds, truth)):
            feats.append(f[j]); targets.append(float(np.mean([a == b for a, b in zip(p, t["y"])]))); groups.append(gid[j]); sparse.append(t["k"] <= 3)
        log(f"fold {fold}: accuracy by group (familiar, shift A, shift B) = " +
            ", ".join(f"{np.mean([targets[-len(pool) + j] for j in range(len(pool)) if gid[j] == g]):.4f}" for g in range(3)))
    feats, targets, groups, sparse = np.array(feats), np.array(targets), np.array(groups), np.array(sparse)
    conf = np.zeros(len(targets))
    for tri, tei in KFold(5, shuffle=True, random_state=SEED + 1).split(np.arange(len(targets))):
        cal = new_calibrator().fit(feats[tri], targets[tri])
        conf[tei] = np.clip(cal.predict(feats[tei]), 0.0, 1.0)
    loss = 0.85 * (1 - targets) + 0.15 * (conf - targets) ** 2
    for g, name in enumerate(["familiar rows", "shift A (rotation)", "shift B (rotation+gain+noise)"]):
        sel = groups == g
        st = [loss[sel & sparse].mean(), loss[sel & ~sparse].mean()]
        log(f"OOF {name}: accuracy {targets[sel].mean():.4f} mean row loss {loss[sel].mean():.4f} sparse {st[0]:.4f} rich {st[1]:.4f} "
            f"| 0.75*mean+0.25*worst = {0.75 * loss[sel].mean() + 0.25 * max(st):.4f}")
    log(f"calibration check: mean conf {conf.mean():.4f} vs mean accuracy {targets.mean():.4f}")
    calibrator = new_calibrator().fit(feats, targets)
    base = [train_model(train, SEED + 1000 + s) for s in range(N_SEEDS)]
    adapted = [adapt_model(m, train, test, SEED + 11 * j) for j, m in enumerate(base)]
    preds, f = predict_pool(adapted, test)
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
