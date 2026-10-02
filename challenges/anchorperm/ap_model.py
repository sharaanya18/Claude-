"""AnchorPerm model pieces (dev module; assembled into solution.py later)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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
        sd = (((s1 - mu) ** 2) * nm[:, None, :].float()).sum(2, keepdim=True).div(cntA).sqrt().clamp_min(1e-6)
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


def train_model(rows, epochs=30, bs=64, lr=2e-3, seed=0, noise_max=0.6, gain_max=1.0, log=print, use_tower=True, ctx_layers=0, drop=0.0, wd=1e-2, noise_min=0.0, requant=False, rich=0, rot_prob=0.0):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = PairScorer(use_tower=use_tower, ctx_layers=ctx_layers, drop=drop, rich=rich)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(rows) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Q0, A0, nm0, pi0 = pad_rows(rows)
    ns = [r["n"] for r in rows]
    for ep in range(epochs):
        model.train(); order = torch.randperm(len(rows), generator=gen); tot = 0.0
        for s in range(0, len(rows), bs):
            ix = order[s:s + bs]
            Q, A = augment(Q0[ix], A0[ix], nm0[ix], gen, noise_max, gain_max, noise_min, requant, rot_prob)
            am = sample_anchors([ns[i] for i in ix.tolist()], gen)
            logits = model(Q, A, nm0[ix], am, pi0[ix])
            loss = listwise_loss(logits, nm0[ix], am, pi0[ix])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss.detach()) * len(ix)
        if ep % 5 == 0 or ep == epochs - 1: log(f"epoch {ep + 1}/{epochs} loss {tot / len(rows):.4f}")
    return model


def evaluate(model, rows):
    lg = predict_logits(model, rows).numpy(); accs = []; res = []
    for l, r in zip(lg, rows):
        pred, S = decode_row(l, r)
        accs.append(np.mean([p == t for p, t in zip(pred, r["y"])])); res.append((pred, S))
    return np.array(accs), res
