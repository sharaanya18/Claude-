"""E1 prototype: add rotation-invariant channel (hidden-hidden graph-matching term + norm-about-centroid product) to the head; train with fully-rotated episodes so the head learns when to trust it."""
import sys, os, numpy as np, torch, inspect, textwrap
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
import ap_model as M
from dev_common import rot
torch.set_num_threads(4)
src = textwrap.dedent(inspect.getsource(M.PairScorer.forward))
src = src.replace("f = torch.cat([s1[..., None], prod[..., None], absd[..., None], sq[..., None], mx[..., None], ctx, rel[..., None]], -1)",
"""ex = invariant_feats(Qn, An, nm, am, pi, Zq, Za, prod)
    f = torch.cat([s1[..., None], prod[..., None], absd[..., None], sq[..., None], mx[..., None], ctx, rel[..., None], ex], -1)""")
def invariant_feats(Qn, An, nm, am, pi, Zq, Za, prod, iters=5, tau=2.0):
    B, N, _ = Qn.shape
    aa_mask = torch.zeros(B, N).scatter_add_(1, pi, am.float()) > 0
    hq = (nm & ~am).float(); ha = (nm & ~aa_mask).float()
    Zq0 = Zq * hq[:, None, :] * hq[:, :, None]; Za0 = Za * ha[:, None, :] * ha[:, :, None]
    Zq0 = Zq0 * (1 - torch.eye(N))[None]; Za0 = Za0 * (1 - torch.eye(N))[None]
    mask = (hq[:, :, None] * ha[:, None, :]) > 0
    S = prod * 1.0
    for _ in range(iters):
        L = (S / tau).masked_fill(~mask, -1e4)
        for _ in range(10):
            L = L - torch.logsumexp(L, 2, keepdim=True); L = L - torch.logsumexp(L, 1, keepdim=True)
        P = torch.exp(L) * mask
        S = prod + torch.bmm(torch.bmm(Zq0, P), Za0.transpose(1, 2)) / hq.sum(1).clamp_min(1)[:, None, None]
    def znorm(X):
        v = X.norm(dim=-1); m = nm.float(); mu = (v * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        sd = (((v - mu) ** 2) * m).sum(1, keepdim=True).div(m.sum(1, keepdim=True)).sqrt().clamp_min(1e-6); return (v - mu) / sd * m
    nq, na = znorm(Qn), znorm(An)
    return torch.stack([S, nq[:, :, None] * na[:, None, :], (nq[:, :, None] - na[:, None, :]).abs()], -1)
M.invariant_feats = invariant_feats
ns = {}; exec(src, M.__dict__, ns)
class PS2(M.PairScorer):
    def __init__(self, **kw):
        super().__init__(**kw); self.head = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
PS2.forward = ns["forward"]
N = 500
sets = {"clean": va[:N], "prot0.7": amp(va[:N], 1.0, 8, 4, 0.7), "prot1.0": amp(va[:N], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:N], 0.5, 5), "rotfull": rot(va[:N], 2)}
Base = M.PairScorer
for name, cls, rp in (("base rot0.2", Base, 0.2), ("inv rot0.0", PS2, 0.0), ("inv rot0.2", PS2, 0.2)):
    M.PairScorer = cls
    model = M.train_model(fit, epochs=40, gain_max=1.6, rot_prob=rp, log=lambda *a: None)
    print(name, {k: round(float(M.evaluate(model, v)[0].mean()), 4) for k, v in sets.items()}, flush=True)
M.PairScorer = Base
