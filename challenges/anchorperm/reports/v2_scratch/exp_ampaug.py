import sys, os, numpy as np, torch
sys.path.insert(0, "/home/user/Claude-/challenges/anchorperm/reports/v2_scratch")
from sims import *
import ap_model as M
from dev_noise import noisy
torch.set_num_threads(4)
orig_aug = M.augment
PCt = {k: torch.tensor(PC[k].copy(), dtype=torch.float32) for k in ("Q", "A")}
def make_aug(mode, p=0.6, gmax=2.0):
    def aug(Q, A, nm, gen, *a, **kw):
        B = Q.shape[0]; out = []
        for X, k in ((Q, "Q"), (A, "A")):
            m = nm[..., None].float(); mu = (X * m).sum(1, keepdim=True) / m.sum(1, keepdim=True); Xc = (X - mu) * m
            r = int(torch.randint(2, 13, (1,), generator=gen))
            if mode == "pc": V = PCt[k][:, :r][None].expand(B, 32, r)
            else: V = torch.linalg.qr(torch.randn(B, 32, r, generator=gen))[0]
            g = 1 + torch.rand(B, 1, 1, generator=gen) * (gmax - 1); use = (torch.rand(B, 1, 1, generator=gen) < p).float()
            Xc = Xc + use * (g - 1) * torch.bmm(torch.bmm(Xc, V), V.transpose(1, 2))
            out.append(torch.clamp(torch.round(mu + Xc), -2, 2) * m)
        return orig_aug(out[0], out[1], nm, gen, *a, **kw)
    return aug
N = 500
sets = {"clean": va[:N], "amp1.8r8": amp(va[:N], 1.8, 8, 1), "amp2.2r4": amp(va[:N], 2.2, 4, 7), "amp1.6+prot0.7": amp(va[:N], 1.6, 8, 6, 0.7), "noise0.5": noisy(va[:N], 0.5, 3)}
for mode in ("none", "rand", "pc"):
    M.augment = orig_aug if mode == "none" else make_aug(mode)
    model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
    print(mode, {k: round(float(M.evaluate(model, v)[0].mean()), 4) for k, v in sets.items()}, flush=True)
