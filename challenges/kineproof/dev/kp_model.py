"""Temporal model + metric-aligned loss + grouped-CV trainer (dev harness).

Architecture: dilated residual temporal CNN (TCN).
  Why this and not a transformer: 1,093 trials from 32 people is small. The
  task is a LOCAL-IN-TIME physical map (force at t is COM acceleration at t)
  plus a medium-range gait-phase context. A dilated conv stack has exactly that
  inductive bias, is translation-equivariant along time (the window is an
  arbitrary 1.7 s cut, so equivariance is correct), and has a receptive field
  we can set explicitly. Kernel 5 with dilations 1,2,4,8,16,32 over 6 blocks of
  2 convs gives RF = 1 + 4*(1+2+4+8+16+32) = 253 frames ~ the whole window.

Loss: EXACTLY the metric's error term.
  metric: E_ia = sum_t|p-y| / sum_t|y|,  score = mean_ia max(0, 1-E_ia)
  loss  : mean_ia [ sum_t|p-y| / sum_t|y| ]   == 1 - score (before clipping)
  So minimising this loss maximises the official score directly. Note that this
  automatically balances the axes: force_y's denominator is ~11x force_x's and
  ~22x force_z's, so a plain L1 would be dominated by the vertical axis and
  cap the score near 1/3 (the description says as much).
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResBlock(nn.Module):
    def __init__(self, c: int, dil: int, k: int = 5, drop: float = 0.0):
        super().__init__()
        pad = dil * (k - 1) // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad, dilation=dil)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, k, padding=pad, dilation=dil)
        self.n2 = nn.GroupNorm(8, c)
        self.do = nn.Dropout(drop)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.n1(self.c1(x)))
        h = self.do(h)
        h = self.n2(self.c2(h))
        return torch.nn.functional.gelu(x + h)


class ForceTCN(nn.Module):
    """(B, C_in, T) frame features + (B, A) trial scalars -> (B, 3, T) force."""

    def __init__(self, c_in: int, n_anthro: int, width: int = 96, blocks: int = 6,
                 k: int = 5, drop: float = 0.05, dil_cap: int = 32,
                 phys_residual: bool = True):
        super().__init__()
        self.phys_residual = phys_residual
        self.stem = nn.Conv1d(c_in + n_anthro, width, k, padding=k // 2)
        self.sn = nn.GroupNorm(8, width)
        dils = [min(2 ** i, dil_cap) for i in range(blocks)]
        self.blocks = nn.ModuleList([ResBlock(width, d, k, drop) for d in dils])
        self.head = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 3, 1))
        # start the head near zero so training begins at the physics prior
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, xf, xa, prior=None):
        B, _, T = xf.shape
        x = torch.cat([xf, xa[:, :, None].expand(-1, -1, T)], dim=1)
        h = torch.nn.functional.gelu(self.sn(self.stem(x)))
        for b in self.blocks:
            h = b(h)
        out = self.head(h)
        if self.phys_residual and prior is not None:
            out = out + prior          # learned correction on the Newtonian base
        return out


# ------------------------------------------------------------------ loss ----
def metric_loss(pred, truth, den, huber: float = 0.0):
    """Normalised-L1 loss == the metric's error term.

    pred/truth (B,3,T); den (B,3) = sum_t|truth| per trial-axis (label-derived
    weight, training-time only). huber>0 smooths the kink at 0 which stabilises
    early optimisation without changing the optimum materially.
    """
    r = pred - truth
    if huber > 0:
        a = r.abs()
        r = torch.where(a < huber, 0.5 * r * r / huber + 0.5 * huber, a)
    else:
        r = r.abs()
    return (r.sum(dim=2) / den.clamp_min(1e-6)).mean()


# --------------------------------------------------------------- training ---
class Standardiser:
    """Per-channel standardisation fitted on TRAIN rows of a fold only."""

    def __init__(self, FF, AF, idx):
        f = FF[idx]
        self.fm = f.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.fs = (f.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        a = AF[idx]
        self.am = a.mean(axis=0, keepdims=True).astype(np.float32)
        self.as_ = (a.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    def f(self, FF):
        return (FF - self.fm) / self.fs

    def a(self, AF):
        return (AF - self.am) / self.as_


def make_prior(FF, prior_ch):
    """The physics prior channels, in target units, straight from the features."""
    return FF[:, prior_ch, :].copy()


def train_one(FFtr, AFtr, Ytr, FFva, AFva, Yva, Ptr=None, Pva=None, *,
              width=96, blocks=6, k=5, drop=0.05, epochs=40, bs=32, lr=3e-3,
              wd=1e-4, seed=0, huber=0.02, phys_residual=True, mirror_pack=None,
              log=None, device="cpu", ema_decay=0.0):
    """Train one fold. Returns (val_pred, model, history).

    Fixed work plan: fixed epochs / batch size / schedule. No clock anywhere.
    """
    seed_all(seed)
    dev = torch.device(device)
    model = ForceTCN(FFtr.shape[1], AFtr.shape[1], width, blocks, k, drop,
                     phys_residual=phys_residual).to(dev)

    Xf = torch.from_numpy(FFtr); Xa = torch.from_numpy(AFtr)
    Yt = torch.from_numpy(Ytr.astype(np.float32))
    Pr = torch.from_numpy(Ptr.astype(np.float32)) if Ptr is not None else None
    den_t = Yt.abs().sum(dim=2)

    if mirror_pack is not None:
        mFf, mAf, mY, mPr = mirror_pack
        Xf = torch.cat([Xf, torch.from_numpy(mFf)], 0)
        Xa = torch.cat([Xa, torch.from_numpy(mAf)], 0)
        Yt = torch.cat([Yt, torch.from_numpy(mY.astype(np.float32))], 0)
        if Pr is not None:
            Pr = torch.cat([Pr, torch.from_numpy(mPr.astype(np.float32))], 0)
        den_t = Yt.abs().sum(dim=2)

    Vf = torch.from_numpy(FFva); Va = torch.from_numpy(AFva)
    Vy = torch.from_numpy(Yva.astype(np.float32))
    Vp = torch.from_numpy(Pva.astype(np.float32)) if Pva is not None else None
    den_v = Vy.abs().sum(dim=2)

    n = Xf.shape[0]
    steps_per = (n + bs - 1) // bs
    total = steps_per * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total, pct_start=0.15, div_factor=10.0,
        final_div_factor=100.0)

    g = torch.Generator().manual_seed(seed)
    ema = None
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tl = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            out = model(Xf[b], Xa[b], Pr[b] if Pr is not None else None)
            loss = metric_loss(out, Yt[b], den_t[b], huber)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss) * len(b)
        if ema_decay > 0:
            with torch.no_grad():
                sd = model.state_dict()
                if ema is None:
                    ema = {kk: v.detach().clone().float() for kk, v in sd.items()}
                else:
                    for kk, v in sd.items():
                        if ema[kk].dtype.is_floating_point:
                            ema[kk].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)
                        else:
                            ema[kk] = v.detach().clone().float()
        model.eval()
        with torch.no_grad():
            vp = model(Vf, Va, Vp)
            vl = float(metric_loss(vp, Vy, den_v, 0.0))
        hist.append((ep, tl / n, vl))
        if log and (ep % 5 == 0 or ep == epochs - 1):
            log(f"    ep {ep:3d} train {tl/n:.4f}  val_E {vl:.4f}  "
                f"val_score~{1-vl:.4f}  lr {sched.get_last_lr()[0]:.2e}")

    if ema is not None:
        model.load_state_dict({kk: v for kk, v in ema.items()}, strict=True)
    model.eval()
    with torch.no_grad():
        vp = model(Vf, Va, Vp).numpy()
    return vp, model, hist
