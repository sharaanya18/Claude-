import sys; sys.path.insert(0, '.')
import numpy as np, torch
from basis import *
torch.set_default_dtype(torch.float64)
MON = [0, 1, 3, 6, 9, 10]

def basis(d, r_m=6, r_v=6):
    G = grad_basis(d)
    U, s, _ = np.linalg.svd(G, full_matrices=False)
    G2 = np.concatenate([G ** 2, np.ones((G.shape[0], 1)) * 1e-4], 1)
    W, s2, _ = np.linalg.svd(G2, full_matrices=False)
    return U[:, :r_m] * 1.0, s[:r_m], W[:, :r_v], s2[:r_v]

def forward_probs(w, X):
    a = w[:2304].view(192, 12); b = w[2304:2316]; c = w[2316:2364].view(12, 4); e = w[2364:2368]
    z = torch.tanh(X @ a + b) @ c + e
    return torch.softmax(z, dim=-1)

def sim_diag(d, m, v):
    w0 = torch.tensor(d["weights"]); Xm = torch.tensor(d["x_monitor"])[MON]
    out = []
    for j in range(2):
        g = torch.tensor(L.gradient(d["weights"], d["x_diag"][j], d["diagnostic_labels"][j]))
        t = int(d["step"]) + 1
        m2 = 0.9 * m + 0.1 * g; v2 = 0.99 * v + 0.01 * g * g
        w2 = w0 - 0.01 * (m2 / (1 - 0.9 ** t)) / (torch.sqrt(torch.clamp(v2, min=0) / (1 - 0.99 ** t)) + 1e-8)
        out.append(forward_probs(w2, Xm))
    return torch.stack(out)               # (2 diag, 6, 4)

def fit_state(d, r_m=6, r_v=6, lam=1e-3, iters=200, seed=0):
    U, s, W, s2 = basis(d, r_m, r_v); U = torch.tensor(U); W = torch.tensor(W)
    obs = torch.tensor(d["diagnostics"])  # (2 branch, 2 diag, 6, 4)
    rng = np.random.default_rng(seed)
    z = torch.zeros(2, r_m, requires_grad=True); y = torch.zeros(2, r_v, requires_grad=True)
    def state(b):
        m = U @ z[b]; v = torch.clamp(W @ y[b] + 1e-4, min=1e-9); return m, v
    opt = torch.optim.LBFGS([z, y], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = 0
        for b in range(2):
            m, v = state(b); loss = loss + ((sim_diag(d, m, v) - obs[b]) ** 2).sum()
        loss = loss * 1e4 + lam * ((z ** 2).sum() + (y ** 2).sum() * 1e6)
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        ms = np.stack([[(U @ z[b]).numpy(), state(b)[1].numpy()] for b in range(2)])   # (2 branch, 2 kinds, 2368)
        res = sum(((sim_diag(d, *state(b)) - obs[b]) ** 2).sum().item() for b in range(2))
    return ms, res
