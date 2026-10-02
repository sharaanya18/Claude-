"""Batched float64 torch re-implementation of learner.assess for many candidate orders at once (per-request sandbox)."""
import numpy as np, torch
import learner as L
torch.set_num_threads(4)
DT = torch.float64
SZ = [int(np.prod(s)) for s in L.SIZES]

def _unpack(w):
    n = w.shape[0]; o = 0; out = []
    for s, z in zip(L.SIZES, SZ):
        out.append(w[:, o:o + z].reshape((n,) + s)); o += z
    return out

def _grad(w, x, y):
    """w (n,P); x (n,4,192); y (n,4) long -> gradient (n,P) of mean CE per row."""
    a, b, c, dd = _unpack(w)
    h = torch.tanh(torch.bmm(x, a) + b[:, None, :])                 # (n,4,12)
    z = torch.bmm(h, c) + dd[:, None, :]
    q = torch.softmax(z, dim=2)
    q = q.clone(); q.scatter_add_(2, y[:, :, None], -torch.ones_like(q[:, :, :1])); q = q / y.shape[1]
    dh = torch.bmm(q, c.transpose(1, 2)) * (1 - h * h)              # (n,4,12)
    return torch.cat([torch.bmm(x.transpose(1, 2), dh).reshape(w.shape[0], -1), dh.sum(1),
                      torch.bmm(h.transpose(1, 2), q).reshape(w.shape[0], -1), q.sum(1)], 1)

def _step(st, x, y):
    w, m, v, t = st; g = _grad(w, x, y); t = t + 1
    m = L.BETA1 * m + (1 - L.BETA1) * g; v = L.BETA2 * v + (1 - L.BETA2) * g * g
    w = w - L.LR * (m / (1 - L.BETA1 ** t)) / (torch.sqrt(v / (1 - L.BETA2 ** t)) + L.EPS)
    return w, m, v, t

def _probs(w, x):
    a, b, c, dd = _unpack(w); h = torch.tanh(torch.matmul(x[None], a) + b[:, None, :])
    return torch.softmax(torch.matmul(h, c) + dd[:, None, :], dim=2)    # (n,12,4)

def _ce(w, x, y):
    p = _probs(w, x); return -torch.log(torch.clamp(p[:, torch.arange(len(y)), y], min=1e-15)).mean(1)

class Sandbox:
    def __init__(self, d, device="cpu"):
        self.dev = device
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=device)
        self.xb = T(d["x_batches"]); self.yb = torch.as_tensor(d["batch_labels"], dtype=torch.long, device=device)
        self.xm = T(d["x_monitor"]); self.ym = torch.as_tensor(d["monitor_labels"], dtype=torch.long, device=device)
        self.w0 = T(d["weights"]); self.t0 = int(d["step"]); self.cont = [list(map(int, c)) for c in d["continuations"]]
    def run(self, orders, mom, chunk=4096):
        """orders (n,8) ints; mom (2,2,P) array [branch, m|v]. Returns D (n,), losses (n,2)."""
        orders = torch.as_tensor(np.asarray(orders), dtype=torch.long, device=self.dev); n = len(orders)
        Ds = []; Ls = []
        mom = torch.as_tensor(np.asarray(mom), dtype=DT, device=self.dev)
        for s in range(0, n, chunk):
            o = orders[s:s + chunk]; k = len(o); ends = []
            for b in range(2):
                st = (self.w0[None].repeat(k, 1), mom[b, 0][None].repeat(k, 1), mom[b, 1][None].repeat(k, 1), self.t0)
                for pos in range(8):
                    j = o[:, pos]; st = _step(st, self.xb[j], self.yb[j])
                ends.append(st)
            Ls.append(torch.stack([_ce(e[0], self.xm, self.ym) for e in ends], 1))
            dist = 0
            for c in self.cont:
                fin = []
                for e in ends:
                    st = e
                    for j in c:
                        jj = torch.full((k,), j, dtype=torch.long, device=self.dev); st = _step(st, self.xb[jj], self.yb[jj])
                    fin.append(_probs(st[0], self.xm))
                dist = dist + ((fin[0] - fin[1]) ** 2).mean((1, 2))
            Ds.append(dist / len(self.cont))
        return torch.cat(Ds).cpu().numpy(), torch.cat(Ls).cpu().numpy()
