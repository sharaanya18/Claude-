import numpy as np
import solution as S
AGG = {a: i for i, a in enumerate(S.AGGFNS)}
UNIT = {'day': 0, 'week': 1, 'month': 2}

def cand_table(h, b, row, W, lo=0.10, off=0.0):
    cands = S.decode_chart(h, b, lo)
    if not cands:
        return [], np.zeros((0, 0))
    s = np.array([c[1] for c in cands]); xs = np.array([c[0] for c in cands]) + off
    tot = float(h[S.WIDTH_LEFT:W - S.WIDTH_LEFT].sum())
    order = np.argsort(-s); rank = np.empty(len(s)); rank[order] = np.arange(len(s))
    feats = []
    for k, (x, sc, p) in enumerate(cands):
        c = int(round(x)); win = h[max(0, c - 8):c + 9]
        dn = np.abs(xs - xs[k]); dn[k] = 999
        feats.append([sc, win.sum(), rank[k], (s >= 0.3).sum(), (s >= 0.5).sum(), s.sum(), s.max(), np.sort(s)[-2] if len(s) > 1 else 0,
                      tot, dn.min(), (x - S.WIDTH_LEFT) / (W - 2 * S.WIDTH_LEFT), p[0], p[1], p[2], AGG[row.aggfn], UNIT[row.aggunit], np.log(row.n_points)])
    return cands, np.array(feats)
