exec(open("/home/user/Claude-/challenges/anchorperm/reports/v2_scratch/exp_rowadapt.py").read().split("sets = {")[0])
def adapt_lr(r, lam, rdim, em_iters=6, tau=1.0):
    """Delta restricted to span of top-rdim train PCs on each side: W = W1 + Uq M Ua^T, M (rdim x rdim) ridge-fitted per row with EM soft hidden pairs."""
    Q, A = cen(r["Q"]), cen(r["A"]); hq, ha = r["hq"], r["ha"]; Uq, Ua = PC["Q"][:, :rdim], PC["A"][:, :rdim]
    aq = [a for a, b in r["anc"]]; ab = [b for a, b in r["anc"]]; W = W1
    for it in range(em_iters):
        Xs, Ys, ws = [Q[aq]], [A[ab]], [np.ones(len(aq))]
        if it > 0:
            C = (((Q[hq] @ W)[:, None, :] - A[ha][None]) ** 2).sum(-1); L = -C / (2 * s2) / tau
            for _ in range(30): L = L - np.logaddexp.reduce(L, 1, keepdims=True); L = L - np.logaddexp.reduce(L, 0, keepdims=True)
            P = np.exp(L); Xs.append(Q[hq]); Ys.append(P @ A[ha]); ws.append(P.max(1))
        X = np.concatenate(Xs) @ Uq; Y = (np.concatenate(Ys) - np.concatenate(Xs) @ W1) @ Ua; w = np.concatenate(ws)[:, None]
        Mm = np.linalg.solve(X.T @ (w * X) + lam * np.eye(rdim), X.T @ (w * Y)); W = W1 + Uq @ Mm @ Ua.T
    return W
sets = {"clean": va[:300], "prot0.7": amp(va[:300], 1.0, 8, 4, 0.7), "prot1.0": amp(va[:300], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:300], 0.5, 5)}
for name, rows in sets.items():
    out = {"global": np.mean([acc_of(r, W1)[0] for r in rows])}
    for lam in (30, 100, 300): out[f"em_full{lam}"] = np.mean([acc_of(r, adapt(r, lam, "em", em_iters=8))[0] for r in rows])
    for rd, lam in ((6, 3), (6, 10), (12, 10), (12, 30)): out[f"lr{rd}_{lam}"] = np.mean([acc_of(r, adapt_lr(r, lam, rd))[0] for r in rows])
    print(name, {k: round(float(v), 4) for k, v in out.items()}, flush=True)
