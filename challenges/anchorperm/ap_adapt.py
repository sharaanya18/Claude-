import numpy as np, torch, torch.nn.functional as F
import ap_model as M
from scipy.linalg import expm


def regime_maps(theta, seed, gain=1.0):
    g = np.random.default_rng(seed); mats = []
    for _ in range(2):
        G = g.normal(size=(32, 32)) / 32 ** 0.5
        mats.append(expm((G - G.T) / 2 ** 0.5 * theta) * gain)
    return mats


def apply_regime(rows, mats, noise=0.0, seed=0):
    g = np.random.default_rng(seed); out = []
    for r in rows:
        r = dict(r)
        for key, R in zip(("Q", "A"), mats):
            x = r[key] @ R + (g.normal(0, noise, r[key].shape) if noise > 0 else 0)
            r[key] = np.clip(np.round(x), -2, 2).astype(np.float32)
        out.append(r)
    return out


def strip_labels(rows):
    return [{k: v for k, v in r.items() if k not in ("y", "pi")} for r in rows]


def anchor_tensors(rows):
    Q, A, nm, _ = M.pad_rows(rows)
    pi = M.anchored_pi(rows)
    for b in range(len(rows)):
        n = rows[b]["n"]
        free = [i for i in range(M.N_MAX) if i >= n]
        pi[b, n:] = torch.arange(n, M.N_MAX)
    return Q, A, nm, pi


def anchor_loss(model, Q, A, nm, pi, anc_lists, tasks, aug_noise=0.0, gen=None):
    B = len(tasks); am = torch.zeros(B, M.N_MAX, dtype=torch.bool); qi = torch.zeros(B, dtype=torch.long); ai = torch.zeros(B, dtype=torch.long)
    rows_ix = torch.tensor([t[0] for t in tasks])
    for j, (ri, t) in enumerate(tasks):
        for k, (q, a) in enumerate(anc_lists[ri]):
            if k != t: am[j, q] = True
        qi[j], ai[j] = anc_lists[ri][t]
    Qb, Ab, nmb, pib = Q[rows_ix], A[rows_ix], nm[rows_ix], pi[rows_ix]
    if aug_noise > 0:
        Qb = Qb + torch.randn(Qb.shape, generator=gen) * aug_noise * nmb[..., None]; Ab = Ab + torch.randn(Ab.shape, generator=gen) * aug_noise * nmb[..., None]
    logits = model(Qb, Ab, nmb, am, pib)
    ans_consumed = torch.zeros(B, M.N_MAX, dtype=torch.bool)
    for j in range(B):
        for k, (q, a) in enumerate(anc_lists[rows_ix[j].item()]):
            if k != tasks[j][1]: ans_consumed[j, a] = True
    cand_a = nmb & ~ans_consumed; cand_q = nmb & ~am
    big = -1e9
    lr = logits[torch.arange(B), qi].masked_fill(~cand_a, big)
    lc = logits[torch.arange(B), :, ai].masked_fill(~cand_q, big)
    return 0.5 * (F.cross_entropy(lr, ai) + F.cross_entropy(lc, qi))


def adapt(model, train_rows, test_rows, epochs=10, lr=5e-4, bs_a=64, bs_t=32, seed=0, aug_noise=0.15, gain_max=1.6, sp=0.0, freeze=None, wt=0.5, adapter=False, adapter_lr=None):
    import copy
    m = copy.deepcopy(model); gen = torch.Generator().manual_seed(seed)
    rows = strip_labels(test_rows)
    Qs, As, nms, pis = anchor_tensors(rows); anc = [r["anc"] for r in rows]
    tasks = [(i, t) for i, a in enumerate(anc) for t in range(len(a))]
    Q0, A0, nm0, pi0 = M.pad_rows(train_rows); ns = [r["n"] for r in train_rows]
    ref = [p.detach().clone() for p in m.parameters()]
    if freeze:
        for n_, p in m.named_parameters():
            if any(n_.startswith(f) for f in freeze): p.requires_grad_(False)
    groups = []
    if adapter:
        m.adq.requires_grad_(True); m.ada.requires_grad_(True)
        groups.append({'params': [m.adq, m.ada], 'lr': adapter_lr or lr, 'weight_decay': 0.0})
    others = [p for n_, p in m.named_parameters() if p.requires_grad and n_ not in ('adq', 'ada')]
    if others: groups.append({'params': others, 'lr': lr, 'weight_decay': 1e-2})
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * (len(tasks) // bs_a + 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        m.train(); order = torch.randperm(len(tasks), generator=gen)
        for s in range(0, len(tasks), bs_a):
            batch = [tasks[i] for i in order[s:s + bs_a].tolist()]
            la = anchor_loss(m, Qs, As, nms, pis, anc, batch, aug_noise, gen)
            ix = torch.randint(0, len(train_rows), (bs_t,), generator=gen)
            Q, A = M.augment(Q0[ix], A0[ix], nm0[ix], gen, 0.6, gain_max)
            am = M.sample_anchors([ns[i] for i in ix.tolist()], gen)
            lt = M.listwise_loss(m(Q, A, nm0[ix], am, pi0[ix]), nm0[ix], am, pi0[ix])
            loss = la + wt * lt
            if sp > 0: loss = loss + sp * sum(((p - r) ** 2).sum() for p, r in zip(m.parameters(), ref))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval(); return m
