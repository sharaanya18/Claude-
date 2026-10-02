"""Dev harness: grouped-CV for the audio+text tagger. Usage: python3 cvnn.py --mode both --epochs 30"""
import sys, json, time, pickle, argparse, random
sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.model_selection import GroupKFold
from metric import score, parse

CL = ['none', '31', '3131', '3151', '3153', '51', '53', '5331', '5351', '5353']
D = 'dataset/public/'


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Net(nn.Module):
    def __init__(self, nv, n_ssl_layers, H=768, d=128, mode='both', drop=0.3, nF0=6):
        super().__init__()
        self.mode = mode; self.d = d
        self.emb = nn.Embedding(nv, 64, padding_idx=0)
        self.tin = nn.Linear(64 + 3, d)
        self.tenc = nn.GRU(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.lw = nn.Parameter(torch.zeros(n_ssl_layers))
        self.ssl_ln = nn.LayerNorm(H)
        self.ssl_proj = nn.Linear(H, d)
        self.f0_proj = nn.Linear(nF0, d)
        self.aenc = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.sig = nn.Parameter(torch.tensor(0.0))  # learned width of relative-position prior
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, len(CL)))

    def forward(self, ch, cfeat, ssl, f0, amask, cmask):
        # ch (B,N) cfeat (B,N,3) ssl (B,nl,T,H) f0 (B,T,6) amask (B,T) bool valid cmask (B,N)
        t = self.drop(self.tin(torch.cat([self.emb(ch), cfeat], -1)))
        t, _ = self.tenc(t)
        if self.mode == 'text':
            return self.head(torch.cat([t, torch.zeros_like(t)], -1))
        a = 0
        if self.mode in ('both', 'ssl'):
            w = torch.softmax(self.lw, 0)
            s = (ssl.float() * w[None, :, None, None]).sum(1)
            a = a + self.ssl_proj(self.ssl_ln(s))
        if self.mode in ('both', 'f0'):
            a = a + self.f0_proj(f0)
        a, _ = self.aenc(self.drop(a))
        B, T, _ = a.shape; N = ch.shape[1]
        att = torch.einsum('bnd,btd->bnt', self.q(t), self.k(a)) / self.d ** 0.5
        # monotone-ish prior: char n (end boundary) ~ time fraction; width learned
        nlen = cmask.sum(1, keepdim=True).float(); tlen = amask.sum(1, keepdim=True).float()
        cpos = (torch.arange(N, device=ch.device)[None].float() + 1) / nlen  # (B,N)
        tpos = (torch.arange(T, device=ch.device)[None].float() + 0.5) / tlen
        s2 = (0.08 + 0.3 * torch.sigmoid(self.sig))
        att = att - ((cpos[:, :, None] - tpos[:, None, :]) ** 2) / (2 * s2 ** 2)
        att = att.masked_fill(~amask[:, None, :], -1e9)
        ctx = torch.einsum('bnt,btd->bnd', torch.softmax(att, -1), self.v(a))
        return self.head(torch.cat([t, ctx], -1))


def make_data(df, cache, vocab, layers_idx):
    items = []
    for _, r in df.iterrows():
        p = r.phones; L = len(p)
        ch = [vocab.get(c, 1) for c in p]
        cf = [[(i + 1) / L, min(L - i - 1, 6) / 6, float(i == L - 1)] for i in range(L)]
        c = cache[r.id]
        ssl = c['ssl'][layers_idx]
        T = ssl.shape[1]
        f0 = torch.from_numpy(c['f0']).T[None]  # (1,6,Tf)
        f0 = Fn.adaptive_avg_pool1d(f0, T)[0].T.numpy()
        y = np.zeros(L, dtype=np.int64)
        if 'tones' in r and isinstance(r.get('tones'), str):
            for o, cc in parse(r.tones): y[o - 1] = CL.index(cc)
        items.append(dict(ch=np.array(ch), cf=np.array(cf, dtype=np.float32), ssl=ssl, f0=f0, y=y))
    return items


def collate(items, train=False, tmask=0.0):
    B = len(items); N = max(len(i['ch']) for i in items); T = max(i['f0'].shape[0] for i in items)
    nl, H = items[0]['ssl'].shape[0], items[0]['ssl'].shape[2]
    ch = torch.zeros(B, N, dtype=torch.long); cf = torch.zeros(B, N, 3); ssl = torch.zeros(B, nl, T, H, dtype=torch.float16)
    f0 = torch.zeros(B, T, items[0]['f0'].shape[1]); am = torch.zeros(B, T, dtype=torch.bool); cm = torch.zeros(B, N, dtype=torch.bool)
    y = torch.full((B, N), -100, dtype=torch.long)
    for b, it in enumerate(items):
        n, t = len(it['ch']), it['f0'].shape[0]
        ch[b, :n] = torch.from_numpy(it['ch']); cf[b, :n] = torch.from_numpy(it['cf'])
        ssl[b, :, :t] = torch.from_numpy(it['ssl']); f0[b, :t] = torch.from_numpy(it['f0'])
        am[b, :t] = True; cm[b, :n] = True; y[b, :n] = torch.from_numpy(it['y'])
    if train and tmask > 0:  # time masking of audio frames (SpecAugment-style) on features only
        for b in range(B):
            t = int(am[b].sum()); w = int(t * tmask)
            if w > 0:
                s = np.random.randint(0, max(1, t - w)); ssl[b, :, s:s + w] = 0; f0[b, s:s + w] = 0
    return ch, cf, ssl, f0, am, cm, y


def decode(prob, L):
    P = []
    for i in range(L):
        if prob[i, 0] < 0.5:
            P.append((i + 1, CL[1 + int(np.argmax(prob[i, 1:]))]))
    return P


def predict(model, items, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(items), bs):
            ch, cf, ssl, f0, am, cm, y = collate(items[s:s + bs])
            pr = torch.softmax(model(ch, cf, ssl, f0, am, cm), -1).numpy()
            for b, it in enumerate(items[s:s + bs]): out.append(pr[b, :len(it['ch'])])
    return out


def train_model(tr_items, nv, nl, args, seed):
    seed_all(seed)
    model = Net(nv, nl, mode=args.mode, drop=args.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr_items) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    w = torch.ones(len(CL)); w[0] = args.w_none
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(tr_items)); tot = 0
        for s in range(0, len(perm), args.bs):
            ch, cf, ssl, f0, am, cm, y = collate([tr_items[i] for i in perm[s:s + args.bs]], True, args.tmask)
            lg = model(ch, cf, ssl, f0, am, cm)
            loss = Fn.cross_entropy(lg.reshape(-1, len(CL)), y.reshape(-1), weight=w, ignore_index=-100, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both'); ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--drop', type=float, default=0.3); ap.add_argument('--tmask', type=float, default=0.1)
    ap.add_argument('--w_none', type=float, default=0.5); ap.add_argument('--layers', default='0,1,2,3')
    ap.add_argument('--cache', default='working/cache_wavlm-base-plus.pkl'); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_num_threads(4)
    tr = pd.read_csv(D + 'train.csv', keep_default_na=False)
    cache = pickle.load(open(args.cache, 'rb'))
    vocab = {c: i + 2 for i, c in enumerate(sorted(set(''.join(tr.phones))))}
    li = [int(v) for v in args.layers.split(',')]
    items = make_data(tr, cache, vocab, li)
    ref = [parse(s) for s in tr.tones]
    grp = tr.phones.factorize()[0]
    oof = [None] * len(tr); t0 = time.time(); fs = []
    for f, (a, b) in enumerate(GroupKFold(args.folds).split(items, groups=grp)):
        m = train_model([items[i] for i in a], len(vocab) + 2, len(li), args, 1000 + f)
        pr = predict(m, [items[i] for i in b])
        for i, p in zip(b, pr): oof[i] = p
        fs.append(score([decode(oof[i], len(items[i]['ch'])) for i in b], [ref[i] for i in b]))
        print(f'fold {f} {fs[-1]:.4f} [{time.time()-t0:.0f}s]', flush=True)
    pred = [decode(oof[i], len(items[i]['ch'])) for i in range(len(tr))]
    print(f'RESULT mode={args.mode} {args.tag} CV={score(pred, ref):.4f} folds={np.round(fs,4)} std={np.std(fs):.4f}')
    pickle.dump(oof, open(f'working/oof_{args.mode}{args.tag}.pkl', 'wb'))
