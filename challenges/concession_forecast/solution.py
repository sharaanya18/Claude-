import json
import os
import random
import sys
import time
from pathlib import Path
PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/public')
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./working/submission.csv')
WORK = SUBMISSION_OUT.parent
WORK.mkdir(parents=True, exist_ok=True)
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = str(WORK / 'hf_cache')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
SEED = 42
DEVICE = 'cuda'
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN = 512
TOPIC_BUDGET = 96
PRESSURE_BUDGET = 160
N_FOLDS = 4
EPOCHS = 2
BATCH = 8
ACCUM = 2
LR = 2e-05
WARMUP = 0.1
AUX_W = 0.5
FAMILY_DIST = 0.5
THRESH_GRID = np.round(np.linspace(0.04, 0.7, 34), 3)
KINDS = ('position', 'correction')
PATHS = np.array([[p >> 3 - t & 1 for t in range(4)] for p in range(16)])
T0 = time.time()
LOG = open(WORK / 'train_log.txt', 'w')

def log(msg):
    line = f'[{time.time() - T0:7.0f}s] {msg}'
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
INTERVALS = [(a, b) for a in range(4) for b in range(a, 4)]
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]

def comparisons(Z):
    cs = np.concatenate([np.zeros(Z.shape[:2] + (1,), int), np.cumsum(Z, 2)], 2)
    C = np.stack([cs[:, :, b + 1] - cs[:, :, a] for a, b in INTERVALS], 2)
    Ci = np.stack([C[:, i] for i, _ in PAIRS], 1)
    Cj = np.stack([C[:, j] for _, j in PAIRS], 1)
    return (np.sign(Ci - Cj), Ci, Cj)

def macro_f1(tp, fp, fn):
    return float(np.mean([0.0 if 2 * tp[c] + fp[c] + fn[c] == 0 else 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in tp]))

def kind_score(Zp, Zg):
    Rg, Cig, Cjg = comparisons(Zg)
    Rp, Cip, Cjp = comparisons(Zp)
    rec = (Rp == Rg) & (Cip == Cig) & (Cjp == Cjg)
    tp = {c: int(((Rg == c) & rec).sum()) for c in (-1, 0, 1)}
    fn = {c: int(((Rg == c) & ~rec).sum()) for c in (-1, 0, 1)}
    fp = {c: int(((Rp == c) & ~rec).sum()) for c in (-1, 0, 1)}
    tpb = {c: int(((Zg == c) & (Zp == c)).sum()) for c in (0, 1)}
    fpb = {c: int(((Zg != c) & (Zp == c)).sum()) for c in (0, 1)}
    fnb = {c: int(((Zg == c) & (Zp != c)).sum()) for c in (0, 1)}
    exact = float((Zp == Zg).all((1, 2)).mean())
    return 0.7 * macro_f1(tp, fp, fn) + 0.15 * macro_f1(tpb, fpb, fnb) + 0.15 * exact

def load(public_dir, name, with_labels):
    df = pd.read_csv(public_dir / f'{name}.csv')
    if with_labels:
        lab = pd.read_csv(public_dir / 'train_targets.csv')
        df = df.merge(lab, on='target_id', how='left', validate='one_to_one')
        assert df['prediction'].notna().all()
    eps = []
    for h, r in enumerate(df.itertuples(index=False)):
        E = json.loads(r.episodes)
        assert len(E) == 4
        Y = json.loads(r.prediction)['concessions'] if with_labels else [[0] * 4] * 4
        for i, e in enumerate(E):
            assert len(e['pressure']) == 4
            eps.append(dict(h=h, slot=i, kind=r.interaction_kind, y=list(map(int, Y[i])), **e))
    return (df, pd.DataFrame(eps))

def encode(tok, ep):
    topic = tok(f"{ep['kind']}. Reference: {ep['reference']} User: {ep['opening_user']}", add_special_tokens=False)['input_ids'][:TOPIC_BUDGET]
    pres = []
    if ep['kind'] == 'correction':
        for t, p in enumerate(ep['pressure']):
            pres += tok(f' [{t + 1}] {p}', add_special_tokens=False)['input_ids'][:PRESSURE_BUDGET // 4]
    asst = tok(' Assistant: ' + ep['opening_assistant'], add_special_tokens=False)['input_ids']
    room = MAX_LEN - 3 - len(topic) - len(pres)
    if len(asst) > room:
        head = 2 * room // 3
        asst = asst[:head] + asst[len(asst) - (room - head):]
    return [tok.cls_token_id] + topic + [tok.sep_token_id] + asst + pres + [tok.sep_token_id]

def families(train_eps):
    fam = np.zeros(len(train_eps), int)
    offset = 0
    for kind in KINDS:
        idx = np.where(train_eps['kind'].values == kind)[0]
        text = (train_eps['reference'].values[idx] + ' ' + train_eps['opening_user'].values[idx]).tolist()
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english').fit_transform(text)
        Z = TruncatedSVD(200, random_state=SEED).fit_transform(X)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-09
        lab = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=FAMILY_DIST).fit_predict(Z)
        fam[idx] = lab + offset
        offset += lab.max() + 1
    return fam

class PathModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        d = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(2 * d, 2 * 16)

    def forward(self, ids, mask, kind):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = torch.cat([h[:, 0], (h * m).sum(1) / m.sum(1)], -1)
        out = self.head(self.drop(pooled)).view(-1, 2, 16)
        return out[torch.arange(len(kind), device=ids.device), kind]

def collate(seqs, pad_id):
    L = max((len(s) for s in seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return (ids.to(DEVICE), mask.to(DEVICE))
PATHS_T = None

def path_marginals(logp):
    return torch.exp(logp) @ PATHS_T

def train_fold(seqs, kinds, ys, pad_id, tag):
    seed_everything()
    model = PathModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = EPOCHS * ((len(seqs) + BATCH * ACCUM - 1) // (BATCH * ACCUM))
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * n_steps), n_steps)
    scaler = torch.amp.GradScaler('cuda')
    path_id = torch.tensor((ys * np.array([8, 4, 2, 1])).sum(1), device=DEVICE)
    yt = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
    kt = torch.tensor(kinds, device=DEVICE)
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(seqs))
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for k, s in enumerate(range(0, len(order), BATCH)):
            b = order[s:s + BATCH]
            ids, mask = collate([seqs[i] for i in b], pad_id)
            bt = torch.as_tensor(b, device=DEVICE)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(ids, mask, kt[bt])
            logp = torch.log_softmax(logits.float(), -1)
            marg = path_marginals(logp).clamp(1e-06, 1 - 1e-06)
            loss = F.nll_loss(logp, path_id[bt]) + AUX_W * F.binary_cross_entropy(marg, yt[bt])
            scaler.scale(loss / ACCUM).backward()
            tot += loss.item() * len(b)
            if (k + 1) % ACCUM == 0 or s + BATCH >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        log(f'[{tag}] epoch {ep + 1}/{EPOCHS} loss {tot / len(order):.4f}')
    return model

@torch.no_grad()
def predict(model, seqs, kinds, pad_id):
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), 16))
    kt = torch.tensor(kinds, device=DEVICE)
    for s in range(0, len(order), 64):
        b = order[s:s + 64]
        ids, mask = collate([seqs[i] for i in b], pad_id)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(ids, mask, kt[torch.as_tensor(b, device=DEVICE)])
        out[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out

def to_hearings(step_probs, eps, n_hearings):
    P = np.zeros((n_hearings, 4, 4))
    P[eps['h'].values, eps['slot'].values] = step_probs
    return P

def fit_thresholds(P, Y):
    th = np.full(4, 0.99)
    best = kind_score((P > th).astype(int), Y)
    for _ in range(3):
        for t in range(4):
            for v in THRESH_GRID:
                cand = th.copy()
                cand[t] = v
                s = kind_score((P > cand).astype(int), Y)
                if s > best:
                    best, th = (s, cand)
    return (th, best)

def validate_submission(sub, sample_path, test_ids):
    sample = pd.read_csv(sample_path)
    assert list(sub.columns) == ['target_id', 'prediction'] == list(sample.columns)
    assert sub['target_id'].is_unique and set(sub['target_id']) == set(sample['target_id']) == set(test_ids)
    for p in sub['prediction']:
        o = json.loads(p)
        assert list(o) == ['concessions'] and len(o['concessions']) == 4
        for row in o['concessions']:
            assert len(row) == 4 and all((type(v) is int and v in (0, 1) for v in row))

def main():
    global PATHS_T
    seed_everything()
    PATHS_T = torch.tensor(PATHS, dtype=torch.float32, device=DEVICE)
    tr_df, tr = load(PUBLIC_DIR, 'train', True)
    te_df, te = load(PUBLIC_DIR, 'test', False)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_seqs = [encode(tok, r) for r in tr.to_dict('records')]
    te_seqs = [encode(tok, r) for r in te.to_dict('records')]
    kind_id = {k: i for i, k in enumerate(KINDS)}
    tr_k = tr['kind'].map(kind_id).values
    te_k = te['kind'].map(kind_id).values
    ys = np.array(tr['y'].tolist())
    log(f'train episodes {len(tr)}, test episodes {len(te)}, mean tokens {np.mean([len(s) for s in tr_seqs]):.0f}')
    fam = families(tr)
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=fam))
    log(f'paraphrase families {len(set(fam))}; fold sizes {[len(v) for _, v in folds]}')
    oof = np.zeros((len(tr), 16))
    te_prob = np.zeros((len(te), 16))
    ckpt = WORK / 'checkpoints'
    ckpt.mkdir(exist_ok=True)
    for f, (trn, val) in enumerate(folds):
        model = train_fold([tr_seqs[i] for i in trn], tr_k[trn], ys[trn], tok.pad_token_id, f'fold {f}')
        oof[val] = predict(model, [tr_seqs[i] for i in val], tr_k[val], tok.pad_token_id)
        te_prob += predict(model, te_seqs, te_k, tok.pad_token_id) / N_FOLDS
        torch.save({k: v.half() for k, v in model.state_dict().items()}, ckpt / f'fold{f}.pt')
        del model
        torch.cuda.empty_cache()
        log(f'fold {f} done; OOF path log-loss {-np.mean(np.log(oof[val][np.arange(len(val)), (ys[val] * [8, 4, 2, 1]).sum(1)] + 1e-09)):.4f}')
    oof_step = oof @ PATHS
    te_step = te_prob @ PATHS
    np.save(WORK / 'oof_path_probs.npy', oof)
    Ptr = to_hearings(oof_step, tr, len(tr_df))
    Ytr = to_hearings(ys, tr, len(tr_df)).astype(int)
    Pte = to_hearings(te_step, te, len(te_df))
    thresholds, cv = ({}, [])
    for kind in KINDS:
        s = tr_df['interaction_kind'].values == kind
        th, best = fit_thresholds(Ptr[s], Ytr[s])
        zero = kind_score(np.zeros_like(Ytr[s]), Ytr[s])
        idx = np.random.default_rng(SEED).permutation(int(s.sum()))
        A, B = (idx[:len(idx) // 2], idx[len(idx) // 2:])
        tA, _ = fit_thresholds(Ptr[s][A], Ytr[s][A])
        tB, _ = fit_thresholds(Ptr[s][B], Ytr[s][B])
        cross = 0.5 * (kind_score((Ptr[s][B] > tA).astype(int), Ytr[s][B]) + kind_score((Ptr[s][A] > tB).astype(int), Ytr[s][A]))
        thresholds[kind] = th
        cv.append(cross)
        log(f'{kind}: thresholds {th.tolist()} | OOF score in-sample {100 * best:.2f}, cross-fitted {100 * cross:.2f}, all-zero {100 * zero:.2f}')
    log(f'estimated score (cross-fitted, mean over kinds): {100 * np.mean(cv):.2f}')
    rows = []
    for h, r in enumerate(te_df.itertuples(index=False)):
        Z = (Pte[h] > thresholds[r.interaction_kind]).astype(int)
        rows.append((r.target_id, json.dumps({'concessions': Z.tolist()}, separators=(',', ':'))))
    sub = pd.DataFrame(rows, columns=['target_id', 'prediction'])
    validate_submission(sub, PUBLIC_DIR / 'sample_submission.csv', te_df['target_id'])
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} ({len(sub)} hearings); predicted concession rate {np.mean([np.mean(json.loads(p)['concessions']) for p in sub.prediction]):.3f}")
if __name__ == '__main__':
    main()
