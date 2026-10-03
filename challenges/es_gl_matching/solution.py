"""Spanish clue -> Galician descriptor matching inside folios.

Pipeline (all fitting on TRAIN folios only; every test folio is scored on its own):
  1. Two multilingual encoders (BAAI/bge-m3 and intfloat/multilingual-e5-large, pretrained weights from the HF hub) are
     fine-tuned on the training folios with a multi-positive contrastive loss over dense and late-interaction (MaxSim)
     similarities; the other items of the same folio are the hard negatives, and passages that recur across training folios
     are linked into concepts to supply extra positives.
  2. Each test folio gets a learned clue x descriptor score matrix (average of the two models); a one-to-one assignment is
     decoded with the Hungarian algorithm on the Sinkhorn-normalised matrix.
  3. Confidence = a 3-parameter calibration map fitted with the exact challenge score on a held-out slice of TRAIN families
     (family groups are built from train text only; those folios are not used to fit the encoders).

Requirements map:
  * task-trained neural model on one A10G, full run < 90 min: DEVICE constant, fixed epochs/batch (two models, 3 epochs each).
  * the GPU-trained encoders generate every scored comparison matrix; CPU only batches, normalises and runs the decode.
  * no TF-IDF / BM25 / n-gram / dictionary / fixed-embedding matching in the predictor (TF-IDF is used only to build
    validation groups from TRAIN text); no external data; no label or id tricks; byte anchors are never read.
  * output: target_id,prediction with {"descriptor_id":..., "confidence":...}, confidence in [1/n, 1].
"""
import json
import os
import random
import sys
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(SUBMISSION_OUT.parent / "hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 42
DEVICE = "cuda"
MODEL_CONFIGS = [("BAAI/bge-m3", "cls", ""), ("intfloat/multilingual-e5-large", "mean", "query: ")]
MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 256
POOLING = "cls"
PREFIX = ""
COLBERT_W = 1.0
EXTRA_POS = True
EPOCHS = 3
FOLIOS_PER_BATCH = 4
LR = 1.5e-5
TEMP = 0.05
N_CV_FOLDS = 6
SINKHORN_ITERS = 60
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load_folios(public_dir):
    with open(public_dir / "folios.jsonl", encoding="utf-8") as f:
        return {o["folio_id"]: o for o in map(json.loads, f)}


def folio_arrays(folio, gold=None):
    clue_ids = [c["target_id"] for c in folio["spanish_clues"]]
    desc_ids = [c["descriptor_id"] for c in folio["galician_descriptors"]]
    out = {"folio_id": folio["folio_id"], "clue_ids": clue_ids, "desc_ids": desc_ids,
           "es": [c["text"] for c in folio["spanish_clues"]], "gl": [c["text"] for c in folio["galician_descriptors"]]}
    if gold is not None:
        pos = {d: i for i, d in enumerate(desc_ids)}
        out["gold"] = np.array([pos[gold[t]] for t in clue_ids])
    return out


def build_groups(items, n_folds):
    """Family-style groups for validation only: link folios whose gold-pair texts are near duplicates (train text only)."""
    texts, owner = [], []
    for k, it in enumerate(items):
        for i, g in enumerate(it["gold"]):
            texts.append(it["es"][i] + " " + it["gl"][g])
            owner.append(k)
    owner = np.array(owner)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000).fit_transform(texts)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in range(0, X.shape[0], 2000):
        sim = (X[s:s + 2000] @ X.T).toarray()
        for r in range(sim.shape[0]):
            sim[r, s + r] = 0
        rows, cols = np.where(sim > GROUP_LINK_SIM)
        for r, c in zip(rows, cols):
            a, b = find(owner[s + r]), find(owner[c])
            if a != b:
                parent[a] = b
    comp = np.array([find(k) for k in range(len(items))])
    sizes = pd.Series(comp).value_counts()
    fold_of_comp, load = {}, np.zeros(n_folds)
    for c in sizes.index:
        f = int(np.argmin(load))
        fold_of_comp[c] = f
        load[f] += sizes[c]
    return np.array([fold_of_comp[c] for c in comp]), comp


GROUP_LINK_SIM = 0.55


# ---------------------------------------------------------------- model
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = AutoModel.from_pretrained(MODEL_NAME)
        self.net.gradient_checkpointing_enable()

    def forward(self, ids, mask):
        H = self.net(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        if POOLING == "cls":
            d = H[:, 0]
        else:
            m = mask[..., None].float()
            d = (H * m).sum(1) / m.sum(1)
        return F.normalize(d, dim=-1), F.normalize(H, dim=-1)


def maxsim(Ta, Ma, Tb, Mb):
    """Late-interaction similarity between every item of a (na,La,d) and every item of b (nb,Lb,d): mean over a-tokens of the
    best b-token match, averaged with the reverse direction."""
    sim = torch.einsum("ied,jfd->ijef", Ta, Tb)
    neg = torch.finfo(sim.dtype).min
    ma = Ma.bool()[:, None, :, None]
    mb = Mb.bool()[None, :, None, :]
    s_ab = sim.masked_fill(~mb, neg).max(3).values
    s_ba = sim.masked_fill(~ma, neg).max(2).values
    fa, fb = Ma.float()[:, None, :], Mb.float()[None, :, :]
    a2b = (s_ab * fa).sum(2) / fa.sum(2)
    b2a = (s_ba * fb).sum(2) / fb.sum(2)
    return 0.5 * (a2b + b2a)


def tokenize(tok, texts):
    return tok([PREFIX + t for t in texts], truncation=True, max_length=MAX_LEN)["input_ids"]


def pad_batch(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(DEVICE), mask.to(DEVICE)


def concept_clusters(items):
    """Union-find over passages: a clue and its gold descriptor share a concept; identical passage texts in different
    folios therefore join their folios' concepts (training labels only)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for it in items:
        for i, g in enumerate(it["gold"]):
            parent[find(("es", it["es"][i]))] = find(("gl", it["gl"][g]))
    ids, tok_of, members = {}, {}, {}
    for it in items:
        for side in ("es", "gl"):
            for t, tk in zip(it[side], it[side + "_tok"]):
                key = (side, t)
                ids[key] = find(key)
                tok_of[key] = tk
                members.setdefault(find(key), {"es": {}, "gl": {}})[side][t] = tk
    names = {r: n for n, r in enumerate(sorted({v for v in ids.values()}, key=str))}
    return {k: names[v] for k, v in ids.items()}, {names[r]: m for r, m in members.items()}, names


def supcon(sim, pos):
    """Multi-positive InfoNCE over rows of sim; pos is a boolean mask of the same shape."""
    neg_inf = torch.finfo(sim.dtype).min
    lp = torch.logsumexp(sim.masked_fill(~pos, neg_inf), dim=1) - torch.logsumexp(sim, dim=1)
    ok = pos.any(1)
    return -lp[ok].mean()


def build_batch(batch, cid, members, rng):
    es, gl, ce, cg = [], [], [], []
    seen_gl = set()
    for it in batch:
        for t, tk in zip(it["es"], it["es_tok"]):
            es.append(tk)
            ce.append(cid[("es", t)])
        for t, tk in zip(it["gl"], it["gl_tok"]):
            gl.append(tk)
            cg.append(cid[("gl", t)])
            seen_gl.add(t)
    if EXTRA_POS:
        for it in batch:
            for t in it["es"]:
                mates = [(x, tk) for x, tk in members[cid[("es", t)]]["gl"].items() if x not in seen_gl]
                if mates:
                    x, tk = mates[int(rng.integers(len(mates)))]
                    gl.append(tk)
                    cg.append(cid[("gl", x)])
                    seen_gl.add(x)
    return es, gl, ce, cg


def train_model(items, tok, tag):
    """Fine-tune the encoder on the given training folios (items carry tokenised texts under 'es_tok'/'gl_tok')."""
    seed_everything()
    model = Encoder().to(DEVICE)
    cid, members, names = concept_clusters(items)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = (len(items) + FOLIOS_PER_BATCH - 1) // FOLIOS_PER_BATCH
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * EPOCHS * steps_per_epoch), EPOCHS * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(SEED)
    pad_id = tok.pad_token_id
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(items))
        tot = 0.0
        for s in range(0, len(perm), FOLIOS_PER_BATCH):
            batch = [items[k] for k in perm[s:s + FOLIOS_PER_BATCH]]
            es, gl, ce, cg = build_batch(batch, cid, members, rng)
            pos = torch.tensor(ce, device=DEVICE)[:, None] == torch.tensor(cg, device=DEVICE)[None, :]
            pe, pg = pad_batch(es, pad_id), pad_batch(gl, pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                e, Te = model(*pe)
                g, Tg = model(*pg)
                cs = maxsim(Te, pe[1], Tg, pg[1])
            sim = e @ g.T / TEMP
            cs = cs.float() / TEMP
            loss = 0.5 * (supcon(sim, pos) + supcon(sim.T, pos.T)) + 0.5 * (supcon(cs, pos) + supcon(cs.T, pos.T))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        log(f"[{tag}] epoch {ep + 1}/{EPOCHS} loss {tot / steps_per_epoch:.4f}")
    return model


def similarity_matrices(model, items, tok):
    """Per folio: (dense, late-interaction) clue x descriptor matrices from the fine-tuned encoder."""
    pad_id = tok.pad_token_id
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for it in items:
            pe, pg = pad_batch(it["es_tok"], pad_id), pad_batch(it["gl_tok"], pad_id)
            e, Te = model(*pe)
            g, Tg = model(*pg)
            out.append(((e @ g.T).cpu().numpy(), maxsim(Te, pe[1], Tg, pg[1]).float().cpu().numpy()))
    return out


def combine(pair):
    return pair[0] + COLBERT_W * pair[1]


# ---------------------------------------------------------------- decode + calibration
def sinkhorn(S, tau, iters=SINKHORN_ITERS):
    logK = (S - S.max()) / tau
    for _ in range(iters):
        logK = logK - logsumexp(logK, axis=1, keepdims=True)
        logK = logK - logsumexp(logK, axis=0, keepdims=True)
    logK = logK - logsumexp(logK, axis=1, keepdims=True)
    return np.exp(logK)


def decode(S, params):
    tau, a, b = params
    P = sinkhorn(S, tau)
    r, c = linear_sum_assignment(-np.log(P + 1e-12))
    n = S.shape[0]
    p = np.clip(P[r, c], 1e-6, 1 - 1e-6)
    conf = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
    conf = np.clip(conf, 1 / n, 1.0)
    return c, conf


def brier_skill(conf, correct, n):
    o = (1 - conf) / (n - 1)
    B = np.where(correct, (1 - conf) ** 2 + (n - 1) * o ** 2, conf ** 2 + (1 - o) ** 2 + (n - 2) * o ** 2)
    return np.maximum(0.0, 1 - B / (1 - 1 / n))


def folio_score(pred, gold, conf):
    n = len(gold)
    corr = pred == gold
    h = n // 2
    l, r = corr[:h].mean(), corr[h:].mean()
    return 0.6 * corr.mean() + 0.2 * np.sqrt(l * r) + 0.2 * brier_skill(conf, corr, n).mean()


def total_score(sims, items, bands, params):
    per = {}
    for S, it, bd in zip(sims, items, bands):
        c, conf = decode(S, params)
        per.setdefault(bd, []).append(folio_score(c, it["gold"], conf))
    return 100 * np.mean([np.mean(v) for v in per.values()])


def fit_calibration(sims, items, bands):
    best = None
    for x0 in [(0.05, 1.0, 0.0), (0.03, 1.0, 0.5), (0.1, 0.7, 0.0)]:
        res = minimize(lambda z: -total_score(sims, items, bands, (abs(z[0]) + 1e-3, z[1], z[2])), x0, method="Nelder-Mead",
                       options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    return (abs(best.x[0]) + 1e-3, best.x[1], best.x[2]), -best.fun


def validate_submission(sub, sample_path, test_folio_of):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns)
    assert sorted(sub["target_id"]) == sorted(sample["target_id"]) and sub["target_id"].is_unique
    for t, p in zip(sub["target_id"], sub["prediction"]):
        o = json.loads(p)
        assert set(o) == {"descriptor_id", "confidence"} and np.isfinite(o["confidence"])
        n = test_folio_of[t][1]
        assert 1 / n - 1e-9 <= o["confidence"] <= 1 + 1e-9 and o["descriptor_id"] in test_folio_of[t][0]


def prepare(items, tok):
    for it in items:
        it["es_tok"], it["gl_tok"] = tokenize(tok, it["es"]), tokenize(tok, it["gl"])


def main():
    global MODEL_NAME, POOLING, PREFIX
    seed_everything()
    folios = load_folios(PUBLIC_DIR)
    train_df = pd.read_csv(PUBLIC_DIR / "train.csv")
    test_df = pd.read_csv(PUBLIC_DIR / "test.csv")
    tt = pd.read_csv(PUBLIC_DIR / "train_targets.csv")
    gold = {r.target_id: json.loads(r.prediction)["descriptor_id"] for r in tt.itertuples()}
    tr = train_df.drop_duplicates("folio_id")
    items = [folio_arrays(folios[f], gold) for f in tr["folio_id"]]
    bands = list(tr["evidence_band"])
    fold, comp = build_groups(items, N_CV_FOLDS)
    fit_idx = [i for i in range(len(items)) if fold[i] != N_CV_FOLDS - 1]
    cal_idx = [i for i in range(len(items)) if fold[i] == N_CV_FOLDS - 1]
    log(f"{len(items)} train folios, {len(set(comp))} groups; fit {len(fit_idx)} folios, calibration holdout {len(cal_idx)} folios")
    te = test_df.drop_duplicates("folio_id")
    t_items = [folio_arrays(folios[f]) for f in te["folio_id"]]
    fit_items, cal_items = [items[i] for i in fit_idx], [items[i] for i in cal_idx]
    cal_bands = [bands[i] for i in cal_idx]

    cal_sims, test_sims = [], []
    for name, pool, prefix in MODEL_CONFIGS:
        MODEL_NAME, POOLING, PREFIX = name, pool, prefix
        tok = AutoTokenizer.from_pretrained(name)
        for group in (fit_items, cal_items, t_items):
            prepare(group, tok)
        model = train_model(fit_items, tok, name.split("/")[-1])
        cal_sims.append(similarity_matrices(model, cal_items, tok))
        test_sims.append(similarity_matrices(model, t_items, tok))
        acc = np.mean([np.mean(linear_sum_assignment(-combine(p))[1] == it["gold"]) for p, it in zip(cal_sims[-1], cal_items)])
        log(f"{name}: held-out-family Hungarian accuracy {acc:.3f}")
        del model
        torch.cuda.empty_cache()

    cal_ens = [np.mean([combine(m[k]) for m in cal_sims], 0) for k in range(len(cal_items))]
    acc = np.mean([np.mean(linear_sum_assignment(-S)[1] == it["gold"]) for S, it in zip(cal_ens, cal_items)])
    params, cal_score = fit_calibration(cal_ens, cal_items, cal_bands)
    log(f"ensemble held-out accuracy {acc:.3f}; calibration {np.round(params, 3)}; held-out score {cal_score:.2f}")

    rows, test_folio_of = [], {}
    for k, it in enumerate(t_items):
        S = np.mean([combine(m[k]) for m in test_sims], 0)
        c, conf = decode(S, params)
        n = len(c)
        for i, t in enumerate(it["clue_ids"]):
            rows.append((t, json.dumps({"descriptor_id": it["desc_ids"][c[i]], "confidence": float(conf[i])}, separators=(",", ":"))))
            test_folio_of[t] = (set(it["desc_ids"]), n)
    sub = pd.DataFrame(rows, columns=["target_id", "prediction"])
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv", test_folio_of)
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
