"""Spanish clue -> Galician descriptor matching inside folios.

Pipeline (all fitting on TRAIN folios only; every test folio is scored on its own):
  1. A multilingual bi-encoder (BAAI/bge-m3 weights from the HF hub) is fine-tuned on the training folios with an
     in-batch contrastive loss in which the other items of the same folio are the hard negatives.
  2. Each folio gets a learned clue x descriptor similarity matrix; a one-to-one assignment is decoded with the
     Hungarian algorithm on the (Sinkhorn-normalised) matrix.
  3. Confidence = a small calibration map (3 parameters) fitted on out-of-fold similarity matrices of family-grouped
     folds, using the exact challenge score.

Requirements map:
  * task-trained neural model on one A10G, full run < 90 min: DEVICE constant, fixed epochs/folds/batch (profiled).
  * the GPU-trained encoder generates every scored comparison matrix; CPU only batches, normalises and runs the
    one-to-one decode.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 42
DEVICE = "cuda"
MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 256
POOLING = "cls"
PREFIX = ""
COLBERT_W = 1.0
EPOCHS = 3
FOLIOS_PER_BATCH = 4
LR = 1.5e-5
TEMP = 0.05
N_CV_FOLDS = 3
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


def train_model(items, tok, tag):
    """Fine-tune the encoder on the given training folios (items carry tokenised texts under 'es_tok'/'gl_tok')."""
    seed_everything()
    model = Encoder().to(DEVICE)
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
            es, gl, tgt, off = [], [], [], 0
            for it in batch:
                n = len(it["es_tok"])
                es += it["es_tok"]
                gl += it["gl_tok"]
                tgt += list(off + it["gold"])
                off += n
            tgt = torch.tensor(tgt, device=DEVICE)
            pe, pg = pad_batch(es, pad_id), pad_batch(gl, pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                e, Te = model(*pe)
                g, Tg = model(*pg)
                cs = maxsim(Te, pe[1], Tg, pg[1])
            inv = torch.empty_like(tgt)
            inv[tgt] = torch.arange(len(tgt), device=DEVICE)
            sim = e @ g.T / TEMP
            cs = cs.float() / TEMP
            loss = 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, inv)) + 0.5 * (F.cross_entropy(cs, tgt) + F.cross_entropy(cs.T, inv))
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
    K = np.exp((S - S.max()) / tau)
    for _ in range(iters):
        K = K / K.sum(1, keepdims=True)
        K = K / K.sum(0, keepdims=True)
    return K / K.sum(1, keepdims=True)


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


def main():
    seed_everything()
    folios = load_folios(PUBLIC_DIR)
    train_df = pd.read_csv(PUBLIC_DIR / "train.csv")
    test_df = pd.read_csv(PUBLIC_DIR / "test.csv")
    tt = pd.read_csv(PUBLIC_DIR / "train_targets.csv")
    gold = {r.target_id: json.loads(r.prediction)["descriptor_id"] for r in tt.itertuples()}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    tr = train_df.drop_duplicates("folio_id")
    items = [folio_arrays(folios[f], gold) for f in tr["folio_id"]]
    bands = list(tr["evidence_band"])
    for it in items:
        it["es_tok"], it["gl_tok"] = tokenize(tok, it["es"]), tokenize(tok, it["gl"])
    fold, comp = build_groups(items, N_CV_FOLDS)
    log(f"{len(items)} train folios, {len(set(comp))} groups, fold sizes {np.bincount(fold)}")

    oof = [None] * len(items)
    for k in range(N_CV_FOLDS):
        tr_idx = [i for i in range(len(items)) if fold[i] != k]
        va_idx = [i for i in range(len(items)) if fold[i] == k]
        model = train_model([items[i] for i in tr_idx], tok, f"cv{k}")
        for i, S in zip(va_idx, similarity_matrices(model, [items[i] for i in va_idx], tok)):
            oof[i] = combine(S)
        acc = np.mean([np.mean(linear_sum_assignment(-oof[i])[1] == items[i]["gold"]) for i in va_idx])
        log(f"fold {k}: hungarian accuracy {acc:.3f}")
        del model
        torch.cuda.empty_cache()
    params, cv_score = fit_calibration(oof, items, bands)
    log(f"calibration {np.round(params, 3)}; OOF score (in-sample calibration) {cv_score:.2f}")

    model = train_model(items, tok, "final")
    te = test_df.drop_duplicates("folio_id")
    t_items = [folio_arrays(folios[f]) for f in te["folio_id"]]
    for it in t_items:
        it["es_tok"], it["gl_tok"] = tokenize(tok, it["es"]), tokenize(tok, it["gl"])
    rows, test_folio_of = [], {}
    for it, S in zip(t_items, similarity_matrices(model, t_items, tok)):
        c, conf = decode(combine(S), params)
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
