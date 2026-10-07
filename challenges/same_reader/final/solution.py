import os
import random
import sys
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SEED = 42
N_FOLDS = 5
DEVICE = "cuda"
USE_AMP = True
NUM_THREADS = 4
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FT_NAME = "distilroberta-base"
FT_SEEDS = (42, 43, 44)
FT_EPOCHS = 2
FT_BATCH = 32
FT_LR = 3e-5
FT_HEAD_LR = 1e-3
FT_MAX_LEN = 72
TFIDF_C = 0.3
EMB_C, EMB_W = 0.3, 0.5
LGB_ROUNDS = 300
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
    torch.set_num_threads(NUM_THREADS)


def validate_submission(sub, sample_path):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub["id"].astype(str).tolist() == sample["id"].astype(str).tolist(), "id mismatch/order"
    assert sub["id"].is_unique
    v = sub["p_same_reader"].to_numpy(dtype=float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), "bad probabilities"


def load_data():
    train = pd.read_csv(PUBLIC_DIR / "train.csv").merge(
        pd.read_csv(PUBLIC_DIR / "train_labels.csv"), on="id", how="left", validate="one_to_one")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    assert train["p_same_reader"].notna().all()
    return train.reset_index(drop=True), test


def make_folds(train):
    y = train["p_same_reader"].to_numpy().astype(int)
    cnt = train["annotator_count"].to_numpy().astype(int)
    norm = train["text"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    groups = pd.factorize(norm)[0]
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(train, y * 10 + cnt, groups))


def cat_frame(df):
    cnt = df["annotator_count"].astype(int).astype(str)
    a, b = df["emotion_a"], df["emotion_b"]
    return pd.DataFrame({"cnt": cnt, "a": a, "b": b, "ca": cnt + "_" + a, "cb": cnt + "_" + b, "pair": a + "|" + b})


def fit_predict_tfidf(tr_df, tr_y, eval_dfs):
    enc = OneHotEncoder(handle_unknown="ignore")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_tr = hstack([enc.fit_transform(cat_frame(tr_df)), vec.fit_transform(tr_df["text"].astype(str))]).tocsr()
    clf = LogisticRegression(C=TFIDF_C, max_iter=3000, random_state=SEED).fit(x_tr, tr_y)
    return [clf.decision_function(hstack([enc.transform(cat_frame(d)), vec.transform(d["text"].astype(str))]).tocsr())
            for d in eval_dfs]


def embed_texts(texts, tok, enc_model, batch=64):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), enc_model.config.hidden_size), dtype=np.float32)
    enc_model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            x = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            h = enc_model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def fit_predict_emb(tr_df, tr_y, emb_tr, eval_sets):
    enc = OneHotEncoder(handle_unknown="ignore")
    x_tr = hstack([enc.fit_transform(cat_frame(tr_df)), csr_matrix(EMB_W * emb_tr)]).tocsr()
    clf = LogisticRegression(C=EMB_C, max_iter=3000, random_state=SEED).fit(x_tr, tr_y)
    return [clf.decision_function(hstack([enc.transform(cat_frame(d)), csr_matrix(EMB_W * e)]).tocsr())
            for d, e in eval_sets]


def lgb_features(df, emotions):
    idx = {e: i for i, e in enumerate(emotions)}
    ia, ib = df["emotion_a"].map(idx), df["emotion_b"].map(idx)
    X = pd.DataFrame({"cnt": df["annotator_count"].to_numpy()}, index=df.index)
    X["a"] = pd.Categorical(ia, categories=range(len(emotions)))
    X["b"] = pd.Categorical(ib, categories=range(len(emotions)))
    for i, e in enumerate(emotions):
        X["has_" + e] = ((ia == i) | (ib == i)).astype(int).to_numpy()
    t = df["text"].astype(str)
    X["len"] = t.str.len().to_numpy()
    X["nw"] = t.str.split().str.len().to_numpy()
    X["q"] = t.str.count(r"\?").to_numpy()
    X["ex"] = t.str.count("!").to_numpy()
    X["name"] = t.str.count(r"\[NAME\]").to_numpy()
    X["up"] = t.apply(lambda s: sum(c.isupper() for c in s) / max(1, len(s))).to_numpy()
    return X


def fit_predict_lgb(tr_df, tr_y, eval_dfs, emotions):
    model = lgb.LGBMClassifier(
        learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5, cat_smooth=20, min_data_per_group=40, cat_l2=10,
        n_estimators=LGB_ROUNDS, random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=NUM_THREADS, verbose=-1)
    model.fit(lgb_features(tr_df, emotions), tr_y)
    out = []
    for d in eval_dfs:
        p = model.predict_proba(lgb_features(d, emotions))[:, 1]
        out.append(np.log(p / (1 - p)))
    return out


class PairClassifier(nn.Module):

    def __init__(self, n_emotions):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(FT_NAME)
        hid = self.backbone.config.hidden_size
        self.emo = nn.Embedding(n_emotions, 32)
        self.rater = nn.Embedding(3, 8)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(hid + 40, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, ids, mask, a, b, c):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        z = torch.cat([self.drop(pooled), self.emo(a) + self.emo(b), self.rater(c)], dim=1)
        return self.head(z.float()).squeeze(-1)


def encode_pairs(df, tok, emo_idx):
    prefix = (df["emotion_a"] + " and " + df["emotion_b"] + " (" + df["annotator_count"].astype(str) + " raters)").tolist()
    ids = tok(prefix, df["text"].astype(str).tolist(), truncation="only_second", max_length=FT_MAX_LEN)["input_ids"]
    return ids, df["emotion_a"].map(emo_idx).to_numpy(), df["emotion_b"].map(emo_idx).to_numpy(), df["annotator_count"].to_numpy() - 3


def make_batch(X, idx, pad_id):
    rows = [X[0][i] for i in idx]
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [pad_id] * (width - len(r)) for r in rows], device=DEVICE)
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=DEVICE)
    t = lambda arr: torch.tensor(arr[idx], device=DEVICE)
    return ids, mask, t(X[1]), t(X[2]), t(X[3])


def ft_logits(model, X, pad_id, bs=128):
    model.eval()
    order = np.argsort([len(r) for r in X[0]], kind="stable")
    out = np.zeros(len(X[0]))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            out[idx] = model(*make_batch(X, idx, pad_id)).float().cpu().numpy()
    return out


def fine_tune_fold(X_tr, y_tr, X_ev_list, tok, n_emotions, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = PairClassifier(n_emotions).to(DEVICE)
    head_params = list(model.emo.parameters()) + list(model.rater.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": FT_LR},
                             {"params": head_params, "lr": FT_HEAD_LR}], weight_decay=0.01)
    n = len(y_tr)
    steps = FT_EPOCHS * ((n + FT_BATCH - 1) // FT_BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    gen = torch.Generator()
    gen.manual_seed(seed)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(FT_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        for i in range(0, n, FT_BATCH):
            idx = perm[i:i + FT_BATCH]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                logit = model(*make_batch(X_tr, idx, tok.pad_token_id))
            loss = lossf(logit.float(), y_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    outs = [ft_logits(model, X, tok.pad_token_id) for X in X_ev_list]
    del model, opt
    torch.cuda.empty_cache()
    return outs


def blend_weights(z_oof, y, grid_step=0.1):
    k = z_oof.shape[1]
    best_w, best_ap = np.ones(k) / k, -1.0
    ticks = np.round(np.arange(0, 1 + 1e-9, grid_step), 6)
    def rec(prefix, left, slots):
        if slots == 1:
            yield prefix + [left]
            return
        for t in ticks:
            if t <= left + 1e-9:
                yield from rec(prefix + [t], round(left - t, 6), slots - 1)
    for w in rec([], 1.0, k):
        w = np.asarray(w)
        ap = average_precision_score(y, z_oof @ w)
        if ap > best_ap + 1e-12:
            best_ap, best_w = ap, w
    return best_w, best_ap


def main():
    seed_everything()
    train, test = load_data()
    sample_path = PUBLIC_DIR / "sample_submission.csv"
    y = train["p_same_reader"].to_numpy().astype(int)
    folds = make_folds(train)
    emotions = sorted(set(train["emotion_a"]) | set(train["emotion_b"]) | set(test["emotion_a"]) | set(test["emotion_b"]))
    emo_idx = {e: i for i, e in enumerate(emotions)}
    log(f"train {train.shape} test {test.shape} positive rate {y.mean():.4f}")


    tok_e = AutoTokenizer.from_pretrained(ENCODER_NAME)
    enc_model = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    emb_tr = embed_texts(train["text"].astype(str).tolist(), tok_e, enc_model)
    emb_te = embed_texts(test["text"].astype(str).tolist(), tok_e, enc_model)
    del enc_model
    torch.cuda.empty_cache()
    log("embeddings done")

    names = ["tfidf_lr", "emb_lr", "lgbm", "finetune"]
    oof = {n: np.zeros(len(train)) for n in names}
    tst = {n: np.zeros(len(test)) for n in names}

    tok_f = AutoTokenizer.from_pretrained(FT_NAME)
    X_all = encode_pairs(train, tok_f, emo_idx)
    X_test = encode_pairs(test, tok_f, emo_idx)
    sub_x = lambda X, ix: (([X[0][i] for i in ix]), X[1][ix], X[2][ix], X[3][ix])

    for f, (a, b) in enumerate(folds):
        tr_df, va_df = train.iloc[a], train.iloc[b]

        for n, (s_va, s_te) in (
                ("tfidf_lr", fit_predict_tfidf(tr_df, y[a], [va_df, test])),
                ("emb_lr", fit_predict_emb(tr_df, y[a], emb_tr[a], [(va_df, emb_tr[b]), (test, emb_te)])),
                ("lgbm", fit_predict_lgb(tr_df, y[a], [va_df, test], emotions))):
            oof[n][b] = s_va
            tst[n] += s_te / N_FOLDS

        for seed in FT_SEEDS:
            lo_va, lo_te = fine_tune_fold(sub_x(X_all, a), y[a], [sub_x(X_all, b), X_test], tok_f, len(emotions), seed)
            oof["finetune"][b] += lo_va / len(FT_SEEDS)
            tst["finetune"] += lo_te / (len(FT_SEEDS) * N_FOLDS)
        msg = " ".join(f"{n}={average_precision_score(y[b], oof[n][b]):.4f}" for n in names)
        log(f"fold {f}: {msg}")

    for n in names:
        folds_ap = [average_precision_score(y[b], oof[n][b]) for _, b in folds]
        log(f"OOF {n}: fold-mean AP {np.mean(folds_ap):.4f} sd {np.std(folds_ap):.4f} pooled {average_precision_score(y, oof[n]):.4f}")


    mu = {n: oof[n].mean() for n in names}
    sd = {n: oof[n].std() for n in names}
    z_oof = np.column_stack([(oof[n] - mu[n]) / sd[n] for n in names])
    z_tst = np.column_stack([(tst[n] - mu[n]) / sd[n] for n in names])
    w, w_ap = blend_weights(z_oof, y)
    eq_ap = average_precision_score(y, z_oof.mean(1))
    log(f"blend weights {dict(zip(names, w.round(2)))} OOF AP {w_ap:.4f} (equal weights {eq_ap:.4f})")

    w = 0.5 * w + 0.5 * np.ones(len(names)) / len(names)
    log(f"final OOF AP with shrunk weights {average_precision_score(y, z_oof @ w):.4f}")


    prob = 1.0 / (1.0 + np.exp(-(z_tst @ w)))
    sub = pd.DataFrame({"id": test["id"].values, "p_same_reader": np.clip(prob, 0.0, 1.0)})
    validate_submission(sub, sample_path)
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape} unique probs={sub['p_same_reader'].nunique()}")


if __name__ == "__main__":
    main()
