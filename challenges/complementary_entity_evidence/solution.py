# eris-template-version: 3
"""Complementary Entity Evidence Selection: anchor-conditioned role tagger + expected-union pair decode.

Reads <public_dir>, trains a transformer inside this script, writes <submission_out> (id, prediction).

Challenge requirements map (how each explicit requirement of the description is met):
  * hardware / runtime: one A10G, 90 min -> device fixed to cuda, fixed work plan (see constants), time is logged only.
  * required methods: real training of a pretrained encoder (full fine-tuning) on the supplied training slates; in-script cohort CV
    chooses the epoch count and the calibration temperature; the final model is refit on 100% of train.
  * prohibited: no web, no annotation mirrors, no external or synthetic examples, no generated sentences or pseudo-labels, no id
    reverse-mapping (ids are used only to group rows into slates and to align the output), no candidate-order signal, no test-set
    statistics (test rows are only run through the frozen trained model, one slate at a time).
  * output: id,prediction for every test id, finite floats; larger = selected earlier within its slate.

Design (why):
  * The hidden target of a candidate is its incident role inventory minus the seed's known roles. So the seed only acts as a HARD MASK
    (a known role can never be "new"); the learning problem is "which roles does this sentence assert about the anchor?".
  * The model is a sentence-level multi-label role tagger. Labels are partially observed (a role listed in known_roles is unobserved for
    that row), hence a masked binary cross-entropy. The same sentence can appear in several train slates with different seeds, so its
    observed labels are merged across appearances; train seed sentences are extra examples whose label is exactly their known_roles.
  * The metric rewards the UNION of the chosen pair relative to the best pair, so the decode picks the pair with the largest expected
    union of new roles under the calibrated role probabilities (redundant candidates are penalised automatically).
"""
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ---- fixed plan (constants; time is never used in a condition) -------------------------------------
MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_REVISION = "main"         # DEV NOTE: pin to an exact commit sha before the final submission
DEVICE = "cuda"
AMP_DTYPE = "bfloat16"          # fixed; bf16 is native on the A10G. dev_run.py may set "float32" on pre-Ampere dev GPUs (no bf16 hardware)
SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 4                  # the LR schedule is defined for MAX_EPOCHS; the final refit trains the CV-chosen epoch count
BATCH_SIZE = 16
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
LLRD = 0.9                      # layer-wise LR decay
WARMUP = 0.1
MAX_LEN = 160
FINAL_SEEDS = 2
T0 = time.time()                # LOGGING ONLY


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---- data ------------------------------------------------------------------------------------------
def slate_key(row_id):
    """ids look like ev_<slate-key>_<candidate-key>; the middle key only groups rows into slates."""
    return row_id.split("_")[1]


def cohort_key(anchor, anchor_type):
    return re.sub(r"\s+", " ", str(anchor).strip().lower()) + "|" + str(anchor_type).strip().lower()


def mark_anchor(text, start, end, anchor_type):
    """Wrap the anchor mention in typed entity markers so the encoder knows which entity's roles to tag."""
    if start is None or end is None or not (0 <= start < end <= len(text)):
        return f"[ENT-{anchor_type}] " + text
    return text[:start] + f" [ENT] {text[start:end]} [/ENT] " + text[end:]


def load_train():
    tr = pd.read_csv(PUBLIC_DIR / "train.csv")
    tg = pd.read_csv(PUBLIC_DIR / "train_targets.csv")
    tg["target"] = tg["target"].map(json.loads)
    tr = tr.merge(tg, on="id", how="left", validate="one_to_one")
    assert tr["target"].notna().all(), "every training row needs a target"
    tr["known"] = tr["known_roles"].map(json.loads)
    tr["new"] = tr["target"].map(lambda t: t["roles"])
    tr["slate"] = tr["id"].map(slate_key)
    tr["cohort"] = [cohort_key(a, t) for a, t in zip(tr["anchor"], tr["anchor_type"])]
    return tr


def build_examples(tr, roles):
    """One example per distinct sentence. y[r]=1 if observed present; m[r]=1 if the label for role r is observed.
    A candidate row says: roles in `new` are present; roles outside known U new are absent; roles in `known` are unobserved.
    Train seed sentences (query_text) are fully observed: present = known_roles, all others absent."""
    ridx = {r: i for i, r in enumerate(roles)}
    R = len(roles)
    ex = {}

    def get(text, start, end, atype, cohort):
        key = (text, start, end, atype)
        if key not in ex:
            ex[key] = dict(text=mark_anchor(text, start, end, atype), y=np.zeros(R, np.float32), m=np.zeros(R, np.float32),
                           cohort=cohort, conflict=0)
        return ex[key]

    for row in tr.itertuples(index=False):
        e = get(row.candidate_text, int(row.anchor_start), int(row.anchor_end), row.anchor_type, row.cohort)
        known = {ridx[r] for r in row.known if r in ridx}
        new = {ridx[r] for r in row.new if r in ridx}
        for i in range(R):
            if i in new:
                if e["m"][i] and e["y"][i] == 0:
                    e["conflict"] += 1
                e["y"][i], e["m"][i] = 1.0, 1.0
            elif i not in known:
                if e["m"][i] and e["y"][i] == 1:
                    e["conflict"] += 1
                elif not e["m"][i]:
                    e["y"][i], e["m"][i] = 0.0, 1.0
        # seed sentence: fully observed inventory
        a = str(row.anchor)
        pos = str(row.query_text).find(a)
        s, t = (pos, pos + len(a)) if pos >= 0 else (None, None)
        sd = get(row.query_text, s, t, row.anchor_type, row.cohort)
        sd["y"][:] = 0.0
        sd["m"][:] = 1.0
        for i in known:
            sd["y"][i] = 1.0
    out = list(ex.values())
    log(f"examples: {len(out)} distinct sentences (incl. seeds); label conflicts across appearances: {sum(e['conflict'] for e in out)}")
    return out


# ---- exact metric (re-implemented from the description) and decode ------------------------------------
def slate_score(sel, role_sets):
    ids = list(role_sets)
    opt = max(len(role_sets[a] | role_sets[b]) for a, b in combinations(ids, 2))
    return len(role_sets[sel[0]] | role_sets[sel[1]]) / opt


def decode_pair(probs, known_idx, ids):
    """Pick the pair maximising E|union of new roles| under independent role probabilities.
    probs: array (n_cand, R); known_idx: per-candidate set of role indices to hard-mask (known roles can never be new)."""
    p = probs.copy()
    for i in range(len(ids)):
        for r in known_idx[i]:
            p[i, r] = 0.0
    best, best_val = None, -1.0
    for a, b in combinations(range(len(ids)), 2):
        val = float(np.sum(1.0 - (1.0 - p[a]) * (1.0 - p[b])))
        if val > best_val + 1e-12 or (abs(val - best_val) <= 1e-12 and (ids[a], ids[b]) < (ids[best[0]], ids[best[1]])):
            best, best_val = (a, b), val
    a, b = best
    marg = p.sum(1) - 0.0
    scores = {}
    order = [i for i in sorted(range(len(ids)), key=lambda i: (-marg[i], ids[i])) if i not in (a, b)]
    for rank, i in enumerate(order):
        scores[ids[i]] = -float(rank + 1)
    first, second = sorted((a, b), key=lambda i: (-marg[i], ids[i]))
    scores[ids[first]] = 1001.0
    scores[ids[second]] = 1000.0
    return scores


# ---- model -----------------------------------------------------------------------------------------
class RoleTagger(nn.Module):
    def __init__(self, n_roles, tok):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.enc.resize_token_embeddings(len(tok))
        h = self.enc.config.hidden_size
        self.ent_id = tok.convert_tokens_to_ids("[ENT]")
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(2 * h, n_roles)

    def forward(self, input_ids, attention_mask):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hs[:, 0]
        is_ent = (input_ids == self.ent_id)
        pos = torch.where(is_ent.any(1), is_ent.float().argmax(1), torch.zeros(len(input_ids), dtype=torch.long, device=input_ids.device))
        ent = hs[torch.arange(len(hs), device=hs.device), pos]
        return self.head(self.drop(torch.cat([cls, ent], -1)))


def make_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tok.add_special_tokens({"additional_special_tokens": ["[ENT]", "[/ENT]"]})
    return tok


def batches(texts, tok, bs, shuffle, seed):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        enc = tok([texts[j] for j in sel], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield sel, enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def param_groups(model):
    groups, n_layers = [], model.enc.config.num_hidden_layers
    groups.append(dict(params=list(model.head.parameters()), lr=LR_HEAD))
    groups.append(dict(params=list(model.enc.embeddings.parameters()), lr=LR_ENCODER * LLRD ** (n_layers + 1)))
    for i, layer in enumerate(model.enc.encoder.layer):
        groups.append(dict(params=list(layer.parameters()), lr=LR_ENCODER * LLRD ** (n_layers - i)))
    rest = [p for n, p in model.enc.named_parameters() if not n.startswith(("embeddings", "encoder.layer"))]
    if rest:
        groups.append(dict(params=rest, lr=LR_ENCODER))
    return groups


@torch.no_grad()
def predict_logits(model, tok, texts):
    model.eval()
    out = np.zeros((len(texts), model.head.out_features), np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))        # length-sorted for speed; output realigned
    for sel, ids, am in batches([texts[i] for i in order], tok, 64, False, 0):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_DTYPE == "bfloat16"):
            lg = model(ids, am)
        for k, j in enumerate(sel):
            out[order[j]] = lg[k].float().cpu().numpy()
    return out


def train_model(examples, n_roles, tok, epochs, seed, eval_hook=None):
    """Fine-tune the tagger for `epochs` epochs of the MAX_EPOCHS schedule. eval_hook(epoch, model) is called after each epoch."""
    seed_everything(seed)
    model = RoleTagger(n_roles, tok).to(DEVICE)
    texts = [e["text"] for e in examples]
    Y = torch.tensor(np.stack([e["y"] for e in examples]))
    M = torch.tensor(np.stack([e["m"] for e in examples]))
    steps = MAX_EPOCHS * math.ceil(len(texts) / BATCH_SIZE)
    opt = torch.optim.AdamW(param_groups(model), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for sel, ids, am in batches(texts, tok, BATCH_SIZE, True, seed * 1000 + ep):
            y, m = Y[sel].to(DEVICE), M[sel].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_DTYPE == "bfloat16"):
                lg = model(ids, am)
            loss = (nn.functional.binary_cross_entropy_with_logits(lg.float(), y, reduction="none") * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            cnt += 1
        log(f"  seed {seed} epoch {ep + 1}/{epochs} train loss {tot / max(1, cnt):.4f}")
        if eval_hook is not None:
            eval_hook(ep + 1, model)
    return model


# ---- cohort folds, CV ----------------------------------------------------------------------------------
def make_folds(tr):
    """Greedy balanced assignment of whole cohorts to folds (a cohort never straddles train/validation)."""
    n_slates = tr.groupby("cohort")["slate"].nunique().to_dict()
    order = sorted(n_slates, key=lambda c: (-n_slates[c], hashlib.md5(c.encode()).hexdigest()))
    load = [0] * N_FOLDS
    fold_of = {}
    for c in order:
        f = min(range(N_FOLDS), key=lambda k: (load[k], k))
        fold_of[c] = f
        load[f] += n_slates[c]
    log(f"cohort folds: slates per fold {load}")
    return fold_of


def slate_probs(df, ex_index, P, ridx):
    """Yield (slate_id, ids, probs[n,R], known_idx) for the rows of df using sentence-level probabilities P."""
    for sid, g in df.groupby("slate", sort=True):
        ids = list(g["id"])
        rows = [ex_index[(r.candidate_text, int(r.anchor_start), int(r.anchor_end), r.anchor_type)] for r in g.itertuples(index=False)]
        known = [{ridx[x] for x in k if x in ridx} for k in g["known"]]
        yield sid, ids, np.stack([P[i] for i in rows]), known


def cv_scores(tr, ridx, ex_index, P_by_epoch, fold_of, gold):
    """Mean Marginal Role Coverage@2 per epoch over all train slates, from out-of-fold probabilities."""
    res = {}
    for ep, P in P_by_epoch.items():
        scores = []
        for sid, ids, probs, known in slate_probs(tr, ex_index, P, ridx):
            sc = decode_pair(probs, known, ids)
            sel = sorted(ids, key=lambda i: (-sc[i], i))[:2]
            scores.append(slate_score(sel, {i: gold[i] for i in ids}))
        res[ep] = float(np.mean(scores))
    return res


def prepare(tr):
    """Role vocabulary (TRAIN only), distinct-sentence examples and the candidate-row -> example index."""
    roles = sorted({r for k in tr["known"] for r in k} | {r for n in tr["new"] for r in n})
    ridx = {r: i for i, r in enumerate(roles)}
    examples = build_examples(tr, roles)
    text_to_idx = {e["text"]: i for i, e in enumerate(examples)}
    ex_index = {}
    for r in tr.itertuples(index=False):
        k = (r.candidate_text, int(r.anchor_start), int(r.anchor_end), r.anchor_type)
        ex_index[k] = text_to_idx[mark_anchor(*k)]
    return roles, ridx, examples, ex_index


def fit_temperature(L, Y, M):
    """One scalar temperature fitted on out-of-fold logits by masked BCE (train-only evidence, 1 free constant)."""
    best_T, best_nll = 1.0, 1e18
    for T in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5):
        p = 1.0 / (1.0 + np.exp(-L / T))
        nll = float(-(M * (Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9))).sum() / M.sum())
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def cross_validate(tr, roles, ridx, examples, ex_index, tok):
    """Cohort-grouped K-fold. Returns OOF logits per epoch, the exact metric per epoch, the temperature per epoch."""
    fold_of = make_folds(tr)
    L_by_epoch = {ep: np.zeros((len(examples), len(roles)), np.float32) for ep in range(1, MAX_EPOCHS + 1)}
    covered = np.zeros(len(examples), bool)
    for f in range(N_FOLDS):
        tr_ex = [e for e in examples if fold_of[e["cohort"]] != f]
        va_idx = [i for i, e in enumerate(examples) if fold_of[e["cohort"]] == f]
        va_texts = [examples[i]["text"] for i in va_idx]
        log(f"fold {f}: train sentences {len(tr_ex)}, validation sentences {len(va_idx)}")

        def hook(ep, model, va_idx=va_idx, va_texts=va_texts):
            L_by_epoch[ep][va_idx] = predict_logits(model, tok, va_texts)

        train_model(tr_ex, len(roles), tok, MAX_EPOCHS, SEED + f, eval_hook=hook)
        covered[va_idx] = True
        torch.cuda.empty_cache()
    assert covered.all()
    Yall = np.stack([e["y"] for e in examples])
    Mall = np.stack([e["m"] for e in examples])
    gold = {r.id: set(r.new) for r in tr.itertuples(index=False)}
    temps = {ep: fit_temperature(L, Yall, Mall) for ep, L in L_by_epoch.items()}
    probs = {ep: 1.0 / (1.0 + np.exp(-L / temps[ep])) for ep, L in L_by_epoch.items()}
    cv = cv_scores(tr, ridx, ex_index, probs, fold_of, gold)
    for ep in sorted(cv):
        log(f"CV epoch {ep}: Marginal Role Coverage@2 = {cv[ep]:.4f} (temperature {temps[ep]})")
    return L_by_epoch, cv, temps


def main():
    seed_everything(SEED)
    tr = load_train()
    te = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    te["known"] = te["known_roles"].map(json.loads)
    te["slate"] = te["id"].map(slate_key)
    log(f"train {tr.shape} test {te.shape}")
    tok = make_tok()
    roles, ridx, examples, ex_index = prepare(tr)
    log(f"{len(roles)} roles")

    # 1) cohort CV: out-of-fold logits per epoch -> exact metric -> epoch count and temperature
    _, cv, temps = cross_validate(tr, roles, ridx, examples, ex_index, tok)
    best_ep = max(cv, key=lambda e: (cv[e], -e))
    T_final = temps[best_ep]
    log(f"chosen epochs for the final refit: {best_ep}; temperature {T_final}; OOF score {cv[best_ep]:.4f}")

    # 2) final refit on 100% of train (FINAL_SEEDS seeds); each test slate is decoded on its own
    te_keys = {}
    for r in te.itertuples(index=False):
        te_keys.setdefault((r.candidate_text, int(r.anchor_start), int(r.anchor_end), r.anchor_type), None)
    te_texts = [mark_anchor(*k) for k in te_keys]
    te_index = {k: i for i, k in enumerate(te_keys)}
    logits = np.zeros((len(te_texts), len(roles)), np.float32)
    for s in range(FINAL_SEEDS):
        model = train_model(examples, len(roles), tok, best_ep, SEED + 100 + s)
        logits += predict_logits(model, tok, te_texts) / FINAL_SEEDS
        del model
        torch.cuda.empty_cache()
    probs = 1.0 / (1.0 + np.exp(-logits / T_final))
    pred = {}
    for sid, g in te.groupby("slate", sort=True):
        ids = list(g["id"])
        rows = [te_index[(r.candidate_text, int(r.anchor_start), int(r.anchor_end), r.anchor_type)] for r in g.itertuples(index=False)]
        known = [{ridx[x] for x in k if x in ridx} for k in g["known"]]
        pred.update(decode_pair(np.stack([probs[i] for i in rows]), known, ids))

    sub = pd.DataFrame({"id": sample["id"], "prediction": [pred[i] for i in sample["id"]]})
    assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample) == len(te)
    assert sub["id"].is_unique and np.isfinite(sub["prediction"].to_numpy(float)).all()
    tmp = SUBMISSION_OUT.with_suffix(".tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, SUBMISSION_OUT)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
