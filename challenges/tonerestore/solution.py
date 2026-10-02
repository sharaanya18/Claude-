"""Tone-marker restoration (Chao pitch contours anchored in a tone-free IPA string).

Usage: python3 solution.py <public_dir> <submission_out>

Requirements map (challenge text -> what this file does)
- Real training: a neural tagger (char-level BiGRU text encoder + cross-attention over an audio encoder) is trained
  from the labelled rows in this script; every marker/contour prediction comes from that trained model.
- Audio evidence: (a) per-clip F0/voicing/energy tracks from a YIN pitch tracker, normalised inside each clip only;
  (b) frozen layers of a general-purpose pretrained acoustic model (WavLM-base+, Hugging Face weights, allowed).
- Scaffold: marker offsets count NFD code points of `phones` (never normalised); the model labels every code point
  with {no marker, one of 9 contour strings}; the reference grammar is enforced in validate_submission().
- No test-set statistics: vocabulary is built from train phones only; each test row is predicted from its own audio and
  string; nothing is fitted, normalised or clustered across test rows. Test only enters through model.forward().
- No lexical retrieval: test forms are never looked up in train; IDs/paths are not used as features.
- Validation: 5-fold GroupKFold where all repetitions of one tone-free form share a fold, exact official metric on OOF.
- Determinism: fixed seeds, folds, epochs, batch size, thread count; no wall-clock or hardware branches (time is logged only).
"""
import os, sys, json, random, time
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
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchaudio
from sklearn.model_selection import GroupKFold
from transformers import AutoModel

# ----------------------------------------------------------------------------- fixed work plan
DEVICE = "cuda"            # local CPU smoke tests flip this constant (and are set back before submitting)
SEED = 42
N_FOLDS = 5
SEEDS_PER_FOLD = 2         # fold x seed models are averaged for the test prediction
EPOCHS = 25
BATCH = 32
LR = 2e-3
DROPOUT = 0.3
TIME_MASK = 0.1            # fraction of audio frames masked in training (SpecAugment-style)
W_NONE = 0.5               # CE weight of the "no marker" class
MODE = "both"              # "text" | "f0" | "ssl" | "both"  (experiment switch; final plan uses "both")
SSL_NAME = "microsoft/wavlm-base-plus"
SSL_LAYERS = [3, 6, 9, 12]
MAX_ROWS = 0               # 0 = all rows (dev smoke tests only)
NUM_THREADS = 4

CL = ["none", "31", "3131", "3151", "3153", "51", "53", "5331", "5351", "5353"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)  # elapsed time is telemetry only


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(NUM_THREADS)


# ----------------------------------------------------------------------------- metric (official, from the task text)
def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def row_score(P, T):
    if not P:
        return 0.0
    A = 2 * len(set(P) & set(T)) / (len(P) + len(T))
    cp, ct = [c for _, c in P], [c for _, c in T]
    C = 1 - _lev(cp, ct) / max(len(cp), len(ct))
    return (6 * A + 3 * C + float(P == T)) / 10


def score(preds, refs):
    return float(np.mean([row_score(p, t) for p, t in zip(preds, refs)]))


def parse(s):
    return [(int(o), c) for o, c in json.loads(s)]


# ----------------------------------------------------------------------------- acoustic front-end (per clip only)
def yin_track(x, sr, fmin=70.0, fmax=450.0, hop_ms=10, thresh=0.15):
    """Batched-over-frames YIN: returns per-frame log2 f0 (nan if unvoiced), voicing strength, log energy."""
    hop = int(sr * hop_ms / 1000)
    tau_min, tau_max = int(sr / fmax), int(sr / fmin)
    W = 2 * tau_max
    xt = torch.nn.functional.pad(torch.from_numpy(x).double(), (W // 2, W // 2 + W))
    fr = xt.unfold(0, W, hop)[: len(x) // hop + 1]
    J = W - tau_max
    nfft = 1 << int(np.ceil(np.log2(W + J)))
    r = torch.fft.irfft(torch.fft.rfft(fr, nfft) * torch.conj(torch.fft.rfft(fr[:, :J], nfft)), nfft)[:, : tau_max + 1]
    cs = torch.cat([torch.zeros(len(fr), 1, dtype=fr.dtype), torch.cumsum(fr ** 2, 1)], 1)
    taus = torch.arange(tau_max + 1)
    d = (cs[:, J: J + 1] - cs[:, 0:1]) + (cs[:, J + taus] - cs[:, taus]) - 2 * r
    d[:, 0] = 0
    cm = torch.cumsum(d, 1)
    c = (d[:, 1:] * torch.arange(1, tau_max + 1) / (cm[:, 1:] + 1e-12))[:, tau_min - 1:]
    below = c < thresh
    idx = torch.where(below.any(1), below.double().argmax(1), c.argmin(1))
    for _ in range(40):  # slide down to the local minimum after the first dip below threshold
        nxt = torch.clamp(idx + 1, max=c.shape[1] - 1)
        mv = (c.gather(1, nxt[:, None])[:, 0] < c.gather(1, idx[:, None])[:, 0]) & below.any(1)
        idx = torch.where(mv, nxt, idx)
    i0 = idx.clamp(1, c.shape[1] - 2)
    y0, y1, y2 = [c.gather(1, (i0 + k)[:, None])[:, 0] for k in (-1, 0, 1)]
    den = y0 - 2 * y1 + y2
    shift = torch.where(den.abs() > 1e-9, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-1, 1)
    f0 = sr / ((i0 + tau_min).double() + shift)
    vo = (1 - y1).clamp(0, 1)
    en = torch.log(fr[:, tau_max - hop: tau_max + hop].pow(2).mean(1) + 1e-8)
    lf = torch.where(vo > 0.7, torch.log2(f0), torch.full_like(f0, float("nan")))
    return lf.float().numpy(), vo.float().numpy(), en.float().numpy()


def f0_features(x, sr):
    """(T, 6) frame features, all normalised inside the clip: rel. log-f0 (semitones/6), voiced flag, voicing, delta, energy z."""
    lf, vo, en = yin_track(x, sr)
    v = np.isfinite(lf)
    if v.sum() >= 3:
        s = lf.copy()
        for t in range(len(lf)):  # 5-frame median smoothing of voiced frames (kills octave blips)
            w = lf[max(0, t - 2): t + 3]
            w = w[np.isfinite(w)]
            s[t] = np.median(w) if (v[t] and len(w)) else np.nan
        lf, v = s, np.isfinite(s)
        rel = np.where(v, (lf - np.median(lf[v])) * 12.0, 0.0)
    else:
        rel = np.zeros_like(lf)
    d = np.zeros_like(rel)
    d[1:] = np.where(v[1:] & v[:-1], rel[1:] - rel[:-1], 0.0)
    ez = (en - en.mean()) / (en.std() + 1e-6)
    return np.stack([rel / 6.0, v.astype(np.float32), vo, d, ez, ez * v], 1).astype(np.float32)


def extract_audio_features(df, ssl_model):
    out = {}
    for k, (rid, p) in enumerate(zip(df.id, df.audio_path)):
        x, sr = sf.read(PUBLIC_DIR / p, dtype="float32")
        f0 = f0_features(x, sr)
        S = None
        if ssl_model is not None:
            xt = torch.from_numpy(x)
            xt = torchaudio.functional.resample(xt, sr, 16000) if sr != 16000 else xt
            xt = ((xt - xt.mean()) / (xt.std() + 1e-7))[None].to(DEVICE)
            with torch.no_grad():
                hs = ssl_model(xt, output_hidden_states=True).hidden_states
            S = torch.stack([hs[l][0] for l in SSL_LAYERS], 0).half().cpu().numpy()  # (L, T, 768)
        T = S.shape[1] if S is not None else (len(f0) + 1) // 2
        f0p = Fn.adaptive_avg_pool1d(torch.from_numpy(f0).T[None], T)[0].T.numpy()  # onto the 20 ms SSL grid
        out[rid] = dict(f0=f0p, ssl=S)
        if k % 500 == 0:
            log(f"audio features {k}/{len(df)}")
    return out


# ----------------------------------------------------------------------------- data / model
def build_items(df, feats, vocab, labelled):
    items = []
    for _, r in df.iterrows():
        p = r.phones
        L = len(p)
        y = np.zeros(L, dtype=np.int64)
        if labelled:
            for o, c in parse(r.tones):
                y[o - 1] = CL.index(c)
        f = feats[r.id]
        cf = np.array([[(i + 1) / L, min(L - i - 1, 6) / 6, float(i == L - 1)] for i in range(L)], dtype=np.float32)
        items.append(dict(ch=np.array([vocab.get(c, 1) for c in p]), cf=cf, f0=f["f0"], ssl=f["ssl"], y=y))
    return items


def collate(items, train=False):
    B = len(items)
    N = max(len(i["ch"]) for i in items)
    T = max(i["f0"].shape[0] for i in items)
    has_ssl = items[0]["ssl"] is not None
    ch = torch.zeros(B, N, dtype=torch.long); cf = torch.zeros(B, N, 3)
    f0 = torch.zeros(B, T, items[0]["f0"].shape[1]); am = torch.zeros(B, T, dtype=torch.bool)
    cm = torch.zeros(B, N, dtype=torch.bool); y = torch.full((B, N), -100, dtype=torch.long)
    ssl = torch.zeros(B, len(SSL_LAYERS), T, 768, dtype=torch.float16) if has_ssl else torch.zeros(B, 1, 1, 768, dtype=torch.float16)
    for b, it in enumerate(items):
        n, t = len(it["ch"]), it["f0"].shape[0]
        ch[b, :n] = torch.from_numpy(it["ch"]); cf[b, :n] = torch.from_numpy(it["cf"])
        f0[b, :t] = torch.from_numpy(it["f0"]); am[b, :t] = True; cm[b, :n] = True; y[b, :n] = torch.from_numpy(it["y"])
        if has_ssl:
            ssl[b, :, :t] = torch.from_numpy(it["ssl"])
    if train and TIME_MASK > 0:
        for b in range(B):
            t = int(am[b].sum()); w = int(t * TIME_MASK)
            if w > 0:
                s = np.random.randint(0, max(1, t - w))
                f0[b, s: s + w] = 0
                if has_ssl:
                    ssl[b, :, s: s + w] = 0
    return [v.to(DEVICE) for v in (ch, cf, ssl, f0, am, cm, y)]


class Net(nn.Module):
    def __init__(self, nv, d=128, drop=DROPOUT):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(nv, 64, padding_idx=0)
        self.tin = nn.Linear(64 + 3, d)
        self.tenc = nn.GRU(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.lw = nn.Parameter(torch.zeros(len(SSL_LAYERS)))  # learned softmax mix of SSL layers
        self.ssl_ln = nn.LayerNorm(768); self.ssl_proj = nn.Linear(768, d)
        self.f0_proj = nn.Linear(6, d)
        self.aenc = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.sig = nn.Parameter(torch.tensor(0.0))  # width of the monotone char-time alignment prior
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, len(CL)))

    def forward(self, ch, cf, ssl, f0, am, cm):
        t, _ = self.tenc(self.drop(self.tin(torch.cat([self.emb(ch), cf], -1))))
        if MODE == "text":
            return self.head(torch.cat([t, torch.zeros_like(t)], -1))
        a = 0
        if MODE in ("both", "ssl"):
            w = torch.softmax(self.lw, 0)
            a = a + self.ssl_proj(self.ssl_ln((ssl.float() * w[None, :, None, None]).sum(1)))
        if MODE in ("both", "f0"):
            a = a + self.f0_proj(f0)
        a, _ = self.aenc(self.drop(a))
        N, T = ch.shape[1], a.shape[1]
        att = torch.einsum("bnd,btd->bnt", self.q(t), self.k(a)) / self.d ** 0.5
        cpos = (torch.arange(N, device=ch.device)[None].float() + 1) / cm.sum(1, keepdim=True).float()
        tpos = (torch.arange(T, device=ch.device)[None].float() + 0.5) / am.sum(1, keepdim=True).float()
        s2 = 0.08 + 0.3 * torch.sigmoid(self.sig)
        att = att - ((cpos[:, :, None] - tpos[:, None, :]) ** 2) / (2 * s2 ** 2)
        att = att.masked_fill(~am[:, None, :], -1e9)
        ctx = torch.einsum("bnt,btd->bnd", torch.softmax(att, -1), self.v(a))
        return self.head(torch.cat([t, ctx], -1))


def train_model(items, nv, seed):
    seed_everything(seed)
    model = Net(nv).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(items) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    w = torch.ones(len(CL), device=DEVICE); w[0] = W_NONE
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(items))
        for s in range(0, len(perm), BATCH):
            ch, cf, ssl, f0, am, cm, y = collate([items[i] for i in perm[s: s + BATCH]], train=True)
            lg = model(ch, cf, ssl, f0, am, cm)
            loss = Fn.cross_entropy(lg.reshape(-1, len(CL)), y.reshape(-1), weight=w, ignore_index=-100, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model


def predict(model, items, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(items), bs):
            ch, cf, ssl, f0, am, cm, y = collate(items[s: s + bs])
            pr = torch.softmax(model(ch, cf, ssl, f0, am, cm), -1).cpu().numpy()
            out += [pr[b, : len(it["ch"])] for b, it in enumerate(items[s: s + bs])]
    return out


def decode(prob, thr=0.5):
    """Marker at every code point whose P(no marker) < thr, with the arg-max contour among the 9 contour classes."""
    return [(i + 1, CL[1 + int(np.argmax(prob[i, 1:]))]) for i in range(len(prob)) if prob[i, 0] < thr]


def validate_submission(sub, sample_path, phones_by_id):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns) == ["id", "tones"], "columns"
    assert len(sub) == len(sample) and sub["id"].tolist() == sample["id"].tolist(), "ids / order"
    assert sub["id"].is_unique
    for rid, cell in zip(sub["id"], sub["tones"]):
        m = json.loads(cell)
        assert 1 <= len(m) <= 32 and len(cell) <= 4096, f"marker count {rid}"
        offs = [o for o, _ in m]
        assert all(type(o) is int for o in offs) and offs == sorted(set(offs)), f"offsets {rid}"
        assert all(1 <= o <= len(phones_by_id[rid]) for o in offs), f"offset range {rid}"
        assert all(isinstance(c, str) and 1 <= len(c) <= 8 and set(c) <= set("12345") for _, c in m), f"contour {rid}"


def main():
    seed_everything()
    train = pd.read_csv(PUBLIC_DIR / "train.csv", keep_default_na=False)
    test = pd.read_csv(PUBLIC_DIR / "test.csv", keep_default_na=False)
    if MAX_ROWS:
        train, test = train.iloc[:MAX_ROWS].reset_index(drop=True), test.iloc[:MAX_ROWS].reset_index(drop=True)
    log(f"train {train.shape} test {test.shape} mode={MODE}")

    ssl_model = None
    if MODE in ("both", "ssl"):
        ssl_model = AutoModel.from_pretrained(SSL_NAME).eval().to(DEVICE)  # frozen general-purpose acoustic encoder
    feats = extract_audio_features(pd.concat([train[["id", "audio_path"]], test[["id", "audio_path"]]], ignore_index=True), ssl_model)
    del ssl_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    vocab = {c: i + 2 for i, c in enumerate(sorted(set("".join(train.phones))))}  # train-only vocabulary; 0=pad, 1=unk
    tr_items, te_items = build_items(train, feats, vocab, True), build_items(test, feats, vocab, False)
    refs = [parse(s) for s in train.tones]
    groups = train.phones.factorize()[0]  # repetitions of one tone-free form stay together

    oof = [None] * len(train)
    test_prob = [np.zeros((len(it["ch"]), len(CL))) for it in te_items]
    n_models = 0
    for f, (a, b) in enumerate(GroupKFold(N_FOLDS).split(tr_items, groups=groups)):
        oof_f = [np.zeros((len(tr_items[i]["ch"]), len(CL))) for i in b]
        for s in range(SEEDS_PER_FOLD):
            model = train_model([tr_items[i] for i in a], len(vocab) + 2, SEED + 100 * f + s)
            for j, p in enumerate(predict(model, [tr_items[i] for i in b])):
                oof_f[j] += p / SEEDS_PER_FOLD
            for j, p in enumerate(predict(model, te_items)):
                test_prob[j] += p
            n_models += 1
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        for i, p in zip(b, oof_f):
            oof[i] = p
        log(f"fold {f}: CV {score([decode(oof[i]) for i in b], [refs[i] for i in b]):.4f}")
    test_prob = [p / n_models for p in test_prob]

    # decision threshold tuned on OOF only (one scalar, same grid for every run)
    grid = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    cv = {t: score([decode(p, t) for p in oof], refs) for t in grid}
    thr = max(cv, key=cv.get)
    log("OOF score by threshold: " + " ".join(f"{t}:{v:.4f}" for t, v in cv.items()) + f" -> thr {thr}")
    log(f"FINAL OOF CV (exact metric) = {cv[thr]:.4f}")

    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    pred_by_id = {}
    for rid, p in zip(test.id, test_prob):
        m = decode(p, thr)
        if not m:  # a row with no marker above threshold: keep the single most likely marker (grammar needs >=1)
            i = int(np.argmin(p[:, 0]))
            m = [(i + 1, CL[1 + int(np.argmax(p[i, 1:]))])]
        pred_by_id[rid] = m
    sub = pd.DataFrame({"id": sample["id"], "tones": [json.dumps([[o, c] for o, c in pred_by_id[r]], separators=(",", ":")) for r in sample["id"]]})
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv", dict(zip(test.id, test.phones)))
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
