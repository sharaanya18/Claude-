"""Dataset audit for Missing Paragraph Ranking. Train-only statistics where it matters;
test text is summarised only for shape/noise diagnostics that the description already states."""
import sys, re, collections, random, math
from pathlib import Path
import numpy as np, pandas as pd

P = Path(sys.argv[1])
tr = pd.read_csv(P/"train.csv", keep_default_na=False)
te = pd.read_csv(P/"test.csv", keep_default_na=False)
cand = pd.read_csv(P/"candidates.csv", keep_default_na=False)
lab = pd.read_csv(P/"train_labels.csv", keep_default_na=False)
ss = pd.read_csv(P/"sample_submission.csv", keep_default_na=False)

print("== SHAPES ==")
for n, d in [("train", tr), ("test", te), ("candidates", cand), ("labels", lab), ("sample_sub", ss)]:
    print(f"{n:12s} {d.shape}  cols={list(d.columns)}")

print("\n== ID CONSISTENCY ==")
print("train query ids unique:", tr.query_id.is_unique, " test:", te.query_id.is_unique)
print("candidate ids unique:", cand.candidate_id.is_unique)
print("labels cover all train queries:", set(lab.query_id) == set(tr.query_id))
print("sample_sub ids == test ids (as set):", set(ss.query_id) == set(te.query_id))
print("sample_sub ids == test ids (same order):", list(ss.query_id) == list(te.query_id))

tr_pools = {q: c.split() for q, c in zip(tr.query_id, tr.candidates)}
te_pools = {q: c.split() for q, c in zip(te.query_id, te.candidates)}
ss_rank = {q: c.split() for q, c in zip(ss.query_id, ss.ranking)}
print("all train pools size 20:", all(len(v) == 20 for v in tr_pools.values()))
print("all test pools size 20:", all(len(v) == 20 for v in te_pools.values()))
print("train pools have dup ids:", any(len(set(v)) != 20 for v in tr_pools.values()))
print("test pools have dup ids:", any(len(set(v)) != 20 for v in te_pools.values()))
print("sample_sub ranking == test pool as set:", all(set(ss_rank[q]) == set(te_pools[q]) for q in te_pools))
print("sample_sub ranking == test pool same order:", all(ss_rank[q] == te_pools[q] for q in te_pools))

tr_cids = set().union(*tr_pools.values())
te_cids = set().union(*te_pools.values())
all_cids = set(cand.candidate_id)
print(f"\ndistinct candidates in train pools : {len(tr_cids)}")
print(f"distinct candidates in test pools  : {len(te_cids)}")
print(f"overlap train/test candidate sets  : {len(tr_cids & te_cids)}")
print(f"candidates.csv rows                : {len(all_cids)}")
print(f"candidates.csv == train|test union : {all_cids == (tr_cids | te_cids)}")
print("all labels are inside their own pool:",
      all(t in tr_pools[q] for q, t in zip(lab.query_id, lab.target_id)))
print("labels distinct (each abstract once):", lab.target_id.is_unique)
print("train targets == train candidate set:", set(lab.target_id) == tr_cids)

print("\n== CANDIDATE FREQUENCY (structure check only; banned as a feature) ==")
ftr = collections.Counter(c for v in tr_pools.values() for c in v)
fte = collections.Counter(c for v in te_pools.values() for c in v)
print("train appearances per candidate: min/max/mean =",
      min(ftr.values()), max(ftr.values()), round(np.mean(list(ftr.values())), 3))
print("test  appearances per candidate: min/max/mean =",
      min(fte.values()), max(fte.values()), round(np.mean(list(fte.values())), 3))
tgt = set(lab.target_id)
import numpy as _np
f_is_tgt = [ftr[c] for c in tr_cids if c in tgt]
print("train: is-a-target candidates:", len(f_is_tgt), "non-target:", len(tr_cids) - len(f_is_tgt))
# how many test candidates are never a correct answer? (we cannot know, but count pool slots)
print("test pool slots:", sum(len(v) for v in te_pools.values()), " -> implies ~",
      sum(len(v) for v in te_pools.values()) / np.mean(list(fte.values())), "distinct")

print("\n== TEXT BASICS ==")
cmap = dict(zip(cand.candidate_id, cand.snippet_b))
def stats(name, texts):
    L = np.array([len(t) for t in texts])
    print(f"{name:22s} n={len(texts):5d} len min/med/max = {L.min()}/{int(np.median(L))}/{L.max()}"
          f"  mean={L.mean():.2f}  empty={(L==0).sum()}  dup_texts={len(texts)-len(set(texts))}")
stats("train snippet_a", list(tr.snippet_a))
stats("test snippet_a", list(te.snippet_a))
stats("train candidates b", [cmap[c] for c in sorted(tr_cids)])
stats("test candidates b", [cmap[c] for c in sorted(te_cids)])

allchars = collections.Counter()
for t in list(tr.snippet_a) + [cmap[c] for c in sorted(tr_cids)]:
    allchars.update(t)
print("\ndistinct chars in TRAIN text:", len(allchars))
print("charset:", "".join(sorted(allchars)))
te_chars = collections.Counter()
for t in list(te.snippet_a) + [cmap[c] for c in sorted(te_cids)]:
    te_chars.update(t)
print("chars present in TEST but not TRAIN:", sorted(set(te_chars) - set(allchars)))
print("top 25 chars (train):", allchars.most_common(25))

def classprof(counter):
    tot = sum(counter.values())
    low = sum(v for k, v in counter.items() if k.islower())
    up = sum(v for k, v in counter.items() if k.isupper())
    dig = sum(v for k, v in counter.items() if k.isdigit())
    sp = counter.get(" ", 0)
    return dict(total=tot, lower=low/tot, upper=up/tot, digit=dig/tot, space=sp/tot,
                other=(tot-low-up-dig-sp)/tot)
print("train char classes:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in classprof(allchars).items()})
print("test  char classes:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in classprof(te_chars).items()})

print("\n== LETTER DISTRIBUTION (corruption fingerprint) ==")
def letterfreq(counter):
    tot = sum(v for k, v in counter.items() if k.isalpha())
    return np.array([counter.get(chr(97+i), 0)/tot for i in range(26)])
ltr, lte = letterfreq(allchars), letterfreq(te_chars)
uni = np.ones(26)/26
print("train letter dist  :", np.round(ltr, 4))
print("test  letter dist  :", np.round(lte, 4))
# model: observed = (1-p)*clean + p*uniform  => estimate p by regressing toward uniform
def est_p(obs, clean):
    # least squares for p in obs = (1-p) clean + p uni
    d = uni - clean
    return float(np.dot(obs - clean, d) / np.dot(d, d))
print("TV distance train vs uniform:", round(0.5*np.abs(ltr-uni).sum(), 4))
print("TV distance test  vs uniform:", round(0.5*np.abs(lte-uni).sum(), 4))
print("ratio test/train TV (=(1-p_te)/(1-p_tr)):", round((0.5*np.abs(lte-uni).sum())/(0.5*np.abs(ltr-uni).sum()), 4))

print("\n== WORD-LEVEL ==")
def toks(t): return t.split()
tr_tokens = collections.Counter(w for t in list(tr.snippet_a) for w in toks(t))
print("train snippet_a tokens:", sum(tr_tokens.values()), "distinct:", len(tr_tokens))
print("most common tokens:", tr_tokens.most_common(30))
wl = np.array([len(w) for t in list(tr.snippet_a) for w in toks(t)])
print("token length mean/median/p90:", round(wl.mean(), 2), int(np.median(wl)), int(np.percentile(wl, 90)))
print("tokens per snippet_a (train) mean:", round(np.mean([len(toks(t)) for t in tr.snippet_a]), 2))
print("tokens per snippet_b (train) mean:", round(np.mean([len(toks(cmap[c])) for c in sorted(tr_cids)]), 2))

print("\n== STRUCTURAL MARKERS ==")
for pat, name in [(r"^\*", "starts with *"), (r"abs[a-z]ract|abstrac", "abstract-like"),
                  (r"\$", "dollar/math"), (r"\d", "contains digit"),
                  (r"\\", "backslash"), (r"[A-Z]", "uppercase")]:
    a = np.mean([bool(re.search(pat, t)) for t in tr.snippet_a])
    b = np.mean([bool(re.search(pat, cmap[c])) for c in sorted(tr_cids)])
    c_ = np.mean([bool(re.search(pat, t)) for t in te.snippet_a])
    print(f"{name:18s} trainA={a:.4f} trainB={b:.4f} testA={c_:.4f}")

print("\n== EXACT / NEAR DUPLICATE TEXT ==")
texts = list(tr.snippet_a) + [cmap[c] for c in sorted(all_cids)]
print("exact duplicate snippets across everything:", len(texts) - len(set(texts)))
