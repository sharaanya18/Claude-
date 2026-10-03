import sys, os
root = os.environ.get("DATA_ROOT", "dataset/public")
sys.argv = ["x", root, "/kaggle/working/dev.csv" if os.path.exists("/kaggle/working") else "working/dev.csv"]
import numpy as np, pandas as pd, json, time, torch
from scipy.optimize import linear_sum_assignment
import solution as S
folios = S.load_folios(S.PUBLIC_DIR); tr = pd.read_csv(S.PUBLIC_DIR / "train.csv").drop_duplicates("folio_id")
tt = pd.read_csv(S.PUBLIC_DIR / "train_targets.csv"); gold = {r.target_id: json.loads(r.prediction)["descriptor_id"] for r in tt.itertuples()}
tok = S.AutoTokenizer.from_pretrained(S.MODEL_NAME)
items = [S.folio_arrays(folios[f], gold) for f in tr.folio_id]; bands = list(tr.evidence_band)
for it in items: it["es_tok"], it["gl_tok"] = S.tokenize(tok, it["es"]), S.tokenize(tok, it["gl"])
fold, comp = S.build_groups(items, S.N_CV_FOLDS)
k = 0; tr_idx = [i for i in range(len(items)) if fold[i] != k]; va_idx = [i for i in range(len(items)) if fold[i] == k]
print("train/val folios", len(tr_idx), len(va_idx), flush=True)
t = time.time(); model = S.train_model([items[i] for i in tr_idx], tok, "dev"); print("train secs", time.time() - t, flush=True)
pairs = S.similarity_matrices(model, [items[i] for i in va_idx], tok); vi = [items[i] for i in va_idx]; vb = [bands[i] for i in va_idx]
def acc_of(mats): return np.mean([np.mean(linear_sum_assignment(-m)[1] == it["gold"]) for m, it in zip(mats, vi)])
print("val hungarian acc dense %.3f colbert %.3f combined %.3f | argmax dense %.3f" % (acc_of([p[0] for p in pairs]), acc_of([p[1] for p in pairs]), acc_of([S.combine(p) for p in pairs]), np.mean([np.mean(p[0].argmax(1) == it["gold"]) for p, it in zip(pairs, vi)])), flush=True)
np.savez("dev_sims.npz", va_idx=np.array(va_idx), fold=fold, **{f"d{i}": p[0] for i, p in enumerate(pairs)}, **{f"c{i}": p[1] for i, p in enumerate(pairs)})
for bd in sorted(set(vb)): print("band", bd, "hungarian acc dense %.3f combined %.3f" % (np.mean([np.mean(linear_sum_assignment(-pairs[i][0])[1] == vi[i]["gold"]) for i in range(len(vi)) if vb[i] == bd]), np.mean([np.mean(linear_sum_assignment(-S.combine(pairs[i]))[1] == vi[i]["gold"]) for i in range(len(vi)) if vb[i] == bd])), flush=True)
sims = [S.combine(p) for p in pairs]
ev = list(range(0, len(vi), 2)); od = list(range(1, len(vi), 2)); sc = []
for a_, b_ in [(ev, od), (od, ev)]:
    p, _ = S.fit_calibration([sims[i] for i in a_], [vi[i] for i in a_], [vb[i] for i in a_])
    sc.append(S.total_score([sims[i] for i in b_], [vi[i] for i in b_], [vb[i] for i in b_], p)); print("params", np.round(p, 3), "cross-fit score", round(sc[-1], 2), flush=True)
print("DEV SCORE (cross-fit calibration) %.2f" % np.mean(sc))
np.savez("dev_sims.npz", va_idx=np.array(va_idx), fold=fold, **{f"d{i}": p[0] for i, p in enumerate(pairs)}, **{f"c{i}": p[1] for i, p in enumerate(pairs)})
