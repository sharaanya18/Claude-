import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as AP
D = Path(__file__).resolve().parents[1] / "dataset/public"
def load():
    tr = pd.read_csv(D/"train.csv").merge(pd.read_csv(D/"train_labels.csv"), on="id", validate="one_to_one")
    te = pd.read_csv(D/"test.csv")
    return tr, te
def splits(tr, seed=42, k=5):
    y = tr.p_same_reader.values.astype(int); cnt = tr.annotator_count.values
    norm = tr.text.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    g = pd.factorize(norm)[0]
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(tr, y*10+cnt, g))
def cv(fit_predict, tr, seeds=(42,), k=5, verbose=True):
    y = tr.p_same_reader.values.astype(int); res=[]
    oofs=[]
    for s in seeds:
        oof = np.zeros(len(tr)); folds=[]
        for a,b in splits(tr,s,k):
            oof[b] = fit_predict(tr.iloc[a], y[a], tr.iloc[b]); folds.append(AP(y[b], oof[b]))
        res.append(np.mean(folds)); oofs.append(oof)
        if verbose: print(f"  seed {s}: mean {np.mean(folds):.4f} sd {np.std(folds):.4f} folds {[round(f,3) for f in folds]} pooled {AP(y,oof):.4f}", flush=True)
    return np.mean(res), oofs
