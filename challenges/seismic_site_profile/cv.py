"""Site-held-out CV harness. The supplied folds hold out whole stations, which
is what the evaluation split does, so a fold's validation records are all from
stations the model has never seen. Scored record-level with the exact metric."""
import numpy as np, pandas as pd
from pathlib import Path
import metric

P = Path("dataset/public")
BANDS = ["v1","v2","v3","v4"]
B2I = {b:i for i,b in enumerate(BANDS)}

def load():
    c = np.load("feat_cache.npz", allow_pickle=True)
    X, ids, cols, gtag = c["X"], list(c["ids"]), list(c["cols"]), list(c["gtag"])
    pos = {k:i for i,k in enumerate(ids)}
    tr = pd.read_csv(P/"train.csv"); te = pd.read_csv(P/"test.csv")
    fo = pd.read_csv(P/"folds.csv")
    tr = tr.merge(fo, on="id")
    Y = np.array([[B2I[b] for b in s.split("|")] for s in tr.profile])
    Xtr = X[[pos[i] for i in tr.id]]
    Xte = X[[pos[i] for i in te.id]]
    return dict(Xtr=Xtr, Y=Y, site=tr.site.values, fold=tr.fold.values,
                tr_ids=tr.id.values, Xte=Xte, te_ids=te.id.values,
                cols=np.array(cols), gtag=np.array(gtag))

def sel(D, groups):
    m = np.isin(D["gtag"], list(groups))
    return D["Xtr"][:, m], D["Xte"][:, m], m

def score_oof(Y, Pred):
    """Pred: (n,4) int. Exact metric at record level over all OOF rows."""
    G = np.array(BANDS, dtype=object)[Y]
    Pr = np.array(BANDS, dtype=object)[Pred]
    return metric.score_cells(G, Pr, return_detail=True)

def run_cv(D, groups, fit_predict, n_folds=5, verbose=True, seed=0):
    """fit_predict(Xtr, Ytr, Xva, seed) -> (nva,4,4) probabilities."""
    Xa, _, _ = sel(D, groups)
    Y, fold = D["Y"], D["fold"]
    Prob = np.zeros((len(Y), 4, 4))
    for f in range(n_folds):
        va = fold == f; tr = ~va
        Prob[va] = fit_predict(Xa[tr], Y[tr], Xa[va], seed)
    Pred = Prob.argmax(axis=2)
    s, d = score_oof(Y, Pred)
    per_fold = []
    for f in range(n_folds):
        va = fold == f
        sf, _ = score_oof(Y[va], Pred[va])
        per_fold.append(sf)
    if verbose:
        print(f"  OOF {s:.4f} | folds {' '.join(f'{x:.3f}' for x in per_fold)} "
              f"mean {np.mean(per_fold):.4f} std {np.std(per_fold):.4f} | "
              f"cellF1 {' '.join(f'{x:.3f}' for x in d['cell_f1'])}")
    return dict(score=s, detail=d, per_fold=per_fold, fold_mean=float(np.mean(per_fold)),
                fold_std=float(np.std(per_fold)), prob=Prob, pred=Pred)


def load_crops():
    """(n, ncrop, d) crop features aligned to the train/test row order."""
    c = np.load("crop_cache.npz", allow_pickle=True)
    X, ids = c["X"], list(c["ids"])
    pos = {k: i for i, k in enumerate(ids)}
    tr = pd.read_csv(P/"train.csv").merge(pd.read_csv(P/"folds.csv"), on="id")
    te = pd.read_csv(P/"test.csv")
    return X[[pos[i] for i in tr.id]], X[[pos[i] for i in te.id]]


def run_cv_views(D, groups, fit_predict, Vtr=None, n_folds=5, seeds=(0,),
                 aug=True, verbose=True, tag=""):
    """CV with several VIEWS of each record (the full span plus sub-window crops).

    Views are extra TRAINING rows carrying the same label - augmentation of a
    real sample - and at inference the views of one record have their
    probabilities averaged. Nothing is pooled across records, so the inference
    path is identical for one record on its own.
    """
    Xa, _, m = sel(D, groups)
    Y, fold = D["Y"], D["fold"]
    Va = None if Vtr is None else Vtr[:, :, m]           # (n, ncrop, d)
    Prob = np.zeros((len(Y), 4, 4))
    for f in range(n_folds):
        va = fold == f; tr = ~va
        if Va is None:
            Xt, Yt = Xa[tr], Y[tr]
            views = [Xa[va]]
        else:
            nv = Va.shape[1]
            if aug:
                Xt = np.vstack([Xa[tr]] + [Va[tr, j] for j in range(nv)])
                Yt = np.vstack([Y[tr]] * (nv + 1))
            else:
                Xt, Yt = Xa[tr], Y[tr]
            views = [Xa[va]] + [Va[va, j] for j in range(nv)]
        acc = np.zeros((int(va.sum()), 4, 4))
        for s in seeds:
            for v in views:
                acc += fit_predict(Xt, Yt, v, s)
        Prob[va] = acc / (len(seeds) * len(views))
    Pred = Prob.argmax(axis=2)
    s_, d = score_oof(Y, Pred)
    pf = [score_oof(Y[fold == f], Pred[fold == f])[0] for f in range(n_folds)]
    if verbose:
        print(f"  {tag:28s} OOF {s_:.4f} | folds {' '.join(f'{x:.3f}' for x in pf)} "
              f"mean {np.mean(pf):.4f} std {np.std(pf):.4f} | "
              f"cell {' '.join(f'{x:.3f}' for x in d['cell_f1'])}")
    return dict(score=s_, detail=d, per_fold=pf, prob=Prob, pred=Pred)


def run_repeated(D, groups, fit_predict, folds_list, Vtr=None, seeds=(0,),
                 aug=False, tag=""):
    """Same candidate over several station-held-out splits. Reports the mean and
    spread across splits, which is the number decisions are made on: one split
    leaves an OOF standard error of the same size as the gains being chased."""
    res = []
    for i, f in enumerate(folds_list):
        Dx = dict(D); Dx["fold"] = f
        r = run_cv_views(Dx, groups, fit_predict, Vtr=Vtr, seeds=seeds, aug=aug,
                         verbose=False)
        res.append(r)
    sc = [r["score"] for r in res]
    cells = np.mean([r["detail"]["cell_f1"] for r in res], axis=0)
    print(f"  {tag:30s} MEAN {np.mean(sc):.4f} +-{np.std(sc):.4f} over "
          f"{len(sc)} splits [{' '.join(f'{x:.3f}' for x in sc)}] | "
          f"cell {' '.join(f'{x:.3f}' for x in cells)}")
    return dict(mean=float(np.mean(sc)), std=float(np.std(sc)), scores=sc, res=res)
