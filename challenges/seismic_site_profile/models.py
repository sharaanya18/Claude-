"""Model wrappers. Every one returns (n,4,4) class probabilities per record."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


def per_cell(make, w=None):
    """Independent classifier per depth cell."""
    def fp(Xtr, Ytr, Xva, seed):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            m = make(seed)
            sw = None if w is None else w
            try:
                m.fit(Xtr, Ytr[:, c], **({} if sw is None else {"sample_weight": sw}))
            except TypeError:
                m.fit(Xtr, Ytr[:, c])
            pr = m.predict_proba(Xva)
            cls = m[-1].classes_ if hasattr(m, "steps") else m.classes_
            for j, cl in enumerate(cls):
                out[:, c, cl] = pr[:, j]
        return out
    return fp


def LR(C=0.03, pca=None):
    def mk(s):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=pca, random_state=0))
        steps.append(LogisticRegression(C=C, max_iter=4000))
        return make_pipeline(*steps)
    return per_cell(mk)


def HGB(lr=0.06, leaf=15, it=200, l2=1.0, mins=20, ff=1.0):
    return per_cell(lambda s: HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaf, max_iter=it, l2_regularization=l2,
        min_samples_leaf=mins, max_features=ff, random_state=s, early_stopping=False))


def ET(n=600, leaf=3, mf="sqrt"):
    return per_cell(lambda s: ExtraTreesClassifier(
        n_estimators=n, min_samples_leaf=leaf, max_features=mf,
        random_state=s, n_jobs=4))


def KNN(k=25, metric="cosine", pca=None):
    def mk(s):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=pca, random_state=0))
        steps.append(KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric))
        return make_pipeline(*steps)
    return per_cell(mk)


def RIDGE_LATENT(alpha=100.0, pca=None):
    """Ordinal route: regress the numeric band (0..3) per cell, then turn the
    continuous score into a distribution over bands with fixed, TRAIN-derived
    thresholds. Exploits the band ordering and the 0.71-0.88 correlation
    between depth cells via a shared multi-output fit."""
    def fp(Xtr, Ytr, Xva, seed):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=pca, random_state=0))
        steps.append(Ridge(alpha=alpha))
        m = make_pipeline(*steps)
        m.fit(Xtr, Ytr.astype(float))
        str_ = m.predict(Xtr)
        sva = m.predict(Xva)
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            # thresholds at the TRAIN class priors, read off TRAIN scores only
            pri = np.bincount(Ytr[:, c], minlength=4) / len(Ytr)
            qs = np.cumsum(pri)[:3]
            thr = np.quantile(str_[:, c], qs)
            # soft assignment: distance to each band's train-score centroid
            cen = np.array([str_[Ytr[:, c] == k, c].mean() if (Ytr[:, c] == k).any()
                            else k for k in range(4)])
            sd = str_[:, c].std() + 1e-9
            d = -((sva[:, c][:, None] - cen[None, :]) / sd) ** 2
            p = np.exp(d - d.max(axis=1, keepdims=True))
            out[:, c] = p / p.sum(axis=1, keepdims=True)
        return out
    return fp


# ---------------------------------------------------------------------------
# Families chosen for this problem's structure rather than off the shelf:
# 239 independent labelled stations, ~400 correlated features, four ORDERED
# bands per cell and 0.71-0.88 correlation between the cells.
# ---------------------------------------------------------------------------
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _centroid_probs(str_, sva, Ytr):
    """Turn a continuous per-cell score into a band distribution using the
    TRAIN score centroid and spread of each band. Train-derived constants only,
    applied to one record at a time."""
    out = np.zeros((len(sva), 4, 4))
    for c in range(4):
        cen, sd = [], []
        for k in range(4):
            m = Ytr[:, c] == k
            cen.append(str_[m, c].mean() if m.any() else float(k))
            sd.append(str_[m, c].std() if m.sum() > 2 else 1.0)
        cen = np.array(cen); sd = np.clip(np.array(sd), 1e-6, None)
        d = -0.5 * ((sva[:, c][:, None] - cen[None, :]) / sd[None, :]) ** 2 \
            - np.log(sd)[None, :]
        d -= d.max(axis=1, keepdims=True)
        p = np.exp(d)
        out[:, c] = p / p.sum(axis=1, keepdims=True)
    return out


def PLS(k=6):
    """Partial least squares to k latent components predicting all four numeric
    band targets jointly. Built for exactly this shape: many correlated inputs,
    few samples, several correlated ordered outputs - the shared stiffness
    factor is what the first components recover."""
    def fp(Xtr, Ytr, Xva, seed):
        m = make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False))
        m.fit(Xtr, Ytr.astype(float))
        return _centroid_probs(m.predict(Xtr), m.predict(Xva), Ytr)
    return fp


def ORDINAL(C=0.03):
    """Cumulative-link decomposition: per cell three binary models for
    P(band > k). Uses the band ORDER, so each classifier sees all records
    instead of splitting them four ways - more sample-efficient than a 4-way
    multinomial on 239 units."""
    def fp(Xtr, Ytr, Xva, seed):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            cum = np.ones((len(Xva), 5))        # P(y > -1) = 1, P(y > 3) = 0
            cum[:, 4] = 0.0
            for k in range(3):
                t = (Ytr[:, c] > k).astype(int)
                if len(set(t)) < 2:
                    cum[:, k + 1] = float(t[0])
                    continue
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=4000))
                m.fit(Xtr, t)
                cum[:, k + 1] = m.predict_proba(Xva)[:, 1]
            # enforce monotone cumulative probabilities, then difference
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = cum[:, :4] - cum[:, 1:5]
            p = np.clip(p, 1e-6, None)
            out[:, c] = p / p.sum(axis=1, keepdims=True)
        return out
    return fp


def LDA_KNN(k=30, shrink=0.3):
    """Supervised metric: shrunk LDA gives a 3-D discriminant space per cell,
    then a neighbour vote in it. The brief's 5-NN tier shows similarity in the
    right space is strong; LDA learns that space instead of assuming it."""
    def mk(s):
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrink),
            KNeighborsClassifier(n_neighbors=k, weights="distance"))
    return per_cell(mk)


def LDA(shrink=0.3):
    return per_cell(lambda s: make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrink)))


def per_cell_weighted(make):
    """Same as per_cell but accepting per-row sample weights, used to give every
    STATION equal total weight. The corpus has 3-5 near-duplicate records per
    station sharing one answer, so unweighted fitting lets a 5-record station
    count 67% more than a 3-record one and encourages memorising repeats."""
    def fp(Xtr, Ytr, Xva, seed, sw=None):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            m = make(seed)
            if sw is None:
                m.fit(Xtr, Ytr[:, c])
            elif hasattr(m, "steps"):
                m.fit(Xtr, Ytr[:, c], **{m.steps[-1][0] + "__sample_weight": sw})
            else:
                m.fit(Xtr, Ytr[:, c], sample_weight=sw)
            pr = m.predict_proba(Xva)
            cls = m[-1].classes_ if hasattr(m, "steps") else m.classes_
            for j, cl in enumerate(cls):
                out[:, c, cl] = pr[:, j]
        return out
    return fp


def with_site_weights(fp_weighted, site):
    """Bind station-equalising weights to a weighted fitter."""
    import collections
    cnt = collections.Counter(site)

    def fp(Xtr, Ytr, Xva, seed, idx=None):
        sw = None
        if idx is not None:
            sw = np.array([1.0 / cnt[s] for s in site[idx]])
            sw = sw * len(sw) / sw.sum()
        return fp_weighted(Xtr, Ytr, Xva, seed, sw)
    return fp


def _hgb_kwargs(lr, leaf, it, l2, ff, seed):
    kw = dict(learning_rate=lr, max_leaf_nodes=leaf, max_iter=it,
              l2_regularization=l2, min_samples_leaf=20,
              random_state=seed, early_stopping=False)
    if ff is not None:          # omitted entirely when None: needs sklearn>=1.4
        kw["max_features"] = ff
    return kw


def ORDINAL_GB(lr=0.06, leaf=15, it=200, ff=None, l2=1.0):
    """Cumulative-link ordinal model with boosted trees as the binary learner:
    three fits per cell for P(band > k). Combines the two things that worked
    independently - the band ordering and gradient boosting - and costs about
    the same as one four-class booster (3 binary trees per round against 4)."""
    def fp(Xtr, Ytr, Xva, seed):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            cum = np.ones((len(Xva), 5)); cum[:, 4] = 0.0
            for k in range(3):
                t = (Ytr[:, c] > k).astype(int)
                if len(set(t.tolist())) < 2:
                    cum[:, k + 1] = float(t[0]); continue
                m = HistGradientBoostingClassifier(**_hgb_kwargs(lr, leaf, it, l2, ff, seed))
                m.fit(Xtr, t)
                cum[:, k + 1] = m.predict_proba(Xva)[:, 1]
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            out[:, c] = p / p.sum(axis=1, keepdims=True)
        return out
    return fp


def ORDINAL_ET(n=800, leaf=3):
    """Same cumulative-link construction with extremely randomised trees."""
    def fp(Xtr, Ytr, Xva, seed):
        out = np.zeros((len(Xva), 4, 4))
        for c in range(4):
            cum = np.ones((len(Xva), 5)); cum[:, 4] = 0.0
            for k in range(3):
                t = (Ytr[:, c] > k).astype(int)
                if len(set(t.tolist())) < 2:
                    cum[:, k + 1] = float(t[0]); continue
                m = ExtraTreesClassifier(n_estimators=n, min_samples_leaf=leaf,
                                         max_features="sqrt", random_state=seed,
                                         n_jobs=4)
                m.fit(Xtr, t)
                cum[:, k + 1] = m.predict_proba(Xva)[:, 1]
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            out[:, c] = p / p.sum(axis=1, keepdims=True)
        return out
    return fp


def select_icc(frac=0.5, inside_fold=True):
    """Feature selection by station-consistency x label relation, FITTED INSIDE
    the training fold. A feature is kept when its variance is mostly between
    stations rather than between earthquakes at one station (so it describes the
    site) and it relates to the station's overall band level. Both statistics
    need the station key, which only training records have - the selected
    indices are then a fixed list applied to any record."""
    def score_feats(X, Y, site):
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        codes = pd.factorize(site)[0]
        ns = codes.max() + 1
        mu = np.zeros((ns, X.shape[1])); cnt = np.zeros(ns)
        for i in range(ns):
            m = codes == i
            mu[i] = Xs[m].mean(0); cnt[i] = m.sum()
        within = np.zeros(X.shape[1])
        for i in range(ns):
            m = codes == i
            within += ((Xs[m] - mu[i]) ** 2).sum(0)
        within /= max(len(Xs) - ns, 1)
        between = (cnt[:, None] * (mu - Xs.mean(0)) ** 2).sum(0) / max(ns - 1, 1)
        icc = np.clip((between - within) / (between + within * (cnt.mean() - 1) + 1e-12), 0, 1)
        yb = Y.mean(1)
        sy = np.array([yb[codes == i][0] for i in range(ns)])
        rho = np.abs(np.array([np.corrcoef(np.argsort(np.argsort(mu[:, j])),
                                           np.argsort(np.argsort(sy)))[0, 1]
                               for j in range(X.shape[1])]))
        return icc * np.nan_to_num(rho)
    return score_feats


import pandas as pd


def with_selection(base, site, frac=0.5):
    """Wrap a model so each fold selects its own features on training rows."""
    scorer = select_icc()

    def fp(Xtr, Ytr, Xva, seed, tr_site=None):
        if tr_site is None:
            return base(Xtr, Ytr, Xva, seed)
        sc = scorer(Xtr, Ytr, tr_site)
        k = max(int(len(sc) * frac), 10)
        idx = np.argsort(-sc)[:k]
        return base(Xtr[:, idx], Ytr, Xva[:, idx], seed)
    return fp
