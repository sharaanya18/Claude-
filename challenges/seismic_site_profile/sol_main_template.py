# ===========================================================================
# Models. Every one is trained from scratch on the supplied training records
# inside this script. No pretrained weights, no external data, no GPU.
# ===========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BANDS = ("v1", "v2", "v3", "v4")


def _per_cell(make, Xtr, Ytr, Xs):
    """Fit one classifier per depth cell; score every matrix in Xs with that
    same fit. Xs is a list so the held-out stations and the evaluation records
    are predicted from ONE training pass instead of two."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        m = make()
        m.fit(Xtr, Ytr[:, c])
        cls = m[-1].classes_ if hasattr(m, "steps") else m.classes_
        for o, X in zip(outs, Xs):
            pr = m.predict_proba(X)
            for j, cl in enumerate(cls):
                o[:, c, cl] = pr[:, j]
    return outs


def _centroid_probs(s_tr, s_list, Ytr):
    """Continuous per-cell score -> band distribution, using each band's TRAIN
    score centroid and spread. Train-derived constants, applied per record."""
    outs = [np.zeros((len(s), 4, 4)) for s in s_list]
    for c in range(4):
        cen, sd = [], []
        for k in range(4):
            m = Ytr[:, c] == k
            cen.append(s_tr[m, c].mean() if m.any() else float(k))
            sd.append(s_tr[m, c].std() if m.sum() > 2 else 1.0)
        cen = np.array(cen)
        sd = np.clip(np.array(sd), 1e-6, None)
        for o, sv in zip(outs, s_list):
            d = -0.5 * ((sv[:, c][:, None] - cen[None, :]) / sd[None, :]) ** 2 \
                - np.log(sd)[None, :]
            d -= d.max(axis=1, keepdims=True)
            p = np.exp(d)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_hgb(Xtr, Ytr, Xs, seed):
    """Gradient-boosted trees: the strongest single family on these features."""
    return _per_cell(lambda: HistGradientBoostingClassifier(
        learning_rate=HGB_LR, max_leaf_nodes=HGB_LEAF, max_iter=HGB_IT,
        l2_regularization=HGB_L2, min_samples_leaf=20,
        random_state=seed, early_stopping=False), Xtr, Ytr, Xs)


def m_et(Xtr, Ytr, Xs, seed):
    """Extremely randomised trees: decorrelated from the boosted model."""
    return _per_cell(lambda: ExtraTreesClassifier(
        n_estimators=ET_N, min_samples_leaf=ET_LEAF, max_features="sqrt",
        random_state=seed, n_jobs=N_THREADS), Xtr, Ytr, Xs)


def m_lr(Xtr, Ytr, Xs, seed):
    """Strongly regularised multinomial logistic regression: a smooth linear
    decision surface, which 239 independent stations can actually support."""
    return _per_cell(lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=LR_C, max_iter=4000)),
        Xtr, Ytr, Xs)


def m_pls(Xtr, Ytr, Xs, seed):
    """Partial least squares on all four numeric band targets at once. Suits
    the shape of this problem: many correlated inputs, few samples, and four
    strongly correlated ordered outputs driven by one stiffness factor."""
    m = make_pipeline(StandardScaler(),
                      PLSRegression(n_components=PLS_K, scale=False))
    m.fit(Xtr, Ytr.astype(float))
    return _centroid_probs(m.predict(Xtr), [m.predict(X) for X in Xs], Ytr)


def m_ord(Xtr, Ytr, Xs, seed):
    """Cumulative-link ordinal model: three binary fits per cell for
    P(band > k). Uses the band ordering, so each fit sees every record."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        cums = [np.ones((len(X), 5)) for X in Xs]
        for cum in cums:
            cum[:, 4] = 0.0
        for k in range(3):
            t = (Ytr[:, c] > k).astype(int)
            if len(set(t.tolist())) < 2:
                for cum in cums:
                    cum[:, k + 1] = float(t[0])
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=ORD_C, max_iter=4000))
            m.fit(Xtr, t)
            for cum, X in zip(cums, Xs):
                cum[:, k + 1] = m.predict_proba(X)[:, 1]
        for o, cum in zip(outs, cums):
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_ordgb(Xtr, Ytr, Xs, seed):
    """The cumulative-link construction with boosted trees as the binary
    learner: three fits per cell for P(band > k). Combines the two things that
    measured best independently - the band ordering and gradient boosting - and
    costs about what one four-class booster does (3 binary trees per round
    against 4)."""
    outs = [np.zeros((len(X), 4, 4)) for X in Xs]
    for c in range(4):
        cums = [np.ones((len(X), 5)) for X in Xs]
        for cum in cums:
            cum[:, 4] = 0.0
        for k in range(3):
            t = (Ytr[:, c] > k).astype(int)
            if len(set(t.tolist())) < 2:
                for cum in cums:
                    cum[:, k + 1] = float(t[0])
                continue
            # NOTE: `max_features` is deliberately not passed. It needs
            # scikit-learn >= 1.4, and cross-validation showed the default is
            # BETTER here anyway (0.2470 against 0.2394), so omitting it
            # removes a version dependency and improves the score.
            m = HistGradientBoostingClassifier(
                learning_rate=HGB_LR, max_leaf_nodes=HGB_LEAF, max_iter=HGB_IT,
                l2_regularization=HGB_L2,
                min_samples_leaf=20, random_state=seed, early_stopping=False)
            m.fit(Xtr, t)
            for cum, X in zip(cums, Xs):
                cum[:, k + 1] = m.predict_proba(X)[:, 1]
        for o, cum in zip(outs, cums):
            for k in range(3, 0, -1):
                cum[:, k] = np.minimum(cum[:, k], cum[:, k - 1])
            p = np.clip(cum[:, :4] - cum[:, 1:5], 1e-6, None)
            o[:, c] = p / p.sum(axis=1, keepdims=True)
    return outs


def m_ldak(Xtr, Ytr, Xs, seed):
    """Shrunk LDA to a 3-D discriminant space, then a neighbour vote in it."""
    return _per_cell(lambda: make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="eigen", shrinkage=LDA_SHRINK),
        KNeighborsClassifier(n_neighbors=LDA_K, weights="distance")),
        Xtr, Ytr, Xs)


# The ensemble members, chosen by equal-weight forward selection on repeated
# station-held-out cross-validation (see final_approach.md).
MODELS = [(n, globals()["m_" + n]) for n in MODEL_NAMES]


# ===========================================================================
# Decode. Macro-F1 scores a band that is never predicted as 0, so a plain
# argmax that avoids the two middle bands throws away half of each cell.
# A per-band weight vector is fitted on the out-of-fold TRAINING predictions
# so that the predicted marginal matches the training band frequencies; it is
# then a fixed 4x4 constant applied to one record at a time.
# ===========================================================================
def fit_band_weights(prob, Y, iters=200):
    W = np.ones((4, 4))
    for c in range(4):
        target = np.bincount(Y[:, c], minlength=4) / len(Y)
        w = np.ones(4)
        for _ in range(iters):
            freq = np.bincount((prob[:, c, :] * w).argmax(axis=1), minlength=4) / len(Y)
            w = w * ((target + 1e-3) / (freq + 1e-3)) ** 0.25
            w = w / w.mean()
        W[c] = w
    return W


def apply_band_weights(prob, W):
    return (prob * W[None, :, :]).argmax(axis=2)


# ---------------------------------------------------------------------------
# Structured decode. The four per-cell models are fitted independently, and
# measurement shows they UNDER-couple the column: the correlation between
# predicted cell bands (0.66-0.76) is lower than between the true ones
# (0.71-0.88). Only 68 of the 256 possible profiles occur in training, so the
# empirical distribution over profiles carries real information the per-cell
# models never see. Re-weighting each record's four distributions by that
# distribution and taking the marginals restores the coupling.
#
# The prior is counted from TRAINING labels only and is a fixed table at
# inference; the re-weighting uses one record's own four distributions and
# nothing else, so the prediction stays per-record. LAMBDA was chosen from the
# middle of a flat region of cross-validated scores (0.7-1.3), not at its edge.
# ---------------------------------------------------------------------------
def fit_profile_prior(Y):
    profiles, counts = np.unique(Y, axis=0, return_counts=True)
    return profiles, counts.astype(float)


def joint_decode(prob, profiles, counts, lam=JOINT_LAMBDA):
    logpri = np.log(counts / counts.sum())
    lp = np.log(prob + 1e-9)
    # score every observed profile for every record, then marginalise per cell
    sc = lp[:, np.arange(4)[None, :], profiles].sum(axis=2) + lam * logpri[None, :]
    sc -= sc.max(axis=1, keepdims=True)
    post = np.exp(sc)
    post /= post.sum(axis=1, keepdims=True)
    out = np.zeros_like(prob)
    for c in range(4):
        for k in range(4):
            out[:, c, k] = post[:, profiles[:, c] == k].sum(axis=1)
    return out


# ===========================================================================
# Submission validation: a malformed file is rejected outright and scores
# nothing, so this raises before anything is written.
# ===========================================================================
def validate_submission(sub, sample_path, test_ids):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), \
        f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    ids = sub["id"].astype(str).tolist()
    assert ids == [str(i) for i in test_ids], "id set/order does not match test.csv"
    assert len(set(ids)) == len(ids), "duplicate ids"
    for p in sub["profile"].astype(str):
        parts = p.split("|")
        assert len(parts) == 4, f"profile {p!r} does not have four cells"
        for b in parts:
            assert b in BANDS, f"band {b!r} outside v1..v4"
    assert not sub["profile"].isna().any(), "NaN profile"
    assert (sub["profile"].astype(str).str.len() > 0).all(), "empty profile"
