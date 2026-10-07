

# ======================================================= submission validation
def validate_submission(sub, sample_path):
    """Assert the submission grammar before anything is written (a malformed CSV scores
    below zero and still costs a credit)."""
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), \
        "columns %s != %s" % (list(sub.columns), list(sample.columns))
    assert len(sub) == len(sample), "rows %d != %d" % (len(sub), len(sample))
    id_col = sample.columns[0]
    assert sub[id_col].astype(str).tolist() == sample[id_col].astype(str).tolist(), "id mismatch/order"
    assert not sub[id_col].duplicated().any(), "duplicate item_id"
    for c in sample.columns[1:]:
        s = sub[c]
        assert not s.isna().any(), "NaN in %s" % c
        assert (s.astype(str).str.len() > 0).all(), "empty string in %s" % c
    for v in sub["amending_ids"]:
        ids = json.loads(v)
        assert isinstance(ids, list) and all(isinstance(x, int) for x in ids), "bad amending_ids: %r" % v


# ======================================================= model training helpers
RET_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=NUM_THREADS, deterministic=True, force_row_wise=True)
GATE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                   verbose=-1, num_threads=NUM_THREADS, deterministic=True, force_row_wise=True)
RANK_FEATS = ["r_days", "r_idf", "r_pnapp", "n_q", "s_max_pnins"]


def add_rank_feats(X, qi, nq):
    """Within-query normalisations: how a candidate compares with the others for ITS query.

    Computed per query from that query's own candidates only - no statistic crosses rows.
    """
    Z = np.zeros((X.shape[0], len(RANK_FEATS)), dtype=np.float32)
    ix = {k: FEATS.index(k) for k in ("days", "idf_ov", "p_napp", "p_nins")}
    for k in range(nq):
        m = np.where(qi == k)[0]
        if not len(m):
            continue
        for j, col in enumerate(("days", "idf_ov", "p_napp")):
            v = X[m, ix[col]]
            Z[m, j] = np.argsort(np.argsort(-v, kind="stable")) / max(1, len(m) - 1)
        Z[m, 3] = len(m)
        Z[m, 4] = X[m, ix["p_nins"]].max()
    return np.hstack([X, Z])


def fit_bagged(params, X, y, rounds):
    """Train one booster per seed; averaging two seeds is a cheap variance reduction."""
    out = []
    for sd in SEEDS:
        p = dict(params)
        p.update(seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        out.append(lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=rounds))
    return out


def predict_bagged(models, X):
    if not len(X):
        return np.zeros(0)
    return np.mean([m.predict(X) for m in models], axis=0)


def act_folds(acts, n_folds=N_FOLDS):
    """Folds grouped by Act, mirroring the hidden split (test Acts are disjoint from train).

    Acts are placed largest-first into the currently lightest fold, so folds are balanced in
    query count; the assignment is a deterministic function of the data, not of a seed.
    """
    cnt = collections.Counter(acts)
    load = [0] * n_folds
    fold_of = {}
    for a, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        j = int(np.argmin(load))
        fold_of[a] = j
        load[j] += n
    return np.array([fold_of[a] for a in acts]), load


def build_rows(C, FB, queries):
    X, qi, cu = retrieval_matrix(C, FB, queries)
    return add_rank_feats(X, qi, len(queries)), qi, cu


def gate_training_rows(C, queries, rows_meta, oof, qi, cu, truth):
    """Walk each training query's plan once, labelling every parsed edit with the metric
    gain it produced.  Plans are built from OUT-OF-FOLD retrieval scores so the gate is
    trained on the same kind of (imperfect) provision sets it will see at inference."""
    GX, GY, gq = [], [], []
    for k in range(len(queries)):
        m = qi == k
        p = oof[m]; rr = cu[m]
        o = np.argsort(-p, kind="stable")
        sel = [j for j in o if p[j] >= RET_THR] or ([o[0]] if len(o) else [])
        instr, ctxs = build_plan(C, queries[k], [int(rr[j]) for j in sel], [float(p[j]) for j in sel])
        gx, gy, _ = plan_rows(queries[k]["en"], instr, ctxs, truth[rows_meta[k]][1])
        GX += gx; GY += gy; gq += [k] * len(gx)
    X = np.asarray(GX, dtype=np.float32).reshape(-1, len(GFEATS))
    return X, np.asarray(GY, dtype=np.float32), np.asarray(gq, dtype=np.int32)


def main():
    seed_everything()
    for f in ("corpus.csv", "train.csv", "train_targets.csv", "test.csv", "sample_submission.csv"):
        assert (PUBLIC_DIR / f).exists(), "missing input file: %s" % f

    log("building corpus index")
    C = build(str(PUBLIC_DIR / "corpus.csv"))
    FB = FeatureBuilder(C)

    train = pd.read_csv(PUBLIC_DIR / "train.csv", keep_default_na=False)
    targets = pd.read_csv(PUBLIC_DIR / "train_targets.csv", keep_default_na=False)
    test = pd.read_csv(PUBLIC_DIR / "test.csv", keep_default_na=False)
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    truth = {r.item_id: (set(json.loads(r.amending_ids)), norm(r.text_at_date))
             for r in targets.itertuples()}
    tr_rows = list(train.itertuples())
    tr_ids = [r.item_id for r in tr_rows]
    tr_queries = [make_query(r) for r in tr_rows]
    te_rows = list(test.itertuples())
    te_queries = [make_query(r) for r in te_rows]
    log("train %d queries / test %d queries" % (len(tr_queries), len(te_queries)))

    # ---------- MODEL 1: retrieval reranker, fitted on the training queries only ----------
    Xtr, qitr, cutr = build_rows(C, FB, tr_queries)
    ytr = np.array([1 if C.uid[cutr[k]] in truth[tr_ids[qitr[k]]][0] else 0
                    for k in range(len(cutr))])
    log("retrieval matrix %s, %d positives" % (Xtr.shape, int(ytr.sum())))

    fq, load = act_folds(train.act_citation.to_numpy())
    log("Act-grouped folds (query counts): %s" % load)
    oof = np.zeros(len(ytr))
    for f in range(N_FOLDS):
        m = fq[qitr] != f
        oof[~m] = predict_bagged(fit_bagged(RET_PARAMS, Xtr[m], ytr[m], RET_ROUNDS), Xtr[~m])
    # honest in-script report of the retrieval term on held-out Acts
    def selected_uids(p, cols):
        keep = [int(C.uid[cols[j]]) for j in range(len(p)) if p[j] >= RET_THR]
        if not keep and len(p):
            keep = [int(C.uid[cols[int(np.argmax(p))]])]
        return set(keep)

    fold_f1 = []
    for k in range(len(tr_queries)):
        m = np.where(qitr == k)[0]
        fold_f1.append(f1(selected_uids(oof[m], cutr[m]), {int(x) for x in truth[tr_ids[k]][0]}))
    oof_f1 = float(np.mean(fold_f1))
    log("Act-grouped OOF amending_ids F1 = %.4f" % oof_f1)

    ret_models = fit_bagged(RET_PARAMS, Xtr, ytr, RET_ROUNDS)

    # ---------- MODEL 2: edit gate, trained on metric gains measured from OOF plans ----------
    GX, GY, gq = gate_training_rows(C, tr_queries, tr_ids, oof, qitr, cutr, truth)
    glab = (GY > 1e-9).astype(int)
    log("gate matrix %s, %.3f of edits improve the metric" % (GX.shape, glab.mean()))
    gate_models = fit_bagged(GATE_PARAMS, GX, glab, GATE_ROUNDS)

    # ---------- inference: one test row at a time ----------
    Xte, qite, cute = build_rows(C, FB, te_queries)
    pte = predict_bagged(ret_models, Xte)
    log("scored %d test candidates" % len(cute))

    out_ids, out_text = [], []
    for k, r in enumerate(te_rows):
        Q = te_queries[k]
        m = np.where(qite == k)[0]
        if len(m):
            p = pte[m]
            o = np.argsort(-p, kind="stable")
            sel = [j for j in o if p[j] >= RET_THR]
            if not sel:
                sel = [o[0]]                       # always claim the single best candidate
            rowsel = [int(cute[m[j]]) for j in sel]
            scores = [float(p[j]) for j in sel]
            ids = sorted(int(C.uid[i]) for i in rowsel)
        else:
            rowsel, scores, ids = [], [], []
        instr, ctxs = build_plan(C, Q, rowsel, scores)
        if instr:
            gx, _, _ = plan_rows(Q["en"], instr, ctxs, None)
            gp = predict_bagged(gate_models, np.asarray(gx, dtype=np.float32).reshape(-1, len(GFEATS)))
            txt = apply_gated(Q["en"], instr, ctxs, gp, GATE_THR)
        else:
            txt = Q["en"]
        # a prediction must never be empty: an empty cell reloads as NaN and fails the check
        if not txt.strip():
            txt = Q["en"]
        out_ids.append(json.dumps(ids, separators=(",", ":")))
        out_text.append(txt)
        if (k + 1) % 250 == 0:
            log("predicted %d/%d" % (k + 1, len(te_rows)))

    sub = pd.DataFrame({"item_id": [r.item_id for r in te_rows],
                        "amending_ids": out_ids,
                        "text_at_date": out_text})
    sub = sub.set_index("item_id").reindex(sample["item_id"].astype(str)).reset_index()
    sub["amending_ids"] = sub["amending_ids"].fillna("[]")
    sub["text_at_date"] = sub["text_at_date"].fillna("")
    for i in range(len(sub)):
        if not str(sub.at[i, "text_at_date"]).strip():
            sub.at[i, "text_at_date"] = sample.at[i, "text_at_date"]
    sub = sub[list(sample.columns)]
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv")
    sub.to_csv(SUBMISSION_OUT, index=False)
    log("wrote %s shape=%s" % (SUBMISSION_OUT, sub.shape))
    back = pd.read_csv(SUBMISSION_OUT, keep_default_na=False)
    validate_submission(back, PUBLIC_DIR / "sample_submission.csv")
    nz = sum(1 for a, b in zip(back["text_at_date"], test["enacted_text"]) if norm(a) != norm(b))
    log("reload OK | rows with an edited text: %d/%d | mean ids per row: %.2f"
        % (nz, len(back), float(np.mean([len(json.loads(v)) for v in back["amending_ids"]]))))


if __name__ == "__main__":
    main()
