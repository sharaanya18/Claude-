# ===========================================================================
# Driver
# ===========================================================================
def build_features(z, ids, n_views):
    """Feature matrix for a list of ids. View 0 is the record's full valid span;
    the remaining views are overlapping sub-windows of THAT SAME record, used as
    extra training rows and as several looks at one record at inference.
    Nothing is ever computed across records."""
    out = np.zeros((len(ids), n_views, N_FEAT), dtype=np.float64)
    for i, k in enumerate(ids):
        w = np.asarray(z[k], dtype=np.float64)
        views = [w] + (crops(w) if n_views > 1 else [])
        for v in range(n_views):
            d = extract(views[v])
            out[i, v] = np.concatenate([d[g] for g in USE_GROUPS])
    return out


def main():
    seed_everything()
    log("start")
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    folds = pd.read_csv(PUBLIC_DIR / "folds.csv")
    z = np.load(PUBLIC_DIR / "waveforms.npz")

    # folds.csv holds out whole stations, which is how the evaluation set is
    # built, so it is used both for the blend/decode fitting below and as the
    # honest estimate of generalisation to unseen stations.
    train = train.merge(folds, on="id", how="left")
    assert train.fold.notna().all(), "a training id is missing from folds.csv"
    Y = np.array([[BANDS.index(b) for b in p.split("|")] for p in train.profile])
    fold = train.fold.to_numpy().astype(int)

    Xtr = build_features(z, train.id.tolist(), N_VIEWS)
    log(f"train features {Xtr.shape}")
    Xte = build_features(z, test.id.tolist(), N_VIEWS)
    log(f"test features {Xte.shape}")

    # ---- fold models. Each fold model predicts its own held-out stations (the
    # out-of-fold material that the blend weights and the band weights are
    # fitted on) and the evaluation records (averaged over folds, a free
    # bagging ensemble). No evaluation label or statistic is ever used.
    oof = {n: np.zeros((len(Y), 4, 4)) for n, _ in MODELS}
    tep = {n: np.zeros((len(test), 4, 4)) for n, _ in MODELS}
    nf = int(fold.max()) + 1
    for f in range(nf):
        va = fold == f
        tr = ~va
        if N_VIEWS > 1:
            Xt = np.vstack([Xtr[tr, v] for v in range(N_VIEWS)])
            Yt = np.vstack([Y[tr]] * N_VIEWS)
        else:
            Xt, Yt = Xtr[tr, 0], Y[tr]
        for name, fn in MODELS:
            for s in SEEDS:
                for v in range(N_VIEWS):
                    # one training pass scores both the held-out stations and
                    # the evaluation records
                    pv, pt = fn(Xt, Yt, [Xtr[va, v], Xte[:, v]], s)
                    oof[name][va] += pv / (len(SEEDS) * N_VIEWS)
                    tep[name] += pt / (len(SEEDS) * N_VIEWS * nf)
        log(f"fold {f} done")

    for name, _ in MODELS:
        s, d = score_cells_int(Y, oof[name].argmax(axis=2))
        log(f"  OOF {name:5s} {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # ---- blend. Equal weights over the families: with 239 independent
    # stations, fitted weights have more freedom than the out-of-fold set can
    # resolve, and a plain average was at least as good in cross-validation.
    oof_b = np.mean([oof[n] for n, _ in MODELS], axis=0)
    te_b = np.mean([tep[n] for n, _ in MODELS], axis=0)
    s, d = score_cells_int(Y, oof_b.argmax(axis=2))
    log(f"  OOF blend argmax {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # ---- structured decode. The out-of-fold figure below is computed with the
    # profile prior counted on the OTHER folds only, so it is an honest
    # estimate for stations the model has not seen.
    oof_dec = np.zeros_like(oof_b)
    for f in range(nf):
        va = fold == f
        prof, cnt = fit_profile_prior(Y[~va])
        oof_dec[va] = joint_decode(oof_b[va], prof, cnt)
    s, d = score_cells_int(Y, oof_dec.argmax(axis=2))
    log(f"  OOF blend decoded {s:.4f} cells " + " ".join(f"{x:.3f}" for x in d))

    # for the evaluation records the prior is counted on all training labels
    prof, cnt = fit_profile_prior(Y)
    log(f"  profile prior: {len(prof)} distinct profiles over {int(cnt.sum())} records")
    pred = joint_decode(te_b, prof, cnt).argmax(axis=2)
    profiles = ["|".join(BANDS[k] for k in row) for row in pred]

    sub = pd.DataFrame({"id": test.id.to_numpy(), "profile": profiles})
    sub = sub[list(sample.columns)]
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv", test.id.tolist())
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")
    for c in range(4):
        log(f"  cell{c} predicted band counts "
            f"{np.bincount(pred[:, c], minlength=4).tolist()} "
            f"(train {np.bincount(Y[:, c], minlength=4).tolist()})")


if __name__ == "__main__":
    main()
