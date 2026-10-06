"""Exact reimplementation of the Seismic Site Profile grader.

score = (cell_score - blind) / (1 - blind), clamped to [0.001, 1.0]
  cell_score = mean over the 4 depth cells of macro-F1 within that cell
  blind      = mean over the 4 depth cells of 1 / (#bands present in that cell)

The macro-F1 is averaged over the bands PRESENT IN GOLD for that cell. That
reading is pinned by the two identities stated in the brief:
  * predicting one band everywhere scores at most 1/K  -> only gold bands averaged
  * drawing at the gold band frequencies scores exactly 1/K
    (F1 of class i under independent draws = p_i, so the mean is 1/K)
Both are asserted in the self-test below.
"""
import numpy as np

BANDS = ("v1", "v2", "v3", "v4")


def f1_per_class(gold, pred, label):
    g = gold == label
    p = pred == label
    tp = int(np.sum(g & p))
    fp = int(np.sum(~g & p))
    fn = int(np.sum(g & ~p))
    if tp == 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def macro_f1_cell(gold, pred):
    """Macro-F1 over the bands present in gold."""
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    labels = np.unique(gold)
    return float(np.mean([f1_per_class(gold, pred, l) for l in labels])), len(labels)


def parse_profiles(profiles):
    """(n,4) array of band strings; malformed rows become '' (scored WRONG)."""
    out = np.full((len(profiles), 4), "", dtype=object)
    for i, s in enumerate(profiles):
        if not isinstance(s, str):
            continue
        parts = s.split("|")
        if len(parts) != 4:
            continue
        for j, p in enumerate(parts):
            out[i, j] = p if p in BANDS else ""
    return out


def score(gold_profiles, pred_profiles, return_detail=False):
    G = parse_profiles(gold_profiles)
    P = parse_profiles(pred_profiles)
    assert len(G) == len(P)
    cell_f1, ks = [], []
    for c in range(4):
        f1, k = macro_f1_cell(G[:, c], P[:, c])
        cell_f1.append(f1)
        ks.append(k)
    cell_score = float(np.mean(cell_f1))
    blind = float(np.mean([1.0 / k for k in ks]))
    raw = (cell_score - blind) / (1.0 - blind)
    s = min(1.0, max(0.001, raw))
    if return_detail:
        return s, {"raw": raw, "cell_score": cell_score, "blind": blind,
                   "cell_f1": cell_f1, "k": ks}
    return s


def score_cells(gold_cells, pred_cells, return_detail=False):
    """Same metric from (n,4) arrays of band strings, skipping re-parsing."""
    G = np.asarray(gold_cells, dtype=object)
    P = np.asarray(pred_cells, dtype=object)
    cell_f1, ks = [], []
    for c in range(4):
        f1, k = macro_f1_cell(G[:, c], P[:, c])
        cell_f1.append(f1)
        ks.append(k)
    cell_score = float(np.mean(cell_f1))
    blind = float(np.mean([1.0 / k for k in ks]))
    raw = (cell_score - blind) / (1.0 - blind)
    s = min(1.0, max(0.001, raw))
    if return_detail:
        return s, {"raw": raw, "cell_score": cell_score, "blind": blind,
                   "cell_f1": cell_f1, "k": ks}
    return s


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4000
    # --- perfect prediction -> 1.0
    g = ["|".join(rng.choice(BANDS, 4)) for _ in range(n)]
    assert abs(score(g, g) - 1.0) < 1e-12, score(g, g)

    # --- a single constant profile -> floor 0.001
    const = ["v2|v2|v2|v2"] * n
    assert score(g, const) == 0.001, score(g, const)

    # --- random draw at the gold band frequencies -> raw ~ 0 (floor 0.001)
    G = parse_profiles(g)
    draws = np.empty((n, 4), dtype=object)
    for c in range(4):
        vals, cnt = np.unique(G[:, c], return_counts=True)
        draws[:, c] = rng.choice(vals, n, p=cnt / cnt.sum())
    s, d = score_cells(G, draws, return_detail=True)
    assert abs(d["cell_score"] - 0.25) < 0.02, d
    assert abs(d["raw"]) < 0.03, d

    # --- predicting one band everywhere scores at most 1/K in that cell
    one = np.full((n, 4), "v1", dtype=object)
    s2, d2 = score_cells(G, one, return_detail=True)
    assert all(f <= 0.25 + 1e-9 for f in d2["cell_f1"]), d2

    # --- blind reference tracks the number of gold bands present
    g3 = ["v1|v1|v1|v1"] * 10 + ["v2|v2|v2|v2"] * 10          # K=2 in every cell
    _, d3 = score_cells(parse_profiles(g3), parse_profiles(g3), return_detail=True)
    assert abs(d3["blind"] - 0.5) < 1e-12, d3

    # --- malformed / unknown band is WRONG, not a crash, and costs only its cells
    pred = list(g)
    pred[0] = "v1|V2|v3"          # wrong arity AND bad case -> whole row's 4 cells wrong
    s4 = score(g, pred)
    assert 0.98 < s4 < 1.0, s4
    pred2 = list(g); pred2[0] = g[0].replace("v", "V", 1)   # one bad cell only
    s5 = score(g, pred2)
    assert s4 < s5 < 1.0, (s4, s5)

    # --- reproduce the brief's quoted 1.0000 -> 0.9965 for one unrecognised band.
    # One row of 333 made unrecognisable spoils all four of its cells. If the
    # record's band has m members in a cell, that class keeps F1 = 2(m-1)/(2m-1),
    # so each cell's macro loses 1/(4(2m-1)) and the reported score loses
    # 1/(3(2m-1)). m = 48 gives exactly 0.9965, which pins both the
    # macro-over-gold-bands reading and K = 4.
    for m, want in ((48, 0.9965), (83, 0.9980)):
        g6 = (["v1|v1|v1|v1"] * m + ["v2|v2|v2|v2"] * m +
              ["v3|v3|v3|v3"] * m + ["v4|v4|v4|v4"] * m)
        p6 = list(g6); p6[0] = "v9|v9|v9|v9"
        got = score(g6, p6)
        closed = 1.0 - 1.0 / (3.0 * (2 * m - 1))
        assert abs(got - closed) < 1e-9, (got, closed)
        print(f"  m={m}: one unrecognisable row of {4*m} -> {got:.4f} "
              f"(closed form {closed:.4f})")
        assert abs(got - want) < 5e-5, (got, want)

    print("metric.py: all self-tests passed")
