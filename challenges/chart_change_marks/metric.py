"""Exact re-implementation of the challenge grader (from the description only)."""
import json
import numpy as np

TOL = 5.0


def chart_score(pred, truth):
    """pred/truth: lists of dicts {x, p}. Greedy closest-pair matching within TOL px,
    each truth mark used once (and each predicted mark once). RPS credit per matched pair."""
    n_p, n_t = len(pred), len(truth)
    if n_p == 0 and n_t == 0:
        return 1.0
    pairs = []
    for i, a in enumerate(pred):
        for j, b in enumerate(truth):
            d = abs(a["x"] - b["x"])
            if d <= TOL:
                pairs.append((d, i, j))
    pairs.sort()
    ui, uj, credit = set(), set(), 0.0
    for d, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j)
        P = np.cumsum(pred[i]["p"])[:2]
        R = np.cumsum(truth[j]["p"])[:2]
        credit += 1.0 - float(((P - R) ** 2).sum()) / 2.0
    return 2.0 * credit / (n_p + n_t)


def cell_scores(df_true, preds, cells):
    """df_true: marks (lists) per row; preds: marks per row; cells: cell tag per row."""
    ch = np.array([chart_score(p, t) for p, t in zip(preds, df_true)])
    emp = np.array([chart_score([], t) for t in df_true])
    cells = np.asarray(cells)
    out = {}
    for c in np.unique(cells):
        m = cells == c
        b = emp[m].mean()
        out[c] = (ch[m].mean() - b) / (1 - b)
    return out, float(np.mean(list(out.values())))


if __name__ == "__main__":
    t = [{"x": 100, "p": [0, 0, 1]}]
    assert chart_score([], []) == 1.0 and chart_score([{"x": 5, "p": [0, 0, 1]}], []) == 0.0
    assert abs(chart_score([{"x": 103, "p": [0, 0, 1]}], t) - 1.0) < 1e-9
    assert chart_score([{"x": 106, "p": [0, 0, 1]}], t) == 0.0
    assert abs(chart_score([{"x": 100, "p": [1, 0, 0]}], t) - 0.0) < 1e-9  # P=(1,1) vs R=(0,0)
    # hedge [.27,.21,.52] vs firm: P=(.27,.48) R=(0,0) -> 1-(.0729+.2304)/2
    assert abs(chart_score([{"x": 100, "p": [.27, .21, .52]}], t) - (1 - (.0729 + .2304) / 2)) < 1e-9
    # one hit one extra: 2*1/(2+1)
    assert abs(chart_score([{"x": 100, "p": [0, 0, 1]}, {"x": 300, "p": [0, 0, 1]}], t) - 2 / 3) < 1e-9
    print("metric tests ok")
