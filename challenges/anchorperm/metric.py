"""Exact AnchorPerm row loss and (given strata) final score. Lower is better.
row loss = 0.85 * L_seq + 0.15 * (c - (1 - L_seq))**2 ;  score = 0.75 * mean(row loss) + 0.25 * max over strata of the stratum mean."""
import numpy as np


def row_loss(pred, truth, conf):
    m = len(truth)
    if len(pred) != m or len(set(pred)) != m or not (0.0 <= conf <= 1.0):
        return 1.0
    l_seq = float(np.mean([p != t for p, t in zip(pred, truth)]))
    r = 1.0 - l_seq
    return 0.85 * l_seq + 0.15 * (conf - r) ** 2


def stratum(set_size, anchor_count, transfer):
    return ("transfer" if transfer else "familiar") + ("_sparse" if anchor_count <= 3 else "_rich")


def final_score(losses, strata):
    losses = np.asarray(losses, float)
    strata = np.asarray(strata)
    worst = max(losses[strata == s].mean() for s in set(strata))
    return 0.75 * losses.mean() + 0.25 * worst
