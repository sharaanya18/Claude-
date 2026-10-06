"""Shared KineProof data access + the exact official metric.

Dev-side module (NOT the submitted solution). Keeps one definition of:
  - how motion arrays and target waveforms are loaded,
  - the marker name order,
  - the official score.

Compliance note for the whole dev tree: TEST arrays are loaded only for
structural validation (shape/dtype/finiteness) and for final inference.
No statistic is ever computed over test rows and fed back into a model,
a feature, a hyperparameter or a constant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC = Path(__file__).resolve().parents[1] / "dataset_public"

T = 256          # frames per trial
N_MARKERS = 22
AXES = ("force_x", "force_y", "force_z")

# zero-based marker order, verbatim from the challenge description
MARKERS = [
    "R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "L.Iliac.Crest", "R.Iliac.Crest",
    "R.GTR", "R.Knee", "R.HF", "R.TT", "R.Ankle", "R.Heel", "R.MT1", "R.MT5",
    "L.GTR", "L.Knee", "L.HF", "L.TT", "L.Ankle", "L.Heel", "L.MT1", "L.MT5",
]
MI = {m: i for i, m in enumerate(MARKERS)}

# left/right mirror permutation: index i -> index of the contralateral marker
MIRROR = [MI[("L." + m[2:]) if m.startswith("R.") else ("R." + m[2:])] for m in MARKERS]


def manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(PUBLIC / f"{split}.csv")


def load_motion(split: str, ids=None) -> tuple[list[str], np.ndarray]:
    """Return (ids, array of shape (N, T, 22, 3) float32) in manifest order."""
    man = manifest(split)
    if ids is not None:
        man = man.set_index("sample_id").loc[list(ids)].reset_index()
    out = np.empty((len(man), T, N_MARKERS, 3), dtype=np.float32)
    for k, rel in enumerate(man["motion_file"]):
        out[k] = np.load(PUBLIC / rel)
    return man["sample_id"].tolist(), out


def load_targets(ids=None) -> tuple[list[str], np.ndarray]:
    """Return (ids, array of shape (N, 3, T) float64), axis order x, y, z."""
    tg = pd.read_csv(PUBLIC / "train_targets.csv")
    if ids is not None:
        tg = tg.set_index("sample_id").loc[list(ids)].reset_index()
    out = np.empty((len(tg), 3, T), dtype=np.float64)
    for k in range(len(tg)):
        for a, col in enumerate(AXES):
            v = json.loads(tg[col].iloc[k])
            out[k, a] = v
    return tg["sample_id"].tolist(), out


# ---------------------------------------------------------------- metric ----
def axis_skill(pred: np.ndarray, truth: np.ndarray) -> float:
    """Official per-axis skill, transcribed from the challenge's grader source.

        baseline_error = sum(|truth|)
        error          = sum(|pred - truth|)
        return  float(error == 0) if baseline_error == 0 else max(0, 1 - error/baseline)
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    baseline_error = np.abs(truth).sum()
    error = np.abs(pred - truth).sum()
    if baseline_error == 0:
        return float(error == 0)
    return max(0.0, 1.0 - error / baseline_error)


def row_valid(pred_row: np.ndarray) -> bool:
    """A row contributes zero entirely if any submitted axis is malformed."""
    a = np.asarray(pred_row, dtype=float)
    return a.shape == (3, T) and bool(np.isfinite(a).all())


def score(pred: np.ndarray, truth: np.ndarray, per_axis: bool = False):
    """Official score. pred/truth shape (N, 3, T).

    Averages axis_skill over x/y/z then over trials. A row whose submitted
    axes are malformed/non-finite contributes zero (all three axes zero).
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    assert pred.shape == truth.shape and pred.shape[1:] == (3, T)
    n = pred.shape[0]
    per = np.zeros((n, 3))
    for i in range(n):
        if not row_valid(pred[i]):
            continue                      # malformed row -> zero, others still score
        for a in range(3):
            per[i, a] = axis_skill(pred[i, a], truth[i, a])
    if per_axis:
        return per.mean(), per.mean(axis=0), per
    return per.mean()
