import json, numpy as np, pandas as pd
from pathlib import Path
import sys
PUB = Path(__file__).resolve().parent / "dataset" / "public"
sys.path.insert(0, str(PUB))
import learner as L


def load_request(row, with_labels=True):
    z = np.load(PUB / row["experiment_path"], allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["request"] = json.loads(row["request"])
    d["x_batches"] = np.stack([L.features(b) for b in d["batch_images"]])
    d["x_monitor"] = L.features(d["monitor_images"])
    d["x_diag"] = np.stack([L.features(b) for b in d["diagnostic_images"]])
    if with_labels and "prediction" in row and isinstance(row["prediction"], str):
        pred = json.loads(row["prediction"])
        d["reference_order"] = pred["reference_order"]
        m = np.load(PUB / pred["moment_path"], allow_pickle=False)["moments"]
        d["moments"] = m
    return d


def states(d, moments=None):
    m = d["moments"] if moments is None else moments
    t = int(d["step"])
    return [(d["weights"].copy(), m[b, 0].copy(), m[b, 1].copy(), t) for b in range(2)]


def quality(d, order, moments=None):
    D, losses = L.assess(states(d, moments), order, d["x_batches"], d["batch_labels"], d["x_monitor"], d["monitor_labels"],
                         [list(c) for c in d["continuations"]])
    lim = d["request"]["divergence_limit"]
    cap = d["request"]["capability_limit"]
    if (losses > cap).any():
        return 0.0, D, losses
    return lim / max(D, lim), D, losses
