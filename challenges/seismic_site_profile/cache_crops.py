"""Multi-crop features. 239 labelled stations is very little, so each record is
also described by overlapping sub-windows of its own valid span: extra TRAINING
rows with the same label (augmentation of a real sample, not synthetic data) and,
at inference, several views of the SAME record whose predictions are averaged.
Everything stays inside one record - no pooling across records."""
import numpy as np, time
from pathlib import Path
import features

NCROP = 3
FRAC = 0.62


def crops(w):
    nz = np.abs(w).max(axis=0) > 0
    last = int(np.nonzero(nz)[0][-1]) + 1 if nz.any() else w.shape[1]
    last = max(last, features.NPERSEG + 256)
    L = int(last * FRAC)
    if L < features.NPERSEG:
        L = min(last, features.NPERSEG)
    offs = np.linspace(0, last - L, NCROP).astype(int)
    return [w[:, o:o + L] for o in offs]


if __name__ == "__main__":
    P = Path("dataset/public")
    z = np.load(P / "waveforms.npz")
    ids = list(z.keys())
    t0 = time.time()
    out = []
    for k in ids:
        w = z[k]
        for c in crops(w):
            d = features.extract(c)
            out.append(np.concatenate([d[g] for g in features.GROUPS]))
    X = np.array(out, dtype=np.float32).reshape(len(ids), NCROP, -1)
    print("Xcrop", X.shape, "s", round(time.time() - t0, 1), "finite", np.isfinite(X).all())
    np.savez_compressed("crop_cache.npz", X=X, ids=np.array(ids))
