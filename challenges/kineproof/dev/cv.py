"""Grouped cross-person CV driver for the temporal model. One experiment per run.

Usage:
  python3 cv.py <exp_name> [key=value ...]

Writes a row to reports/experiments.csv and the OOF predictions to
working/oof_<exp_name>.npy so later experiments can blend without retraining.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from kp_data import load_motion, load_targets, score
from kp_feat import anthro_features, frame_features, mirror, mirror_targets
from kp_model import Standardiser, make_prior, metric_loss, train_one
from groups import anthro, folds_by_person, person_groups

OUT = Path(__file__).resolve().parents[1]
WORK = OUT / "working"
REP = OUT / "reports"
T0 = time.time()
PRIOR_CH = [3, 4, 5]          # com_a31 Newtonian estimate, already in target units

DEF = dict(width=96, blocks=6, k=5, drop=0.05, epochs=40, bs=32, lr=3e-3,
           wd=1e-4, huber=0.02, phys_residual=1, mirror=1, seeds=1,
           folds=5, split_seed=0, ema=0.0, noise=0.0)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)


def get_cache():
    """Build (and cache) features for the original and mirrored motion.

    Uncompressed .npy so the cache saves/loads in ~1 s instead of ~90 s
    (savez_compressed on 400 MB is single-threaded zlib and dominated the run).
    """
    keys = ["FF", "AF", "Y", "mFF", "mAF", "mY"]
    paths = {k: WORK / f"cache_{k}.npy" for k in keys}
    paths["lab"] = WORK / "cache_lab.npy"
    paths["ids"] = WORK / "cache_ids.npy"
    if all(p.exists() for p in paths.values()):
        d = {k: np.load(p) for k, p in paths.items()}   # 418 MB total, fits RAM
        return (d["FF"], d["AF"], d["Y"], d["lab"], d["ids"],
                d["mFF"], d["mAF"], d["mY"])
    log("building cache (original + mirrored) ...")
    ids, X = load_motion("train")
    _, Y = load_targets(ids)
    FF, AF = frame_features(X), anthro_features(X)
    log("  original features done")
    Xm = mirror(X)
    mFF, mAF = frame_features(Xm), anthro_features(Xm)
    mY = mirror_targets(Y)
    log("  mirrored features done")
    lab, _, _, _ = person_groups(anthro(X), 32)
    vals = dict(FF=FF, AF=AF, Y=Y.astype(np.float32), mFF=mFF, mAF=mAF,
                mY=mY.astype(np.float32), lab=lab, ids=np.array(ids))
    for k, p in paths.items():
        np.save(p, vals[k])
    log("cached working/cache_*.npy")
    return (vals["FF"], vals["AF"], vals["Y"], vals["lab"], vals["ids"],
            vals["mFF"], vals["mAF"], vals["mY"])


def verify_mirror(FF, AF, Y, mFF, mAF, mY):
    """The mirror must be a physically exact symmetry; check it, don't assume."""
    # force_x and force_y unchanged, force_z negated
    assert np.allclose(mY[:, 0], Y[:, 0]) and np.allclose(mY[:, 1], Y[:, 1])
    assert np.allclose(mY[:, 2], -Y[:, 2])
    # the Newtonian prior channels must transform the same way as the target
    log(f"  mirror check: prior f_x identical? "
        f"{np.abs(mFF[:,3]-FF[:,3]).max():.2e}  "
        f"f_y identical? {np.abs(mFF[:,4]-FF[:,4]).max():.2e}  "
        f"f_z negated? {np.abs(mFF[:,5]+FF[:,5]).max():.2e}")


def run(name, cfg):
    FF, AF, Y, lab, ids, mFF, mAF, mY = get_cache()
    verify_mirror(FF, AF, Y, mFF, mAF, mY)
    N = len(ids)
    nf = int(cfg["folds"])
    fold = folds_by_person(lab, nf, int(cfg["split_seed"]))
    log(f"exp={name} cfg={ {k: v for k, v in cfg.items()} }")
    log(f"N={N} people={len(np.unique(lab))} folds={[int((fold==f).sum()) for f in range(nf)]}")

    nseeds = int(cfg["seeds"])
    OOF = np.zeros((nseeds, N, 3, Y.shape[2]), dtype=np.float64)
    PR = np.ascontiguousarray(FF[:, PRIOR_CH, :])
    mPR = np.ascontiguousarray(mFF[:, PRIOR_CH, :])
    for fi in range(nf):
        tr = np.where(fold != fi)[0]
        va = np.where(fold == fi)[0]
        st = Standardiser(FF, AF, tr)
        FFtr, AFtr = st.f(FF[tr]), st.a(AF[tr])
        FFva, AFva = st.f(FF[va]), st.a(AF[va])
        Ptr, Pva = PR[tr], PR[va]
        mp = None
        if int(cfg["mirror"]):
            mp = (st.f(mFF[tr]), st.a(mAF[tr]), mY[tr],
                  mPR[tr])
        for si in range(nseeds):
            log(f"  fold {fi} seed {si}: train {len(tr)} val {len(va)}"
                f"{' (+mirror)' if mp is not None else ''}")
            vp, _, hist = train_one(
                FFtr, AFtr, Y[tr], FFva, AFva, Y[va], Ptr, Pva,
                width=int(cfg["width"]), blocks=int(cfg["blocks"]), k=int(cfg["k"]),
                drop=float(cfg["drop"]), epochs=int(cfg["epochs"]), bs=int(cfg["bs"]),
                lr=float(cfg["lr"]), wd=float(cfg["wd"]), huber=float(cfg["huber"]),
                phys_residual=bool(int(cfg["phys_residual"])),
                seed=1000 * si + fi, mirror_pack=mp, log=log,
                ema_decay=float(cfg["ema"]))
            OOF[si, va] = vp

    P = OOF.mean(axis=0)
    tot, pax, per = score(P, Y, per_axis=True)
    pf = [per[fold == f].mean() for f in range(nf)]
    # per-person score, for the worst-person term the playbook asks for
    pp = {int(c): float(per[lab == c].mean()) for c in np.unique(lab)}
    log("")
    log(f"RESULT {name}: TOTAL {tot:.4f}   x {pax[0]:.4f}  y {pax[1]:.4f}  z {pax[2]:.4f}")
    log(f"  per-fold  [{' '.join(f'{v:.4f}' for v in pf)}]  mean {np.mean(pf):.4f} sd {np.std(pf):.4f}")
    log(f"  per-person: worst {min(pp.values()):.4f}  best {max(pp.values()):.4f}  "
        f"median {np.median(list(pp.values())):.4f}")
    log(f"  worst 5 people: " +
        " ".join(f"{k}:{v:.3f}" for k, v in sorted(pp.items(), key=lambda t: t[1])[:5]))

    np.save(WORK / f"oof_{name}.npy", P)
    np.save(WORK / f"oofseeds_{name}.npy", OOF)
    np.save(WORK / f"fold_{name}.npy", fold)

    REP.mkdir(exist_ok=True)
    fcsv = REP / "experiments.csv"
    new = not fcsv.exists()
    with open(fcsv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["experiment_id", "total", "x", "y", "z", "fold_scores",
                        "fold_sd", "worst_person", "best_person", "config",
                        "runtime_s", "notes"])
        w.writerow([name, f"{tot:.4f}", f"{pax[0]:.4f}", f"{pax[1]:.4f}",
                    f"{pax[2]:.4f}", " ".join(f"{v:.4f}" for v in pf),
                    f"{np.std(pf):.4f}", f"{min(pp.values()):.4f}",
                    f"{max(pp.values()):.4f}", json.dumps(cfg),
                    f"{time.time()-T0:.0f}", ""])
    log(f"logged to reports/experiments.csv; OOF -> working/oof_{name}.npy")
    return tot, pax, per, fold


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "e000"
    cfg = dict(DEF)
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown key {k}; known {sorted(cfg)}"
        cfg[k] = v
    run(name, cfg)
