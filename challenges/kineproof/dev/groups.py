"""Phase 3: derive and VALIDATE approximate person groups from anthropometry.

The description states the hidden split is person-disjoint, that 32 people form
train, and that the person key is an HMAC-SHA-256 digest under an evaluator-only
key -- i.e. person identity is deliberately unrecoverable from ids. Any
validation that is not person-disjoint is optimistic and useless here, so we
must APPROXIMATE persons from motion content.

Legitimacy: segment lengths are a deterministic function of the public TRAIN
marker arrays. No id, no row order, no file metadata, no provenance, no test
statistic is used. (Test groups are never needed -- we never validate on test.)

Writes: working/groups.npz, reports/split_audit_raw.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from kp_data import MI, PUBLIC, load_motion

OUT = Path(__file__).resolve().parents[1]
WORK = OUT / "working"
L = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


# Rigid-body segments only: pairs whose within-trial sd was far below their
# across-trial sd in the audit (joint-spanning pairs like thigh/shank move, so
# we take the MEDIAN over time, which is robust to the swing).
RIGID = [
    ("pelvis_width", "R.ASIS", "L.ASIS"),
    ("pelvis_depth_R", "R.ASIS", "R.PSIS"),
    ("pelvis_depth_L", "L.ASIS", "L.PSIS"),
    ("crest_width", "R.Iliac.Crest", "L.Iliac.Crest"),
    ("gtr_width", "R.GTR", "L.GTR"),
    ("thigh_R", "R.GTR", "R.Knee"), ("thigh_L", "L.GTR", "L.Knee"),
    ("shank_R", "R.Knee", "R.Ankle"), ("shank_L", "L.Knee", "L.Ankle"),
    ("TT_ankle_R", "R.TT", "R.Ankle"), ("TT_ankle_L", "L.TT", "L.Ankle"),
    ("foot_R", "R.Heel", "R.MT1"), ("foot_L", "L.Heel", "L.MT1"),
    ("foot5_R", "R.Heel", "R.MT5"), ("foot5_L", "L.Heel", "L.MT5"),
    ("forefoot_R", "R.MT1", "R.MT5"), ("forefoot_L", "L.MT1", "L.MT5"),
    ("asis_gtr_R", "R.ASIS", "R.GTR"), ("asis_gtr_L", "L.ASIS", "L.GTR"),
    ("knee_HF_R", "R.Knee", "R.HF"), ("knee_HF_L", "L.Knee", "L.HF"),
    # cross-segment, still rigid within a person
    ("asis_knee_R", "R.ASIS", "R.Knee"), ("asis_knee_L", "L.ASIS", "L.Knee"),
    ("psis_crest_R", "R.PSIS", "R.Iliac.Crest"),
    ("psis_crest_L", "L.PSIS", "L.Iliac.Crest"),
]


def anthro(A: np.ndarray) -> np.ndarray:
    """(N,T,22,3) -> (N, n_seg) median-over-time segment lengths, metres."""
    out = np.empty((len(A), len(RIGID)), dtype=np.float64)
    for j, (_, m1, m2) in enumerate(RIGID):
        d = np.linalg.norm(A[:, :, MI[m1], :] - A[:, :, MI[m2], :], axis=-1)
        out[:, j] = np.median(d, axis=1)
    return out


def anthro_within_sd(A: np.ndarray) -> np.ndarray:
    """per-trial within-trial sd of each segment length -> measurement noise floor"""
    out = np.empty((len(A), len(RIGID)), dtype=np.float64)
    for j, (_, m1, m2) in enumerate(RIGID):
        d = np.linalg.norm(A[:, :, MI[m1], :] - A[:, :, MI[m2], :], axis=-1)
        out[:, j] = d.std(axis=1)
    return out


def person_groups(F: np.ndarray, n_people: int = 32):
    """Cluster trials into n_people groups in standardised anthropometric space.

    Average-linkage agglomerative clustering. Deterministic, no randomness.
    Returns (labels, standardised features, mean/sd used).
    """
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = (F - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, 0.0)
    lk = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(lk, n_people, criterion="maxclust") - 1
    return lab, Z, D, lk


def folds_by_person(lab, k=5, seed=0):
    """Assign whole PERSONS to folds, greedily balancing trial counts.

    Deterministic given (lab, k, seed). Whole persons move together, which is
    what mirrors the hidden person-disjoint split.
    """
    rng = np.random.default_rng(seed)
    people = np.unique(lab)
    cnt = np.array([(lab == c).sum() for c in people])
    order = people[np.argsort(-cnt + rng.random(len(cnt)) * 0.1)]
    load = np.zeros(k)
    pf = {}
    for c in order:
        f = int(np.argmin(load))
        pf[c] = f
        load[f] += (lab == c).sum()
    return np.array([pf[c] for c in lab])


if __name__ == "__main__":
    ids, X = load_motion("train")
    F = anthro(X)
    Wsd = anthro_within_sd(X)
    names = [r[0] for r in RIGID]

    p("=" * 78)
    p("PERSON-GROUP DERIVATION AND VALIDATION (train only)")
    p("=" * 78)
    p(f"trials {len(ids)}   segments {len(RIGID)}   description says 32 people")

    lab, Z, D, lk = person_groups(F, 32)
    sz = np.bincount(lab)
    p(f"\nk=32 cluster sizes: min {sz.min()} max {sz.max()} median {int(np.median(sz))} "
      f"mean {sz.mean():.1f}  (1093/32 = {1093/32:.1f})")
    p(f"  sizes sorted: {sorted(sz.tolist())}")

    # ---- TEST 1: are clusters as internally consistent as ONE person is? ----
    p("\n" + "-" * 78)
    p("TEST 1  within-cluster segment sd vs the measurement noise floor")
    p("-" * 78)
    p("If a cluster is one person, the spread of its trials' segment lengths should")
    p("be close to the within-trial measurement sd (marker jitter / soft tissue),")
    p("not to the population spread.")
    p(f"\n{'segment':<16} {'pop sd mm':>10} {'within-clu':>11} {'within-trial':>13} "
      f"{'ratio pop/clu':>14}")
    rows = []
    for j, nm in enumerate(names):
        pop = F[:, j].std() * 1000
        wc = np.mean([F[lab == c, j].std() for c in range(32) if (lab == c).sum() > 1]) * 1000
        wt = Wsd[:, j].mean() * 1000
        rows.append((pop, wc, wt))
        p(f"{nm:<16} {pop:10.2f} {wc:11.2f} {wt:13.2f} {pop/max(wc,1e-9):14.2f}")
    pop_m = np.mean([r[0] for r in rows]); wc_m = np.mean([r[1] for r in rows])
    wt_m = np.mean([r[2] for r in rows])
    p(f"{'MEAN':<16} {pop_m:10.2f} {wc_m:11.2f} {wt_m:13.2f} {pop_m/wc_m:14.2f}")
    p(f"\nInterpretation: within-cluster spread {wc_m:.2f} mm vs population {pop_m:.2f} mm")
    p(f"  -> clusters are {pop_m/wc_m:.1f}x tighter than the population.")
    p(f"  within-cluster ({wc_m:.2f} mm) vs within-trial noise ({wt_m:.2f} mm): "
      f"{'same order -> clusters behave like single subjects' if wc_m < 4*wt_m else 'clusters still mix subjects'}")

    # ---- TEST 2: is 32 the right k? silhouette-style gap scan ----
    p("\n" + "-" * 78)
    p("TEST 2  how sharply does k=32 stand out?")
    p("-" * 78)
    Dinf = D.copy(); np.fill_diagonal(Dinf, np.inf)
    p(f"{'k':>4} {'n':>4} {'min':>5} {'max':>5} {'within':>8} {'between':>9} "
      f"{'ratio':>7} {'singletons':>11}")
    for k in [8, 16, 24, 28, 30, 32, 34, 36, 40, 48, 64, 96]:
        lb = fcluster(lk, k, criterion="maxclust") - 1
        s = np.bincount(lb)
        wi, be = [], []
        for c in range(len(s)):
            m = lb == c
            if m.sum() > 1:
                sub = D[np.ix_(m, m)]
                wi.append(sub[np.triu_indices(m.sum(), 1)].mean())
            be.append(D[np.ix_(m, ~m)].mean())
        p(f"{k:>4} {len(s):>4} {s.min():>5} {s.max():>5} {np.mean(wi):8.3f} "
          f"{np.mean(be):9.3f} {np.mean(be)/np.mean(wi):7.2f} {int((s==1).sum()):>11}")
    p("\nA ratio that stops improving past k=32 means extra clusters only split")
    p("real people; a ratio still climbing means 32 clusters still merge people.")

    # ---- TEST 3: cophenetic gap -- is there a natural threshold? ----
    p("\n" + "-" * 78)
    p("TEST 3  merge-height profile of the dendrogram")
    p("-" * 78)
    h = lk[:, 2]
    p("merge heights near the 32-cluster cut (index from the top):")
    p("  " + " ".join(f"{h[-i]:.3f}" for i in range(1, 46)))
    cut_lo, cut_hi = h[len(h) - 32], h[len(h) - 31]
    p(f"\nheight just below the k=32 cut: {h[len(h)-33]:.4f}")
    p(f"height AT the k=32 cut          : {cut_lo:.4f}")
    p(f"height just above (k=31)        : {cut_hi:.4f}")
    p(f"relative jump at the cut        : {cut_hi/max(cut_lo,1e-9):.3f}x")

    # ---- TEST 4: does the grouping actually change what CV reports? ----
    p("\n" + "-" * 78)
    p("TEST 4  leakage gauge: nearest-neighbour distance, random vs grouped folds")
    p("-" * 78)
    rng = np.random.default_rng(0)
    rand_fold = rng.integers(0, 5, len(ids))
    # grouped: whole clusters to folds, balanced by size
    order = np.argsort(-np.bincount(lab))
    load = np.zeros(5); gf = np.zeros(32, dtype=int)
    for c in order:
        f = int(np.argmin(load)); gf[c] = f; load[f] += (lab == c).sum()
    grp_fold = gf[lab]
    for nm, fold in [("random 5-fold", rand_fold), ("person-grouped 5-fold", grp_fold)]:
        ds = []
        for f in range(5):
            v, t = fold == f, fold != f
            ds.append(D[np.ix_(v, t)].min(axis=1).mean())
        p(f"  {nm:<24} mean val->train NN distance = {np.mean(ds):.4f}")
    p("  A much larger distance under grouped folds confirms that random folds")
    p("  place near-replicate trials of the same walker on both sides.")
    p(f"\n  grouped fold sizes (trials): {[int((grp_fold==f).sum()) for f in range(5)]}")
    p(f"  grouped fold sizes (people) : {[int(len(set(lab[grp_fold==f]))) for f in range(5)]}")

    np.savez_compressed(WORK / "groups.npz", ids=np.array(ids), lab=lab,
                        anthro=F, names=np.array(names), grp_fold=grp_fold)
    with open(OUT / "reports" / "split_audit_raw.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    p("\nwrote working/groups.npz and reports/split_audit_raw.txt")
