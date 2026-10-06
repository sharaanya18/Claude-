"""Honest estimate of the two selection steps, at zero extra compute.

Both the ensemble subset and the joint-prior strength were chosen by looking at
the same two splits they are reported on, which is how a cross-validated number
drifts above what a leaderboard returns. Here the choice is made on ONE split
and scored on the OTHER, both ways round, so the reported figure includes the
cost of having chosen. Everything runs off the cached out-of-fold
probabilities, so it needs no refitting.
"""
import numpy as np, warnings, itertools
warnings.filterwarnings("ignore")
import cv, decode

Z = np.load("oof_pool.npz")
D = cv.load(); Y = D["Y"]
SPLITS = [k.split("|")[1] for k in Z.files if k.startswith("fold|")]
ALL = ["ordgb", "hgb", "ord", "et", "lr", "pls", "ldak"]
LAMS = [None, 0.3, 0.5, 0.7, 1.0, 1.3]
# candidate subsets: every family, the ranked prefixes, and the structural trio
SUBSETS = {"all7": ALL, "top3": ["ordgb", "ord", "et"],
           "top2": ["ordgb", "ord"], "top4": ["ordgb", "ord", "et", "hgb"],
           "trees+ord": ["ordgb", "hgb", "et", "ord"],
           "ordgb_only": ["ordgb"]}


def evaluate(s, names, lam):
    fold = Z[f"fold|{s}"]
    P = np.mean([Z[f"{s}|{m}"] for m in names], axis=0)
    Pred = np.zeros((len(Y), 4), dtype=int)
    for f in sorted(set(fold)):
        va = fold == f; tr = ~va
        Q = P[va]
        if lam is not None:
            prof, cnt = np.unique(Y[tr], axis=0, return_counts=True)
            Q = decode.joint_prior_decode(Q, prof, lam, cnt.astype(float))
        Pred[va] = Q.argmax(axis=2)
    return cv.score_oof(Y, Pred)[0]

table = {(k, l): {s: evaluate(s, v, l) for s in SPLITS}
         for k, v in SUBSETS.items() for l in LAMS}

print("full grid (chosen and reported on the same splits - optimistic):")
print(f"  {'subset':11s} " + "  ".join(f"lam={str(l):>4}" for l in LAMS))
for k in SUBSETS:
    print(f"  {k:11s} " + "  ".join(
        f"{np.mean(list(table[(k, l)].values())):8.4f}" for l in LAMS))

print("\nselect on one split, score on the other (includes the cost of choosing):")
tot = []
for s in SPLITS:
    other = [x for x in SPLITS if x != s]
    best = max(table, key=lambda kl: np.mean([table[kl][o] for o in other]))
    got = table[best][s]
    tot.append(got)
    print(f"  chose {best[0]:11s} lam={str(best[1]):>4} on {other} -> scored {got:.4f} on {s}")
print(f"  honest mean: {np.mean(tot):.4f}")

print("\nfor comparison, the fixed a-priori choice (no selection at all):")
for k, l in (("all7", None), ("all7", 0.5), ("top3", 1.0)):
    print(f"  {k} lam={l}: mean {np.mean(list(table[(k, l)].values())):.4f} "
          f"per split {[round(table[(k, l)][s], 4) for s in SPLITS]}")
