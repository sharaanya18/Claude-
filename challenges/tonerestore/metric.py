"""Exact anchored-contour metric from CHALLENGE.md. P/T: lists of (offset:int, contour:str)."""
import json


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def row_score(P, T):
    P = [(int(o), str(c)) for o, c in P]
    T = [(int(o), str(c)) for o, c in T]
    if not P:
        return 0.0
    A = 2 * len(set(P) & set(T)) / (len(P) + len(T))
    cp, ct = [c for _, c in P], [c for _, c in T]
    C = 1 - _lev(cp, ct) / max(len(cp), len(ct))
    E = 1.0 if P == T else 0.0
    return (6 * A + 3 * C + E) / 10


def score(preds, refs):
    return sum(row_score(p, t) for p, t in zip(preds, refs)) / len(refs)


def parse(s):
    return [(int(o), c) for o, c in json.loads(s)]


if __name__ == "__main__":
    T = [(4, "3131"), (6, "53"), (9, "53")]
    assert abs(score([T], [T]) - 1.0) < 1e-12
    assert score([[]], [T]) == 0.0
    # one wrong contour, same offsets: A=2/3, C=2/3, E=0 -> (4+2)/10
    P = [(4, "3131"), (6, "31"), (9, "53")]
    assert abs(row_score(P, T) - (6 * 2 / 3 + 3 * 2 / 3) / 10) < 1e-12
    # right contours wrong offsets: A=0, C=1, E=0 -> .3
    assert abs(row_score([(1, "3131"), (2, "53"), (3, "53")], T) - 0.3) < 1e-12
    print("metric ok")
