import json, numpy as np

def folio_score(order_clues, gold_ids, pred_ids, conf, n):
    """order_clues: clue target_ids in public order; gold_ids/pred_ids/conf aligned lists; exact challenge metric for one folio."""
    corr = np.array([p == g for p, g in zip(pred_ids, gold_ids)], float)
    a = corr.mean(); h = n // 2
    l = corr[:h].mean(); r = corr[h:].mean()
    sk = []
    for c, ok in zip(conf, corr):
        probs_other = (1 - c) / (n - 1)
        B = (c - ok) ** 2 + (n - 1) * (probs_other - 0) ** 2 if ok else (c - 0) ** 2 + (probs_other - 1) ** 2 + (n - 2) * probs_other ** 2
        sk.append(max(0.0, 1 - B / (1 - 1 / n)))
    return 0.6 * a + 0.2 * np.sqrt(l * r) + 0.2 * np.mean(sk)

def total_score(folio_rows):
    """folio_rows: list of (band, folio_score). mean over bands of mean over folios, x100."""
    bands = {}
    for b, s in folio_rows: bands.setdefault(b, []).append(s)
    return 100 * np.mean([np.mean(v) for v in bands.values()])

if __name__ == "__main__":
    n = 8; ids = [f"d{i}" for i in range(n)]
    assert abs(folio_score(ids, ids, ids, [1.0] * n, n) - 1.0) < 1e-12
    uni = folio_score(ids, ids, ids, [1 / n] * n, n)
    print("all correct conf=1/n:", uni, "(0.6+0.2+0)")
    wrong = ids[1:] + ids[:1]
    print("all wrong conf=1.0:", folio_score(ids, ids, wrong, [1.0] * n, n))
    print("all wrong conf=1/n:", folio_score(ids, ids, wrong, [1 / n] * n, n))
