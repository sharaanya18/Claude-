import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, itertools, time
from multiprocessing import Pool
from sim import *
tr = pd.read_csv(PUB/"train.csv")
PERMS = list(itertools.permutations(range(8)))
def work(args):
    i, lo, hi = args
    d = load_request(tr.iloc[i]); out = np.zeros(hi-lo)
    for k in range(lo, hi): out[k-lo] = quality(d, PERMS[k])[0]
    return out
if __name__ == "__main__":
    for i in [int(a) for a in sys.argv[1:]]:
        t = time.time(); chunks = [(i, a, min(a+2520, 40320)) for a in range(0, 40320, 2520)]
        with Pool(4) as p: q = np.concatenate(p.map(work, chunks))
        np.save(f"exh_{i}.npy", q)
        d = load_request(tr.iloc[i]); ref = tuple(d["reference_order"]); rq = q[PERMS.index(ref)]
        print(i, "share of orders with quality>=0.999:", round((q >= 0.999).mean(), 4), "| share quality 0:", round((q == 0).mean(), 3),
              "| mean q:", round(q.mean(), 3), "| max q:", round(q.max(), 3), "| ref q:", round(rq, 3), "| ref percentile:", round((q < rq - 1e-9).mean(), 4), f"[{time.time()-t:.0f}s]", flush=True)
