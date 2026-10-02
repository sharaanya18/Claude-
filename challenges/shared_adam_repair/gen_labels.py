import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from sim import *
N_ORD = 400
tr = pd.read_csv(PUB/"train.csv")
def work(i):
    rng = np.random.default_rng(1000 + i); d = load_request(tr.iloc[i])
    orders = np.array([rng.permutation(8) for _ in range(N_ORD)]); orders = np.vstack([orders, np.array(d["reference_order"])[None], np.arange(8)[None], np.arange(7, -1, -1)[None]])
    D = np.zeros(len(orders)); lo = np.zeros((len(orders), 2))
    for k, o in enumerate(orders):
        _, D[k], lo[k] = quality(d, list(o))
    return i, orders.astype(np.int8), D, lo
if __name__ == "__main__":
    t = time.time(); out = {}
    with Pool(4) as p:
        for n, (i, o, D, lo) in enumerate(p.imap_unordered(work, range(len(tr)), chunksize=8)):
            out[i] = (o, D, lo)
            if n % 200 == 0: print(n, round(time.time() - t), flush=True)
    idx = sorted(out)
    np.savez_compressed("order_labels.npz", orders=np.stack([out[i][0] for i in idx]), D=np.stack([out[i][1] for i in idx]), losses=np.stack([out[i][2] for i in idx]))
    print("done", round(time.time() - t), flush=True)
