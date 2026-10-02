import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, time
from feats import *
tr = pd.read_csv(PUB / "train.csv"); lab = np.load("order_labels.npz")
orders, D, losses = lab["orders"].astype(int), lab["D"], lab["losses"]
t = time.time(); Xs = []; Qs = []; RF = []
for i in range(len(tr)):
    d = load_request(tr.iloc[i]); rf = request_features(d); req = d["request"]
    q = req["divergence_limit"] / np.maximum(D[i], req["divergence_limit"]); q = q * (losses[i].max(1) <= req["capability_limit"])
    Xs.append(order_features(orders[i], rf).astype(np.float32)); Qs.append(q); RF.append(rf["r"])
    if i % 400 == 0: print(i, round(time.time() - t), flush=True)
np.savez_compressed("ds.npz", X=np.stack(Xs), Q=np.stack(Qs), orders=orders, D=D, losses=losses)
print("saved", np.stack(Xs).shape, round(time.time() - t))
