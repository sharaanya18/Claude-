import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
tr = pd.read_csv("dataset/public/train.csv"); ds = np.load("ds.npz")
X, Q = ds["X"], ds["Q"]; n, m, f = X.shape
groups = tr.experiment_group.values
rand = Q[:, :400].mean(1); can = Q[:, 401]; rev = Q[:, 402]; ref = Q[:, 400]
print("candidate-set stats: random %.3f canonical %.3f reverse %.3f reference %.3f | best-of-400 %.3f" % (rand.mean(), can.mean(), rev.mean(), ref.mean(), Q[:, :400].max(1).mean()))
res = {"model": [], "oracle_cands": []}
for name, tgt in [("q", lambda Qi: Qi), ("rank", lambda Qi: Qi.argsort().argsort() / (len(Qi) - 1))]:
    picks = np.zeros(n)
    for fold, (a, b) in enumerate(GroupKFold(4).split(np.arange(n), groups=groups)):
        Xa = X[a][:, :400].reshape(-1, f); ya = np.concatenate([tgt(Q[i, :400]) for i in a])
        mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6, min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Xa, ya)
        for i in b: picks[i] = Q[i, int(np.argmax(mdl.predict(X[i, :400])))]
    print(name, "pick quality (CV by scene): mean %.3f | per-fold:" % picks.mean(), [round(picks[b].mean(), 3) for a, b in GroupKFold(4).split(np.arange(n), groups=groups)], "| vs random %.3f" % rand.mean(), flush=True)
