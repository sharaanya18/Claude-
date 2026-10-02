import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from sim import *; from vest import *
tr=pd.read_csv(PUB/"train.csv")
rng=np.random.default_rng(5)
sel=[]
for g,df in tr.groupby("experiment_group"): sel+=list(rng.choice(df.index.values,30,replace=False))
def f(i):
    d=load_request(tr.iloc[i]); return v_features(d), np.log(d["moments"][:,1].mean(0)+1e-30).astype(np.float32), tr.experiment_group[i]
if __name__=="__main__":
    t=time.time()
    with Pool(4) as p: out=p.map(f,sel,chunksize=4)
    X=np.stack([o[0] for o in out]); Y=np.stack([o[1] for o in out]); G=np.array([o[2] for o in out])
    np.savez_compressed("vest_data.npz",X=X,Y=Y,G=G,sel=np.array(sel)); print("built",X.shape,time.time()-t)
