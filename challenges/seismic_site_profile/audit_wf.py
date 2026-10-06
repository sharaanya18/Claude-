import numpy as np, pandas as pd
from pathlib import Path
P = Path("/home/user/Claude-/challenges/seismic_site_profile/dataset/public")
tr=pd.read_csv(P/"train.csv"); te=pd.read_csv(P/"test.csv"); fo=pd.read_csv(P/"folds.csv")
z = np.load(P/"waveforms.npz")
keys = list(z.keys())
print("n keys", len(keys))
print("keys == train+test:", set(keys)==set(tr.id)|set(te.id))
a = z[keys[0]]; print("shape", a.shape, "dtype", a.dtype)
shapes=set(); 
import time
t0=time.time()
W = np.stack([z[k] for k in keys])  # memory: 1512*3*3000*4 = 54MB
print("stacked", W.shape, W.dtype, "load s", round(time.time()-t0,1), "MB", W.nbytes//10**6)
print("nan", np.isnan(W).sum(), "inf", np.isinf(W).sum())
mx = np.abs(W).max(axis=(1,2))
print("max|x| per record: min",mx.min(),"max",mx.max(), "all==1:", np.allclose(mx,1.0))
# which component holds the max
argc = np.array([np.unravel_index(np.abs(w).argmax(), w.shape)[0] for w in W])
print("component holding global max (0=E,1=N,2=Z):", np.bincount(argc)/len(argc))
# per-component rms
rms = np.sqrt((W**2).mean(axis=2))
print("per-comp rms mean", rms.mean(axis=0).round(4), "std", rms.std(axis=0).round(4))
print("H/V rms ratio: mean", (np.sqrt((rms[:,0]**2+rms[:,1]**2)/2)/rms[:,2]).mean().round(3))
# zero traces
zero = (rms<1e-8).sum(axis=0); print("near-zero comps", zero)
# constant segments / padding: count leading+trailing exact zeros
lead = np.array([(np.abs(w).max(axis=0)!=0).argmax() for w in W])
rev  = np.array([(np.abs(w[:, ::-1]).max(axis=0)!=0).argmax() for w in W])
print("leading zero samples: max", lead.max(), "nonzero frac", (lead>0).mean())
print("trailing zero samples: max", rev.max(), "nonzero frac", (rev>0).mean())
# mean offset (is it detrended?)
print("abs mean/rms per comp:", (np.abs(W.mean(axis=2))/rms).mean(axis=0).round(4))
# clipping: fraction of samples at +-1
clip = (np.abs(W)>0.999).sum(axis=(1,2))
print("samples within 0.1% of 1.0: median", np.median(clip), "max", clip.max())
# Are train and test from same event pool? check cross-correlation of a few
np.save("/tmp/claude-0/-home-user-Claude-/325ffff6-eef0-5109-829b-701b602a737a/scratchpad/Wkeys.npy", np.array(keys))
