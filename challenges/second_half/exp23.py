import numpy as np, pickle, torch
exec(open('exp21.py').read().split("for l2 in (1e-3")[0])
oofZ=np.zeros((N,6,6),np.float32)
for f in range(5):
    tri=torch.tensor(np.where(fold!=f)[0]); vai=np.where(fold==f)[0]
    th,le,lo=fit(X[tri],true_idx[tri],0.1)
    with torch.no_grad(): oofZ[vai]=Zof(th,X[vai],torch.exp(le)).numpy()
np.save('/tmp/LKZ.npy',oofZ)
print('LK OOF',round(score(np.array([hungarian(z) for z in oofZ]),Y),4))
