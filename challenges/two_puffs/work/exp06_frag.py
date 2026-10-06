import sys, time, json
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, runner as R, spiro, dist_decode as dd, frag
t0=time.time(); dat=R.load()
Y=dat['Y']; Sf=np.load(f"{R.W}/oof_Sf.npy"); Sv=np.load(f"{R.W}/oof_Sv.npy")
gf,gv=np.load(f"{R.W}/ladder.npy")
P=np.load(f"{R.W}/oof_dist.npy")
rho=0.529
qf,qv=frag.posterior_draws(rho, M=128)
Et=np.empty(len(Y)); Et2=np.empty(len(Y))
for i in range(len(Y)):
    Et[i],Et2[i]=frag.ats_moments(dat['sessions'][i],gf,Sf[i],gv,Sv[i],qf,qv)
print('[%.0fs] sampled. analytic vs sampled E[t] corr %.5f  mean diff %.4f'%(
    time.time()-t0, np.corrcoef(P[:,8],Et)[0,1], np.mean(np.abs(P[:,8]-Et))), flush=True)
phi=np.clip(4*(Et-Et2),0,1)
tf=4*Y[:,8]*(1-Y[:,8])
print('Somers D of true fragility vs:')
print('  4p(1-p), p=analytic decode : %.4f  -> FDS %.4f'%(spiro.somers_d(tf,4*P[:,8]*(1-P[:,8])), spiro.fds(P[:,8],Y[:,8])))
print('  4p(1-p), p=sampled E[t]    : %.4f'%spiro.somers_d(tf,4*Et*(1-Et)))
print('  phi = E[4t(1-t)]           : %.4f  -> FDS %.4f'%(spiro.somers_d(tf,phi),(1+spiro.somers_d(tf,phi))/2))
print('  Var(t|x) spread: mean %.4f sd %.4f'%(np.mean(Et2-Et**2), np.std(Et2-Et**2)), flush=True)
ptil=frag.frag_to_prob(phi, Et)
print('\nsharpening blend sweep (w=0 -> pure E[t]):')
best=None
for w in np.arange(0,1.01,0.1):
    pm=(1-w)*P[:,8]+w*ptil
    Q=P.copy(); Q[:,8]=np.clip(pm,0,1)
    s=spiro.official_score(Q,Y)
    print('  w=%.1f score %.4f rcs %.4f fds %.4f'%(w,s['score'],s['rcs'],s['fds']))
    if best is None or s['score']>best[0]: best=(s['score'],w)
print('best w=%.1f score %.4f'%(best[1],best[0]))
np.save(f"{R.W}/oof_ats_moments.npy", np.column_stack([Et,Et2]))
