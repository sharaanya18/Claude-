import sys, time
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, runner as R, spiro
t0=time.time(); dat=R.load()
QP=np.array([0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.875,0.925,0.96,0.98,0.99,0.995])
cands={
 'quantile17': (R.quantile_ladder(dat['yf'],QP), R.quantile_ladder(dat['yv'],QP)),
 'uniform31':  (R.uniform_ladder(-0.30,0.60,0.03),)*1 + (R.uniform_ladder(-0.30,0.60,0.03),),
 'uniform19':  (R.uniform_ladder(-0.25,0.65,0.05), R.uniform_ladder(-0.25,0.65,0.05)),
 'uniform46':  (R.uniform_ladder(-0.35,0.55,0.02), R.uniform_ladder(-0.35,0.55,0.02)),
}
for nm,(gf,gv) in cands.items():
    r=R.run_cv(dat,gf,gv,R.PAR_BASE,400,seed=0)
    fs=R.fold_scores(r['P'],dat['Y'],r['fold'])
    print('%-11s K=%2d  score %.4f rcs %.4f fds %.4f  rho %.2f  folds %s [%.0fs]'%(
        nm,len(gf),r['score']['score'],r['score']['rcs'],r['score']['fds'],r['rho'],np.round(fs,3),time.time()-t0))
