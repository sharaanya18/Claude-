"""Refined inversion: wider grid, identification intervals, censoring flags, joint ats refinement."""
import sys, json, time
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, boot

D='/home/user/Claude-/challenges/two_puffs/dataset'; W='/home/user/Claude-/challenges/two_puffs/work'
t0=time.time()
B=pd.read_csv(f'{W}/blow_indices.csv'); B=B[B.src=='blows.jsonl']
acc=B[B.acceptable==1]
sess={pid:(g.fev1.values,g.fvc.values) for pid,g in acc.groupby('pid')}
train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values

GRID=np.exp(np.arange(np.log(0.30), np.log(3.0)+1e-9, 0.0015))
LG=np.log(GRID)
out=[]
parts={}
for i,pid in enumerate(train.id):
    f,v=sess[pid]; P=boot.Participant(f,v); parts[pid]=P
    pf=P.p_fev1(GRID, spiro.FEV1_THR); pv=P.p_fvc(GRID, spiro.FVC_THR)
    ef=((pf-Y[i,0:5][None,:])**2).sum(1); ev=((pv-Y[i,5:8][None,:])**2).sum(1)
    jf,jv=int(np.argmin(ef)),int(np.argmin(ev))
    # identification interval: grid points within 1e-6 of the minimum (the objective is piecewise constant)
    okf=np.where(ef<=ef[jf]+1e-9)[0]; okv=np.where(ev<=ev[jv]+1e-9)[0]
    out.append(dict(id=pid,
        Rf=float(np.exp(LG[okf].mean())), Rv=float(np.exp(LG[okv].mean())),
        Rf_lo=GRID[okf[0]], Rf_hi=GRID[okf[-1]], Rv_lo=GRID[okv[0]], Rv_hi=GRID[okv[-1]],
        wf=float(LG[okf[-1]]-LG[okf[0]]), wv=float(LG[okv[-1]]-LG[okv[0]]),
        res_f=float(np.sqrt(ef[jf]/5)), res_v=float(np.sqrt(ev[jv]/3))))
L=pd.DataFrame(out)
print('[%.0fs] identification width (log units): fev1 %s'%(time.time()-t0, L.wf.describe()[['mean','50%','75%','90%' if False else 'max']].round(3).to_dict()))
print('  wf quantiles', L.wf.quantile([.5,.75,.9,.95,.99]).round(3).to_dict())
print('  wv quantiles', L.wv.quantile([.5,.75,.9,.95,.99]).round(3).to_dict())
print('censored-left  (Rf_lo at grid min): %d   censored-right: %d'%((L.Rf_lo<=GRID[0]+1e-9).sum(),(L.Rf_hi>=GRID[-1]-1e-9).sum()))
print('censored-left v (Rv_lo at grid min): %d   censored-right: %d'%((L.Rv_lo<=GRID[0]+1e-9).sum(),(L.Rv_hi>=GRID[-1]-1e-9).sum()))
print('Rf dist', L.Rf.quantile([.01,.05,.25,.5,.75,.95,.99]).round(3).to_dict())
print('Rv dist', L.Rv.quantile([.01,.05,.25,.5,.75,.95,.99]).round(3).to_dict())
print('corr(logRf,logRv) = %.3f'%np.corrcoef(np.log(L.Rf),np.log(L.Rv))[0,1])

rec=np.zeros_like(Y)
for i,pid in enumerate(train.id):
    P=parts[pid]
    rec[i,0:5]=P.p_fev1(np.array([L.Rf[i]]),spiro.FEV1_THR)[0]
    rec[i,5:8]=P.p_fvc(np.array([L.Rv[i]]),spiro.FVC_THR)[0]
    rec[i,8]=P.p_ats(L.Rf[i],L.Rv[i])
print('\nORACLE (margin-fitted R):', {k:round(v,4) for k,v in spiro.official_score(rec,Y).items()})

# --- local 2D refinement that also matches ats (weighted as the metric weights it) ---
rec2=rec.copy(); Rf2=L.Rf.values.copy(); Rv2=L.Rv.values.copy()
mult=np.exp(np.arange(-0.10,0.1001,0.005))
for i,pid in enumerate(train.id):
    P=parts[pid]
    cf=Rf2[i]*mult; cv=Rv2[i]*mult
    pf=P.p_fev1(cf,spiro.FEV1_THR); pv=P.p_fvc(cv,spiro.FVC_THR)
    ef=((pf-Y[i,0:5][None,:])**2).sum(1); ev=((pv-Y[i,5:8][None,:])**2).sum(1)
    best=None
    for a in range(len(mult)):
        for b in range(len(mult)):
            pa=P.p_ats(cf[a],cv[b])
            obj=ef[a]+ev[b]+(pa-Y[i,8])**2
            if best is None or obj<best[0]: best=(obj,a,b,pa)
    _,a,b,pa=best
    Rf2[i],Rv2[i]=cf[a],cv[b]
    rec2[i,0:5]=pf[a]; rec2[i,5:8]=pv[b]; rec2[i,8]=pa
print('ORACLE (joint-fitted R incl ats):', {k:round(v,4) for k,v in spiro.official_score(rec2,Y).items()})
print('per-target RMSE joint:', dict(zip(spiro.TARGETS, np.sqrt(((rec2-Y)**2).mean(0)).round(4))))
L['Rf2']=Rf2; L['Rv2']=Rv2
L.to_csv(f'{W}/latent_R.csv',index=False)
print('[%.0fs] done'%(time.time()-t0))
