"""Repeated-CV verification of the FINAL pipeline, by importing solution.py itself.

Three split seeds x 5 stratified folds, the exact official metric, every post-hoc constant
cross-fitted inside the fold structure, plus an untouched sanity holdout.
"""
import sys, time, json, os
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs')
from pathlib import Path
import numpy as np, pandas as pd
import solution as S

W='/home/user/Claude-/challenges/two_puffs/work'
PUB=Path('/home/user/Claude-/challenges/two_puffs/dataset')
t0=time.time()
cache=f'{W}/verify_cache.npz'
train, test, sample, X, P, Y, fev, fvc = S.build_tables(PUB)
coefs=S.fit_reference(P); XD=S.design_matrix(X, coefs); names=list(XD.columns)
Xtr=XD.loc[train['id']].values.astype(np.float64)
sess=[S.Session(fev[p], fvc[p], S.LAM) for p in train['id']]
if os.path.exists(cache):
    z=np.load(cache); df, dv = z['df'], z['dv']
else:
    scale=float(np.mean([s.if_.mean() for s in sess]))
    dgrid=np.arange(-0.8,2.0,0.002)*scale
    df=np.empty(len(Y)); dv=np.empty(len(Y))
    for i in range(len(Y)): df[i],dv[i],_=S.invert_latent(sess[i],Y[i],dgrid)
    np.savez(cache, df=df, dv=dv)
print('[%.0fs] setup done'%(time.time()-t0), flush=True)
gf=S.build_ladder(df); gv=S.build_ladder(dv); ramp=S.RAMP_FRAC*float(gf[-1]-gf[0])
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8])

def one_seed(seed):
    fold=S.make_folds(Y, train['n_acceptable'].values, seed=seed)
    Sf=np.zeros((len(Y),len(gf))); Sv=np.zeros((len(Y),len(gv))); Pd=np.zeros((len(Y),9))
    for k in range(S.N_FOLDS):
        tr=fold!=k; va=fold==k
        Sf[va]=S.predict_survival(S.fit_survival(Xtr[tr],df[tr],gf,names),Xtr[va],gf)
        Sv[va]=S.predict_survival(S.fit_survival(Xtr[tr],dv[tr],gv,names),Xtr[va],gv)
        Pd[va]=S.predict_direct(S.fit_direct(Xtr[tr],Y[tr],names),Xtr[va])
    rho=float(np.corrcoef(S.norm_ppf(S.pit_values(Sf,gf,df,ramp)),
                          S.norm_ppf(S.pit_values(Sv,gv,dv,ramp)))[0,1])
    Pg=S.decode_all(sess,gf,Sf,gv,Sv,rho,ramp)
    wg=np.arange(0,0.41,0.05)
    ws=[S.official_score(np.clip((1-w)*Pg+w*Pd,0,1),Y)['score'] for w in wg]
    wd=float(wg[int(np.argmax(ws))])
    Po=np.clip((1-wd)*Pg+wd*Pd,0,1)
    T=S.ats_posterior(sess,gf,Sf,gv,Sv,rho,ramp)
    phi,_=S.borda_fragility(T); lo,hi=float(phi.min()),float(phi.max())
    Pf=Po.copy()
    for k in range(S.N_FOLDS):
        tr=fold!=k; va=fold==k
        iso=S.fit_fragility_map(phi[tr],Po[tr,8],wgt[tr])
        Pf[va,8]=S.apply_fragility_map(iso,phi[va],Po[va,8],lo,hi)
    return fold, Pg, Po, Pf, rho, wd

rows=[]
allP={}
for seed in [42, 7, 2024]:
    fold,Pg,Po,Pf,rho,wd=one_seed(seed)
    sg=S.official_score(Pg,Y); so=S.official_score(Po,Y); sf=S.official_score(Pf,Y)
    fs=np.array([S.official_score(Pf[fold==k],Y[fold==k])['score'] for k in range(S.N_FOLDS)])
    print('seed %-5d rho %.3f w_dir %.2f | gen %.4f | +direct %.4f | +frag %.4f (rcs %.4f fds %.4f) | folds %s mean %.4f sd %.4f worst %.4f [%.0fs]'%(
        seed,rho,wd,sg['score'],so['score'],sf['score'],sf['rcs'],sf['fds'],np.round(fs,4),fs.mean(),fs.std(),fs.min(),time.time()-t0), flush=True)
    rows.append(dict(seed=seed,gen=sg['score'],direct=so['score'],final=sf['score'],
                     rcs=sf['rcs'],fds=sf['fds'],fold_mean=fs.mean(),fold_sd=fs.std(),fold_worst=fs.min(),
                     rho=rho,w_dir=wd))
    allP[seed]=(fold,Pf)
R=pd.DataFrame(rows); R.to_csv(f'{W}/verify_seeds.csv',index=False)
print('\nacross seeds: final %.4f +- %.4f  (min %.4f max %.4f)'%(R.final.mean(),R.final.std(),R.final.min(),R.final.max()))
np.save(f'{W}/verify_final_oof.npy', allP[42][1])
# sanity holdout: 20% of participants never used for any selection step in this script
rs=np.random.RandomState(999); idx=rs.permutation(len(Y)); hold=idx[:int(0.2*len(Y))]
fold42,Pf42=allP[42]
print('sanity holdout (20%% of participants, seed 999): %.4f   rest: %.4f'%(
    S.official_score(Pf42[hold],Y[hold])['score'], S.official_score(Pf42[np.setdiff1d(idx,hold)],Y[np.setdiff1d(idx,hold)])['score']))
print('[%.0fs] done'%(time.time()-t0), flush=True)
