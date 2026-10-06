"""Experiment 12: better ranking statistics for the fragility half.

FDS is a concordance, so the quantity to rank by is not necessarily E[frag|x].  For a pairwise
concordance the ideal order puts i above j when P(frag_i > frag_j) > P(frag_j > frag_i); the
tractable approximation is the Borda / expected-marginal-rank statistic

    b_i = E_{frag ~ posterior_i} [ F(frag) ],      F = population marginal CDF of fragility,

which differs from E[frag] exactly when posterior SHAPES differ between participants.  Fragility
is a deterministic function of the latent, frag(delta) = 4 t(delta) (1 - t(delta)), so the whole
posterior of fragility is available from the latent posterior.  Several statistics are compared.
"""
import sys, time
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs')
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
from pathlib import Path
import numpy as np, pandas as pd
import solution as S
from scipy.special import ndtr

W='/home/user/Claude-/challenges/two_puffs/work'
PUB=Path('/home/user/Claude-/challenges/two_puffs/dataset')
t0=time.time()
train, test, sample, X, P, Y, fev, fvc = S.build_tables(PUB)
coefs=S.fit_reference(P); XD=S.design_matrix(X,coefs); names=list(XD.columns)
Xtr=XD.loc[train['id']].values.astype(np.float64)
sess=[S.Session(fev[p],fvc[p],S.LAM) for p in train['id']]
import os
if os.path.exists(f'{W}/verify_cache.npz'):
    z=np.load(f'{W}/verify_cache.npz'); df,dv=z['df'],z['dv']
else:
    scale=float(np.mean([s.if_.mean() for s in sess])); dgrid=np.arange(-0.8,2.0,0.002)*scale
    df=np.empty(len(Y)); dv=np.empty(len(Y))
    for i in range(len(Y)): df[i],dv[i],_=S.invert_latent(sess[i],Y[i],dgrid)
    np.savez(f'{W}/verify_cache.npz', df=df, dv=dv)
gf=S.build_ladder(df); gv=S.build_ladder(dv); ramp=S.RAMP_FRAC*float(gf[-1]-gf[0])
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8]); tfrag=4*Y[:,8]*(1-Y[:,8])
fold=S.make_folds(Y, train['n_acceptable'].values)
print('[%.0fs] setup'%(time.time()-t0), flush=True)

Sf=np.zeros((len(Y),len(gf))); Sv=np.zeros((len(Y),len(gv))); Pd=np.zeros((len(Y),9))
for k in range(S.N_FOLDS):
    tr=fold!=k; va=fold==k
    Sf[va]=S.predict_survival(S.fit_survival(Xtr[tr],df[tr],gf,names),Xtr[va],gf)
    Sv[va]=S.predict_survival(S.fit_survival(Xtr[tr],dv[tr],gv,names),Xtr[va],gv)
    Pd[va]=S.predict_direct(S.fit_direct(Xtr[tr],Y[tr],names),Xtr[va])
    print('  fold %d [%.0fs]'%(k,time.time()-t0), flush=True)
rho=float(np.corrcoef(S.norm_ppf(S.pit_values(Sf,gf,df,ramp)),S.norm_ppf(S.pit_values(Sv,gv,dv,ramp)))[0,1])
Pg=S.decode_all(sess,gf,Sf,gv,Sv,rho,ramp)
wg=np.arange(0,0.41,0.05)
ws=[S.official_score(np.clip((1-w)*Pg+w*Pd,0,1),Y)['score'] for w in wg]
wd=float(wg[int(np.argmax(ws))]); Po=np.clip((1-wd)*Pg+wd*Pd,0,1)
np.save(f'{W}/e12_Po.npy',Po); np.save(f'{W}/e12_Sf.npy',Sf); np.save(f'{W}/e12_Sv.npy',Sv)
np.save(f'{W}/e12_meta.npy',np.array([rho,wd]))
print('base OOF: gen %.4f  +direct %.4f (w=%.2f)'%(S.official_score(Pg,Y)['score'],S.official_score(Po,Y)['score'],wd), flush=True)

# --- full posterior of t (and hence of fragility) per participant ---
M=256
rng=np.random.RandomState(20261006)
z1=rng.standard_normal(M); z2=rho*z1+np.sqrt(max(1e-9,1-rho**2))*rng.standard_normal(M)
qf,qv=ndtr(z1),ndtr(z2)
T=np.empty((len(Y),M))
for i,s in enumerate(sess):
    xs,ys=S.survival_curve(gf,Sf[i],ramp); uf=np.interp(qf,1-ys,xs)
    xs,ys=S.survival_curve(gv,Sv[i],ramp); uv=np.interp(qv,1-ys,xs)
    T[i]=((uf[:,None]>=s.t_fev1()[None,:])|(uv[:,None]>=s.t_fvc()[None,:])).astype(np.float64)@s.pw
FR=4*T*(1-T)
print('[%.0fs] posterior of fragility built'%(time.time()-t0), flush=True)

Et=T.mean(1)
stats={
 '4p(1-p) of blended ats': 4*Po[:,8]*(1-Po[:,8]),
 'phi = E[frag]':          FR.mean(1),
 'posterior median frag':  np.median(FR,axis=1),
 'Borda E[F(frag)]':       None,
 'P(frag > 0.25)':         (FR>0.25).mean(1),
 'P(frag > 0.50)':         (FR>0.50).mean(1),
 'P(frag > 0.75)':         (FR>0.75).mean(1),
 '1 - P(frag < 0.02)':     1.0-(FR<0.02).mean(1),
}
pool=np.sort(FR.ravel())
stats['Borda E[F(frag)]']=np.searchsorted(pool,FR,side='right').mean(1)/len(pool)
print('\nSomers D of true fragility vs each ranking statistic:')
res={}
for nm,g in stats.items():
    d=S.somers_d(tfrag,g); res[nm]=d
    print('  %-24s D=%.4f  FDS=%.4f'%(nm,d,(1+d)/2))
# cross-fitted transfer for the best few
print('\ncross-fitted ordering transfer (final score):')
best=None
for nm,g in sorted(stats.items(), key=lambda kv:-res[kv[0]])[:5]:
    lo,hi=float(g.min()),float(g.max()); Pf=Po.copy()
    for k in range(S.N_FOLDS):
        tr=fold!=k; va=fold==k
        iso=S.fit_fragility_map(g[tr],Po[tr,8],wgt[tr])
        Pf[va,8]=S.apply_fragility_map(iso,g[va],Po[va,8],lo,hi)
    s=S.official_score(Pf,Y)
    print('  %-24s score %.4f (rcs %.4f fds %.4f)'%(nm,s['score'],s['rcs'],s['fds']), flush=True)
    if best is None or s['score']>best[0]: best=(s['score'],nm)
# combinations of the two leading statistics by rank average
r1=pd.Series(stats['phi = E[frag]']).rank().values
r2=pd.Series(stats['Borda E[F(frag)]']).rank().values
r3=pd.Series(stats['P(frag > 0.25)']).rank().values
for nm,g in [('rank(phi)+rank(Borda)',r1+r2),('rank(phi)+rank(P>0.25)',r1+r3),('all three ranks',r1+r2+r3)]:
    lo,hi=float(g.min()),float(g.max()); Pf=Po.copy()
    for k in range(S.N_FOLDS):
        tr=fold!=k; va=fold==k
        iso=S.fit_fragility_map(g[tr],Po[tr,8],wgt[tr])
        Pf[va,8]=S.apply_fragility_map(iso,g[va],Po[va,8],lo,hi)
    s=S.official_score(Pf,Y)
    print('  %-24s score %.4f (rcs %.4f fds %.4f)  D=%.4f'%(nm,s['score'],s['rcs'],s['fds'],S.somers_d(tfrag,g)), flush=True)
np.save(f'{W}/e12_FR.npy',FR)
print('[%.0fs] done'%(time.time()-t0), flush=True)
