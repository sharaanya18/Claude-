"""Experiment 10: feature-block ablations on identical folds (pool reference, loop shape, scatter)."""
import sys, time, json, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, genlatent as gl, dist_decode as dd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.special import ndtr

W=harness.W; D=harness.D; t0=time.time()
X=pd.read_csv(f'{W}/X_raw.csv',index_col=0); P=pd.read_csv(f'{W}/pool_raw.csv',index_col=0)
train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
models=harness.pool_reference(P); XD=harness.design(X,models)
LAM=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
strat=harness.strata(Y, train.n_acceptable.values); fold=harness.folds(strat,5,0)
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8])
QP=np.array([0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.875,0.925,0.96,0.98,0.99,0.995])
PAR=dict(objective='binary',learning_rate=0.04,num_leaves=15,min_data_in_leaf=340,feature_fraction=0.45,
         bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=4,seed=1,
         deterministic=True,force_row_wise=True)

try:
    LB=pd.read_csv(f'{W}/loop_blows.csv',index_col=0); have_loop=True
except Exception:
    have_loop=False

sess=[gl.GenSession(*accvals[p], LAM) for p in train.id]
scale=float(np.mean([s.if_.mean() for s in sess]))
dgrid=np.round(np.arange(-0.8,2.0,0.002)*scale,8)
lt=np.load(f'{W}/lam{LAM:.1f}_latent.npy') if __import__('os').path.exists(f'{W}/lam{LAM:.1f}_latent.npy') else None
if lt is None:
    df=np.empty(len(Y)); dv=np.empty(len(Y))
    for i in range(len(Y)): df[i],dv[i],_=gl.invert(sess[i],Y[i],dgrid)
    np.save(f'{W}/lam{LAM:.1f}_latent.npy',np.column_stack([df,dv]))
else:
    df,dv=lt.T
print('[%.0fs] latent ready (LAM=%.1f)'%(time.time()-t0,LAM), flush=True)

def ladder(y):
    g=np.maximum.accumulate(np.quantile(y,QP))
    for i in range(1,len(g)):
        if g[i]<=g[i-1]: g[i]=g[i-1]+1e-5
    return g
def fit(Xa,ya,grid,names):
    K=len(grid); Z=np.hstack([np.repeat(Xa,K,axis=0),np.tile(grid,len(ya))[:,None]])
    lab=(np.repeat(ya,K)>=np.tile(grid,len(ya))).astype(np.float64)
    p=dict(PAR); p['monotone_constraints']=[0]*Xa.shape[1]+[-1]
    return lgb.train(p,lgb.Dataset(Z,lab,feature_name=list(names)+['level']),num_boost_round=400)
def pred(m,Xa,grid):
    K=len(grid); Z=np.hstack([np.repeat(Xa,K,axis=0),np.tile(grid,len(Xa))[:,None]])
    return m.predict(Z).reshape(len(Xa),K)
def pit(S,grid,y,rp):
    o=np.empty(len(y))
    for i in range(len(y)):
        xs,ys=gl.survival_curve(grid,S[i],rp); o[i]=1.0-np.interp(y[i],xs,ys)
    return np.clip(o,1e-4,1-1e-4)

def run(XDv, tag):
    Xtr=XDv.loc[train.id].values.astype(np.float64); names=list(XDv.columns)
    gf=ladder(df); gv=ladder(dv); ramp=0.25*(gf[-1]-gf[0])
    Sf=np.zeros((len(Y),len(gf))); Sv=np.zeros((len(Y),len(gv)))
    for k in range(5):
        tr=fold!=k; va=fold==k
        Sf[va]=pred(fit(Xtr[tr],df[tr],gf,names),Xtr[va],gf)
        Sv[va]=pred(fit(Xtr[tr],dv[tr],gv,names),Xtr[va],gv)
    rho=float(np.corrcoef(dd.norm_ppf(pit(Sf,gf,df,ramp)),dd.norm_ppf(pit(Sv,gv,dv,ramp)))[0,1])
    C=dd.Copula(rho)
    Pr=np.array([gl.decode_participant(sess[i],gf,Sf[i],gv,Sv[i],C,ramp,dd.norm_ppf) for i in range(len(Y))])
    s=spiro.official_score(Pr,Y)
    rs=np.random.RandomState(20261006); z1=rs.standard_normal(128)
    z2=rho*z1+np.sqrt(max(1e-9,1-rho**2))*rs.standard_normal(128); q1,q2=ndtr(z1),ndtr(z2)
    Et=np.empty(len(Y)); Et2=np.empty(len(Y))
    for i,gs in enumerate(sess):
        xs,ys=gl.survival_curve(gf,Sf[i],ramp); uf=np.interp(q1,1-ys,xs)
        xs,ys=gl.survival_curve(gv,Sv[i],ramp); uv=np.interp(q2,1-ys,xs)
        t=((uf[:,None]>=gs.tf()[None,:])|(uv[:,None]>=gs.tv()[None,:])).astype(np.float64)@gs.pw
        Et[i]=t.mean(); Et2[i]=(t**2).mean()
    phi=np.clip(4*(Et-Et2),0,1)
    a=np.abs(Pr[:,8]-0.5); side=np.where(Pr[:,8]>=0.5,1.0,-1.0); mm=np.zeros(len(Y))
    for k in range(5):
        tr=fold!=k; va=fold==k
        iso=IsotonicRegression(increasing=False,out_of_bounds='clip'); iso.fit(phi[tr],a[tr],sample_weight=wgt[tr])
        mm[va]=iso.predict(phi[va])
    gr=(phi-phi.min())/max(phi.max()-phi.min(),1e-9)
    Q=Pr.copy(); Q[:,8]=np.clip(0.5+side*np.clip(mm-1e-6*gr,0,0.5),0,1)
    s2=spiro.official_score(Q,Y)
    print('%-26s p=%3d  decode %.4f | +frag %.4f (rcs %.4f fds %.4f) [%.0fs]'%(
        tag,Xtr.shape[1],s['score'],s2['score'],s2['rcs'],s2['fds'],time.time()-t0), flush=True)
    return s2['score']

poolcols=[c for c in XD.columns if c.startswith('ref_') or c.startswith('pp_')]
scat=[c for c in XD.columns if any(c.endswith(e) for e in ('_std','_cv','_rng','_rng_rel','_iqr','_mad','_mad_rel','_best2','_best2_rel','_best_med_rel'))]
run(XD, 'A full')
run(XD.drop(columns=poolcols), 'B no pool reference')
run(XD.drop(columns=scat), 'C no within-session scatter')
if have_loop:
    XL=pd.concat([XD, LB.loc[XD.index].astype(float)], axis=1)
    run(XL, 'D + loop shape')
core=['fev1_max','fvc_max','pef_max','ratio_med','fev1_cv','fvc_cv','pef_cv','n_acceptable','n_blows',
      'age','male','height','weight','bmi']+poolcols
run(XD[[c for c in core if c in XD.columns]], 'E compact')
