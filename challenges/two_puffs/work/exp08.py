"""Experiment 8: the full stack, one increment at a time, on identical folds."""
import sys, time, json
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, runner as R, spiro, harness, pipeline as PL, dist_decode as dd

t0=time.time(); dat=R.load()
Y=dat['Y']; Xtr=dat['Xtr']; yf=dat['yf']; yv=dat['yv']; names=dat['names']; sess=dat['sessions']
wgt=1.0+4.0*Y[:,8]*(1.0-Y[:,8])
strat=harness.strata(Y, dat['train'].n_acceptable.values)
fold=harness.folds(strat,5,0)
gf=PL.ladder(yf); gv=PL.ladder(yv)

def log(*a): print(*a, flush=True)

# ---- stage 4: survival OOF, single variant vs 3-variant ensemble
for tag, var in (('single', PL.VARIANTS[:1]), ('ens3', PL.VARIANTS)):
    Sf=np.zeros((len(yf),len(gf))); Sv=np.zeros((len(yv),len(gv)))
    for k in range(5):
        tr=fold!=k; va=fold==k
        Sf[va]=PL.predict_survival(PL.fit_survival(Xtr[tr],yf[tr],gf,names,var),Xtr[va],gf)
        Sv[va]=PL.predict_survival(PL.fit_survival(Xtr[tr],yv[tr],gv,names,var),Xtr[va],gv)
    rho=float(np.corrcoef(dd.norm_ppf(PL.pit(Sf,gf,yf)),dd.norm_ppf(PL.pit(Sv,gv,yv)))[0,1])
    P=PL.decode(sess,gf,Sf,gv,Sv,rho)
    s=spiro.official_score(P,Y)
    log('[%4.0fs] survival %-6s rho %.3f -> score %.4f rcs %.4f fds %.4f'%(time.time()-t0,tag,rho,s['score'],s['rcs'],s['fds']))
    if tag=='ens3':
        np.save(f'{R.W}/e8_Sf.npy',Sf); np.save(f'{R.W}/e8_Sv.npy',Sv); np.save(f'{R.W}/e8_P.npy',P)
        np.save(f'{R.W}/e8_rho.npy',np.array([rho])); best_S=(Sf,Sv,rho,P)

Sf,Sv,rho,P=best_S

# ---- PIT calibration diagnostics: is the conditional latent law honest?
for nm,(S,g,y) in (('fev1',(Sf,gf,yf)),('fvc',(Sv,gv,yv))):
    u=PL.pit(S,g,y)
    h=np.histogram(u,bins=10,range=(0,1))[0]/len(u)
    log('PIT %-4s decile shares %s  (uniform = 0.100)'%(nm,np.round(h,3)))

# ---- stage 5: structural calibration of the latent law (shift / spread)
log('\nlatent-law shift/spread sweep:')
bestcal=(s['score'],0.0,1.0)
for shift in [-0.01,0.0,0.01]:
    for spread in [0.85,0.95,1.0,1.1,1.25]:
        _,gf2=PL.calibrate_survival(Sf,gf,shift,spread); _,gv2=PL.calibrate_survival(Sv,gv,shift,spread)
        Pc=PL.decode(sess,gf2,Sf,gv2,Sv,rho)
        sc=spiro.official_score(Pc,Y)
        if sc['score']>bestcal[0]: bestcal=(sc['score'],shift,spread)
        log('  shift %+.2f spread %.2f -> %.4f (rcs %.4f fds %.4f)'%(shift,spread,sc['score'],sc['rcs'],sc['fds']))
log('best calibration %s'%(bestcal,))
_,gf2=PL.calibrate_survival(Sf,gf,bestcal[1],bestcal[2]); _,gv2=PL.calibrate_survival(Sv,gv,bestcal[1],bestcal[2])
P=PL.decode(sess,gf2,Sf,gv2,Sv,rho)
log('calibrated decode: %.4f'%spiro.official_score(P,Y)['score'])

# ---- stage 8: direct nine-target member and blend
Pd=np.zeros((len(Y),9))
for k in range(5):
    tr=fold!=k
    Pd[fold==k]=PL.predict_direct(PL.fit_direct(Xtr[tr],Y[tr],names),Xtr[fold==k])
np.save(f'{R.W}/e8_Pd.npy',Pd)
log('\ndirect nine-target member: %.4f'%spiro.official_score(Pd,Y)['score'])
log('blend sweep (generative vs direct):')
for w in np.arange(0,0.61,0.1):
    Q=np.clip((1-w)*P+w*Pd,0,1); sc=spiro.official_score(Q,Y)
    log('  w_direct %.1f -> %.4f (rcs %.4f fds %.4f)'%(w,sc['score'],sc['rcs'],sc['fds']))
log('[%.0fs] done'%(time.time()-t0))
