import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, decode as dec
W=harness.W; D=harness.D
X=pd.read_csv(f'{W}/X_raw.csv',index_col=0); P=pd.read_csv(f'{W}/pool_raw.csv',index_col=0)
L=pd.read_csv(f'{W}/latent_R.csv'); train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
sessions=[dec.PreSession(*accvals[p]) for p in train.id]
yf=np.log(L.Rf2.values); yv=np.log(L.Rv2.values)
of,ov=np.load(f'{W}/oof_latent.npy').T

print('=== CHECK A: decode with TRUE latent and a near-point-mass law (should ~= oracle) ===')
law=dec.ResidualLaw(np.random.RandomState(0).randn(2000)*1e-4, np.random.RandomState(1).randn(2000)*1e-4)
Pr=dec.decode_all(sessions, yf, yv, law)
print(' ', {k:round(v,4) for k,v in spiro.official_score(Pr,Y).items()})

print('\n=== latent target distributions ===')
for nm,y in (('logRf',yf),('logRv',yv)):
    print(nm,'mean %.4f sd %.4f'%(y.mean(),y.std()), 'pct', np.round(np.percentile(y,[0.5,1,5,25,50,75,95,99,99.5]),3))
print('logRv <= log(0.4):', (yv<=np.log(0.4)).sum(), ' logRf outside [log.7,log1.6]:', ((yf<np.log(0.7))|(yf>np.log(1.6))).sum())
print('\n=== OOF residual distributions ===')
rf=yf-of; rv=yv-ov
for nm,r in (('res_f',rf),('res_v',rv)):
    print(nm,'sd %.4f  IQR/1.349 %.4f  pct'%(r.std(),(np.percentile(r,75)-np.percentile(r,25))/1.349),
          np.round(np.percentile(r,[1,5,25,50,75,95,99]),3), 'kurtosis %.1f'%(((r-r.mean())**4).mean()/r.var()**2))
print('corr(res) %.3f'%np.corrcoef(rf,rv)[0,1])

print('\n=== CHECK B: no-feature generative baseline (constant latent mean, marginal law) ===')
for trim in [(None,None),((0.7,1.7),(0.5,1.7))]:
    yf2=yf.copy(); yv2=yv.copy()
    if trim[0]: yf2=np.clip(yf2,np.log(trim[0][0]),np.log(trim[0][1])); yv2=np.clip(yv2,np.log(trim[1][0]),np.log(trim[1][1]))
    law=dec.ResidualLaw(yf2-yf2.mean(), yv2-yv2.mean())
    Pr=dec.decode_all(sessions, np.full(len(yf),yf2.mean()), np.full(len(yf),yv2.mean()), law)
    print(' trim',trim[0],{k:round(v,4) for k,v in spiro.official_score(Pr,Y).items()})
print('train-mean constant:', {k:round(v,4) for k,v in spiro.official_score(np.tile(Y.mean(0),(len(Y),1)),Y).items()})

print('\n=== CHECK C: in-sample decode with fitted means + trimmed gaussian law, scale sweep ===')
rfc=np.clip(rf,-0.4,0.4); rvc=np.clip(rv,-0.4,0.4)
rng=np.random.RandomState(7)
for s in [0.6,0.8,1.0,1.2]:
    g=rng.multivariate_normal([0,0],[[rfc.var()*s*s, 0.3*s*s*rfc.std()*rvc.std()],[0.3*s*s*rfc.std()*rvc.std(), rvc.var()*s*s]],4000)
    law=dec.ResidualLaw(g[:,0],g[:,1])
    Pr=dec.decode_all(sessions, of, ov, law)
    print('  gauss scale %.1f'%s, {k:round(v,4) for k,v in spiro.official_score(Pr,Y).items()})
