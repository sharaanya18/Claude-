import sys, json, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, harness, decode as dec
W=harness.W; D=harness.D
L=pd.read_csv(f'{W}/latent_R.csv'); train=pd.read_csv(f'{D}/train.csv')
Y=pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values
accvals=pickle.load(open(f'{W}/accvals.pkl','rb'))
sessions=[dec.PreSession(*accvals[p]) for p in train.id]
yf=np.log(L.Rf2.values); yv=np.log(L.Rv2.values)
of,ov=np.load(f'{W}/oof_latent.npy').T
Pg=np.load(f'{W}/oof_gen.npy'); Pb=np.load(f'{W}/oof_gbdt.npy')

print('exact-zero / exact-one share per target:')
for j,t in enumerate(spiro.TARGETS):
    print('  %-8s zeros %.3f ones %.3f  truth mean %.4f | gen mean %.4f | gbdt mean %.4f'%(
        t,(Y[:,j]==0).mean(),(Y[:,j]==1).mean(),Y[:,j].mean(),Pg[:,j].mean(),Pb[:,j].mean()))
print('\nper-target Brier (lower better):')
for j,t in enumerate(spiro.TARGETS):
    print('  %-8s gen %.5f   gbdt %.5f   delta %+.5f'%(t,((Pg[:,j]-Y[:,j])**2).mean(),((Pb[:,j]-Y[:,j])**2).mean(),
        ((Pg[:,j]-Y[:,j])**2).mean()-((Pb[:,j]-Y[:,j])**2).mean()))
print('\nprediction spread: truth sd / gen sd / gbdt sd')
for j,t in enumerate(spiro.TARGETS):
    print('  %-8s %.4f  %.4f  %.4f'%(t,Y[:,j].std(),Pg[:,j].std(),Pb[:,j].std()))

print('\n=== is the latent target itself the problem? score of decode using TRUE latent with the OOF-residual law ===')
law=dec.ResidualLaw(yf-of, yv-ov)
print('  true mean + empirical law:', {k:round(v,4) for k,v in spiro.official_score(dec.decode_all(sessions,yf,yv,law),Y).items()})

print('\n=== tight symmetric gaussian law sweep on OOF means (in-sample law choice, indicative) ===')
rng=np.random.RandomState(3); z=rng.multivariate_normal([0,0],[[1,0.36],[0.36,1]],6000)
for sf in [0.02,0.04,0.06,0.08,0.11]:
    for sv in [0.04,0.07,0.10,0.14,0.19]:
        law=dec.ResidualLaw(z[:,0]*sf, z[:,1]*sv)
        s=spiro.official_score(dec.decode_all(sessions,of,ov,law),Y)
        print('  sf=%.2f sv=%.2f -> score %.4f rcs %.4f fds %.4f'%(sf,sv,s['score'],s['rcs'],s['fds']))
