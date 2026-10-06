import sys, json, time
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro

D = '/home/user/Claude-/challenges/two_puffs/dataset'
t0 = time.time()
train = pd.read_csv(f'{D}/train.csv')
test = pd.read_csv(f'{D}/test.csv')
print('train', train.shape, 'test', test.shape)
print(train.dtypes.to_dict())

tj = train['target_json'].apply(json.loads)
Y = pd.DataFrame(list(tj))[spiro.TARGETS]
print('\n=== target stats ===')
print(Y.describe().T[['mean','std','min','25%','50%','75%','max']].round(4))
# granularity
allv = Y.values.ravel()
print('multiples of 1/4000:', np.allclose(allv*4000, np.round(allv*4000)))
print('frac exactly 0:', (allv==0).mean().round(4), 'frac exactly 1:', (allv==1).mean().round(4))
print('\nmonotonic fev1 arm:', (np.diff(Y[['fev1_00','fev1_05','fev1_10','fev1_15','fev1_20']].values,axis=1)<=1e-12).all())
print('monotonic fvc arm:', (np.diff(Y[['fvc_00','fvc_05','fvc_10']].values,axis=1)<=1e-12).all())
print('\ncorr fev1_10 vs fvc_05:', np.corrcoef(Y.fev1_10, Y.fvc_05)[0,1].round(4))
print('corr matrix:\n', Y.corr().round(3))
print('\nats: mean %.4f  >0.5 frac %.4f' % (Y.ats.mean(), (Y.ats>0.5).mean()))
print('fragile (0.05<ats<0.95): %.4f  near-cert-neg(<=0.05): %.4f  near-cert-pos(>=0.95): %.4f' % (
    ((Y.ats>0.05)&(Y.ats<0.95)).mean(), (Y.ats<=0.05).mean(), (Y.ats>=0.95).mean()))

print('\n=== demographics ===')
for c in ['age_years','height_cm','weight_kg','bmi','n_blows','n_acceptable']:
    print(c, 'train', train[c].describe()[['mean','std','min','50%','max']].round(2).to_dict(),
          '| test mean %.2f' % test[c].mean())
print('sex train', train.sex.value_counts(normalize=True).round(3).to_dict(), 'test', test.sex.value_counts(normalize=True).round(3).to_dict())
print('eth train', train.ethnicity.value_counts(normalize=True).round(3).to_dict())
print('eth test', test.ethnicity.value_counts(normalize=True).round(3).to_dict())
print('n_acceptable dist train', train.n_acceptable.value_counts().sort_index().to_dict())
print('n_acceptable dist test ', test.n_acceptable.value_counts().sort_index().to_dict())
print('n_blows dist train', train.n_blows.value_counts().sort_index().to_dict())

# constant baselines under the exact metric (on train, in-sample constant = optimistic but a reference)
Yv = Y.values
print('\n=== constant baselines on TRAIN (in-sample) ===')
print('0.5 everywhere        :', spiro.official_score(np.full_like(Yv,0.5), Yv))
m = Yv.mean(0)
print('train-mean everywhere :', spiro.official_score(np.tile(m,(len(Yv),1)), Yv))
print('train-mean vector:', dict(zip(spiro.TARGETS, m.round(4))))
print('[%.0fs] done' % (time.time()-t0))
