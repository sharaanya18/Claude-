"""Invert the published targets for a latent per-participant response multiplier (Rf, Rv).

If the generating process is 'post session blows look like the pre session blows scaled by R',
then fitting the two scalars per participant must reproduce the 9 probabilities almost exactly,
and the resulting score must land near the documented oracle (0.9116).  That is the test.
"""
import sys, json, time
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro, boot

D = '/home/user/Claude-/challenges/two_puffs/dataset'
W = '/home/user/Claude-/challenges/two_puffs/work'
t0 = time.time()

B = pd.read_csv(f'{W}/blow_indices.csv')
B = B[B.src == 'blows.jsonl']
acc = B[B.acceptable == 1]
sess = {pid: (g.fev1.values, g.fvc.values) for pid, g in acc.groupby('pid')}

train = pd.read_csv(f'{D}/train.csv')
Y = pd.DataFrame(list(train.target_json.apply(json.loads)))[spiro.TARGETS].values

GRID = np.round(np.arange(0.50, 2.2001, 0.0025), 6)   # candidate response multipliers

parts, Rf, Rv, resid_f, resid_v = {}, [], [], [], []
for pid in train.id:
    f, v = sess[pid]
    P = boot.Participant(f, v)
    parts[pid] = P
    parts[pid] = P
    pf = P.p_fev1(GRID, spiro.FEV1_THR)            # (G,5)
    pv = P.p_fvc(GRID, spiro.FVC_THR)              # (G,3)
    i = len(Rf)
    ef = ((pf - Y[i, 0:5][None, :]) ** 2).sum(1)
    ev = ((pv - Y[i, 5:8][None, :]) ** 2).sum(1)
    jf, jv = int(np.argmin(ef)), int(np.argmin(ev))
    Rf.append(GRID[jf]); Rv.append(GRID[jv])
    resid_f.append(ef[jf] / 5.0); resid_v.append(ev[jv] / 3.0)

Rf = np.array(Rf); Rv = np.array(Rv)
resid_f = np.array(resid_f); resid_v = np.array(resid_v)
print('[%.0fs] inverted. Rf: %s' % (time.time()-t0, np.round(np.percentile(Rf,[1,5,25,50,75,95,99]),3)))
print('Rv:', np.round(np.percentile(Rv,[1,5,25,50,75,95,99]),3))
print('per-event residual RMSE fev1 arm %.4f  fvc arm %.4f' % (np.sqrt(resid_f.mean()), np.sqrt(resid_v.mean())))
print('frac participants with fev1 residual rmse < 0.02: %.3f' % (np.sqrt(resid_f) < 0.02).mean())

# reconstruct all nine and score
rec = np.zeros_like(Y)
for i, pid in enumerate(train.id):
    P = parts[pid]
    rec[i, 0:5] = P.p_fev1(np.array([Rf[i]]), spiro.FEV1_THR)[0]
    rec[i, 5:8] = P.p_fvc(np.array([Rv[i]]), spiro.FVC_THR)[0]
    rec[i, 8] = P.p_ats(Rf[i], Rv[i])
print('\n=== ORACLE CHECK: fitted (Rf,Rv) reconstruction vs truth, on train ===')
print(spiro.official_score(rec, Y))
per = np.sqrt(((rec - Y) ** 2).mean(0))
print('per-target RMSE:', dict(zip(spiro.TARGETS, per.round(4))))

np.save(f'{W}/Rf.npy', Rf); np.save(f'{W}/Rv.npy', Rv)
pd.DataFrame({'id': train.id, 'Rf': Rf, 'Rv': Rv,
              'res_f': np.sqrt(resid_f), 'res_v': np.sqrt(resid_v)}).to_csv(f'{W}/latent_R.csv', index=False)
print('[%.0fs] done' % (time.time()-t0))
