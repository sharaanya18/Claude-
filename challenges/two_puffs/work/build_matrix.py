import sys, json, time, pickle
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, feats
D='/home/user/Claude-/challenges/two_puffs/dataset'; W='/home/user/Claude-/challenges/two_puffs/work'
t0=time.time()
train=pd.read_csv(f'{D}/train.csv'); test=pd.read_csv(f'{D}/test.csv')
demo=pd.concat([train.drop(columns=['target_json']), test]).set_index('id')

rows={}; accvals={}
with open(f'{D}/blows.jsonl') as fh:
    for line in fh:
        r=json.loads(line); pid=r['pid']
        row,bf=feats.participant_feats(r['blows'], demo.loc[pid])
        rows[pid]=row
        acc=[x for x in bf if x['acceptable']>0.5] or bf
        accvals[pid]=(np.array([x['fev1'] for x in acc]), np.array([x['fvc'] for x in acc]))
print('[%.0fs] blows done %d'%(time.time()-t0, len(rows)))

prows={}
with open(f'{D}/pool.jsonl') as fh:
    for line in fh:
        r=json.loads(line); pid=r['pid']
        dm={k:r[k] for k in ('age_years','sex','ethnicity','height_cm','weight_kg','bmi')}
        row,_=feats.participant_feats(r['blows'], dm)
        row['ethnicity']=r['ethnicity']
        prows[pid]=row
print('[%.0fs] pool done %d'%(time.time()-t0, len(prows)))

X=pd.DataFrame(rows).T
P=pd.DataFrame(prows).T
X['ethnicity']=demo.loc[X.index,'ethnicity'].values
X.index.name='id'
X.to_csv(f'{W}/X_raw.csv'); P.to_csv(f'{W}/pool_raw.csv')
with open(f'{W}/accvals.pkl','wb') as fh: pickle.dump(accvals,fh)
print('X',X.shape,'P',P.shape,'[%.0fs]'%(time.time()-t0))
