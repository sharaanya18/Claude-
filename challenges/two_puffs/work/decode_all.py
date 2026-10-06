import sys, json, time, base64
sys.path.insert(0, '/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, spiro

D = '/home/user/Claude-/challenges/two_puffs/dataset'
t0=time.time()
rows=[]; bytestats={'min':255,'max':0,'below62':0,'total':0,'censored':0}
hist=np.zeros(256, dtype=np.int64)
for src in ['blows.jsonl','pool.jsonl']:
    with open(f'{D}/{src}') as fh:
        for line in fh:
            r=json.loads(line)
            for b in r['blows']:
                buf=base64.b64decode(b['flow_b64']); raw=np.frombuffer(buf,np.uint8)
                hist += np.bincount(raw, minlength=256)
                bytestats['min']=min(bytestats['min'],int(raw.min())); bytestats['max']=max(bytestats['max'],int(raw.max()))
                bytestats['below62'] += int((raw<62).any()); bytestats['total']+=1
                bytestats['censored'] += int(len(raw)>=2044)
                ix=spiro.blow_indices(b); ix.pop('_v'); ix.pop('_f')
                ix['pid']=r['pid']; ix['src']=src
                rows.append(ix)
B=pd.DataFrame(rows)
print('byte stats', bytestats, '[%.0fs]'%(time.time()-t0))
print('n_points == len(bytes)?', )
B.to_csv('/home/user/Claude-/challenges/two_puffs/work/blow_indices.csv', index=False)
print('blows total', len(B), B.groupby('src').size().to_dict())
print('\n=== per-blow indices (blows.jsonl, acceptable) ===')
sub=B[(B.src=='blows.jsonl')&(B.acceptable==1)]
print(sub[['fev1','fvc','pef','i0','ipk','n','n_points','v0','vmax']].describe().T.round(3))
print('\n=== unacceptable ===')
sub2=B[(B.src=='blows.jsonl')&(B.acceptable==0)]
print(sub2[['fev1','fvc','pef','n']].describe().T.round(3))
print('\nacceptable/plateau cross:', pd.crosstab(B.acceptable,B.plateau).to_dict())
print('fev1/fvc ratio accept:', (sub.fev1/sub.fvc).describe().round(3).to_dict())
print('neg/zero fvc:', (sub.fvc<=0).sum(), 'fvc>8L:', (sub.fvc>8).sum(), 'fev1>fvc:', (sub.fev1>sub.fvc+1e-9).sum())
print('i0==0 frac:', (sub.i0==0).mean().round(4), ' i0 describe:', sub.i0.describe().round(2).to_dict())
print('censored(n>=2044) frac all blows: %.4f'%(B.n_points>=2044).mean())
