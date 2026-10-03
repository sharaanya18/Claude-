import sys; sys.argv=["x","dataset/public","/tmp/x.csv"]
import numpy as np, pandas as pd, json, time
import solution as S
folios=S.load_folios(S.PUBLIC_DIR); tr=pd.read_csv(S.PUBLIC_DIR/"train.csv").drop_duplicates("folio_id")
tt=pd.read_csv(S.PUBLIC_DIR/"train_targets.csv"); gold={r.target_id:json.loads(r.prediction)["descriptor_id"] for r in tt.itertuples()}
items=[S.folio_arrays(folios[f],gold) for f in tr.folio_id]
for th in [0.45,0.55,0.7]:
    S.GROUP_LINK_SIM=th; t=time.time(); fold,comp=S.build_groups(items,3); sz=pd.Series(comp).value_counts()
    print(th,"groups",len(sz),"largest",sz.iloc[:5].tolist(),"singletons",(sz==1).sum(),"fold sizes",np.bincount(fold),"%.0fs"%(time.time()-t),flush=True)
