import sys; sys.path.insert(0,'/home/user/Claude-/challenges/chart_change_marks/exp')
import pickle
ARGS=list(sys.argv)
from common import *
name=ARGS[1]; epochs=int(ARGS[2]); extra=eval(ARGS[3]) if len(ARGS)>3 else {}
tr,X,Wd,M,marks,truth=load()
folds=tr.fold.values
oh,ob=S.run_cv(tr,X,Wd,M,marks,folds,epochs=epochs,**extra)
pickle.dump((oh,ob),open(f'/home/user/Claude-/challenges/chart_change_marks/working/oof_{name}.pkl','wb'))
for thr in [0.2,0.3,0.4,0.5]:
    pr=S.decode_all(oh,ob,thr); cs,m=evaluate(tr,truth,pr); print(thr,round(m,4),{k:round(v,3) for k,v in cs.items()})
