import sys; sys.path.insert(0,'/home/user/Claude-/challenges/chart_change_marks/exp')
import pickle
ARGS=list(sys.argv)
from common import *
name=ARGS[1]; epochs=int(ARGS[2]); fl=eval(ARGS[3]); extra=eval(ARGS[4]) if len(ARGS)>4 else {}
tr,X,Wd,M,marks,truth=load()
folds=tr.fold.values
oh=[None]*len(tr); ob=[None]*len(tr)
for f in fl:
    te=np.where(folds==f)[0]; tri=np.where(folds!=f)[0]
    S.log(f'fold {f}')
    net=S._train_fold(X,Wd,M,marks,tri,epochs=epochs,seed=S.SEED+f,**extra)
    hh,bb=S.predict_net(net,X,Wd,M,te)
    for k,i in enumerate(te): oh[i]=hh[k]; ob[i]=bb[k]
pickle.dump((oh,ob),open(f'/home/user/Claude-/challenges/chart_change_marks/working/oof_{name}.pkl','wb'))
