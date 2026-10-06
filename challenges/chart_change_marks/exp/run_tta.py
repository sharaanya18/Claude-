import sys; sys.path.insert(0,'/home/user/Claude-/challenges/chart_change_marks/exp')
import pickle, torch
ARGS=list(sys.argv)
from common import *
tr,X,Wd,M,marks,truth=load()
folds=tr.fold.values
for f in [0,3]:
    te=np.where(folds==f)[0]; tri=np.where(folds!=f)[0]
    S.log(f'fold {f}')
    net=S._train_fold(X,Wd,M,marks,tri,epochs=24,seed=S.SEED+f)
    torch.save(net.state_dict(),f'/home/user/Claude-/challenges/chart_change_marks/working/net_f{f}.pt')
    for name,views in [('none',((0,0),)),('h',((0,0),(1,0))),('hv',((0,0),(1,0),(0,1),(1,1)))]:
        hh,bb=S.predict_net(net,X,Wd,M,te,views=views)
        pickle.dump((te,hh,bb),open(f'/home/user/Claude-/challenges/chart_change_marks/working/tta_{name}_f{f}.pkl','wb'))
