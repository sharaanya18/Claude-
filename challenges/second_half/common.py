"""Dev helpers (not part of solution.py): data loading + exact metric."""
import json, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
P='dataset/public/'
def load():
    L=pd.read_csv(P+'listens.csv').sort_values(['session','relative_position']).reset_index(drop=True)
    C=pd.read_csv(P+'continuations.csv').sort_values(['continuation','rank'])
    tr=pd.read_csv(P+'train.csv'); te=pd.read_csv(P+'test.csv')
    for d in (tr,te):
        d['pre']=d.prefixes.map(json.loads); d['cand']=d.candidates.map(json.loads)
    tr['y']=list(tr[[f'match_{i}' for i in range(1,7)]].to_numpy()-1)
    return L,C,tr,te
def score(pred,true):  # arrays (N,6) of 0-based
    A=(np.asarray(pred)==np.asarray(true)).mean()
    return (A-1/6)/(1-1/6)
def hungarian(M):  # M 6x6 score (higher better) -> pred position per prefix
    r,c=linear_sum_assignment(-M); p=np.zeros(M.shape[0],int); p[r]=c; return p
