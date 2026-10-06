import sys, json, time, pickle
sys.path.insert(0, '/home/user/Claude-/challenges/chart_change_marks')
sys.argv = ['x']
import numpy as np, pandas as pd
import solution as S
import metric as Mt
from pathlib import Path
D = Path('/home/user/Claude-/challenges/chart_change_marks/dataset/public')
CACHE = Path('/home/user/Claude-/challenges/chart_change_marks/working/cache.pkl')

def load():
    tr = pd.read_csv(D/'train.csv').merge(pd.read_csv(D/'folds.csv'), on='id')
    if CACHE.exists():
        X, Wd, M = pickle.load(open(CACHE, 'rb'))
    else:
        X, Wd, M = S.load_split(tr, D/'train_images'); CACHE.parent.mkdir(exist_ok=True)
        pickle.dump((X, Wd, M), open(CACHE, 'wb'))
    marks = [S.parse_marks(s) for s in tr.marks]
    truth = [json.loads(s) for s in tr.marks]
    tr['cell'] = tr.dataset_name + '|' + tr.fieldtype
    return tr, X, Wd, M, marks, truth

def evaluate(tr, truth, preds, mask=None):
    idx = np.arange(len(tr)) if mask is None else np.where(mask)[0]
    cs, mean = Mt.cell_scores([truth[i] for i in idx], [preds[i] for i in idx], tr.cell.values[idx])
    return cs, mean

PROXY = ['pas|categorical','lims|categorical','micro|datetime','lims|datetime','antibiotics|datetime','pas|uniqueidentifier','micro|uniqueidentifier','antibiotics|uniqueidentifier']
# test-cell look-alikes (same fieldtype or same source system, held out in CV); 4 groups mirror the 4 hidden cells
GROUPS = {'micro/pas categorical':['pas|categorical','lims|categorical'], 'datetime':['micro|datetime','lims|datetime','antibiotics|datetime'],
          'uniqueid':['pas|uniqueidentifier','micro|uniqueidentifier','antibiotics|uniqueidentifier']}

def summary(tr, truth, preds):
    ch = np.array([Mt.chart_score(p, t) for p, t in zip(preds, truth)])
    emp = np.array([Mt.chart_score([], t) for t in truth])
    out = {}
    for c in PROXY:
        m = (tr.cell.values == c)
        b = emp[m].mean(); out[c] = (ch[m].mean() - b) / (1 - b)
    return out, float(np.mean(list(out.values())))
