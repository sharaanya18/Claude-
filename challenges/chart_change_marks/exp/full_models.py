import sys, os
ARGS = list(sys.argv)
seeds = [int(v) for v in ARGS[1].split(',')]
epochs = int(ARGS[2])
threads = int(ARGS[3])
outdir = ARGS[4]
data = ARGS[5]
sys.argv = ['x', data, os.path.join(outdir, 'tmp.csv')]
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/home/user/Claude-/challenges/chart_change_marks')
import numpy as np, pandas as pd
import solution as S
from pathlib import Path
S.NUM_THREADS = threads
D = Path(data)
tr = pd.read_csv(D / 'train.csv'); te = pd.read_csv(D / 'test.csv')
X, Wd, M = S.load_split(tr, D / 'train_images')
Xt, Wt, Mt = S.load_split(te, D / 'test_images')
marks = [S.parse_marks(s) for s in tr['marks']]
S.log('data ready')
for sd in seeds:
    net = S._train_fold(X, Wd, M, marks, np.arange(len(tr)), epochs=epochs, seed=sd)
    hh, bb = S.predict_net(net, Xt, Wt, Mt, np.arange(len(te)))
    H = np.zeros((len(te), 928), np.float32); B = np.zeros((len(te), 3, 928), np.float16)
    for i, (h, b) in enumerate(zip(hh, bb)):
        H[i, :len(h)] = h; B[i, :, :b.shape[1]] = b
    np.savez(os.path.join(outdir, f'full_seed{sd}.npz'), H=H, B=B, W=Wt)
    S.log(f'saved seed {sd}')
