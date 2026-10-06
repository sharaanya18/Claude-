import sys, glob, json
ARGS = list(sys.argv)
sys.argv = ['x', '/home/user/Claude-/challenges/chart_change_marks/dataset/public', '/tmp/none.csv']
sys.path.insert(0, '/home/user/Claude-/challenges/chart_change_marks')
import numpy as np, pandas as pd
import solution as S
files = ARGS[1].split(',')
thr = float(ARGS[2]); temp = float(ARGS[3]); out = ARGS[4]
D = S.PUBLIC_DIR
te = pd.read_csv(D / 'test.csv'); sample = pd.read_csv(D / 'sample_submission.csv', keep_default_na=False)
Hs = []; Bs = []
for f in files:
    z = np.load(f); Hs.append(z['H']); Bs.append(z['B'].astype(np.float32)); W = z['W']
H = np.mean(Hs, 0); B = np.mean(Bs, 0)
prior = np.array([0.27, 0.21, 0.52])
pred = []
for i in range(len(te)):
    w = int(W[i]); marks = S.decode_chart(H[i, :w], B[i, :, :w], thr)
    pred.append(S.apply_calib([{"x": round(x + 1.0, 2), "p": [float(v) for v in p]} for x, s, p in marks], temp, 0.0, prior))
sub = sample.copy(); sub['marks'] = [json.dumps(p) for p in pred]; sub['id'] = te['id'].values
S.validate_submission(sub, sample)
sub.to_csv(out, index=False)
print('models', len(files), 'rows', len(sub), 'marks', sum(len(p) for p in pred), 'empty', sum(len(p) == 0 for p in pred))
