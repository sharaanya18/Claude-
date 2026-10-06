"""Optional feature block: normalised flow-volume loop shape + a pool-fitted PCA of it.

Phase 7 of the brief, and the one place the published reference rungs went backwards
(0.6017 -> 0.5962 when loop shape was added to a nine-target model).  Kept as an ablatable
block so validation, not belief, decides.  The PCA basis is learned from the shipped pool,
which the challenge explicitly permits ("representations learned from the shipped pool are
welcome"); it never sees a target or a test row at fit time.
"""
import sys, json, time
sys.path.insert(0,'/home/user/Claude-/challenges/two_puffs/work')
import numpy as np, pandas as pd, feats

NGRID = 12
FRACS = np.linspace(0.05, 0.95, NGRID)


def loop_curve(blow):
    """Flow at fixed exhaled-volume fractions, normalised by PEF (scale-free loop shape)."""
    v, f = feats.decode(blow)
    n = len(v)
    ipk = int(np.argmax(f)); pef = float(f[ipk])
    if pef <= 0:
        return np.zeros(NGRID)
    i0 = max(0, min(int(np.round(ipk - v[ipk]/(pef*feats.DT))), n-1))
    vol = v - v[i0]
    imax = int(np.argmax(vol)); vmax = float(vol[imax])
    if vmax <= 0 or imax <= ipk:
        return np.zeros(NGRID)
    seg = np.maximum.accumulate(vol[ipk:imax+1]); fseg = f[ipk:imax+1]
    return np.clip(np.interp(FRACS*vmax, seg, fseg)/pef, 0.0, 2.0)


def participant_curves(blows):
    cs = [loop_curve(b) for b in blows if b["acceptable"] == "Y"] or [loop_curve(b) for b in blows]
    A = np.array(cs)
    return A.mean(0), (A.std(0) if len(A) > 1 else np.zeros(NGRID))


def build(path, keyed_demo=False):
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            m, s = participant_curves(r['blows'])
            out[r['pid']] = np.concatenate([m, s])
    cols = ['loop_m%d' % i for i in range(NGRID)] + ['loop_s%d' % i for i in range(NGRID)]
    return pd.DataFrame(out).T.rename(columns=dict(enumerate(cols)))


if __name__ == '__main__':
    D = '/home/user/Claude-/challenges/two_puffs/dataset'
    W = '/home/user/Claude-/challenges/two_puffs/work'
    t0 = time.time()
    A = build(f'{D}/blows.jsonl'); A.to_csv(f'{W}/loop_blows.csv')
    print('blows', A.shape, '[%.0fs]' % (time.time()-t0), flush=True)
    B = build(f'{D}/pool.jsonl'); B.to_csv(f'{W}/loop_pool.csv')
    print('pool', B.shape, '[%.0fs]' % (time.time()-t0), flush=True)
