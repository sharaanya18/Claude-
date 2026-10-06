"""Phase 1: full KineProof dataset audit.

Compliance boundary enforced here: TEST arrays are checked for STRUCTURAL
integrity only (count, shape, dtype, finiteness, pelvis-origin convention).
No distributional profile of test is computed, because this repo's rule
(.claude/agents/eris-data-auditor.md, CLAUDE.md 2.3 #5, research P-A10) is
that test feature distributions must not inform any design decision.
Writes: reports/audit_raw.txt  (+ npz caches under working/)
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kp_data import AXES, MARKERS, MI, MIRROR, N_MARKERS, PUBLIC, T, load_motion, load_targets

OUT = Path(__file__).resolve().parents[1]
WORK = OUT / "working"
WORK.mkdir(exist_ok=True)
L = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def sec(t):
    p("\n" + "=" * 78)
    p("== " + t)
    p("=" * 78)


# ------------------------------------------------------------ 1. integrity --
sec("1. FILE INTEGRITY AND SCHEMA")

tr_man = pd.read_csv(PUBLIC / "train.csv")
te_man = pd.read_csv(PUBLIC / "test.csv")
tg = pd.read_csv(PUBLIC / "train_targets.csv")
ss = pd.read_csv(PUBLIC / "sample_submission.csv")

p(f"train.csv rows             : {len(tr_man)}   cols {list(tr_man.columns)}")
p(f"test.csv rows              : {len(te_man)}   cols {list(te_man.columns)}")
p(f"train_targets.csv rows     : {len(tg)}       cols {list(tg.columns)}")
p(f"sample_submission.csv rows : {len(ss)}       cols {list(ss.columns)}")
p(f"train .npy on disk         : {len(list((PUBLIC/'train').glob('*.npy')))}")
p(f"test  .npy on disk         : {len(list((PUBLIC/'test').glob('*.npy')))}")
p(f"description claims 1093 train / 322 test -> "
  f"{'MATCH' if (len(tr_man), len(te_man)) == (1093, 322) else 'MISMATCH'}")

p(f"\ntrain ids unique           : {tr_man.sample_id.is_unique}")
p(f"test  ids unique           : {te_man.sample_id.is_unique}")
p(f"targets cover train exactly: {set(tg.sample_id) == set(tr_man.sample_id)}")
p(f"sample_sub ids == test ids : {set(ss.sample_id) == set(te_man.sample_id)}")
p(f"sample_sub id ORDER == test: {ss.sample_id.tolist() == te_man.sample_id.tolist()}")
p(f"motion_file == '<split>/<id>.npy' for all train: "
  f"{all(r.motion_file == f'train/{r.sample_id}.npy' for r in tr_man.itertuples())}")
p(f"motion_file == '<split>/<id>.npy' for all test : "
  f"{all(r.motion_file == f'test/{r.sample_id}.npy' for r in te_man.itertuples())}")
p(f"id format: all match 's_' + 16 hex: "
  f"{all(len(s) == 18 and s.startswith('s_') and all(c in '0123456789abcdef' for c in s[2:]) for s in tr_man.sample_id)}")
p("NOTE: ids are opaque 16-hex digests. The description says person assignment")
p("      uses HMAC-SHA-256 under an evaluator-only key, so person identity is")
p("      NOT recoverable from ids by design. No attempt is made to do so.")

# load everything
tr_ids, X = load_motion("train")
te_ids, Xte = load_motion("test")
_, Y = load_targets(tr_ids)
p(f"\nmotion train array {X.shape} {X.dtype}   test {Xte.shape} {Xte.dtype}")
p(f"targets array      {Y.shape} {Y.dtype}")
p(f"all train shapes (256,22,3): {X.shape[1:] == (T, N_MARKERS, 3)}")
p(f"all test  shapes (256,22,3): {Xte.shape[1:] == (T, N_MARKERS, 3)}")
p(f"train motion finite        : {np.isfinite(X).all()}")
p(f"test  motion finite        : {np.isfinite(Xte).all()}")
p(f"targets finite             : {np.isfinite(Y).all()}")
p(f"exact zeros in motion      : {int((X == 0).sum())} of {X.size} "
  f"({100*(X==0).mean():.4f}%)")

# pelvis origin convention: "coordinates relative to the first-frame pelvis origin"
PELVIS = [MI["R.ASIS"], MI["L.ASIS"], MI["R.PSIS"], MI["L.PSIS"]]
pel0_tr = X[:, 0, PELVIS, :].mean(axis=1)          # (N,3) pelvis centre, frame 0
pel0_te = Xte[:, 0, PELVIS, :].mean(axis=1)
p(f"\nframe-0 pelvis centre (4 ASIS/PSIS mean), train: "
  f"max|.| = {np.abs(pel0_tr).max():.6g}")
p(f"frame-0 pelvis centre, test : max|.| = {np.abs(pel0_te).max():.6g}")
p("  -> if ~0, the origin is the 4-marker pelvis centroid at frame 0.")
for name, idxs in [("ASIS mid", [MI["R.ASIS"], MI["L.ASIS"]]),
                   ("PSIS mid", [MI["R.PSIS"], MI["L.PSIS"]]),
                   ("4-marker", PELVIS),
                   ("all-22", list(range(N_MARKERS)))]:
    v = X[:, 0, idxs, :].mean(axis=1)
    p(f"   candidate origin {name:<9}: train max|.| = {np.abs(v).max():.6g}")

# --------------------------------------------------- 2. coordinate ranges ---
sec("2. MARKER COORDINATE STATISTICS (train only)")

p(f"{'marker':<16} {'x mean':>9} {'x sd':>8} {'y mean':>9} {'y sd':>8} "
  f"{'z mean':>9} {'z sd':>8}   {'|vel| m/fr':>10}")
vel = np.diff(X, axis=1)                        # (N,255,22,3)
spd = np.linalg.norm(vel, axis=-1)              # (N,255,22)
for m in range(N_MARKERS):
    a = X[:, :, m, :]
    p(f"{MARKERS[m]:<16} {a[...,0].mean():9.4f} {a[...,0].std():8.4f} "
      f"{a[...,1].mean():9.4f} {a[...,1].std():8.4f} "
      f"{a[...,2].mean():9.4f} {a[...,2].std():8.4f}   {spd[:,:,m].mean():10.5f}")

glob_rng = [(X[..., k].min(), X[..., k].max()) for k in range(3)]
p(f"\nglobal coord range  x [{glob_rng[0][0]:.3f},{glob_rng[0][1]:.3f}] "
  f"y [{glob_rng[1][0]:.3f},{glob_rng[1][1]:.3f}] z [{glob_rng[2][0]:.3f},{glob_rng[2][1]:.3f}]")
p("Interpreting the lab axes from the motion itself:")
for k, nm in enumerate("xyz"):
    disp = X[:, -1, :, k].mean(axis=1) - X[:, 0, :, k].mean(axis=1)
    p(f"  axis {nm}: net body displacement over the 256 frames "
      f"mean {disp.mean():+.4f}  sd {disp.std():.4f}  "
      f"|mean| {np.abs(disp).mean():.4f}  frac>0 {(disp>0).mean():.3f}")
p("  (largest |net displacement| = direction of travel; the vertical axis should")
p("   show near-zero net displacement but marker heights spanning ~1 leg length.)")
# heights: which axis separates heel from iliac crest the most?
for k, nm in enumerate("xyz"):
    d = (X[:, :, MI["R.Iliac.Crest"], k] - X[:, :, MI["R.Heel"], k]).mean()
    p(f"  axis {nm}: mean (R.Iliac.Crest - R.Heel) = {d:+.4f}  "
      f"-> {'VERTICAL candidate' if abs(d) > 0.5 else ''}")

# ------------------------------------------------------- 3. target audit ----
sec("3. TARGET (FORCE) WAVEFORM AUDIT")

den = np.abs(Y).sum(axis=2)                      # (N,3) the metric denominator
p(f"{'axis':<8} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9} "
  f"{'mean|y|':>9} {'sum|y| mean':>12} {'sum|y| min':>11}")
for a, nm in enumerate(AXES):
    y = Y[:, a]
    p(f"{nm:<8} {y.mean():9.4f} {y.std():8.4f} {y.min():9.4f} {y.max():9.4f} "
      f"{np.abs(y).mean():9.4f} {den[:,a].mean():12.3f} {den[:,a].min():11.3f}")

p(f"\nany trial-axis with sum|y| == 0 (exact-zero axis)? "
  f"{int((den == 0).sum())} of {den.size}")
p(f"smallest sum|y| per axis: " + "  ".join(f"{AXES[a]}={den[:,a].min():.4f}" for a in range(3)))
p("\nMETRIC CONSEQUENCE: the denominator sum|y| differs between axes by")
p(f"  ratio y/x = {den[:,1].mean()/den[:,0].mean():.2f}x, y/z = {den[:,1].mean()/den[:,2].mean():.2f}x.")
p("  Each axis is normalised by its OWN denominator and then weighted equally,")
p("  so an unweighted L1/L2 loss would be dominated by force_y. A metric-aligned")
p("  loss must divide each trial-axis residual by that trial-axis sum|y|.")

p("\nper-axis waveform shape, mean over trials (every 16th frame):")
for a, nm in enumerate(AXES):
    mw = Y[:, a].mean(axis=0)
    p(f"  {nm}: " + " ".join(f"{mw[t]:+.3f}" for t in range(0, T, 16)))
p("\nper-axis sd across trials at each frame (every 16th):")
for a, nm in enumerate(AXES):
    sw = Y[:, a].std(axis=0)
    p(f"  {nm}: " + " ".join(f"{sw[t]:.3f}" for t in range(0, T, 16)))

# vertical force should sit near +1 BW in stance (body weight) if sign is +up
p(f"\nforce_y: frac of all samples > 0 = {(Y[:,1]>0).mean():.4f}; "
  f"mean = {Y[:,1].mean():.4f} BW; 99th pct = {np.percentile(Y[:,1],99):.3f}")
p(f"force_y per-trial mean: min {Y[:,1].mean(axis=1).min():.3f} "
  f"max {Y[:,1].mean(axis=1).max():.3f} mean {Y[:,1].mean(axis=1).mean():.3f}")
p("  -> a 256-frame window of overground walking with 5 summed plates should")
p("     average near +1 BW vertically whenever the walker is fully on the plates;")
p("     lower means part of the window has the walker off the instrumented area.")
p(f"frac of trials whose force_y is < 0.05 BW for >25% of frames: "
  f"{((np.abs(Y[:,1])<0.05).mean(axis=1) > 0.25).mean():.4f}")
p(f"frac of frames with |force_y| < 0.02 BW (effectively unloaded): "
  f"{(np.abs(Y[:,1])<0.02).mean():.4f}")

# how many near-zero (flight/off-plate) runs
nz = (np.abs(Y[:, 1]) < 0.02)
p(f"trials with ANY unloaded frame: {(nz.any(axis=1)).mean():.4f}; "
  f"trials entirely loaded: {(~nz.any(axis=1)).mean():.4f}")

# smoothness: how much high-frequency content?
for a, nm in enumerate(AXES):
    d1 = np.abs(np.diff(Y[:, a], axis=1)).sum(axis=1)
    p(f"{nm}: total variation / sum|y| = {np.median(d1/den[:,a]):.4f} (median) "
      f"-> {'smooth' if np.median(d1/den[:,a]) < 0.2 else 'has transients'}")

# ----------------------------------------------- 4. duplicates / outliers --
sec("4. DUPLICATES, NEAR-DUPLICATES, OUTLIERS")

h = [hashlib.sha256(X[i].tobytes()).hexdigest() for i in range(len(tr_ids))]
hc = Counter(h)
p(f"exact duplicate motion arrays in train: {sum(v-1 for v in hc.values() if v>1)}")
hte = [hashlib.sha256(Xte[i].tobytes()).hexdigest() for i in range(len(te_ids))]
p(f"exact duplicate motion arrays in test : {sum(v-1 for v in Counter(hte).values() if v>1)}")
p(f"train/test exact array overlap        : {len(set(h) & set(hte))}  "
  f"(must be 0: no trial is reused across the split)")

hy = [hashlib.sha256(np.ascontiguousarray(Y[i]).tobytes()).hexdigest() for i in range(len(Y))]
p(f"exact duplicate target waveforms      : {sum(v-1 for v in Counter(hy).values() if v>1)}")

# near-duplicate motion: cosine distance on a cheap descriptor
desc = np.concatenate([X.mean(axis=1).reshape(len(X), -1),
                       X.std(axis=1).reshape(len(X), -1)], axis=1)
dn = desc / np.linalg.norm(desc, axis=1, keepdims=True)
S = dn @ dn.T
np.fill_diagonal(S, -1)
p(f"\nnear-duplicate descriptor cosine: max off-diagonal {S.max():.6f}, "
  f"pairs > 0.9999: {int((S > 0.9999).sum()//2)}, > 0.999: {int((S>0.999).sum()//2)}")

# outlier trials by motion scale
sc = np.linalg.norm(X.reshape(len(X), -1), axis=1)
p(f"motion Frobenius norm: mean {sc.mean():.2f} sd {sc.std():.2f} "
  f"min {sc.min():.2f} max {sc.max():.2f}")
q = np.percentile(sc, [0.5, 1, 50, 99, 99.5])
p(f"  percentiles 0.5/1/50/99/99.5: " + " ".join(f"{v:.2f}" for v in q))
p(f"  trials beyond 4 sd of motion norm: {int((np.abs(sc-sc.mean())>4*sc.std()).sum())}")

# target outliers
py = Y[:, 1].max(axis=1)
p(f"peak force_y per trial: mean {py.mean():.3f} sd {py.std():.3f} "
  f"min {py.min():.3f} max {py.max():.3f}")
p(f"  trials with peak force_y > 2.0 BW (implausible for walking): "
  f"{int((py>2.0).sum())}")
p(f"  trials with peak force_y < 0.5 BW (barely on the plates): {int((py<0.5).sum())}")

# ------------------------------------------- 5. anthropometry / person id --
sec("5. SUBJECT STRUCTURE FROM ANTHROPOMETRY")
p("Segment lengths are near-constant within a person and differ between people,")
p("so they are the legitimate route to approximate person groups. They are a")
p("function of PUBLIC TRAIN INPUTS only -- no ids, no order, no provenance.")

SEG = [("pelvis_width", "R.ASIS", "L.ASIS"),
       ("pelvis_depth_R", "R.ASIS", "R.PSIS"),
       ("pelvis_depth_L", "L.ASIS", "L.PSIS"),
       ("crest_width", "R.Iliac.Crest", "L.Iliac.Crest"),
       ("gtr_width", "R.GTR", "L.GTR"),
       ("thigh_R", "R.GTR", "R.Knee"), ("thigh_L", "L.GTR", "L.Knee"),
       ("shank_R", "R.Knee", "R.Ankle"), ("shank_L", "L.Knee", "L.Ankle"),
       ("knee_HF_R", "R.Knee", "R.HF"), ("knee_HF_L", "L.Knee", "L.HF"),
       ("TT_ankle_R", "R.TT", "R.Ankle"), ("TT_ankle_L", "L.TT", "L.Ankle"),
       ("foot_R", "R.Heel", "R.MT1"), ("foot_L", "L.Heel", "L.MT1"),
       ("foot5_R", "R.Heel", "R.MT5"), ("foot5_L", "L.Heel", "L.MT5"),
       ("forefoot_R", "R.MT1", "R.MT5"), ("forefoot_L", "L.MT1", "L.MT5"),
       ("asis_gtr_R", "R.ASIS", "R.GTR"), ("asis_gtr_L", "L.ASIS", "L.GTR")]


def seg_table(A):
    """median-over-time segment length per trial -> (N, n_seg)"""
    out = np.empty((len(A), len(SEG)), dtype=np.float64)
    for j, (_, m1, m2) in enumerate(SEG):
        d = np.linalg.norm(A[:, :, MI[m1], :] - A[:, :, MI[m2], :], axis=-1)
        out[:, j] = np.median(d, axis=1)
    return out


Str = seg_table(X)
Ste = seg_table(Xte)
np.savez_compressed(WORK / "seg.npz", train=Str, test=Ste,
                    names=np.array([s[0] for s in SEG]), tr_ids=np.array(tr_ids),
                    te_ids=np.array(te_ids))

p(f"\n{'segment':<16} {'mean m':>8} {'sd m':>8} {'cv%':>7} "
  f"{'within-trial sd':>16}")
for j, (nm, m1, m2) in enumerate(SEG):
    d = np.linalg.norm(X[:, :, MI[m1], :] - X[:, :, MI[m2], :], axis=-1)
    p(f"{nm:<16} {Str[:,j].mean():8.4f} {Str[:,j].std():8.4f} "
      f"{100*Str[:,j].std()/Str[:,j].mean():7.2f} {d.std(axis=1).mean():16.5f}")
p("\nA segment whose within-trial sd is far below its across-trial sd is a")
p("rigid body dimension -> good person fingerprint. High within-trial sd means")
p("the marker pair spans a joint that actually moves (not usable as a length).")

# how separable are people? Look at the gap structure of pairwise distances
Z = (Str - Str.mean(0)) / (Str.std(0) + 1e-9)
D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
nn = D.min(axis=1)
p(f"\nstandardised anthropometric space: nearest-neighbour distance")
p(f"  mean {nn.mean():.4f} median {np.median(nn):.4f} "
  f"p90 {np.percentile(nn,90):.4f} max {nn.max():.4f}")
sortD = np.sort(D, axis=1)
p(f"  distance to 1st/10th/30th/40th/60th NN (median over trials): " +
  " ".join(f"{np.median(sortD[:,k]):.3f}" for k in [0, 9, 29, 39, 59]))
p("  If ~34 trials per person (1093/32), a jump in this curve around k~30-40")
p("  marks the edge of a person's own cluster.")

# hierarchical clustering into exactly 32 groups
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
Dc = D.copy()
np.fill_diagonal(Dc, 0.0)
lk = linkage(squareform(Dc, checks=False), method="average")
for k in (16, 24, 32, 40, 48, 64):
    lab = fcluster(lk, k, criterion="maxclust")
    sz = np.bincount(lab)[1:]
    p(f"  k={k:<3} cluster sizes: n={len(sz)} min={sz.min()} max={sz.max()} "
      f"median={int(np.median(sz))}  (ideal ~{1093/k:.0f})")

lab32 = fcluster(lk, 32, criterion="maxclust")
np.savez_compressed(WORK / "clusters.npz", lab32=lab32, D=D.astype(np.float32))
p(f"\nk=32 within-cluster mean NN distance {np.mean([nn[lab32==c].mean() for c in np.unique(lab32)]):.4f}")

# purity proxy: how tightly do trials in one cluster agree on anthropometry?
wi, be = [], []
for c in np.unique(lab32):
    m = lab32 == c
    if m.sum() > 1:
        wi.append(D[np.ix_(m, m)][np.isfinite(D[np.ix_(m, m)])].mean())
    be.append(D[np.ix_(m, ~m)].mean())
p(f"k=32: mean WITHIN-cluster distance {np.mean(wi):.4f}, "
  f"mean BETWEEN-cluster distance {np.mean(be):.4f}, ratio {np.mean(be)/np.mean(wi):.2f}x")

# ------------------------------------------------ 6. temporal structure ----
sec("6. TEMPORAL / GAIT STRUCTURE")

p("Window is 256 frames ~ 1.7 s, so the source marker rate is ~150 Hz.")
p(f"  256 / 1.7 s = {256/1.7:.1f} Hz")

# heel/toe vertical trace tells us stance/swing; find the vertical axis first
vax = int(np.argmax([abs((X[:, :, MI['R.Iliac.Crest'], k] - X[:, :, MI['R.Heel'], k]).mean())
                     for k in range(3)]))
tax = int(np.argmax([np.abs(X[:, -1, :, k].mean(axis=1) - X[:, 0, :, k].mean(axis=1)).mean()
                     for k in range(3)]))
p(f"inferred vertical axis index = {vax} ('{'xyz'[vax]}'), "
  f"travel axis index = {tax} ('{'xyz'[tax]}')")
p(f"  (description: force_y is vertical, force_x forward, force_z lateral;")
p(f"   marker axes need not use the same ordering -- verified above from motion.)")

hv = X[:, :, MI["R.Heel"], vax]
p(f"R.Heel vertical: per-trial range mean {np.ptp(hv,axis=1).mean():.4f} m")
fs = np.linalg.norm(np.diff(X[:, :, MI["R.Heel"], :], axis=1), axis=-1)
p(f"R.Heel speed: mean {fs.mean():.5f} m/frame, max {fs.max():.5f}")
# count heel-strike-like minima of heel height per trial
cnt = []
for i in range(len(X)):
    v = hv[i]
    loc = [t for t in range(2, T-2) if v[t] <= v[t-1] and v[t] <= v[t+1]
           and v[t] < v.mean()]
    # merge adjacent
    merged = [t for k, t in enumerate(loc) if k == 0 or t - loc[k-1] > 20]
    cnt.append(len(merged))
p(f"R.Heel low-point count per trial (proxy for strides in window): "
  f"mean {np.mean(cnt):.2f} median {np.median(cnt):.0f} "
  f"distribution {dict(sorted(Counter(cnt).items()))}")
p("  -> ~1.7 s at normal cadence (~0.55 s/step) holds ~3 steps / ~1.5 cycles;")
p("     the description warns some trials lack a complete gait cycle.")

# where in the window is the force active?
act = (np.abs(Y[:, 1]) > 0.05)
p(f"\nfraction of trials loaded at frame 0 / 128 / 255: "
  f"{act[:,0].mean():.3f} / {act[:,128].mean():.3f} / {act[:,255].mean():.3f}")
first = np.array([np.argmax(a) if a.any() else -1 for a in act])
last = np.array([T-1-np.argmax(a[::-1]) if a.any() else -1 for a in act])
p(f"first loaded frame: median {np.median(first):.0f}  p90 {np.percentile(first,90):.0f}")
p(f"last loaded frame : median {np.median(last):.0f}   p10 {np.percentile(last,10):.0f}")

# --------------------------------------------- 7. physics / information ----
sec("7. PHYSICS LINK: IS GRF PREDICTABLE FROM COM ACCELERATION?")
p("Newton: sum(F_ext) = m*a_com, and GRF/(m*g) = a_com/g + gravity term.")
p("With markers only on the lower body we can only approximate the COM, but the")
p("correlation tells us how much of the force is directly kinematic.")

# crude COM proxy: pelvis centroid (carries most of the trunk mass motion)
com = X[:, :, PELVIS, :].mean(axis=2)                      # (N,T,3) pelvis centroid per frame
dt = 1.0 / (T / 1.7)
acc = np.gradient(np.gradient(com, dt, axis=1), dt, axis=1)  # (N,T,3) m/s^2
g = 9.80665
for a, nm in enumerate(AXES):
    best = None
    for k in range(3):
        c = np.corrcoef(acc[:, :, k].ravel(), Y[:, a].ravel())[0, 1]
        if best is None or abs(c) > abs(best[1]):
            best = ("xyz"[k], c)
        p(f"  corr({nm}, pelvis_acc_{'xyz'[k]}) = {c:+.4f}")
    p(f"  -> {nm} best matched by pelvis_acc_{best[0]} ({best[1]:+.4f})")

# vertical: a_v/g + 1 should track force_y
pred_y = acc[:, :, vax] / g + 1.0
from kp_data import score as kp_score
Ptest = np.zeros_like(Y)
Ptest[:, 1] = pred_y
p(f"\nNaive Newtonian vertical estimate (pelvis acc / g + 1) scored with the")
p(f"official metric on force_y alone: "
  f"{np.mean([__import__('kp_data').axis_skill(pred_y[i], Y[i,1]) for i in range(len(Y))]):.4f}")

with open(OUT / "reports" / "audit_raw.txt", "w") as f:
    f.write("\n".join(L) + "\n")
p(f"\nwrote reports/audit_raw.txt ({len(L)} lines)")
