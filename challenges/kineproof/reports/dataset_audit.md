# dataset_audit.md — KineProof, measured from the real files

Phase 1 deliverable. Every number below was computed from the actual bundle
(`dev/audit.py`, `dev/groups.py`, raw output in `reports/audit_raw.txt` and
`reports/split_audit_raw.txt`), not taken from the description.

Bundle provenance: `public.zip`, 91,335,451 bytes, md5 `b04f49abcc609a8cf0334fe77df79120`.

**Compliance boundary observed throughout.** Test arrays were read only for
structural validation (count, shape, dtype, finiteness, origin convention) and
for final inference. No distributional profile of the test split was computed,
because test feature distributions must not inform any design decision
(CLAUDE.md §2.3 #5, repo auditor rule, research P-A10). The prompt's Phase 1
asks whether train and test differ statistically; I deliberately did **not**
answer that beyond structural integrity, and §9 records the cost of that choice.

---

## 1. Integrity and schema — all clean, matches the description exactly

| Item | Measured | Description claims | OK |
|---|---|---|---|
| train rows / `.npy` files | 1093 / 1093 | 1,093 | ✅ |
| test rows / `.npy` files | 322 / 322 | 322 | ✅ |
| `train_targets.csv` rows | 1093, covers train exactly | train only | ✅ |
| `sample_submission.csv` | 322 rows, ids == test ids **in the same order** | — | ✅ |
| motion array shape / dtype | **all** `(256, 22, 3)` `float32` | (256,22,3) float32 | ✅ |
| targets | `(1093, 3, 256)`, all finite | 256 finite floats | ✅ |
| non-finite marker values | none in train, none in test | — | ✅ |
| `motion_file` convention | `<split>/<sample_id>.npy` for every row | enumerated in manifests | ✅ |
| `sample_id` format | `s_` + 16 lowercase hex, unique | opaque identifier | ✅ |

Missing files: none. Shape inconsistencies: none. Exact zeros in motion: 16 of
18,467,328 values (0.0001 %) — numerically negligible, not a missing-data code.

**Ids are opaque and person identity is unrecoverable by construction.** The
description states the split assignment sorts an HMAC-SHA-256 digest of a
private source-person key under an evaluator-only key. No attempt was made to
invert that, and doing so to reach labels would be a Held-out Answer Ingestion
violation.

## 2. Coordinate system, recovered from the motion itself

The description names the force axes (x forward, y vertical, z lateral) but not
the marker axes. Measured from the data:

- **Origin**: the frame-0 **R.ASIS/L.ASIS midpoint** is exactly `0` for every
  trial (max |·| = 0.000000). The 4-marker pelvis centroid is not (max 0.1176),
  nor PSIS-mid (0.2353), nor the all-22 centroid (0.5230). So the subtracted
  origin is the ASIS midpoint at frame 0.
- **Marker axis x = direction of travel**: net displacement over the window is
  `+2.048 ± 0.546 m`, and **positive in 100 % of trials** (so every walker
  travels in +x; no sign ambiguity to handle). 2.05 m / 1.7 s ≈ 1.2 m/s, normal
  walking speed.
- **Marker axis y = vertical**: net displacement ≈ 0 (mean +0.005 m), and
  `R.Iliac.Crest − R.Heel` = **+0.870 m** on y versus +0.091 (x) and +0.127 (z).
- **Marker axis z = lateral**: net displacement ≈ 0, and right-side markers sit
  at positive z, left-side at negative z (e.g. R.ASIS z = +0.124, L.ASIS = −0.124).

**Marker axes therefore align 1:1 with the force axes.** Confirmed by the
physics in §7: each force axis correlates most strongly with the *same* marker
axis' acceleration, with a positive sign.

Global ranges: x ∈ [−0.618, 3.688], y ∈ [−1.038, 0.184], z ∈ [−0.405, 0.435] m.

## 3. Target waveforms

| axis | mean | sd | min | max | mean\|y\| | **Σ\|y\| (the metric denominator)** |
|---|---|---|---|---|---|---|
| force_x (forward) | +0.0025 | 0.1007 | −0.589 | +0.692 | 0.0717 | mean **18.36**, min 5.21 |
| force_y (vertical) | +0.8138 | 0.4222 | −1.010 | +3.653 | 0.8181 | mean **209.44**, min 38.90 |
| force_z (lateral) | +0.0018 | 0.0467 | −0.350 | +0.345 | 0.0371 | mean **9.50**, min 1.84 |

**No trial-axis has Σ|y| = 0** (0 of 3279), so the metric's "entirely zero axis"
branch never fires on train. The smallest denominators (x: 5.21, z: 1.84) are
the most metric-sensitive rows.

**The single most important number in this table:** the denominators differ by
**11.4× (y/x)** and **22.0× (y/z)**. Each axis is normalised by its own
denominator and then weighted equally. An unweighted L1 or MSE loss is therefore
dominated by force_y and, exactly as the description warns, caps the score near
one third. This directly dictates the loss design (see `final_approach.md`).

Waveform character:
- **force_y** is highly stereotyped: mean trace runs 0.86 → 0.71 → 0.96 → 0.58
  across the window, 89.4 % of samples positive, per-trial mean 0.814 BW. This
  is why a constant waveform already scores 0.619 on y.
- **force_x and force_z are zero-mean and essentially have no useful average**
  (mean +0.0025 and +0.0018; a constant waveform scores 0.052 and 0.054). All
  of the horizontal signal is trial-specific.
- Total variation / Σ|y| (median): x 0.116, y 0.028, z 0.091 — all smooth, no
  sharp impact transients to chase (consistent with walking, not running).

**Off-plate structure (a major modelling constraint).** The force is the sum of
five plates, and the walker is not always on them:
- 9.84 % of all frames have |force_y| < 0.02 BW (effectively unloaded);
- 60.8 % of trials contain at least one unloaded frame; only 39.3 % are loaded
  throughout; 16.1 % of trials are unloaded for more than a quarter of the window;
- loaded at frame 0 in 89.1 % of trials, at frame 128 in 98.8 %, at frame 255 in
  only 61.0 % — the window typically ends with the walker leaving the plates.

So the task is a waveform regression **plus a contact-envelope detection**. And
because the absolute lab position was removed by the origin subtraction, where
the plates are is not directly observable — this is an irreducible uncertainty
floor, not something more capacity can fix.

## 4. Duplicates, near-duplicates, outliers

- Exact duplicate motion arrays: **0** in train, **0** in test.
- **Train/test array overlap: 0** — no trial is reused across the split, as stated.
- Exact duplicate target waveforms: **0**.
- Near-duplicates by mean/sd descriptor cosine: max off-diagonal 0.999985;
  253 pairs > 0.9999 and 24,244 pairs > 0.999. These are *not* duplicated
  trials — they are repeated trials of the **same walker**, which is precisely
  the sibling-leakage structure that forces person-grouped CV (§5).
- Motion Frobenius norm 96.1 ± 21.2 (range 55.4–149.5); **0 trials beyond 4 sd**.
- Peak force_y per trial 1.540 ± 0.400 (range 0.308–3.653). 86 trials peak above
  2.0 BW and 1 trial below 0.5 BW. These are kept: the metric scores every trial
  equally and dropping them would bias the model away from the fast walkers.

## 5. Subject structure — persons recovered essentially exactly

Segment lengths are near-constant within a walker, so median-over-time rigid
segment lengths (25 marker pairs) are a legitimate person fingerprint computed
purely from public train inputs. Three independent tests agree that
average-linkage clustering in standardised anthropometric space recovers the
32 people:

**Test 1 — internal consistency vs the measurement noise floor.**
Mean within-cluster segment spread **1.40 mm** against a population spread of
**22.05 mm** (15.8× tighter) and a *within-trial* marker-jitter sd of 4.25 mm.
The clusters are tighter than single-trial marker noise, i.e. they behave like
repeated measurements of one body. Per segment the ratio reaches 57–60× for the
shanks (pop 35.7 mm vs within-cluster 0.62 mm).

**Test 2 — k=32 cluster sizes.** min 8, max 56, median 35, mean 34.2, against
the expected 1093/32 = 34.2. Size list:
`8, 22, 22, 26, 28, 28, 28, 29, 29, 31, 31, 32, 32, 32, 33, 35, 36, 37, 37, 37, 37, 37, 38, 38, 39, 40, 40, 41, 43, 45, 46, 56`.

**Test 3 — the dendrogram has a natural cut at exactly 32 (decisive).** Merge
heights from the top run `… 2.738, 2.712, 2.667,` then **`1.540, 1.522, 1.451, …`**.
The 31 merges that join *different* people all sit at height ≥ 2.667; every
merge from the 32nd onward sits at ≤ 1.540 — a clean 1.73× gap at precisely the
cut the description implies. **The data independently confirm "32 people"
without using any label.**

**Leakage gauge.** Mean validation→train nearest-neighbour distance in
anthropometric space: **0.195 under random 5-fold** versus **3.318 under
person-grouped 5-fold** — a 17× gap. Random trial-level CV would place
near-replicate trials of the same walker on both sides of every fold. Any score
from random CV on this dataset is meaningless.

Adopted split: whole persons to folds, greedily balanced on trial count →
fold sizes `[214, 213, 213, 220, 233]` trials and `[6, 6, 6, 7, 7]` people.

**Bias direction (required statement).** Clusters coarser than true persons ⇒
pessimistic; finer ⇒ optimistic. Tests 1 and 3 indicate the granularity is
right, so the residual bias is small; if anything a 56-trial cluster that is
really two similar-sized people would make the estimate slightly *pessimistic*,
which is the safe direction.

## 6. Temporal / gait structure

- 256 frames over "about 1.7 s" ⇒ **150 Hz** source rate (256/1.7 = 150.6);
  `dt = 1/150 s`. Used for every derivative.
- R.Heel vertical excursion 0.212 m per trial; heel speed ≤ 0.041 m/frame.
- Heel low-points per trial (stride proxy): mean 2.58, median 3, distribution
  `{1: 85, 2: 454, 3: 409, 4: 128, 5: 15, 6: 2}` — about 3 steps ≈ 1.5 gait
  cycles per window, and 85 trials with only one, matching the description's
  warning that some trials do not contain a complete gait cycle.
- First loaded frame: median 0 (p90 = 2). Last loaded frame: median 255 (p10 = 221).

## 7. Physics link — how much force is directly kinematic

Newton for the whole body with the target normalised by `m·g`:
`f_x = a_com,x/g`, `f_y = a_com,y/g + 1`, `f_z = a_com,z/g`. **Mass cancels**,
so the target is a pure kinematic quantity. That is the structural reason this
task can generalise to unseen walkers at all.

Measured with a pelvis-centroid COM proxy and Savitzky–Golay second derivatives
(verified exact on polynomials and against scipy):

| | pelvis_acc_x | pelvis_acc_y | pelvis_acc_z |
|---|---|---|---|
| force_x | **+0.518** | +0.197 | −0.019 |
| force_y | +0.047 | **+0.311** | +0.057 |
| force_z | −0.010 | +0.002 | **+0.175** |

Diagonal dominance with positive signs confirms the axis alignment of §2.

**A zero-training Newtonian estimate scores 0.616 on force_y** (pelvis
acceleration ÷ g + 1, official metric). With no model at all. That tells us the
physics is directly accessible and should be handed to the network explicitly
rather than rediscovered — which is what the solution does (prior channels plus
an additive residual base).

The correlations are only *moderate* because (a) the pelvis is a poor proxy for a
whole-body COM whose trunk/arms/head are not instrumented, and (b) double
differentiation of markers amplifies noise. Both are exactly what a learned
model can fix.

## 8. Information-ceiling diagnostics (all cross-person, exact metric)

| probe | total | x | y | z |
|---|---|---|---|---|
| all-zero submission | 0.000 | 0.000 | 0.000 | 0.000 |
| constant mean waveform | 0.227 | 0.024 | 0.612 | 0.045 |
| constant **median** waveform (L1-optimal) | 0.242 | 0.052 | 0.619 | 0.054 |
| **ORACLE** same-person median waveform | 0.285 | 0.095 | 0.638 | 0.122 |
| untrained Newtonian (SG-31) | 0.281 | 0.001 | 0.654 | 0.186 |
| untrained Newtonian (SG-51) | 0.298 | 0.001 | 0.639 | 0.255 |
| ridge on trial scalars only | 0.268 | 0.117 | 0.612 | 0.076 |
| kNN retrieval on trial scalars (K=15) | 0.274 | 0.106 | 0.634 | 0.083 |
| **per-frame ridge on kinematics (λ=1e4)** | **0.486** | 0.414 | 0.612 | 0.431 |

Four conclusions that shaped the whole approach:

1. **Knowing the person is worth almost nothing.** The oracle same-person median
   (0.285) barely beats the global median (0.242), and both are far below a
   per-frame linear model (0.486). The force is set by *this trial's* kinematics,
   not by the walker's average. **Cross-person generalisation is therefore much
   less dangerous here than the split design suggests** — the map is largely
   person-independent physics. (It also kills retrieval/template methods as a
   primary: §10 of the prompt's plan is a dead end on this data, measured.)
2. **The horizontal axes are where the score lives.** force_y is ~0.61 from a
   constant and climbs slowly; force_x and force_z go from ~0.05 to ~0.43 with a
   linear model. Since each axis is 1/3 of the score, reaching 0.80 overall
   requires roughly 0.8 on *each* axis — the vertical axis alone cannot carry it.
3. **A linear per-frame model already reaches 0.486**, below the printed AI
   baseline of ~0.60 but well above every constant/retrieval probe. The
   remaining gap is nonlinearity and temporal context — which is what the TCN adds.
4. **Target compression is not free.** Oracle projection of the true waveform
   onto a train-fitted basis: PCA 32 comps → 0.910, 64 → 0.952, 128 → 0.978;
   DCT 32 → 0.915, 64 → 0.958, 128 → 0.982. Even 128 of 256 DCT coefficients
   costs 1.8 % of the ceiling, and 32 costs 8.5 %. **Predicting the waveform
   directly in the time domain is correct**; a low-dimensional basis target would
   impose a ceiling the metric can see (it punishes exactly the peaks that
   smoothing removes). This settles the prompt's Phase 8 empirically.

## 9. Known gaps in this audit

- **Train-vs-test distributional comparison was deliberately not performed**
  (see the compliance boundary at the top). Cost of the conservative choice: if
  the 10 held-out people differ systematically from the 32 training people in
  speed, size or gait, I will not know. Mitigation is to rehearse the shift from
  *train only* — per-person CV with a worst-person term, and reporting
  per-held-out-person spread rather than only the mean.
- Mass, height, sex and age of the walkers are not supplied, so anthropometry is
  the only subject descriptor available.
- The 86 high-peak trials (> 2.0 BW) were not individually inspected for whether
  they are fast walking or a protocol difference.
