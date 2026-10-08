# Eris plan: KineProof, Echoes of an Invisible Floor (BLIND strategist run)

Status: blind plan written from `corpus/kineproof_floor/CHALLENGE.md` (itself a *reconstruction*, not verbatim
platform text), CLAUDE.md and the eris-playbook references. **No dataset was available.** Every number about rows,
columns, units, frame rate, people, the official metric formula and the submission grammar is an **assumption /
unverified** unless it is quoted from CHALLENGE.md, and even those quotes are second-hand. Each assumption below has a
train-only diagnostic that confirms or refutes it in the first hour of real work (roadmap step 1).

Family classification (SKILL step 1): primarily **audio-signal / sensor streams** (marker trajectories in, force
waveform out) with **tabular-and-scientific** "domain theory as architecture" (F8, C11) and a latent-structure element
(which foot is on an instrumented plate) handled like **structured-assignment** (exact marginalisation over a tiny
latent set, principle F2-2).

---

## Contract & decision unit

**Input per trial (assumed from CHALLENGE.md):** one row in `train.csv` / `test.csv` with an id and a `motion_file`
column pointing at a per-trial marker array of shape `(256 frames, 22 markers, 3 coords)`. File format (npy / csv /
npz), marker names, coordinate units (mm vs m), axis convention (which axis is vertical) and frame rate are
**unverified**. `train_targets.csv` holds the force sequences for training trials (layout unverified: either one row
per trial with three JSON arrays, or long format id/frame/force_x/force_y/force_z).

**One valid answer (assumed):** for each test trial, three finite real-valued sequences of length exactly 256
(`force_x`, `force_y`, `force_z`), serialised exactly as `sample_submission.csv` dictates (CHALLENGE.md says rank-1
wrote one JSON array of 256 floats per axis per trial; **unverified**: could also be long format, one row per
trial-frame). The script must read the sample file, mirror its columns/order/ids, and emit the same grammar.

**Invalid vs low-scoring:** invalid = wrong columns/order/ids, row count mismatch, array length != 256, NaN/inf,
empty strings, malformed JSON (scores below 0 and burns a credit). Low-scoring = anything well-formed.

**Metric (implied, unverified):** for trial i and axis a,
`r_ia = sum_t |p_iat - y_iat| / sum_t |y_iat|`, capped at 1 (CHALLENGE.md: "capped at 1 with a shallow slope beyond
the cap"; whether the *official* metric is a hard `min(r,1)` and the slope is only rank-1's training surrogate is
unknown). Score = mean over trials and axes (lower is better; possibly reported as `1 - error`). Term-by-term:
- Every trial and every axis weighs the same regardless of force magnitude. The small shear axes (anterior-posterior,
  medio-lateral) count exactly as much as the large vertical axis: they are where the score will be won or lost.
- **Predicting all zeros gives exactly r = 1** on every axis (the cap value). So the cap makes 0 the "abstain"
  answer: any axis prediction worse than zero is (almost) no worse than zero, and any real signal is a pure gain.
  On noisy shear axes the expected-loss-optimal answer is a *shrunk* version of the model mean (posterior median of
  an L1 loss pulled toward 0 when the sign/timing is uncertain).
- L1 numerator: rewards the per-frame posterior **median**, not the mean; robust to occasional spikes.
- Normaliser uses the true signal only, so it is a per-trial-axis constant: training with this loss is equivalent to
  an L1 loss with per-example weight `1/sum|y_ia|` (metric weights, A2/A7) plus the cap.
- Edge case to check in train: trials with `sum|y_ia| == 0` (no foot ever touches a plate). Official handling unknown
  (eps? excluded?). If such trials exist, an exact-zero prediction matters; see decode.

**True independent unit:** the **person** (32 people per CHALLENGE.md, unverified), then the trial. Trials from one
person share body mass, anthropometry, marker placement and gait style; trials may also be overlapping windows cut
from one longer walk (check). The test set is most plausibly person-disjoint (biomechanics generalisation standard;
rank-2 hard-coding `TRAIN_PEOPLE = 32` hints that "people" is an explicit axis). Treat as person-disjoint unless the
description states otherwise.

**Pipeline stages (diagnose separately):**
1. *Whole-body force (Newton):* `W(t) = m * (a_com(t) - g)` from marker kinematics. Quality = how well the
   CoM acceleration and the mass are estimated.
2. *Per-foot split:* during double support the total is shared between feet (`D`, the between-feet difference).
3. *Plate gating (the "invisible floor"):* the target is only the part of `W` carried by feet that stand on an
   instrumented plate; gates `g_R(t), g_L(t)` in [0,1]. Plates are not visible in the data; their layout has to be
   learned from where feet are when force appears.
4. *Decode:* per-axis shrinkage / serialisation.
Diagnose each with oracles on train (see Validation design): oracle-gate ceiling, oracle-mass ceiling, Newton-only.

---

## Compliance regime

- **Domain:** sensor/signal regression, tabular-scientific. No pretrained weights needed or useful (no HF mocap
  backbone worth the risk). Likely filed under a non-fine-tuning bucket; if it turns out to be labelled
  "From-scratch", the plan is already from-scratch (no pretrained anything). If labelled "Fine-tuning" (unlikely for
  mocap), raise as a reviewer question; nothing domain-appropriate exists on HF.
- **Hardware:** CHALLENGE.md says both inspected solutions are CPU-only (`THREADS = 10` and `2`), "consistent with a
  stated CPU-only environment". Plan **CPU-only**: `device = "cpu"` hard-coded, `torch.set_num_threads(N_THREADS)`
  with a fixed constant (10 if the description says 10 cores; assumption), no CUDA calls, no `cuda.is_available()`.
  Runtime limit unknown: plan for **<= 40 min** total on 10 CPU cores (>= 30% headroom against a 1 h limit; if the
  description states 1.5 h, keep the plan anyway and add seeds, not stages).
- **Explicit bans I must assume (CLAUDE.md §2.3):** no external mocap/force data, no synthetic labelled data, no
  pretrained "gait-to-GRF" models, no test statistics (no test-wide normalisation, no per-person clustering of test
  trials to pool mass estimates), no pseudo-labels, no hand-coded plate rectangles, no hard-coded body masses.
- **Allowed and central:** published physics (Newton's second law, CoM from markers) as *architecture* for a trained
  model (F8/C11, "domain theory as architecture"); hand-computed kinematic features (velocities, accelerations, foot
  heights) as inputs to the trained model; per-trial (row-local) TTA if not banned.
- **Strip-the-ML test (self-audit):** remove all trained parts and what remains is `W0 = m0*(a_uniform_centroid - g)`
  with a constant mass and no gates. That predicts whole-body force while the target is *plate-only* force; it is wrong
  in every frame where a foot is off a plate, has no per-foot split and no mass. Every load-bearing piece (gates =
  learned plate map + contact detection, D split, mass/scale head, residual corrections, CoM marker weights,
  derivative filters) is trained. The physics only sets the output parameterisation. Passes, and the docstring will
  say so with a logged ablation (Newton-only OOF vs full model OOF).
- **Whole-test aggregation:** none. Each test trial's prediction is a function of its own marker array plus
  train-fitted parameters. Feature scalers are fitted on train only. Half-rows test (drop half of test rows, the kept
  predictions must be bit-identical) will be run.
- **Sibling leakage:** no feature cross-references other trials. Person grouping is for validation only. The person
  fingerprint is computed from the trial's own markers (row-local).
- **Every constant derivable in-script:** shrink factors, blend weights and mass-head regularisation are fitted inside
  `solution.py` on OOF; architecture sizes, epochs and LR are fixed design constants profiled offline (documented as
  defaults, not "tuned on submissions"). No constant may come from watching public scores.
- **Grey item to note:** the plate map is learned from absolute foot coordinates, i.e. the model learns the lab
  layout. That is the physical data-generating process the task title points at ("invisible floor"), learned from
  train labels; it is not provenance/id exploitation. One reviewer question listed below.

---

## Data findings

No data available: these are **expected** findings plus the exact train-only diagnostic for each (run before
modelling; record results in `reports/`). Numbers marked (A) are assumptions.

1. **Shapes:** `N_train` trials (A: 600–4,000; 32 people → ~20–120 trials per person), each `(256,22,3)`; targets
   `(256,3)`. Test row count only used for runtime/memory.
2. **Units and axes:** determine marker unit by typical inter-marker distances (mm if ~100s). Determine the vertical
   marker axis as the axis along which head markers are always above foot markers. Determine the vertical *force* axis
   as the axis with the largest mean |force| (~body weight during single support); sign convention (reaction force
   positive up or plate-load positive down) from its mean sign. These conventions are learned by the model anyway
   (learned 3x3 map, see approach) but must be known for diagnostics.
3. **Force units / mass identifiability:** fit per person `m_p` by least squares of `y_vert ≈ m_p (a_com_vert + g)`
   on frames where the force is clearly "full" (ratio cluster near the top). If `m_p` is ~constant across people
   (CV < 3%) the forces are body-weight-normalised → no mass head needed. If it varies (typical 50–100 kg, CV ~15%)
   the forces are in Newtons and mass must be estimated from anthropometry (expected error contribution ~5–10% on the
   vertical axis for unseen people: this is the main *information-ceiling* term for force_z).
4. **The invisible floor:** for each frame compute `ratio(t) = y_vert(t) / W0_vert(t)` (W0 with per-person fitted
   mass). Expected trimodal histogram: ~0 (no foot on a plate), ~1 (all load on plates), intermediate (double support
   with one foot on the floor between plates). Record the share of frames per mode. For the onset frames (force rises
   from ~0) record the striking foot's horizontal position: if positions cluster at a few fixed locations across all
   people, the plate layout is fixed in the lab frame (learnable map). If the clusters drift by person/session, the
   coordinate frame changes per session and absolute-position gating must be conditioned on something else (flag:
   high risk).
5. **Marker identity:** use column names / file field names if they exist (e.g. heel/toe/ankle L/R). If unnamed,
   derive foot markers in-script from train only: the 4–6 markers with lowest mean height; split into L/R by sign of
   lateral offset from the pelvis centroid; assert stable assignment on every train trial.
6. **Missingness / occlusion:** count NaN or zero-filled markers; fill per trial by linear interpolation along time
   + a binary mask channel (row-local).
7. **Duplicates / windows:** hash frames; check whether consecutive trials share frames (overlapping windows of one
   recording). Chain them into the person group either way.
8. **Edge effects:** fraction of trials whose force is non-zero at frame 0 / 255 (contact already in progress):
   finite differences at the edges need reflective padding plus an "edge distance" channel.
9. **All-zero trials:** count trials with `sum|y_axis| == 0` for any axis (metric division by zero; decide decode).
10. **Irreducible ambiguity / ceilings (train oracles):**
    - Zero predictor: exactly 1.0 per axis (floor every model must beat by far).
    - Newton-only with per-person oracle mass and *no* gating: expected poor (maybe 0.4–0.8 on vertical), shows how much
      the gating carries.
    - Newton with oracle mass and oracle gates (gates = clipped `ratio(t)`): ceiling of the physics parameterisation
      for vertical; residual = marker noise + CoM error. For shear axes the same with fitted per-frame D.
    - Person-disjoint vs within-person gap of a plain model: measures how much person identity (mass, style) matters.
11. **Leakage suspects:** any column in `train.csv` beyond id/motion_file (person id, trial number, walking speed,
    side) – known at prediction time? Person id must never be a feature (unseen at test, fragile). File name / order
    must never be a feature.

---

## Validation design

- **Split:** GroupKFold by **person**, K = 8 folds of 4 people each (A: 32 people), **2 split seeds** for decisions
  (re-drawn person→fold assignment), paired comparisons on identical folds, Nadeau-Bengio corrected t (V16) plus sign
  consistency across >= 6/8 folds before accepting a change. Groups = person id if provided; otherwise derived in-script
  by union-find over (a) overlapping frame windows and (b) anthropometric fingerprint (median over frames of rigid
  intra-segment marker distances, e.g. foot length, pelvis width, shank length; near-identical fingerprint → same
  person), calibrated so that the number of clusters ≈ 32.
- **Bias direction:**
  - If the hidden test is person-disjoint: person-GroupKFold is roughly unbiased, slightly *pessimistic* (trains on
    28/32 people vs 32 for the shipped model).
  - If the hidden test contains the same people (unknown): person-GroupKFold is *pessimistic* by the mass/style
    memorisation gap; random-trial KFold is the matching proxy. Report both; decide mass-head flexibility under the
    person split (the conservative reading costs only on the mass term, which I will quantify as the person-vs-random
    gap of the mass head alone).
  - Trial-level random KFold on overlapping windows would be strongly *optimistic*; never used for decisions.
- **Sanity holdout:** 4 people (one fold) untouched by every selection step until the final check.
- **Metric re-implementation** (exact formula above, both hard cap and soft-slope variants) with unit tests:
  perfect → 0; zeros → exactly 1 per axis; `-y` → capped 1; `2y` → 1; constant train-mean waveform → documented
  baseline; all-zero-target trial → defined eps behaviour (match whatever the description says; default eps=1e-8 and
  log the count). Report per-axis, per-person, per-slice (full-contact trials vs partial-contact trials vs no-contact).
- **Nested / cross-fitted post-hoc steps:** per-axis shrink factors, blend weights and mass-head ridge strength are
  fitted on OOF of folds A (half the people) and scored on folds B, swapped; the cross-fitted number is what is
  reported and what decides "ship or not".
- **Expected proxy-vs-private gap:** person-split CV should land within ~±0.02 of private if the split hypothesis is
  right; if the test shares people, private will be *better* than CV.

---

## Overfit/underfit risks

Effective sample size: **32 people** for anything person-level (mass, style), but **tens of thousands of contact
events and ~10^5–10^6 frames** for frame-level physics (gating, splitting). Different parts sit on different rungs.

| Risk | Kind | Mitigation |
|---|---|---|
| Mass/scale head memorises person fingerprints (32 samples) | overfit | head = linear on 4–8 standardised anthropometric features with strong weight decay; select strength by person-CV from a wide grid incl. very strong end; compare with "constant mass" |
| Plate map overfits to sparse foot positions (edges seen rarely) | overfit | shared tiny MLP on (x,y) with low-frequency Fourier features + smoothness by weight decay; per-marker offset augmentation jitters positions by a few mm so edges are learned softly |
| Backbone memorises person-specific gait | overfit | small TCN (~150–300k params), dropout, weight decay, EMA, fixed epochs; person-disjoint CV catches it |
| Many decode/blend knobs on OOF | overfit | <= 3 per-axis shrink scalars + <= 2 blend weights per axis, cross-fitted |
| Noisy second derivatives of markers make W0 useless | underfit | multiple fixed Savitzky-Golay smoothing widths as parallel channels (nuisance axis F9/D8) + learned temporal conv on top; the network picks |
| Absolute coordinates dropped by "normalise per trial" preprocessing | underfit (fatal) | keep absolute lab-frame foot positions as a separate input stream; only the pose stream is pelvis-centred |
| Down-sampling time to save CPU loses impact transients | underfit | full 256-frame resolution at output; backbone may run at stride 2 internally but the gate/output head works at full rate with skip connection |
| Loss mismatched to metric (MSE) | underfit on metric | train directly on the metric (soft-cap relative L1, per trial-axis), short MSE/Huber warm-up only for the first epochs |
| Too-small CPU budget forces tiny model | underfit | profile one epoch first; spend budget on the primary, not on extra members |
| Coordinate frame differs across sessions | shift | diagnostic 4; if true, gate input becomes foot position relative to a per-trial learned anchor (e.g. trial start) and the plan downgrades gating to contact-only + learned prior |

---

## Recommended approach (primary + fallback)

### Primary: physics-structured temporal network ("Newton + split + gates") trained on the metric

Compliance: clean (from-scratch trained neural network; physics only as output parameterisation; all fitting on
train). Pattern ids: F8/C11 (domain equation as architecture), F2-2 (explicit latent structure), F2-5 (inductive bias
where evidence lives: local in time, absolute in space for the plate map), A2/A7 (metric weights in loss), A17
(per-slot calibration), G12 (fixed schedule + EMA), G11 (augmentation consistent with geometry), D1 (diverse members).

**Preprocessing (row-local, per trial):**
1. Load `(256,22,3)`; interpolate missing markers along time; mask channel.
2. Convert to one physical unit (metres) by an in-script train-derived scale check (assert one unit across trials).
3. Kinematics via fixed Savitzky-Golay filters at 3 widths (e.g. 7, 15, 31 frames; polyorder 3): smoothed position,
   velocity, acceleration per marker per axis, with reflective edges + "distance to sequence edge" channel.
4. Two input streams:
   - *Pose stream* (translation-invariant): marker positions minus pelvis/marker-centroid per frame, velocities,
     accelerations (all markers, all widths) → ~22×3×(1+2×3) ≈ 460 channels; standardised with train-fit means/stds.
   - *Floor stream* (absolute lab frame): horizontal positions and heights of foot markers (L and R separately), their
     velocities; absolute horizontal position of the centroid.
5. Per-trial anthropometric vector (row-local): median over frames of ~6–10 rigid inter-marker distances (foot length,
   shank, thigh, pelvis width, shoulder width, standing height proxy).

**Model:**
- *CoM acceleration:* `a_com(t) = sum_m softmax(w)_m * acc_m(t)` with learned marker weights `w` (22 params, init
  uniform) applied to a learned convex mix of the 3 SG widths. Newton term
  `W0(t) = s_trial * (A @ a_com(t) + b)` where `A` (3×3) and `b` (3, absorbs gravity and force sign/axis conventions)
  are learned, initialised from the train-diagnosed vertical axis/sign, and `s_trial = exp(mass head(anthropometrics))`
  (linear, strongly regularised; collapses to a constant if forces are BW-normalised).
- *Backbone:* non-causal dilated residual TCN over 256 frames on the pose stream + floor stream (hidden 64–96,
  6–8 blocks, kernel 5, dilations 1,2,4,8,16,32 → receptive field covering a full stride; dropout 0.1; GroupNorm).
  Rationale: evidence for contact/split is local in time (heel strike, toe-off), so translation-equivariant in time.
- *Heads per frame:*
  - `dW(t)` residual correction to W0 (3 axes, init 0).
  - `D(t)`: between-feet difference (3 axes), so `F_R = (W + D)/2`, `F_L = (W - D)/2` with `W = W0 + dW`.
  - Gates `g_R(t), g_L(t) = sigmoid(contact_logit_f(t) + plate_logit(x_f(t), y_f(t)))`, where `plate_logit` is ONE
    shared tiny MLP (2→32→32→1 on Fourier-featured (x, y) of the foot reference point, mirrored per foot) = the learned
    invisible floor; `contact_logit_f` comes from the backbone (foot height/velocity evidence).
  - Output `F(t) = g_R(t) * F_R(t) + g_L(t) * F_L(t)` (3 axes).
  - Optional exact-marginal variant (roadmap step 4b): treat (R on plate?, L on plate?) as a 4-state latent with
    softmax probabilities and output the expectation. With independent sigmoids it is the same expectation; the variant
    only matters if the states are made mutually dependent (e.g. a foot spanning two plates). Keep sigmoids in the
    primary.
- Parameter count ~0.2–0.4M; learned physics parameters (w, A, b, mass head) ~40 — state them in the log.

**Loss:** the metric surrogate per trial-axis: `r = sum|F - y| / (sum|y| + eps)`, `loss = r if r <= 1 else
1 + 0.1*(r - 1)`, averaged over axes and trials (A2/A7 weighting automatic). Warm-up: first 3 epochs add a small
Huber term on standardised forces for stable gradients. Auxiliary (G13, small weight 0.1): total-force consistency
`|W - y|` on frames whose OOF-independent train label ratio says "full contact" (train labels only; derived per batch
from the batch's own targets, no cross-trial statistic) — tests whether it helps; drop if no paired gain.

**Training recipe (fixed plan):** AdamW (lr 2e-3, wd 1e-4 on backbone, strong wd on mass head), batch 32 trials,
cosine schedule with 2-epoch warm-up, fixed **40 epochs** (A; set after profiling), EMA decay 0.999 evaluated at the
end, grad clip 1.0, seeds fixed. Augmentation: (i) per-marker constant 3D offset per trial (std ~5 mm, train-only;
leaves velocity/acceleration and labels unchanged and imitates marker-placement variation; CHALLENGE.md cites rank-1
using it); (ii) small global horizontal translation (std ~1–2 cm) of *all* markers, which changes only the floor stream
slightly (teaches the plate map soft edges; keep only if paired CV gain); (iii) no mirroring / no rotation by default
(plate layout breaks symmetry; see roadmap diagnostics).

**Ensemble and shipping:** 8 person-fold models (each on 28 people) give OOF for calibration and form the shipped
average (fold-averaging chosen over a 100% refit because CPU budget cannot afford both; the 12.5% data loss is cheaper
than a second full pass, and the 8-model average is variance reduction). Roadmap step 6 tests "refit on 100% × 2 seeds"
against "8-fold average" if the budget allows; ship the measured winner.

**Second member (diversity of assumption, D1):** frame-level **LightGBM** (objective `l1`, sample weight
`1/sum|y_trial_axis|` to mimic the metric) on hand features: W0 per axis from uniform marker weights at 3 smoothing
widths, foot positions/heights/velocities L and R, foot-to-CoM horizontal offsets, lags/leads ±(2,5,10,20) frames,
window stats (C4), anthropometrics. One model per axis, fixed rounds (set by inner CV in-script once: mean best
iteration × 1.1), `deterministic=True`, fixed threads. It encodes no explicit gating structure, so its errors differ.
Blend per axis: `p = alpha_a * NN + (1 - alpha_a) * GBDT` with alpha on a grid {0, 0.1, …, 1} cross-fitted by person
halves; ship the NN alone if the GBDT does not earn a cross-fitted gain beyond noise. The NN must stay the majority
member (compliance legibility: the primary is the structured network).

**Decode:** per-axis shrink `p_a ← c_a * p_a`, `c_a` in [0.7, 1.1] fitted on OOF, cross-fitted (A17). For trials the
model is confident are no-contact (both gates < τ over all frames), emit exact zeros only if all-zero targets exist in
train and the cross-fitted metric improves. Serialise per sample grammar.

### Fallback (if gating diagnostics fail: plate map not fixed in lab frame, or structured model unstable)

Plain dilated TCN (same streams, same loss, same augmentation, no physics heads) predicting `F(t)` directly, with W0
(uniform-weight Newton estimate) fed as an *input channel* (physics as feature rather than as architecture), + the same
LightGBM member and per-axis shrink. Clean compliance; expected a few points worse on vertical, similar on shear.

**Expected score (estimate, not a promise):** relative-L1 error averaged over axes ~0.20–0.35 for the primary on
person-split CV (vertical ~0.08–0.15 if gates are learned well and mass is identifiable/normalised; shear ~0.25–0.5
each). Reasoning: vertical force is mostly explained by Newton × gate; shear components are small, noisy and depend on
the split between feet, so they will dominate the error. Zero baseline = 1.0. If forces are in Newtons and test is
person-disjoint, add ~0.03–0.06 on vertical for mass uncertainty.

---

## Rejected options

- **Hand-coded plate rectangles / contact thresholds (rule-based gating):** strip-the-ML failure and hard-coded data
  constants (Q1, Q3). The plate map must be a trained sub-network.
- **Pure Newtonian estimate (no learning):** rule-based and also wrong (target is plate-only force).
- **Large transformer / bidirectional LSTM from scratch as primary:** CPU-slow, 32 people → memorisation risk; the TCN
  with physics heads encodes the structure with fewer parameters (principle 5, F2-13 primary-design gate). Kept as a
  roadmap diversity member only if budget remains and it earns a paired gain.
- **Pretrained time-series foundation models (Chronos, Moirai, TimesFM) or HF pose models:** inference-only or grey,
  domain mismatch, CPU cost; no evidence they help on mocap→GRF.
- **GBDT as primary:** poor at temporal context and multiplicative gate × force interaction; demoted to member.
- **Left/right mirror augmentation by default:** the plate layout is not guaranteed symmetric; mirrored trials would
  carry wrong labels. Only after a train diagnostic proves symmetry (F2-6: measure, don't assume).
- **Time-reversal augmentation:** physically `F = m a` is time-reversal symmetric, but braking/propulsion split and
  plate-dynamics artefacts are not; roadmap experiment only.
- **Per-person pooling at test (cluster test trials by fingerprint, average mass estimate):** whole-test aggregation,
  banned (§2.3 #5, R). Mass head is row-local.
- **Person-id / trial-order / filename features:** banned/fragile.
- **Optuna HPO over architecture:** 32 people cannot resolve many trials; selection noise. Fixed design + a tiny,
  cross-fitted grid for mass-head strength and decode scalars.
- **Frame-level random CV:** leaks within-person and overlapping windows (optimistic).

---

## Fixed work plan & runtime budget

All counts fixed constants; time used only for logging; `device="cpu"`; `torch.set_num_threads(10)` (A: 10 cores;
set to the description's stated core count), `torch.use_deterministic_algorithms(True, warn_only=True)`; seeds for
random/numpy/torch/DataLoader Generator; `num_workers=0` (data in memory); LightGBM `num_threads=10,
deterministic=True, force_row_wise=True, seed`.

| Stage | Work | Estimate (10 CPU cores, assuming ~2,000 trials; scale linearly) |
|---|---|---|
| Load + validate inputs | read csv + motion files, assert shapes `(256,22,3)`, finite after fill | 1–2 min |
| Feature tensors | SG filters, streams, anthropometrics (vectorised numpy) | ~1 min |
| Primary NN, 8 folds × 1 seed × 40 epochs | TCN ~0.3M params; ~0.4 GFLOP per trial fwd+bwd → ~6–10 s/epoch | 8 × ~5 min ≈ 30–35 min (**too tight: profile; if > 4 min/fold reduce to 6 folds or hidden 64 / internal stride 2**) |
| LightGBM per axis | ~500k frame rows × ~150 features, 8 folds + reuse fold models for test | 4–6 min |
| Calibration + blend (cross-fitted) | numpy | < 1 min |
| Predict test + validate + write | 8 NN + 24 GBDT models | 1–2 min |
| **Total** | | **target 35–40 min, >= 30% headroom vs 1 h** |

Memory: 2,000 × 256 × ~500 channels float32 ≈ 1 GB; GBDT frame matrix ~0.6 GB; fine within any CPU box (A: 62 GB).

Script validations: assert train/test schema, ids unique, motion files exist, array shape `(256,22,3)` (or the shape
observed in train: assert identical in test), no NaN after fill; output: columns/order/ids identical to sample, each
cell parses to 256 finite floats (or long format with exactly 256 frames per id), reload with
`keep_default_na=False`; raise before writing on any failure. No placeholder/fallback output.

---

## Metric-aware training & decode

- **Loss = metric** (per trial-axis relative L1, soft cap slope 0.1 beyond 1) → metric's own per-example weights are
  applied automatically (A2/A7). Per-frame L1 → median-seeking; good for spiky impacts.
- **Cap-aware shrinkage:** because zero scores exactly the cap, the loss itself pulls uncertain shear predictions
  toward 0. Post-hoc per-axis scalar `c_a` (A17) fitted on OOF, cross-fitted; expected `c_a` ≈ 1 for vertical, < 1 for
  medio-lateral.
- **Structure in the loss, not the decode:** gating/splitting are inside the forward model, so the decode is trivial
  (no search). The only decode constants: 3 shrink scalars + up to 3 blend alphas (6 total, well under the ~6-knob
  limit for a few hundred-thousand OOF frames grouped in 32 people).
- **Oracle checks (F2-15):** (a) gold targets serialised and re-read through the validator and the metric → exactly 0
  error; (b) Newton + oracle gates + oracle mass on train → ceiling of the parameterisation; (c) zero predictor = 1.0;
  (d) uniform-weight Newton without gates = the "physics-only" yardstick the trained model must beat by a wide margin
  (strip-the-ML evidence, logged).
- **Back-solving (A1):** the metric's normaliser is the label's own L1 mass: nothing to back-solve; but the latent
  per-person mass and per-frame gate are "hidden quantities behind the target" reconstructed by the diagnostics and
  learned by the heads.
- **Per-axis model choice:** if GBDT beats the NN on one axis under cross-fitted OOF, alpha handles it (per-output
  choice, F2-1).

---

## Structural signals

1. **Newton's second law** `sum F = m (a_com - g)`: architecture base (F8). Learned CoM marker weights and learned
   axis/sign map `A, b` mean no hard-coded conventions.
2. **Per-foot superposition + plate gating** (the description's "invisible floor"): `F = g_R F_R + g_L F_L`,
   `F_R + F_L = W`. Gates learned from absolute foot position (plate map, shared across feet: one floor) × contact
   evidence (foot height/velocity).
3. **Floor is fixed in lab coordinates** (to verify, Data findings 4): shared plate MLP across feet and trials;
   translation augmentation kept tiny so the map stays absolute.
4. **Rigid per-marker offsets preserve kinematics and labels:** per-marker offset augmentation (CHALLENGE.md cites
   rank-1); transforms every geometry-tied input consistently (G11) because velocity/acceleration are recomputed
   from the offset positions.
5. **Left/right foot interchangeability:** the plate map and contact head share weights across feet (mirrored lateral
   coordinate for the contact head only, not for the plate map). A train diagnostic checks lateral mirror symmetry of
   the plate layout before any mirror augmentation is tried.
6. **Body mass constant per person:** row-local mass head on anthropometrics; never pooled across test trials.
7. **Temporal locality:** contact events are local; non-causal TCN, translation-equivariant in time; edge channel for
   frames 0/255.
8. **Physical sanity (assert on train, not enforced as rules):** vertical plate force has one sign; when both gates are
   ~0 the force is ~0; shear magnitude << vertical. Logged as checks on OOF predictions.

---

## Experiment roadmap

Each step: one change, paired on identical person folds (8 folds × 2 split seeds), accept only on corrected-t
gain + sign consistency; log to `reports/experiments.md`.

1. **Contract & diagnostics** (stop when all pass): confirm file formats, sample grammar, axis/unit conventions,
   metric re-implementation + unit tests, gold-roundtrip oracle = 0, Data findings 3/4/7/9/10, people grouping (32?).
   Decision gates: BW-normalised vs Newtons (mass head on/off); plate layout fixed (primary) vs session-dependent
   (fallback).
2. **Cheap baselines end-to-end, valid file:** zero (1.0), physics-only uniform Newton, LightGBM frame model. Record
   per-axis OOF. First credit: LightGBM baseline only if it beats physics-only clearly (valid pipeline proof).
3. **Primary structure:** plain TCN (fallback) → + W0 base → + D split → + gates with plate MLP. Each step paired.
   Expect the gates to be the largest single gain on vertical.
4. **Metric-aware:** loss = soft-capped relative L1 vs Huber (paired); cap slope {0.05, 0.1, 0.3}; per-axis shrink
   cross-fitted; (4b) exact 4-state gate latent only if (3) shows gate errors at plate edges / double-plate cases.
5. **Augmentation & robustness:** per-marker offsets (std 2/5/10 mm), tiny global translation, then the
   diagnostic-gated experiments: mirror (only if the layout is symmetric), time reversal.
6. **Diversity & shipping form:** blend NN + GBDT per axis (cross-fitted); 8-fold average vs 100% refit × 2 seeds
   (if runtime allows); a second NN seed per fold only if seed std > half the targeted gain.
7. **Bounded in-script selection:** mass-head ridge grid (5 values, strong end included) inside the script on inner
   person folds; nothing else searched.
8. **Final:** freeze constants, full run from clean `working/`, run twice and diff (bit-identical expected on CPU with
   fixed threads), half-rows test, compliance scan, validator. Credits: baseline → primary single → blend → final.
9. **Fresh-seed finalist check (P-A09):** re-score top 2 configs on 3 fresh person splits + sanity holdout.

---

## Compliance audit

- Test file used only for ids/row count and one-trial-at-a-time inference: YES (design).
- Scalers/encoders fitted on train only; no test statistics, no pooling across test trials: YES.
- No wall-clock in control flow; no hardware/env branches (CPU hard-coded); fixed epochs/folds/rounds/threads; seeds:
  YES (design); verify with double-run diff.
- No hard-coded tuned constants: shrink/blend/mass-head strength fitted in-script; architecture/epochs are fixed design
  defaults documented in the docstring with their profiling rationale.
- No external data, no pretrained weights, no synthetic labelled data (offset augmentation modifies real samples
  within training): YES.
- Strip-the-ML: physics-only yardstick logged and far worse than the trained model (to be shown in OOF table).
- No person id / filename / order features; groups used only for CV.
- Readable source well under 512 KB; docstring requirements map (CPU-only, training in script, metric, no test use).
- Output written once at the end after validation; failures raise.

---

## Open questions & assumptions

Reviewer / description questions:
1. Is the hidden test person-disjoint from train? (Changes how flexible the mass head may be; plan is safe under both.)
2. Official metric: hard `min(r,1)` or soft slope? Handling of trials with zero true force on an axis?
3. Exact submission grammar (JSON arrays per axis per trial vs long format; float formatting).
4. Runtime limit and core count for the CPU-only environment (plan assumes 10 cores, <= 1 h).
5. Learning the plate layout from absolute foot coordinates (a property of the lab, learned from train labels) is
   assumed legitimate as the task's stated physical structure; confirm it is not considered "exploiting data
   generation".

Assumptions (all unverified, each with its diagnostic in Data findings): `(256,22,3)` marker arrays via
`motion_file`; three force axes × 256 frames; 32 training people; forces possibly in Newtons; plates fixed in lab
frame; foot markers identifiable by name or height; no overlapping windows across the people split; CPU-only.

What I could not verify: anything about the actual data (sizes, units, axes, missingness), the official metric and
submission grammar, the runtime limit, whether test people overlap train people, and the CPU epoch time behind the
runtime table (must be profiled before fixing epochs/folds).
