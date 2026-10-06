# final_approach.md — KineProof

Phase 17 deliverable, written before the final training run. Scores quoted from
`reports/dataset_audit.md` §8 (cross-person 5-fold, exact official metric) and
from the in-script held-out-person validation printed by `solution.py`.

---

## 1. Problem understanding

Given 256 frames of 22 lower-body marker positions (metres, origin = frame-0
ASIS midpoint), reconstruct the 3-axis ground-reaction-force waveform at the
same 256 time points, in body-weight units. 1,093 train trials from 32 people;
322 test trials from **10 people never seen in training**.

The metric is a per-trial, per-axis **normalised L1 skill**:

```
E_ia = Σ_t |p_ita − y_ita| / Σ_t |y_ita|      score = mean_ia max(0, 1 − E_ia)
```

Three properties of this metric drive everything:
- Each axis is normalised by **its own** Σ|y| and then weighted equally.
  Measured denominators: force_y mean 209.4, force_x 18.4, force_z 9.5 — i.e.
  **11.4× and 22.0× smaller** for the horizontals. Vertical-only reconstruction
  caps at one third, as the description states.
- It is scale-sensitive and timing-sensitive in equal measure: a 1-sample shift
  of a pure spike scores **0**, a 1.5× amplitude error scores 0.5.
- A single non-finite value zeroes that whole row; a wrong header/row count/id
  raises a global error. Output grammar is a hard gate, not a soft penalty.

## 2. Platform findings (detail: `platform_analysis.md`)

- Submissions cost a credit each and **there is no free score probe**, so every
  selection decision is made on local cross-person CV.
- The **private** slice decides rank; the printed AI baseline (≈ 0.60) is the
  payout gate; review happens after close, so compliance-adjusted value is what
  matters.
- **Deterministic Execution** has actually blocked submissions in this repo's
  history, every time for wall-clock-dependent control flow. Fixed work plan;
  time for `print()` only; no device/env branches; no fallback write.
- Patterns that converged across unrelated challenges and apply here: put the
  metric in the loss; mirror the hidden split in CV; representation before
  machinery; refit on 100 % of train with fixed counts; diversity over seeds.

## 3. Dataset findings (detail: `dataset_audit.md`)

- Bundle is clean: 1093/322, all `(256,22,3) float32`, all finite, no duplicate
  or shared arrays between splits, ids opaque and person-unrecoverable by design.
- **Marker axes align 1:1 with force axes** (x forward — net +2.05 m, positive in
  100 % of trials; y vertical; z lateral), recovered from the motion itself.
- Sampling rate is **150 Hz** (256 frames / 1.7 s).
- **Persons are recoverable almost exactly from anthropometry**: the dendrogram
  has a natural gap at precisely 32 clusters (merge heights 2.667 → 1.540),
  within-cluster segment spread 1.40 mm vs 22.05 mm population and *below* the
  4.25 mm single-trial marker noise. Random folds are 17× more leaked than
  grouped folds (val→train NN distance 0.195 vs 3.318).
- **Mass cancels** in the target: `f = a_com/g` (+1 vertically). The target is a
  pure kinematic quantity, which is the structural reason unseen walkers are
  tractable. An untrained Newtonian estimate already scores **0.616 on force_y**.
- 9.8 % of frames are effectively unloaded and 60.8 % of trials contain an
  unloaded stretch, so there is a **contact envelope** to learn on top of the
  waveform — and because absolute lab position was removed, where the plates are
  is unobservable. That is an irreducible uncertainty floor.

## 4. Private-LB reference findings

**None available.** The supplied reference file is for a different challenge
(cross-lingual legal-provision ranking). See `private_lb_reference_analysis.md`.
No part of this approach derives from a reference solution; what I took from the
file was methodological (run the strip-the-ML check rather than assert it; log
the no-model baseline permanently) and one explicit rejection (its wall-clock
training safeguard is the exact pattern the platform checker rejects).

## 5. Validation strategy

- **Person-disjoint, always.** Persons derived by average-linkage clustering of
  25 median-over-time rigid segment lengths, cut at the 32 the description
  states. Whole persons to folds, balanced on trial count.
- Dev harness: 5-fold cross-person CV → fold sizes `[214,213,213,220,233]`
  trials, `[6,6,6,7,7]` people. Reported as mean ± per-fold sd, plus per-person
  worst/median/best, because the private set is only 10 people and the
  person-to-person spread is the real risk.
- In `solution.py`: one person-disjoint holdout (7 whole persons, 311 trials)
  trained with the identical fixed plan, scored with the exact metric, printed,
  and compared against the best constant waveform — then the model is refit on
  100 % of train. The holdout is **diagnostic only**; nothing is selected on it,
  so no selection optimism enters.
- **Bias direction:** clusters coarser than true persons ⇒ pessimistic, finer ⇒
  optimistic. Evidence says the granularity is right; the residual risk (a
  56-trial cluster that is really two similar people) biases *pessimistic*,
  the safe direction.
- Deliberate omission: no train-vs-test distribution comparison (see §10).

## 6. Feature representation

187 per-frame channels + 27 per-trial scalars, all local-in-time functions of a
single trial's own markers (so no statistic crosses rows or the split):

1. **The Newtonian estimate itself, in target units** — `a/g` and `a/g + 1` from
   four COM/pelvis variants at two smoothing widths. Handing the model the
   physics rather than making it rediscover it.
2. COM and pelvis velocity, height, horizontal speed.
3. All 22 marker positions in the **moving pelvis frame** (translation invariant).
4. Foot/shank/knee velocity and acceleration in the lab frame — these carry the
   loading transients that create the force peaks.
5. Explicit **contact cues**: foot height above its own trial minimum, foot
   speed, fore-aft foot position relative to pelvis, inter-foot geometry. This
   is what a biomechanist reads for heel-strike and toe-off, and it is what the
   off-plate envelope needs.
6. Six joint angles in cosine form (smooth, bounded, no `arccos`).

Derivatives use **Savitzky–Golay** filters (windows 9/15/31 frames = 60/100/207 ms
at 150 Hz, cubic), because plain finite differences amplify marker noise by
1/dt². Verified exact on polynomials, identical to scipy, and correct on a
sine-response check.

Per-trial scalars: segment lengths, body proportions (ratios to leg length),
pelvis height, walking speed, cadence from the dominant vertical-pelvis
frequency, and acceleration-spread summaries. Because the target is
mass-normalised, what matters is the walker's mass *distribution* and gait,
not mass.

## 7. Model architecture

**Dilated residual temporal CNN** (`ForceTCN`): stem conv → 6 residual blocks of
2× (conv k=5, GroupNorm, GELU) with dilations 1,2,4,8,16,32 → 1×1 head to 3
channels. Width 72. Receptive field `1 + 4·(1+2+4+8+16+32) = 253` frames ≈ the
whole window.

Why this rather than a transformer or a large seq2seq model:
- 1,093 trials from **32** people set the capacity, not 1,093 rows. Distinct
  people are the independent unit.
- The target is a local-in-time physical map (force at *t* is COM acceleration
  at *t*) plus medium-range gait-phase context — exactly a dilated conv's
  inductive bias.
- The 256-frame window is an arbitrary 1.7 s cut of a longer trial, so
  **translation equivariance along time is correct**; a model with absolute
  positional encoding would learn window-position artefacts that cannot transfer.
- The repo's recorded pattern: in six prior problems the simplest solution won
  while more machinery ranked in the bottom half.

**Physics residual base.** The output layer is zero-initialised and the model's
output is `learned_correction + Newtonian_prior`. Training therefore *starts
exactly at the 0.616-on-y physics solution* and only ever learns the correction
from a lower-body COM proxy to the true whole-body COM, plus the contact
envelope. This is the "domain theory as architecture" pattern (playbook F8).

## 8. Target representation

**Direct 256-point time-domain prediction**, deliberately not a basis.
Oracle-projection measurement settles it: PCA/DCT at 32 components caps the
score at 0.910/0.915, at 64 at 0.952/0.958, and even 128 of 256 DCT
coefficients caps it at 0.982. A basis target would impose a ceiling the metric
can see, because the metric punishes precisely the peaks that smoothing removes.

## 9. Loss

The loss **is** the metric's error term:

```
loss = mean_{i,a} [ Σ_t |p − y| / Σ_t |y| ]   ==  1 − score   (before clipping)
```

with a small Huber region (0.02 BW) to smooth the L1 kink early in optimisation.
Minimising it maximises the official score directly, and it reproduces the
metric's axis balancing exactly — a plain L1/MSE would be dominated by force_y
and cap the result near one third. `Σ_t|y|` is a label-derived per-trial-axis
weight used at **training time only**; inference never needs it.

AdamW, one-cycle LR (max 3e-3, 15 % warmup), weight decay 1e-4, grad-clip 1.0,
28 epochs, batch 32 — all fixed, no clock, no validation-triggered stop.

## 10. Augmentation

**One augmentation, physically exact: left/right mirror.** Swap every
contralateral marker and negate the lateral axis; then force_x and force_y are
unchanged and force_z is negated. This maps a real walking trial to another real
walking trial — it fabricates nothing — and doubles the data. `solution.py`
asserts the symmetry holds on the targets before using it.

Rejected: spatial rescaling (gravity does not scale, so body-weight-normalised
forces do not transform cleanly); time reversal (not physical); synthetic
trials (banned). Uniform time-rescaling *is* exactly derivable
(`f_x → s²f_x`, `f_y → s²(f_y−1)+1`) and is the most promising untested lever —
see §14.

## 11. Retrieval

**Measured and rejected as a primary or a member.** kNN retrieval on trial
scalars scores 0.274 (K=15) and the *oracle* same-person median waveform only
0.285, barely above the global median's 0.242 and far below a per-frame linear
model's 0.486. On this data the force is set by the individual trial's
kinematics, not by any template — so retrieval has little to contribute. This
also means cross-person generalisation is **less** dangerous than the split
design suggests: the map is largely person-independent physics.

## 12. Ensemble and post-processing

3 full-data refits at different seeds, **simple average** — variance reduction
with no fitted weights, which is what the literature favours under noise
(estimating blend weights adds variance; forward selection overfits).
**No post-processing at all**: no smoothing, no clipping, no calibration. Each
would need its constants fitted on OOF with a nested check, and none showed a
principled reason to exist. Predictions are written exactly as the model emits
them, at 6 decimals.

## 13. Anti-overfitting measures

- Person-disjoint validation everywhere; random CV never used for any decision.
- Capacity set by **32 people**, not 1,093 rows: 6 blocks × width 72 (~0.5 M
  parameters), dropout 0.05, weight decay 1e-4.
- The physics residual base means the model is regularised toward a correct
  solution rather than toward zero.
- Translation-equivariant architecture — no window-position artefact to exploit.
- Fixed epoch count (regularise by *where you stop*), not validation-triggered
  early stopping.
- **No selection was performed on the holdout**: the config was fixed from the
  baseline measurements and a single architecture benchmark, so there is no
  winner's-curse correction to apply (N candidates compared on the holdout = 1).
- Runtime assertions that the model beats the constant-waveform baseline and
  that predictions are non-degenerate.

## 14. Validation evidence and expected score

Measured cross-person, exact metric (`dataset_audit.md` §8):

| | total | x | y | z |
|---|---|---|---|---|
| all-zero | 0.000 | 0.000 | 0.000 | 0.000 |
| constant median waveform | 0.242 | 0.052 | 0.619 | 0.054 |
| untrained Newtonian | 0.281–0.298 | 0.001 | 0.654 | 0.186–0.255 |
| kNN retrieval / oracle-person | 0.274 / 0.285 | | | |
| per-frame linear ridge | 0.486 | 0.414 | 0.612 | 0.431 |
| **TCN, person-disjoint holdout (7 unseen people, 1 seed, 782 trials)** | **0.5816** | 0.5085 | 0.7365 | 0.4997 |

**Expected private score — stated as an estimate with its uncertainty, not a
prediction.** The three quantities must be kept separate:
- *Measured*: the held-out-person score printed by `solution.py` on 7 unseen
  training people (and the 5-fold number in `reports/experiments.csv`).
- *Estimated generalisation*: the private set is **10 people**, so
  person-to-person spread dominates the error bar. Group splits are noisier than
  random ones, and the literature's typical ID→OOD level drop is a few points.
- *Uncertain assumptions*: that my derived person clusters match the real
  people (strongly evidenced, not proven); that the 10 held-out walkers are not
  systematically different in speed/size/gait from the 32 training ones (**not
  checked, by choice** — see §10 of the audit).

### Measured result and what it does and does not say

**Measured: 0.5816** total (x 0.5085, y 0.7365, z 0.4997) on 7 whole unseen
people (311 trials), exact official metric. Reference points: best constant
waveform 0.2469 (**+0.3347**), per-frame linear ridge 0.486 (**+0.096**),
printed AI baseline ≈ 0.60 (**−0.018**).

**Per-unseen-person scores: 0.379, 0.496, 0.546, 0.585, 0.657, 0.659, 0.687.**
This spread of 0.31 between the best and worst walker is the most important
number in this report. The private set is 10 people; with a per-person sd of
about 0.10, the standard error of a 10-person mean is roughly 0.03, so the
private score could plausibly land anywhere in a band several hundredths wide
purely from which walkers were drawn.

Three reasons the shipped model should score **above** 0.5816, none of them
measured:
1. the holdout model saw 782 trials from 25 people; the shipped model is refit
   on all 1,093 trials from 32 people (V9 records +0.007…+0.04 from this alone);
2. the shipped model averages 3 seeds, the holdout was a single seed;
3. the holdout deliberately set aside the **7 largest** person clusters, which
   removes 28 % of the training trials — a harder-than-average split, chosen
   because it errs pessimistic.

One reason it could score **below**: the train-vs-validation gap
(train normalised-L1 0.274 ⇒ ~0.726, versus 0.582 held out) shows real
overfitting, and **the epoch count was never validated against a per-epoch
validation curve** — 28 epochs is a fixed plan chosen a priori, not a measured
optimum. This is the single most likely place a quick win is being left behind.

**I do not claim 0.80, and this approach as shipped does not reach it.** Keeping
the prompt's three categories separate:
- *measured validation score*: **0.5816** on 7 unseen people;
- *estimated generalisation*: mid-to-high 0.5s to low 0.6s, driven mainly by the
  full-data refit and seed averaging, with a ±0.03-ish band from the 10-person draw;
- *uncertain assumptions*: that the derived clusters are the real people
  (strongly evidenced, unproven); that the 10 hidden walkers resemble the 32
  training ones (**unchecked by choice**); that 28 epochs is near-optimal (untested).

## 15. Why this approach over the alternatives tested

| Alternative | Why not |
|---|---|
| Constant / mean / median waveform | 0.242 measured; also fails strip-the-ML |
| Retrieval / template / kNN | 0.274 measured, and the *oracle* person template is only 0.285 — the signal is not person-level |
| Ridge on trial scalars | 0.268 — the waveform is not a function of trial summaries |
| Per-frame linear ridge | 0.486 — strong, and the honest linear floor the TCN must beat, but misses nonlinearity and gait-phase context |
| PCA/DCT coefficient target | measured ceiling 0.910 at 32 comps, 0.982 even at 128 — an avoidable cap |
| Transformer / large seq2seq | wrong inductive bias for 32 independent subjects; absolute positions are a window artefact |
| Unweighted L1/MSE loss | would let force_y (11–22× larger denominator) dominate and cap near one third |
| Test-time normalisation, pseudo-labelling, test-based calibration | banned (CLAUDE.md §2.3 #5); half-rows test would fail |

## 16. Remaining risks (honest list)

1. **Phase 4 was never executed** — no KineProof reference exists, so there is no
   external calibration of what a top score looks like.
2. **No multi-seed, multi-split-seed confirmation of the finalist.** With the
   time available I ran a single fixed configuration; the playbook's
   fresh-split-seed confirmation of finalists (V18/P-A09) was not done.
3. **Untested levers**, in the order I would try them: (a) the exactly derivable
   time-rescaling augmentation; (b) a second model family (per-frame GBDT or a
   U-Net with strided down/up-sampling) for genuine assumption diversity;
   (c) width/depth capacity ladder; (d) an explicit auxiliary contact-probability
   head, since the off-plate envelope is a distinct sub-problem; (e) more seeds.
4. **The off-plate envelope is partly unknowable** (absolute plate position was
   removed), which is a real ceiling, not a modelling failure.
5. **Person clustering is an approximation.** Evidence is strong (dendrogram gap
   at exactly 32; within-cluster spread below marker noise) but it is inferred,
   not given.
6. **Train-vs-test shift unmeasured by choice.** If the held-out walkers differ
   systematically, the held-out-person estimate is optimistic in a way I cannot
   bound from train alone.
