# Eris plan: Two Puffs (BLIND run, no dataset available)

Status: blind strategist plan. Sources read: `corpus/two_puffs/CHALLENGE.md` (a reconstruction that is itself marked
uncertain), `CLAUDE.md`, the eris-playbook references. **No data was available.** Every row count, column name,
distribution, metric formula and runtime limit below is an **ASSUMPTION / UNVERIFIED** unless it is quoted from the
CHALLENGE.md text. Where the target definition could plausibly differ, a plan variant is given (section "Open questions").

Problem families: tabular-and-scientific (session indices + demographics, probability targets) + audio-signal (raw
flow waveforms, optional self-supervised waveform encoder) + structured output (9 coupled probabilities that are all
functionals of one joint distribution).

---

## Contract & decision unit

**Decision unit.** One participant = one row of the submission (assumption: one session per participant in the
test file; if a participant can appear in several rows, the unit for grouping is still the person, see Validation).

**One valid answer (assumed grammar, unverified).** An id column plus 9 probabilities in [0, 1]:
`P(dFEV1 >= 0%)`, `P(dFEV1 >= 5%)`, `P(dFEV1 >= 10%)`, `P(dFEV1 >= 15%)`, `P(dFEV1 >= 20%)`,
`P(dFVC >= 0%)`, `P(dFVC >= 5%)`, `P(dFVC >= 10%)`, `P(ATS/ERS positive)` where
`dX = 100 * (X_post - X_pre) / X_pre` and ATS/ERS positive = `(dFEV1 >= 12% AND FEV1_post - FEV1_pre >= 0.2 L) OR
(dFVC >= 12% AND FVC_post - FVC_pre >= 0.2 L)` (CHALLENGE.md). Column names/order are taken from
`sample_submission.csv` at run time and mapped to events by an explicit, asserted name table (raise on any unknown
column, never guess).

**Invalid vs low-scoring.** Invalid: wrong columns/order, row count or id order mismatch, NaN/inf, values outside
[0, 1] (a malformed CSV ranks below 0). Low-scoring but valid: incoherent probabilities (e.g. `P(>=10%) > P(>=5%)`),
poorly calibrated probabilities, constant base rates.

**Metric (assumption).** A proper scoring rule averaged over rows and the 9 columns, most likely mean Brier
(squared error in probability space, consistent with rank-1's gradient `2*(P-T)*w/9`), possibly log loss; targets T
may be binary realised events or soft probabilities ("the label ... is itself a probability"); per-row weights `w`
exist in rank-1's loss but it is unknown whether they come from the metric or the solver. Plan: implement
`brier9(P, T, w=None)` and `logloss9(P, T, w=None, eps)` and train on a loss that is proper for both (squared error
+ soft-target BCE are both minimised by the true conditional probability). If the description gives row weights or
per-column weights, they go into the loss exactly (A2/A7). Calibration-sensitive, not rank-only: the base-rate
component dominates, so calibration is worth as much as discrimination here.

**Pipeline stages.** (1) signal decoding: base64 flow bytes -> flow(t) -> volume(t) with BTPS -> per-blow indices by
the published ATS/ERS algorithms (C12); (2) learned evidence: a conditional *joint distribution* of
(dFEV1, dFVC) given the pre-bronchodilator session and demographics; (3) decode: integrate the 9 events over that
distribution (coherent by construction), then per-slot calibration. Diagnose separately: (1) by oracle checks against
any provided indices/flags/labels; (2) by NLL/CRPS of the continuous gains under grouped CV; (3) by the gap between
"events decoded from the distribution" and "direct per-event classifiers" on the same folds.

**Core structural insight.** The 9 outputs are not 9 independent labels. They are threshold functionals of two
continuous latent quantities (dFEV1, dFVC) plus the pre values in litres (needed for the 0.2 L criterion). The five
FEV1 columns are a discretised CDF, the three FVC columns another, and the ATS column is a union of two half-planes
whose FEV1 boundary is row-specific: `dFEV1 >= max(12, 20 / FEV1_pre[L])`. A model that outputs one joint
distribution and integrates it produces monotone, mutually consistent probabilities and uses the continuous gains
(if given) as a far richer training signal than 9 bits per row (A1, A8, A18; principle F2.2).

## Compliance regime

- **Regime (assumption):** From-scratch + CPU-only. Evidence: rank-2's docstring "No pretrained parameters, stored
  predictions or external records are loaded" and "trains a waveform representation on the unlabeled reference pool";
  both inspected solutions pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and have no CUDA. Plan: `device = "cpu"` hardcoded,
  no HF/timm downloads, no pretrained tokenizers/weights, everything trained on the supplied data in-script.
- **Allowed and used:** LightGBM, PyTorch MLP/1D-CNN trained from scratch, numpy/scipy signal processing,
  scikit-learn transformers fit on train only, in-script HPO with fixed trials.
- **Domain algorithms as features (allowed, C12/F8):** ATS/ERS 2019 spirometry computations (BTPS scaling,
  back-extrapolated time zero, FEV1 at 1 s, FVC, PEF, FEF25-75, MEF75/50/25, FEV0.5/0.75/6, FET, plateau check,
  repeatability) are deterministic signal parsing that feed a trained model. They do not predict the target (the
  bronchodilator response). Strip-the-ML test: remove the trained models and only the pre values remain, with no
  prediction of the post response beyond a train base rate -> passes clearly.
- **Target-definition constants** (thresholds 0/5/10/15/20/12 %, 0.2 L) are part of the task definition, not tuned
  constants; they appear only in the target builder and the event integrator, with a comment citing the description.
- **Banned / avoided:** external data, including **GLI-2012 / NHANES reference-equation tables** (lookup tables
  of spline coefficients are external data; instead a "predicted normal" model is fit in-script on TRAIN (+ the
  supplied unlabeled pool) inputs); any use of test waveforms for self-supervised pretraining, scalers, PCA or the
  predicted-normal fit (CLAUDE.md 2.3 #5); pseudo-labels; synthetic sessions; matching test rows to the pool or to
  train by provenance/ids (2.4 provenance); time/hardware/env branches.
- **Grey item to note:** the unlabeled reference pool is used for representation learning and the predicted-normal
  fit. It is supplied challenge data and not the test set; if the pool turns out to contain the test participants'
  sessions it must be filtered out by an explicit, documented rule from the description (not by matching), or not
  used at all. Reviewer question Q3.
- **Self-audits on this plan:** strip-the-ML passes (above). No whole-test aggregation: every prediction is a
  function of the row's own session + frozen train-fit models; residual banks and calibration maps come from TRAIN
  OOF only; half-rows test must give identical kept-row predictions. Sibling leakage: no feature cross-references a
  row with other participants; repeated visits of one person are grouped. Every constant is either a task-definition
  constant, a documented data-contract fact (byte format, sample rate) asserted on every row, or chosen by an in-script
  train-only search.
- **Ethnicity/sex features:** used as model inputs because they are supplied and physiologically relevant to lung
  volumes; if the description bans them, drop (one-line config, not a branch).

## Data findings

No dataset available. What follows is the **profiling checklist the implementer must run first**, with the decision
each finding drives. Expected values are domain priors, labelled as such.

1. **Files and schema (unverified).** Expect `train.*` (participants with labels), `test.*` (participants without),
   `sample_submission.csv`, possibly a per-blow table (`participant_id, blow_idx, flow_b64, btps, acceptable,
   plateau, effort`) and an unlabeled reference pool. Record: row counts, which columns exist only in train
   (leakage-column audit I3: anything post-bronchodilator present in train but not test is a *label source*, never a
   feature), whether train carries post-BD blows, post-BD summary values, the 9 event columns, or soft probabilities.
2. **Waveform decoding (highest technical risk).** Decode `flow_b64` with candidate dtypes (int16 LE, uint16,
   int8 deltas, float32) and the documented/derivable sample rate; accept a format only if, on every train blow,
   flow is mostly positive during expiration, volume (cumsum * dt * btps) is monotone after PEF, FVC lies in
   0.3-8 L and FET in 1-20 s (physiological ranges, assertion bounds not tuned constants). If train or the pool
   provides reference indices (FEV1/FVC per blow), require our computation to reproduce them (oracle, target:
   |error| < 10 mL on > 99 % of blows) and fit any unknown scale by least squares on train only.
3. **Target profile.** Base rates per event (domain prior, unverified: P(dFEV1 >= 0) ~ 0.65-0.8, P(>= 12 %) ~ 0.1-0.25,
   ATS-positive ~ 0.1-0.3, depends on the cohort: asthma/COPD cohorts much higher than population screens);
   distribution of continuous dFEV1/dFVC (heavy right tail, small negative mass); correlation of dFEV1 and dFVC
   (prior: 0.5-0.7). Ties at exactly 0 % if volumes are quantised to 10 mL: check `>=` vs `>` against provided labels.
4. **Measurement noise.** Within-session spread of FEV1/FVC across acceptable blows (prior: repeatability SD
   ~ 50-150 mL), number of blows and acceptable blows, effort grade counts. This is the irreducible noise floor of
   the gain: SD(dFEV1) from measurement alone ~ 3-6 % for a 2-3 L FEV1, so thresholds 0 % and 5 % are inherently
   uncertain (information ceiling, see V10).
5. **Group structure.** Duplicate participants (same age/sex/height/weight/ethnicity and near-identical blows),
   repeated visits, exact waveform hash duplicates across train rows: union-find via `make_groups.py` on
   (demographic tuple, blow hashes). Report group-size distribution; giant groups (> 5 % rows) are inspected.
6. **Information ceiling.** (a) Constant base-rate Brier per slot = p(1-p); (b) "noise-only" ceiling: simulate the
   gain as pure measurement noise from the observed repeatability to estimate how much of P(dFEV1 >= 0/5 %) is
   unpredictable; (c) if soft targets are given, the Brier floor is E[T(1-T)], not 0. These bound what modelling
   can gain and tell us to spend effort on calibration.
7. **Id/order audit.** Id format, row order vs label (sorted by label or date?), file sizes; a metadata-only model
   (ids, n_blows only) under grouped CV must show no lift beyond what n_blows legitimately carries (V19b).

## Validation design

- **Hidden split (assumption):** random by participant. Alternatives the description might state: by site/device or
  by time. If stated, mirror it exactly (whole sites/periods per fold).
- **Scheme:** `StratifiedGroupKFold(n_splits=5)` x 3 split seeds; groups = union-find participant groups (above);
  strata = ATS label x dFEV1 bin {<0, 0-5, 5-10, 10-15, 15-20, >=20} (or the argmax-bin of soft labels). All 9 events
  have both classes in every fold (B14/V6). Plus a **sanity holdout** of 15 % of groups that touches no selection
  until the final check.
- **Bias direction.** Random-by-participant CV is roughly unbiased if the hidden split is random and repeat visits
  are grouped (slightly optimistic if undetected repeat visits remain). If the hidden split is by site/time, this CV
  is optimistic mainly in calibration (base-rate shift): report also a leave-one-cluster-out score with KMeans
  clusters on demographics + device-like features (V7) as the pessimistic bound.
- **Metric implementation and unit tests:** `brier9` and `logloss9` (+ per-slot breakdown). Tests: perfect
  predictions -> 0 (binary T) or E[T(1-T)] (soft T); reversed (1-T) -> maximum; constant 0.5 -> 0.25 Brier;
  per-slot train base rate -> documented floor; monotone-projection never increases the metric when the truth is
  monotone (property test on random data).
- **Oracle checks before modelling (lesson 15):** (i) if continuous post/pre values are given, the target builder must
  reproduce every provided event label exactly (tests `>=` vs `>`, rounding, ATS 2005 rule vs ERS/ATS 2021
  "% predicted" rule); (ii) the event integrator fed a *degenerate* distribution at the true gains must reproduce the
  9 labels (proves threshold/ATS geometry and column mapping); (iii) our pre FEV1/FVC must match any provided pre
  values.
- **Selection discipline:** every selection (loss mix lambda, mixture K, GBDT depth/rounds, blend weights, per-slot
  calibration) is nested or cross-fitted: fit on folds' inner splits or on group-half A of the OOF, score on half B,
  swap (V3). Accept a change only with the Nadeau-Bengio corrected paired t > ~2 and sign consistency over >= 11 of 15
  fold-repeats (V16). Report best CV minus winner's-curse allowance (V18) with N candidates stated.
- **Yardsticks:** (a) per-slot train base rate; (b) 9 direct LightGBM classifiers (M3) - the honest "agent baseline"
  analogue; every structured model must beat (b) on the paired folds.

## Overfit/underfit risks

| Risk | Kind | Mitigation |
|---|---|---|
| Low signal-to-noise (BD response is weakly predictable; measurement noise floor) | overfit | strongly regularised heads (shallow GBDT, min_data_in_leaf >= 40, MLP width <= 128, dropout, weight decay, EMA), few knobs, repeated CV, corrected t |
| Wrong byte format / sample rate -> garbage curves | catastrophic | oracle checks vs provided indices/flags, physiological assertions on every row, raise on failure (no silent fallback) |
| Target builder mismatch (`>=`, rounding, ATS variant, denominator) | catastrophic for labels | oracle reproduction of provided labels; variants as D8 nuisance only if data cannot decide |
| 9 independent classifiers waste the continuous gain information | underfit | joint distributional model (M1/M2) trained on continuous gains |
| Many per-slot calibration/blend parameters on small OOF | overfit | <= 2 params per slot, cross-fitted, shipped only if cross-fitted gain > 0 |
| Hidden repeat visits/near-duplicates across folds | optimistic CV | union-find groups; random-vs-group CV gap as gauge (V19c) |
| Self-supervised waveform encoder memorising | overfit | trained label-free on pool+train blows, frozen, compressed to <= 16 dims; enters only with measured paired gain |
| From-scratch NN on a few thousand rows underperforms GBDT | underfit/variance | NN gets GBDT OOF location/scale as cross-fitted inputs (D5), 5 seeds averaged, fixed epochs + EMA (G12) |
| Base-rate shift between train and private | calibration | per-slot calibration fit on grouped OOF only (V21); no test-based recalibration (banned) |
| Fold models trained on 80 % | underfit | refit on 100 % with fixed counts (V9), calibration maps transferred (B9) after a transfer check |

## Recommended approach (primary + fallback)

### Stage 0 - signal processing and features (shared by all members; deterministic, train-fit transforms only)

Per blow (C12, C4, C9):
- decode -> flow(t) [L/s] * btps; volume(t) = cumulative trapezoid; time zero by back-extrapolation from the point of
  peak slope; back-extrapolated volume (BEV); FEV0.5, FEV0.75, FEV1, FEV3, FEV6, FVC, PEF, time-to-PEF, FET,
  FEF25-75, MEF75/50/25 (flow at 25/50/75 % FVC remaining), end-of-test volume change in the last 1 s.
- curve shape: flow-volume curve resampled at 20 fixed volume fractions (of FVC) normalised by PEF; concavity index
  (area between the straight PEF->end line and the curve); log-flow decay slope over the last 50 % of volume
  (time-constant of emptying); FEV1/FVC, FEV0.75/FVC, FEF25-75/FVC, MEF25/PEF, FEV1/FEV6. Obstruction shape
  (scooping) is the physiological driver of reversibility.

Per session (blows are an unordered set; canonical order = ATS ranking by FEV1+FVC, D9):
- ATS selection: largest FEV1 and largest FVC among acceptable blows (the published rule uses all blows when none is
  acceptable; that is a data rule written as such, logged with a count, not an environment fallback); indices of the
  best blow; mean/max/min/SD across acceptable blows.
- repeatability: FEV1 best - 2nd best, FVC best - 2nd best (mL and %), SD/CV, number of blows, number acceptable,
  number with plateau, effort-grade counts and best grade; "best-of-N" bias proxies (n acceptable, spread).
- explicit ATS geometry for trees (C1): `need_pct_fev1 = max(12, 20 / FEV1_pre)`, `need_pct_fvc = max(12, 20 / FVC_pre)`,
  `abs_02_pct_fev1 = 20 / FEV1_pre`.
- demographics: age, sex, height, weight, BMI, ethnicity (categorical), number of blows.
- **predicted-normal model (in-script, train-only):** ridge on log(FEV1), log(FVC), FEV1/FVC with log(height),
  age spline (5 knots), sex, ethnicity and interactions, fit on TRAIN (+ pool) pre-session inputs only -> residual
  z-scores and % predicted. Low % predicted and low FEV1/FVC carry most of the response signal. (Replaces
  external GLI tables.)
- All scalers / quantile transforms fit on the training fold (inside CV) or on full train for the final refit.

### Members (each produces a coherent 9-vector)

**M2 - LightGBM location-scale + train residual bank (lean core; built first).**
1. LightGBM regressors for `m1 = E[dFEV1 | x]` and `m2 = E[dFVC | x]` (Huber objective, shallow: num_leaves 15,
   min_data_in_leaf 40, lr 0.03, feature_fraction 0.7, bagging 0.8, lambda_l2 5; rounds = fixed from inner-CV mean
   best iteration x 1.1).
2. Scale models: LightGBM on `log|OOF residual|` for each target -> `s1(x), s2(x)` (heteroscedastic noise: small
   FEV1, poor repeatability and obstruction widen the gain distribution).
3. Residual bank: standardised OOF residual *pairs* `(r1, r2)` from TRAIN (keeps the empirical copula and the heavy
   right tail).
4. Event decode for a row: `P(event) = mean_k 1[event(m1 + s1*r1_k, m2 + s2*r2_k, FEV1_pre, FVC_pre)]` over the bank
   (fixed, train-only; deterministic). Monotone and coherent by construction; the ATS boundary is exact per row.
   Strip-the-ML: removing the trained m/s models leaves an unconditional bank = base rates.

**M1 - Joint mixture-density network trained on the metric (structural primary, from scratch, CPU).**
- Inputs: Stage-0 features (quantile-normalised, train-fit) + cross-fitted M2 outputs (m1, m2, s1, s2) as stacked
  features (D5) + (step 5, optional) 16-d session waveform embedding.
- Output: K-component bivariate Gaussian mixture over (dFEV1, dFVC) in %: weights, means, two scales, correlation
  per component (K in {1, 2, 3} chosen by nested CV; prior expectation K=2: "non-responder" and "responder" modes).
- Event layer (exact, differentiable): FEV1/FVC threshold events via mixture marginal Phi; ATS union via
  `P(A) + P(B) - P(A and B)` with `P(A and B)` from a fixed 32-node Gauss-Legendre quadrature of the conditional
  normal CDF, with row-specific boundaries `max(12, 20/FEV1_pre)` and `max(12, 20/FVC_pre)`. If the official pre values
  differ from ours (oracle check iii), add a learned per-row noise on the pre value (rank-1's "plausible true
  (FEV1, FVC) pairs"), integrated by the same quadrature.
- Loss: `L = Brier9(P_events, T; metric weights) + lambda * NLL(observed gains) + mu * CRPS_dense` where CRPS_dense is the
  Brier summed over a dense 1 %-grid of thresholds for both gains (this is the metric extended to all thresholds:
  metric-aligned and information-rich). If gains are only known as the 9 binary events, NLL is replaced by the
  **interval-censored** likelihood (the 5 FEV1 labels bracket dFEV1, the 3 FVC labels bracket dFVC; A1/A6).
  If targets are soft probabilities, use Brier/soft-BCE on them; optionally back-solve a per-row latent
  (mu, sigma) by probit regression of Phi^-1(T_k) on the thresholds t_k (A1) and add it as an auxiliary target.
- Net: 2 x 128 MLP, GELU, dropout 0.2, AdamW lr 2e-3, wd 1e-3, batch 256, 150 fixed epochs, cosine schedule, EMA 0.999,
  output-bias init at train base rates (lesson 7), 5 seeds averaged. Log train vs held-out Brier per epoch once to
  fix the epoch count as a constant (lesson 16).

**M3 - 9 direct LightGBM classifiers (yardstick, diversity member).** `objective="cross_entropy"` (accepts soft
labels), shallow as M2, 3 seeds, then row-wise monotone projection (weighted pool-adjacent-violators across the 5
FEV1 thresholds and the 3 FVC thresholds). Enters the blend only if it adds paired gain.

### Shipped pipeline (primary)
`P = per-slot cross-fitted blend of {M1, M2, (M3)}` with non-negative weights summing to 1 per slot (<= 3 members,
fit on group-half OOF, checked against equal weights, V16) -> **per-slot Platt recalibration** on logit (2 params
per slot, cross-fitted, shipped only if the cross-fitted Brier gain > 0; A17) -> row-wise monotone projection across
thresholds -> clip to [1e-4, 1 - 1e-4] (only matters for log loss) -> validate -> write. Final models refit on 100 % of
train with fixed counts; calibration/blend constants transferred from the OOF stage after a transfer check (B9).

Expected private result (ESTIMATE, not a promise): Brier skill vs per-slot base rate of roughly 0.05-0.20 overall,
highest on the 15/20 % and ATS slots (obstruction-driven), lowest on the 0 % slots (noise-dominated). The structured
members should beat 9 direct classifiers by a small but consistent margin (~2-5 % relative Brier) mainly through
coherence and the continuous-gain signal. Reasoning: weak predictability of BD response and a measurement-noise floor.

### Fallback
M2 alone (+ per-slot Platt, monotone by construction). It needs no neural training, runs in a few minutes on CPU,
and is coherent. Ship it as the first credit (baseline) once the oracle checks pass.

## Rejected options

- **9 independent classifiers as the primary** (the likely "AI baseline" shape): ignore threshold monotonicity, the
  continuous gains and the ATS geometry; kept only as yardstick M3.
- **End-to-end 1D-CNN on raw labelled curves as primary:** labels are few relative to waveform dimensionality and
  noisy; CNN enters only as a label-free self-supervised encoder (roadmap step 5) whose frozen low-dim embedding must
  earn a paired gain.
- **External reference equations (GLI-2012 tables)** and any external spirometry data: external-data ban.
- **Pretrained time-series/foundation models** (Chronos, TabPFN): banned by from-scratch regime and inference-only.
- **Self-supervised pretraining on test waveforms, scalers/PCA fit on train+test, test-batch normalisation,
  calibrating on test predictions:** CLAUDE.md 2.3 #5.
- **Hand rule "obstructed -> high P(response)":** strip-the-ML failure; obstruction enters only as features.
- **Sampling-based joint decoders / large mixtures / deep ensembles of many families:** heavier than needed
  (lesson 13); the mixture density with closed-form events is the lean structural design.
- **Time guards, placeholder submissions, try/except model fallbacks, cuda/cpu switches** (H1, B4, H3, H4): DO NOT
  ADOPT (Deterministic Execution).
- **Augmentation by fabricating sessions (mixing blows across participants):** synthetic data. Only defensible
  augmentation: dropping a *non-best* blow within a session (resampling which real slice is observed, L), tested in
  step 5, because it changes repeatability features but not the label's pre value.

## Fixed work plan & runtime budget

Hardware assumption: CPU-only (stated by neither text with certainty; inferred from both inspected solutions).
Threads pinned: `OMP_NUM_THREADS=MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`, LightGBM `num_threads=4,
deterministic=True, force_row_wise=True`, seeds everywhere, `device="cpu"`. No time in any condition.
Data-size assumption for the estimate: ~3-10k labelled participants, ~3-8 blows each, ~1-6k samples per blow,
pool of similar size. Re-profile on the real data and rescale counts (as constants) before freezing.

| Stage | Work (fixed) | Est. CPU time |
|---|---|---|
| Decode + per-blow indices + session features (train, pool, test) | vectorised numpy | 2-4 min |
| Predicted-normal ridge (train+pool inputs) | closed form | < 10 s |
| M2: 2 mean + 2 scale LGBM x 5 folds x 3 repeats + refit | ~400 rounds each | 3-5 min |
| M3: 9 LGBM x 5 folds x 1 repeat + refit, 3 seeds | ~400 rounds each | 4-6 min |
| M1: MDN 5 folds x 3 repeats x 1 seed (CV) + 5-seed full refit | 150 epochs MLP | 6-10 min |
| Nested selection grids (K in {1,2,3}, lambda in {0, 0.3, 1}, mu in {0, 0.3}) on repeat 1 only | 18 configs x 5 folds | 6-8 min |
| (Step 5, optional) SSL 1D-CNN encoder on pool+train blows | 15 fixed epochs | 8-10 min |
| Blend, calibration, projection, validation, write | | < 1 min |
| **Total** | | **~30-45 min** |

If the challenge states a limit (unknown), keep >= 30 % headroom by cutting repeats/selection-grid size as fixed
constants (never at run time). Memory: < 4 GB.

Script-level validation: assert input schema (required columns, base64 decodes, btps finite and in a physiological
range), unique ids, every test participant has >= 1 blow, all 9 outputs finite and in [0, 1], monotone within each
threshold family, columns/ids/order equal to sample_submission, reload with `keep_default_na=False`. Any failure
raises; no fallback file.

## Metric-aware training & decode

1. **Loss = metric** (A2/A7/A17): Brier over the 9 events with the metric's row/column weights (if any) is the
   primary loss of M1; soft-target BCE substitutes when the metric is log loss.
2. **Back-solve richer targets** (A1): continuous gains when given; interval-censored gains when only events are given
   (A6 one-sided terms); probit back-solve of (mu, sigma) when soft T are given.
3. **Cumulative structure** (A8): threshold events decoded from one CDF -> monotone across thresholds without
   post-hoc fixes; PAV projection only after per-slot calibration/blending can break it.
4. **ATS geometry exact per row** (A3/A18): the decode integrates the joint over the union region with the
   row-specific absolute boundary; no separate ATS classifier needed (M3's ATS head is a diversity check only).
5. **Calibration** (lesson 7, V21): output biases initialised at base rates; per-slot Platt fitted on grouped OOF,
   cross-fitted; ship only on a positive cross-fitted gain.
6. **Blending in a comparable space** (D2): all members output calibrated probabilities of the same events; blend in
   probability space with per-slot non-negative weights, <= 3 free weights per slot, nested.
7. **No decision thresholds** to tune: the output is the probability itself (expected Brier is minimised by the
   calibrated probability).

## Structural signals

- **Threshold nesting:** `1[d >= 20] <= 1[d >= 15] <= ... <= 1[d >= 0]` for FEV1 and FVC -> one CDF per gain.
- **ATS coupling:** ATS is a deterministic function of (dFEV1, dFVC, FEV1_pre, FVC_pre) -> derived, not learned
  separately. Verify on train that the provided ATS label equals the formula on 100 % of rows when gains are given
  (if not, the label uses another rule: switch to the variant that reproduces it).
- **dFEV1-dFVC dependence:** captured by the bivariate mixture correlation (M1) and the paired residual bank (M2).
- **Absolute-volume criterion:** small lungs need larger % gains -> `20/FEV1_pre` as an explicit feature and in the
  decode geometry.
- **Measurement noise / regression to the mean:** pre = best of N blows; a session with a noisy or low best blow
  predicts a larger measured gain -> repeatability and N-acceptable features; scale head models heteroscedasticity.
- **Blow-set permutation invariance:** sorted canonical order (ATS ranking) + symmetric pooling (mean/max/SD) (D9);
  SSL encoder pools blows by mean+max.
- **Within-session blow pairs as natural positives** for the SSL encoder (two blows of one person share
  physiology): contrastive training uses real pairs only, label-free, train+pool only.
- **Physiology in features, not rules:** obstruction (low FEV1/FVC, concave FV curve, low MEF25/PEF) is the known
  driver of reversibility; it reaches the prediction only through trained models.

## Experiment roadmap

1. **Contract & metric** (stop when green): column-name table vs sample_submission; `brier9`/`logloss9` unit tests;
   target builder reproduces provided labels (oracle i); event integrator reproduces labels from degenerate
   distributions (oracle ii); our pre values match provided ones (oracle iii); waveform decoding passes physiological
   assertions on 100 % of train blows. Record per-slot base-rate Brier under grouped CV (floor).
2. **Cheapest honest baseline end-to-end:** M3 on Stage-0 summary indices + demographics; full script writes a valid
   CSV; double-run diff identical. Record grouped CV (5x3) per slot.
3. **Structure:** M2 (location-scale + residual bank). Paired comparison vs M3 on identical folds. Then add
   curve-shape and predicted-normal features (one block at a time, corrected t). **Credit 1: M2 (or M3 if M2 loses).**
4. **Metric-aware neural member:** M1 with Brier9 loss; grid K x lambda x mu nested on repeat 1; stack M2 OOF
   features in; 5-seed refit. Accept if it beats M2 paired or adds blend gain.
   **Credit 2: best single structured model.**
5. **Diversity / representation:** (a) SSL 1D-CNN blow encoder on pool+train blows (masked-reconstruction or
   within-session contrastive), frozen, 16-d session embedding into M1/M2; (b) non-best-blow dropout augmentation
   for M1; (c) M3 as blend member. Each enters only with a paired gain beyond noise.
6. **Bounded HPO** (if steps 3-5 plateau): Optuna TPESampler(seed=0), n_trials=25, n_jobs=1, on M2's LGBM depth/leaves/
   regularisation, objective = grouped 5-fold Brier9 of the decoded events, evaluated on a different split seed than
   the reported one (nested). **Credit 3: blended + calibrated ensemble.**
7. **Freeze:** sanity holdout scored once; finalists re-scored on 3 fresh split seeds; run the final script twice
   from a clean `working/` and diff; half-rows test; compliance scan; runtime profile. **Credit 4: final.** Keep 2
   credits in reserve for checker glitches.

Stop rules: if after step 4 the structured members are within noise of M3, ship the simplest coherent model (M2) and
stop adding machinery; if per-slot Brier is within ~2 % of the noise-only ceiling on the 0/5 % slots, stop working on
those slots and focus on 15/20 %/ATS calibration.

## Compliance audit

- Test file read only for one-row-at-a-time feature extraction and prediction; no scaler/PCA/vocabulary/SSL/
  predicted-normal fit on test rows; residual bank and calibration maps from TRAIN OOF only. Half-rows test planned.
- No `time` in conditions; no cuda/cpu or import fallbacks; fixed folds/epochs/rounds/trials/threads; seeded LightGBM,
  torch, numpy, Optuna. The "use all blows when none acceptable" rule is a documented data rule with a logged count,
  identical on every machine.
- No external data (no GLI tables), no pretrained weights or tokenizers (from-scratch), no internet use at all.
- Hardcoded constants: task-definition thresholds (description), physiological assertion bounds (validity checks,
  not tuned), data-contract format constants (asserted). All modelling hyperparameters either principled defaults
  stated in comments or chosen by the in-script nested search.
- Strip-the-ML: without trained models only a train base rate is left. Requirements map in the docstring: from-
  scratch, CPU, train-only fitting, per-row inference, fixed plan, metric implemented, outputs validated.
- Source plain, < 512 KB, no blobs; comments explain each fit/predict.

## Open questions & assumptions

Reviewer / description questions:
- Q1. Exact submission columns and metric (Brier vs log loss, any row/column weights)? Plan handles both; the loss
  switches by a constant set from the description, not by a runtime check.
- Q2. Is a post-bronchodilator session (or its indices) given for **test** rows? (Variant C below.)
- Q3. Is the unlabeled reference pool disjoint from test participants, and may it be used for self-supervised
  representation learning and the predicted-normal fit?
- Q4. Runtime and hardware limit (CPU cores, minutes)?
- Q5. Are ethnicity/sex permitted as inputs?

Variants that change the plan materially:
- **Variant A (assumed):** test = pre-BD session only; train carries post-BD data (blows or values) -> continuous
  gains available; plan as written.
- **Variant B: train labels are only the 9 columns (binary or soft).** No continuous gains: drop M2's regressions
  (replace by an ordinal LightGBM over the 6 dFEV1 intervals and 4 dFVC intervals: multiclass softmax -> CDF), train
  M1 by interval-censored likelihood + Brier9; for soft T, probit back-solve per-row (mu, sigma) as auxiliary
  targets (A1). Brier floor becomes E[T(1-T)].
- **Variant C: test includes both sessions (pre and post).** The task becomes a measurement-noise problem: the
  gain is measured, the label is a probability that the *true* change crosses each threshold. Then the features of the
  post session are inputs; M1's mixture is centred on the measured gain with a learned scale driven by within-session
  repeatability of both sessions (rank-1's "discretised joint distribution over plausible true (FEV1, FVC) pairs"),
  and M2 regresses the residual (label-implied latent minus measured). Same decode, validation and compliance.
- **Variant D: ATS label uses ERS/ATS 2021 (> 10 % of predicted).** Detected by oracle (i); predicted value would then
  come from the in-script predicted-normal model, which cannot reproduce official GLI predictions exactly: ask the
  reviewer whether the official predicted values are supplied as a column.
- Assumed CPU-only and from-scratch (inferred from inspected code, not stated). If a GPU and pretrained weights are
  allowed, the plan is unchanged except the SSL encoder could be larger; no pretrained spirometry model exists
  anyway.

Could not verify: any data fact, the byte format and sample rate of `flow_b64`, the metric, column names, row counts,
runtime limit, presence and contents of the reference pool, whether the AI baseline score is printed.
