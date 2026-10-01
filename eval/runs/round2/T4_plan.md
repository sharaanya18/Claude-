# Eris plan: T4 shared-workspace motion-mode sequence prediction

Status of evidence: NO dataset was available for this run. Everything under "Data findings" is a diagnostic to run on TRAIN plus an expected result, all marked UNVERIFIED. Column names are unknown; I use `id`, a 4x7 history (28 bits, oldest first) and a 10-letter target string `y` as placeholders. Numbers such as run times are estimates, not measurements.

---

## Contract & decision unit

**One valid answer.** For each test `id`, a string of exactly 10 characters, each in {S,A,B,C,D,E,F}. A valid-but-bad answer is any such string. Invalid (scores below zero / dead last, CLAUDE.md section 5): wrong columns or order, wrong row count, an id mismatch, a length other than 10, a letter outside the 7-letter set, an empty string (NaN on reload), and so on. Reload with `keep_default_na=False` and assert all of this against `sample_submission.csv` before writing.

**Metric, term by term** (from the paraphrase; exact formulas are open questions Q1-Q4):
- 0.3 x exact macro-F1: per-step class match over the 7 modes, averaged over classes.
- 0.5 x tolerant macro-F1: same, but a predicted step counts as matching a true step of the same class within +-1 step, with greedy one-to-one matching. It rewards boundary-hedged runs: a transition predicted 1 step early or late is mostly forgiven.
- 0.2 x onset utility: onset = first non-S step. Utility = 1 - gap/G, floored at 0 (decay length G unknown, assumed about 10). It is 0 if the prediction OR the truth never leaves S.

**Consequences I derive from the metric (assumptions, to be verified by unit tests):**
1. Onset asymmetry. If the truth has no onset, the onset term is 0 whatever we predict. Predicting "never active" gains nothing there, but predicting "never active" when the truth does activate loses the whole term. So the onset term always favours committing to an onset; the F1 terms (false-positive tokens) push back. This trade-off must be resolved by expected utility under the model posterior, not by argmax.
2. Perfect predictions do not score 1.0 if some rows have no onset. Compute that ceiling on train first.
3. The linear decay plus an absolute-error flavour means the optimal onset guess is the posterior MEDIAN of the onset time given onset exists, not the mode.
4. The tolerant term makes the median/hedged boundary preferable to the modal boundary.

**Decision unit.** The row is the unit of prediction. "Every row is scored independently and must be predicted independently of other rows" is a hard rule: a row's output is a function of its own 4x7 history plus a train-fit model. The unit of INDEPENDENT LABELLED INFORMATION is not the row. Rows may be fixed-length windows cut from longer recordings, so neighbouring windows share most of their content. The effective sample size is the number of recordings and workflow types, not the few thousand rows.

**Pipeline stages, to be diagnosed separately:**
- (a) Representation / candidate coverage: is the true 10-step string reachable among the model's top-M sequences, and is the true onset within +-1 of an enumerated candidate? Report oracle recall@M.
- (b) Scoring: calibrated per-step marginals p_t(m|x) and an onset posterior P(onset=t|x) for t in 1..10, plus "never".
- (c) Decoding: choose the output string by minimum Bayes risk (expected metric) over candidates. Score the decode on its own, using the true posterior replaced by the model's.

---

## Compliance regime

**Domain.** Tabular / small-sequence structured prediction from a 28-bit input. There are no images or text, and no pretrained backbone is applicable (CLAUDE.md 6.1 / 6.5 flavour). From-scratch small neural nets and GBDTs are allowed. The "train in-script, no external data, CPU or GPU, about 1 hour" limits are respected. I treat CLAUDE.md's 50 min target as binding.

**Explicit bans extracted from the description:**
- No external data.
- No test-set-wide statistics. This covers any normalisation, clustering, prior estimation or vocabulary fit using test rows.
- Predict each row independently.

**Ban implied by the above and stated loudly.** Stitching test windows into longer recordings (a window's history reveals the next window's target) is a leak and is a whole-test operation. It is banned. Do not use `id` order or id adjacency as a feature, and do not use cross-row lookups at inference. Everything about types and recordings (groups for validation) is derived from TRAIN only and used for CV only.

**Grey areas, with the reading chosen:**
- MBR decoding with constants fitted in-script on train OOF. The description (as paraphrased) does not ban metric-aware decoding. It is per-row and uses only train-fit constants, so I treat it as allowed. If a reviewer says decode-time optimisation is banned, the in-rules mechanism is training-time: class-weighted likelihood plus an onset auxiliary loss, so the same goal is reached through the loss. Plan for the loss-only variant as a measured fallback (Experiment roadmap step 4b).
- Reconstructing longer train recordings from train windows to create additional training windows at new offsets. These are real labelled units recombined, not fabricated entities, but it is borderline "synthetic data". It is OFF by default (Q7). It is also only possible if the history's channel semantics map onto the target modes, which is itself unverified.
- Hand-engineered features (xor deltas, run lengths, pair mismatches) fed to a trained model: allowed.
- GBDT as a second family: allowed for tabular.

**Strip-the-ML test.** Remove the trained model and there is nothing left: the decode needs its posterior, and no workflow grammar or lookup is hard-coded. Not allowed and not used: a hand-written table "history pattern -> future string", hand-coded workflow grammars, or a hard-coded A<->B, C<->D symmetry applied without a train check (the swap is a structural claim from the description that I assert on train, see Structural signals).

---

## Data findings

All items are UNVERIFIED hypotheses. Each is a diagnostic to run on train (never test beyond schema/row count/id format) with the result I expect.

**D1. Schema and sizes.** Row count (expected "a few thousand"), history encoding (28 bit columns or a packed string), target as a 10-letter string, any extra columns (workflow id, recording id, offset). Expected: no workflow/recording id in test. Any such train-only column is used for GROUPS only, never as a feature.

**D2. Target distribution.** Exact class frequencies of the 7 modes over all 10 x N tokens, per step t, and per row (fraction of rows all-S; fraction with an onset; distribution of onset time 1..10; number of distinct runs per row; run-length histogram).
- Expected: S dominates (probably 50-80 percent of tokens); E and F (or whichever modes are rare) are in the low single digits, so macro-F1 is dominated by rare-class precision/recall.
- Expected: runs are persistent (long dwells), so rows are explained by 1-3 segments. Onset time is not uniform: mass at t=1 (continuation of motion already in history) and a secondary mass at later steps.
- Compute the onset-term ceiling = fraction of rows with a true onset (rows with no onset can never score on the 0.2 term).

**D3. Channel-to-mode semantics.** Cross-tabulate the history's last snapshot (7 channels) against the first target step mode, and each history channel against each mode. Check whether a snapshot is one-hot (exactly one channel on) or multi-hot, and whether channel j corresponds to mode j.
- Expected (hypothesis): history channels are indicator flags of the same 7 modes, one-hot or nearly. If not, learn the mapping rather than assume it; the NN gets raw bits either way.
- Because the "persistence" baseline (copy last snapshot mode for all 10 steps) is the natural reference, compute its score: it gives the floor and shows how much of the answer is carried by the last snapshot.

**D4. Duplicates and ambiguity ceiling.** Count unique 28-bit histories; the multiplicity histogram; for each unique history the entropy of its target strings. Compute (i) the in-sample Bayes oracle (per-pattern best MBR output) and (ii) the leave-one-out version.
- Expected: heavy duplication (28 bits but strongly structured), so many rows share exact histories with different futures: genuine ambiguity (workflow phase is unobservable from 4 snapshots). The LOO oracle is the realistic ceiling; the in-sample oracle overstates it.
- This is also the memorisation trap: exact-pattern target statistics look great in-sample and fail on unseen types.

**D5. Recording / window structure (to build groups).** If the data are windows cut with stride s < 14 from longer recordings, adjacent windows overlap. Test, on train only: (i) are ids sequential in recording order? Check whether row i+1's history snapshots are a shifted copy of row i's snapshots plus its first target steps (using the D3 mapping); (ii) otherwise detect overlaps between the target of row i and the history of row j via the mapping and union-find. Report number of recordings, windows per recording, and the stride.
- Expected: a few dozen to a few hundred recordings, 10-100 windows each; most windows have a near-duplicate partner. If unrecoverable, fall back to history-signature clusters (D6).

**D6. Workflow / history types.** Cluster train histories into types: agglomerative or union-find on Hamming distance over the 28-bit vector (and over the canonicalised form under arm swap), with a fixed cluster count 8-12. Report cluster sizes, the per-cluster target distribution, and the per-cluster fraction of identical histories.
- Expected: a handful of dominant types and a tail of rare ones, matching the warning that the private test may contain types rare or absent in train. Per-cluster difficulty is expected to vary a lot, so leave-type-out scores will fall well below random-split scores.

**D7. Arm symmetry audit.** Compare counts of A vs B and of C vs D in targets; compare the 28-bit history distribution before and after swapping the two arms' channels; fit a quick model on original rows and score the swapped rows (and vice versa) to see whether the swap preserves the label law.
- Expected: approximately symmetric frequencies; any big gap (more than a few SE) means the arms are not exchangeable and the swap augmentation must be dropped or only used with an arm-identity flag.
- Check which of E/F (if any) are unaffected by the swap, and whether the pair mapping is (A,B) arms with (C,D) their commanded counterparts or (A,C)/(B,D) as arm pairs. The description says A/C and B/D are observed/commanded pairs, which I read as arm 1 = {A,C}, arm 2 = {B,D}, so the swap is A<->B, C<->D. Verify by checking that the pair bigram statistics are identical across arms after the swap.

**D8. Causal lag between observed and commanded.** Compute the bigram matrix of target modes P(m_t | m_{t-1}) and the lagged cross-statistics P(A_t | C_{t-1}) and P(B_t | D_{t-1}), in both targets and the history.
- Expected: commanded modes lead observed modes by about 1 step (the "action at t-1 explains the change at t" structure), giving a strongly structured transition matrix. A fixed 1-step lag would be a coupling to exploit through model structure, not rules.

**D9. Leakage suspects.** Any column not known at prediction time, id-ordering correlation with the label, and per-row scores of a trivial model that are too good. A "claimed-perfect" signal is a leak tell.

**D10. Metric unit tests (before any modelling).** Implement the metric once in a function with the interpretation flags (Q1-Q4) and test: perfect string; all-S; constant single-letter; reversed; shifted by +-1 (tolerant high, exact low, onset utility drops by 1/G); base-rate sampled strings; a row with no onset; a prediction with onset when truth has none. Record the composite score of the persistence baseline and the all-S baseline.

---

## Validation design

**Reconstructing how the test was split (from the description + train only).** The statement that the private test "may contain history/workflow types that are rare or absent in train" means a covariate/type shift: leave-type-out is the closest simulation. A plain random K-fold would put near-duplicate windows of the same recording in both train and validation and report an inflated score. I therefore plan two schemes and select with both.

**V1: recording-grouped K-fold (leak-free).** Groups = recordings recovered by D5 (union-find over overlapping windows), or if that fails, connected components of near-duplicate histories at Hamming distance <= 2 (computed on train only). GroupKFold with 5 folds, repeated with 3 different group-to-fold assignments (seeds). Reports in-distribution generalisation.

**V2: leave-type-out (test-like for the stated shift).** Types from D6 clustering (fixed cluster count, fixed seed). Each fold holds out 1-2 whole clusters, preferring rare ones (so train has seen "close but not the same" workflows); 5 folds, 3 seeds of cluster-to-fold assignment. Reports extrapolation to unseen types. The type clustering uses train only and exists for CV only; shipped inference does not use it.

**Selection rule.** A change must beat noise on V2 (paired, same folds, mean gain larger than 1 SE across folds and consistent in sign across the 3 seeds) and not lose more than 1 SE on V1. If V1 and V2 disagree, V2 wins (it matches the stated deployment shift), but a V1 loss is logged. Report mean +- std per scheme; do not compare numbers from different splits.

**Nested checks for post-hoc steps.** Hyperparameter (K, weight decay, learning rate) selection: fixed small grid inside the script, scored on V2/V1 OOF. Decode constants (per-class weights, onset weight sharing, G sensitivity), recalibration constants and blend weights are each fitted on OOF predictions from the other folds and evaluated on the held-out fold (cross-fitted), so the final reported OOF number includes no selection on itself. A final untouched sanity fold (one V2 cluster fold and one V1 fold held out before any selection) is reported once.

**Expectation management.** A self-built proxy over-estimates the real score; plan for the real number to be below V1 and probably around or below V2. Report the V1-V2 gap; it is the best available estimate of type-shift risk.

**Metric diagnostics reported with every model:** the three components separately; per-class F1; per-onset-time utility; calibration (ECE per class and per step); stage diagnostics (oracle recall of the candidate pool, ranking given reachable, decode given good scores).

---

## Overfit/underfit risks

**Overfit risks (with mitigations):**
1. Few independent groups (recordings and types, perhaps tens to a few hundred) behind a few thousand rows. Mitigation: grouped CV (V1/V2); capacity ladder (below); strong weight decay, dropout, small hidden size; fixed-length training without validation-triggered stopping.
2. Exact 28-bit patterns memorised (high-cardinality identity). Mitigation: no pattern-level target encoding as a primary feature; features are structural (xor deltas, run lengths, pair mismatches, counts), and the NN is regularised; V2 exposes this risk directly.
3. Many tuned knobs on a noisy CV (K, wd, decode weights, calibration, blend weight). Mitigation: a very small grid (about 12 configurations, ranked on V2 with V1 as veto), cross-fitted decode constants, and a plateau rule (pick the middle of the stable region, not the argmax).
4. Rare-class decode weights fitted on very few positives. Mitigation: weights follow the closed-form F1 fixed point (threshold = F/2) from OOF counts, shrunk toward the train class prior; no free grid over corners; bound by the label prior.
5. Selection on the reported OOF. Mitigation: nested/cross-fitted as above.
6. Leakage through near-duplicates across folds. Mitigation: V1 groups by overlap, and V2 holds out whole clusters.

**Underfit risks (with mitigations):**
1. Dropping step-specific onset information with a single flat multi-class head. Mitigation: onset auxiliary head (11-way: onset 1..10 or none) and an explicit run/hazard structure in the output (latent-structure model below).
2. Too small a latent mixture (K=1 cannot represent two plausible workflows with the same history). Mitigation: K in {1, 4, 8} selected by V2; the K=1 rung is the plain input-dependent chain.
3. Loss mismatched with the metric (plain CE favours S, hurts rare classes and onset). Mitigation: fixed class-weights (1/sqrt(freq) from train counts), onset auxiliary loss, per-slot recalibration, and MBR decode.
4. Throwing away order information in preprocessing (for example counts only). Mitigation: raw 28 bits always go in alongside engineered features.
5. Heavy regularisation killing the minority types. Mitigation: per-output (per class) regularisation can be compared in the GBDT member; the NN uses a shared moderate setting and the decode corrects class bias.

---

## Recommended approach (primary + fallback)

### Primary: arm-equivariant latent-phase chain model with exact marginal likelihood, plus MBR decode

**Representation.** Input x = the raw 4x7 bits, plus engineered features (below). Optionally arm-equivariant: arm-1 channels {A,C} and arm-2 channels {B,D} processed by shared-weight blocks and combined symmetrically, so the A<->B / C<->D symmetry holds by construction (only if D7 passes; otherwise a plain MLP with the swap used as augmentation or not at all).

**Engineered features (all train-fit-free, row-local):**
- Per-channel count of active snapshots (0-4), last-snapshot flags, and the 3x7 xor change flags between consecutive snapshots.
- Number of channels changed per transition, number of distinct snapshots in the history, whether the history is constant, and time since last change.
- Run length of the current state per channel.
- Pair features: observed-vs-commanded mismatch for each arm at lag 0 and lag 1 (A vs C, B vs D), because commanded leads observed.
- Arm-symmetric summaries (sum and absolute difference of the two arms' feature vectors), which are permutation-symmetric by design.
- Nothing is fit on test; no vocabularies or scalers beyond constants.

**Model.** A small PyTorch network (about 10k-30k parameters):
- Encoder: 2 hidden layers (64 units, GELU/ReLU, dropout 0.2-0.3, weight decay), producing (a) mixture gate logits over K latent "phases" (K in {1,4,8} chosen in script), and (b) per-step emission potentials e_t(m|x), low-rank in (step x mode) to limit parameters.
- Output: the 10-step string is generated by a mixture of K chains. Each phase k has a learned initial distribution, a learned 7x7 transition potential (tied across steps with a small learned step-dependent hazard shift; at most 7+49 parameters per phase plus a step embedding), combined with the input-dependent emissions. P(y|x) = sum_k g_k(x) * prod_t P_k(y_t | y_{t-1}, t, x). The forward algorithm (7 states x 10 steps x K) is exact and cheap on GPU.
- Training loss: the exact negative log-likelihood of the whole 10-step string under the mixture (explicit marginal likelihood, no implicit latent inference by a classifier), class-weighted by 1/sqrt(freq) from train counts, plus (i) an auxiliary per-step marginal cross-entropy, (ii) an auxiliary onset cross-entropy (11-way, using the model's exact onset posterior from the forward pass), and (iii) a small any-activity-per-window auxiliary. All have fixed weights, not tuned on the reported OOF.
- Learned parameters are few and listed in the plan so the strip-the-ML test passes: gate, emissions, transitions, hazard shift.
- Training recipe: AdamW, fixed epochs (about 150-200 full-batch or large-batch steps), a fixed cosine schedule, EMA/weight averaging over the last quarter, fixed seeds; no validation-triggered stopping.
- Calibration: per-class bias (7) and per-step-bucket temperature (about 3) fitted on OOF marginals, cross-fitted; biases initialised at the base rate.

**Refit.** After CV (V1/V2) selects K and weight decay, refit on 100% of train with 5 seeds (fixed) and average the final posteriors (average log-probabilities of marginals/onset, or mixture over seeds; identical code path for OOF and final). Fold models are used only to produce OOF numbers and calibration.

**Decode: minimum Bayes risk over the model's posterior.** For each row, build a candidate set of strings: (i) the class-weighted per-step argmax, (ii) the Viterbi path, (iii) the M=32 best sequences (exact M-best from the chain), (iv) onset-shifted variants (+-1, +-2 of each of the top candidates), (v) all-S. Compute S=64 sequences sampled from the posterior with a fixed-seed generator (deterministic), then choose the candidate that maximises the Monte-Carlo expected row metric: 0.3 x exact term + 0.5 x tolerant term (greedy +-1 matching exactly as the metric) + 0.2 x onset utility with the true onset time distribution from the samples. Under the pooled-macro reading (Q1) the per-class token contributions are weighted by the F-score fixed point (predict class c at step t only when p_t(c) is above F_c/2 and scale by 1/D_c, with D_c = predicted + true count of class c from train OOF), instead of the per-row macro. The constants (G sensitivity, class weights, shrinkage toward the prior) come from an in-script OOF search or fixed-point and are cross-fitted.
- This is not a pure constraint search: it consumes the model's posterior (a learned model is essential).
- Cost of the reading: the pooled vs per-row choice is implemented as a switch fixed by the description before submission, with both versions' OOF scores logged; ship the MBR variant that is compliant under the more conservative reading (per-row exact metric, no pooled denominators) unless the description says pooled.

**Expected result (an estimate, not a promise, no data seen):** composite score on V1 perhaps 0.55-0.75 and on V2 perhaps 0.45-0.65; private LB likely a few points below V2 given type shift. The wide range reflects not knowing the ambiguity ceiling (D4) and the no-onset fraction (D2).

### Fallback: per-step tabular heads (LightGBM) + onset classifier, same MBR decode

- Long format: one row per (window, step t) with t as a feature, 7-class LightGBM with the same raw+engineered features, plus a second small LightGBM for the 11-way onset class. Per-step marginals and an onset posterior feed the same MBR decode (samples drawn from independent steps conditioned on the onset posterior, since there is no chain; the Viterbi/M-best candidates are replaced by per-step argmax variants).
- Regularisation: shallow trees (max_depth 4-6), min_child_samples large (about 50), strong L2, feature fraction 0.7, fixed number of rounds (about 300), `deterministic=True`, fixed seed and thread count.
- Role: (a) fallback if the latent model fails V2, (b) diversity of assumption (flexible tabular marginals vs structured chain), blended with the primary only if the blend gains more than noise: blend per-step log-probabilities with a single weight fitted on OOF (cross-fitted), not raw probability averaging.

---

## Rejected options

1. **Nearest-pattern / lookup table (history -> majority future string).** Rule-based, memorises recordings, and fails on unseen types, which is exactly the stated private-test shift. Used only as a diagnostic ceiling (D4), never shipped.
2. **Sequence transformer / LSTM / seq2seq with beam search.** Input has 28 bits and only a few thousand rows from few independent groups; the capacity ladder says stop at the lowest rung that wins. Beam search adds runtime-dependent decoding risk. Rejected unless the chain model clearly underfits on V2.
3. **Hand-written workflow grammar / state machine / regex.** Banned by the compliance rules (strip-the-ML).
4. **Stitching test windows into recordings, or using id order across rows.** Banned (whole-test aggregation; predicts rows from siblings).
5. **Pseudo-labelling or self-training on test, test-time adaptation, label-free prior estimation from test.** Banned.
6. **Plain per-step cross-entropy classifier with argmax output as the primary.** Ignores onset coupling, run structure, and the asymmetric onset utility; kept only as the baseline rung.
7. **Large ensembles of near-identical GBDTs or NNs differing by seed only.** Little gain; the budget goes to diversity of assumption (chain vs tabular) instead.
8. **Validation-triggered early stopping or wall-clock cutoffs.** Banned by determinism rules and noisy with few groups.
9. **Reconstructing longer train recordings to enlarge training data.** Not part of the primary (grey; Q7). Re-evaluate only after a reviewer answer and only if the channel-to-mode mapping from D3 is exact.

---

## Fixed work plan & runtime budget

All counts are fixed constants (CLAUDE.md section 3); time is logged only. Device fixed to `cuda` (A10G); tiny models, so the GPU is barely loaded. No `os.cpu_count()`, no import fallbacks, `deterministic=True` for LightGBM, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch, fixed `Generator`s.

| Stage | Fixed plan | Estimated A10G time (UNMEASURED) |
|---|---|---|
| Load, validate schema, feature build | one pass, vectorised | < 0.5 min |
| Group building (D5/D6) on train | union-find + clustering, fixed seeds | < 1 min |
| NN grid on V1/V2 | 12 configs (K in {1,4,8} x 4 wd/dropout settings) x 5 folds x 1 split seed | 12 x 5 x ~10-15 s = 10-15 min |
| NN repeat of the best 2 configs on 3 split seeds | 2 x 5 x 2 extra seeds | 3-5 min |
| LightGBM fallback member | 5 folds x 2 configs x (7-class + onset), 300 rounds, long format about 50k rows | 6-10 min |
| Calibration + decode constants, cross-fitted | closed-form fixed point + small fixed grid | 1-2 min |
| MBR decode OOF + test | M=32, S=64, vectorised | 1-3 min |
| Final refit on 100%: NN 5 seeds (+ LightGBM if blended) | fixed | 2-4 min |
| Validate and write submission | re-read file | < 0.5 min |

Total estimate about 25-40 min; target <= 45 min and at least 30 percent headroom against the 50 min target. These numbers must be profiled locally (one epoch, one fold) and the counts hard-coded; if the profile exceeds the budget, cut the grid (fixed reduction), not branch on time. Memory: a few hundred MB on GPU, a few GB host RAM. The grid size may be reduced to 8 configs without touching the plan's logic.

**In-script validation.** Input: the schema, id format, row count, finite features. Output: exactly 10 characters per row, letters in the allowed set, row count equal to `sample_submission.csv`, ids in sample order, no empties, and re-read the written file with `keep_default_na=False`. Do not write a constant-output fallback early unless allowed (an all-S file is a valid but silent degraded path; if used it must be logged loudly; I prefer a loud failure).

---

## Metric-aware training & decode

1. **Back-solve the metric.** All three terms are deterministic functions of the 10-letter string, so the full target string (not just per-step labels) is the training target. The onset is derived exactly: onset time in 1..10 or none. Train an onset head with its own CE because the onset term has a separate (linear-gap) utility.
2. **Hierarchical averaging.** Replicate exactly what the metric does (per row then mean, or pooled then macro): implement both in one function and log both for every model (Q1). The class weights in the loss follow whichever is chosen: with the per-row reading, the class-rarity weights are mild (1/sqrt(freq)); with the pooled reading, apply the F-score denominators in the decode.
3. **Closed-form F-threshold.** For the pooled reading, derive the per-class decision rule algebraically: predict class c at step t when p_t(c) is above F_c/2, with the per-class weight 1/D_c (D_c = predicted count + true count of class c). Compute F_c and D_c by a fixed-point iteration on OOF, bounded by the class prior so rare classes cannot reach degenerate corners (no unbounded grid search).
4. **Expected utility over the valid output set.** Argmax of the per-step probability is not the optimum for this composite. Use the MBR decode above, with the exact row metric (including the greedy +-1 tolerant matching and the onset gap utility) evaluated against posterior samples.
5. **Onset.** Choose the onset time as the posterior median given activation, then compare with the F1-driven alternative inside the MBR candidates (the candidate set includes onset shifts of +-1, +-2). Assess the G sensitivity (G in {5, 9, 10}) and pick the candidate rule robust across them.
6. **Ordinal structure.** The onset time and run durations are ordinal; the exact chain posterior already encodes cumulative at-least-k events. If the primary underfits, add cumulative at-least-k heads for onset.
7. **Hard constraints at decode.** The output must be 10 letters from {S,A..F}; enforce by construction. If train shows hard constraints (for example certain modes never adjacent, E/F only after A/B, mutually exclusive pairs), recover them from TRAIN transition counts and mask impossible transitions in the chain (verify on every train case), combined with the model evidence, never as a pure constraint search.
8. **Calibration.** Per-slot (per class bias and per step-bucket temperature) logistic recalibration on OOF marginals, cross-fitted, with base-rate initialisation. The correctness of the posterior drives the MBR, so ECE per class and per step is a gating diagnostic.
9. **Loss-only variant (if decode tuning is disallowed).** Move the class weights into the NLL (1/sqrt(freq), plus a small extra weight on onset and on rare modes), keep the argmax-like decode with the per-step median hedge only. State the measured OOF cost versus the MBR decode so the reviewer can decide.

---

## Structural signals

Every invariant the data-generating process plausibly guarantees, and how each is turned into signal (all subject to the D-series checks):

1. **Arm exchange symmetry** (A<->B, C<->D, with the swap applied to history channels AND the 10-step target together). Use as (i) augmentation with doubled training data, (ii) an equivariant architecture with shared weights across the two arm blocks, or (iii) test-time averaging of the prediction and the un-swapped prediction of the swapped input. Assert D7 on train first, and compare gains (i) vs (ii) vs (iii) under V2. If the swap is only approximate, include an arm-identity flag.
2. **Observed/commanded lag.** The commanded mode at t-1 explains the observed mode at t. Causal alignment: pair features at lag 1 in the history, transitions in the chain encode the lag, and for the forecast horizon the first step is the most predictable (anchored to the last snapshot) while later steps are forecast-only; weight step losses uniformly but diagnose per-step F1 separately.
3. **Workflow phase as an explicit latent variable.** Rows are windows of longer recordings whose generating process is a workflow/state machine; the mixture of K chains is the learned version of that (no hand-written grammar). The gate g_k(x) infers the phase from the history; durations come from learned transition/hazard potentials.
4. **Run/segment structure.** Mode runs are persistent; the chain's self-transition and the hazard terms encode persistence; the tolerant metric uses the same notion.
5. **Window overlap = group structure.** Overlapping windows from the same recording are used (only on train) to define CV groups. Whether overlapping windows can be recombined into extra training windows is deferred (Q7).
6. **Onset coupling.** The onset is a function of the string, so the onset posterior is derived from the same chain (exact), and an auxiliary head enforces coherence.
7. **Symmetric / canonical forms.** Arm-symmetric summary features (sum and absolute difference of arm statistics) and canonical channel ordering are permutation-symmetric by design.
8. **Absent classes in a row.** Per-row macro averaging over a union of true and predicted classes (or over all 7 with a zero_division convention, Q3) shapes the decode: emitting a spurious class in a row can cost a whole class F1 under the per-row reading. The MBR evaluation uses the exact convention.
9. **Class-prior consistency.** Fixed rule: a row's decode must not use test-wide class frequency. Priors come from train only.

---

## Experiment roadmap

Stop criteria are given per step. One change per experiment; log id, change, V1 and V2 mean +- std, per-fold, three metric components, and runtime.

1. **Contract, metric and validation (no models).** Implement the metric with flags, unit-test it (D10), compute the persistence/all-S/prior baselines, onset-term ceiling, and the oracle ambiguity ceiling (D4). Build V1 and V2 groups and verify no recording/type straddles a fold. Stop: tests pass and the splits are leak-free.
2. **Cheap valid baseline, end to end.** Per-step multiclass logistic regression or a small LightGBM on raw + engineered features with plain argmax; write and validate `submission.csv` through the exact command. Stop: valid CSV and baseline V1/V2 recorded.
3. **Representation and structure.** (a) Per-step MLP; (b) add engineered features; (c) arm-swap augmentation or equivariance (D7-gated); (d) onset auxiliary head; (e) K=1 chain; (f) K=4, 8 latent mixture. Compare paired on the same folds with 3 split seeds, constants fixed (epochs, EMA window). Stop at the lowest rung that wins on V2 by more than noise; V1 as veto.
4. **Metric-aware loss and decode.** (a) Class weights in the loss; (b) loss-only variant (compliance fallback); (c) per-slot calibration cross-fitted; (d) MBR decode with candidate set and posterior samples; (e) G sensitivity. Stop: each step must beat noise in the three components jointly, with the onset term not collapsing.
5. **Diversity of assumption.** LightGBM per-step + onset member, blend by per-step log-prob weight (one weight, cross-fitted). Only if the blend beats the best single member by more than noise on V2.
6. **Bounded in-script HPO.** The fixed 12-config grid, fixed seeds, nested check on a held-out fold; plateau rule. No Optuna timeouts; if Optuna is used, fixed `n_trials` with a seeded TPE sampler and `n_jobs=1`.
7. **Final fixed-plan run.** From a clean `working/`, the exact command, run twice and diff the CSVs (expected identical or near-identical with the deterministic setup); runtime within 30 percent headroom; validator passes; prediction distribution (per-class token rates, onset distribution, all-S fraction) compared against OOF as a bug check only.

Credits (6 per problem): baseline (step 2), best single (after step 4), ensemble (step 5), final (step 7). Do not use public LB feedback to tune. Test first the techniques that two or three independent solvers would plausibly converge on, namely arm-swap augmentation, class-weighted loss with threshold decode, and onset-aware decode, before any proxy-driven tweak.

---

## Compliance audit

CLAUDE.md section 7 and the section B self-audits against this plan:

- Red: any code path reading the test file for something other than one-sample prediction? No. Test is used for schema, row count and per-row inference only. No statistics, vocabularies, scalers, clustering, dedup, rank normalisation over the test file, no stitching.
- Red: time inside conditions? None. Fixed epochs, rounds, seeds, grid, M and S in the decode; `time.time()` only in `log()`.
- Red: environment-dependent fallbacks? None. `device = "cuda"` fixed; no `try/except` model switching; no `os.cpu_count()`; fixed thread counts and workers.
- Red: constants tuned offline? None. Every constant is either a fixed design number or found by in-script train-only search (the 12-config grid, F-threshold fixed point, calibration, blend weight), cross-fitted. Class weights come from train counts by a fixed formula. The arm-swap mapping is asserted by a train diagnostic in-script.
- Red: external data, synthetic data, hosted weights, non-allowed libs? None. numpy, pandas, scikit-learn, torch, lightgbm only; no pretrained weights are needed at all. Posterior sampling inside decode is sampling from the model's own prediction, not training data; no new labelled training examples are created. Window-recombination augmentation is off pending Q7.
- Red: would the solution work with the ML removed? No: the decode needs the learned posterior; no lookup or grammar is hard-coded.
- Whole-test aggregation: none. A row's output depends only on its own history plus the train-fit model. Class priors, F-score thresholds and calibration constants are fixed from train OOF.
- Sibling leakage: no cross-row features at inference. In train, grouping by overlap is used only to build CV folds, not as a feature.
- Id/order features: `id` is not used as a feature.
- Genuine learning: the NN's trained parameters carry the prediction; log the parameter-change norm and the V2 gain of the chain over the persistence baseline and over the flat per-step head, to show the training is load-bearing.
- Orange: source readable and under 512 KB, comments explaining each reasoning step, plain code paths. Metric function, decode and validator are straightforward numpy.
- Challenge-specific text honoured: in-script training, no external data, no test-set-wide statistics, per-row independence, about 1 hour limit (plan is 25-40 min).

---

## Open questions & assumptions

**Questions for the reviewer / challenge text (each with the plan under both readings):**

- **Q1. Pooled vs per-row macro-F1.** "Every row is scored independently" could mean the composite is computed per row and averaged (then MBR with the exact row metric is exactly optimal and no pooled denominators are needed) or that only the prediction must be independent while F1 is pooled over all tokens (then add the F-threshold class weights). Plan A (per-row): decode = MBR on the exact row score. Plan B (pooled): same candidates, tokens weighted by 1/D_c and the p > F_c/2 rule. The cost of choosing wrongly is measured on OOF under both and logged; I recommend the per-row-exact decode as the default if unanswered, because it is also near-optimal under pooling.
- **Q2. Onset decay length G.** The paraphrase says "decays linearly with the step gap". Is the utility max(0, 1 - gap/G) with G = 9, 10, or other? Plan: parameterise G; if unknown, use the decode that is robust over G in {5, 9, 10}.
- **Q3. Class conventions.** Which classes enter the macro average (all 7, or only those present in truth or prediction), and what is the zero-division convention? This changes the cost of emitting a rare class spuriously.
- **Q4. Rows with no onset in the onset term.** Is the 0.2 term averaged over all rows (no-onset rows contribute 0) or only over rows with a true onset? The first reading gives a ceiling below 1 and the onset-hedging asymmetry above; the second makes onset commitment free.
- **Q5. Arm symmetry mapping.** Is the exchange exactly A<->B and C<->D, and are E and F unchanged by it? Are history channels in the same order as the target modes? To be asserted by D3/D7 on train; ask if ambiguous.
- **Q6. Is decode-time (MBR) tuning allowed?** If it is banned, the loss-only variant is the in-rules mechanism with its measured cost.
- **Q7. Is recombining train windows into longer recordings for extra training windows acceptable (real labelled units, not fabricated), or is it "synthetic data"?** Default: not used.
- **Q8. Are there train-only metadata columns (workflow id, recording id, offset)?** If so they define groups only and are never features.

**Assumptions (all UNVERIFIED):** history channels share semantics with target modes (D3); windows overlap and form recordings (D5); the two arms are approximately exchangeable (D7); S dominates the token distribution (D2); duplicated histories with different futures exist (D4); runtime figures are rough estimates until profiled on an A10G.

**What this plan could not verify (no data):** every number in Data findings, the channel-to-mode map, the group count, the ambiguity ceiling, the no-onset fraction, whether the latent mixture beats the K=1 chain, whether the arm swap helps, runtimes, and the expected score range (given only as an estimate, with the reasoning above).
