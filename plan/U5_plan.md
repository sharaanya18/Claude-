# U5 plan: snowflake air temperature from three 80x80 views (From Scratch, CPU only)

Sources read: eris-strategist.md, CLAUDE.md, tasks/user/U5_snowflake_temperature.md (nothing else; no dataset, no internet).
Facts quoted from the description are marked (D). Everything about the data that the description does not state is **UNVERIFIED** and is a diagnostic to run or a hypothesis. No solution code is in this plan.

---

## Contract & decision unit

**One valid answer.** `submission.csv` with columns `id,temperature_c`, exactly one row per test id (4,958 rows (D)), no missing, duplicate or unknown ids, any order allowed (D). Keep `sample_submission.csv` order anyway, `to_csv(index=False)`, finite floats (one decimal is the label format, not required of predictions). A malformed row is scored as the reference value -4.8, i.e. contributes zero score (D). Invalid and low-scoring are therefore equivalent per row, but a wrong id set or column set is dead last, so validate before writing.

**Id to image mapping.** `test.csv` / `train.csv` carry an `evidence` string of the form `test_images.npy:17` (D). The image row must be taken from that string (plain deterministic number extraction), not from file order. Assert: file name matches the expected npy, index in range, indices unique, and the set of indices covers all images. The parsed index is a pointer only; it must never be fed to any model (row order is "uninformative" and fingerprinting row order or identifiers is banned (D)).

**Metric (D).** e = MAE of predictions; b = MAE of constant -4.8 (the train median) on the same rows; score = max(0, 1 - e/b). Per term:
- Pure L1 on the temperature scale, so the optimum is the conditional **median**; no calibration to a mean, no squared-loss target.
- Linear in e: score gain = MAE reduction / b. With the quartiles (-8.5, -4.8, -1.9) (D) b is roughly 3.3 to 3.9 C (UNVERIFIED estimate; compute exactly on train), so +0.10 score is about 0.35 to 0.4 C of MAE.
- Floor at 0: the leaderboard cannot reward predictions worse than constant, but it does not punish them either, so downside risk is capped and shrinkage toward the median is a cheap hedge when a model is unreliable on shifted weeks. Never use the clipped score for model selection (it hides negative folds); select on the unclipped pooled `1 - sum(e)/sum(b)`.
- b depends on the evaluated rows: public and private shares each have their own b (D). If the test weeks are colder or warmer than -4.8 on average, b gets larger and the same MAE scores higher; this is outside our control.
- No per-example weights, no hierarchy in the metric. (But see decision unit: the test is 13 weeks with at most 80 flakes per day, so the effective test weighting is more day-balanced than train.)

**True independent unit: the calendar week.** 23 train weeks, 13 test weeks, no overlap (D). Within a week and within a day flakes share a snowfall and a thermometer reading, so rows are heavily correlated; effective sample size for generalisation is about 23 weeks (and perhaps 60 to 100 days; train has at least 24,939/400 = 63 days, test has at least 4,958/80 = 62 days, both UNVERIFIED lower bounds). Rows per week: about 1,084 train, about 381 test (arithmetic from D).

**Pipeline stages (diagnose separately).** (1) Representation: per-view geometry/texture features and/or learned conv features, which have to capture crystal habit (needle/column vs plate/star), riming/clumping/melt state, size. (2) Scoring: a regressor that outputs a conditional median (or a conditional distribution whose median is read off). (3) Decode: trivial (clip to the train range, optional 2-parameter OOF calibration). There is no candidate-coverage problem. Predictions must be a function of the row's own three views plus train-fit models only.

**Information ceiling is low and known to be low (D):** the thermometer is beside the cameras, not where the crystal grew, and weeks differ more than flakes. Expect modest absolute scores; the central engineering problem is week-to-week transfer, not in-week fit.

---

## Compliance regime

Domain: computer vision / "From Scratch" / CPU only (D). Map to CLAUDE.md section 6.3 and 6.8, with the challenge text overriding everything it touches (section 2.4).

**Explicit hard constraints from the description (treat every one as a ban):**
1. CPU only: 10 cores, 62 GB RAM, no GPU, no network, 1.5 h wall clock for training plus inference together (D).
2. From scratch: every fitted or learned state, including scalers, thresholds, calibration maps and any hyperparameter chosen by looking at data, must be created from the released training rows inside `solution.py` during the graded run. Networks start at random. No pretrained weights, embeddings or feature extractors (also bars timm/HF weights, which CLAUDE.md otherwise allows).
3. No external data; no attempt to date a row, identify a snowflake, or match against outside snowflake collections (automatic disqualification).
4. No rule engine as predictor: hand-written image processing may shape features, but the temperature must come from components fitted in the script.
5. No hosted inference or network access.
6. No use of test rows beyond ordinary per-row inference: no training on test, no pseudo-labels, no test-time adaptation, no statistic fitted over the test data as a whole, and **no grouping of test rows that look like the same snowfall** (so no kNN smoothing across test rows, no test clustering, no test-based rank/z normalisation, no batch-statistic BatchNorm at inference).
7. No hard-coded predictions, no fingerprinting identifiers or row order.

**Allowed (D):** CNN/attention networks (shared or per-view encoders), own features (size, outline, texture, brightness, branching, symmetry) into linear/tree/kernel/neighbour models, any metric-fitting loss such as absolute error, output calibration fitted on own week folds, augmentations built from train rows (flips, right-angle rotations of a view, small shifts, brightness changes, reordering the views) and averaging predictions over such transforms at inference.

**Conflicts between the description and CLAUDE.md / the general guidebook (the description wins; each resolution is listed):**

| CLAUDE.md / agent file says | Description says | Resolution |
|---|---|---|
| Assume one NVIDIA A10G, `device="cuda"`, fp16/bf16 autocast, GradScaler, GPU boosters, channels_last+AMP | CPU only, 10 cores | Hardcode `device="cpu"`, fp32, no autocast, no GPU tree params. All run-time estimates below are for the described 10-core CPU, not an A10G. The dev sandbox may have a different CPU: profile one fold and one epoch there before fixing counts (UNVERIFIED how many cores the sandbox has). |
| Target 35-50 min, ceiling 1.5 h with grace rarely granted, plan for 1 h worst case | 1.5 h wall clock for train+inference together, stated as the limit | Treat 90 min as the hard limit, no grace. Plan <= 55 min on the described machine (>= 38% headroom), because CPU timing is harder to predict than GPU timing and an overrun scores zero. |
| `os.cpu_count()`-derived workers forbidden; fixed thread counts | 10 cores stated | Hardcode `torch.set_num_threads(10)`, LightGBM `num_threads=10`, `joblib n_jobs=10`, DataLoader replaced by in-process tensor batching (`num_workers=0`). If the grader machine has fewer than 10 cores the run is slower but identical; keep headroom for that. |
| Internet allowed for HF/timm pretrained weights | no network, no pretrained anything | No `from_pretrained`, no `pretrained=True`, no `timm` (hand-write a lean net; also avoids a not-listed dependency). Set no HF env vars. |
| Early stopping on the validation metric is fine | Fitted state must come from train rows | Early stopping on the held-out fold biases the pooled OOF (selection on the evaluation set). Use fixed schedules plus in-script snapshot selection with a nested check (see Fixed work plan). |
| Strategist: derive own groups by union-find over content overlap/near-duplicates | Banned: identifying a snowflake or dating a row; weeks are already the guaranteed split in `validation_folds.json` | Use `validation_folds.json` only. Do not cluster train images into pseudo-snowfalls or pseudo-days, even for CV. |
| Strategist: repeated CV with different split seeds | Folds are fixed by week and only 5 given | Only one week-grouped split exists. Repeat model seeds (for noise) but not the split; judge on pooled OOF plus per-fold paired signs. |
| Guidebook: batch-level BN at inference is fine | No statistic fitted over the test data as a whole | Use GroupNorm (or BatchNorm in eval mode with train running stats, which is a train-fit state). Inference of a row must not depend on which other rows share its batch. |
| Strategist/CLAUDE.md: every hard-coded constant derivable by in-script train-only search | Same, and stated even more strictly (any hyperparameter chosen by looking at data) | Put every cheap numeric knob (GBDT config, number of rounds, CNN snapshot epoch, blend weight, calibration, loss variant) into an in-script, fixed-size, seeded search with a nested check. Keep dev-time-only choices to coarse structural design (capacity rung, input resolution) and document them as design choices; flagged as reviewer question 2. |
| Constants such as -4.8, -15.5, 3.7 | stated in the text | Compute the median and the train min/max inside the script from train labels; do not hard-code. |

**Self-audits on the plan (CLAUDE.md section 7 + agent section B):**
- Strip-the-ML: remove GBDT and CNN and nothing predicts temperature (features alone are just numbers). Pass. Weak spot: if one single feature (log flake area) carries most of the score, the CNN/GBDT only refines a one-variable rule; run the ablation "GBDT on area-only features" and report it so the reviewer sees the learned model does more than a size rule.
- No whole-test aggregation: every operation on test is per row (feature extraction, CNN forward, TTA over flips of that row, clip). No test normalisation, no vocabulary or scaler touches test. Pass. Unit test: predicting a row alone equals predicting it inside a batch.
- Sibling leakage: only relevant to train CV (weeks given). The three views of one row are the same flake by construction (that is the input, not leakage). Pass.
- Time/env branching: none; time only in `log()`.
- Row order / ids: ids used only to write the output; image index parsed from `evidence` only as a pointer.

---

## Data findings

Verified from the description (D): train 24,939 rows, test 4,958; images `uint8 (N,3,80,80)`, flake bright, background 0; one fixed pixel scale (true size preserved, large flakes can run off the edge); three fixed cameras, view order the same in every row, views not aligned with each other; target -15.5 to 3.7 C, median -4.8, quartiles -8.5 and -1.9, one decimal; 23 train weeks, 13 test weeks; train days thinned to <= 400 flakes, test days <= 80; falling snow only; 5 week-grouped folds in `validation_folds.json` (keys: n_folds, leakage_unit, guarantee, rows_per_fold, fold_of); fold scores spread widely; test weeks may sit warmer or colder.

Not verified (no data available). Run these on **train only** (test only for schema, row count, id format, memory sizing) and treat the stated expectations as hypotheses:

1. **Integrity.** dtype/shape/value range, fraction of views that are all zero, evidence-string parse (every id maps to a unique in-range row), `validation_folds.json` keys, rows per fold, `fold_of` covers every train row once. Expectation: 5 folds of roughly 5,000 rows, 4 to 5 weeks each.
2. **Target by fold.** Per-fold mean/median/std/quantiles and per-fold constant-(-4.8) MAE. Also the oracle "per-fold median" predictor's score: that is the share of variance sitting in week-level offsets that no flake can explain. Hypothesis: fold medians differ by several degrees, so a per-fold-median oracle scores well above 0 while the pooled constant scores exactly 0; this quantifies the week-shift risk and the size of the "irrecoverable" part.
3. **Exact b.** Compute b on pooled train and per fold; convert score points to degrees.
4. **Size diagnostics.** Per view: nonzero-pixel count (area), bounding-box extents, fraction of flakes touching the 80x80 border (censored size), max extent across the three views. Hypotheses: heavy right-skewed area; edge-touching is a few percent to tens of percent; size correlates positively with temperature (aggregation near 0 C, small sharp flakes in hard cold, per the description) with the sign stable across folds. Run per-feature Spearman per fold; keep features whose sign is stable across all 5 folds and flag the ones that flip (those are week-confounded).
5. **Cameras differ.** Per-view area, mean brightness and bbox-aspect distributions; whether view 0/1/2 have systematically different sharpness or scale. Hypothesis: yes (different geometry), so per-view encoders or a shared encoder with a view embedding should beat a naive shared-and-permuted one. Test view-permutation invariance by comparing GBDT with per-view features in fixed camera order against sorted-by-size order.
6. **Orientation.** Mean image and second-moment orientation per view. If gravity-aligned structure exists (falling plates present a preferred aspect), rot90 may destroy signal; if not, rot90 is free augmentation. Decide augmentation set by paired CV, not by assumption.
7. **Brightness / camera-state nuisance.** Distribution of nonzero-pixel intensity per view per fold, saturation fraction, speckle noise outside the flake. Hypothesis: part of this drifts by week (lens condition, illumination), which a model can use to memorise weeks. Test with a **nuisance-only GBDT** (global brightness, saturation, speckle count, no shape) under the week folds: score > 0 means nuisance carries week-linked temperature signal that may not transfer; then neutralise it (brightness jitter, per-image intensity normalisation) and compare. Do not train a "fold classifier" in the solution; as a train-only diagnostic keep it to distribution tables, because a week-dating classifier invites the "dating a row" reading.
8. **Random-split vs week-split gap.** Train a fast GBDT on handcrafted features with (a) ordinary random 5-fold and (b) the provided week folds. Hypothesis: (a) scores much higher (days/weeks leak), the gap is the memorisation budget. Report it to calibrate how optimistic any non-grouped number would be; never use (a) for selection.
9. **Duplicate / near-duplicate rows in train** (hash of the three views). Diagnostic only; expectation is none or few; nothing is used in the solution.
10. **Row index vs label.** Spearman of row index and label on train, expected about 0 (D says uninformative). Diagnostic only; never used.
11. **Label structure.** Histogram at 0.1 C resolution, counts in the tails (< -12, > 0), heaping at integer values. Hypothesis: sparse tails (few flakes near -15 or above 0), so the model will under-extend there; the capped output range plus MAE favours conservative predictions in the tails.
12. **Habit table.** Group train rows by 2 C temperature bins and show mean area, branching count, hexagonal-symmetry energy, compactness. Hypothesis: non-monotone habit signals (needle/column/plate regimes) visible as bumps in some features; verifies the features carry the physical story rather than just size.
13. **Information ceiling probes (train, week-grouped).** (a) area-only GBDT; (b) full handcrafted GBDT; (c) per-fold-median oracle. Hypothesis: (a) already captures much of what (b) gets; (b) gap above (a) is the learnable shape signal; (c) is far above both (the week offset is the unexplained part).
14. **Runtime sizing.** Memory: train 24,939 x 3 x 80 x 80 uint8 = 479 MB; float32 would be 1.9 GB, trivial against 62 GB. Time one CNN epoch and the feature extractor on a 2,000-row sample.

Expected outcome (hypotheses, UNVERIFIED): week-grouped pooled OOF score for the best model of 0.15 to 0.30; per-fold scores spread from about 0 (or negative before clipping) to about 0.4; the 13-week test lands below pooled OOF.

---

## Validation design

**Reproduction of the split (D).** Week-grouped: every calendar week lies in a single fold; test weeks are disjoint from train weeks; test days are thinned harder (<= 80) than train days (<= 400). Use `validation_folds.json` `fold_of` as the only grouping. Do not build content-derived groups (banned reading).

**Scheme.** Leave-one-fold-out over the 5 provided folds, all models trained on the other 4 folds (about 18 weeks vs 23 at shipping time). Metric re-implemented exactly (see below). Report: pooled OOF unclipped score `1 - sum(e)/sum(b)` (primary, as the description instructs), per-fold scores (5 numbers, report mean and std across folds), pooled MAE in degrees, and per-fold bias (mean signed error) to see warm/cold shift.

**Metric unit tests.** perfect predictions give 1.0; constant train median gives exactly 0.0 on pooled rows; constant train mean gives <= 0 (unclipped) pooled; reversed predictions around the median give a negative unclipped score and a clipped 0; a constant equal to the per-fold median gives >= 0 per fold (the oracle); predicting only integer-rounded perfect labels gives about 1 minus a tiny term.

**Noise handling.** Only 5 week clusters, so the standard error of any difference is large. Rule: accept a change only if (a) pooled OOF improves, (b) the paired per-fold difference is positive in at least 4 of 5 folds, and (c) the gain exceeds the seed-to-seed spread measured with 3 model seeds on the same folds (CNN: dev runs; GBDT: cheap). Per-row bootstrap CIs are misleading because rows are not independent (days/weeks); if a bootstrap is shown, resample **folds** only and say it is 5-cluster.

**Post-hoc selection needs its own held-out check.** All in-script selections (GBDT config, boosting rounds, CNN snapshot epoch, blend weight, calibration slope/intercept, loss variant) are made on pooled OOF of 4 folds and scored on the 5th (leave-one-fold-out of the selection, using stored OOF predictions; cheap because OOF predictions are already stored). Report the nested number next to the plain pooled one; the difference is the selection optimism.

**Proxy bias, direction and size.**
- Week-grouped CV removes whole weeks: honest for week shift, but trains on 18 not 23 weeks, so slightly **pessimistic** for the shipped model (more weeks help specifically because week count is the effective sample size).
- CV days are thicker (<= 400 per day) than test days (<= 80): CV score is more driven by a few heavy days, so it is probably slightly **optimistic or at least noisier** than a day-balanced test; sign unknown, size unknown.
- The test has only 13 weeks and may be warmer/colder than train (D): expect the realised test score to be **below** pooled OOF by an amount comparable to the fold-to-fold std, with a left tail (a band, not a point). Estimate for the private score: 0.08 to 0.25, centre about 0.15 (UNVERIFIED; reasoning: description says part of every error is irrecoverable, weeks differ more than flakes, expect a gap, and the leaderboard share is a subset of 13 weeks).
- Random-split CV (diagnostic 8) is strongly optimistic and must not be used for anything except sizing the memorisation gap.

---

## Overfit/underfit risks

**Overfitting**
1. **Week fingerprints** (23 effective groups). Camera illumination/brightness/noise state or weather-correlated imaging artefacts let a flexible CNN or deep GBDT memorise which week a flake came from and output that week's temperature. Mitigations: grouped CV catches it; brightness-scale jitter and per-image intensity normalisation (compare with and without by paired CV); shallow capacity; weight decay; GroupNorm; small flips/shifts; large `min_data_in_leaf` (hundreds) so no leaf can be one day's flakes; nuisance-only GBDT diagnostic; keep absolute-brightness features out of the GBDT unless they pass the stability check (sign stable across folds).
2. **Day dominance**: train days have up to 400 near-correlated flakes. Strong subsampling-style regularisation (row subsampling in GBDT, dropout, flake-level not day-level augmentation). Cannot reweight by day (days are not given and dating is banned).
3. **Selection on 5 noisy folds.** Few knobs, small grids, nested check, paired-fold sign rule.
4. **High-capacity head on tiny effective sample**: use the capacity ladder: handcrafted features + regularised GBDT, then a small from-scratch CNN (about 0.2 to 0.4 M parameters); stop at the lowest rung that wins under week-grouped CV.
5. **Extrapolation to warm/cold weeks**: trees cannot extrapolate beyond train range; clip outputs to the train min/max and let MAE favour the median.

**Underfitting**
1. Information thrown away by down-sizing: keep 80x80 resolution in the first layers (stride-2 5x5 stem is the max reduction); fine branching and rime speckle sit at pixel scale.
2. Size and absolute scale lost by pooling/normalisation: view is a fixed pixel scale and size is informative. Use sum/mean pooling of unnormalised activations plus max pooling, avoid per-image normalisation that erases flake area, and concatenate explicit log-area/extent/edge-touch scalars into the CNN head (hybrid input is allowed (D)).
3. L1 gradient is constant-magnitude and optimises slowly: smooth-L1 (small delta) warm phase then pure L1, or an ordinal/distributional head (see decode) whose median is read off.
4. Too few epochs for a CPU-limited from-scratch net: budgeted explicitly below; GBDT on handcrafted features is the underfit insurance.
5. Heavy regularisation can collapse predictions to the median and give score about 0: monitor prediction spread vs target spread (OOF std ratio) and slope of the OOF calibration regression; a slope well above 1 signals over-shrinking.
6. Loss/metric mismatch: use L1 or median-reading heads; never MSE then report MAE.

---

## Recommended approach (primary + fallback)

### Primary: two-arm blend, both from scratch, one conditional-median output
**Arm A: handcrafted geometry/texture features into LightGBM (L1 objective).**
- Per view (cameras fixed, so keep per-view columns in camera order) and cross-view aggregates (min/max/mean over the three views, plus sorted-by-size versions): area, bbox width/height/extents, perimeter, convex-hull solidity, compactness, eccentricity and orientation from second moments, Hu moments, radial intensity profile (binned by radius from the centroid), angular harmonic energy in polar coordinates (m = 6 for hexagonal plates/stars, m = 2/4 for columns/needles), skeleton length, number of skeleton endpoints and branch points (dendrite-style branching), number of connected components (clumps, rime droplets), fraction of very bright pixels, intensity mean/std/percentiles over the flake, local-gradient energy (sharpness), 2D FFT radial power spectrum bins, edge-touch flags and a censored-size indicator. Hand-written image processing here only shapes inputs; the output comes from the fitted trees (D allows it).
- LightGBM: objective L1 (compare Huber with a small alpha), learning rate about 0.03, strongly regularised (num_leaves 7 to 31, `min_data_in_leaf` 100 to 800, `feature_fraction` 0.3 to 0.7, `bagging_fraction` 0.5 to 0.8, `lambda_l2`), `deterministic=True`, `force_row_wise=True`, `num_threads=10`, fixed seeds.
- In-script search: a fixed grid of 8 configs on the 5 week folds with nested leave-one-fold-out selection (reuse stored OOF predictions), number of rounds chosen from checkpointed fold curves by pooled OOF.
- Ship: fit on 100% of train with the selected config (cheap, adds the 23-week benefit); fixed rounds from the in-script selection.

**Arm B: from-scratch CNN over the three views, trained with an absolute-error loss.**
- Shared small conv encoder over the 80x80 views (stem 5x5 stride 2, 3 to 4 stages, 16-32-64-96 channels, GroupNorm, ReLU/GELU, no pretrained anything), plus a learned per-view embedding (cameras differ) so that features are tagged by camera. Pool each view with mean + max (and size scalars), concatenate in fixed camera order, small MLP head (dropout), output scalar temperature (standardised by train median/MAD for conditioning) in the L1 variant, or a 1-C-bin ordinal head (see Metric-aware training) in the distributional variant. Capacity about 0.2 to 0.4 M parameters: the lowest rung that can learn texture; do not climb to wider/deeper without paired-fold gain.
- Optimiser AdamW, warmup + cosine, grad clip 1.0, EMA of weights, fixed epochs (about 8 to 12, set after profiling), fixed seeds, batch 128, in-process batching (no DataLoader workers).
- Augmentations (all listed as allowed (D)): horizontal/vertical flips, small shifts (+-4 px, zero fill since the background is 0), brightness scale jitter, right-angle rotations and view reordering **only if** paired CV shows they do not hurt (orientation and camera identity may carry signal). In-network view-level dropout (zero an entire view with small probability) as a regulariser against reliance on one camera. No mixup/cutmix (fabricates blended flakes, which risks the synthetic-data reading).
- Test-time: average over the same flip set (and rot90/reorder only if training used it); each row is processed independently.
- Ship: the 5 fold models averaged (each trained on about 18 weeks; also yields the OOF predictions), plus optionally a 100%-train refit only if profiling leaves headroom and a CV transfer check is positive.

**Blend:** `p = w * CNN + (1 - w) * GBDT` with one weight w in [0,1] fitted on pooled OOF (grid of 11 values), nested check; then an optional 2-parameter linear OOF calibration `a + b*p` (accepted only if the nested check improves in >= 4/5 folds), then clip to [train min, train max]. Because both arms predict medians on the same scale, simple weighted average beats raw-probability tricks; no stacking learner needed with 5 clusters.

**Why this is the leanest design that encodes the structure:** per-view physics features carry the habit signal at almost no overfit cost; the CNN adds learned texture/branching evidence the hand features miss; the blend is diverse in assumption (hand features vs learned filters), one weight. Joint cross-view attention, mixtures, sampled decoding stay out of the primary (primary-design gate).

### Fallback: Arm A alone
Same GBDT pipeline with fewer features if time is short. Cheap (about 15 min total), fully valid, deterministic, no GPU-like numerics risk. Ship it if the CNN does not beat Arm A by more than noise under the week folds, or if the CNN budget cannot be met with at least 30% headroom. It is also the safe default if a run-to-run diff of the CNN shows instability.

### Secondary candidate (diversity, only if both above are measured and time remains)
Kernel/neighbour model on a few standardised, stability-checked handcrafted features (e.g. SVR/ridge-on-random-Fourier or kNN median): allowed (D), but week-fingerprint risk is high for neighbour methods and the gain is likely within noise; test it only as a roadmap step 5 member.

---

## Rejected options

- **Any pretrained backbone, embedding or feature extractor**: banned (D). Also rules out timm/HF downloads that CLAUDE.md otherwise allows.
- **Self-supervised or autoencoder pretraining on train images in-script**: permitted in principle (train data only) but infeasible on 10 CPU cores within the budget and unlikely to beat supervised training with 25k labelled rows. Pretraining or contrastive learning on **test** images is banned.
- **Grouping test rows by snowfall, kNN/graph smoothing across test rows, test-day clustering, test-time adaptation, pseudo-labels**: explicitly banned (D). This is probably the single biggest available lever (flakes in the same snowfall share one temperature) and it is off the table; do not even prototype it on train in a way that could migrate into the script.
- **Dating rows / identifying snowflakes / matching outside collections**: automatic disqualification (D). Also no pseudo-week clustering of train for CV.
- **Large or high-resolution deep networks, wide ResNets, attention over all patches of three views**: infeasible on CPU in 55 min and the wrong rung of the capacity ladder for about 23 weeks.
- **Raw-pixel GBDT or raw-pixel kNN as the main model**: poor inductive bias and maximal week-fingerprint exposure; raw-pixel tabular models are also a rejection pattern in CLAUDE.md for CV.
- **Mixup/CutMix/synthetic flakes**: fabricates blended entities; avoid (synthetic-data reading).
- **Rule-engine mapping from size or habit to temperature**: banned (D).
- **Mean-target losses (MSE) or log-target transforms**: metric is MAE in degrees; targets span negative values.
- **Early stopping on the held-out fold as selection**: optimistic OOF; replaced by fixed schedule + nested snapshot selection.
- **Day/time reweighting**: needs dating the rows.

---

## Fixed work plan & runtime budget

**Hardware assumption (the description's, not CLAUDE.md's): 10 CPU cores, 62 GB RAM, no GPU, 90 min hard limit for train+inference.** Target <= 55 min wall (38% headroom). All figures are estimates, UNVERIFIED until profiled on a machine like the grader.

Fixed constants (placeholders to be fixed after profiling, then never changed by clocks or environment): seeds {42}, `torch.set_num_threads(10)`, LightGBM `num_threads=10`, `joblib n_jobs=10`, batch 128, `device="cpu"`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `PYTHONHASHSEED=0`, `CUBLAS_*` not needed on CPU, MKL/oneDNN thread counts fixed via env set before importing torch (`OMP_NUM_THREADS=10`, `MKL_NUM_THREADS=10`).

| Stage | Work | Estimate (10 cores) |
|---|---|---|
| S0 Load/validate | read csv/json/npy, parse evidence, assert mapping and fold file | < 0.5 min |
| S1 Handcrafted features | 29,897 rows x 3 views = about 90k images; vectorised moments, FFT, polar harmonics; scipy.ndimage labeling; skimage skeleton in a fixed `n_jobs=10` pool | 3 to 5 min |
| S2 GBDT grid | 8 configs x 5 folds, about 400 to 600 features x 20k rows, about 600 rounds each, about 10 to 15 s per fit | 8 to 12 min |
| S2b GBDT final | 100% fit with selected config | 1 to 2 min |
| S3 CNN OOF | 5 folds x E=10 epochs x about 20k rows; target throughput about 500 to 800 rows/s (about 3 x 15 MFLOP per row forward) so about 30 to 40 s per epoch, about 6 min per fold; plus snapshot inference with flip-TTA on held-out and test rows at 3 snapshot epochs | 28 to 35 min |
| S4 Blend/calibration/nested checks/writing | arithmetic on stored OOF arrays | < 1 min |
| Logs, validation, re-read of CSV | | < 0.5 min |
| **Total** | | **about 42 to 55 min** |

Rules for stability of this plan:
- If profiling shows the CNN cannot fit, cut width/epochs/folds statically (for example 3 snapshot epochs, E=8, smaller stem), then re-time; do not add any clock-based branch. A CPU-only dev replay is mandatory because there is no GPU path to hide behind.
- Memory: uint8 arrays held (0.48 GB train, 0.095 GB test), float32 batches built on the fly; GBDT matrices about 25k x 600 x 4 B = 60 MB; peak well under 10 GB.
- Determinism on CPU: fixed thread counts, `deterministic=True` and `force_row_wise=True` for LightGBM (multi-thread histogram sums are the usual non-determinism), `torch.Generator` for shuffling/augmentation draws, in-process batching, seeds for numpy/random/torch. Run the full script twice and diff predictions (expect equality or ~1e-6 for the CNN; if the CNN differs by more, set single-thread BLAS for the final or average over fewer, more stable members).
- Snapshot-epoch selection: save held-out and test predictions at 3 fixed epochs per fold (for example epochs 6, 8, 10 of the EMA weights), choose one global epoch by pooled OOF with the nested leave-one-fold-out check. This puts the "when to stop" choice inside the script without wall-clock or fold-triggered early stopping.
- Validation inside the script before writing: columns equal `sample_submission.csv`, ids identical set and order, row count 4,958, no NaN/inf, values within [train min, train max], re-read the written CSV. Raise before writing a broken file.
- No fallback path that silently differs: if `validation_folds.json` is missing or malformed, raise (see open question 1).

---

## Metric-aware training & decode

(i) **Back-solving the metric.** score = 1 - e/b with e an L1 error, b fixed by the test rows: the optimum is the per-row conditional median; any monotone information in the target scale is irrelevant. Labels have one decimal, so predictions need no rounding (rounding gains nothing and loses if wrong).

(ii) **Weights and hierarchy.** The metric is a flat mean over rows; no per-example weights are justified. Week/day reweighting would need dating the rows (banned), so none.

(iii) **Distributional alternative to plain L1 (roadmap step 4).** Instead of a scalar output, predict the class distribution over about 19 bins of 1 C (from the train min to max, computed in-script) with cumulative "at least k" binary heads (ordinal BCE) or a softmax with a distance-aware soft label; read off the **median of the predicted CDF** (linearly interpolated) as the output. Rationale: week-level ambiguity makes the conditional distribution broad and possibly bimodal; the median of a well-fit predictive distribution is the MAE optimum, and the ordinal loss gives denser gradients than constant-magnitude L1. Compare against scalar smooth-L1/L1 under the same folds and keep the better (paired folds); do not blend the two unless each earns it. The same median-from-distribution trick can be applied to the GBDT (LightGBM multiclass/ordinal binning), but L1 or quantile 0.5 is the lean default and the multiclass GBDT costs about 19 x more trees; only test it if time remains.

(iv) **Decode.** No enumerable output set; output is a real number. Steps: median read-off, blend, optional linear OOF calibration, clip to [train min, train max]. Calibration is fitted per outer fold on the other four folds' OOF and reported cross-fitted; accepted only if it helps in >= 4/5 folds. A slope > 1 on OOF predictions would mean the models are over-shrunk (common with L1 + heavy regularisation); a slope < 1 means over-dispersion that week shift will punish.

(v) **Shift hedge (decide by evidence, not by guess).** The test has 13 weeks that may be warmer or colder than train. Where the fold-level bias (mean signed error) is systematically toward the train mean, no remedy is allowed from test statistics. The only in-rules hedge is shrinkage fitted on train week folds: the blend weight and calibration selected by OOF already encode how much shrinkage the week folds reward.

(vi) **Censored/edge cases.** Flakes running off the image edge have a lower-bound size; add edge-touch flags and the extent as features so the models learn the censoring rather than reading it as a small flake; do not hard-code a correction.

(vii) **Loss details.** CNN: smooth-L1 with a small delta (about 0.5 C on the standardised scale equivalent) in the first epochs, then pure L1; targets standardised by train median and MAD (computed in-script); output bias initialised at the train median (0 in standardised units). GBDT: objective `l1` (or `huber`), `boost_from_average=True` so leaves start at the median.

---

## Structural signals

1. **Three cameras of one flake**: every row is the same crystal from three fixed directions. Use per-view columns in fixed camera order (cameras differ), cross-view sorted statistics (max/min/mean extent: the largest view approximates the true 3D size), and cross-view agreement (spread of area/aspect across views separates compact 3D shapes (rimed, graupel, melted) from thin plates/needles whose projected size depends on viewing angle). Check swap invariance by a train diagnostic rather than assuming it.
2. **Fixed pixel scale**: absolute size is information; keep size-preserving operations; scale augmentation (zoom) is forbidden by logic (not in the allowed list and would corrupt the size signal).
3. **Background is zero**: flake area, mask-based shape features and mask-restricted intensity statistics are exact; noise/speckle outside the flake is a camera-state fingerprint to be audited (nuisance-only diagnostic), not a feature to be exploited.
4. **Symmetry**: horizontal/vertical flips are almost surely label-preserving for crystal habit; rot90 and reordering of views are allowed (D) but to be verified by paired CV because camera orientation may carry gravity-aligned information. Hexagonal (m = 6) angular harmonics are theory-motivated features for plates/stars/dendrites; columns/needles show up as elongation/aspect plus m = 2 energy.
5. **Physical habit ranges** (from the description: needles/columns in some ranges, plates/stars in others, clumping/riming/melt near 0 C, small sharp flakes in hard cold): encode as features (aspect ratio, branching count, compactness, speckle fraction, edge sharpness) feeding a learned scorer; never as thresholds mapped to temperature (rule-engine ban).
6. **Hierarchy of evidence**: the temperature at the cameras is a noisy proxy of the growth temperature aloft; the label noise is week-correlated. Use L1/median targets (robust) and heavy regularisation; treat the achievable score as modest.
7. **Per-row independence at inference** (a hard structural constraint from the ban on test grouping): prediction function f(views of this row) only; GroupNorm / eval-mode BN with train running stats; unit test that single-row inference equals batched inference.

---

## Experiment roadmap

Order and stop criteria (do not start a later step before the earlier ones are solid; one change per experiment; log table: id, change, pooled OOF, per-fold scores, std across folds, seed spread, est. runtime):

1. **Contract, metric, validation (stop when unit tests pass).** Parse evidence, read `validation_folds.json`, implement metric (pooled unclipped, per-fold, clipped), unit-test on perfect/reversed/constant/mean predictions; run diagnostics 1 to 14 above on train. Stop criterion: constant -4.8 scores exactly 0 pooled; fold file covers every row once.
2. **Cheapest valid end-to-end baseline** (about 20 min): area/extent-only features + LightGBM L1 on the week folds, write a valid CSV, run the validator. Record the area-only ceiling (this is also the "does the learned model do more than a size rule" ablation).
3. **Representation and structure**: full handcrafted feature set into GBDT (Arm A). Ablate feature groups (size, outline, branching/skeleton, symmetry harmonics, texture/FFT, brightness, cross-view aggregates) by paired folds; drop brightness-only features if the nuisance-only diagnostic scores > 0 and they do not pass sign stability. Stop when added groups stop beating seed noise and the 4/5-fold sign rule.
4. **Metric-aware loss/decode**: L1 vs Huber GBDT; CNN scalar smooth-L1→L1 vs ordinal-median head; OOF calibration. Stop when the best variant beats the others by the paired-fold rule.
5. **From-scratch CNN (Arm B) on the capacity ladder**: start at the smallest encoder, shared + view embedding, flips + shifts + brightness jitter; add rot90/view reordering/view dropout/explicit scalar inputs one at a time. Compare per-view encoders vs shared (parameter count rises 3x). Log train loss vs held-out score per epoch (memorisation check: loss falls while held-out plateaus), fix epochs and snapshot set as constants. Report the zero-information floor (constant median = 0) and Arm A as the yardsticks the CNN must beat. Stop climbing at the first rung that beats the previous rung under the paired-fold rule.
6. **Blend and in-script search**: fixed 8-config GBDT grid with nested selection; blend weight grid of 11; calibration; snapshot selection. Verify nested vs plain pooled gap (optimism).
7. **Diversity (only if measured gain > noise)**: second CNN seed or a per-view-encoder variant; kernel/neighbour member; each needs standalone quality close to the primary.
8. **Final fixed-plan run from a clean working dir, executed twice and diffed**; check prediction distribution vs OOF distribution (mean/std/range, no constants), runtime with >= 35% headroom, CSV validator, compliance checklist.

Credit use: baseline (step 2), best single (best of Arm A / Arm B), blend, final. Judging uses pooled OOF; the public leaderboard is a weak sanity check on 13 weeks.

---

## Compliance audit

CLAUDE.md section 7 against this plan:
- Test file read for anything other than one-row-at-a-time prediction? No. Test is read for schema/ids/images and per-row inference only.
- time-based branches? None; `time.time()` only in `log()`.
- `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks? None; fixed CPU config.
- Hard-coded constants tuned offline? Numeric knobs searched in-script (nested). Coarse structural design (net size, stem stride, feature list, augmentation family) is dev-time judgement and is the residual risk; question 2.
- External data / synthetic data / self-hosted weights / non-allowed library? None. Libraries: numpy, pandas, scipy, scikit-image (core Kaggle stack; verify installed), lightgbm, torch. No timm/transformers.
- Strip-the-ML: passes; area-only ablation reported.
- Source < 512 KB plain Python, no blobs; comments explain reasoning.
- Challenge restrictions honoured: CPU only, 10 cores, 1.5 h, from scratch (random-init weights, in-script scalers/thresholds/calibration), no test grouping, no dating, no pretrained, no rule engine.
- Agent section B self-audits: no whole-test aggregation (rank/z-score across test absent); no related-row grouping at test; label-derived statistics: none used as features (no target encoding of any group id; train labels only enter through model fitting). Nothing derived from train+test.
- Output validity: `validate_submission` against `sample_submission.csv`; evidence-to-row mapping asserted.

Residual "grey" items to surface (see below): reading `validation_folds.json` during the graded run; the choice of dev-time design constants; in-network view dropout; fold-model averaging at test time; whether skeleton/branching feature extractors count as "hand-written image processing that shapes features" (explicitly allowed (D)).

---

## Open questions & assumptions

**Reviewer questions**
1. Will `validation_folds.json` be present in the public dir during the graded run? The plan reads it to build the week folds used for the in-script nested selection, calibration and snapshot choice. Reading it is explicitly consistent with "calibration fitted on your own week folds" (D), but if absent the script must fail loudly (no fallback). Reading A: present, use as is. Reading B: absent; then every in-script choice would need a different grouping, and the only compliant substitute without dating rows would be fixed design choices (GBDT default config, fixed epochs, blend 0.5, no calibration). Check `task_manifest.json` at implementation time (not read here).
2. "Any threshold or hyperparameter chosen by looking at data must be created inside solution.py." Does a fixed architecture/augmentation family, selected during development and then hard-coded, count? Conservative reading: only choose data-dependent numeric knobs inside the script (done) and keep dev-time choices coarse. Cost of the conservative reading: the CNN search is limited to a few snapshot epochs and the loss variant; extra search time would eat the CPU budget.
3. Is in-network view-level dropout acceptable? It is a regulariser inside the model, not an augmentation in the allowed list. If not, drop it (small expected loss).
4. Is averaging 5 fold models plus a 100%-data GBDT acceptable as the shipped predictor (no test use beyond per-row inference)? Expected yes.
5. Does hand-computed skeletonisation/connected-component branching count as "features you compute yourself" (D lists branching, outline, symmetry)? Expected yes.

**Assumptions to verify at implementation time**
- The grader machine has the stated 10 cores, 62 GB; the dev sandbox timing correlates with it. If the sandbox has fewer cores, scale the estimates and keep headroom.
- scikit-image and scipy are available; if skimage is not, the skeleton features are implemented with scipy/numpy morphology (fixed code path, no try/except fallback).
- The three camera streams differ in scale or sharpness (hypothesis 5); if they do not, a shared encoder without a view embedding is simpler.
- Heavy use of size and shape features is not a "rule engine": confirmed by the strip-the-ML test and the area-only ablation.
- Efficient from-scratch CNN throughput (500 to 800 rows/s) is an estimate; profile first and fix epochs/folds accordingly.
- Expected pooled week-grouped OOF 0.15 to 0.30 and private 0.08 to 0.25 are estimates only; no number is promised.

**What could not be verified in this run:** every property of the data beyond what the description states (distributions, fold structure, nuisance drift, camera differences, duplicates), the sandbox CPU and library availability, the CNN throughput, and the contents of `validation_folds.json` / `task_manifest.json`.
