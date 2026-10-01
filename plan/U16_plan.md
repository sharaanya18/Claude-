# U16 Flash Refuge - Eris plan (strategist)

Status: written from the challenge text and CLAUDE.md only. No dataset files were available, so nothing about the data is verified. Every statement under "Data findings" that is not quoted from the description is a hypothesis (marked UNVERIFIED) together with the exact diagnostic that would confirm or refute it. All numbers for N (scenes, queries, rooms) are placeholders to be replaced after diagnostic D0.

## Contract & decision unit

**One valid answer.** For each test `id`, a JSON array of at most 70 distinct integer masks. Each mask has bits in 0..7 and 2 to 6 bits set, no mask is a strict subset of another (antichain), and no mask has more than `budget` bits. `[]` is the written form of the empty family. Output file columns: exactly `id,portfolios`, every test id once, `to_csv(index=False)`.

**Invalid vs low-scoring.** Invalid (rejected, scores dead last): malformed JSON, bool/float/string items, repeated masks, more than 70 masks, a mask with fewer than 2 or more than 6 bits, a non-antichain, duplicate or missing ids, extra columns, empty field. Valid but wrong: any mask outside the true family (including masks over budget) is a false positive.

**What the metric rewards.** Per case: 1 if true and predicted families are both empty; 0 if exactly one is empty; else Dice over exact masks, 2|T∩P|/(|T|+|P|). Final = mean over cases, floored at 0.01 and capped at 1. Consequences:
- It is a set-F1 over masks, so predicting extra masks costs precision and omitting masks costs recall.
- "Empty" is an all-or-nothing outcome. If the truth is empty and we output anything, the case is 0. If the truth is non-empty and we output `[]`, the case is 0.
- Near-miss families score partial credit only when masks match exactly.

**The deterministic structure (this is the key reduction).** The true family is a deterministic function of the 8 x J binary matrix A (A[k,j] = "candidate k adequate in region j") and the budget:
- Coverage is monotone in the portfolio, so a portfolio S is successful iff every region has at least 2 adequate members within S. It is inclusion-minimal iff it is successful and every single-member removal fails. The enumeration therefore needs only "success" and "all single removals fail" over the at most 255 non-empty subsets of 8 candidates.
- A[k,j] is determined by the gain g[k,j] = (median_cand[k,j] + 0.02) / (median_scout0[j] + 0.02) against the interval [lower_j, upper_j] (inclusive). The scout-0 median is exactly computable from the inputs, so the only unknown is median_cand[k,j], the median luminance of candidate k in rectangle j.
- So the learnable quantity is the 8 per-scene candidate luminance fields (or just their rectangle medians). The query, the budget and the family enumeration are exact code (the task definition, not a solver; see Compliance regime).

**True independent unit.** The physical room (`group_id`), not the query. Queries of one scene share the same 8 fields, scenes of one room share lighting and geometry, and the 8 candidates of a scene share scene-level exposure errors. The effective sample size is the number of rooms, which is the quantity that decides how much model capacity is safe.

**Pipeline stages (diagnosed separately).**
1. Response estimation: predict log(candidate luminance + 0.02) fields (or region medians) from scouts and probe observations. Diagnose by region-level log-gain MAE and by bit accuracy of A.
2. Calibrated uncertainty: turn a point gain into P(A[k,j]) and a sampling distribution over A, with errors shared across regions of the same candidate.
3. Decoding: exact enumeration of minimal portfolios under each sample, then expected-Dice-optimal selection of the output family, including the empty family as a candidate.
- Candidate-coverage analogue: the oracle (gold-A) decode must reproduce 100% of training targets. This proves formula, offset, inclusive bounds, median definition and enumerator are correct before any model exists.

## Compliance regime

**Domain.** Computer vision, regression of illumination response plus structured set decoding. Not labelled "from scratch" or "fine-tuning"; the description permits general pretrained visual weights and any architecture. The compute statement is: one GPU, 24 GB, one hour for setup, training and complete test inference. I treat the one-hour number as a hard ceiling and plan for <= ~40 min on an A10G (CLAUDE.md's 50 min target is overridden by the shorter stated limit).

**Explicit bans in the description (hard constraints).**
1. Do not substitute a training scene's measurement map for a test scene's prediction. This forbids nearest-neighbour retrieval/transplant of training maps or training gains into test scenes (the model must produce the test prediction from the test scene's own inputs).
2. Do not identify or retrieve the original capture archive, recover test photographs from outside repositories, fingerprint public imagery against external material, or look up source identifiers. (No internet access beyond pretrained-weight download; the plan never tries to recognise the scenes.)
3. Do not use creator construction artifacts, archive ordering, filenames or file sizes as label channels. So: no `scene_id`, `id`, `scene_path`, row order, file size, npz byte patterns or file order as features. `group_id` is used for fold construction only (and only train has it as a validation instrument).
4. Output rules (bans on format): exact column names, no extra columns, no repeated masks, no floats/bools/strings in the array, at most 70 masks, antichain, mask size 2..6, over-budget masks count as false positives.
5. Semantic rules that are not "bans" but constrain implementation: luminance is computed before the median (no per-channel medians); candidate maps are never added together; scout 0 is the denominator for every candidate; past scouts are never selectable.
6. From CLAUDE.md (Eris-wide): no external or synthetic data, no pseudo-labels, no fitting on test (scalers, vocabularies, PCA, calibrators, noise models), no whole-test aggregation, no wall-clock branching, no environment fallbacks, no hard-coded offline-tuned constants, weights only from HF/timm with pinned revisions.

**Explicitly allowed by the description.** General pretrained visual weights; any architecture; any correct method of enumerating portfolios from learned response estimates (so exact enumeration and an expected-Dice decode over a learned noise model are within the stated rules); training on `candidate_luminance`, on derived patch gains, or directly on portfolio targets; using sphere observations as model inputs; using the released query as the task evidence.

**Where the description is silent (my assumptions).**
- CPU count and RAM are not stated: assume small (<= 4 cores, >= 16 GB). Keep all training data as GPU tensors, no DataLoader workers, no `os.cpu_count()`. Fix `torch.set_num_threads(4)`.
- Network access for the pretrained-weight download is not stated explicitly; CLAUDE.md says HF/timm download is allowed. Assume reachable and pin the revision. No try/except fallback (a failed download should fail loudly; if network is risky, an alternative "no pretrained weights" configuration is a hard-coded design decision made before submission, not a runtime branch). Download time counts toward the one hour, so choose a small/base-size backbone (<= ~350 MB).
- Number and size of test scenes/queries are not stated: runtime is planned for an assumed range and re-profiled; counts are then hard-coded.
- Whether GPU is exactly an A10G is not stated ("a single GPU with 24 GB"): plan memory for 24 GB and use bf16/fp16 autocast with the same precision path in dev and grading.
- The description says the evaluator may score only one shard; not relevant to modelling, but the full test must still be predicted and valid.
- Description does not say whether per-scene reuse of computations across a scene's test queries is acceptable. I treat per-scene caching of the 8 predicted fields as legitimate (it is a function of that scene's own tensors only) and forbid anything that pools different test queries' regions/bounds.
- It does not say whether using query bounds as model input is acceptable; the query is a stated model input ("each query specifies...") and test.csv has the same feature columns. I include it only in an optional, ablated second stage (see Recommended approach, Open questions).

**Domain-by-domain allowed/grey/banned.**
- Allowed: frozen or fine-tuned pretrained backbone; trained conv decoder on released maps; derived region gains from released maps; flip augmentation applied consistently to images, spheres, labels and rectangles; flip TTA; physical-model features (linear-light interpolation between scout probes) as inputs to a trained model; calibrated sampling and expected-Dice decoding.
- Grey (reviewer questions listed at the end): query-aware calibration (bounds as features); hand-built physics prior as an input channel; use of train scout images as extra relighting targets.
- Banned: retrieval of training maps for test scenes; any identification of the archive; any use of ids/paths/file sizes; any test-set statistic; time-dependent control flow; synthetic labelled data (rectangles cut from real train maps are derived labels, not synthetic scenes).

**Strip-the-ML test.** Remove the trained predictor and replace it with the best non-learned stand-in (global exposure ratio from probe means, or the linear-interpolation prior alone, or a constant): the family decode then scores far below the trained system (diagnostic D5/D6 and roadmap step 2 measure this honestly). The enumerator and the expected-Dice selector are the task definition and the metric's decision rule, not a solver of the physical question. If the physics-prior-only variant turns out to score within noise of the trained model, the plan flags it and promotes the learned component by making the prior only an input (never an output shortcut).

## Data findings

Known from the description (not data findings, just contract): train inputs per scene are scouts (3,160,240,3) uint8, scout calibration (3,32,32,3), candidate calibration (8,32,32,3); a training measurement file gives candidate_luminance float32 (8,160,240); train.csv has `id, scene_id, group_id, scene_path, query, portfolios`; test.csv has the same minus `portfolios`; query has budget in {4,5,6} and 3 to 6 regions.

Everything below is UNVERIFIED. Each item is a diagnostic to run on TRAIN (test only for schema, row counts, id format, array shapes, for runtime planning), with the hypothesis I expect and what it decides.

- **D0 Inventory.** Rows in train/test, number of scenes, number of rooms (group_ids), queries per scene, scenes per room, regions per query histogram, budget histogram, test scene count and test query count (counts only). Hypothesis: a few hundred training scenes in tens to low hundreds of rooms, tens of queries per scene. Decides: the capacity-ladder rung (below) and the runtime plan. Decision table: < ~100 rooms means stay on the low-capacity rungs; 100 to 400 rooms means a small trained conv decoder with a physical prior; > 400 rooms may justify partial fine-tuning of a pretrained encoder.
- **D1 Gold-oracle decode.** From `candidate_luminance` and scout 0 (luminance from uint8 using `settings.json` weights, float64, NumPy median with its even-count averaging), compute A and the family for every train query and compare with `portfolios`. Hypothesis: >= 99.9% exact (any gap points to a median/precision/tie convention to fix, or to label noise = hard ceiling). Also report: true-family size distribution (|T|, max |T|, share with > 70 masks, expected 0), empty-family share overall and by budget and by number of regions, antichain and mask-size assertions.
- **D1b Trivial baselines on train.** Score of "always []" (equals the empty share), "always the modal family given budget", and "a constant-gain prediction (gain = 1)". These are floors the model must clear by a margin larger than cluster-bootstrap noise.
- **D2 Label/unit consistency.** Distribution of `candidate_luminance` (min/quantiles/max, fraction above 1.0, quantisation to multiples of 1/255) versus scout luminance; confirm both live on the same scale since the gain mixes them. Hypothesis: same display-luminance scale. If candidate luminance is on a different scale or exposure, the gain definition (and the ceiling) changes.
- **D3 Gain and bound structure.** Distribution of log g[k,j] per scene (overall, per candidate rank, per region); share of adequate (k,j) pairs; number of adequate candidates per region (0,1,2,>=3); log lower, log upper, log width (upper/lower), and the margin of every true gain to the nearest bound. Hypothesis: widths and margins are moderate, and a non-trivial share of (k,j) bits sit within the achievable prediction error of a bound (this fraction sets the ceiling).
- **D3b Information-ceiling curve.** Add synthetic noise to the TRUE log gains (shared per (scene,candidate) with sd in {0.03, 0.06, 0.1, 0.2} plus independent per (k,j) with sd in {0.03, 0.06, 0.1, 0.2}), decode argmax-A families, and score them. This maps "log-gain accuracy" to "metric" and tells how accurate the predictor must be. It is a measurement on train labels, not a model. Expect the curve to be steep because one flipped bit can create/remove masks and flip empty vs non-empty.
- **D4 Query-generator structure (the exploitable-bias question).** For each train query, check whether interval centres sit near the true gains of a small subset of candidates (planted portfolio), how often every region contains >= 2 true-adequate candidates (non-empty families) versus a tightened region (empty), and how budget relates to the planted size. Hypothesis (unverified): families are mostly non-empty with planted-like structure and empties arise from one deliberately broken region, so interval centre/width and the number/size of regions carry information about A that a trained query-aware scorer can use. This decides whether stage 2 (query-aware calibration) is worth building and what to ask a reviewer.
- **D5 Probe observations.** Visualise a sample of scout and candidate calibration patches: chrome-like (specular) vs diffuse-grey appearance, brightest-pixel location, mean luminance. Compute for each (scene,candidate): global exposure ratio of the candidate probe to the scout-0 probe, brightness centroid shift versus scout probes. Test Rung 0: predict g = exposure ratio everywhere and report R^2 of log g and bit accuracy of A; and the physics-prior test: express each candidate probe as a non-negative linear combination of the 3 scout probes (in approximately linearised intensities), residual of that fit, and the log-MAE of g predicted by the same coefficients applied to the scout region medians. Hypothesis: the probes encode light direction/intensity well enough that the linear prior is a strong (not perfect) predictor, with larger residuals when the candidate light lies outside the scouts' span. This calibrates how much the learned part must add.
- **D6 Variance decomposition of log g.** Fraction of variance of log g[k,j] explained by (scene,candidate) global exposure vs region-specific spatial terms vs noise from small rectangles. Hypothesis: a large but not dominant global component; the spatial part is what the dense/ROI model adds.
- **D7 Rectangle geometry.** Area (px), aspect, position density (a heatmap over the 160x240 frame pooled over train queries), overlap between regions within a query, share of tiny rectangles (< ~100 px where medians are noisy). Used for: loss weighting by empirical rectangle density, sampling derived rectangles for the region-level model, and the heteroscedastic noise model (sigma vs area).
- **D8 Group structure.** Scenes per group_id; near-duplicate/mirrored scene detection across different group_ids via thumbnails of scout 0 (and scout calibration statistics) compared under identity, h-flip, v-flip, h+v flip; union-find over group_id, scene_id and near-duplicate edges to get own groups. The description says reflections are applied consistently, so mirrored copies of the same room are plausible. Hypothesis: `group_id` is already clean, but a few cross-group mirrored duplicates may exist; the check is cheap and the cost of missing them is optimistic CV.
- **D9 Scout/candidate light exchangeability.** Mean/sd of scouts 0,1,2 luminance and probe statistics; are scouts 1 and 2 statistically interchangeable (swap test on probe stats)? Is the candidate probe distribution inside the scout probe distribution (distribution shift between "scout lights" and "candidate lights" within train)? Decides whether scouts 1/2 can be swap-augmented and whether using scouts as extra training targets is safe.
- **D10 Candidate redundancy within scenes.** Pairwise correlation of the 8 candidates' log-gain fields; near-identical candidates; how often two candidates have identical A columns for the query regions. Informs correlated errors and the expected number of masks.
- **D11 Test schema only.** Test scene count, queries per test scene, regions per query, budget values, test npz tensor shapes equal the declared ones. No feature distribution study.
- **D12 Runtime profiling.** Time one forward/backward of the dense net at the planned resolution and batch, the backbone feature extraction over N scenes, and the enumeration/MBR throughput per 1000 queries (measure, then hard-code counts).

**Expected-but-unverified structure of the data (hypotheses H1..H6).**
- H1: Lighting commands vary direction/intensity/colour, and the probes (spheres) reveal them; scouts show 3 different lights; candidates are 8 further lights.
- H2: log g is dominated by a global per-candidate component plus a spatially varying component driven by surface orientation/distance relative to the light.
- H3: Offset 0.02 matters for dark regions (gain compresses toward 1): model log(L + 0.02), the same transform as the metric's gain.
- H4: Query bounds are placed relative to true gains (planted structure), so margins to bounds are not uniform.
- H5: Many queries per scene share the same 8 fields, so per-scene prediction is amortised.
- H6: The ceiling of exact-mask Dice is limited by the share of bits within prediction error of a bound (D3b).

## Validation design

**How the test split was made (from the description, UNVERIFIED beyond that).** "Training rooms, public evaluation rooms and private evaluation rooms are disjoint": the splits are room-disjoint. Hence validation must be room-grouped (and robust to repeated views of the same room, mirrored copies).

**Scheme.**
- Groups: union-find over `group_id`, `scene_id` (if one group has several scenes, they stay together) and near-duplicate/mirror edges from D8. These are train-only computations.
- Folds: 5-fold GroupKFold over derived groups, assignment balanced by number of queries and empty-family share (greedy assignment on sorted groups), fixed seed. A second split seed (different group-to-fold assignment) is used for the cheap stages (calibration, decode variants, ridge/GBDT region model) to check that conclusions are not an artefact of one split; the dense net is trained once per fold (cost).
- Held-out sanity: one set of rooms (~15% of groups) is set aside during development and never touched by HPO, blend weights, noise-model fitting or decode selection until the final comparison; the final script then uses all rooms in its 5 folds (the holdout exists in dev only).
- Effective sample size is rooms. Reporting: mean ± fold std AND a cluster bootstrap over rooms (resample rooms, average over their queries) for the CI of any comparison. Accept a change only if the paired difference (same folds, same rooms) exceeds the bootstrap standard error and is directionally consistent in most folds and in the second split seed.
- Metric implementation: exact per-case Dice with the empty rules. Unit tests: perfect prediction = 1; both-empty = 1; empty vs non-empty = 0 (both ways); disjoint masks = 0; superset flood (T of size m, P = T plus extras) = 2m/(2m+e); reversed/shuffled masks = 0; plus a base-rate predictor equals the always-[] score. Enumerator unit tests: brute-force Python itertools on random A vs the vectorised enumerator; gold-oracle reproduces train `portfolios`.

**Two-level evaluation (each post-hoc selection has its own held-out check).**
- Level 1 (response model): OOF region log-gain MAE/RMSE on the actual query rectangles, split by candidate rank, region area bucket, and bit accuracy/Brier/log-loss of A, all from the fold model that did not see the room.
- Level 2 (noise model, decode, optional calibrator): fit on OOF rows of the other folds and apply to the held-out fold's OOF rows (cross-fitted), so the reported family Dice is not optimistic about the few decode constants.
- Ablation table (paired on the same folds): argmax-A hard decode vs expected-Dice decode vs expected-Dice + query-aware calibrator.

**Bias direction of each proxy.**
- Group-disjoint OOF on train rooms mirrors the test split axis (rooms disjoint): neutral by construction.
- Fold models see 80% of the rooms while shipped ensemble members (fold models, same 80%) are no larger: no size mismatch; a 100% refit would be slightly better than OOF suggests (pessimistic by a small amount, unquantified).
- OOF is optimistic if mirrored/duplicate rooms leak across folds (guarded by D8) and if the noise model/decode constants are fit on the same OOF rows they are scored on (guarded by cross-fitting). Rough expected sign of the net bias: slightly optimistic (a few points of Dice), plan for the real private score to land at or below the OOF number.
- Public vs private LB are small disjoint room sets: expect large public/private noise; trust OOF/cluster bootstrap, not public LB.

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Few independent rooms (effective n small): a deep net memorises rooms. Mitigation: capacity ladder (below) with a physical prior as residual base, small trainable decoder (~1 to 3 M params), dropout on frozen feature channels, flip and scout-swap augmentation, fixed epoch count and EMA (no validation-triggered stopping), weight decay.
2. Repeated queries per scene inflate row count but not information: weight loss per scene/room, use rectangles derived from maps with scene-balanced sampling, not per-query counts.
3. Selection on OOF: noise model, decode choice and any calibrator are low-dimensional (a handful of constants), cross-fitted, ablated paired.
4. Query-bound features (stage 2) can exploit the generator and silently overfit its quirks: ship only if it beats the query-agnostic decode by more than the bootstrap error across rooms and in the second split seed; reviewer question logged.
5. Hard constants tuned by watching public submissions: forbidden; all decode constants come from in-script OOF fitting.
6. Mirrored/duplicate rooms across folds: D8 union-find with flip-aware edges.
7. Over-confident calibrated probabilities because OOF models are single while shipped is an average: direction is conservative (fine), stated.

**Underfit risks and mitigations.**
1. Under-resolved rectangles: the median over small rectangles needs fine resolution; run the decoder at 80x120 and upsample bilinearly to 160x240 before the median; if tiny rectangles dominate (D7), predict at full 160x240 in the last stage.
2. Backbone that is illumination-invariant (semantic features discard intensity): do not rely on frozen semantic features alone; intensity information comes from raw log-luminance and the physical prior; semantic features supply material/surface cues only.
3. Loss mismatched to the metric: the metric is a threshold on a ratio of medians; train in log(L+0.02) with L1/Huber (median-like) and add region-level (rectangle-median) loss on rectangles drawn like the query rectangles, then calibrate spread by OOF.
4. Discarding probe information: feed the full 32x32x3 probes of the candidate and all scouts (not just means) through a small trained encoder.
5. Too little training: 20 epochs on the dense model is a hypothesis; profile and check the train/OOF loss gap each epoch (log only) before freezing the epoch count as a constant.
6. Variance of the uncertainty model: use heteroscedastic sigma with few parameters (area, prior-vs-net disagreement, flip-TTA disagreement), fit by Gaussian NLL on OOF; if the OOF log-likelihood gain over constant sigma is below noise, ship constant sigma.

**Capacity ladder (applied conditional on D0; stop at the lowest rung that wins under room-grouped CV).**
- Rung 0: global exposure ratio from probe means (floor; also the strip-the-ML stand-in).
- Rung 1: region-level regressor on engineered features (log scout region medians, physical-prior region gain, probe descriptors, pooled frozen backbone features of the rectangle), ridge/GBDT with per-output regularisation grid selected in-script by grouped CV.
- Rung 2: small trained conv decoder conditioned on probes with the physical prior as residual base (primary).
- Rung 3: partial fine-tune of the pretrained encoder (LP-FT: warm start from rung 2 head, tiny LR, 1 to 2 epochs). Only if D0 gives >= ~400 rooms and rung 3 beats rung 2 by more than the paired bootstrap error; log parameter-change norm and CV gain over the probe.

## Recommended approach (primary + fallback)

### Primary: probe-conditioned log-luminance field predictor + calibrated expected-Dice family decode

Stage A. Per-scene precomputation (functions of that scene's own tensors only; same code for train and test):
- Scout luminance maps from uint8 using `settings.json` weights; log(lum + 0.02) at 80x120 for the 3 scouts, plus chroma (log R/G, B/G or similar) so colour-dependent lights can be modelled.
- Probe descriptors for each light: scouts v=0..2 and candidate k=0..7 (raw 32x32x3 patches kept for a small trained encoder; also mean luminance, brightness centroid, and low-dimensional statistics as extra features).
- Physical prior ("sphere-guided linear relighting"): in approximately linearised intensity (gamma-expand the display values with a fixed constant), solve per candidate the non-negative least-squares coefficients c_kv expressing the candidate probe as a combination of the 3 scout probes (3 unknowns, closed form/small iterative solve, vectorised). The prior image for candidate k is sum_v c_kv * lin(scout_v) mapped back to log(lum + 0.02). It is an input feature channel (and base of the residual), not the answer. Rationale: light transport is linear in the incident illumination, and the probes measure the incident illumination for every scout and candidate.
- Frozen pretrained backbone features of the scouts (patch tokens from a pinned DINOv2-small/base or a timm ConvNeXt at a fixed input size such as 224x336), reduced by a learned 1x1 projection to ~32 channels with dropout, for material/surface cues. Features cached in memory per scene inside the script.

Stage B. Trainable decoder (target model): a small U-Net-style conv decoder (3 levels, 32/64/128 channels) with global FiLM conditioning from a probe encoder (a small CNN over the concatenated candidate probe and the three scout probes, producing a conditioning vector; scouts 1 and 2 pooled symmetrically; candidates have no index embedding). Input channels: 3 scout log-luminance maps, chroma, physics-prior map, projected frozen features. Output: residual added to the physics-prior log(L+0.02) map at 80x120, bilinearly upsampled to 160x240. One sample = (scene, candidate); batches of 8 candidates x several scenes.
- Trainable parameter count is a few million; the learning signal is millions of supervised pixels in `candidate_luminance` (8 maps per scene).
- Loss: pixel-wise L1/Huber on log(L+0.02) at 80x120 against 2x2-downsampled labels, weighted by the empirical rectangle-position density from train queries (D7) to concentrate capacity where regions occur; plus a region loss: |log median_pred(rect) - log median_true(rect)| on rectangles resampled from the train rectangle distribution (derived patch gains, explicitly allowed). Add the region loss only if it improves OOF region MAE by more than noise (roadmap step 4).
- Augmentation: horizontal flip (and vertical only if D9/diagnostics show symmetric statistics) applied consistently to scouts, probes, labels, rectangles; swap of scouts 1 and 2 if D9 supports exchangeability; random candidate order. Test-time: flip TTA with inverse transform, average in log domain; the disagreement between TTA views is a per-row uncertainty feature.
- Optional extra supervised pairs: for TRAIN scenes only, use each scout v with its own probe as an additional (light, image) training pair with its own luminance as target (identity case). This is real labelled data (a captured image with its probe), not synthetic. Enter only after measuring a gain (roadmap step 5). Sanity unit test: feeding a scout's own probe as the candidate probe must return that scout's image closely on train scenes.

Stage C. Gain and uncertainty:
- For every query region j and candidate k: median of predicted luminance in the rectangle (NumPy-median semantics on the upsampled map), gain mu_kj = (median_pred + 0.02) / (median_scout0_j + 0.02) where the denominator is computed exactly from the input.
- Noise model on log g, fitted on OOF residuals with Gaussian NLL: r_kj = a_(scene,k) + e_kj, a shared across regions of the same candidate (sd tau), e region-level with sd sigma(area, TTA disagreement, prior-vs-net gap). About 6 scalars. Fit cross-fitted over folds. If a heavier tail is warranted (checked with residual QQ on OOF), use Student-t with a single df parameter.
- Marginals p_kj = P(lower_j <= g_kj <= upper_j) from this model, reported with OOF reliability (Brier, log-loss, reliability bins).

Stage D. Exact family decode (expected-Dice optimal over the valid output set):
1. Draw S=256 samples of A (and a separate S'=256 evaluation set) from the noise model with the shared candidate offset.
2. For every sample enumerate the minimal successful portfolios with size <= budget (vectorised over 255 subsets x J regions; monotone-coverage shortcut).
3. Mask marginals q_m = fraction of samples whose family contains mask m. Candidate output families: greedy antichain by descending q_m, evaluating every prefix length (0..70) and the empty family, plus the modal sample families; choose the one with the highest sample-estimated expected Dice (empty-vs-empty = 1, empty-vs-non-empty = 0). Antichain is maintained by skipping masks comparable (subset/superset) to chosen masks; cap at 70 by construction.
4. Compare against hard decode (argmax A at point predictions). Keep the better under the paired cross-fitted OOF comparison; the expectation is that the expected-Dice decode wins mostly by choosing `[]` and by trimming low-q masks.

Stage E (optional, only if the roadmap step shows a gain beyond noise): query-aware calibrator. A GBDT or small logistic model over per-(k,j) features: mu, sigma, margins to log lower/upper, interval width and centre, the rank of candidate k among the 8 predicted gains in region j (within-row relative transform), number of regions, budget, region area, tTA disagreement; outputs p_kj, trained on OOF rows only, cross-fitted by room. This replaces the Gaussian marginals in Stage D (sampling by Gaussian copula with the same shared candidate factor). Uses only the row's own inputs and train-fit parameters. This is the grey component (query-aware); its ablation (with vs without bounds features) is part of the compliance note.

Shipping: average the log fields of the 5 fold models (each trained on 80% of rooms) plus flip TTA, then apply the OOF-fitted noise model and decode. Reason: no extra training run inside the one-hour budget, an ensemble of 5 plus TTA reduces variance, and the noise model was fitted on the same model family (single-model OOF residuals are an upper bound of the ensemble's error, so the decode is slightly conservative: direction stated, magnitude unverified). If profiling shows spare time, a fixed 100% refit as a 6th member is a roadmap option; adopt only on a measured gain on the dev holdout.

### Fallback: region-level regressor (Rung 1) with the same calibration and decode
- Rows: (scene, candidate, rectangle) with rectangles from train queries (the actual ones) plus a fixed number of rectangles resampled from the empirical train rectangle distribution on each training scene, labels from the true maps (derived patch gains).
- Features: log median luminance of each scout in the rectangle (and ratios to scout 0, log), the physical-prior region gain, probe descriptors for candidate and scouts (mean luminance, brightness centroid, linear-combination coefficients c_kv and residual), rectangle geometry (area, position), pooled frozen backbone features of the rectangle (mean of patch tokens in the box, per scout), and within-row relatives (candidate's prior gain minus the mean over its 8 candidates).
- Model: ridge with a wide per-output grid selected by grouped CV, and/or LightGBM with grouped early-stopping-free fixed rounds, objective L1 on log g.
- Reason it is the fallback: cheaper, lower variance when rooms are few (D0), and doubles as the honest frozen/zero-shot yardstick and a diversity member (different assumption: tabular region regressor vs dense spatial decoder). If both reach comparable OOF quality, blend in log-gain space with a single scalar weight fit cross-fitted on OOF (not fixed averaging of raw probabilities).

**Why this fits this data.** The target is a ratio of medians in arbitrary rectangles under a lighting condition described by a probe. A dense, probe-conditioned field predictor lets any rectangle be scored without retraining and gets millions of pixel labels; the physical prior encodes the one mechanism the data guarantee (linear light transport with known probes) and keeps the learned part small; and exact enumeration plus expected-Dice selection handles the all-or-nothing exact-mask metric.

## Rejected options

- **Retrieve or transplant training gains/maps for test scenes (kNN over scenes):** explicitly banned.
- **Direct family classification (predict masks or 238 subset flags end to end from scene + query):** the label space is huge, discards the physics, and fails to generalise across rooms; the family is a deterministic function of A, so learn A, not the family. (It remains legitimate only as a diagnostic.)
- **Dense relighting generator (GAN/diffusion/NeRF-like):** not required by the description, too heavy for one hour, and region medians do not need photorealism.
- **Full fine-tune of a large backbone:** with a hundred-ish rooms it memorises rooms; kept only as rung 3 under an explicit data-size gate.
- **Point-estimate argmax decode as the final decoder:** ablation baseline only; the exact-mask Dice with all-or-nothing empties rewards decoding under uncertainty.
- **Pooling test queries of the same scene (e.g. joint inference over bounds) or any test-time adaptation:** whole-test aggregation, banned by CLAUDE.md.
- **Per-channel medians, summing candidate maps, using a past scout as a portfolio member:** violate the stated query semantics.
- **Hard-coded gain thresholds or offline-tuned decode constants:** forbidden; constants are fitted in-script on OOF.
- **Large HPO:** CV noise with few rooms would select noise; only a tiny in-script grouped-CV grid for the ridge fallback and per-output regularisation.
- **Using scene_id/id/scene_path/row order/file size as features:** explicitly banned.

## Fixed work plan & runtime budget

Constraint: one hour total including setup, training and complete test inference. Target <= ~38 min on an A10G (>= ~35% headroom). All counts are hard-coded constants after profiling (D12); no time-based branching anywhere; time appears only in log lines. Estimates below are UNVERIFIED planning numbers under assumed N ~ 400 train scenes, ~150 test scenes, ~10k train queries, ~3k test queries.

| Stage | Fixed plan | Est. A10G time |
|---|---|---|
| Imports, pinned backbone download, seeds | one small/base backbone, revision pinned | 1.5 to 3 min (network-dependent, silent in description) |
| Load npz to GPU tensors, luminance maps, groups, D8 near-dup union-find | all in memory (~1.6 MB/scene) | ~1 min |
| Frozen backbone features (train+test scouts) | batch 32, fixed size 224x336, fp16 | ~1 min |
| Physics prior (NNLS per candidate), probe features | vectorised, closed-form/small fixed iterations | < 1 min |
| Dense decoder, 5 group folds | 20 epochs/fold, AdamW, one-cycle or cosine with warmup, EMA, bf16 autocast, batch of scene-candidate samples fixed, flip + swap aug | ~3 min/fold = ~15 min |
| OOF region medians, noise-model fit, calibrator (if kept) | Gaussian NLL, ~6 parameters; optional GBDT fixed rounds | ~2 min |
| OOF expected-Dice decode for diagnostics (train queries) | S=256+256 samples, chunked on GPU | ~2 min |
| Test inference (5 fold models x flip TTA), decode, validate, write | fixed | ~3 min |
| Fallback ridge/GBDT region model (if used as blend member) | fixed grid, fixed rounds | ~3 min |
| Validation report print/write under working/ | prints + `validation_report.txt` | < 1 min |
| **Total** | | **~29 to 33 min, headroom ~45%** |

Memory: scenes as GPU tensors (~1.6 MB each; 600 scenes ~1 GB); decoder activations at 80x120 with batch 64 are small (< 6 GB); decode tensors chunked by queries so (Q x S x 255 x J) bool/int8 stays under ~2 GB per chunk. No DataLoader (`num_workers` not applicable); deterministic generators for any shuffling; fixed threads.

Determinism: `PYTHONHASHSEED`, `random`, `numpy`, `torch` + CUDA seeds, `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before the torch import, seeded `Generator` for the decode samples (fixed seed per query derived from a constant plus the position in the sorted test frame, not from ids), fixed fold assignment seed, TPE/Optuna not used. Pinned pretrained revision. No `torch.cuda.is_available()` switches, no `try/except` import fallbacks, `device="cuda"` fixed.

Input/output validation inside the script: assert tensor shapes against `settings.json`; assert gold-oracle reproduces training targets on a fixed small train subset at start (stage check that fails loudly if the enumerator/metric convention is wrong); `validate_submission` from CLAUDE.md plus portfolio-specific checks (JSON parse back, ints only, each mask in 3..255 with 2..6 bits, <= budget bits, unique, antichain, <= 70, non-empty field) run on the DataFrame before writing and again on the reloaded CSV with `keep_default_na=False`.

## Metric-aware training & decode

(i) **Back-solve the metric.** gain = (median_cand + 0.02)/(median_scout0 + 0.02): invertible given the exact scout-0 median, so the model targets log(median_cand + 0.02) and the adequacy test becomes a comparison on log scale (log lower <= log(median_cand+0.02) - log(median_scout0+0.02) <= log upper). Training labels are log(L+0.02) pixel maps and region log-medians derived from the released `candidate_luminance`.
(ii) **Weighting follows the metric's averaging.** The metric averages per query (queries equal weight). Region loss uses rectangles resampled to match the train query rectangle distribution; per-scene weights avoid letting scenes with many queries dominate the dense loss (a scene's weight is bounded), and CV cluster bootstrap is at room level.
(iii) **Not ordinal;** no cumulative heads.
(iv) **Composite metric -> expected utility.** The output is chosen by expected Dice over the valid output set under the model's posterior over A (Stage D), including the empty family. Argmax of A is not Dice-optimal because exact masks and the empty rule make the loss highly nonlinear; constants (sigma model, tau) are fitted on OOF and cross-fitted. The optimal-F structure is used: masks ranked by q_m and thresholded at about half the optimal expected Dice (found by evaluating all prefixes on the sample set; no grid on test).
(v) **Closed-form counts.** For a case with expected family size E|T| and chosen |P| the expected Dice is approximately 2 sum_{m in P} q_m / (E|T| + |P|); the prefix search above is the exact sample-based version, bounded by 70 masks, antichain, and the budget, so it cannot reach degenerate corners.
(vi) **Hard constraints inside the valid output space.** Budget, antichain, <= 70 masks, mask size 2..6 enforced at decode, combined with model evidence (no constraint search without the predictor).
(vii) **Correlated errors.** Shared (scene, candidate) offset in the noise model; samples of A are jointly drawn so a candidate that is uniformly too bright flips many regions together (the primary driver of family-level errors). An overdispersed marginal (Student-t) is used only if the OOF tail check demands it.
(viii) **Probabilistic outputs need calibration.** Check OOF reliability of p_kj; fit at most a tiny per-slot (candidate-rank, area-bucket) recalibration of sigma, cross-fitted, report with and without.

## Structural signals

- Coverage is monotone and minimality reduces to single-member removal; enumeration over 255 subsets is exact and cheap.
- Mask size is automatically >= 2 and bounded by the budget; the antichain property holds automatically within a sample's family; the 70-mask cap needs a guard only on sampled/greedy unions.
- Candidates have no identity across scenes: no candidate-index embeddings; each is represented by its own probe; random candidate permutation as augmentation; decode is permutation-equivariant.
- Scout 0 is the reference for all gains and is special; scouts 1 and 2 are probably exchangeable (D9): symmetric pooling + swap augmentation if supported.
- Reflections are applied consistently to images, labels, rectangles and probes: horizontal flip augmentation and flip TTA with inverse mapping of the output field; assert on every train scene that a flipped sample's flipped prediction maps back to the same rectangle median under the augmentation implementation (unit test). Vertical flip is used only after a diagnostic shows no OOF loss.
- Light transport is linear in incident light: probes yield linear-combination coefficients (physics prior), and the residual net learns what the three scouts cannot span.
- The offset 0.02 compresses gains in dark regions: model log(L+0.02), and expect larger relative uncertainty in bright/small rectangles by area, not by darkness.
- Scouts are real captures of known lights for train scenes: their own probe/image pairs are identity training pairs (optional extra supervision; train scenes only; never test scouts).
- Per-scene sharing: all queries of a scene use the same 8 predicted fields; compute once per scene (a function of the scene's tensors only).
- Rectangle median uses NumPy semantics (mean of the two middle values for even counts); decode uses the exact median of the predicted luminance, not a mean, so the model's quantity matches the metric's.
- Query bounds (stage E only): "at least two adequate candidates per region" plus planted-like structure (D4) means the bounds carry information about A; use only inside the row, ablated, and logged.

## Experiment roadmap

1. **Contract, metric, validation (stop when all unit tests pass and gold-oracle = 100% or the residual mismatch is explained and bounded).** Implement exact gain/adequacy/enumeration/metric; run D0, D1, D1b, D2; build groups (D8); write fold assignment. Decision: sets expected ceiling and the floor.
2. **Cheapest end-to-end baseline (Rung 0 + decode).** Global exposure-ratio gains, constant sigma, argmax and expected-Dice decode; valid CSV; record OOF Dice per fold, mean ± std, cluster bootstrap CI. Also the physical-prior-only variant (D5) as the strip-the-ML yardstick. Stop: valid file, floors recorded. Spend credit 1 here only after a clean local end-to-end run.
3. **Region-level regressor (Rung 1 fallback) with calibrated expected-Dice decode.** Per-output ridge grid (wide, strong end included) and optional GBDT; repeated split seeds; compare with step 2 paired. Stop when no further gain beyond bootstrap SE. This is the honest frozen/zero-shot floor the dense decoder must beat.
4. **Dense probe-conditioned decoder (primary) on the same folds.** Start with pixel loss only; log epoch-wise train loss vs held-out OOF region MAE (memorisation check), freeze the epoch count and EMA decay as constants; then test (each paired): + region loss; + flip/scout-swap augmentation; + frozen semantic features; + physics prior on/off (to prove the net adds beyond the prior); + scout identity pairs. Accept a change only if it beats the paired bootstrap SE. Capacity-ladder gate: if rooms < ~100, do not go beyond rung 1/2; rung 3 only above ~400 rooms.
5. **Uncertainty and decode.** Fit the noise model (shared + region-level, sigma(area, TTA disagreement, prior-net gap)); evaluate by Brier/log-loss/reliability and by family Dice; compare constant vs heteroscedastic sigma; compare hard vs expected-Dice decode; cross-fit everything. Stop when additional noise parameters do not improve cross-fitted OOF NLL beyond noise.
6. **Optional query-aware calibrator (Stage E).** Paired ablation without bounds features vs with; keep only if the gain is consistent across folds and across the second split seed. If kept, spend one reviewer question on it before the final submission.
7. **Diversity.** Blend dense decoder + region regressor in log-gain space with a scalar weight fit cross-fitted on OOF, only if both are close in quality and the blend gain exceeds the bootstrap SE. Otherwise ship the single better one.
8. **Freeze the fixed work plan and profile (D12).** Fix epochs/folds/samples; verify runtime estimate; full run from a clean `working/`; run twice and diff the CSV (identical or near-identical); distribution sanity of output (share of empty families, mean |P|, vs OOF values).
9. **Submissions.** Credit 1: baseline (step 2/3). Credit 2: best single (step 4/5). Credit 3: blended/calibrated (step 6/7). Final: frozen plan. The free CSV check is used to validate format and compare candidates. Do not tune on public LB.

## Compliance audit

CLAUDE.md section 7 and strategist section B self-audit against this plan (all UNVERIFIED until the solution is written and run):

- Test file read only for producing per-scene, per-query predictions: yes. Per-scene tensors from test used only to predict that scene; no statistic fit on test; no pooling across test queries or scenes (noise model, sigma, decode constants all from train OOF).
- No whole-test aggregation: yes. The decode of one query depends on its own scene tensors, its own query and train-fit parameters.
- Wall-clock: time only inside log prints. No `elapsed()` in conditions, loops, arguments, timeouts.
- Environment fallbacks: none. `device="cuda"` fixed; no `cpu_count`; no import fallbacks; pinned backbone revision; seeds fixed.
- Hard-coded constants: model hyperparameters are generic defaults (AdamW, one-cycle, EMA) and the epoch count/resolution are structural plan constants set by profiling; noise-model parameters, decode choices (prefix length, q-threshold), ridge regularisation and any blend weight are fitted in-script on train OOF. No offline-found numbers pasted from earlier submissions.
- External data / synthetic data / self-hosted weights: none. Only the released data and pinned HF/timm pretrained weights. Rectangles cut from released real maps are derived labels, not synthetic scenes. No retrieval of training maps for test scenes.
- Strip-the-ML: without the trained predictor the system reduces to the exposure/prior floor; the learned decoder is load-bearing and measured vs the prior alone. The enumerator and decode are the metric's definition. If the prior alone is within noise of the trained model, the plan flags it (compliance risk) and shifts weight to the learned parts.
- Related-row ("sibling") leakage: the physical-prior coefficients are computed from the scene's own probes and scouts, not from its own targets; groups defined by union-find with flip-aware near-duplicate edges; query-bound features (if used) are within-row only.
- Label-derived statistics: noise model and calibrator trained on OOF residuals from fold models that never saw the room; nested/cross-fitted.
- Engineered-negatives/generator exploitation: Stage E uses query bounds; logged as the single exploit-the-generator risk with an ablation.
- Banned identification/retrieval: no attempt to identify the capture archive, no external lookups, no filename/id/path/size features, no byte-pattern use.
- Source: plain readable UTF-8 `solution.py`, < 512 KB, no blobs, no `exec`/`eval`, comments on reasoning at each stage, validator asserting submission format before writing.
- Output format: portfolios array of ints; no duplicates; antichain; masks 2..6 bits; <= budget bits; <= 70 masks; every test id; exact `id,portfolios` columns; empty family written as `[]`, reload test with `keep_default_na=False`.
- Runtime: ~29 to 33 min estimated (< 60 min limit, ~45% headroom); to be re-profiled (D12).

## Open questions & assumptions

**Reviewer questions (one line each, with the plan under each reading).**
1. Is using the query's bounds/width/centre as features of a trained adequacy calibrator acceptable ("exploits the generator")? If not: ship query-agnostic Stage D only (the primary already works without it).
2. Is a physics-motivated prior (linear combination of scouts from probe observations) fed as an input channel to a trained decoder acceptable, or does it look rule-based? Plan: keep it as input/residual base, report the strip-the-ML ablation, keep the learned residual load-bearing.
3. Is a frozen pretrained backbone plus a trained decoder acceptable here (description permits general pretrained visual weights and any architecture)? Yes as read; plan also works with the backbone removed (rung 2 without the semantic channel) if the reviewer objects.
4. Are train scouts as extra labelled (light, image) training pairs for train scenes acceptable? If not, drop (optional step).
5. Is per-scene caching of predicted fields across a scene's test queries acceptable (it is a function of that scene only)? If not, recompute per query (runtime cost is small).
6. May the pretrained backbone be downloaded from HF at grading time (network and time counted in the hour)? If not, drop the semantic channel; the primary does not depend on it by design.

**Assumptions where the description is silent.** CPU/RAM small; GPU is A10G-class 24 GB; HF download reachable and counted in the hour; N scenes/rooms/queries as in the runtime table; per-case independence of queries across test scenes (no pooling); reflections used as augmentation are valid for the lighting statistics (to be tested in step 4, vertical flip especially); mirrored near-duplicates may exist across `group_id`s (checked by D8).

**What I could not verify.** Everything numerical: dataset size, rooms, empty-family share, query/bound structure, probe appearance, gold-oracle exactness, achievable log-gain error, runtime. No dataset or model run was available.

**Estimate (not a promise, UNVERIFIED).** Exact-mask Dice is a hard metric: the floor is the empty-family share (always `[]`); I would expect the exposure-only baseline to be near or slightly above that floor, the region-level regressor with calibrated decode to add a moderate amount, and the dense probe-conditioned decoder with expected-Dice decode to land roughly in a 0.30 to 0.60 band on OOF, with the private score at or a few points below OOF (rooms disjoint, small shards, higher variance). The ceiling curve D3b should replace this guess with a concrete required log-gain accuracy as soon as the data are available.
