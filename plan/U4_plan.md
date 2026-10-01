# U4 plan: surface scale and sun geometry from rover views (Eris Strategist)

Status of evidence. No dataset files were available to this run and only three files were read: the agent file, CLAUDE.md and the challenge text. Everything under "Data findings" that is not arithmetic on the challenge text is an UNVERIFIED hypothesis plus the exact diagnostic that would test it. No solution code is written here. Numeric runtime and score figures are estimates, not measurements.

## Contract & decision unit

**One valid answer.** A CSV with header exactly `row_id, standoff_m, footprint_m, sun_elev_deg, sun_bearing_deg`, exactly 6,468 rows, one per `row_id` in test.csv, no duplicates, no missing/extra/renamed column, every value finite. The row_id values come from `sample_submission.csv`, whose values are random placeholders and must never be copied. Out-of-band values are clipped by the scorer (standoff to [2.20, 30.0], footprint to [0.03, 10.0], elevation to [3, 89], bearing to [0, 180]), so they are not invalid. Invalid means: wrong columns, wrong or duplicate ids, wrong row count, NaN/inf. Everything else is merely low-scoring. A malformed file scores below the 0.05 floor and still burns a credit.

**Decision unit.** One image (192x192 RGB) maps to four numbers. There are no cross-row output constraints. The independent unit for validation is `capture_group` (a rover working location). Test groups are all absent from train. Below it sits a probable "frame" unit (see Data findings, hypothesis H3): one source frame yields several crops that share Sun elevation and bearing.

**Metric, term by term (View Reconstruction Skill).** Per row:
- `raw = S_view^0.25 * S_axes^0.75`, with `S_axes = mean(S_foot, S_stand, S_elev, S_bear)`.
- Final score is `clip((mean(raw) - 0.4934220017)/(1 - 0.4934220017), 0.05, 1)`. The mean is over rows, equal weight per row, so large groups weigh more.
- Each axis term is a Laplace kernel of the error in a transformed space:
  - `S_foot = exp(-|ln w_pred - ln w_true| / 0.88505864)`
  - `S_stand = exp(-|ln r_pred - ln r_true| / 0.63674155)`
  - `S_elev = exp(-|e_pred - e_true|_rad / 0.25593667)`
  - `S_bear = exp(-|b_pred - b_true|_rad / 1.00008771)`
- Sensitivity at zero error (loss of term per unit error): footprint 1.13 per ln-unit, standoff 1.57 per ln-unit, elevation 3.9 per radian (0.068 per degree), bearing 1.0 per radian (0.0175 per degree). A degree of elevation error costs about 4x a degree of bearing error, but bearing errors will be larger.
- `S_view` is a joint, geometry-based term. With H = 1.995 m the camera pitch is fixed by standoff (depression angle = asin(H/r), roughly 62.4 degrees at r = 2.25 down to 5.6 degrees at r = 20.4) and focal length is `f = 192 r / w` pixels. Three world marks (a vertical mark at the view centre, a ground mark toward the Sun at azimuth b, a ground mark `0.5 w` beyond the centre) are projected with predicted and true parameters. `disp` is the mean pixel displacement over the three marks, divided by 192 and capped at 3. `S_view = exp(-disp/0.25529558)`. Sun elevation does not enter `S_view`; only r, w and b do.
- The blend is geometric: `S_view` has a 0.25 exponent on a log scale, so a row where the marks fly far off loses a lot even if the axes are fine. Worst case `disp = 3` gives `S_view` of about 8e-6, but realistic displacements (marks stay within about 100 px of the centre) bound the loss.
- Value of each target if made perfect from the best constant (from the text): elevation 0.1706, footprint 0.1548, standoff 0.1054, bearing 0.1035; both solar 0.3727; both length 0.4051; all four 1.0. The singles sum to 0.534, so gains compound through the geometric blend and `S_view`. Elevation and footprint matter most.

**Pipeline stages (no candidate set exists).** (1) representation: fine-tuned backbone; (2) per-target posterior (distributional heads); (3) decode: choose each output by expected utility under the posterior. Plan to diagnose separately: posterior calibration per head (NLL, reliability per bin); decode quality given the posterior (mean vs median vs mode vs expected-utility, on OOF); quantisation loss of the bins (feed gold labels snapped to bin centres through the metric; must cost < 0.002).

## Compliance regime

**Domain.** Computer vision, labelled Fine-Tuning, Hard, A10G. CLAUDE.md section 6.3 and 6.7 apply with the overrides below. Fine-tuning a public backbone (HF/timm, pinned revision) is explicitly permitted and "worth roughly a factor of four"; training from scratch is also permitted.

**Explicit bans in the description, each treated as a hard constraint:**
- (a) Test-time augmentation of any kind. This means: no flip, rotation, D4, multi-crop or multi-scale averaging at inference; no flip-averaged pooled features; no per-sample re-encoding with several views. Inference is exactly one forward pass of each trained model on the unmodified image. Averaging the outputs of several independently trained models (the 4 fold models) is an "ensemble of models you trained" and is explicitly allowed. Doubling the forward pass inside the training loss (an invariance/consistency regulariser between two D4 views) is training-time only and is allowed; it is not used in the primary.
- (b) No pseudo-labelling, self-training or other transductive use of test.
- (c) No lookup tables or values fitted to test labels.
- (d) Nothing fitted, calibrated or normalised on test statistics, and no touching test beyond the final prediction pass. This includes per-`capture_group` aggregation or smoothing of test predictions (tempting because sun values are probably shared within a frame/group; banned), group-level normalisation, rank/z-score across the test file, and any train-vs-test similarity or shift check. From test I use only: row ids, row count, and the image files for the final forward pass.
- (e) No shipped pre-trained-for-this-task weights; all fine-tuning happens in the script.
- (f) No external data. Only the challenge images plus backbone weights from HF/timm.
- (g) No development decision based on test.
- (h) No reading of text, markers or identifiers visible in images, no OCR.
- (i) No matching a released view against anything outside the challenge. Train-vs-train similarity (to build validation groups) is inside the challenge and is allowed; train-vs-test matching is never done.
- (j) No hand-coded heuristics as the solution: no fixed thresholds, edge/texture/frequency rules, formulas with hand-chosen constants, template matching or nearest-neighbour lookup on raw pixels. The plan uses no hand features. The decode uses only the metric's own published constants and a learned posterior.
- (k) No incidental artefacts: no file size, JPEG quantisation tables, encoder fingerprints, file order or row_id order as features, splits or sampling order. Images are decoded to pixels only; the training order is a seeded shuffle that never looks at file or row order.

**Augmentation clause.** The description allows training augmentation "provided every augmentation is one the targets are genuinely invariant to". The only augmentation guaranteed invariant is the in-plane rotation and mirror (the text states all four targets are unchanged by it). Primary uses the exact D4 group (90-degree rotations and flips: lossless, no interpolation, no border fill), applied on GPU with a seeded generator. Rejected because targets are not invariant to them: zoom/random-resized-crop and progressive resizing (change footprint and pixel-scale cues), brightness/contrast/colour jitter, blur, noise, JPEG re-compression (illumination and texture cues carry sun elevation and footprint), mixup, cutmix, RandAugment. Arbitrary-angle rotation is not used in the primary (interpolation blur and corner fill are not in the test distribution; measure only if the corner diagnostic says the released views have no fill).

**Conflicts between the description and CLAUDE.md / the guidebook / agent file (description wins per CLAUDE.md 2.4):**
1. TTA: CLAUDE.md 2.2 and 6.3 list TTA as clearly allowed; the description bans it of any kind. Follow the description. The agent file's F2 items on flip-averaged pooling and "matching test-time averaging" do not apply.
2. Augmentation recipe: CLAUDE.md 6.3 recommends RandAugment/Mixup/CutMix/progressive resizing/label smoothing; the description restricts to target-invariant augmentations. Follow the description (D4 only).
3. Capacity ladder (agent file F2.1: frozen features first and stop at the lowest rung that wins) versus CLAUDE.md 6.7 (Fine-Tuning label: frozen embedding plus linear head or GBDT is grey/likely rejected) and the description (fine-tuning worth about 4x). Resolution: climb the ladder only as a development diagnostic (frozen probe = honest floor); ship a genuine end-to-end fine-tune. Log the parameter-change norm to show training is load-bearing.
4. Checkpoint selection: CLAUDE.md section 10 says best-checkpoint per fold by validation metric; the agent file prefers fixed schedules. Use fixed epochs plus weight EMA, no validation-triggered checkpoint choice (also keeps OOF honest).
5. Refit on 100% of data (agent file) versus compute: ship the mean of the 4 fold models instead of an extra full-data model; state it as a deliberate trade.
6. Runtime: the description does not state a limit ("assume platform default"). CLAUDE.md gives 50 min target, 1.5 h ceiling, one section says 1 h. Plan <= 42 min worst case so that a 1 h cap still leaves headroom.
7. CLAUDE.md 2.1 asks that tuned hyperparameters be searched inside the script. The plan uses standard fine-tuning defaults and, in the script, only searches a tiny train-only set (posterior temperature scalars). Dev-time comparisons of at most 2-3 recipe values are on train-only grouped CV and are listed as a reviewer question.

**Strip-the-ML test.** Remove the network and nothing predictive remains: the decode needs a posterior, bins come from train label ranges, and there are no rules. Passes.

## Data findings

Legend: [TEXT] = stated or arithmetic from the challenge text. [UNVERIFIED] = hypothesis; the diagnostic to run on train is given. All diagnostics are train-only.

**From the text [TEXT]:**
- 15,114 train and 6,468 test rows (test is 30% of 21,582); 192x192 RGB JPEG, metadata stripped; test locations absent from train.
- Target ranges: standoff 2.25-20.4 m (ln range 0.81-3.02, factor 9), footprint 0.06-3.4 m (ln range -2.81-1.22, factor 57), elevation 5.5-88 deg, bearing 0-180 deg.
- Ground sampling per pixel = w/192, i.e. 0.31 to 17.7 mm per pixel: footprint is the physical pixel scale. Standoff is the camera pitch: depression asin(1.995/r).
- Group-mean oracle scores "the floor"; sample_submission values are random placeholders.
- Arithmetic hints on target distributions (derived from "best constant scores exactly 0.50 per term"):
  - bearing: for uniform [0,180] with the constant at 90 deg, mean `exp(-|x|)` is (1 - e^(-pi/2))/(pi/2) = 0.504. This matches the 0.50 constant, so the bearing is close to uniform on [0, 180] (hypothesis H1).
  - elevation: the same calculation for uniform [5.5, 88] gives about 0.33, below 0.50, so elevation is more concentrated than uniform (H2).
  - footprint: uniform in ln w would give about 0.39 at centre, below 0.50, so ln w is more concentrated than uniform (H2b).

**Diagnostics to run on train, with expected outcomes [UNVERIFIED]:**
- D1 Schema and integrity: dtypes, nulls, unique row_id and file existence for all 15,114 images, image mode/size all 192x192x3 RGB, targets finite and inside the stated ranges. Expect clean. Also count `capture_group` values, rows per group (min/median/max) and the group-size distribution (needed for fold design; the number of groups is unknown, could be 100-300; if under about 40 the CV is noisy and 4 folds are coarse).
- D2 Target distributions: quantiles, skew, histograms for ln r, ln w, e, b; correlation matrix of (ln r, ln w, ln(w/r), e, b). Hypotheses: H1 bearing near-uniform; H2 elevation concentrated in a band with tails; ln w broad; ln w and ln r positively correlated; ln(w/r), the angular footprint, may be much narrower than ln w (discrete zoom settings times crop fractions); e and b nearly independent.
- D3 Frame structure: within each group, count rows sharing exactly identical (sun_elev_deg, sun_bearing_deg). H3: identical pairs mark crops of one frame (same Sun, same facing), so effective sample size is the number of frames, not 15k, and r and w vary within a frame (crop position and magnification). Also report ICC (between-group variance fraction) per target; expect low-to-moderate, consistent with the text that group means score the floor. Report the group-mean-oracle score computed with the exact metric (expect near 0.05).
- D4 Image statistics: per-channel mean/std; corner and border pixel statistics to detect any rotation fill or circular mask (decides whether arbitrary-angle rotation is ever usable); exact duplicate images within train (hash) and near-duplicates via downsampled correlation within train only; duplicate rate across different capture_groups (hypothesis: near zero; if not zero, merge groups into super-groups for validation).
- D5 Order leak audit: does train.csv row order or row_id correlate with capture_group or targets? Expected: possible ordering by group. Never use it; the shuffle is seeded.
- D6 Cue existence (diagnostic only, never features): correlation of image-global luminance with elevation (expect positive), of high-frequency energy / Laplacian variance with ln w (expect negative as pixels get coarser), of orientation-anisotropy of the gradient structure tensor with ln r (expect larger anisotropy at large r because of foreshortening). Purpose: confirm each target is readable and which receptive field is needed. Writing these as rules is banned by (j).
- D7 Frozen-probe floor under the same group folds: pooled features of one pretrained backbone (no flip averaging) plus a heavily regularised linear head, scored with the exact metric. Expected chance-corrected score about 0.05-0.15 (the text's "factor of four" for fine-tuning suggests the frozen floor is a quarter of the fine-tune). This floor is what the fine-tune must beat; the probe is never shipped.
- D8 Metric unit tests from train labels alone (these also pin down interpretation of `S_view`): perfect = 1.0; reversed/constant/base-rate behave as expected; the best constant search reproduces REF = 0.4934220017 within 1e-3 and each term's constant gives mean 0.50 over train; the seven "value of each target" numbers (0.1054, 0.1548, 0.1706, 0.1035, 0.3727, 0.4051, 1.0) are reproduced when replacing one or more targets with truth over the best constants. If only one reading of the `S_view` mark definition (marks built from predicted vs true construction) reproduces these, that reading is the one.
- D9 Bin quantisation oracle: snap true labels to bin centres and score; expect loss < 0.002.
- D10 Information ceiling probes (expected, to check): the 180-degree ambiguity of the camera facing in a narrow-field crop may make bearing bimodal (b versus 180 - b); check later from OOF posterior entropy per bearing bin.

## Validation design

**How the test split was probably made.** Whole locations (capture_group) were held out: all 6,468 test rows come from groups not in train, about 30% of rows. No time or terrain stratification is stated. Reproduce this with group-held-out folds.

**Folds.** 4 folds, `StratifiedGroupKFold`-style (groups stay whole; stratify on binned elevation and binned ln r so each fold has similar target ranges), seeded. Groups are `capture_group`, merged by union-find with any train-only near-duplicate or shared-exact-image links found in D4 (super-groups). Frames (identical sun values) always stay inside one group. Repeat the split with a second seed only for cheap dev comparisons; the shipped run uses a single seed because each fold costs a full training.

**Reported numbers.** Exact metric (per-fold and mean +- std), plus per-term S values, plus a group-weighted version as a secondary view. The metric re-implementation is unit-tested per D8 before any model is trained.

**Selection steps need their own checks.** Posterior temperature (one scalar per head) is fit on OOF with a 2-way group split inside OOF for the reported number (cross-fit). Architecture and recipe choices are made on paired folds and judged against fold-to-fold noise; accept a gain only if consistent across folds and larger than the paired std-error. Never choose with the public LB.

**Direction of bias of this proxy.**
- Pessimistic (about 0.01-0.03 score): each fold model trains on 75% of groups; the shipped ensemble of 4 such models is slightly better than any one.
- Optimistic (about 0.01): few selection knobs (temperature, one or two recipe choices) used on the same OOF.
- Optimistic if train groups have geographic neighbours that straddle folds (D4 merges check this), neutral otherwise since test groups relate to train the same way.
- Could be optimistic in one scenario: if the test holds out terrain types (crater floor vs delta) rather than random locations. The text does not say; treat as unknown.
- Net: expect the private score near the OOF value, more likely slightly below it than above it.

**Metric sanity set.** Perfect, constant, reversed targets, base-rate, group-mean oracle, bin-snapped gold, D4-invariance check (score on a held-out fold with D4 views of the same images, as diagnostic only; this is dev measurement, not inference, and never uses test).

## Overfit/underfit risks

**Overfit**
- Effective sample size is frames or groups, not 15,114 rows (H3). A 28M-parameter model can memorise terrain-to-sun links. Mitigation: group-held-out validation, D4 augmentation, drop-path 0.1, weight decay 0.05, layer-wise LR decay, weight EMA, a fixed short schedule (about 20 epochs), a fold-model ensemble, no group identity or order features.
- Selection on OOF: kept to temperature scalars and at most 2-3 recipe values; reported cross-fitted.
- Sun cues carried by terrain albedo or sky/ground colour rather than shadows (a group-specific shortcut). Mitigation: group CV exposes it; if the gap to the random-split CV is large, increase regularisation or lower the LR.

**Underfit**
- Resolution: the scale cue lives in fine texture; resizing or down-sampling would destroy it. Mitigation: native 192 (ConvNeXt) or reflect-pad to 196 (patch-14 ViT) with no resampling.
- Backbone too small or too little training. Mitigation: ConvNeXt-Tiny primary; measure Small and a ViT before concluding; 20 epochs minimum, checked by a learning-curve (train vs held-out score per epoch).
- Receptive field: tilt and facing direction are global perspective cues; ConvNeXt-T at 192 has a final 6x6 map with global coverage. A global-attention member (ViT) adds diversity.
- Loss mismatch with metric: the metric is a Laplace kernel in ln/rad space; plain MSE would pull to the mean for multimodal posteriors. Mitigation: distributional heads plus expected-utility decode (Metric-aware section); fallback is direct L1 in ln/rad space, the metric's own distance.
- Information thrown away by augmentation: only D4 is used; no photometric or scale augmentation.

## Recommended approach (primary + fallback)

**Primary: ConvNeXt-Tiny, genuine full fine-tune, distributional heads, expected-utility decode, 4-fold group ensemble.**
- Representation: timm `convnext_tiny.fb_in22k_ft_in1k` (pin the Hub commit at implementation; its LayerNorm design gives no batch-statistic dependence at test). Native 192x192 input, ImageNet mean/std (the backbone's own constants), no resize. Pooled global features (mean pool of the final map plus the classifier-token-equivalent) into a small MLP trunk.
- Targets and heads: five soft-histogram heads on train-derived ranges: ln r (about 32-48 bins), ln w (about 32-48 bins), elevation (about 32 bins over 3-89 deg), bearing (about 36 bins over 0-180 deg), plus an auxiliary head on ln(w/r). Bin widths (about 0.05-0.09 ln-units, 2.7 deg, 5 deg) are far below the metric's error scales (0.64, 0.89 ln-units; 14.7 deg; 57 deg). Loss: HL-Gauss cross-entropy (Gaussian-smoothed bin targets of sigma about 0.75 bin; this is ordinal smoothing, not class label smoothing). Equal head weights. Optional aux: L1 on `H/r` (smooth tilt proxy), kept only if it measures a gain.
- Augmentation: exact D4 only (uniform random element per sample, on GPU, seeded).
- Optimisation: AdamW, standard defaults (backbone LR about 1e-4 with layer-wise decay about 0.8, head LR about 1e-3, weight decay 0.05), batch 64, warmup 1 epoch then cosine over about 20 epochs, bf16 autocast, channels-last, EMA of weights (fixed decay) used for inference, drop-path 0.1. These are starting defaults, to be confirmed by dev-time comparisons.
- Folds: 4 group folds. Each fold model gives (i) OOF posteriors and (ii) test posteriors from one plain forward pass on each unmodified test image. Test posterior = mean of the 4 fold models' temperature-scaled probabilities.
- Decode: per head, pick the value maximising expected metric term under the posterior (Metric-aware section). Clip to the bands. Write and re-validate.
- Compliance status: clean (fine-tune of public backbone, no TTA, no test fitting, no hand rules). Reviewer questions listed below.

**Fallback (lowest risk): same backbone, folds and augmentation, four direct regression outputs trained with L1 on (ln r, ln w, e in rad, b in rad)**, i.e. the metric's own per-axis distance (median regression), no bins, no decode step. Choose this if the distributional head does not beat it under paired fold comparison or if the decode shows no OOF gain over the posterior median.

**Second family for diversity (roadmap, gated on a measured gain over noise): DINOv2-S/14 (or ViT-B/16 at 192) fine-tuned with the same heads.** Different assumption (global attention and self-supervised geometry-aware features vs local convolutional texture). Reflect-pad 192 to 196 for patch 14 so that pixel scale is untouched. Blend only if each family alone is within about 0.02 score of the other and the blend beats the best single by more than the paired fold noise; blend by averaging temperature-scaled posteriors (same bins), not by raw regression averages.

## Rejected options

- Frozen features plus ridge/GBDT/kNN as the shipped model: Fine-Tuning label (CLAUDE.md 6.7) makes it grey/likely rejected, and the text says fine-tuning is worth about 4x. Kept only as the dev-time floor (D7).
- Hand-built features (texture slope, anisotropy, shadow-length rules, Hough/horizon finding) with or without GBDT: bans (j) and the strip test; also the point of the task is that the model must discover cues.
- TTA of any kind (flip/rotation averaging, multi-crop, multi-scale): banned (a).
- Per-group aggregation of test predictions (e.g. sharing sun elevation across a test frame or group): banned (b)/(d) and CLAUDE.md 2.3.5, even though it could plausibly help.
- Pseudo-labels, self-training, test-time normalisation, test-fitted calibration: banned.
- Mixup, cutmix, RandAugment, zoom/random-resized-crop, progressive resizing, colour/brightness jitter, blur, noise: violate the "targets genuinely invariant" augmentation clause or destroy scale/illumination cues.
- Label-transforming zoom augmentation (central crop by factor c transforms footprint to c*w exactly): not target-invariant per the clause, and adds an up-sampling blur absent in test. Not used.
- Training from scratch: allowed but weaker at 15k images; no benefit over a fine-tuned 22k backbone.
- Larger backbones (ConvNeXt-B/L, ViT-L) in the primary: runtime and memorisation risk; consider only after the lean design is measured.
- Frame-aware contrastive batches (positives from the same frame): heavy, unmeasured; roadmap-only if time.
- Joint (r, w, b) decode optimising S_view directly: not in the primary (see roadmap step 7).
- Photometric-invariant elevation: not invariant; no.
- Group-equivariant architectures or multi-view internal pooling (rotation-averaging inside the model): too close to built-in TTA; avoided (reviewer question).

## Fixed work plan & runtime budget

All counts are hardcoded constants after local profiling; time is used only for logging; device is fixed to CUDA on one A10G; seeds fixed; deterministic flags on (`cudnn.deterministic`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`); no `torch.compile`; no environment-dependent fallbacks.

| Stage | Plan | A10G estimate (UNVERIFIED) |
|---|---|---|
| Load | decode 15,114 train + 6,468 test JPEGs with PIL into uint8 tensors held on GPU (about 1.67 GB train + 0.72 GB test); no DataLoader workers | 0.5-1.5 min |
| Input checks | schema, ids, image shape, target ranges, group build | < 0.5 min |
| CV training | 4 folds x about 11.3k images x 20 epochs = 0.91M image-passes; ConvNeXt-T at 192, bf16, batch 64, throughput assumed 650-900 img/s | 17-25 min |
| OOF + test inference | 4 x 3.8k OOF + 4 x 6.5k test = about 41k single forward passes | < 1.5 min |
| Temperature fit, decode, metric report, write, re-read validation | tiny (vectorised, grid over bin centres) | < 1 min |
| **Total** | | **about 22-32 min** |

Headroom: 50-min target leaves at least 35%; the worst case is also under 42 min for a 1 h cap. If profiling shows throughput below about 500 img/s, reduce epochs (for example to 16) in the constant, not in a time branch. If a second family is approved and measured, the added 4-fold training of about 18-25 min pushes the total to about 45-55 min, above the budget; then reduce both to 3 folds or 14 epochs, or drop the second family. Memory: ConvNeXt-T batch 64 at 192 in bf16 is about 5-7 GB; peak < 12 GB with data on GPU.

**Robustness.** Input validation (train csv columns, row counts 15,114/6,468, every image file present and 192x192x3, targets finite and in bands). Output validation: columns and order exactly as sample, ids identical and in order, 6,468 rows, unique, finite, clipped to bands; re-read the written file with `keep_default_na=False`. No constant-output fallback is written (a silent degraded path is worse; a crash is loud). Final check: run twice from a clean `working/` and diff the CSVs (expect near-equal; mean absolute difference reported).

## Metric-aware training & decode

(i) **Back-solve the metric.** Each axis term is `exp(-|err|/s)` with err in ln-metres (standoff, footprint) or radians (elevation, bearing). So train in exactly those spaces: ln r, ln w, e in radians, b in radians. The posterior over the bin centres is also in those spaces.

(ii) **Weights and averaging.** The metric is a plain row mean (no per-group averaging); use equal row weights in the loss. Report a group-weighted number as a diagnostic only.

(iii) **Distributional heads with HL-Gauss cross-entropy** (ordinal Gaussian-smoothed bin targets). It trains stably, gives the full posterior, and represents multimodality (for example a bearing ambiguity b versus 180 - b, hypothesis H4) that a regression head would average away.

(iv) **Expected-utility decode, per axis.** For each row and head, compute the candidate `c` (grid of bin centres refined by a factor of a few) that maximises `sum_k p_k * exp(-|x_k - c|/s)` with the metric's own constants (0.88505864, 0.63674155, 0.25593667 rad, 1.00008771 rad). The utility is a Laplace-kernel-smoothed posterior, so the optimum lies between the median and the mode (larger s makes it more median-like, as for bearing). It is a pure function of that row's posterior, so there is no whole-test aggregation. The constants are the scorer's, not fitted. Compare on OOF against posterior mean, median and mode; ship the best by paired folds.

(v) **Posterior calibration.** One temperature per head (and optionally one per head for a mixing floor), fit by NLL on OOF over a fixed small grid (for example 15 values) with cross-fitting inside OOF for reporting. Temperature-scale each fold model's logits before averaging the 4 fold probabilities for test. Note the mismatch: the temperature is fit on single-model OOF, but test uses a 4-model mixture (slightly broader); measure on dev by comparing decode on a single fold model versus a 2-model mean.

(vi) **Coupling S_view.** The primary treats r, w, b as conditionally independent marginals. A joint decode over (r, w, b) maximising expected full `raw` (sample true values from the product posterior, project the three marks for each candidate) is a roadmap step (7) and enters only if a measured OOF gain beats noise. A training-time auxiliary term using the differentiable `S_view` on soft predictions is the alternative if the joint decode is too costly.

(vii) **Auxiliaries from structure.** Head on ln(w/r) (angular footprint), tested against predicting ln w directly (check both directions of the coupling ln w = ln r + ln(w/r)); optional L1 aux on H/r. Each kept only if it gains over noise.

(viii) **Oracle checks before modelling.** Gold labels through the metric give 1.0; gold labels snapped to bin centres then decoded must reproduce the gold within half a bin and cost < 0.002 score; the decode runtime is measured on 3.8k OOF rows.

## Structural signals

- **Exact D4 invariance** (stated): rotation by 90 deg multiples and mirror leave all four targets unchanged; used as training augmentation only (no TTA). The unknown original in-plane rotation means the model must infer the facing direction from perspective cues itself; D4 adds orientation diversity at no resampling cost.
- **Mirror and the bearing range**: bearing is an unsigned angle in [0, 180] (a bounded interval, not circular), so mirroring is valid and 0 versus 180 are far apart.
- **Facing direction is the geometry anchor**: the foreshortening axis (direction along which ground scale changes) fixes the camera's facing direction up to a possible 180-degree ambiguity; shadows and shading give the Sun direction in image coordinates; bearing is their relative angle. A learned model must combine these; do not hand-code (ban (j)). Expect a bimodal bearing posterior for low-gradient crops (H4); the decode handles it.
- **Standoff is pitch**: r = H/sin(depression); only the foreshortening anisotropy and the perspective gradient carry it, since there are no size cues.
- **Footprint is pixel scale**: w/192 is the ground sampling distance; texture and grain size in pixels relative to the learned prior of natural grain sizes carry it. Two outputs coupled by the angular footprint w/r (field of view of the crop); test whether ln(w/r) is narrower than ln w (H: discrete zoom levels times crop fractions).
- **Sun elevation**: shadow length scales as cot(elevation) and brightness as sin(elevation); both are learnable, not rules.
- **Frames**: rows of one frame share elevation and bearing (H3). Used only in validation (frames never split) and possibly as diagnostics; not used at test (cross-row use is banned).
- **Metric structure**: Laplace kernels in ln/rad space; best-constant shapes suggest bearing is nearly uniform (H1) and elevation concentrated (H2). Consequences for decode and bins.
- **No label-derived group statistics** (priors, target encodings) are used; capture_group is used only for CV splits.

## Experiment roadmap

1. **Contract, metric, validation (stop when D8 unit tests pass within 1e-3, bin oracle < 0.002).** Implement the exact metric (both `S_view` readings), reproduce REF and the seven value numbers; build group folds plus super-group merge; audit D1-D6.
2. **Cheap floors and valid end-to-end baseline.** Constant (about 0.05), group-mean oracle, frozen probe D7 (floor, never shipped), then a first fine-tune: ConvNeXt-T, L1 regression in ln/rad space, D4, 2 of the 4 folds, about 8 epochs. Writes a valid CSV; profile epoch time and fix the final epochs/folds. Stop when the CSV validator passes and the probe floor is beaten by a clear margin.
3. **Representation and structure.** Native 192 vs a modest up-sample (for example 256) as one paired comparison (resolution lever); Tiny vs Small; the ln(w/r) auxiliary; optional H/r aux. Paired, fixed epochs, same folds. Stop each comparison when the gain is below the fold std-error.
4. **Metric-aware loss and decode.** Distributional heads versus L1 regression, paired on the same folds; expected-utility decode versus median/mean/mode; temperature scalars cross-fitted. Accept if gain exceeds noise; otherwise ship the fallback.
5. **Diversity.** DINOv2-S/14 or ViT-B/16 alone first; blend posteriors only when members are close in quality and the blend beats the best single by more than noise. Respect the runtime budget (see above).
6. **In-script bounded search.** Only temperature scalars (fixed grid) and, if a second family is shipped, a single mixing weight on a fixed small grid, cross-fitted. No Optuna.
7. **Optional joint decode** over (r, w, b) for `S_view`, or a differentiable `S_view` aux loss. Only if time and the measured gain exceed noise.
8. **Final fixed-plan run.** Clean `working/`, exact platform command, run twice and diff; report OOF score per fold, per-term S, runtime, validator output, compliance audit. Credits (6): baseline (step 2), best single (step 4), ensemble (step 5, if used), final; keep two spare.

When two or three independent solvers would plausibly converge on a technique (ConvNeXt fine-tune with D4 augmentation and L1/ln targets), test it first; only afterwards the distributional/decode variant.

## Compliance audit

CLAUDE.md section 7 reds, plus the challenge bans:
- Test file use: only ids/row count and the single final forward pass; no statistics, vocab, scaler, clustering, dedup, rank-normalisation, group aggregation or train-vs-test similarity. Pass.
- Time in control flow: none; time only in log lines. Pass.
- Env-dependent fallbacks (`cuda.is_available`, `cpu_count`, import try/except): none; device fixed. Pass. (The plan assumes timm is installed; if not, the implementer must choose a single HF `transformers` ConvNeXt and not an import fallback.)
- Hard-coded tuned constants: bin ranges derived in-script from train labels; metric constants come from the description; recipe defaults are standard values; temperature grid searched in-script on train OOF only; `H` (1.995 m) appears only if the optional H/r aux is kept (listed as a reviewer question). Dev-time sweeps of at most 2-3 recipe values are on train-only grouped CV (reviewer question).
- External data / synthetic data / hosted weights / non-allowed libs: none; only the public backbone from HF/timm at a pinned revision. Pass.
- TTA (banned): none; the 4-model ensemble is a model ensemble, not input augmentation. Pass.
- Pseudo-labelling or transduction: none. Pass.
- Reading text/markers in images, OCR, matching to outside material, file-size/JPEG/order artefacts: none. Pass.
- Hand-coded heuristics as the solution: none; strip-the-ML passes (without the trained net there is no output).
- Related-row leakage: folds are group-held-out with super-group merge; test rows are never cross-referenced.
- Fine-tuning load-bearing: full fine-tune, log parameter-change norm and the CV gain over the frozen probe.
- Reproducibility: fixed seeds, fixed folds, deterministic kernels, one code path, source well under 512 KB, readable comments.
- Augmentations: D4 only, each target-invariant per the description.
- Output validity: validator against sample_submission.csv, clipping to bands, finite values, re-read check.

## Open questions & assumptions

Reviewer questions, each with the plan under each reading:
1. Is averaging the outputs of the 4 fold-trained models "TTA"? Reading A (no, they are independent trained models, stated as allowed): ship the mean posterior. Reading B (any averaging is TTA): ship one model (the fold with best OOF is not allowed to be chosen post hoc, so train one model on 100% of data) with a longer budget. Primary assumes A.
2. Does a training-time invariance (consistency) loss between two D4 views, or a group-equivariant architecture, count as TTA? Plan avoids both; D4 random augmentation is the only invariance mechanism.
3. Are standard fine-tuning default hyperparameters, with at most 2-3 development-time values compared on train-only grouped CV, acceptable (CLAUDE.md asks for HPO inside the script)? Plan keeps in-script search to temperature scalars only.
4. Is an expected-utility decode that uses the metric's own constants "metric-aware decoding" in a banned sense, or just a learned-posterior decode? Under the strictest reading, ship the L1 regression fallback (median regression with the metric's distance), which has the same goal in training instead of decoding.
5. Is the auxiliary target H/r (uses the fixed camera height from the scoring formula) a "formula with hand-chosen constants"? It is a target reparametrisation, not a rule that produces outputs; if disallowed, drop it (the net learns the relation).
6. Exact `S_view` definition: marks built from predicted vs true construction values. Both implemented; D8 unit tests pick the right one; the plan is unchanged either way.
7. Runtime limit not stated: assumed platform default; the plan targets <= 42 min worst case.

Assumptions: (a) timm and torch are present on the grader (CLAUDE.md treats timm as available); (b) HF Hub download of the pinned backbone works at runtime; (c) bf16 autocast on A10G with deterministic flags runs without raising; (d) A10G throughput of 650-900 img/s for ConvNeXt-T at 192 is an estimate to be profiled before fixing epochs; (e) the 4-fold ensemble is acceptable instead of a separate 100% refit.

**Expected private-score range (ESTIMATE, not a promise).** Chance-corrected score of about 0.30-0.55, point estimate about 0.40, reasoned from: (i) the text's claim that fine-tuning is worth about 4x over a weaker baseline (frozen floor expected about 0.05-0.15); (ii) a worked example: typical errors of 0.3 ln-units on footprint (S about 0.71), 0.2 ln-units on standoff (0.73), 6 degrees on elevation (0.66), 25 degrees on bearing (0.64), `S_view` about 0.6 give raw about 0.66, score about 0.33; halving all errors gives raw about 0.82, score about 0.64. The lower half of the range applies if bearing stays near its floor because the facing direction is ambiguous. The real private number is more likely to land slightly below the OOF value than above it (reasons in Validation design).

**What could not be verified.** All of Data findings beyond text arithmetic (group count, frames, distributions, rotation fill, duplicates, order patterns), the A10G throughput and runtime, the pinned model revision, the `S_view` reading, the probe floor, and every score estimate.
