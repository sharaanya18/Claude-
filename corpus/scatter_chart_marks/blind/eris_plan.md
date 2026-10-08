# Eris plan: scatter_chart_marks (change-mark localisation on data-quality charts) — BLIND strategist run

Status of inputs: the only task text is `corpus/scatter_chart_marks/CHALLENGE.md`, itself a *reconstruction* (not
verbatim platform text). **No dataset was supplied.** Every statement below about row counts, columns, metric formula,
hardware or runtime limit is an **assumption** unless it is quoted from CHALLENGE.md, and each is listed with the train
check that settles it (section "Data findings" is therefore a profiling protocol, not findings).
Pattern ids refer to `.claude/skills/eris-playbook/references/*` (A = metric/decoding, B = validation/selection,
C = features, D = ensembling, E = pretrained use, G = training, V = validation recipes, L = learned patterns).

---

## Contract & decision unit

**Decision unit.** One chart = one rendered image (scatter/line plot of an aggregated data-quality metric over time) plus
its metadata row (`aggfn`, `aggunit`, `fieldtype`, `dataset_name`; names taken from CHALLENGE.md, unverified).
The answer for a chart is a **set** (possibly empty) of horizontal positions of change marks. It is a 1-D keypoint
detection problem along the time axis with an explicit "no marks" outcome (CHALLENGE.md).

**One valid answer (assumption until `sample_submission.csv` is read).** One row per chart id, in sample order, holding
either (a) a serialised list of x-positions (pixel columns of the original image, or a time-bin index, or a fraction of
plot width) or (b) a has-marks flag plus positions. The empty answer is legal and must be encoded exactly as the sample
encodes it (`[]`, `""`, `none`, ...). **Validity hazard:** if empty = empty string, it reloads as NaN (CLAUDE.md §9);
the in-script validator reloads with `keep_default_na=False` and compares the empty token against the sample's.

**Invalid vs merely low-scoring.** Invalid: wrong columns/order/ids, NaN, unparsable list, positions outside the
image/plot range or wrong unit (pixel vs bin), duplicated positions if the grammar forbids them, non-integers if bins are
integers. Low-scoring: missed marks, false marks, "marks" on an empty chart.

**What the metric rewards (assumed; formula not seen).** CHALLENGE.md implies tolerance-window position matching that
penalises both misses and false positives. The plan supports the three plausible families and switches to the real one
as soon as the description is read:
- M1 per-chart F1 with one-to-one matching within tolerance δ (Hungarian or greedy by distance), averaged over charts,
  empty-gold & empty-pred = 1, one side empty = 0. Then the empty decision is worth a whole chart (gate head matters).
- M2 global (micro) precision/recall/F1 over all marks with tolerance δ. Then empty charts only matter through FPs.
- M3 a distance-graded score (e.g. 1 − min(|Δx|/δ,1) per matched mark, or AP over tolerance thresholds).
Decode differs per family (see "Metric-aware training & decode"); training is shared.

**Pipeline stages, diagnosed separately (principles §3).**
1. Evidence: a per-column (per-x) mark heatmap + chart-level P(empty) from a trained network.
2. Candidate coverage: local maxima of the heatmap after suppression — measure oracle recall (fraction of gold marks
   with a candidate within δ) at a generous threshold.
3. Ranking/calibration: does peak height order true vs false candidates (AP of candidates)?
4. Decode: choose the emitted subset (threshold / expected-F1 / empty gate) — measure with gold-perfect heat (oracle
   decode) and with OOF heat.

---

## Compliance regime

**Domain:** computer vision (rendered charts) with a 1-D localisation output. CLAUDE.md §6.3 regime: a trained deep
model on the pixels is required; tabular-on-pixels is banned; tabular-on-image-features is grey.

**Open question 1 — pretrained timm backbones (flagged by CHALLENGE.md, NOT resolved here).** Two readings:
- Reading A (default per CLAUDE.md §1/§2.2 when the real description is silent): ImageNet-pretrained timm weights from
  the HF hub may be loaded and fine-tuned.
- Reading B (if the real text says "from scratch"/"no pretrained"/"no external checkpoints", CLAUDE.md §2.4, §6.8):
  no pretrained weights of any kind.
Action: read the real description's model/compute lines; if they are silent or ambiguous, ask a reviewer in writing
("May solution.py download and fine-tune an ImageNet-pretrained timm backbone such as convnext_atto.d2_in1k from the HF
hub?") and save the answer. **The plan is built so that the from-scratch arm (Arm S) is compliant under both readings
and is always built first**; the pretrained arm (Arm P) is added only under Reading A. Measured cost of the
conservative choice = paired grouped-CV gap Arm P − Arm S (roadmap step 4), reported so the owner can decide.

**Open question 2 — hardware.** CHALLENGE.md suggests a CPU-first/CPU-only environment (rank-1 hard-codes CPU).
If the real text says CPU-only: `device = "cpu"` hardcoded, fixed `torch.set_num_threads(N)` matching the stated cores,
no AMP. If it says A10G: `device = "cuda"` hardcoded. **Never** `cuda if available else cpu` (Deterministic Execution,
engineering §1; the CHALLENGE.md note that rank-2 does this is not evidence it is accepted, L010).

**Explicit bans to honour (from CLAUDE.md, description wording still to be checked):** no test statistics of any kind
(no normalisation, thresholds, clustering or dataset-level pooling across test charts); no pseudo-labels; no synthetic
charts (rendering new labelled charts is synthetic data, §2.3 #6; augmenting real charts is fine); no hand-coded
detector that solves the task (template matching for the mark glyph, CUSUM/PELT on an extracted series); decode
constants (`NMS` radius, peak and empty thresholds) must be found by an in-script OOF search, never hardcoded from
earlier submissions (Q3); TTA only if the description does not ban it.

**Self-audits on the design.**
- Strip-the-ML: remove the network and nothing predicts marks (hand per-column statistics are only extra input channels
  to the trained network; the decoder only reads the network's heatmap). Passes.
- No whole-test aggregation: each chart's output = f(its own image, its own metadata, train-fit weights, OOF-fit decode
  constants). Half-rows test must pass (drop half the test charts, kept charts' outputs identical).
- Sibling leakage: charts of the same `dataset_name` very likely share mark dates (one data-quality incident shows in
  several fields' charts). Any feature that looks up other charts of the same dataset (train labels keyed by dataset and
  date, or other test charts) is a sibling/ID feature → excluded. `dataset_name` is used **only** to build CV groups,
  never as a model input.
- Every constant derivable by an in-script train-only search: plot-area crop bounds are derived in-script from train
  images (pixel statistics, no labels) or the full image is used; σ of the heatmap target, NMS radius and thresholds are
  chosen on OOF inside the script.
- Genuine training is load-bearing in both arms (Arm S is trained from random init; Arm P fully fine-tuned, parameter
  change norm logged and asserted > 0).

---

## Data findings

No data available; nothing below is measured. This is the ordered profiling protocol (train only; test used only for
schema, row count, image size distribution for runtime planning), with the decision each item drives.

1. **Files and schema.** List `public/`: images dir, `train.csv`/`test.csv`, label format (list column? one row per
   mark?), `sample_submission.csv` columns and the empty-answer token. Decides output grammar and the validator.
2. **Label unit.** Are gold positions pixel x (original image), plot-relative fraction, or time-bin index? If bins,
   is the bin→pixel mapping recoverable per chart (number of points, `aggunit`)? Decides the output head's coordinate
   system and the inverse mapping.
3. **Are marks visible in the input image?** Overlay gold x on ~30 train images. Case V (the glyph is rendered in the
   input; detection of a drawn mark) vs Case L (marks are labels of *where the series changes*; nothing drawn). Same
   architecture, different emphasis: Case V → resolution, glyph-scale features, precision; Case L → long receptive
   field across time, extracted-series channels, calibration of ambiguous changes. Case V also triggers the V15
   saturated-metric routine (near-perfect CV must be proven to be learned, not a leak).
4. **Image geometry.** Sizes (CHALLENGE.md hints height ~64 and width ~512 after rank-2's resize, unverified), fixed
   layout or not (axes, tick labels, legend, margins → plot-area columns), RGB vs grayscale, mark colour. Decides input
   resolution (never downsample x below the mark width or below δ/2 per output bin) and whether to crop to the plot area.
5. **Label statistics.** % empty charts (exact), marks-per-chart histogram, minimum gap between marks on one chart (sets
   the NMS radius lower bound and the decodable ceiling), x-position histogram (edge marks? uniform?), by `aggfn`,
   `aggunit`, `fieldtype`.
6. **Groups.** Charts per `dataset_name`; within a dataset, do marks share x/time positions across charts (sibling
   test)? Exact/near-duplicate images (perceptual hash on train, union-find with `make_groups.py`-style logic).
7. **Metadata cardinalities** and whether `aggunit` changes the x-scale (series length / points per column).
8. **Ambiguity ceiling.** Near-identical charts with different marks (irreducible label noise); inter-mark position
   jitter between sibling charts of one dataset (proxy for annotation noise ~ the useful δ scale).
9. **Leakage tells.** Filename/id order vs labels, file size vs empty/non-empty (an id/size-only model under grouped CV
   must show no lift, V19b). Never used as features.

---

## Validation design

**Hidden split (assumption).** The description (reconstructed) is silent. Strictest plausible reading: test charts come
from datasets not seen in train, or at least charts are random. Build **StratifiedGroupKFold, 5 folds, groups =
union-find over `dataset_name` ∪ near-duplicate images**, stratified on marks-count bucket {0,1,2,3+} × `aggfn`.
- Arm S (cheap): 5 folds × 2 split seeds. Arm P: 5 folds × 1 seed (+ second seed only for the final finalists, V18).
- Sanity hold-out: ~15% of groups untouched by any selection until the final check (V4).
- Also run the same model under plain random KFold once: the random-vs-group gap measures sibling leakage (V19c). If
  gap is large, group CV is the honest proxy and the private score of anything sibling-dependent would fall.
- Bias direction: dataset-group CV is **pessimistic** if the real test is a random chart split (siblings would help a
  little there, but we do not exploit them), **unbiased** if the test is dataset-disjoint. Tuned decode constants
  evaluated in-sample are **optimistic** (by ~0.01–0.03 on a few hundred charts); always report the cross-fitted number.

**Metric re-implementation and unit tests** (before any model): implement M1/M2/M3 with a δ parameter and the exact
matching rule once known. Tests: perfect → 1; all-empty predictions → score = %empty charts (M1) / 0 recall (M2);
predictions shifted by δ+1 → 0 matches; duplicate predictions on one gold → one TP + FPs; reversed (mirror x) → low.
Print the constant baselines (all empty; one mark at the train-modal x) as floors.

**Oracle decode check (A, principles §3, lesson 15):** rasterise gold marks into the Gaussian target at the model's
output resolution, run the full decoder (peak finding + suppression + sub-bin refinement + inverse coordinate map +
serialisation), score with the metric: must be ≥ 0.999. If not, the output resolution, NMS radius or coordinate mapping
is wrong (e.g. two gold marks closer than the suppression radius). Repeat for every candidate input size.

**Nested selection.** Each post-hoc step (thresholds, NMS radius, empty gate, blend weight, calibration map) is fitted on
OOF of group-half A and scored on half B, then swapped (V3). Accept a change only if the paired gain on identical folds
passes the corrected resampled t (V16) and is sign-consistent on ≥ 4/5 folds.

---

## Overfit/underfit risks

| Risk | Where | Mitigation |
|---|---|---|
| Few independent units (datasets) behind many charts | CV and capacity | groups by dataset; report #groups; small heads; strong augmentation; capacity ladder (tiny backbones only) |
| Memorising dataset-specific mark dates (siblings) | any model seeing many charts of one dataset | group CV; never feed `dataset_name`; random-vs-group gap logged |
| Downsampling x destroys small marks / precise positions | input resize, backbone stride | keep x resolution ≥ native/2 and output stride ≤ δ/2; anisotropic resize (shrink height, not width); oracle check per size; sub-bin quadratic refinement |
| Too short receptive field along time (Case L) | 2D CNN with small RF | dilated 1-D conv stack over columns (RF ≥ 256 columns) on top of the 2D features |
| Threshold/NMS overfit to OOF | decode | ≤ 3 decode constants; cross-fitted; plateau choice; bounded grids |
| Empty-gate miscalibration (M1 gives a whole chart for empty) | gate head | dedicated chart-level head with BCE, Platt-calibrated on OOF, decoded by expected utility |
| Under-training on CPU budget | Arm P on CPU | tiny backbones (convnext_atto 3.7M, efficientnet_b0 5.3M), small inputs, fixed epochs from profiling; ship fold ensemble instead of refit if refit doesn't fit the budget |
| Loss mismatch with set/tolerance metric | heatmap BCE | Gaussian σ tied to δ; penalty-reduced focal loss; calibration of peak heights to P(match) |
| timm tag missing in the platform image | Arm P | pick tags present in timm ≥ 0.9 (convnext_atto.d2_in1k, efficientnet_b0.ra_in1k, resnet18.a1_in1k); verify on a Kaggle-image run; no try/except fallback |

---

## Recommended approach (primary + fallback)

**Shared framework (both arms): "image → per-column 1-D heatmap + chart-level empty head", translation-equivariant
along x (lesson F2.5), decoded per chart.**

Input pipeline (inside the script, from raw files every run):
- Read image, convert to float RGB (grayscale only if train shows charts are monochrome and marks are not coloured).
- Crop to the plot area if layout is fixed: bounds derived in-script from TRAIN images (median column/row darkness
  profile → axis lines), asserted on every train image; otherwise use the full image. (CHALLENGE.md mentions rank-1
  geometry constants `C0/R0/R1`; we derive them, never paste them.)
- Anisotropic resize: height → H (Arm S 64, Arm P 96–128), width → W = native plot width or 512–1024 (whichever keeps
  ≥ 2 output bins per δ). Positions are carried in continuous coordinates with an exact inverse map.
- Per-image hand channels (own image only, no test stats) concatenated to the 1-D features: per-column ink count,
  per-column mean/min row of non-background pixels (the plotted value), its first difference, per-column
  non-gray-colour fraction (Case V glyph colour). These are inputs to a trained model (C4/F4), never a decision rule.
- Metadata: `aggfn`, `aggunit`, `fieldtype` → small learned embeddings (vocabulary fit on train; unseen → "UNK" index),
  injected by FiLM into the 1-D stack and into the empty head. `dataset_name` excluded (group key only).

**Arm S — from scratch, compliant under every reading (always built; primary under Reading B).**
- 2-D stem: 3–4 conv blocks (3×3, GroupNorm, GELU) with stride (2,1) — height-only downsampling — channels 32→64.
- Collapse height by learned attention pooling + max pooling over rows → (C, W) sequence.
- 1-D trunk: 8–10 residual dilated Conv1d blocks (k=3, dilations 1,2,4,…,128, width 128) → RF ≥ 512 columns;
  optional 1-layer BiGRU(64) (decide by paired CV).
- Heads: per-column heat logit (Gaussian target), per-column offset (only if output stride > 1), chart-level empty logit
  from [attention-pooled features ‖ max heat ‖ metadata], auxiliary mark-count head (Poisson NLL; aux only).
- ~0.5–1.5 M parameters; CPU-trainable.

**Arm P — pretrained tiny CNN fine-tuned (primary under Reading A).**
- timm `convnext_atto.d2_in1k` (Apache-2.0, 3.7 M) as first choice; `efficientnet_b0` or `hgnetv2_b0` as the diversity
  second member only if present in the platform's timm version (verify, pin tag; no fallback code).
- `features_only` at strides 4/8/16; collapse each stage's height with attention pooling, upsample along x to stride 2
  (or 4 if δ is wide) and sum (a 1-D FPN), concatenate hand channels, then the **same** 1-D dilated trunk + heads as
  Arm S (the long-range temporal reasoning lives there).
- Full fine-tune (CPU/GPU budget permitting) with backbone LR 10× below head LR after a short head-only warm-up (LP-FT,
  E-B), AdamW, cosine, EMA of weights, fixed epochs; assert parameter-change norm > 0.

**Why this fits the task:** the target lives on one axis, so a 1-D dense prediction with height pooling is the smallest
model that preserves every valid answer (F6 "smallest learnable decision"); dense per-column scoring is
translation-equivariant (marks anywhere in time), handles 0..k marks natively, and the empty head addresses the explicit
no-marks case. A 2-D detector (boxes) adds a y-coordinate nobody scores.

**Primary (Reading A):** Arm P (convnext_atto) + Arm S blended (equal-weight average of calibrated per-column logits,
weight checked against cross-fitted OOF-fit weight), because the two differ in assumption (ImageNet edge filters at 2-D
vs from-scratch column tokens with hand channels). Ship only if the blend beats the best single arm by more than noise;
otherwise ship the best single arm. **Required margin for keeping Arm P at all:** it must beat Arm S by ≥ 1 corrected SE
on group CV; Arm P adds review risk under an unresolved weights question, so a tie goes to Arm S.

**Fallback (Reading B or no written answer that pretrained is allowed when the text is ambiguous):** Arm S alone,
2 seeds (or 2 architectural variants: with/without BiGRU) averaged, fold models or refit per the budget rule below.

---

## Rejected options

- **Classical change-point detection (CUSUM/PELT/BinSeg) on the pixel-extracted series, or template matching of the mark
  glyph** as the predictor: strip-the-ML failure (§2.1, Q1). Extracted-series channels survive only as inputs.
- **GBDT/logistic per-column classifier on hand features** as primary: "tabular model on image-derived features" is grey
  in the CV regime; kept only as a 30-minute yardstick to size the information in the hand channels.
- **Sibling/dataset-level label lookups** (train marks of the same `dataset_name` at the same date as a feature, or
  pooling test charts of one dataset): ID feature + sibling leakage + cross-row test use. Rejected; the random-vs-group
  gap will show what it would have been worth.
- **Large backbones (convnext_base, DINOv2/EVA ViTs)**: patch-16 ViTs quantise x to 16 px (too coarse for small marks at
  δ of a few px), and the CPU budget cannot fine-tune them with OOF. Reconsider only on A10G with measured gain.
- **2-D object detectors (FasterRCNN/YOLO/ultralytics)**: wrong output space, torchvision weights come from
  download.pytorch.org (not HF/timm), ultralytics not in the image.
- **Sequence generation of coordinate lists**: unnecessary; dense heatmap + decode preserves every answer.
- **Pseudo-labelling test, test-time BN/adaptation, test-wide thresholds**: banned (§2.3 #5).
- **Rendering synthetic charts with planted marks**: synthetic labelled data, banned (§2.3 #6). Copy-paste of real
  marks between charts (Case V) is borderline (fabricated hybrid charts) → not used.
- **Validation-triggered early stopping in the final refit / time guards**: fixed epochs only (CLAUDE.md §3).

---

## Fixed work plan & runtime budget

All counts are constants at the top of `solution.py`, set after local profiling (time one epoch per arm, multiply,
hardcode). No clock, hardware, environment or try/except branches. Seeds for `random`, `numpy`, `torch`, DataLoader
generator; `num_workers` fixed (0 on CPU is safest for determinism); `torch.set_num_threads(T)` fixed;
`use_deterministic_algorithms(True, warn_only=True)`; double-run diff before upload.

Images are decoded and resized **once** into a uint8 tensor in RAM (N × 3 × H × W; at N = 10k, 3×128×1024 ≈ 3.9 GB —
use H = 64/96 or grayscale if N is large; assumption: N in the low thousands).

Budget formula: `time ≈ (folds + refit) × epochs × N / train_img_per_s + (N_oof + N_test × views) / infer_img_per_s`.

**Scenario CPU-only (assumed likely; ~8–10 cores, target ≤ 45–50 min with 30% headroom against the stated limit):**
| Stage | Plan | Estimate (assumption N≈3k train, 1k test; MUST be profiled) |
|---|---|---|
| Load + resize + hand channels | once | 1–2 min |
| Arm S, 64×512 input | 5-fold × 1 seed OOF, 30 epochs, batch 32; ship the 5 fold models (no refit) | ~150–400 img/s train → 8–20 min |
| Arm P (Reading A), convnext_atto at 96×768 | 4-fold, 12 epochs, ship fold models | ~20–40 img/s → 15–30 min — likely too slow; then reduce to H=64, W=512, 3 folds, or drop Arm P from the CPU plan if Arm S wins |
| Decode search on OOF + cross-fit | grid ≤ 400 configs, vectorised | < 1 min |
| Test inference (+ hflip TTA if valid) | fold average | 1–2 min |
On CPU, ship the fold ensemble rather than refitting: the OOF used to fit decode constants then comes from exactly the
shipped model type (B9 transfer issue avoided) and saves one fit.

**Scenario A10G:** Arm P at 128×1024, 5 folds × 25 epochs + full refit with the same epoch count, bf16 autocast,
channels_last; Arm S 5 folds × 2 seeds; total ≈ 15–25 min for N ≈ 5k. Refit on 100% when it fits (V9).

Output validation inside the script: id set/order vs sample, list parse round-trip, positions finite and inside
[0, W_orig) (or bin range), sorted, unique, empty token as in sample; reload with `keep_default_na=False`; raise on any
failure; write once at the end (no placeholder file).

---

## Metric-aware training & decode

**Targets.** For each chart, per-column target = max over gold marks of exp(−(x−x_k)²/(2σ²)) at output resolution;
σ chosen from {δ/4, δ/2, δ} on OOF (one nested choice). Exact peak value 1 at the rounded centre; continuous offset
target if stride > 1.
**Losses.** Penalty-reduced focal loss on the heatmap (CenterNet form, α=2, β=4) — or BCE on Gaussian soft labels;
pick one by paired CV — plus BCE on the chart-level empty logit, plus 0.1 × Poisson NLL count head. Output-bias init at
the base rate (empty head bias = logit(%empty); heat bias = logit(mean target)). If the metric averages per chart (M1),
weight each chart equally in the loss (normalise heat loss by number of marks + 1) so long, mark-dense charts do not
dominate (A7).
**Augmentation (real charts only, geometry-consistent, every geometry-tied label moved with the image):**
x-translation/crop-pad with mark shift (positions near the edge kept only if the mark stays inside), mild x-stretch
0.85–1.15 with label rescale, intensity/contrast jitter, small blur/JPEG-like noise, vertical shift/scale of the plot
area. **Horizontal flip** (x → W−x) and **vertical flip** only after a train diagnostic: train with vs without on
identical folds; keep if not harmful (a step change is still a change under time reversal; a drawn glyph may be
asymmetric, axis labels move — crop to the plot area first). TTA = the same flips, un-transformed (x → W−x) before
averaging heatmaps, kept only if OOF gains (and only if the description does not ban TTA).
**Decode (per chart, no test statistics).**
1. Average member heatmaps in logit space (after per-member Platt calibration on OOF; D2), light 1-D smoothing.
2. Candidates = local maxima via 1-D max-pool equality within radius r (NMS radius) — r chosen on OOF from a grid
   bounded below by 1 bin and above by min train gold gap (oracle check guarantees decodability).
3. Sub-bin refinement by parabolic fit over the 3 bins around each peak (parameter-free); inverse-map to original units.
4. Calibrate candidate heights → P(candidate matches a gold within δ) by isotonic on OOF (cross-fitted).
5. Emission rule, by metric family:
   - M1 (per-chart F1): expected-utility decode. Sort candidates by p_i; for k = 0..K compute
     E[F1_k] ≈ (1 − p_empty) · 2Σ_{i≤k} p_i / (k + E[n | non-empty]) and E[F1_0] = p_empty, with E[n] from the count
     head (calibrated on OOF). Emit argmax_k (A3, metric-and-decoding §3). Compare against the simple two-threshold rule
     (τ_peak, τ_empty grid on OOF); ship whichever wins cross-fitted.
   - M2 (micro F1): single global τ_peak from the closed-form F-threshold fixed point (τ = F*/2 on calibrated p; A12),
     empty gate only as FP control.
   - M3: same as M1 with the distance-graded utility in the expectation.
   Total free decode constants ≤ 3 (σ, r, and τ or none for expected utility), bounded grids, plateau choice.
6. Report: candidate oracle recall (coverage), candidate AP (ranking), decoded score (decode) per fold.

---

## Structural signals

- **Translation equivariance along time**: marks can appear anywhere → dense per-column head, x-shift augmentation,
  no flatten/global-position head.
- **Height irrelevance of the answer**: only x is scored → height pooled early; vertical augmentations are free.
- **Explicit empty outcome**: dedicated gate head + expected-utility decision (C7).
- **Set output, order-free, minimum separation**: NMS radius bounded by the train minimum gap; sorted unique output.
- **Time reversal / value flip symmetries**: candidate augmentations, adopted only after the train diagnostic (lesson 6:
  test, don't assume).
- **Metadata conditions the scale** (`aggunit` sets points per pixel; `aggfn`/`fieldtype` set what "a change" looks
  like): FiLM conditioning; per-slice OOF report (by `aggunit`, `aggfn`) to find the weak slice.
- **Dataset siblings**: used only for grouping (validation), never as signal.
- **Causal reading of a change (Case L)**: a change at x is evidenced by the contrast between the windows left and right
  of x → the 1-D trunk sees both sides (non-causal dilated conv / BiGRU), and hand channels include the per-column
  plotted value so the trunk can learn level/variance shifts.

---

## Experiment roadmap

Each step: paired grouped CV (same folds), report mean ± std and per-fold, keep only gains beyond the corrected SE.
1. **Contract** (½ day): read real description + sample; fix metric family, δ, unit, empty token; implement metric +
   unit tests; validator; answer Case V vs Case L; write the profiling numbers into this plan. Ask the pretrained-weights
   question in writing now. Stop when oracle decode ≥ 0.999 at the chosen resolution.
2. **Floors**: constant baselines; GBDT per-column yardstick on hand channels (diagnostic only).
3. **Arm S baseline end-to-end** (fixed plan, 5-fold OOF, simple two-threshold decode), valid CSV from the exact command,
   run twice and diff. First credit: this baseline if it clears the printed AI baseline.
4. **Representation**: input resolution ladder (W 512 → native, H 64 → 128) with oracle check; hand channels on/off;
   metadata FiLM on/off; Arm P (Reading A) vs Arm S paired → this is the measured cost of Reading B.
5. **Metric-aware**: σ choice, focal vs BCE, empty head + count head, expected-utility decode vs thresholds (cross-fit),
   isotonic calibration. Augmentation diagnostics (flip H/V, x-stretch).
6. **Diversity**: Arm P + Arm S blend; second backbone (efficientnet_b0 / hgnetv2_b0) only if timm in the image has it
   and the blend gain clears noise. OOF correlation matrix printed.
7. **Small in-script HPO** (fixed ≤ 12 Optuna trials on Arm S LR/width/dropout, seeded TPE, scored on a fixed fold
   subset, nested check) only if steps 3–6 leave runtime headroom.
8. **Freeze**: final fixed plan from a clean `working/`, run twice and diff, half-rows test, compliance scan,
   runtime ≤ 70% of the limit; second credit = best single, third = blend (if different), final = frozen best.

---

## Compliance audit

- 🔴 Test file read only for one-chart-at-a-time inference: yes by design (no test normalisation, no dataset pooling).
- 🔴 Clock in conditions: none (time only in `log`).
- 🔴 Hardware/env branches: none; device hardcoded per the description (CPU or cuda), threads/workers fixed.
- 🔴 Hardcoded tuned constants: crop bounds, σ, NMS radius, thresholds all derived in-script from train/OOF; fixed
  training counts come from profiling (principled defaults, allowed).
- 🔴 External/synthetic data: none; augmentation of real charts only; no rendered synthetic charts.
- 🔴 Strip-the-ML: no rule detector remains without the network.
- 🔴 Pretrained weights: Arm P only under Reading A (description silent → CLAUDE.md default allows HF/timm; ambiguous →
  written reviewer answer first). Arm S compliant under both readings.
- 🟠 Readable source < 512 KB, docstring requirements map (inputs, model, training, validation, decode, determinism).
- 🟠 `dataset_name` used only for CV groups — state so in a comment (reviewers may otherwise read it as an ID feature).
- 🟢 Seeds, fixed workers/threads, comments on every fit/predict, double-run diff, validator green.

---

## Open questions & assumptions

1. **Pretrained timm backbones allowed?** (reviewer, written). Plan under both readings above; Arm S always compliant.
2. **Hardware/runtime limit**: CPU-only (assumed likely) vs A10G; exact limit (assume ≤ 1 h, plan ≤ 45 min).
3. **Metric formula** (M1/M2/M3, δ, matching rule, empty-empty credit) — unverified; decode switches on it.
4. **Submission grammar** (list serialisation, empty token, unit = pixel/bin/fraction) — unverified.
5. **Are marks rendered in the input images (Case V) or latent change points (Case L)?** — decides emphasis, not design.
6. **Hidden split** (chart-random vs dataset-disjoint) — assumed dataset-disjoint for validation (conservative).
7. **TTA allowed?** default yes per CLAUDE.md §2.2 unless the description bans multi-view inference.
8. **Reviewer question to ask once:** "Is using `dataset_name` only as a cross-validation grouping key acceptable?" (it
   is never a model input).
9. Row counts, image sizes, %empty, marks-per-chart — all unknown; the runtime table is an estimate to be replaced by
   profiled numbers.

**Expected score (estimate, not a promise):** Case V (visible glyphs) with a correctly sized resolution: tolerance-F1 in
the high 0.9s is plausible because the evidence is local and unambiguous; the ranking will be decided by empty-gate
calibration, edge marks and close pairs. Case L (latent change points): roughly 0.5–0.75 F1-type, limited by label
ambiguity; the dataset-group CV is the honest proxy and the private score should land within ~0.02 of the cross-fitted
CV if the split assumption holds, lower if the test shares datasets in a way we chose not to exploit.
