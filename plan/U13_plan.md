# U13 Warehouse object detection with selective human review: Eris build plan

Status of this plan: written from the challenge text and CLAUDE.md only. No dataset was available, so everything under "Data findings" is a list of diagnostics plus UNVERIFIED hypotheses. Every number that depends on the data (image count, group count, image size, throughput) is an assumption and is marked as such. No test image, test statistic, eval/ or rubric file was opened.

## Contract & decision unit

**One valid answer (per row of test.csv, keyed by `id`):**
- `detections`: JSON array of `[class_id, xmin, ymin, xmax, ymax]`, class_id an int in 0..4, coordinates normalised to image width/height, continuous convention (no +1 pixel), `0 <= xmin < xmax <= 1`, `0 <= ymin < ymax <= 1`. `[]` is valid. No confidence column. Duplicates are separate predictions.
- `review`: either `[]` or exactly one `[xmin, ymin, xmax, ymax]` with the same range rules and `(xmax-xmin)*(ymax-ymin) <= 0.10`. Boundary-touching counts as contained.
- File: `working/submission.csv` (the Eris entry point writes to `sys.argv[2]`), columns `id,detections,review` in the order of `sample_submission.csv`, same ids. Row and detection order are free.

**Invalid (scores below zero, burns a credit) vs merely low-scoring:**
- Invalid: wrong columns or ids, non-JSON strings, class id outside 0..4 or non-int, `xmin >= xmax`, coordinate outside [0,1], NaN, review area above 0.10 (float rounding included), more than one review box, a review box given as a nested list. Plan to write review area at most 0.099 after rounding to 6 decimals and to re-read and re-validate the written file.
- Low-scoring but valid: empty detections with no review (score 0 on any image with reference objects, per the description), duplicates, wrong classes, loose boxes.

**What the metric rewards, term by term**
1. Per image: submitted boxes whose entire box lies in the review window are removed; every reference box entirely inside the window is added as a perfect box. Reference boxes only partly in the window stay in the auto set.
2. Per image and per class: maximum-cardinality one-to-one matching between remaining submitted and remaining reference boxes, eligible only at IoU >= 0.75 and same class. Only the match COUNT matters (any maximum matching gives the same count), so a bipartite eligibility matrix with Hungarian or Hopcroft-Karp is exact; greedy-by-IoU is not.
3. Per warehouse (group_id) and class, pool over the warehouse's images:
   - TP = automatic matches + fully reviewed reference objects
   - predicted = remaining submitted + fully reviewed reference objects
   - reference = all reference objects
   - class F1 = 2*TP/(predicted + reference)
4. Warehouse score = mean of class F1 over classes with at least one prediction OR one reference object in that warehouse. A class absent from both sides is omitted. A warehouse with no classes at all scores 1. Final score = unweighted mean over warehouses.

**Consequences that drive the design (derived from the formula, exact reasoning given):**
- Strict IoU 0.75 makes localisation precision, not recall, the main ceiling, especially for small, stacked and blurred objects.
- Pooling is per warehouse AND per class, and warehouses and classes are weighted equally regardless of size. A rare class in a small warehouse has the same weight as pallets in a big one. Loss weighting and thresholds must reflect this hierarchical averaging.
- Spurious class penalty: false detections in a class that has no reference object in that warehouse give class F1 = 0 and still enter the class average. One spurious forklift detection in a warehouse whose reference has 2 classes drops that warehouse score by about 1/3. So rare-class precision is worth far more than rare-class recall at the margin. This is the most unusual feature of the metric and must be handled through a trained, per-class calibrated decision, not hand rules.
- Review gain, linearised. Let a window contain, for one class in one warehouse, `FN_in` reference objects not auto-matched and `FP_in` submitted boxes that are not auto-matches. Writing `D = predicted + reference` and `F` for the class F1, then
  `dF ~ [ (2 - F) * FN_in + F * FP_in ] / D`.
  Both terms are nonnegative, so growing a window never hurts: always use the maximum permitted area (0.099). The only free choices are aspect ratio and position. Value comes from windows that (a) fully contain many objects the detector would miss or mislocalise (small, stacked, clustered, blurred), and (b) fully contain many uncertain false-positive-prone detections. Large objects (forklifts, large stillages) are rarely fully containable in a 10% window and are poor review targets.
- Boxes inside the chosen window are free (they are replaced by the reference), so low-confidence candidates in the window cost nothing.

**True independent unit:** the warehouse (group_id). Several photographs of the same physical objects exist inside a warehouse, so images within a group are near-duplicates for generalisation purposes. The effective sample size for validation and for any fitted policy is the number of train warehouses, not the number of images or boxes.

**Decision unit at inference:** one image. Prediction for an image must be a function of that image alone plus a train-fit model (no cross-image information, see Compliance regime). The output object is (detection set, one window). Pipeline stages to diagnose separately:
1. Candidate coverage: oracle recall at IoU >= 0.75 of the full low-threshold candidate pool, per class and size bucket.
2. Scoring and ranking: per-class AP at IoU 0.75, and how well calibrated "match at 0.75" probability is.
3. Decode: thresholds, duplicate suppression, class-presence handling, scored by the exact metric.
4. Review: window policy scored against oracle window, random window, central window and no window.

## Compliance regime

**Domain:** computer vision, object detection (CLAUDE.md 6.3). A trained/fine-tuned CNN or transformer detector on pixels is required; no raw pixels into a tabular model. Not a "from scratch" or "fine-tuning-labelled" challenge by the text. Pretrained COCO or ImageNet weights are explicitly allowed by the description.

**Explicit bans in the description, each treated as a hard constraint:**

| Ban (description text) | How the plan honours it |
|---|---|
| Additional training images or annotations beyond the supplied set | Train only on `train.csv` images. No synthetic or LLM-generated boxes, no pseudo-labels (also banned by Eris 2.3). |
| Pretrained weights trained specifically on industrial or warehouse datasets | Use only COCO-trained or ImageNet-trained general weights. Reject any checkpoint whose card mentions Objects365, LVIS-extra, SKU, logistics, warehouse, or any industrial set, and reject web-scale foundation weights (CLIP, DINOv2, SAM) because the description whitelists only COCO and ImageNet. Prefer the plain-COCO checkpoint name over variants named `*_coco_o365` or similar. Pin the model revision. |
| Looking up the source collection or recovering external annotations | Never search for, name or compare to any public dataset. No internet use beyond pretrained weight download. |
| Manual annotation or human inspection of test images to choose predictions or review windows | No test image is ever viewed, plotted or hand-checked. Visual debugging only on train/OOF images. Review windows come only from the trained policy. |
| External inference APIs or remote model execution | None. All inference in the script on the local A10G. |
| Fitting or adapting model parameters, thresholds or review policies on test images | All constants (class thresholds, calibrator, NMS/fusion IoU, review constants) are fitted in-script on TRAIN out-of-fold detections only. No test-time adaptation, no BN-statistics adaptation (BN frozen, eval mode), no test-score-distribution matching, no "expected objects per image" matching on test. |
| Combining information across different test images during inference | No cross-image NMS, deduplication, matching, ranking, normalisation or voting; no use of `group_id` in test (the column is not read at inference); no batch statistics; no "pick the top-k images to review" logic; no exploiting that photographs of one warehouse show the same objects. Each image is processed independently. Additional engineering rule: outputs must not depend on batch-mates, so every image is resized to one fixed canvas independently (no pad-to-batch-max) and eval-mode BN is used; add a train-only unit test that predictions for a batch of 8 equal predictions for batch size 1 within tolerance. |
| Using IDs, filenames, row order or serialization artifacts to infer targets | No feature from `id`, `image` filename, `group_id` (test), file size, EXIF, JPEG quantisation tables, row order or CSV serialization. `group_id` is used only on train to build validation folds. Read-only use of width/height only to scale coordinates (derive from the decoded image; see open questions). |
| Runtime (90 min incl. installation, weight download, preprocessing, training, validation, inference, writing) on one A10G 24 GB, 62 GB RAM | Plan to about 50-55 min (see runtime). CLAUDE.md forbids `pip install` and wall-clock branching; the stricter rule wins. |

**Where the description is silent, and what this plan assumes:**
- Ensembling, flip/multi-scale TTA and WBF fusion across several models for ONE image: not mentioned, assumed allowed because they combine views of the same image only (Eris 2.2 allows TTA).
- Number of train images, number of warehouses, number of test images, image resolution, CPU cores: unknown. Plan is parameterised and the work counts are fixed after a profiling run at implementation time.
- Maximum detections per image: not capped; assume capping at 300 is fine.
- Whether `width`/`height` columns of test may be read: assume reading them or the decoded image size is fine (normalisation needs the size); the plan reads size from the decoded image to stay clear of the "serialization artifacts" ban.
- Whether a trained post-hoc calibrator on detector outputs counts as "training": assumed yes (it is a trained model on train OOF outputs, within-row features only).
- Installation: the description mentions installation inside the time budget; CLAUDE.md forbids `pip install` and requires the Kaggle stack. Assume no installation; use only installed `torch`, `torchvision`, `transformers`, `timm`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv`/`pillow`. Availability and version of `transformers` RT-DETR classes are UNVERIFIED.
- Metric-aware window selection from the model's own outputs: the task itself defines review as part of the output and bans only fitting the policy on test, so a train-fitted policy applied per image is assumed compliant (the mechanism is "expected-gain search", not test fitting).

**Eris-specific bans applied:** weights only from Hugging Face Hub or timm (so torchvision's `download.pytorch.org` COCO checkpoints and anything from GitHub, e.g. ultralytics YOLO weights, are out); no hand-written rules replacing the detector; no wall-clock control flow; no env-dependent fallbacks; fixed seeds; source under 512,000 bytes and readable.

## Data findings

**Nothing below is verified. These are the exact diagnostics to run on `train.csv` and `images/` (train only), with the hypothesis expected for each. Test is touched only for row count, id format and the number/size of images for runtime planning; test feature distributions are not studied.**

D1. Schema and sanity (train)
- Run: row count; number of unique `group_id`; images per group (min/median/max, number of single-image groups); count of rows with `detections == []`; JSON parse of all `detections`; number of boxes per image; check `0<=xmin<xmax<=1`, `0<=ymin<ymax<=1`; duplicate boxes within an image; class ids outside 0..4.
- Expected: few dozen warehouses or fewer (UNVERIFIED), strongly uneven images per group, a handful of malformed/degenerate boxes. Few groups means huge variance of the warehouse-averaged metric.

D2. Class and presence structure
- Run: boxes per class; images per class; and a group x class presence matrix (how many warehouses contain each class, how many warehouses have only 1-2 classes).
- Expected: pallets dominate; forklift (1) and pallet truck (4) are rare and absent from many warehouses; stillage and small load carrier appear in a subset. If true, the spurious-class penalty is large and rare-class thresholds are decisive. Also expected: warehouse-level F1 per class has very high variance.

D3. Box geometry
- Run: normalised width, height, area, aspect ratio, and pixel size at 1024 long side, per class; fraction with min side below 16 px and below 32 px at 1024; fraction touching an image border; same-class pairwise IoU within an image (share above 0.5 and above 0.75); cross-class IoU above 0.9 (duplicates across classes); relation of stacked pallets (shared x-extent, adjacent y).
- Expected: heavy-tailed sizes with many small load carriers and stacked pallets; same-class overlaps in stacks that break standard NMS at 0.5 (so duplicate suppression must be learned/fused per class and not applied blindly); forklifts large.

D4. Image properties
- Run: width/height distribution; aspect ratios; decoded size vs csv `width`/`height` for every train image; EXIF orientation presence; whether boxes were drawn in the displayed or the stored orientation (check by verifying that decoded size equals csv size after the chosen orientation handling and that boxes of a distinctive class, e.g. forklift, sit plausibly on a train image when plotted, train only).
- Expected: large photographs (possibly several thousand px), mixed orientations; decide one fixed orientation handling and assert it on all train images.

D5. Group structure and near-duplicates
- Run: perceptual hash (dHash on 64x64 grey, implemented in numpy) of every train image; union-find merge of images with Hamming distance below a small constant AND of train groups that contain such near-duplicates; compare merged groups with provided `group_id`.
- Expected: provided `group_id` is already the right unit; merging may join a few groups. Within a group, many views of the same pallets and racks, which makes random splits very optimistic.

D6. Annotation noise ceiling at IoU 0.75
- Run: jitter reference boxes by realistic noise (each coordinate perturbed by 2%, 4%, 6% of the box side, several seeds, applied to reference-as-prediction) and score with the exact metric; split by size bucket.
- Expected: the achievable F1 at IoU 0.75 drops steeply with small box size; this sets the ceiling and shows where review helps most. Interpretation: if a 4% jitter already halves F1 for small objects, a detector cannot score high on them and review windows should concentrate on them.

D7. Metric oracles
- Run: gold boxes as predictions with no review must give exactly 1.0 for every warehouse; gold with classes permuted must give 0; empty must give 0 on any non-empty warehouse; gold with one spurious class box in an absent class must lose the expected amount; gold minus 20% of boxes. Then, per image, the oracle window (best of the grid with the gold boxes) to estimate the upper bound of review gain on an empty detector and on a noisy detector (jittered gold); also a random window and the fixed-centre window.
- Expected: oracle review gain over an empty detector is mostly determined by object density and size, and the fixed-centre window is clearly worse than the oracle; the real policy sits between random and oracle.

D8. Candidate-pool coverage (after the first detector baseline, OOF)
- Run: oracle recall at IoU 0.5 / 0.75 of the pool at score threshold 0.01 (top 100 per image) by class and size; share of misses due to wrong class vs poor localisation vs no box.
- Expected: recall at 0.5 is considerably higher than at 0.75; wrong-class confusions between pallet and stillage and between small load carrier and pallet are the main class errors.

D9. Leak suspects
- Run: correlation of box counts or class mix with `id`, `image` string, row order, file size and `group_id` in train; check that ids and filenames carry no label information (a tell would be perfectly ordered class blocks).
- Expected: opaque ids, no leakage; this check only documents that none of these are used.

D10. Throughput probe (train images, planning only)
- Run: decode + resize time per image at 1024 (fixed canvas) and 1280; forward/backward time and peak memory for the chosen detector at batch 4/8 with AMP; inference time with flip TTA.
- Expected (estimate): about 5-8 train images/s at 1024 for a ResNet-50-class DETR-style detector with AMP on an A10G; decide epochs from this.

## Validation design

**Reproducing the test split from the description:** test warehouses are disjoint from train warehouses. Therefore group hold-out is the correct axis and magnitude: whole warehouses held out, with warehouse-level pooling in the metric. Random image splits would be strongly optimistic because several photographs depict the same physical objects.

**Folds**
- Groups: provided `group_id` merged with train-only near-duplicate union-find (D5), unconditionally.
- K = 3 grouped folds (K = 4 if train has at least about 24 warehouses and the runtime plan allows). Stratify folds greedily so each fold contains warehouses with each rare class (forklift, pallet truck) and balanced image and box counts; this keeps per-class F1 defined in every fold.
- The detector is trained 3 times (one per fold) in the shipped script, so the OOF detections that calibrate and score everything are produced in-script and from the same recipe that ships.
- Detector retraining is not repeated across seeds (compute). Cheap downstream stages (calibrator, thresholds, review constants) are repeated over 3-5 different fold-assignment seeds of cross-fitting and over bootstrap resamples of warehouses, and reported as mean and std.

**Metric implementation:** re-implement the full metric exactly as described (window removal, per-class maximum matching at IoU >= 0.75, warehouse-class pooling, average over classes present on either side, empty cohort 1, mean over warehouses). Unit tests: perfect gold (1.0), all-empty on non-empty data (0), classes permuted (0), constant full-image box of one class, base-rate guess (class prior boxes at random locations), a case where window area is exactly 0.10, a box exactly touching the window edge (contained), a box straddling the window (not contained), a spurious class in an empty class (drops average), a warehouse with no references and no predictions (scores 1), and reversal sanity (swapping prediction and reference roles gives the same F1 because the formula is symmetric in predicted/reference when no review is used). Report the warehouse-bootstrap standard error as the noise level.

**Nested/cross-fitted post-hoc steps:** every post-hoc selection needs its own held-out check, so
- Calibrator (match-at-0.75 probability): fit on OOF candidates of two folds, apply to the third; repeat; report the cross-fitted score.
- Class thresholds, duplicate-suppression IoU, window constants (see decode): chosen on OOF from the two training folds' warehouses, scored on the third; the shipped constants are refit on all OOF.
- Hold back at least one fold-seed pattern (not used for any decision) as the "sanity fold" for the final reported number when the group count allows.

**Accept rule:** a change counts only if it beats the warehouse-bootstrap noise and is paired on identical folds. Stage-by-stage reporting (scores both before and after review): (a) detector AP at 0.75 per class; (b) pre-review metric after thresholds; (c) post-review metric; (d) oracle-window and random-window and central-window references.

**Bias of each proxy (direction, reason):**
- Fold models train on about 2/3 of the warehouses, so OOF is pessimistic relative to a model trained on 100% of train. The shipped three-model ensemble sees all warehouses across its members and is somewhat better than a single fold model, so the OOF proxy is mildly pessimistic (small, a few points at most, ESTIMATE).
- Thresholds, calibrator and window constants are tuned on the same OOF; cross-fitting removes most of this optimism. Residual optimism is small if the number of constants stays below about eight.
- Warehouse scores from few warehouses are very noisy and skewed by rare-class presence; the private warehouses may have different class mixes, so expect the real number to be volatile in either direction. Net band is stated in Open questions.
- Test-time TTA and ensembling are not visible in OOF with a single fold model, except flip TTA which is applied identically; the ensemble effect is pessimistic in OOF.

## Overfit/underfit risks

**Overfit**
- Few warehouses (effective N) and many near-identical photographs: the detector can memorise warehouse backgrounds and layouts. Mitigation: grouped folds; strong photometric and scale augmentation; weight decay and EMA; fixed short schedule; lower learning rate on the pretrained backbone; log train loss vs held-out AP at a few fixed checkpoints in development and fix the epoch count as a constant.
- Many tuned knobs on a tiny warehouse count. Mitigation: at most about eight fitted constants (five class thresholds regularised toward the algebraic F*/2 value, one fusion IoU, one window shrinkage constant, one containment softness), smooth grids, cross-fitting, shrink toward the algebraic default when the cross-fitted gain is within the bootstrap noise.
- Calibrator learned on single-model OOF but applied to ensemble outputs: see the underconfidence bias note under the approach; mitigated by applying the per-model calibration before fusion and by keeping thresholds conservative.
- Absent-class penalty tuned on a handful of warehouses can overfit to which warehouses lack a class. Mitigation: bootstrap over warehouses and choose the smoothest threshold plateau, not the sharpest optimum.
- Per-image class-presence logic must not become a hand rule; keep it as one trained feature of the calibrator.

**Underfit**
- Resolution: small load carriers and stacked pallets can be only tens of pixels in a large photograph; down-sizing to 640 would lose them and IoU 0.75 needs precise edges. Mitigation: fixed canvas at 1024 long side (1280 as a roadmap item tested only if D3 shows many objects under 16 px), flip TTA, optional tiling at inference (per image).
- Too few epochs: DETR-family detectors need many epochs; with a 90-min budget and a 3-fold design, epochs may be limited. Mitigation: start from the COCO-trained detector (heads included), freeze nothing except BN and early stem, use a fixed schedule with warmup and cosine, and profile before fixing counts; fallback design below if throughput is too low.
- Loss not aligned with IoU 0.75: use the detector's IoU-aware classification (varifocal/quality focal) so scores track IoU, and raise the box-quality term weight as a roadmap experiment.
- Class imbalance and warehouse imbalance starving rare classes: see sampling weights below.
- Information thrown away by preprocessing: no centre-crop; keep the full field of view with padding; clip boxes only at the image border.

## Recommended approach (primary + fallback)

**Primary: COCO-pretrained transformer detector fine-tuned end-to-end on grouped folds, learned per-box calibration, expected-utility decode, expected-gain review window.**

1. **Detector.** A Hugging Face `transformers` real-time DETR-class detector, ResNet-50 variant, plain COCO-pretrained checkpoint (the RT-DETR / RT-DETRv2 family), full fine-tune with 5 classes (re-initialise only the classification head), AMP, fixed 1024 canvas, batch size fixed (about 8, UNVERIFIED memory), AdamW with a lower LR on the backbone, warmup + cosine, EMA of weights, grad clip 1.0. Why: transformer detectors output one box per object without NMS (stacked, overlapping same-class objects are not destroyed by NMS), classification is IoU-aware, a strong COCO prior is available, and the HF weights satisfy the "weights only from HF/timm" rule. Exact checkpoint name, revision and class availability in the installed `transformers` version are UNVERIFIED and must be checked first at implementation.
2. **Training data policy.** Sampling weight per image = product of (a) warehouse balancing weight proportional to `n_images_in_group ** -0.5` (a compromise between image-level and the metric's warehouse-equal averaging; design constant, not test-tuned) and (b) repeat-factor weight for images with rare classes (LVIS-style, threshold chosen as a fixed design constant). Augmentation: horizontal flip, scale/translate jitter (multi-scale, keeping boxes with at least a fixed visible fraction), photometric jitter, blur/JPEG-quality jitter for blurred objects. No vertical flip, no rotation beyond a tiny angle, no copy-paste or fabricated composites (synthetic-data risk).
3. **Fold ensemble.** Three fold models give OOF detections for calibration and policy fitting; the shipped test prediction fuses the three fold models plus flip TTA for each image (weighted boxes fusion within one image, same class). No refit on 100% of data because compute does not allow it (documented departure from the refit default; roadmap step tests a 100% refit only if time remains).
4. **Per-box calibrator (second stage, trained, within-row features).** Candidate boxes (low threshold, fixed top-100) from OOF are labelled by whether they match a same-class reference at IoU >= 0.75 (and also at IoU >= 0.5 for an "object-present" head). A small logistic/GBDT model on: detector score, class, log box area and aspect, relative rank within the image and class, count of same-class candidates above fixed scores in the image, max IoU with a higher-scored same-class box, distance to border, fusion agreement between TTA views and between ensemble members (variance of box coordinates, a free signal of localisation uncertainty), image-level class mass (own image only). Output: `q` = P(automatic match at 0.75) and `o` = P(object of that class exists at this location at IoU >= 0.5). This also supplies the class-presence signal for the spurious-class penalty.
5. **Decode (expected utility, section Metric-aware).** Class thresholds as a function of per-class F* (fixed point), regularised by an OOF search over the five thresholds.
6. **Review window (expected gain).** Enumerate windows of area 0.099 on a fixed grid of aspect ratios and positions, score each by expected gain using `q`, `o`, soft containment, class F* and a train-fit shrinkage, pick the best. Per image only, no cross-image terms.
7. **Strip-the-ML test:** removing the detector leaves nothing; the calibrator, thresholds and window policy are small learned or train-fit components wrapped around the detector. The detector fine-tune is load-bearing (the parameter-change norm and the AP gain over zero-shot COCO head initialisation are logged in development).

**Fallback (if the transformer detector is too slow, unstable or unavailable in the installed `transformers`):** `timm` ImageNet backbone (ResNet-50 or a small ConvNeXt, weights from the HF hub via timm) plus a torchvision Faster R-CNN with FPN or FCOS head (heads initialised from scratch, that is the cost: no COCO detection prior because torchvision's COCO weights are hosted off HF/timm), same 3-fold design, same calibrator, decode and review stages, standard class-wise NMS tuned on OOF. Weaker on small data, but all components are compliant. A second, cheaper fallback shape: one grouped hold-out split (about 20% of warehouses) with a single detector trained on the rest, calibrator and policy fitted on the hold-out only, shipped as that single model (noisier calibration, lower cost, no ensemble-calibration mismatch).

**Why not a bigger or fancier primary:** the dominant levers are (1) a strong COCO-pretrained detector at enough resolution, (2) correct grouped validation and exact metric, (3) rare-class precision under the spurious-class penalty, (4) a principled expected-gain review window. Extra members (second architecture, tiling, 1280 px) enter only after the lean design is measured and show a gain larger than the warehouse-bootstrap noise.

## Rejected options

- **Ultralytics/YOLO families:** weights come from GitHub (banned source) and the package is not guaranteed; also non-COCO-only variants may be industrial.
- **torchvision detection weights from download.pytorch.org:** not Hugging Face or timm; grey under CLAUDE.md; used only through a timm backbone in the fallback.
- **Objects365, LVIS-extended or foundation checkpoints (CLIP, DINOv2, SAM, Grounding-DINO, OWL-ViT):** the description whitelists only COCO and ImageNet general-purpose weights; open-vocabulary detectors would also approach inference-only use.
- **Cross-image logic of any kind** (merging detections across photographs of a warehouse, warehouse-level class-presence priors from test, group-level thresholds on test, test-batch normalisation, ranking test images to place windows): banned.
- **Hand-written rules** (size filters, class-specific box shapes, fixed window at the densest area computed from pixel statistics, regex on filenames): violates the strip-the-ML test and the ban on rules replacing the model.
- **Fixed central or fixed best-on-train window as the review policy:** ignores per-image content; used only as a reference baseline in validation.
- **Pseudo-labelling, test-time adaptation, self-training on test images:** banned.
- **Full-data single model without OOF:** leaves thresholds, calibration and review policy without train-only evidence; rejected as primary.
- **Standard class-wise NMS at 0.5 for DETR-family output:** unnecessary for the primary and would damage stacked same-class objects; used only in the fallback.
- **Copy-paste and fabricated composite augmentation:** risk of being classed as synthetic data.
- **Large ensembles of many architectures:** budget and compliance noise; diversity of assumption can only be added after the lean pipeline is measured.

## Fixed work plan & runtime budget

Assumptions (UNVERIFIED): about 1,500 train images, about 600 test images, about 1024 px long side after resizing, A10G, DETR-class R50 detector with AMP at roughly 6 images/s training. Fixed counts must be set after the D10 profiling run and recorded as constants; none depend on wall-clock.

| Stage | Fixed work | Estimate (A10G) |
|---|---|---|
| 0. Read CSVs, schema checks, group build (union-find on dHash) | 1 pass | 1-2 min |
| 1. Weight download (HF), image decode and fixed-canvas resize cached in RAM (uint8) | train + test once | 3-4 min |
| 2. Train 3 fold detectors | 3 folds x E epochs (E set by profiling, about 8-12 under the assumptions), batch 8, workers fixed, seed fixed | about 30-34 min total (about 10-11 min per fold) |
| 3. OOF inference with flip TTA, pool at score >= 0.01, top 100 per image | all train images | 3-4 min |
| 4. Calibrator, thresholds, fusion IoU, review constants (cross-fitted, CPU) | fixed grids, no timeouts, `n_jobs=1` | 3-4 min |
| 5. Test inference: 3 models x flip TTA, fusion, calibration, threshold, window | all test images | 3-5 min |
| 6. Write, re-read, validate against `sample_submission.csv` | 1 pass | under 1 min |
| Total | | about 48-55 min, against 90 min limit: about 40% headroom |

Memory: uint8 cache of 1,500 train images at 1024 x 1024 x 3 is about 4.7 GB RAM (plus test about 1.9 GB), far under 62 GB. GPU: AMP with batch 8 at 1024 for an R50 DETR-class model is about 12-18 GB (ESTIMATE); fix batch size with headroom, use gradient accumulation if the profile says otherwise, free GPU memory between folds.

Determinism: seeds for `random`, `numpy`, `torch`, CUDA, `PYTHONHASHSEED`, DataLoader `Generator`, fixed `num_workers` constant, `torch.backends.cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `device = "cuda"`, no `is_available()` switches, no `cpu_count()`-derived values, no try/except import fallbacks, no time-conditioned control flow (time only inside `print`). Bipartite matching and window enumeration use fixed grids and deterministic tie-breaking (lowest index). The script embeds an in-script validator and re-reads the written CSV.

If throughput at 1024 is too low to afford E >= about 6 epochs per fold, the lever order is: lower resolution to 896, then K = 2 folds, then the single hold-out fallback; each is a fixed constant decided at development time, not a runtime branch.

## Metric-aware training & decode

**Training**
- Metric reconstruction target: the metric only counts matches at IoU >= 0.75 and per-warehouse pooled F1, so the detector should output boxes whose score tracks IoU (varifocal/quality focal classification) and boxes should be tight. Roadmap experiment: increase the GIoU/L1 weights relative to the default and compare OOF AP at 0.75 paired on the same folds.
- Hierarchical weighting: image sampling weight described above, replicating "warehouse first, class next" averaging without extreme weights; verify on OOF that it helps warehouse-averaged score, not just pooled AP.
- Rare classes: repeat-factor sampling for images that contain forklift or pallet truck; also check per-class AP separately and do not smooth labels.
- Horizontal flip is allowed; check on train that class counts are flip-symmetric (they are by construction) and apply matching test-time flip averaging.

**Decode (expected utility, stated as derived formulas, constants fitted in-script)**
1. Fusion: within each image, weighted boxes fusion of flip views and fold members, same class, with a fusion IoU constant chosen on OOF. Do not suppress same-class overlaps beyond a very high IoU (stacks); cross-class suppression only above a very high IoU and only as an OOF-selected constant, after the D3 diagnostic shows how often different classes legitimately overlap.
2. Inclusion rule: include candidate `i` of class c if `q_i > tau_c`. Starting value from the F1 fixed point: tau_c = F*_c/2 where F*_c is the OOF warehouse-pooled class F1 reached at the optimum (iterate to the fixed point, do not grid-search without a bound). Then a bounded 1-D search per class, lower-bounded by the label prior so thresholds cannot reach degenerate corners (including the corner "predict nothing"), upper-bounded below 1, regularised toward the fixed-point value, cross-fitted.
3. Spurious-class penalty: add to the rare-class decision an expected-cost term using a trained estimate of class presence in the image (feature of the calibrator from `o` mass), so a lone low-confidence forklift is dropped unless the class is probably present. Quantify on OOF how much warehouse-averaged score the penalty term recovers; if the gain is within noise, drop the term and keep the per-class threshold only.
4. Duplicates and box refinement: ensure no pair of same-class submitted boxes has IoU above the fusion constant; averaging of coordinates across TTA views and fold models is the localisation lever for IoU 0.75.

**Review window**
- Candidates: windows of area 0.099 (fixed constant strictly below 0.10) on a fixed grid of aspect ratios (width/height in a fixed geometric ladder, with width and height both at most 1) and positions (grid step as fraction of window size, plus positions snapped to candidate box edges). Exhaustive, vectorised, deterministic.
- Score of a window W: `sum over classes c of w_c * [ (2 - F*_c) * E[FN_in(c,W)] + F*_c * E[FP_in(c,W)] ]`, where for submitted candidates inside W `E[FP_in] = sum (1 - q_i)` over submitted boxes, and `E[FN_in] = E[reference objects inside W] - E[matched inside W]`, using `o` over the low-threshold pool for expected reference objects and `q` for matches, and soft containment: each box edge treated as Gaussian with sigma proportional to the box side with the proportionality fitted on OOF matched pairs (train-only), which prices the chance that a reference edge falls outside the window.
- Class weights `w_c = 1 / (n_c_expected_in_image + lambda * n_c_bar)`, where `n_c_bar` is the train per-image mean count of class c (a train constant) and `lambda` is chosen by the exact metric on OOF, cross-fitted. This is the within-image proxy for `1/D_c` in the linearisation because warehouse-level counts are not available at test without cross-image pooling (which is banned). The common warehouse factor (number of images, number of classes present) does not change the arg-max over windows inside one image.
- Output: the best window, with coordinates rounded to 6 decimals, area asserted below 0.10, boundary rules respected; empty review `[]` only if the best expected gain is at most zero (it essentially never is, since a window can only help).
- Validation of the policy against the references: oracle gold-window, random window, fixed-centre window, and no-review; the policy should beat random and centre by more than bootstrap noise or it is replaced by the simpler of the two.

## Structural signals

Invariants and free signals (each turned into an augmentation, constraint, feature or reference):
- Horizontal flip symmetry of the scene: augmentation and TTA. Vertical flip violates gravity and perspective: banned in augmentation.
- Perspective and gravity structure: lower in the image means nearer and larger, pallets sit on the floor or racks, small load carriers sit on pallets or in stillage; learned by the detector, and exposed to the calibrator through relative position and size features within the image.
- Stack geometry: stacked pallets share near-identical x-extent and have vertically adjacent boxes; these relations are legitimate within-image features for the calibrator (own-image neighbours, not cross-image) and explain why box-edge agreement across TTA views is an informative localisation uncertainty feature.
- Box-edge uncertainty from view/member variance: free signal for the soft-containment sigma and for `q`.
- Review geometry: gain is monotone in window size so area 0.099 always; best windows are dense clusters of small, hard or uncertain objects; objects larger than the window can never be reviewed (so large forklifts and stillages rarely benefit).
- Boxes inside the window are free: any thresholding or duplicate removal inside the window is irrelevant, so window choice should be made before the inclusion rule is evaluated for boxes outside it.
- Metric pooling structure: warehouse-and-class pooled counts; spurious class penalty; empty cohort scores 1 (so an image with no objects and no detections contributes nothing).
- Annotation convention: "annotation catalog can omit ambiguous or barely visible objects" means some real objects are unlabeled; the calibrator learns how likely small, blurry, truncated detections are to be annotated (to be verified with D3/D8, not hard-coded).
- Same physical objects in several photographs of a warehouse: a leakage source for random validation (handled by grouping) and a tempting cross-image prior at test (banned, not used).

## Experiment roadmap

1. **Contract, metric, validation (stop when all unit tests pass and gold = 1.0):** implement the exact metric, JSON/validation functions, grouped folds, near-duplicate merging; run D1-D7 on train; record oracle review gains. Record the warehouse-bootstrap noise.
2. **Strongest cheap baseline, end to end and valid:** one grouped split, one fine-tuned detector at 640-800 px for a few epochs, fixed score threshold 0.5, no review, plus a second submission variant with fixed-centre window. Check validator and CSV. This is the first credit (baseline).
3. **Representation and structure:** move to 1024 (and 1280 if D3 shows many tiny objects) and the warehouse/class-balanced sampler; compare paired on the same folds. Stop when gain is within noise.
4. **Metric-aware decode and review:** calibrator (`q`, `o`), fixed-point per-class thresholds, spurious-class term, expected-gain window; each step compared against the previous one cross-fitted. Add the oracle checks: gold candidate scores through the decoder must reproduce the gold set; gold through the window enumerator must pick an oracle window.
5. **Diversity:** fold ensemble plus flip TTA (already in the primary); then, only if each is measured against bootstrap noise, a second backbone size or multi-scale TTA or tiling; fuse by WBF per image. Score each family alone first.
6. **Bounded constant search in-script:** the handful of constants (thresholds, fusion IoU, window shrinkage, softness) with fixed grids and no timeouts, scored with a nested held-out check. No HPO of the detector beyond a fixed recipe; any epoch or LR choice is made once in development on train grouped CV and recorded as a fixed design constant (see open question 3).
7. **Final fixed-plan run:** clean `working/`, run twice, diff the two submissions (expect near-equal; investigate if boxes differ materially), check output distribution vs OOF (boxes per image, class mix, window area distribution, share of images with review), then upload. Credits: baseline, best single, ensemble, final.

Stop rules: if step 2 shows the pooled OOF warehouse score is below about 0.2, re-inspect D4 (orientation or coordinate convention) and D6 before spending compute on anything else.

## Compliance audit

Against CLAUDE.md section 7 and the Eris section B self-audits:
- Reads test only for one-image-at-a-time inference? Yes by design; plus the independence unit test (batch 8 equals batch 1). `group_id` of test is never read.
- Any `time`/`elapsed` in conditions or arguments? None planned; telemetry only.
- `cuda.is_available()`, `cpu_count()`, import fallbacks? None; fixed `device = "cuda"`.
- Hard-coded constants tuned offline? Class thresholds, fusion IoU, window constants, calibrator are fitted in-script from train OOF. Remaining hand-fixed constants are design constants: window area 0.099, canvas size, top-100 candidates, pool score floor 0.01, sampler exponents, epochs, LR, batch size. Epochs/LR/resolution come from train grouped-CV development runs (never test or leaderboard); flagged as a reviewer question because section 7 forbids offline-tuned constants while 4A allows fixed step counts derived from CV.
- External data, synthetic data, self-hosted weights, GitHub models? None. Weights only from HF/timm, COCO or ImageNet only, revision pinned.
- ML removed would still work? No. All detections come from the trained detector; there is no rule engine.
- Whole-test aggregation? None: no rank/z-score across test, no vocabulary/encoder/scaler/PCA on test, no test-set prior estimates, no cross-image NMS or dedup, no batch statistics.
- Sibling leakage: the calibrator uses only own-image candidates; all fitting uses grouped folds so photographs of one warehouse never straddle train and validation.
- Strip-the-ML: passes. Parameter-change norm and OOF gain over the COCO-initialised head probe logged in development to show the fine-tune is load-bearing.
- Description-specific bans: see table; none violated by design. Window selection is by trained policy, not by looking at test images.
- Source readability and size: plain Python under 512,000 bytes, comments for each step.
- Output validation: validator asserts columns, order, ids, row count, JSON parse, class ints 0..4, coordinate ranges and strict inequalities, review area at most 0.099, no NaN; re-read the written file with `keep_default_na=False`.

## Open questions & assumptions

**Reviewer questions (one line each, with plan under each reading):**
1. Is a COCO-only Hugging Face detector checkpoint (RT-DETR/RT-DETRv2 family) acceptable as "general-purpose COCO pretrained weights"? If not, use the fallback (timm ImageNet backbone with torchvision heads, from scratch heads). Is a checkpoint also trained on Objects365 allowed? Plan assumes no, and costs some performance.
2. Is selecting the review window by expected gain from the detector's own per-image outputs (policy constants fitted on train OOF) compliant? Plan assumes yes; the alternative reading (any detection-conditioned policy is metric exploitation) would reduce review to a fixed or random window and costs the review gain (estimated a few points up to about ten, per D7 on train).
3. Are training-recipe constants (epochs, learning rate, resolution) chosen offline from train-only grouped CV acceptable as fixed design constants? If not, the in-script alternative is a fixed standard recipe with no tuning, with a small cost.
4. Is flip TTA plus weighted fusion of three fold models, all applied to one image at a time, acceptable under "no combining information across test images"? Plan assumes yes (same image only).
5. May the script read width/height from the CSV or from the decoded image? Plan reads the decoded image and asserts equality with train CSV values on train only.
6. Does the mention of installation in the 90-min budget allow `pip install`? Plan assumes no (CLAUDE.md).

**Unverified items:** every number in Data findings; the availability, name and revision of the RT-DETR checkpoint and class in the installed `transformers`; throughput and memory; train/test image counts; number of warehouses; image sizes; whether stacked-same-class overlap is common; whether provided group ids are already the correct merge unit.

**Assumed vs silent:** compute cap (A10G 24 GB, 62 GB RAM, 90 min) is explicit; CPU count, test size and disk are silent and left parameterised; confidence thresholds on test are not tuned; no image-level cap on detections beyond 300; deterministic grading assumed to rerun the script (so two-run diff is part of the roadmap).

**Expected result (ESTIMATE, low confidence, nothing verified):** OOF warehouse-averaged score before review about 0.30-0.50, review adding about 0.03-0.10, so a private score in roughly 0.30-0.55. Reasoning: IoU 0.75 is strict, a COCO-pretrained detector fine-tuned on a handful of warehouses typically generalises imperfectly to unseen warehouses, rare-class warehouse averaging is volatile, and review windows can fully fix only small clustered objects; the OOF proxy is mildly pessimistic (fold models see 2/3 of warehouses) and mildly optimistic from residual constant tuning, so the net bias is small relative to the very large warehouse-level variance.
