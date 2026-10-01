# U20 plan: farm behavior-state frames (goat pen, 3-class macro-F1)

Status of evidence: no dataset files were available. Everything under "Data findings" is a diagnostic to RUN plus a hypothesis, all UNVERIFIED. The description file began at "Overview" (no header or runtime section was supplied), so runtime and compute are assumed from CLAUDE.md. Numbers for N, runtime and score are estimates, not measurements.

## Contract & decision unit

- One valid answer: `submission.csv` with columns `id,prediction`, in that order, exactly one row per `id` in `test.csv`, no extras. `prediction` is an integer in {0,1,2}. Missing, non-finite, non-integer or out-of-range values are invalid (dead last, below 0). A low score is not the same as an invalid file.
- Classes: 0 = one visible behavior state, 1 = exactly two distinct states, 2 = three or more distinct states. This is a count-of-distinct-categories target, capped at 3, over the animals in one overhead frame. It is ordinal (0 < 1 < 2) and monotone in "how many animals, and how varied".
- Metric: `f1_score(y_true, y_pred, labels=[0,1,2], average="macro", zero_division=0)`. Term by term: each class F1 gets weight 1/3. A class never predicted contributes 0. Constant prediction of class c with prior p gives (1/3)*2p/(1+p). It is hard-label, so calibration only matters through the decision rule, not the ranking. The middle class (exactly two states) is sandwiched between two neighbours and is probably the weakest F1 and the main lever.
- Independent unit: not the row. Test is built from complete 100-frame index buckets (every third bucket), with transformed duplicate groups kept together. Train therefore very likely contains (a) near-identical adjacent frames (static pens) and (b) transformed copies of the same source frame. The effective sample size is the number of independent buckets/scenes, not the number of images.
- Pipeline stages to diagnose separately: (1) representation: does the backbone resolve small, partly overlapping animals at 384x288 after blur; (2) head: does the pooling/scoring head turn dense evidence into a group-level class; (3) decode: macro-F1-aware decision constants. No candidate-coverage stage exists.
- Not a ranking task and no hierarchical averaging: the score is plain macro-F1 over the whole test set.

## Compliance regime

Domain: computer vision, image classification. The challenge is not labelled "from scratch" and not labelled "fine-tuning" in the text I have. Regime: CLAUDE.md section 6.3 plus the explicit bans below.

Explicit bans in the description (hard constraints):
1. Do not recover the source COCO annotations or original behavior labels via a public mirror, filename search or any internet lookup. Internet is used only for HF/timm backbone weights.
2. Do not use the source archive's original train/val/test membership, source filenames, or annotation IDs as predictive features. Consequence: `id` and the `image` path string are never model inputs, never hashed, never parsed. They are used only to open the file and to write the submission.
3. Do not build a lookup table from released image files, and do not manually assign hidden test classes. Consequence: no pHash/exact-match/kNN-to-train-label lookup as the predictor, and no hand labelling of any test frame.
4. "Keep the model path end-to-end ... one script" and "Train an image model from the supplied public frames and use a validation split defined using training rows only."

Requirements stated: report macro-F1 AND per-class F1 in local validation (print both per fold and pooled OOF); write the final file to `./working/submission.csv`.

Where the description is silent, and my assumption:
- Runtime and GPU: silent. Assume CLAUDE.md: A10G 24 GB, plan for at most 50 min total, no wall-clock branching.
- Pretrained weights: silent, not labelled from scratch. Assume general-purpose HF/timm weights are allowed (CLAUDE.md 2.2). If a reviewer says from-scratch, fallback C below applies.
- Model family or size caps: silent. Assume none.
- Output path: description says `./working/submission.csv`; CLAUDE.md says read argv. Follow argv, defaulting to `./dataset/public` and `./working/submission.csv` only when argv is absent. Same file at the stated path in the normal run.
- Test-time use: silent; CLAUDE.md governs. Per-image inference plus per-image flip TTA only. No smoothing across neighbouring test frames, no test-fitted normalisation, no pseudo-labels, no class-proportion matching on test.

Mechanism compliance from CLAUDE.md:
- Never feed raw pixels to a tabular model. Model is a pretrained ViT/CNN fine-tuned (LP-FT). Heads on frozen pooled features are only a stage inside it and the grey "frozen embedding + head" configuration is not the shipped model.
- Avoid COCO-detection-pretrained backbones (DETR, Mask R-CNN) to stay clear of any association with the source COCO annotations; use DINOv2 / EVA / ConvNeXt weights (LVD-142M, ImageNet-22k).
- Strip-the-ML test: remove the trained backbone/head and nothing is left (no rules, no regex, no lookup). Passes.

## Data findings

All items below are diagnostics to run on train only (read-only, no training needed beyond a frozen probe) with the hypothesis I expect. UNVERIFIED.

D1. Shapes and schema. `train.csv` (id, image), `train_targets.csv` (id, target), join on id; assert 1:1 and no missing. Row counts of train and test (test: count only, for runtime planning). Hypothesis: a few thousand train images (unknown N; I plan for about 5k train and about 2.5k test). Image size uniformly 384x288 RGB JPEG.
D2. Target distribution. Exact class % in train. Hypothesis: class 0 (one state) is the largest, 1 and 2 smaller and "not identical"; roughly 40-55 / 25-35 / 15-30. Compute the majority-class and prior-constant macro-F1 floors.
D3. Duplicates and near-duplicates. Compute exact hash, and frozen-embedding (DINOv2 CLS and mean-patch) cosine similarity for all pairs, also against the horizontally flipped version (min over flip). Union-find at a cosine threshold chosen from the gap in the pair-similarity histogram (not a magic number). Report group-size distribution. Hypothesis: transformed copies (flip, brightness, small crop) of the same source frame exist, plus static-scene near-duplicates; group count is far below N. Check label agreement within groups: any disagreement is a direct measure of label noise / ceiling.
D4. Row-order structure. For consecutive train.csv rows, compare adjacent-row embedding cosine with the random-pair baseline; also autocorrelation of labels in row order. Hypothesis: if rows are in frame order, adjacent similarity is far above random and the sequence breaks into runs of about 100 per bucket (with 2 of every 3 buckets present, adjacent train buckets look continuous). If rows are shuffled (ids are opaque) this fails and content-only groups are used.
D5. Low-light structure. Per-image mean luminance, fraction of channel-identical (gray/IR-like) images, fraction below a data-driven luminance cut (valley of the luminance histogram). Class distribution inside dark vs normal; whether dark frames form contiguous runs. Hypothesis: night frames are a minority of train, are gray or low-contrast, and may sit in whole buckets. The description says the hidden set contains both ordinary and low-light, so a model that never sees dark frames fails; check train has enough dark frames (target at least a few hundred or at least 10 independent groups).
D6. Residual overlay or border leak. Per-edge-strip pixel variance across images. If a timestamp/overlay remnant survives the crop, it carries time and thus scene identity; mask a fixed thin border as a preprocessing step if so (a preprocessing constant seen in train, to be stated in a comment).
D7. Ordinal structure. Using frozen-probe OOF (grouped folds), build the 3x3 confusion matrix. Hypothesis: errors are almost all between adjacent classes; 0 vs 2 confusions rare. Supports a cumulative (ordinal) head.
D8. Animal-count proxy and ceiling. Without labels for count, report probe accuracy as a function of an image-level proxy (mean patch-token norm, or foreground token fraction). Hypothesis: class rises with number of animals; many class-0 frames contain few or one animal, so part of the signal is "how many animals", which is legitimately image-derived. Information ceiling estimate: grouped-CV macro-F1 of the frozen DINOv2-L probe is the floor, not a ceiling; the ceiling is probably limited by blur (behavior states such as standing/lying/feeding need detail) and annotation incompleteness. Expect label noise to cap macro-F1 well below 0.9.
D9. Flip symmetry check. Agreement rate of frozen-probe predictions between an image and its horizontal flip, and vertical flip, under grouped CV. Hypothesis: horizontal flip invariant; vertical flip may break if shadows/floor give direction. Rotations by 90 degrees are excluded (changes the 4:3 aspect).
D10. Leak tells. Any column other than id/image/target in the csvs; any id-vs-label association (check only to know it must not be used; the model never receives ids); any near-perfect single-feature predictor (luminance, file size) which would signal a bucket-confounded shortcut. Hypothesis: brightness/file size correlates with class through bucket confounding (night buckets differ in class mix); this is the shortcut that grouped CV will punish.

Test-side use limited to schema, row count, id format and image dimensions for runtime planning. No test image statistics.

## Validation design

Reproducing the split: test = every third complete 100-frame index bucket, transformed duplicate groups kept together. So test buckets are interleaved in time with train buckets, and adjacent train buckets sit on both sides of every test bucket.

Scheme A (primary, mirrors deployment):
- Build groups from train only: union-find of (i) near-duplicate/transformed-copy links from D3 and (ii) if D4 confirms rows are frame-ordered, contiguous blocks of about 100 consecutive train rows (bucket proxy). Merge both into group ids.
- Assign groups to 5 folds, interleaving blocks (block index mod 5, then re-balanced on class counts) so that every held-out block has train blocks on both sides, as in the real split. Folds are stratified on the class mix of groups (StratifiedGroupKFold-style, seeded).
- 5 folds minimum; 3 split seeds for the cheap frozen-probe stage, 1 seed for the LP-FT stage (cost) with the finalists re-run on a second seed in development. Report mean and std of fold macro-F1 and per-class F1, plus pooled OOF.
- Bias direction: slightly optimistic. Frames at bucket edges are near-adjacent to train frames across the boundary in the real test too, so that part matches; any scenes recurring across non-adjacent buckets (a static camera revisiting the same animals' arrangement) also leak equally in both. Residual optimism is expected to be small (a few F1 points). Pooled-over-folds OOF F1 is also a little optimistic from decode constants fitted on it, handled below by cross-fitting.

Scheme B (pessimistic sanity check, development only):
- Group by scene clusters (connected components at a looser similarity threshold, or contiguous runs of several hundred rows). This removes more train neighbourhood than the real split does, so it is pessimistic by design. Use it to rank candidates by robustness; large drops between A and B for one candidate signal scene memorisation (LP-FT or latent heads that overfit).
- Also report OOF macro-F1 within the dark subset (D5) and the normal subset separately. A candidate that wins overall but loses on dark frames is rejected unless train has too few dark groups to resolve it.

Holdout sanity set: before the final run, set aside about one sixth of the groups (whole buckets) that never touch hyperparameter choice, blend, or decode-offset fitting; evaluate the single finalist once on it. In the shipped script all data is used and the cross-fitted decode check stands in for it.

Nested/cross-fitted selection steps: head regularisation (per output), class-weight exponent, ordinal vs softmax head, and decode offsets each get their own check by choosing them on the other 4 folds' OOF and scoring on the 5th.

Metric tests (before anything else): implement `macro_f1` exactly as in the description with `zero_division=0`; unit-test on perfect predictions (1.0), reversed labels (0 and 2 swapped, mid kept: <1), constant class (equals (1/3)*2p/(1+p)), and base-rate random predictions (about 1/3 or lower). Also assert the all-three-classes-predicted property on OOF.

Acceptance rule: accept a change only when paired fold-wise gain on the same folds exceeds one standard error and holds in at least 4 of 5 folds and on the second split seed; compare with fixed stopping constants (no validation-triggered stopping).

## Overfit/underfit risks

Overfit:
- Few independent groups. The count of distinct scenes/buckets is likely tens to low hundreds. Mitigation: capacity ladder (frozen then linear head then LP-FT of last 2 blocks only), per-output regularisation grid with a very wide range, group-level sample weights (weight 1/size of duplicate group so repeated frames do not dominate), no full fine-tune.
- Scene memorisation / shortcut features: brightness, camera position, number of background objects correlating with class through bucket confounding. Mitigation: grouped CV by bucket (Scheme A) and scene (Scheme B); brightness/contrast/gamma and grayscale augmentation; log dark-vs-normal OOF.
- Duplicates inflating CV: handled by groups; transformed copies in different folds would leak, so use D3 groups, not only row blocks.
- Tuned knobs: decode offsets (2 numbers), class-weight exponent (1 number), head C (per output). All low-dimensional, cross-fitted, and dropped if the cross-fitted gain is not positive across folds.
- Over-search: no big HPO. At most a small fixed grid in script; many trials on a small noisy CV select noise.

Underfit:
- Resolution/representation: animals are small and the images are already blurred and downsampled to 384x288. Downsizing to 224 would throw away what is left. Mitigation: keep native resolution (resize to 392x294, exactly 4:3 and a multiple of the 14-pixel patch, a 2% upscale) and use a large dense backbone (DINOv2-L/14), pool richly (CLS, mean of patch tokens, GeM/log-sum-exp over patches), flip-average.
- Global pooling can hide small distinct cues; the latent-presence head (below) scores every location before pooling.
- Loss mismatch: plain cross-entropy favours the common class while macro-F1 weights classes equally. Mitigation: class-balanced loss weights (exponent chosen on OOF from a fixed small set) and OOF-fitted decode offsets.
- Ordinal middle class: cumulative heads give the middle class two chances; evaluate vs softmax.
- Too little LP-FT: log the parameter-change norm and the CV gain over the probe; if the norm stayed near 0 or the gain is under noise, the fine-tune is just compliance and the linear probe is the honest rung.

## Recommended approach (primary + fallback)

Primary: DINOv2 ViT-L/14 (HF `facebook/dinov2-large` or timm `vit_large_patch14_dinov2.lvd142m`, revision pinned) at 392x294, LP-FT.
1. Preload all train images once into a uint8 tensor (about 330 KB each), no DataLoader workers, GPU-side augmentation with a seeded `torch.Generator` (removes worker nondeterminism).
2. Stage 1, frozen: extract pooled features for original and horizontal flip views (CLS, patch mean, GeM over patches). Fit per-output regularised heads (softmax and cumulative/ordinal) on grouped folds with a very wide C grid, strong end included; keep the lowest rung that wins. This is the floor.
3. Stage 2, LP-FT: warm-start the head from the probe, fine-tune only the last 2 transformer blocks plus head, tiny backbone LR (about 1e-5 as a literature default), head LR about 1e-3, weight decay about 0.05, 2 epochs, EMA or no EMA fixed, cosine schedule, bf16 autocast, grad clip 1.0. Augmentation: horizontal flip (plus vertical only if D9 supports), small random resized crop/scale (0.85-1.0), brightness/gamma/contrast jitter strong enough to cover dark frames, random grayscale, light unsharp/blur jitter. No label smoothing, no mixup/cutmix across images (mixing frames changes the state count and would create synthetic composition labels).
4. Class-balanced weights (exponent from a fixed small set chosen on OOF), group-size weighting.
5. Test: refit on 100% of train with the fixed plan, predict original + horizontal flip, average logits, apply cross-fit-validated decode offsets. Fold models are used only for OOF numbers and the decode constants.

Structured head variant (roadmap step 4, only if it beats the plain head beyond noise): K=6-8 learned latent state detectors on live patch tokens; per-location score, log-sum-exp pooled over locations to a presence probability p_k per detector; the distribution of the number of distinct present states is the exact Poisson-binomial over {p_k} (DP over K), mapped to classes {1 -> 0, 2 -> 1, >=3 -> 2}; trained by exact marginal likelihood of the observed class. About K x 1024 parameters, translation-equivariant, matches how the label is generated. State the parameter count in the code; it is a few-parameter learned layer, not a rule.

Fallback A (lean): frozen DINOv2-L pooled features, flip-averaged, per-output regularised heads (softmax or ordinal), class-balanced weights, decode offsets. Runs in roughly 10 minutes. Ship the heads as trained torch/sklearn models inside the script. Grey (frozen embedding plus head) but legitimate if LP-FT shows no gain over noise; keep LP-FT in the shipped path anyway if it does not hurt, for compliance.
Fallback B (diversity, later): a second backbone with a different pretraining assumption (EVA02-L/14 in timm, or ConvNeXt-L at 384 in22k) with the same recipe; blend logits by z-score or train-reference rank only when its standalone CV is close to the primary.
Fallback C (only if a reviewer says from-scratch): small residual CNN trained from scratch on 384x288 with heavy augmentation and ordinal head; expect clearly lower score; not a plan unless forced.

Why this ranking: the effective sample size is small, so strong pretrained dense features with a minimal trained layer dominate; resolution and backbone size are the cheapest lever against underfit; the latent head encodes the label-generation structure with few parameters.

## Rejected options

- Full fine-tune of ViT-L/B: memorises scenes with few independent groups (capacity ladder).
- Training from scratch: too little data; only a forced fallback.
- Detection/segmentation pipelines: no box labels; COCO-detection-pretrained models are avoided (association with the source annotations); also needs animal-level labels we do not have.
- kNN/lookup to train labels, pHash or exact-duplicate lookup as the predictor: banned ("lookup table from released image files") and fails the strip test.
- Temporal smoothing or label propagation among test frames, neighbour-bucket tricks, pseudo-labelling, test-time adaptation, class-prior matching on test: whole-test aggregation, banned (CLAUDE.md 2.3 #5).
- Mixup/CutMix across images: creates synthetic labelled compositions with undefined state counts.
- Downsizing to 224 or center-cropping: discards small animals.
- Using row order, `id`, `image` path text, or any source-derived name as a feature: banned.
- Deblurring/privacy-inversion models: out of intent and external weights; a mild unsharp-mask augmentation on the image itself is acceptable, a trained restoration model is not planned.
- Large HPO (Optuna with many trials): CV cannot resolve it; small fixed grids only.

## Fixed work plan & runtime budget

Assumptions (UNVERIFIED): N_train about 5k, N_test about 2.5k, A10G, ViT-L/14 at 588 tokens, about 80 img/s forward in bf16 and about 60 img/s for forward plus backward through the last 2 blocks. Profile one epoch and one trial locally on the real data, then hard-code the counts.

| Stage | Fixed plan | Est. time |
|---|---|---|
| S0 load and decode all images to uint8 | 1 pass, no workers | 1 min |
| S1 frozen feature extraction (orig + hflip), train and test | about 15k forwards | 4 min |
| S2 probe grid on cached features | 5 folds x 3 split seeds x about 6 C values x 2 head types, GPU torch/sklearn | 2 min |
| S3 LP-FT, 5 folds (OOF only, no test inference) | 2 epochs x about 4k imgs + OOF predict both views | 15 min |
| S4 LP-FT refit on 100%, predict test both views | 2 epochs x 5k + 5k forwards | 5 min |
| S5 decode offsets, cross-fit check, validation, write | cheap | 1 min |
| Total | | about 28 min |

- Headroom: 28 vs 50 min target is about 44%; within the 1 h worst case by a wide margin. Optional additions (second backbone, latent head) must keep the total at or below about 40 min estimated, by trimming epochs/folds with fixed counts, never by a clock.
- Memory: ViT-L bf16 weights about 0.6 GB; batch 32 at 588 tokens with grad on 2 blocks well under 12 GB; preloaded uint8 images about 2.5 GB host RAM (train+test).
- Determinism: seeds for random/numpy/torch/cuda, `PYTHONHASHSEED`, `cudnn.deterministic=True`, `benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch, fixed split seeds, fixed GPU augmentation generator, `num_workers=0` (images preloaded). `device="cuda"` hard-coded, no `is_available()` or `cpu_count()` branches, no import fallbacks, time used only in log lines. Pin the pretrained model revision. Fixed epochs and fixed step counts (no validation-triggered stopping); the 100% refit uses the same fixed steps as the CV runs.
- Input/output validation: assert train/target join, image existence and shape, finite predictions, 3 classes present in train; validator from CLAUDE.md section 5 plus integer dtype and {0,1,2} range check, ids equal test.csv ids, re-read written file with `keep_default_na=False`.

## Metric-aware training & decode

- Loss: softmax cross-entropy or cumulative-link BCE (P(y>=1), P(y>=2)), chosen on cross-fitted OOF macro-F1; weighted by class-balance exponent alpha in a fixed small set {0, 0.5, 1} and by group-size weights. Initialise output biases at the log prior.
- Decode: hard-label macro-F1 is not maximised by argmax. Fit two additive logit offsets (class 1 and class 2 relative to class 0) by a small bounded grid on OOF, bounded by the train label prior so the search cannot reach a corner where a class is never predicted. Report it cross-fitted (offsets fit on 4 folds, applied to the 5th) and apply to test only if the cross-fitted gain is positive in most folds; otherwise ship plain argmax. If the loss already includes balanced weights, offsets are expected to be near 0.
- Per-slot calibration: for the cumulative head fit the two thresholds this way; for softmax a temperature before the offsets, both on OOF only.
- Do not adjust to the test class mix. If a class gets no test predictions, log it loudly (the F1 for it would be 0) but do not change the predictions from test statistics.
- Verify decode on OOF with the exact official metric and per-class F1; also check macro-F1 within dark and normal subsets.

## Structural signals

- Ordinal nesting of the classes (0 < 1 < 2): cumulative head.
- Permutation invariance across animals: the label is a function of the set of per-animal states, so a location-wise score pooled by log-sum-exp (and the Poisson-binomial marginal) is the matching bias; avoid flatten heads that bake in absolute layout.
- Symmetries: horizontal flip invariance (to be asserted by D9 diagnostic and used as training augmentation and test-time flip averaging); vertical flip only if the diagnostic supports it. Photometric invariance (dark/normal): brightness/gamma/contrast/grayscale augmentation, since the description says test mixes ordinary and low-light views. Augmentation of real images only; no synthetic labelled data.
- Group structure: transformed duplicate groups share labels; group-weight the loss and fold on them. Duplicate groups with disagreeing labels indicate label noise (reported, not removed unless clearly justified).
- Count-of-animals correlate: an image-derived cue (more animals gives more chance of distinct states); the model learns it through pooled evidence, no hand-written counter.
- Causal/temporal alignment: frames are individual; temporal context from neighbouring frames is not allowed at test (cross-row pooling of test rows), so each prediction depends on its own image only.
- The label's source is per-animal, but only group-level labels are given: learning from aggregated labels; exact marginal likelihood through the DP is the principled form.

## Experiment roadmap

Each step has a stop criterion; log id, change, CV mean +/- std, per-fold, per-class F1, dark vs normal F1, runtime.
1. Contract, metric unit tests, groups (D3/D4) and folds (Scheme A and B) in place. Stop when metric tests pass and group sizes/leakage check are sane.
2. Baseline end-to-end and valid: frozen DINOv2-L pooled features plus softmax head, flip-averaged, valid `submission.csv` (Fallback A). Record OOF macro-F1 and per-class F1. This is the floor and the first credit submission.
3. Representation: native resolution vs 224, pooling variants (CLS, mean, GeM/LSE), backbone size (B vs L), second backbone as a frozen probe. Keep a change only if it beats the noise (one standard error and 4 of 5 folds).
4. Metric-aware loss/decode: class-balanced weights, ordinal vs softmax, decode offsets (cross-fitted), latent-presence Poisson-binomial head. Stop when gains no longer exceed noise.
5. LP-FT of the last 2 blocks with augmentation (flip, photometric, crop). Log parameter-change norm and gain over the probe, per fold, Scheme A and B, dark/normal. Keep if the gain exceeds noise or if needed for compliance and not harmful.
6. Diversity: second backbone with a different pretraining assumption, blended in logit z-score space only if standalone CV is close to the primary. Multi-seed heads are cheap; more LP-FT seeds only if runtime allows.
7. Small fixed in-script search (head C, alpha) with its own cross-fitted check. No large HPO.
8. Freeze the plan, set counts from profiling, run twice from clean `working/`, diff the CSVs (expect near-identical), check class distribution of predictions against OOF/train priors, then upload. Credits: baseline, best single, ensemble, final only.

Expected score (estimate, not a promise): frozen probe grouped-CV macro-F1 about 0.50-0.65; with LP-FT, ordinal head and decode offsets about 0.55-0.72; private LB expected about 0.02-0.06 below Scheme A CV, mostly from boundary-adjacency leakage, dark-frame shift and noisy per-class F1 on a small test set. Reasoning: blurred 384x288 frames, labels from per-animal annotations, a sandwiched middle class and label noise cap the ceiling well below 0.9. The prior-constant floor is about 0.2-0.3.

## Compliance audit

CLAUDE.md section 7 self-audit and section B checks, against this plan:
- Test file read only for per-image inference (and row count/ids): yes. No scaler/vocab/PCA/clustering fit on test or train+test; no rank/z-score normalisation across test; no pseudo-labels; no temporal smoothing across test frames; no test-based class-prior matching: pass.
- Time used in conditions: none; plan fixed by constants, time only in logs: pass.
- `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks, try/except changing work: none, device fixed to cuda: pass.
- Hard-coded constants tuned offline or via earlier submissions: none intended. Decode offsets, alpha and head C are found in-script from train-only OOF; backbone LR/epochs are literature defaults stated in comments and confirmed by CV in development (not by public-LB probing): pass, with the reviewer question below.
- External data/synthetic data/self-hosted weights/GitHub: none; only pinned HF/timm weights: pass. No mixup/cutmix, no generated labelled composites.
- Explicit bans: no COCO annotation recovery, no original split membership/filenames/annotation IDs as features (ids and paths never enter the model), no lookup table of released images, no manual test labels: pass.
- Strip-the-ML test: nothing remains without the trained backbone/head: pass. Model dominates the code.
- Sibling leakage: groups built from train content/blocks for CV and loss weights only; no cross-referencing of a row's group-mates at inference: pass.
- Raw pixels to a tabular model: none (deep model on pixels; frozen-feature heads are a stage inside a deep pipeline): pass.
- Source readable, under 512 KB, no encoded blobs: planned. Comments explain reasoning at each step; seeds fixed; threads/workers fixed (`num_workers=0`): planned.
- Output validity checklist (CLAUDE.md 5) with integer predictions in {0,1,2}, columns `id,prediction`, ids matching test.csv, validator raises before writing: planned.

## Open questions & assumptions

Reviewer questions:
1. Are general-purpose pretrained backbones (DINOv2 LVD-142M, EVA02, ConvNeXt in22k) allowed here? The description does not say "from scratch". Plan A assumes yes.
2. Is using the order of rows in `train.csv` (never `id` or the path text) only to build CV groups acceptable, given the ban on source filenames as predictive features? It is never a model input; if disallowed, groups come from image-content similarity alone (Scheme B-style, more pessimistic).
3. Does an LP-FT with the last 2 blocks trained, on a mostly frozen DINOv2 backbone, count as genuine training/fine-tuning for this challenge? The log of the parameter-change norm and CV gain over the probe is the evidence offered.
4. Are in-script OOF-fitted decode offsets (2 numbers) acceptable post-processing? They are fit on train OOF only.

Assumptions (each would change the plan if wrong):
- Runtime/GPU as in CLAUDE.md (A10G, at most 50 min target); if a shorter limit exists in the missing header, reduce S3 folds and epochs by fixed constants and say so in the code.
- Public dir layout: `train.csv`, `train_targets.csv`, `test.csv`, `sample_submission.csv`, `train_images/`, `test_images/`; image paths are relative to the public directory.
- `sample_submission.csv` ids equal `test.csv` ids in the same order; assert set equality with test.csv and keep sample order.
- Train is large enough for 5-fold grouped CV; if fewer than about 15 independent groups per class exist, use 3-fold with more repeats and treat decode offsets as non-shipped.
- The test split follows the description (every third 100-frame bucket, duplicate groups together); no test image statistics were or will be used to check this.

Could not verify (no data, no GPU in this sandbox): N, class mix, duplicate and group structure, row-order meaning, dark-frame share, per-image luminance and overlay remnants, flip symmetries, backbone throughput, and every score and runtime estimate above.
