# Eris plan: T2 visible flood-failure mechanism recognition

Status of evidence: no dataset was available to the strategist. Everything under "Data findings" is a diagnostic to run plus an UNVERIFIED hypothesis. Numbers from the task text (841/215 rows, 54/13 sites, base rates, positive-site counts, 30 min limit) are taken as given. All runtime figures are estimates, not measurements.

## Contract & decision unit

**One valid answer.** For each test id (about 215), in sample_submission order: `id` plus `prediction`, a JSON object with exactly the four keys `support_scour`, `debris_obstruction_or_impact`, `approach_or_embankment_washout`, `structural_displacement_or_collapse`, each a finite float in [0,1]. Written with `json.dumps` (double quotes, fixed key order) inside a csv written by `to_csv(index=False)`. Reload with `keep_default_na=False` and `json.loads` every row to verify.

**Invalid vs low-scoring.** Invalid: missing or extra keys, NaN/inf, non-JSON string, wrong row count or order, ids not matching. Merely low-scoring: any valid object. Assumed, to confirm with a reviewer: values must lie in [0,1] and need not sum to anything (independent labels).

**Metric, term by term.**
- Per label k: weighted average precision AP_k over ALL test photos pooled (not per site), with photo weight w_i = 1 / (photos at site(i)). Presumably `average_precision_score(y_k, p_k, sample_weight=w)`; the exact tie and interpolation convention is an assumption. Score = mean_k AP_k.
- Rank-only per label: calibration is irrelevant within a label, but the ordering ACROSS sites is what is scored. A site whose photos are uniformly scored high or low (a site-level offset) is penalised even if within-site ranking is perfect. Cross-site comparability of scores is therefore the main generalisation problem, not calibration as such.
- Equal site weighting: a site with 5 photos weighs the same as a site with 30; each photo of a small site carries 6x the weight. Hierarchical: per-photo weight inside pooled ranking, so training must use the same weights.
- No gating (no "wrong subset zeroes the score"), but macro averaging over 4 labels means the rare, hardest labels (collapse, 14 of 54 positive sites; debris 28; scour 21) count as much as washout (44).
- Chance level: weighted prevalence per label, roughly 0.11 to 0.28, macro about 0.16 (the weighted prevalence differs from the quoted photo-level base rates because of the 1/n_site weights; computed in diagnostics).

**True independent unit.** The site (54 train, 13 test, disjoint). Photos within a site share the bridge, camera, lighting, river and often the failure. Effective sample size is about 54 groups, not 841 rows. With about 13 test sites and roughly 3 to 4 positive test sites for the rarest label, the private score will be very noisy.

**Pipeline stages.** (1) Representation: strong frozen backbone producing patch tokens. (2) Scoring: per-label spatially-local scorer with pooling. (3) Light adaptation: LP-FT of the last blocks. There is no candidate pool or combinatorial decode; the "decode" is monotone sigmoid plus flip averaging. Diagnose separately: representation quality (global-pooled probe AP), spatial-pooling gain (patch-LSE head vs global head), adaptation gain (LP-FT vs probe), calibration/offset loss (AP before vs after oracle per-site centring, diagnostic only on train OOF).

## Compliance regime

**Domain.** Computer vision, multi-label image classification, `Fine-tuning`-like platform requirement stated in the task ("real training/fine-tuning inside the script is required").

**Explicit bans from the description and CLAUDE.md.**
- Site prefix (id before first `-`) as a predictive feature or for any site-metadata lookup: banned. Allowed only for grouping.
- External data, pseudo-labels, test statistics of any kind: banned (CLAUDE.md 2.3 #3, #5).
- Runtime 30 min on A10G overrides the 1 h default; no wall-clock branching.
- Weights only from HF/timm, revision pinned; no GitHub.
- Never raw pixels to a tabular model.

**Allowed mechanisms used.** General-purpose pretrained backbone (DINOv2-L from HF; ConvNeXt-L from timm only if added), trained heads, last-block fine-tuning, flip TTA (per-sample), flip augmentation, token-dropout augmentation.

**Grey.** Frozen-feature plus small head is grey in CV / fine-tuning categories. Mitigation (F2.1 rung ladder): ship an LP-FT stage (head warm-started from the probe, last 2 transformer blocks unfrozen, tiny LR, 2 epochs) as the compliance layer; log the parameter-change norm of those blocks and its CV gain over the probe. Reviewer question below.

**Ambiguity with plan under each reading.** (a) Using the train site prefix to compute loss weights 1/n_site and CV groups: description allows grouping; weighting is a metric-mirroring use of the same grouping. Reading 1 (allowed): weights as planned. Reading 2 (stricter, weights disallowed): train unweighted; the measured cost is computed in CV (expected small, because the unweighted loss still ranks photos). Ship reading 1 only if a reviewer agrees; otherwise run reading 2 from the same code with a single constant. (b) The test-time use of the site prefix is banned in both readings, so there is no site-level smoothing or normalisation of test predictions.

**Self-audits.**
- Strip-the-ML: remove the head and fine-tuned blocks and nothing predicts; the only non-learned parts are resize, normalise, flip. Passes.
- No whole-test aggregation: each test row's prediction depends only on its own image plus a train-fit model. In particular NO averaging of predictions over test photos sharing a site prefix, NO per-site or per-test-file z-scoring, NO test-fit scalers (feature standardisation uses train statistics only). This is the single most tempting and most clearly banned trick for this task; do not even prototype it.
- Sibling leakage: no features built from site-mates.
- Hard-coded constants: each tuned quantity (L2 grid, token-dropout rate, head steps) comes from an in-script train-only grouped search; untuned architecture constants (LP-FT LR 1e-5, 2 epochs, resolution, pool layout) are stated design defaults, chosen without peeking at public LB; any that matter are covered by the roadmap sensitivity step and logged.

## Data findings

All items UNVERIFIED hypotheses; each diagnostic is run on train only. For test, only schema, row count (about 215), id format and image size are used, for memory and runtime planning.

1. **Files and schema.** Read train.csv / sample_submission.csv headers, image directory layout and naming, confirm the id maps to a file, confirm label columns (0/1 ints, or a JSON column?). Hypothesis: one row per photo, 4 binary columns or a JSON label; all images 960x640, JPEG. Check for any portrait/rotated or different-size images (EXIF orientation) and handle with a fixed rule.
2. **Site structure.** Photos per site: min / median / max, histogram, resulting weight range 1/n_site. Hypothesis: sites of 5 to 40 photos; if max weight / median weight exceeds about 5, plan weight clipping chosen by in-script grouped CV (option: none vs cap at 5x median).
3. **Label statistics.** Per label: photo-level positive rate (should reproduce 13.6 / 12.6 / 27.7 / 11.3 %), weighted prevalence, positive-site count (44 / 28 / 21 / 14), and, for positive sites, the fraction of their photos that are positive. Hypothesis: labels are strongly site-clustered but not constant within site (some photos at a washout site show only the intact span), so the intra-site correlation is high but not 1; the photo-level signal is real but the site-level offset matters.
4. **Site-vs-photo variance.** (i) Oracle A: predict each photo with its OWN site's leave-nothing-out positive rate (a label-leaking oracle used only as a ceiling); (ii) oracle B: perfect within-site ranking with site-mean from a trained model. Diagnostic of how much AP is determined by between-site offset. Hypothesis: oracle A is far above any image model, meaning the metric rewards site-level recognition; this justifies strong site-invariance regularisation and group CV.
5. **Co-occurrence.** 4x4 label co-occurrence, count of all-negative photos, count of multi-label photos. Hypothesis: collapse co-occurs with scour and displacement-like cues; a sizeable fraction of photos has no label (context shots). If all-negative is above about 30%, add an auxiliary "any mechanism" output.
6. **Duplicates.** Near-duplicates and cross-site "same bridge" detection: cosine similarity of DINOv2 CLS embeddings on train only, union-find at a high threshold, merged with the id site groups to form CV groups. Hypothesis: a few sites are re-visits of the same bridge under different prefixes; merge them for CV.
7. **Visual inspection (train).** View about 10 positives and negatives per label to see where evidence lives. Hypotheses: scour at pier or abutment base (lower, middle image), debris against piers/deck (middle band), washout on the embankment sides (left/right edges), displacement/collapse often large and central. Supports a spatially local head and a high-resolution input rather than 224.
8. **Id/row-order leakage suspects.** Does the numeric suffix after the site prefix, file size, JPEG quality, EXIF or camera model correlate with labels within or across sites? Hypothesis: possibly (photographers shoot overviews first); do NOT use any of them; record only as a leak warning, and also check that image statistics do not trivially identify the site (adversarial site classifier on frozen features only as a diagnostic of how much site identity the features carry).
9. **Resolution sensitivity (cheap probe).** Global-pool linear probe AP under grouped CV at 224 square, 448x672, and (if memory allows) 644x966. Hypothesis: mid-resolution (about 448x672) beats 224 by a clear margin for scour and debris; diminishing return beyond.
10. **Flip invariance.** Train probe on original, evaluate OOF on flipped; hypothesis AP drop below noise, so horizontal flips are valid augmentation and TTA. Vertical flip is not used (gravity, water).
11. **Information ceiling.** Naive best-possible rule on train: AP of the best single global probe feature set; AP when the label is predicted from the site prior only (see 4). Hypothesis: macro weighted AP for a well-regularised frozen probe sits well above chance (about 0.16) but remains limited by site shift.
12. **Irreducible ambiguity.** Fraction of near-identical photos (within-site near-duplicates from 6) with differing labels; hypothesis small, label noise at the boundary of "visible" mechanisms is the main ceiling.

## Validation design

**How the test split was probably made.** Group split by site: 13 held-out sites, disjoint from 54 train sites. Mirror it exactly.

**CV.** StratifiedGroupKFold, 5 folds on the derived groups (site prefix merged with near-duplicate union-find from diagnostic 6), stratified by site-level positive pattern of the rarest labels so each fold has at least 2 to 3 positive sites for collapse (14 positive sites / 5 = about 3 per fold). About 11 sites per fold, close to the 13 test sites. Repeat with 3 different split seeds for all comparisons that decide the architecture; report mean and std of the fold weighted-AP per label and macro, plus site-bootstrap confidence intervals. Compare designs PAIRED on identical folds; accept a change only when its gain exceeds one standard error of the paired fold differences and is consistent across split seeds.

**Metric implementation.** Re-implement weighted AP exactly (sample_weight = 1/n_site computed on the evaluated set). Unit tests: perfect scores give 1.0; reversed scores give very low; constant scores give the weighted prevalence; base-rate predictions equal constant; weights all equal reduce to unweighted AP; test the macro mean over 4 labels; test a toy two-site example by hand. Report both fold-mean AP (best analogue of one 13-site test set) and pooled-OOF AP.

**Shift axis.** The deployment shift is held-out bridge sites (new bridge, new camera, new river, possibly new season), so "leak-free" means whole sites held out. Extra diagnostic: leave-site-offset loss (AP after oracle per-site centring minus raw AP) to measure how much the cross-site ordering costs.

**Nested selection.** Every post-hoc selection step needs its own held-out check: the per-label L2 strength, the head-variant choice per label (global vs patch-LSE), weight clipping, token-dropout rate, and the per-label recalibration are selected on inner folds and evaluated on the outer fold (cross-fitted). The reported number is the outer-fold one. A self-built proxy over-estimates; expect the real private number to land below the CV figure, and note rare-label noise: with about 3 to 4 positive test sites for collapse, one site can swing that label's AP by 0.1 or more.

## Overfit/underfit risks

**Overfit.**
- 54 groups; site identity leaks through background, camera, colour cast. Mitigation: group CV, 1/n_site weights, strong L2 per label chosen on a wide grid including the very strong end, token-dropout and flip augmentation, fixed short schedules, no full fine-tune, no validation-triggered stopping, few head parameters (about 1k per label).
- Many tuned knobs on a noisy CV: keep tuned knobs to L2 (per label), head variant, token-dropout rate; everything else fixed.
- Rare labels with 14 positive sites: very strong regularisation or a shared patch scorer across labels; per-label model choice is only allowed with nested check.
- Selection of the best backbone/resolution on the same OOF used for the final estimate: use paired repeated seeds and the nested estimate.

**Underfit.**
- Too-small or too-low-resolution representation (224 squash of a 960x640 image loses the small debris and scour cues): use DINOv2-large at 448x672 aspect-preserving.
- Global pooling flattening local evidence: use patch-level scorer with log-sum-exp pooling.
- Over-regularised head for the easy label (washout): per-label L2 from the wide grid fixes this.
- LP-FT too timid to matter: that is acceptable (the probe is the real model); log param-change norm.
- Loss/metric mismatch: use the metric's own weights (1/n_site) in a weighted BCE.

## Recommended approach (primary + fallback)

**Primary (lean).** DINOv2-large (HF `facebook/dinov2-large`, pinned revision) at 448x672 (32x48 patches), fp16/bf16 inference, original plus horizontal flip views.
1. Cache, for both views, the hidden states entering the last 2 blocks (for LP-FT) and the final patch tokens (average-pooled 2x2 to a 16x24 grid for the head), plus CLS.
2. Head per label k: a shared-nothing linear patch scorer `s_k(patch) = w_k . LN(token)` followed by log-sum-exp pooling over the patch grid with a learned temperature and a base-rate-initialised bias (translation-equivariant: evidence can be anywhere). Parameters: about 1024 per label plus 2 scalars. Trained with weighted BCE (weights 1/n_site, normalised to mean 1), AdamW, fixed full-batch steps, token dropout (random 25% of patches per step, rate chosen from a 3-point in-script grid), flip augmentation, and L2 per label from a log grid (for example 8 values from 1e-4 to 1e2 including the strong end).
3. In parallel, the global head variant (CLS concatenated with mean of patch tokens, flip-averaged, linear, per-label L2). Choose between the two heads PER LABEL by nested grouped CV; the shipped model uses the winner per label (or their logit average if close, as a diversity-of-assumption blend).
4. LP-FT: warm-start from the chosen head, unfreeze the last 2 transformer blocks plus final norm, LR 1e-5 for the blocks (head LR 10x lower than at probe time), 2 epochs, fixed seeds, token dropout and flip augmentation. Log the L2 norm of the block parameter change and the CV gain over the probe. This is the compliance layer and must be demonstrably non-zero.
5. Optional tiny cross-label stack: a per-label logistic regression on the 4 OOF logits (within-row features only), cross-fitted, kept only if the paired CV gain exceeds noise.
6. Inference: average logits over original and flip, sigmoid, then a per-label monotone recalibration (a, b) fitted on cross-fitted OOF (affects probabilities only, not AP). Refit everything on 100% of train for the shipped test predictions (fold models only to produce OOF numbers, selection and calibration).

**Fallback.** Frozen-feature linear model: concatenated DINOv2-L (CLS plus mean patch, flip-averaged) and, if time, ConvNeXt-L (timm, in22k pretrained, at native 640x960, GAP plus GeM) features, standardised with train statistics, one weighted logistic regression per label with per-label L2 chosen by nested grouped CV. No fine-tuning stage; use only if the primary fails validation or the LP-FT stage is rejected by a reviewer (in which case the question about "genuine training" must be asked first, since this fallback is grey).

**Diversity member (roadmap step 5, only after the primary is measured).** ConvNeXt-L (supervised in22k, convolutional inductive bias) head, blended with the DINOv2 logits by train-reference z-scoring or by feeding its OOF logits into the tiny stack; add it only if its standalone CV is within about one SE of the DINOv2 head and the paired blend gain exceeds noise.

**Why this fits this data.** About 54 groups, 4 labels, evidence that is spatially local (pier base, embankment edge, debris on piers), high resolution photos, and metric cost dominated by cross-site ordering: the lowest rung of the capacity ladder with a translation-equivariant head, strong regularisation and site-weighted loss is the best expected private score; a full fine-tune of a base-size or larger model on 841 photos from 54 sites would memorise sites.

## Rejected options

- Full or partial-block fine-tuning of the whole backbone: memorises sites on 54 groups; runtime-risky inside 30 min; no measured upside.
- Averaging or smoothing predictions across test photos of the same site (via the id prefix): banned (site prefix as a feature, whole-test aggregation). Not implemented, not prototyped.
- Per-test-site or per-test-file normalisation, test-fit scalers, label-free prior estimation on test, pseudo-labels, test-time adaptation: banned.
- Zero-shot CLIP text prompts as the predictor: inference-only; may not replace training.
- GBDT or other tabular models on frozen embeddings as the core: grey, weak with 54 groups; at most an optional diversity member.
- Hand-engineered image features (colour histograms, edge or water-colour thresholds) as the core: rule-based; low value; a stripped-ML risk.
- Object detection / segmentation heads: no box or mask labels.
- 224x224 squash of the image: loses small cues; rejected on the resolution diagnostic.
- Large heterogeneous ensemble (several giant backbones, many seeds): runtime (30 min), and near-identical families add little; one extra family at most.
- Training on synthetic or externally generated flood images: banned.
- Vertical flip, heavy colour or geometric distortions that change failure semantics: rejected; mild brightness/contrast jitter only if the cached-feature pipeline supports it cheaply.

## Fixed work plan & runtime budget

All counts fixed in the file; device `cuda`; seeds fixed; no time-dependent branches; no env-dependent fallbacks; DataLoader with fixed `num_workers` and a seeded Generator; SDPA/eager attention fixed; `torch.use_deterministic_algorithms(True, warn_only=True)`; pinned HF revisions. A10G, 30 min limit. Estimated times (all UNVERIFIED, to be profiled locally):

| Stage | Fixed plan | Est. time |
|---|---|---|
| Read csv, validate schema, decode and resize 1056 JPEGs (fixed resize rule, fixed workers) | 1056 images | 0.5 to 1 min |
| DINOv2-L extraction at 448x672, orig + flip, batch 8, fp16 autocast; cache hidden state before last 2 blocks and pooled final tokens | 2112 forward passes | 2 to 3 min |
| Head grid under nested grouped CV: 5 folds, 2 split seeds for the selection stage, 2 head variants, 8 L2 values, 3 token-drop rates handled as 3 x 8 x 2 = 48 fits per fold-seed, each about 150 full-batch steps at about 1 to 2 s on cached features | about 480 short fits | 4 to 5 min |
| LP-FT of last 2 blocks: 5 folds x 2 epochs + 1 final refit on 100%, orig + flip views | about 12 epoch-equivalents of 2-block fwd/bwd | 2 to 3 min |
| Cross-fitted recalibration and optional tiny stack | negligible | under 0.5 min |
| Final refit of head on 100% of train + test inference (orig + flip) | 215 test images | 1 min |
| Output validation, JSON reload check, write | | under 0.5 min |
| Optional ConvNeXt-L extraction + head (roadmap step 5) | 2112 passes at 640x960 | +3 min |

Estimated total about 11 to 15 min without the diversity member, about 15 to 19 min with it, leaving at least 35% headroom against the 30 min limit (target at most 20 min). Memory: DINOv2-L weights about 0.6 GB; cached hidden states before the last 2 blocks, 2 views x 1056 x 1536 tokens x 1024 dims x 2 bytes = about 6.6 GB (keep on GPU or pinned CPU); pooled head tokens 16x24 grid about 1 GB; peak well under 24 GB, with batch 8 at 1536 tokens (profile and keep a fixed headroom). Reduce risk by freeing GPU memory between stages.

**Robustness inside the script.** Assert id/prefix parse, label dtype, finite features, no NaN logits, finite probabilities in [0,1], row count equals sample_submission, re-read the written CSV and `json.loads` each prediction, assert exact 4 keys. Do not write a constant fallback submission silently; log loudly on any anomaly and fail rather than degrade.

## Metric-aware training & decode

1. **Per-example weights.** Weighted BCE with w_i = 1/n_site(i), normalised to mean 1; optional cap at a multiple of the median chosen in-script by nested CV only if the weight range in diagnostic 2 is extreme. This replicates the hierarchical (site-then-label) averaging.
2. **Ranking nature.** AP is rank-only per label; BCE with base-rate-initialised bias is a proper scoring rule and gives monotone-consistent ordering; no threshold or decode search is needed. A pairwise cross-site ranking auxiliary term (pairs drawn across different training sites) is a roadmap candidate if the offset-loss diagnostic (data finding 4) is large.
3. **Per-label regularisation.** L2 per label selected by weighted AP under grouped CV from a very wide grid including the strong end; the rare labels (collapse, scour) are expected to want stronger regularisation than washout (hypothesis).
4. **Auxiliary loss.** If many photos have no label, add "any mechanism" as a fifth training-only output; used to shape the shared representation of the patch scorer only.
5. **Calibration.** Per-label monotone logistic recalibration (a, b) on cross-fitted OOF logits; reported cross-fitted; does not change AP, produces sensible probabilities in [0,1].
6. **Decode.** Average original and flip logits (TTA, per-sample, allowed), sigmoid, write. No metric-aware decode constants exist, so no decode-time test fitting.
7. **Refit.** Fold models only for OOF; ship refit on 100% of train. Check that the OOF-based calibration transfers by comparing the refit model's train-score distribution with OOF (sanity, not tuning). Use a fixed number of head steps and LP-FT epochs derived from the CV runs in the same script, never a clock.

## Structural signals

- **Locality.** Evidence is local (pier base, embankment edge, debris on piers): patch-level scoring with log-sum-exp pooling; no flatten or global-pool-only head as primary.
- **Flip symmetry.** Horizontal flip is label-preserving for all four mechanisms; verify with the OOF flip diagnostic (data finding 10), assert it, use it as augmentation and TTA. No vertical flip.
- **Site grouping.** Group CV, 1/n_site loss weights, and merged near-duplicate groups; site prefix used only here.
- **Label coupling.** Four labels share one backbone and one patch representation; optional cross-label stack from within-row OOF logits; a physical hierarchy (collapse often implies displacement/scour cues) can be tested by train diagnostic (conditional label rates) and, if it is exact in train, encoded as a decode-time mask or coupling feature, verified on all train rows.
- **Aspect and resolution.** Keep the 3:2 aspect; patch grid 32x48; the 2x2 average pooling halves memory, with the LP-FT blocks seeing full-resolution tokens.
- **Within-site correlation of photos.** Present in train; cannot be exploited at test time without the site prefix (banned); consequently training weight and regularisation are the only handles on it.
- **Hierarchical averaging** of the metric is replicated by the loss weights; the macro average over labels motivates per-label regularisation and per-label model choice rather than a single global setting.

## Experiment roadmap

1. **Contract, metric, validation.** Parse data, build the groups (site plus near-duplicate merge), StratifiedGroupKFold with 3 seeds, weighted-AP implementation with unit tests (perfect, reversed, constant, base-rate, equal-weights reduction, hand-made two-site case). Stop when tests pass and the fold site and positive-site counts look sane (at least 2 positive sites for collapse per fold).
2. **Cheapest valid baseline end to end.** DINOv2-L at 224 global CLS+mean-patch, per-label weighted logistic regression; write a valid submission, validator passes, record the CV table. Stop criterion: baseline macro AP clearly above chance.
3. **Representation and structure.** Resolution ladder (224 -> 448x672 -> if memory allows larger), patch-LSE head vs global head, flip TTA, token dropout, per-label L2 grid. Accept each step only when the paired gain exceeds one SE across 3 split seeds. Stop when two consecutive steps give no gain.
4. **Metric-aware loss.** Weighted BCE vs unweighted; optional weight cap; optional cross-site pairwise auxiliary term and "any mechanism" auxiliary; per-label recalibration. Stop at the first configuration where further changes are within noise.
5. **LP-FT compliance layer.** Last-2-blocks LP-FT at LR 1e-5 (and, if time permits, 5e-6 and 2e-5 as a 3-point check), log parameter change norm and CV gain over the probe. Requirement: measurable parameter change and non-inferior CV (within one SE); if clearly worse, ship the probe and send the reviewer question first.
6. **Diversity (only if the lean design is measured).** ConvNeXt-L at native 640x960 head; standalone CV, then train-reference z-score blend or OOF-stack feature; keep only with gain above noise and runtime still at most 20 min.
7. **In-script bounded HPO.** Only the already-listed tuned knobs (per-label L2, token-drop rate, optional weight cap), fixed trial counts, own nested check, no timeouts. Prefer robust plateaus over sharp optima.
8. **Final fixed-plan run.** Clean run from scratch with the one official command, run twice, diff the submissions (near-identical), check the distribution of test predictions against OOF (mean, spread per label), runtime at most 20 min, validator passes. Credits: baseline, best single, final (and ensemble if step 6 survived).
When two independent solvers would plausibly converge on a technique (hi-res frozen strong backbone with a local pooled head), test it before any proxy-driven tweaking.

## Compliance audit

CLAUDE.md section 7 red items against the plan:
- Test read only for one-sample inference (decode, resize, forward, flip average): yes. No scaler/vocab/clustering fit on test; feature standardisation, if any, uses train statistics only. No dedup, no rank normalisation across test, no site-level aggregation or use of the test site prefix.
- No `time.time()` inside any condition; time only in `log()`.
- No `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks or try/except path switches; `device = "cuda"`, fixed workers and batch size.
- Hard-coded tuned constants: none offline-derived; tuned knobs from in-script grouped search; untuned design constants (resolution, LP-FT LR and epochs, grid ranges) documented as fixed defaults in comments and not set from public-LB feedback.
- No external or synthetic data, no self-hosted weights; only HF/timm pretrained backbone with pinned revision; allowed libraries only (torch, transformers, timm, scikit-learn, numpy, pandas).
- Strip-the-ML: passes; the head and fine-tuned blocks are the predictor.
- Model-heavy part dominates: yes (backbone plus trained head plus LP-FT); the grey spot is frozen-feature-plus-head, mitigated by the LP-FT stage with logged parameter change.
- Challenge restrictions honoured: 30 min limit, no site-prefix feature, no site lookup, no external data.
- Section B self-audits: sibling leakage absent; every constant derivable in-script; whole-test aggregation absent.
- Determinism: fixed seeds for random, numpy, torch, CUDA, DataLoader generator; `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` set before torch import; cudnn deterministic; SDPA or eager attention fixed; tolerance for fp16 reduction noise since constants are few and ensembles averaged.
- Source readable, under 512 KB, no blobs; comments explain each step.

## Open questions & assumptions

**Reviewer questions.**
1. Is a frozen pretrained backbone with a trained spatial-pooling head plus a logged LP-FT of the last two transformer blocks accepted as genuine training/fine-tuning under this challenge's category? (This is the central compliance risk.)
2. Is using the id prefix to compute 1/n_site loss weights and CV groups acceptable, given the description allows it for grouping only? Plan under "no" reading: unweighted loss, same code, one constant flipped.
3. Is horizontal-flip TTA (per-sample, no test statistics) fine? (CLAUDE.md says yes; confirm for this task.)
4. Exact weighted-AP definition (sample_weight with sklearn conventions and tie handling) and whether probabilities must lie in [0,1] only, or also be valid as an independent per-label vector.
5. Is the near-duplicate union-find merging of train sites for CV acceptable (train-only, grouping use)?

**Assumptions (unverified).**
- A one-image-per-row csv with `id` plus 4 label columns, images in a folder resolvable from `id`; all 960x640 JPEGs.
- Site prefix parse is `id.split("-")[0]`.
- Test has 13 sites with no train overlap; the label noise level is moderate.
- DINOv2-large and fp16 inference fit the A10G budget as estimated; the exact HF revision hash and timm model tag to be pinned and checked at implementation.
- Expected performance (estimate only, with reasoning): chance macro weighted AP about 0.16; a well-regularised frozen strong-backbone probe with site-group CV plausibly reaches a CV macro weighted AP of about 0.35 to 0.55; the private score is likely lower and very noisy, perhaps 0.30 to 0.50, because of 13 test sites, rare-label positive-site counts of about 3 to 4 and cross-site offset effects. No score is promised.

**What could not be verified.** Everything about the dataset (schema, site-size distribution, label co-occurrence, near-duplicates, flip invariance, resolution gain, cross-site offset loss), all runtime and memory figures, the exact model identifiers and revisions, and every expected score above.
