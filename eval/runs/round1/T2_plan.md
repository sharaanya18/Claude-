# Eris plan - T2 visible flood-failure mechanism recognition

Status of evidence: no dataset was available for this run. Everything under "Data findings" is a diagnostic to run or a hypothesis, marked UNVERIFIED. Numbers quoted from the task text (841 train, ~215 test, 54/13 sites, base rates, positive-site counts) are given facts; derived arithmetic is shown. All runtimes and scores are estimates.

## Contract & decision unit

**One valid answer.** A CSV with columns `id, prediction`, one row per test photo, same ids and order as `sample_submission.csv`. `prediction` is a JSON object with exactly the 4 keys `support_scour`, `debris_obstruction_or_impact`, `approach_or_embankment_washout`, `structural_displacement_or_collapse`, each a finite float in [0,1]. Invalid (scores below zero, burns a credit): wrong columns or order, missing key, extra key, NaN/inf, value outside [0,1], malformed JSON, row-count or id mismatch, a crash, a timeout above 30 min. Low-scoring but valid: any well-formed probabilities.

**Metric, term by term.**
- For each label l, compute AP_l = average precision over all test photos, with per-photo weight w_i = 1 / n_site(i) (n_site = number of test photos at that photo's site). Presumably this is `sklearn.metrics.average_precision_score(y, s, sample_weight=w)`, i.e. the step-sum form. This is an assumption (the exact formula is paraphrased); re-check against the original description wording.
- Score = mean of the 4 AP_l (macro). Each label counts equally regardless of prevalence.
- Hierarchy: weights make every SITE count equally, so the metric is a site-weighted average. A site with 40 photos contributes the same total weight as a site with 2 photos.
- Rank-only per label: AP is invariant to monotone transforms of each label's scores. Calibration does not matter, but the ranking must be comparable ACROSS photos from different sites (a global ranking per label, not per site).
- No gating term is described. Degenerate case: a label with zero weighted positives in the test split makes AP undefined; with 13 test sites this is unlikely for washout/debris/scour but possible-ish for collapse (expected ~3.4 positive sites of 13, see Data findings).

**True independent unit.** The SITE (54 train, 13 test), not the photo. Photos at one site share the bridge, river, camera, weather and likely the failure mechanism. Effective sample size is closer to 54 than 841; positive sites per label (washout 44, debris 28, scour 21, collapse 14 of 54) are the binding statistic.

**Pipeline stages.** This is not a structured-decode problem: the 4 labels are independent sigmoid outputs with no hard output constraint known from the text (verify co-occurrence on train, see Structural signals). Stages: (1) representation/coverage: does the backbone at the chosen resolution keep the evidence (pier-base scour, debris on piers, embankment washout, displaced spans) visible? (2) scoring/ranking: a global per-label ranking across sites; (3) no decode beyond the monotone map to a probability. Diagnose separately: (a) representation quality via a frozen-feature linear probe on grouped CV (dev diagnostic only, never shipped); (b) fine-tuned ranking quality; (c) decomposition of the weighted AP into between-site ranking (site-mean score vs site-has-positive) and within-site ranking.

## Compliance regime

**Domain.** Computer vision, multi-label image classification, small data. CLAUDE.md section 6.3 applies (fine-tune a CNN/ViT; never raw pixels into tabular models). The platform text states real training/fine-tuning inside the script is required, so section 6.7 strictness is adopted: genuinely fine-tune a pretrained backbone, not frozen embeddings plus a head.

**Allowed.** HF/timm pretrained weights downloaded inside the script; fine-tuning; fixed-work K-fold ensembling; per-image TTA (hflip); augmentation of real train images; train-only fitted calibration.

**Explicit bans extracted from the description.**
1. Site prefix as a predictive feature or to look up site metadata. It may be used only for grouping.
2. External data (none planned). Runtime: 30 min on A10G (overrides the 50 min default).
3. Platform requires real training inside the script.

**Reading choices where the text is ambiguous.**
- Use of the train site prefix for per-photo LOSS WEIGHTS (w = 1/n_site) and for stratified sampling: is that "only for grouping"? It is a train-time use that does not enter the model as a feature and does not touch metadata, but it is a literal use beyond grouping. Compliant-under-every-reading plan: group-only (uniform weights, uniform sampling). Plan: measure the CV gain of site weighting vs uniform (experiment E4). If the gain is within noise, ship uniform (compliant under all readings). If it is a clear gain, ship weighting AND flag it to a reviewer (open question Q1) with the group-only fallback ready. Cost of the strict reading is measured, not assumed.
- Test-time aggregation of predictions over photos sharing a test site prefix (smoothing predictions within a site): BANNED by every plausible reading (it uses the id prefix as a predictive feature and is whole-test/sibling aggregation). The script must never parse the prefix of a test id. This can look attractive if labels are site-correlated; it is excluded.
- EXIF/file-size/JPEG-quality/filename features: not used (leak-prone proxies for site). Only decoded pixels (after a fixed orientation handling) enter the model.
- TTA with hflip is per-sample, allowed. No test-batch statistics: use LayerNorm backbones (ViT/ConvNeXt), model.eval() at inference; no BatchNorm backbones with test-time batch stats.

**Self-audits (plan level).**
- Strip-the-ML test: nothing rule-based exists. Remove the fine-tuned network and there is no prediction. Pass.
- No whole-test aggregation: each prediction is a function of its own pixels plus train-fit weights. The only post-fit steps (z-score reference, Platt scaling) use train OOF statistics, not test. Pass.
- Sibling leakage: the label is a property of the photo, features are the photo itself; no cross-referencing of group-mates. Pass (provided Q1 weighting reading holds; group-mate-derived auxiliary targets such as site-mean labels are rejected for this reason).
- Hard-coded constants: backbone id, pinned revision, resolution, epochs, LR, layer decay, drop-path are architecture/recipe conventions fixed before the final run, not decode constants. No thresholds exist (AP is rank-only). The only fitted scalars (z-score mean/std, Platt a,b per label) are computed in-script from train OOF. Any recipe choice that came from dev experiments (e.g. resolution) is logged as a design choice with its CV table, not smuggled as a tuned magic offset; keep the number of such choices small (<= 6 configs compared).
- Inference-only / frozen-features-plus-head: avoided. Load-bearing fine-tuning: all transformer blocks unfrozen with layer-wise LR decay; after step 1 assert that the max parameter delta in blocks 0, mid, and last is non-zero.

## Data findings

**None of this is verified (no data available). Diagnostics to run on TRAIN only, with the hypotheses I expect.**

Arithmetic derivable from the stated facts (verified arithmetic, interpretation UNVERIFIED):
- Train positive photos: scour 0.136*841 ~ 114; debris 0.126*841 ~ 106; washout 0.277*841 ~ 233; collapse 0.113*841 ~ 95.
- Photos per site: 841/54 ~ 15.6 (train), 215/13 ~ 16.5 (test).
- Positive photos per positive site: scour 114/21 ~ 5.4; debris 106/28 ~ 3.8; washout 233/44 ~ 5.3; collapse 95/14 ~ 6.8. Against ~15.6 photos per site, only about 35% (scour), 24% (debris), 34% (washout), 44% (collapse) of photos at a positive site would show the mechanism if positive sites are average-sized.
- Test expectation by site share (13/54): ~5.1 scour sites, ~6.7 debris, ~10.6 washout, ~3.4 collapse. The collapse AP on test will be determined by only ~3-4 sites. Expect very high private-LB variance.

Diagnostics and expected hypotheses (UNVERIFIED):
1. **Id and file integrity.** Parse train ids: confirm every id has a `-`; list prefix patterns; count photos per site (min/median/max, share of sites with <= 3 photos). H: right-skewed; a few large sites (30-60 photos) and several tiny ones. Report the weight range 1/n and the Kish effective sample size of the weights. Check whether prefix strings that differ may denote the same bridge.
2. **Label table.** Exact photo-level and site-weighted prevalence per label (the constant-prediction AP floor equals the weighted prevalence, so this is the baseline to beat). Label co-occurrence matrix and the count of all-negative photos. H: scour and debris co-occur; collapse co-occurs with scour/washout; a non-trivial share of photos (maybe 40-50%) have no label.
3. **Within-site label purity.** For each label: fraction of photos positive inside positive sites; between-site vs within-site variance of the label (intraclass correlation). H: only partial purity (arithmetic above says ~25-45%), so labels are photo-level visual facts, not site constants. If instead purity is near 1 for some label, a site-identity shortcut exists and grouped CV matters even more.
4. **Between/within decomposition of the ceiling.** Oracle with site-level truth only (score = site has label): its weighted AP bounds what site-level knowledge buys. H: washout (81% of sites positive) has small between-site signal, so AP is mostly within-site ranking; collapse (26% of sites) has larger between-site signal.
5. **Image stats.** Confirm all are 960x640 (3:2), orientation via EXIF (apply `exif_transpose` consistently), corrupt/truncated JPEG count, grayscale count, mean/std per channel. Per-site camera/size heterogeneity (as a leak WARNING, not a feature). H: uniform size; evidence is often small relative to the frame (pier base, debris on a pier), so heavy down-sizing to 224x224 would throw away signal.
6. **Duplicates / near-duplicates.** Perceptual hash (pHash/dHash, hamming <= small fixed threshold) plus frozen-embedding nearest-neighbour similarity across DIFFERENT sites. H: few or no cross-site near-duplicates; a handful of within-site bursts (same view repeated). If cross-site links exist, union-find merges those sites into one CV group.
7. **Irreducible ambiguity.** Near-identical images (hash/embedding) with different labels; fraction of such pairs. H: small but nonzero; sets a ceiling and argues against memorising loss.
8. **Information ceiling.** Frozen-DINOv2 + logistic regression, grouped CV, site-weighted AP per label (dev diagnostic). H: well above the weighted prevalence floor for washout/debris, lower for collapse. Compare resolutions 224x336 vs 336x504 vs 448x672 to test the "detail is lost" hypothesis.
9. **Test side (allowed only):** schema, row count (~215), id format, image size for memory/runtime planning. No test pixel statistics, no feature distributions, no site counts used for any decision.

## Validation design

**How the test split was made (inferred).** Whole sites held out: 54 train sites vs 13 test sites, disjoint, roughly an 80/20 site split (13/67 = 19%). Deployment shift = new bridges (new backgrounds, rivers, cameras, damage severity), not new time or new classes. So validation must hold out whole sites and match the ~11-site validation size per fold.

**Groups.** Do not trust only the id prefix. Build groups by union-find over: (a) shared site prefix; (b) cross-site near-duplicate links (pHash hamming <= fixed small threshold; optionally frozen-embedding cosine mutual nearest neighbours above a fixed high threshold). If diagnostic 6 finds no cross-site links, groups = sites. The script builds groups from train only and never parses a test prefix.

**Splitter.** Custom deterministic multilabel-stratified GROUP K-fold (sklearn `StratifiedGroupKFold` handles one label only): 5 folds over ~54 groups (~11 per fold). Greedy assignment: process groups by rarity (collapse-positive groups first, then debris, scour, washout), assign each to the fold with the largest deficit in positive-group counts across all 4 labels and photo count, seeded tie-breaks. Target: each fold holds ~3 collapse-positive sites (14/5 ~ 2.8), ~4 scour, ~5-6 debris, ~9 washout. Assert every fold has at least 2 positive sites per label.

**Repeats and noise.** With ~11 sites per fold the fold-level AP is very noisy (collapse rests on ~3 sites). Use 3 different split seeds (15 fold-trainings per config in dev). Report: mean per-fold macro wAP, std over folds, pooled-OOF macro wAP per seed, and a CLUSTER bootstrap over sites (resample sites, 1000 times, fixed seed) for a CI on the pooled-OOF score and for PAIRED differences between configs on identical splits. Accept a change only if the paired site-bootstrap interval for the gain excludes zero, or the gain is positive in >= 4 of 5 folds in >= 2 of 3 seeds. Because fold models differ in scale, report both mean-of-fold-AP and pooled-OOF AP; a large gap flags cross-fold calibration drift.

**Metric re-implementation.** `site_weighted_ap(y, s, site)`: w = 1/n_site computed within the evaluated set (within each held-out fold or the pooled OOF; note the weights inside a fold use fold-local site counts, which equals the true weights because whole sites are held out). Unit tests: perfect scores -> 1.0; reversed scores -> low (< weighted prevalence); constant scores -> weighted prevalence (exactly, check tie handling); base-rate predictions -> weighted prevalence; uniform weights reduce to plain AP; a hand-computed 6-photo, 2-site example; invariance to monotone transforms; a label with no positives returns NaN and is handled (skipped in fold means, never silently 0).

**Post-hoc selection checks.** Z-score reference and Platt scaling do not change ranking, so they need no nested check. Blend weights (if two families are used) are equal weights after z-scoring; no free weights fitted. Backbone/resolution/weighting choices are made once on seeds {s1,s2,s3} CV and then confirmed on a fresh split seed s4 that was never used for selection (holdout sanity). Expect the real private score to land BELOW the CV (13 sites, selection optimism).

## Overfit/underfit risks

| Risk | Evidence / reason | Mitigation |
|---|---|---|
| Site-identity memorisation | 841 photos but 54 sites; backgrounds, rivers, cameras are site-specific shortcuts | Group CV; strong augmentation (random-resized-crop scale 0.5-1, hflip, colour jitter, small rotation, no vflip); LLRD; drop-path 0.1-0.2; weight decay; EMA; few epochs with a FIXED schedule; no best-epoch selection |
| Selection noise | ~11 sites per fold; collapse rests on ~3 test sites | <= 6 configs compared; paired site-bootstrap; repeat 3 seeds; prefer simple robust recipe over tuning |
| Tiny-site over-weighting | w = 1/n gives a 1-photo site 40x the weight of a photo at a 40-photo site; heavy upsampling of a few photos | Normalise weights to mean 1 per batch; compare w^1 vs w^0.5 vs uniform in E4 as a logged ablation; ship metric-derived w^1 only if it beats uniform beyond noise, else uniform |
| Label noise from crops | Random crops may remove the visible evidence while keeping the positive label | Mild crop scale (>= 0.5 area, aspect jitter small); try crop-free (flip+color only) arm in E5 |
| Resolution underfit | Evidence is local; 224 down-sizing loses it | Native-aspect input >= 336x504; test 448x672 as the upside arm |
| Too little adaptation (underfit) | Frozen or last-block-only tuning leaves features generic | Unfreeze all blocks with LLRD (head LR 1e-3, backbone base LR ~3e-5, decay 0.75 per block); measure parameter movement |
| Too much training | 4 imbalanced tasks, 673 images per fold model | 8-10 epochs, cosine with 1 epoch warmup, EMA; verify the OOF AP curve is flat/near-peak at the final epoch (log per epoch on the held-out fold for monitoring only) |
| Loss/metric mismatch | AP is rank-only and site-weighted | Weighted BCE as base; optional pairwise ranking term as E6; no pos_weight unless it beats noise |
| Determinism noise | Flash-attention backward and bf16 are non-deterministic | Fold-ensemble averaging; double run diff; if diff is large use math SDPA kernel |
| Cross-fold scale drift in pooled OOF | Each fold model has own scale | Report mean-of-fold AP and pooled AP; z-score per fold only for diagnostics |
| Head instability for rare label | collapse has ~95 positive photos / 14 sites | Shared backbone, 4-way head, no per-label tuning |

## Recommended approach (primary + fallback)

**Primary (clean).** Fine-tune DINOv2 ViT-B/14 (`facebook/dinov2-base` via HF `transformers`, revision pinned to a commit sha to be looked up at implementation time; UNVERIFIED which sha) at native 2:3 aspect, 336x504 input (24x36 = 864 patches; positional embeddings interpolated by the model). Reasoning: self-supervised DINOv2 features are strong for local structural/texture evidence (scour holes, debris, eroded embankments, displaced spans) and fine-tune stably on small data; the representation is the biggest-value lever and it must not be undersized. Head: concat(CLS, mean-pooled patch tokens, GeM/max-pooled patch tokens) -> dropout 0.2 -> Linear(., 4). Optional (E3) variant: 4 learned label queries with one cross-attention layer over patch tokens (label-specific local evidence).
- Loss: BCE-with-logits on 4 outputs, per-photo weight w_i = 1/n_site(i) normalised to mean 1 over the training set (the metric's own weights, subject to the Q1 reading and E4 result); no pos_weight (AP is rank-only).
- Optimiser: AdamW, LLRD 0.75, base LR ~3e-5 backbone / 1e-3 head, wd 0.05, warmup 1 epoch + cosine, 10 epochs, batch 16 (grad-accum 2 if memory needs), bf16 autocast, grad clip 1.0, EMA (decay fixed) evaluated at the final epoch. Deterministic: GPU-side augmentation with a seeded `torch.Generator`, images preloaded as uint8 tensors, `num_workers=0`.
- Ensemble: 5 grouped folds, each fold model scores its held-out sites (OOF for reporting) and the test set (hflip TTA = average of logits for image and mirrored image); test logits averaged across the 5 fold models; sigmoid in float64 on the averaged logit.
- Output: per-label monotone Platt scaling fit on OOF (train only) to produce proper probabilities; ranking unchanged. JSON with the 4 keys in the stated order; full-precision floats clipped to [1e-6, 1-1e-6] only if needed (avoid creating ties by rounding).

**Second family for diversity of assumption (add only if it earns a measured gain).** ConvNeXt (timm, ImageNet-22k pretrained, e.g. `convnext_base.fb_in22k_ft_in1k`; convolutional, supervised, LayerNorm) at ~384x576. Score each family alone first; blend only if within noise of each other in quality; blend by z-scoring each family's logits with its own train OOF mean/std then averaging (no free weights). Budget permitting (see runtime), this is the planned upgrade of the primary.

**Fallback.** DINOv2 ViT-S/14 or ConvNeXt-small at 336x504, same recipe, ~2.5x cheaper: use if the primary's fixed plan measures above the runtime budget or OOMs in the dev profile. This is a plan decided OFFLINE after profiling; the shipped script contains exactly one fixed path and no runtime fallback/branching on device, memory, or clock.

## Rejected options

- Frozen embeddings (DINOv2/CLIP/SigLIP) + logistic regression/GBDT as the solution: grey/likely rejected in a fine-tuning-required task. Kept only as a dev diagnostic for diagnostics 8 and for the roadmap's quick representation ranking.
- Zero-shot CLIP/VLM scoring: inference-only; banned.
- Test-time smoothing of predictions across photos with the same test site prefix: banned (site prefix as predictive feature, whole-test aggregation).
- Pseudo-labelling, test-time adaptation, transductive clustering of the test set, BN adaptation with test batches: banned.
- ViT-L/g or full 640x960 fine-tuning: ~3-4x the cost; does not fit the 30 min limit with headroom for 5 folds.
- Detector/segmentation with synthesized boxes: no box labels; would require external or synthetic annotations.
- Auxiliary targets built from site-mates' labels (e.g. site-mean label) or site-adversarial heads using site ids as a target: sibling leakage / use of site beyond grouping.
- CutMix/Mixup as default: fabricates blended scenes with ambiguous multi-label targets; at most an E5 arm (real-image recombination, label union), not in the primary.
- Large Optuna HPO in script: 30 min limit and a noisy 54-site CV would select noise; recipe is fixed, not searched.
- Per-label separate models: 4x cost, less data per label (collapse has ~95 positive photos).
- Metadata/EXIF/file-size/id features: leak proxies for site; banned in spirit and in the description.

## Fixed work plan & runtime budget

All counts fixed; time used for logging only. Estimates are UNVERIFIED (no profiling possible here); must be profiled on an A10G-like setup before freezing the counts. Target total <= ~20 min (30 min limit, >= 30% headroom => <= 21 min).

| Stage | Plan | Est. time |
|---|---|---|
| Startup, imports, HF weight download (ViT-B ~350 MB) | fixed model id and revision | 0.5-1.5 min (network-dependent) |
| Read + decode ~1056 JPEGs, resize to 336x504, keep uint8 in RAM (~0.55 GB at 336x504) | single process, fixed order | ~0.5 min |
| Group construction (pHash union-find), fold assignment | CPU, trivial | < 0.1 min |
| 5 folds x 10 epochs of ~673 train images, batch 16 | ViT-B, 864 tokens, bf16; ~8-12 s/epoch estimated | ~7-10 min |
| Per-fold held-out + test inference with hflip TTA (~215 + ~170 imgs x 2) | ~10 s/fold | ~1 min |
| OOF metric, z-score/Platt fit, JSON write, validation | trivial | < 0.2 min |
| Primary total | | ~10-13 min |
| Optional ConvNeXt family at ~384x576 | similar 5-fold plan | +6-8 min (total ~17-21 min; only if measured to fit) |

Memory estimate (UNVERIFIED): ViT-B at 864 tokens, batch 16 with SDPA attention in bf16 ~ 10-14 GB peak; ConvNeXt-B at 384x576 batch 16 ~ 10-14 GB; both under 24 GB; keep batch 8 + grad-accum 2 as the safer fixed setting if the profile shows >18 GB. Free GPU memory between folds (`del model; empty_cache()`).

Determinism: seeds for random/numpy/torch/cuda, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` before importing torch, cudnn deterministic, `use_deterministic_algorithms(True, warn_only=True)`, seeded `torch.Generator` for augmentation and sampling, `num_workers=0`, `device="cuda"` hard-coded, no `is_available` or `cpu_count` branches, no try/except import fallbacks. Plan to run twice and diff test predictions (expect Spearman ~0.99+ per label; large swings mean the SDPA kernel or bf16 needs changing).

Validation inside the script: assert sample_submission schema and id order; assert 4 keys per JSON, finite, in [0,1]; assert row count; re-read the written CSV with `keep_default_na=False`, `json.loads` every row, re-check; raise before writing a broken file. No constant-output early fallback (a silent degraded path is worse than a loud failure).

## Metric-aware training & decode

(i) Back-solving: the metric is plain weighted AP (not invertible), but the weights are fully known: w = 1/n_site. So training should see the same weights (Q1 permitting). (ii) Hierarchical averaging: per-photo weights reproduce the site-weighted average; implementation options compared in E4: (a) weighted loss with normalised w, (b) site-balanced sampler (draw site uniformly, then photo within the site, fixed number of draws per epoch; seeded generator), (c) w^0.5 compromise, (d) uniform. All reported with paired site-bootstrap CIs; choose by evidence, default to (d) if gains are within noise (also the compliance-safe reading). (iii) Ordinal heads: not applicable (binary labels). (iv) Expected-utility decode: not applicable (independent per-label ranking; no composite output metric). (v) No thresholds, so no fixed-point algebra or bounded search is needed; the only "decode" is sigmoid of the averaged logit plus an order-preserving Platt map. (vi) Ranking: AP rewards a good global ordering; E6 adds a pairwise logistic ranking loss between positive and negative photos in the batch, weighted by w_i*w_j, with antisymmetric accumulation; accept only if it beats BCE beyond noise. (vii) No hard constraints. (viii) No aggregate counts. (ix) No taxonomy.

Cross-site ranking note: because the metric pools photos across sites per label, a model that is good within a site but whose score scale shifts with site appearance (brightness, scene) loses AP. Colour jitter augmentation and a shared 4-way head help; the between/within decomposition (diagnostic 4) tells which part is limiting.

## Structural signals

- Mirror invariance: scene mechanisms are left-right symmetric; hflip augmentation + hflip TTA (verify no text/geometry-tied inputs; there are none besides the image). Vertical flip NOT valid (water, sky, deck orientation).
- Multi-label co-occurrence: verify on train (diagnostic 2). If the matrix shows strong dependencies (e.g. collapse implies displacement-type evidence, scour precedes collapse), the shared backbone with a joint 4-way head already captures them; a label-correlation head (second-layer MLP over the 4 logits) is an optional E7 arm. Hidden taxonomy: recover only from TRAIN labels; if any exact implication holds on every train photo, record it and test adding it as a soft prior (not a hard decode rule unless it holds on 100% of train and is learned, not hand-coded).
- Site structure as free signal for VALIDATION and weighting only: groups for CV; w = 1/n for loss weighting (Q1). Not as features.
- Near-duplicate bursts inside a site: image-level duplicates in a batch reduce the effective batch diversity; if bursts are frequent, dedupe within a site for the epoch sampler (train only) or lean on the site-balanced sampler.
- Spatial structure: evidence is local and often at a specific part of the frame (water line at piers, banks near approaches). Patch-token pooling (GeM/max, or label-query attention) exploits locality better than a CLS-only head; test in E3.
- Geometry-tied augmentation: crops/rotations apply to the whole image; no auxiliary geometry inputs exist.

## Experiment roadmap

Each step has a stop criterion; do not start later steps before earlier ones are solid. Use submission credits only for the baseline, best single, ensemble and final.

1. **Contract, metric, validation (no models).** Implement `site_weighted_ap` with the unit tests above, the group builder, the multilabel-stratified group-fold splitter, the submission writer and validator. Stop when tests pass, every fold has >= 2 positive sites per label, and the writer output round-trips (`json.loads`, keep_default_na=False). Run the train diagnostics (Data findings 1-7).
2. **Cheap representation ranking (dev only, frozen features, grouped CV, 3 split seeds).** DINOv2-S/B, ConvNeXt-B in22k, a CLIP/SigLIP image tower; resolutions 224x336, 336x504, 448x672; linear-probe wAP. Purpose: pick backbone family and resolution cheaply and test the "detail is lost at low resolution" hypothesis. Stop: pick top-2 families within noise of the best; no frozen-probe model is shipped.
3. **First valid end-to-end fine-tune baseline.** The primary recipe, uniform weights, 5-fold, one seed, fixed plan, exact CLI, validator passing. Record OOF per-label wAP, per-fold std, runtime. This is the first submission candidate. Stop: wAP above the frozen probe from step 2 by more than noise; otherwise debug optimisation (check parameter movement, LR, LLRD) before moving on.
4. **Metric-aware loss and weighting (E4, E6).** uniform vs w vs w^0.5 vs site-balanced sampler; BCE vs BCE+pairwise rank. One change at a time, 3 split seeds, paired site-bootstrap. Stop: a change adopted only if it beats noise; otherwise keep the simplest/compliance-safe option.
5. **Structure and head (E3, E5, E7).** patch-pooling vs CLS-only; label-query attention; crop scale and colour-jitter strength; resolution upside arm 448x672 if runtime allows; hflip TTA gain. Stop: <= 3 arms; keep only those beating noise.
6. **Diversity.** Second family (ConvNeXt) alone vs primary; equal-weight z-score blend; also compare against 2-seed same-family as a control. Keep the blend only if it beats the best single beyond noise AND runtime stays <= ~21 min.
7. **Bounded in-script HPO: skipped by design** (30 min limit, 54-site noise). If a reviewer or a clear dev finding demands one dimension (e.g. epochs), it is a fixed small grid with its own held-out check (cross-fitted), not an Optuna study.
8. **Final fixed-plan run from a clean `working/`**, exact CLI, run twice and diff test predictions; confirm runtime on an A10G-like setup; confirm the OOF-derived stats and the submission's score distribution look alike (mean/std of logits per label vs OOF, no constant columns); then spend a credit. Compare candidate rankings against the public LB for sanity only; trust CV on disagreement (unless a split mismatch is found).

When two or three independent solvers would plausibly converge on a technique (high-resolution fine-tuned DINOv2/ConvNeXt with site-grouped CV and fold ensembling), run it as step 3 before any proxy-driven tweak.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits against this plan (plan-level; re-run on the code):
- Red: test file read only to produce one-image-at-a-time predictions, never parsed for site prefix, no stats/vocab/scaler/rank-normalisation over test. Pass by design.
- Red: no `time.time()` in any condition, loop bound, `min()`, or library timeout argument; epochs/folds/batch/workers fixed. Pass by design.
- Red: no `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks; one fixed path. Pass by design.
- Red: hard-coded tuned constants: none beyond recipe conventions; all fitted scalars (z-score reference, Platt) in-script from train OOF; no thresholds exist. If a dev comparison picked resolution/weighting, it is a documented design choice with the CV table (<= 6 configs). Flag lightly.
- Red: no external data, no synthetic data, no self-hosted weights, no GitHub sources; only HF (pinned revision) / timm weights. Pass.
- Red: strip-the-ML test passes (no rules); not raw pixels into a tabular model; not inference-only; fine-tuning is load-bearing (all blocks unfrozen, parameter-movement assertion).
- Orange: source readable, < 512 KB, no blobs; model-heavy part dominates; comments explain the reasoning; the 30 min challenge limit honoured (planned <= ~20 min).
- Orange: site prefix used in train only for CV groups (and for loss weights/sampler under reading Q1, with a group-only fallback measured); never as a feature, never for lookup, never at test time.
- Orange: `ConvNeXt`/ViT with LayerNorm only; no BatchNorm with test-batch statistics; hflip TTA per image.
- Green: seeds, fixed workers/threads, comments.

## Open questions & assumptions

Reviewer questions:
- Q1. Is using the TRAIN site prefix to compute per-photo loss weights w = 1/n_site (or to drive a site-balanced sampler) allowed, given "only for grouping"? Plan A (weighted) if allowed or if the gain over uniform is large; Plan B (group-only, uniform weights) is compliant under every reading; the measured cost of Plan B is reported from E4.
- Q2. Is averaging predictions over test photos that share a prefix explicitly banned? (Assumed banned; excluded from the plan.)
- Q3. Is a K-fold ensemble of fine-tuned models plus an order-preserving Platt map of OOF-fit scalars acceptable (train-only fitting)? (Assumed yes.)
- Q4. If a second family is added, is a z-score blend using train-OOF statistics acceptable? (Assumed yes.)
- Q5. Exact weighted-AP implementation in the grader (sklearn step-sum with `sample_weight`, versus interpolated AP, handling of a label with no positives)? Assumed sklearn.

Assumptions:
- The 4 labels are independent binary labels (not mutually exclusive), per the multi-label statement; verify co-occurrence on train.
- Test sites resemble train sites in image statistics; no test statistics were or will be used to verify this.
- Pretrained backbone revisions can be pinned to a commit sha (to be looked up when implementing; not verified here). HF download time is assumed small and is included in the budget.
- The dev sandbox may lack a GPU; fine-tune timings must be profiled on A10G-like hardware before the counts are frozen.

What could not be verified (no data, no execution): every dataset statistic beyond the stated facts; within-site label purity; duplicate structure; image-size uniformity; all runtimes and memory figures; the right resolution/weighting; and the achievable score.

Score expectation (estimate, not a promise): grouped-CV macro site-weighted AP of roughly 0.40-0.60 for the primary, with collapse lowest and washout/debris higher, floor = weighted prevalence (~0.1-0.3 per label). The private score on 13 sites is likely to land lower than CV and could swing by +/-0.10 because the collapse AP depends on ~3-4 positive sites. Reasoning: small data, local visual evidence, strong self-supervised backbone, whole-site shift.
