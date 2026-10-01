# U7 plan: dual-angle ground-footprint reconstruction (Eris Strategist)

Status of evidence. Only three files were read: the strategist procedure, `CLAUDE.md`, and the challenge text `tasks/user/U7_dual_angle_footprints.md`. No dataset was available, no test files exist for me, no internet. Everything under "Data findings" is therefore a hypothesis or a diagnostic to run, marked UNVERIFIED. Numbers that follow arithmetically from the description are marked CERTAIN. No solution code is written here.

## Contract & decision unit

**One valid answer (per case).** A JSON string `footprints_json` = list of at most 128 polygons. Each polygon has 3 to 128 `[x,y]` vertices, finite, each in [0,1], origin upper-left, x right, y down, first vertex NOT repeated at the end. `[]` is valid (no building). File: columns exactly `case_id,footprints_json`, the 400 test ids of `sample_submission.csv`, no duplicates.

**Invalid vs low-scoring.** Invalid (score zero, and the text does not say whether per case or for the whole file, so I assume the whole file): invalid JSON, wrong schema, missing/extra/duplicate ids, a polygon with fewer than 3 or more than 128 vertices, non-finite or out-of-[0,1] coordinates, more than 128 polygons, a repeated closing vertex. "Invalid polygons" is not defined further. I will guard conservatively: no consecutive duplicate vertices, non-zero area (except where a tiny polygon is deliberately kept; see below), no self-intersection (validate with shapely, which is on the allowed list, and repair or drop a polygon that fails). Low-scoring but valid: empty list on a non-empty case, speculative polygons, wrong shapes.

**Metric, term by term.**
- Polygon IoU from deterministic scanline rasterisation at 192x192 cells in the shared ground frame (CERTAIN: 225 m / 192 = 1.17 m per cell; images are 160 px = 1.41 m per px, so the raster grid is finer than the image grid). The exact fill rule (cell centre vs corner, even-odd) is not stated: assume cell-centre sampling and test sensitivity to the alternative.
- Exact-identical-polygon rule: identical valid polygons get IoU 1 even if smaller than one cell; two different polygons never match when both rasterise empty. Consequence (CERTAIN logic): every gold polygon whose 192-raster is empty can only be matched by reproducing its vertices exactly, which is not learnable. Those buildings are a recall ceiling and must not be predicted for.
- At each IoU threshold t in {0.50, 0.55, ..., 0.95}: maximum bipartite matching (one-to-one) over pairs with IoU >= t, then F1 = 2TP/(2TP+FP+FN). Mean over the 10 thresholds. Duplicated/overlapping predictions are punished as FP.
- Aggregation ambiguity (unresolved, see Open questions): per-case F1 averaged over cases, vs pooled TP/FP/FN over the whole set. The sentence "Empty predictions score zero on nonempty cases" reads like per-case scoring; the empty-gold/empty-prediction case value (1 or excluded) is not stated. Implement both aggregations and both empty-case conventions; choose decode constants that are near-optimal under both.
- Gating: none beyond validity. Strictness: with 1.41 m px, a 10 m x 10 m house is about 7x7 px; a one-pixel shift of the whole footprint gives IoU about 0.75 (CERTAIN arithmetic: 42/56). So thresholds >= 0.80 are reachable essentially only for larger buildings or with sub-pixel boundary accuracy, which is why trivial baselines score < 0.003 and why a good model's mean F1 will be low in absolute terms.

**True independent unit.** The 450 m parent tile (4 children of 225 m, both views, clipped copies of boundary-crossing footprints all share a parent). Train: 1,600 cases = 400 parents. Test: 400 cases = 100 parents, from a spatially separate eastern region. The decision unit at inference is the case (one chip pair -> one polygon set); I treat parent as the unit for validation grouping, and I never use test siblings jointly (see Compliance).

**Pipeline stages, to be diagnosed separately.**
1. Candidate coverage: does the dense prediction contain, for each gold building, a connected region that could reach IoU >= 0.5 (instance-recall upper bound of the segmentation + instance split)?
2. Scoring/ranking: given reachable candidates, is the confidence ordering good (which instances to keep; calibrated probability that an instance matches at the thresholds)?
3. Decoding: probability maps -> instances -> simplified polygons -> valid JSON. Diagnose with gold/oracle inputs (see Validation design).

## Compliance regime

**Domain classification.** Computer vision, dense prediction (classless instance segmentation / footprint polygonisation). Playbook: CLAUDE.md 6.3 + 2.3/2.4. The description does not state a category label; I assume "Computer vision" (not "Fine-tuning" and not "From-scratch"). Plan B below covers the from-scratch reading.

**Where the description is silent, and my assumption.**
- Compute: not stated. Assume the platform default: one NVIDIA A10G 24 GB, target <= 50 min, hard worst case 1 h (CLAUDE.md section 1). I design for about 33 min planned and <= 45 min with headroom.
- Runtime limit: not stated. Same assumption; a shorter limit would be handled by editing hard-coded fold/epoch constants (never by time branching).
- Domain label (CV / fine-tuning / from-scratch): not stated; assume CV with pretrained ImageNet-type encoders allowed (the description explicitly allows "pretrained visual encoders that do not reveal source identities").
- Whether the platform's general rules apply: not stated. Assume ALL of CLAUDE.md applies on top (no test fitting, no external data, no synthetic data, HF/timm weights only, fixed work plan, determinism, source < 512 KB, plain readable code, `solution.py <public_dir> <submission_out>`). Treated as hard constraints.

**Explicit bans in the challenge text (hard constraints).**
1. Reverse opaque identifiers or recover source coordinates, tile names, collection metadata. Consequence: no parsing of `case_id`, `view_*_path` for information; paths used only to open images; no attempt to infer geography.
2. Query external imagery, maps, building databases, source archives, or hidden answer files for evaluation cells. Consequence: no external data at all; pretrained weights only from HF/timm.
3. Manually annotate evaluation images. Consequence: no hand labelling or hand-tuning on test pictures; I never view test images.
4. Exploit row, filename or polygon ordering as a target proxy. Consequence: shuffle, never use case order, id order or polygon order within `footprints_json` (including as a CV grouping shortcut; I derive groups from content instead).
5. Implicit scope limit: pretrained encoders must not "reveal source identities". Consequence: use generic ImageNet-style timm backbones only; do NOT use remote-sensing encoders pretrained on this kind of archive (risk that weights encode the acquisition source). Reviewer question below.

**Allowed (explicit):** segmentation / detection / polygonisation / registration / multi-view fusion models trained on train only; deterministic image processing; geometric augmentation; cross-validation; either view independently.

**Mechanism-vs-goal check.** The ban on recovering source coordinates does not ban learning spatial context, and does not ban reconstructing the parent grouping of TRAIN cases from image/polygon content for leakage-safe CV (that is a validation device on train, not a recovery of coordinates or names). I flag it as a reviewer question and plan a content-only method that never touches ids or test rows.

**Reading chosen: compliant under every plausible interpretation.** Per-case inference only; trained network is the sole source of footprint geometry; decode constants are searched in-script on train OOF; no cross-case or cross-sibling use of test chips; no remote-sensing-source pretrained weights.

**Self-audits on the plan.**
- Strip-the-ML test: remove the trained network and nothing remains that produces polygons (decode needs the learned interior/contact/distance maps). Decode constants are few (about 5). Passes.
- No whole-test aggregation: no cross-test normalisation, no vocabularies/encoders fit on test, no stitching of the four test children of a parent (that would be a cross-row/sibling use of test rows; description deliberately keeps clipped copies within a parent, which tempts it). Banned in this plan.
- Sibling leakage in training/validation: addressed by deriving parent groups.
- Hard-coded constants must be derivable by in-script train-only search: decode constants (mask threshold, seed threshold, minimum area, simplification epsilon, border snap distance, keep-score threshold) are found in-script on OOF. Network hyperparameters (lr, epochs, loss weights) are standard fixed defaults selected by development CV among a short list; flagged in Open questions.
- Inference-only is banned in several categories: genuine training is load-bearing here (decoder from scratch, encoder fully fine-tuned, multi-head dense losses, fold ensembles).
- Torchvision detection/segmentation pretrained weights download from download.pytorch.org, which is not HF/timm, so they are not used (CLAUDE.md section 1 internet rule).

## Data findings

No data available. All items below are UNVERIFIED. Each lists the exact diagnostic I would run on TRAIN (train.csv, train_labels.csv, train images only) and the expectation, plus the decision it drives. Never run on test.

**D1. Label structure and validity.**
- Run: parse every `footprints_json`; count polygons per case (mean, median, p90, p99, max, fraction of empty cases, fraction of cases with > 128 polygons); vertices per polygon (histogram, fraction > 128, fraction with 4 vertices); check closing-vertex repetition, winding, self-intersection, zero area, duplicate polygons, overlapping polygons (pairwise IoU > 0), coordinates outside [0,1].
- Expected (UNVERIFIED): tens of polygons per chip with a heavy tail; a few percent empty chips; most polygons have 4-8 vertices (fixed sub-pixel simplification tolerance); few or no overlaps; no > 128 cases or only a handful.
- Drives: polygon cap handling, whether to enforce disjoint instances at decode, minimum-vertex policy, empty-case frequency (matters for the empty/empty convention).

**D2. Size and ceiling.**
- Run: rasterise gold at 192 (own scanline implementation); histogram of polygon area in cells; fraction with empty raster, < 4 cells, < 16 cells. Estimate foreground fraction (class imbalance of the interior mask) overall and per case. Border-touching fraction (polygons with a vertex at 0 or 1) and the fraction of gold vertices exactly on the border.
- Expected: heavy small-building mass (outbuildings, sheds) that cannot reach high IoU; foreground about 10-25% (UNVERIFIED); many polygons touch the border (clipping).
- Drives: the un-matchable ceiling (empty-raster golds), instance-balanced loss weights, border snapping rule fit on OOF, minimum-area decode constant.

**D3. Annotation-noise ceiling by jitter.**
- Run: perturb gold polygons by (a) whole-shape shifts of 0.5, 1, 2 cells, (b) vertex jitter of 0.25-1 cell, (c) 5-10% area dilation/erosion, and compute the full metric against the original gold. This is an oracle ceiling curve (no model).
- Expected: score collapses above IoU 0.8 for small buildings; it tells which thresholds a model can realistically win and how much the 0.85-0.95 thresholds are worth.
- Drives: metric-aware loss weighting by building size, sub-pixel contour extraction rather than blocky pixel polygons, expectations.

**D4. Touching/adjacent structure.**
- Run: fraction of gold polygons that share an edge or are within 1 cell of another polygon; gap-width distribution between neighbours; fraction of footprints in connected blobs of 2+ buildings after rasterising the union at 192.
- Expected: a substantial fraction touches (row houses, attached garages).
- Drives: whether a contact/boundary head and watershed split are needed (likely yes) and how much a merge error costs (oracle D6b).

**D5. Views.**
- Run: per view, channel means/stds, histogram overlap between view_a and view_b, Laplacian-variance sharpness; check whether view_a is always the sharper/nadir-like one across all train cases (otherwise A/B roles are mixed per case); phase-correlation / gradient-based global shift between views per case: histogram of shift vector (dx, dy) and its direction consistency; exact or near duplicates between view_a and view_b; JPEG artefact level.
- Expected (UNVERIFIED): a single campaign gives a nearly constant lean direction for the severe view with roof displacement of several to a few tens of pixels depending on height; view_b blurrier/less contrasty along the look direction; A=nadir, B=oblique consistently.
- Drives: (i) receptive field/pyramid depth needed to cover roof displacement; (ii) which flips/rotations are valid augmentations (flips parallel to the lean axis keep the lean direction; others reverse it); (iii) whether to train swap-invariantly (random A/B swap in training, swap-averaged at test) if roles are mixed or if I decide robustness is worth it.

**D6. Oracles through the decoder (before any modelling).**
- (a) Gold masks at 192 -> semantic raster -> polygonise -> rasterise -> full metric: expect near 1.0 minus losses from simplification and the 128-vertex/128-polygon caps; also report decoder runtime per case (needed for the budget). Must reproduce convention (corner vs centre) exactly; any half-cell offset shows up here.
- (b) Gold semantic mask only (touching buildings merged) -> connected components -> metric: measures the cost of instance merging with no model. Compare with gold + contact map.
- (c) Gold with the 1-cell border band removed vs snapped to the border: measures the value of border snapping.
- Drives: decoder design, tolerance choice, border policy.

**D7. Parent-group recovery (needed for leakage-safe CV).**
- Run, on train content only: for every ordered pair of chips (i, j), compare the right-edge strip of i with the left-edge strip of j (and bottom vs top) in view A and view B (a 1,600 x 1,600 descriptor distance matrix of a few-pixel strips; trivial cost); add a second evidence term: gold polygons clipped at the shared edge should have matching vertical (or horizontal) extents on both sides (clipped copies of one building). Mutual best matches with a clear margin give confident adjacencies; connected components restricted to size <= 4 give 2x2 sibling sets; ambiguous chips become singletons. Report the fraction assigned to groups of exactly 4 (target about 100%, expectation UNVERIFIED), margin distribution, and chain-through to neighbouring parents (the unused-column gap does not separate train parents from each other).
- Drives: grouping for CV folds. If recovery is poor, fall back to appearance/density clustering for blocked folds and treat CV as optimistic.
- Not a violation of the ban on recovering source coordinates: no coordinates, names or metadata are recovered, only a content-based partition of train chips.

**D8. Spatial heterogeneity inside train (proxy for west -> east shift).**
- Run: describe each train case by simple label/pixel summaries (building density, median footprint area, mean brightness, green fraction); cluster (k=3-5) and check how much per-cluster score of a first model varies. Also build contiguous parent blocks via the adjacency graph (spectral/graph partition of D7 adjacencies) to emulate "train west -> eval east".
- Expected: neighbourhood styles differ materially (dense urban vs sparse); the east region will differ in density/roof type/vegetation. Cannot be measured on train; only its sensitivity can.

**D9. Ordering / id audit (to confirm we are NOT using it).**
- Run: check, on train only, whether label statistics trend with row order or id string, purely to know a leak exists so that shuffling/ignoring is deliberate. Do not use it.

**D10. Duplicates.**
- Run: perceptual-hash and byte-hash within train; exact view_a==view_b; identical polygon sets across cases (indicates the same parent used twice). Drives grouping; also catches empty/blank chips.

## Validation design

**Reproducing the split from the description (train structure only).** Test = complete 450 m parents from a different (eastern) region; train = western region; one unused tile column between. So the shift is spatial/geographic: different building density, size, roof materials, vegetation, possibly sun/lean differences. Within train, children of one parent and clipped copies of the same building are strongly correlated.

**Primary CV.** 5-fold GroupKFold over recovered parent groups (D7), about 80 parents per fold. Folds assigned deterministically from the seed. Report mean and std across folds; repeat with a second split seed during development (accept a change only if it beats noise: paired same-fold difference > 1 standard error and positive in >= 4 of 5 folds and in both split seeds).

**Spatial-block CV (stress / test-like).** Contiguous parent blocks from the adjacency graph (D8), 5 blocks. Used for model-family decisions that interact with shift (augmentation strength, view fusion, A/B swap), not for every tweak. One block (fold 4 of the primary partition) is a holdout sanity fold untouched by any selection until the final check.

**Bias direction of each proxy.**
- Random-group CV by parents: optimistic against the real shift (same-region neighbouring parents share styles, and any chip adjacency not recovered leaks clipped copies); likely a few to ~15% relative optimistic (UNVERIFIED guess).
- Spatial-block CV: mildly pessimistic (removes whole neighbourhoods from training, and train regions are small) but closer to the real shift in kind; the real eastern shift could be larger or smaller than my blocks.
- Imperfect group recovery makes both optimistic. Real number expected to land below the primary CV.

**Metric re-implementation and unit tests.** Own scanline rasteriser at 192, own matching (scipy maximum bipartite matching on the thresholded IoU graph), F1 per threshold, both aggregation variants, both empty-case conventions. Tests: perfect -> 1.0; empty on non-empty -> 0; both empty (document value); one polygon duplicated -> precision drop at every t; reversed/shuffled polygon order -> unchanged; shifted-by-k-cells curve matches D3; full-chip polygon -> about 0; constant base-rate (mean shape) -> about 0 (should be < 0.003 as the description says); tiny identical polygon with empty raster -> IoU 1; different tiny polygons both empty -> no match.

**Post-hoc selection needs its own held-out check.** Decode constants and the instance-keep rule: fitted on OOF of four folds, scored on the fifth (cross-fitted), reported as the honest number; the final shipped constants fit on all OOF. Fixed-point keep threshold (see decode section) cross-fitted too.

**Stage diagnostics (on OOF).** (1) Candidate coverage: share of gold buildings with any predicted instance at IoU >= 0.5 (and >= 0.75); (2) ranking: AUC of the instance scorer for "matches at 0.5" and for "matches at 0.75"; (3) decode: gold-through-decoder oracle (D6a) and merge/split errors.

**Zero-shot/frozen yardstick.** Report the frozen-encoder + trained-decoder result under the same folds as the floor the fine-tune must beat, plus view ablations on the same folds: A-only, B-only, A+B (paired).

## Overfit/underfit risks

**Overfit.**
- Few independent groups (400 parents; effective sample about 400 despite dense pixel labels). Mitigation: parent-grouped CV, geometric + photometric augmentation, EMA, fixed short schedule (no validation-triggered stopping), weight decay, 5-fold ensemble for variance reduction, view-dropout to prevent over-reliance on one view.
- Spatial overfit to the western region's styles (colour, building types). Mitigation: independent photometric jitter per view (brightness/contrast/gamma/hue small), mild scale/blur jitter, spatial-block CV as stress test, no features that encode place.
- Decode knobs tuned on one OOF set (about 5-6 constants). Mitigation: low-dimensional coordinate search over coarse grids, mid-range plateaus, cross-fitted check.
- Selection on noisy CV between architecture options. Mitigation: paired same-fold comparisons, two split seeds, accept only above noise, keep the lean design if no clear winner.
- Instance scorer overfit: tiny logistic model with few features, cross-fitted.

**Underfit.**
- Resolution loss: 160 px input with stride-32 encoders blurs small buildings; strict IoU needs sub-pixel boundaries. Mitigation: 2x bilinear (deterministic) upsample of inputs (320 px), full-resolution U-Net decoder with skip connections, contour extraction on the probability map at sub-cell precision, not on a thresholded blocky mask.
- Receptive field too small to relate roof/facade evidence in the oblique view to the ground footprint: shifts of several to tens of pixels (UNVERIFIED). Mitigation: pyramid depth covering the measured D5 displacement; fusion at every scale, not only early.
- Capacity ladder (F2.1): frozen strong features alone give poor pixel-precise boundaries; full fine-tune of a base-size model risks memorising parents. Plan: report frozen-encoder + trained decoder as rung 1, then full fine-tune with layer-wise LR decay (encoder lr about 5-10x below decoder) as rung 4 (dense prediction usually needs encoder adaptation across the aerial domain gap); choose by grouped CV. Log encoder parameter change norm and CV gain over the frozen rung (also the compliance layer).
- Loss mismatch: pixel-BCE is dominated by large buildings, while F1 counts each building equally and the thresholds reward boundaries. Mitigation: instance-balanced and boundary-weighted loss variants; soft-IoU/Lovasz term; compare under CV.
- Information lost by decode: simplification, 128-vertex cap, small-building dropping; quantified by D6a.
- Empty/near-empty chips handled by a probability map that is simply low everywhere; the keep rule must not hallucinate polygons (precision).

## Recommended approach (primary + fallback)

**Primary: two-view late-fusion U-Net with learned instance maps, learned instance scorer, metric-aware decode.** (Lean first; extras only after measurement per roadmap, F2.13.)

- Representation: both 160x160 RGB views, each upsampled 2x (bilinear, fixed) to 320x320, normalised with fixed ImageNet stats (a constant, not fitted to test). Shared timm backbone pretrained on ImageNet (ConvNeXt-Tiny or ResNet-50 class; pin revision; features_only pyramid), view-specific input handling (separate lightweight stem/BatchNorm or a learned per-view embedding) because the two views differ strongly; per-scale fusion by concatenation of view features followed by a 1x1 mix, so ground-aligned evidence (the shared ground frame guarantees base-of-building alignment across views while roofs/facades shift) is combined at all scales. U-Net decoder written in plain torch (segmentation_models_pytorch is not on the allowed list); nearest-neighbour or pixel-shuffle upsampling to avoid non-deterministic bilinear backward kernels.
- Heads (dense, shared decoder, translation-equivariant): (1) interior probability, (2) contact/boundary probability for separating touching buildings, (3) normalised distance-to-boundary (per instance) regression for seeds; optional (4) per-pixel "border-clipped" cue is not needed. No flatten/global-pool heads.
- Targets: rasterise gold polygons at 2x the image grid (own rasteriser, same convention as the decoder) to build interior, contact and distance targets. Keep tiny golds in training masks (they are real) but they are excluded from decode by the area constant.
- Loss: BCE + soft Dice/IoU on interior, boundary-weighted pixels, instance-balanced pixel weights (weights ~ 1/sqrt(instance area), clipped) so small and large buildings count comparably, BCE on contact, L1/Huber on the distance map; all variants compared under CV. View-dropout (randomly blank one view with p about 0.15, fixed in plan) for robustness and for built-in A-only/B-only ablations.
- Augmentation: geometric transforms applied jointly to both views and the targets. Valid transform set determined from D5: flips that keep the lean direction are always on; flips/rotations that reverse the lean direction are tested per F2.6 (learn per-view embedding, or measure the lean mapping on train and assert it) and kept only if grouped-CV neutral or positive. Independent photometric jitter per view. Small scale/translation jitter within the fixed GSD, light blur/noise on the sharper view for domain robustness.
- Training recipe: AdamW, cosine with warmup, AMP, channels_last, gradient clipping 1.0, EMA of weights, batch about 16, fixed epochs about 18 per fold, no early stopping and no wall-clock use, five fold models (one per CV fold). Whole dataset kept as uint8 tensors on the GPU, augmentation on GPU, no DataLoader workers (determinism and speed).
- Inference: per fold model, flip-TTA over the validated symmetry subset, average probability maps over TTA then over the five fold models (no per-instance merging across models). Same procedure applied to OOF (single fold model + TTA) for calibration.
- Decode (deterministic function of the learned maps; constants fit in-script on OOF): seeds from the interior probability minus contact probability above `t_seed`; marker-controlled watershed constrained to interior above `t_mask`; drop regions below `min_area` (default aligned with the un-matchable tiny-gold ceiling); sub-cell contour by marching squares on the interior probability (padded by replication so contours reach the chip border), polygon simplification with epsilon `eps` (Douglas-Peucker or similar), border snap within `d_snap` to exactly 0/1 for regions touching the chip edge (structural property: gold is clipped to the cell boundary, check D2), enforce disjointness, vertex <= 128, polygons <= 128 (keep highest-confidence first), orientation/validity via shapely, clip to [0,1], never repeat the first vertex.
- Keep rule and scorer (roadmap step 7, only after the lean decode is measured): small logistic model predicting, for each decoded instance, E[number of IoU thresholds hit]/10 from within-instance features (mean/min interior probability, boundary sharpness, area, compactness, touches-border, TTA/fold disagreement, A-only vs B-only vs fused agreement), trained on OOF with gold matching, cross-fitted. Keep instances with predicted utility above F*/2 where F* is the achievable mean F1 (fixed-point; justified by the F-measure optimality condition that adding an item helps iff its match probability exceeds about half the optimum F). With F* about 0.3 this keeps many borderline instances, which is the opposite of an intuitive 0.5 cut.
- Expected hit on compliance: clean (CV playbook 6.3; hand-engineered decode on top of learned maps is polygonisation, which the description explicitly allows).

**Fallback (and ablation arm): single-stage early-fusion, lighter model.** Stack the two views as 6 channels into a smaller backbone (ResNet-34-class, first-conv weights duplicated and rescaled), same heads and same decode but without the learned instance scorer; plus a nadir-only (view A) variant. This is what ships if the late-fusion model shows no paired CV gain over it, or if runtime must be cut (hard-coded constant change, never branching). Also the minimum-risk valid output: fixed decode with fixed defaults is NOT a fallback written into the script as a silent degraded path; a crash must be loud.

**Plan B for the from-scratch reading.** If a reviewer rules that pretrained weights are not allowed in this challenge: same architecture with a ResNet-ish encoder trained from scratch with GroupNorm/BatchNorm, stronger augmentation, longer schedule within the same budget, no pretrained tokenizer-like components; expected clearly lower score (UNVERIFIED, perhaps 20-40% relative).

**Plan C if A/B roles are mixed in train (D5).** Random A/B swap augmentation + swap-averaged TTA; the network infers view type from content (sharpness/lean). If A is consistently nadir, also test swap-augmentation as a robustness variant: if CV cost is within noise, ship swap-robust (works under both readings of the view convention); otherwise ship ordered and record the assumption.

## Rejected options

- Pure pixel/edge/threshold/rectangle/centroid priors: fail the strip test, and the description reports them below 0.003.
- Mask R-CNN / torchvision detection with pretrained torchvision weights: weights not from HF/timm (not allowed source), 28x28 mask resolution is too coarse for strict IoU on 7-12 px buildings, heavier to make deterministic.
- Direct polygon/vertex-sequence prediction (HiSup/PolyWorld/Mask2Former-type heads): high engineering cost, not first-order for the budget; revisit only after the lean design plateaus and only if time remains (F2.13).
- SAM or any promptable foundation model as a mask generator: inference-only, classless masks need a building classifier on top; fails the genuine-training spirit; large.
- Remote-sensing-pretrained encoders: risk of revealing source identities (explicit scope of allowed encoders), so skipped.
- Cross-sibling stitching of test children of a parent (merging clipped copies, using neighbouring chips as context): whole-test/sibling aggregation, the dataset construction invites it, banned here.
- Using row order, filename, or polygon order, or parsing ids (explicit ban).
- Hand-written rectification of footprints (snap to right angles) as a replacement for learning: the model must produce the geometry; only light learned/constant-fit simplification is used.
- Blending many near-identical models: low diversity of assumption; only a second assumption-diverse family (e.g. a different backbone with early fusion) is considered, after measurement.
- Frozen-encoder-only heads as final: poor boundaries; used only as the yardstick.
- Public-LB-driven tweaking: forbidden by playbook; CV is the source of truth.

## Fixed work plan & runtime budget

All counts below are fixed constants; time is logged only. Single device `cuda`, no environment-dependent switches. Estimates are for an A10G and are UNVERIFIED until profiled (CLAUDE.md 3.7: profile one epoch, then hard-code).

| Stage | Fixed plan | Est. runtime |
|---|---|---|
| Load 3,200 train + 800 test JPEGs to uint8 tensors, parse labels, validate input schema | single process | ~1 min |
| Parent-group recovery (edge-strip distance matrices) + fold assignment | 1,600x1,600 matmuls | ~0.5 min |
| Target rasterisation (2x grid), once | vectorised, CPU | ~1 min |
| 5 fold trainings | 5 x 18 epochs x ~1,280 chips x 2 views at 320 px, batch 16, AMP | ~20 min (about 13 s/epoch, to be profiled) |
| OOF inference with TTA (1,600 chips) | per fold model on its held-out chips | ~1 min |
| Decode-constant search | coordinate descent over about 6 constants, about 20 evaluations on a fixed 800-chip OOF subset, decode ~20 ms/chip | ~5 min |
| Instance scorer + fixed-point keep rule (cross-fitted) | small logistic | ~1 min |
| Test inference: 5 models x TTA x 400 chips + decode + validation + write | | ~2 min |
| Overheads/logging | | ~2 min |
| **Planned total** | | **about 33 min** (budget <= 45 min incl. 30% headroom; ceiling 50 min) |

Memory: dataset tensors about 0.25 GB uint8 at 160 px; training batch 16 x 2 views x 320 px with a ConvNeXt-Tiny-class encoder in AMP about 8-12 GB peak (UNVERIFIED); fits 24 GB with headroom; fixed batch size, free fold models between folds. Decode on CPU with a fixed number of processes only if profiled necessary (single process by default).

Robustness inside the script: assert file existence, shapes (160x160x3), finite values, row count equals `sample_submission.csv`, id order equal; validator rejects any polygon violating the contract (3..128 vertices, coords finite in [0,1], no repeated first vertex, polygons <= 128, no self-intersections), re-read the written CSV and re-validate. No silent fallbacks: a failing component raises with stage context. Seeds fixed (numpy, torch, cuda, PYTHONHASHSEED); `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic`; avoid `F.interpolate` bilinear/`grid_sample` backward kernels in the decoder; evaluate any residual non-determinism by running twice and diffing (roadmap 9).

## Metric-aware training & decode

(i) Back-solve the metric: the score is an average over IoU thresholds of one-to-one match F1, so boundary accuracy dominates the upper thresholds; the learnable quantity is a precise per-instance footprint mask. Training uses the same rasterisation convention as the decoder (verified by D6a), 2x finer than the 192 raster, so the net can place boundaries at sub-cell precision.
(ii) Weight the loss with the metric's own weights: per-instance weights (each building counts the same in F1) via instance-balanced pixel weights; replicate hierarchical averaging in validation (per-case F1 then mean; also pooled).
(iii) Boundary weighting and soft-IoU/Lovasz term for strict thresholds; compare under CV.
(iv) Composite metric -> expected utility decode: each decoded instance has an expected utility (expected number of thresholds hit / 10), predicted by the cross-fitted scorer; keep instances above F*/2 with F* found by fixed-point iteration on OOF (algebraic, not a grid corner search); bounded by the empirical polygon-count prior (cap at 128, cap at the 99th percentile of train counts).
(v) Constraint enforcement inside the valid output space: one-to-one by construction (disjoint instances from the watershed), border snapping for clipped footprints, vertex cap, validity repair via shapely; the constraint step combines with, not replaces, model evidence.
(vi) Instance separation: contact head and watershed turn pixel evidence into the instance set; merge/split errors measured against the D6b oracle.
(vii) Calibration: interior probability maps trained with BCE are the primary calibrated evidence; the per-instance scorer is cross-fitted; report its calibration (reliability) on held-out folds.
(viii) Decode-constant search: `t_mask`, `t_seed`, `min_area`, `eps`, `d_snap`, and the keep threshold, on coarse grids, coordinate descent from fixed defaults, evaluated on OOF with both aggregation variants; the shipped value is the one within one standard error of the best on both (robustness to the metric ambiguity), preferring mid-range plateaus.
(ix) Training/inference consistency: decode constants calibrated on single-fold-model OOF with the same TTA; at test, the five-model average is smoother. Dev check: score a two-model average on a subset of OOF with constants fitted on single-model OOF; if loss exceeds noise, either recalibrate on an ensembled OOF (bagged OOF with several seeds) or choose a full-data refit (5-fold dev confirms), per F2.1 "measured transfer problem".

## Structural signals

- Shared ground frame: ground-level base-of-building evidence is pixel-aligned across views; roof/facade parallax differs. Use per-scale fusion, per-view stems, and let the network learn the displacement (do not hand-code an offset). Measure displacement direction/magnitude on train (D5) to size the receptive field and decide valid flips.
- Lean-direction symmetry: flips that preserve the lean direction are valid augmentations; others require a learned per-view embedding or are rejected by diagnostic (F2.6, F2.16). Verify claimed symmetry on train by comparing CV with and without.
- Clipped boundary: gold footprints are clipped to the cell boundary; snap predicted contours that reach the chip edge to 0/1 (constant `d_snap` fit on OOF); check D2 (fraction of gold vertices on the border).
- One-to-one matching: duplicates are FPs; decode produces disjoint instances; verify on train that gold footprints rarely overlap (D1).
- Sub-cell un-matchable golds: use `min_area` to avoid predicting tiny polygons that cannot match under the empty-raster rule.
- Fixed sub-pixel simplification of gold: gold vertex statistics (D1) set the target simplification level; `eps` fit on OOF; polygons typically low-vertex, so the 128-vertex cap should never bind.
- View complementarity: view-dropout trains each stream to carry evidence alone, and OOF A-only/B-only/fused disagreement is a feature for the instance scorer (spatial evidence agreement across views is a natural confidence cue; not a test-set statistic since each case uses only its own two views).
- Parent structure: used only on train for grouping and leakage control; never used in test.
- No label-derived priors from train are used as place proxies; no count/density prior across test.

## Experiment roadmap

Stop criteria: advance only when the step's gate passes under paired same-fold comparison with two split seeds. Use credits only for baseline, best single, ensemble, final (6 credits per problem).

1. Contract, metric and validation correct and unit-tested. Rasteriser/matcher/aggregations implemented and tested per Validation design; oracles D6a-c run and recorded. Gate: gold -> decode -> metric reproduces near 1.0, convention consistent. Also D1-D5, D7 diagnostics.
2. Strongest cheap baseline, end-to-end and valid: view A only (or fused early-fusion ResNet-34-class), one dev fold, default decode constants, valid CSV with the validator. Gate: valid file, metric well above 0.003 on the dev fold; per-stage recall/ranking/decode diagnostics logged. Candidate for credit 1.
3. Representation and structure: paired comparison A-only vs B-only vs fused late-fusion (same folds); frozen vs LP-FT/last-stage vs full fine-tune with layer-wise LR; resolution (1x vs 2x upsample); receptive field vs measured parallax; flip policy and A/B swap. Gate: fused beats best single view by more than noise; fine-tune beats frozen yardstick by more than noise (record parameter-change norm).
4. Metric-aware loss and decode: instance-balanced weights, boundary weighting, Lovasz/soft-IoU, contact head and watershed, sub-cell contours, border snap. Gate: each earns a measured gain over noise or is removed.
5. Decode constants and keep rule: coordinate-descent constants; scorer + fixed-point keep threshold; cross-fitted evaluation under both aggregation variants. Gate: cross-fitted gain over the fixed-default decode. Candidate for credit 2 (best single).
6. Diversity (only if time budget and gain): second family with a different assumption (early fusion/different backbone); score each alone first, blend only members of close quality, average probability maps (z-score/train-reference scaling), not raw instance lists. Stop if gain < noise. Candidate for credit 3.
7. In-script bounded search: fixed small grids for decode constants only (the search is in the script with fixed trial count); no network HPO inside the script beyond fixed defaults.
8. Spatial-block stress test and sanity fold: holdout fold and blocked CV score reported once; compare with random-group CV to calibrate the expected shift penalty (also per-cluster scores from D8).
9. Final fixed-plan run from clean `working/` with the exact command; run twice and diff submissions (non-determinism check); sanity check of test prediction summary (polygons per case, area distribution, empty fraction) against OOF only as a bug catch, not tuning. Candidate for credit 4. Compliance audit.

**Expected score range (estimate, not a promise).** Trivial baselines are below 0.003. For a two-view U-Net at 1.41 m/px with strict IoU averaging over 0.50-0.95, I would expect OOF mean F1 of roughly 0.20-0.40 (reasoning: F1 at IoU 0.5 of about 0.5-0.7 is typical for building footprint segmentation from overhead imagery; averaging across thresholds up to 0.95 usually scales that by about 0.4-0.55 because strict thresholds are hard for small buildings; sensitive to unlabelled/ambiguous annotations). Private score expected lower than OOF by about 10-25% relative because of the west -> east shift (reasoning above), so about 0.15-0.35. Confidence: low; D3 jitter ceiling and the first fold will tighten this.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits against this plan:
- Test file read only to produce per-sample predictions: yes. No stats, vocab, scaler, clustering, dedup, rank-normalisation or pseudo-labels over test; no sibling stitching. Test-time normalisation uses fixed ImageNet constants (not fitted to test).
- Time used only for logging: planned yes. No `if elapsed`, no `timeout=`; all counts fixed.
- No `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks, or try/except that change models or work: planned yes.
- Hard-coded constants: decode constants and the keep threshold are found in-script on OOF; the network defaults (lr, epochs, loss weights) are standard fixed values chosen from a short development list; flagged (open question 7).
- External data: none; pretrained weights ImageNet-type from HF/timm with pinned revision; no remote-sensing-source weights; no torchvision-hosted weights; no GitHub downloads; libraries limited to numpy, pandas, scipy, scikit-learn, opencv/PIL, shapely (listed), torch, timm, (skimage only if confirmed to be available, otherwise cv2/scipy).
- Strip-the-ML test: passes (all geometry flows from the learned maps).
- Raw pixels into a tabular model: none. The instance scorer uses decode-derived features from the CNN (small calibration head on top of a trained model).
- Source: single plain `solution.py` < 512 KB, readable, with comments explaining reasoning at each step; no blobs, no exec/eval.
- Explicit bans: no id reversal or metadata recovery (ids/paths opened, not parsed); no external imagery/maps/databases; no manual annotation of eval images (never viewed); no ordering exploitation (case order shuffled for folds, polygon order unused).
- Output contract: validator enforces columns, ids and order, JSON parseable, <= 128 polygons, 3..128 vertices, finite [0,1], no closing duplicate, no self-intersection; re-read after writing.
- Determinism: seeds, deterministic algorithms, nearest/pixel-shuffle upsampling in decoder, run-twice diff.
- Open compliance risks: (a) group recovery from content on train (low risk, flagged); (b) pretrained encoder identity scope (low risk with ImageNet-type weights); (c) exploiting data-generation structure via border snapping and clipped geometry (it is a documented property of the labels, learned/fit in-script; low risk).

## Open questions & assumptions

Questions for a reviewer (each with plan under both readings):
1. Metric aggregation: per-case F1 mean, or pooled over the set? Both implemented; constants chosen robust to both. Empty gold and empty prediction: score 1 or excluded? Implemented both; model keeps the empty output whenever evidence is low.
2. What counts as an "invalid polygon" (self-intersection, zero area, duplicate vertices) and does one invalid polygon zero only its case or the whole file? Conservative guard: none of those occurs in the output.
3. Does "pretrained visual encoders that do not reveal source identities" exclude remote-sensing-pretrained encoders? Plan uses generic ImageNet encoders only; remote-sensing encoders would be a possible upside but are skipped.
4. Is reconstructing the 2x2 parent grouping of TRAIN chips from image-edge continuity and clipped-polygon matching acceptable for leakage-safe CV (no coordinates, names or metadata recovered)? If not, fall back to random folds plus appearance/density clustering for blocked folds, accepting more optimistic CV.
5. Domain label: if the challenge is a from-scratch one, Plan B (no pretrained weights); if labelled Fine-tuning, the full fine-tune rung is the compliance layer and frozen-feature heads are not allowed as final.
6. Do the platform's general rules apply here (fixed work plan, no wall-clock branching, determinism, HF/timm-only weights)? Assumed yes.
7. Is selecting a short list of fixed network defaults by development CV (not pasted tuned HPO) acceptable? Assumed yes; decode constants are searched in-script.
8. Is flip/rotation augmentation consistent with the dataset's physical view geometry? Handled by diagnostic (D5, F2.6); not a rule question.

Assumptions made (all UNVERIFIED): A10G, 24 GB, <= 50 min target (1 h worst case); A=near-nadir and B=oblique in all cases and in test the same; image 160 px = 1.41 m/px (CERTAIN arithmetic); fill rule cell-centre; pooled vs per-case ambiguity covered; roof displacement and lean direction roughly constant in the campaign; the east region differs materially from the west; timm available; shapely and scipy available (shapely is in the allowed list; scipy core Kaggle).

Cannot be verified without data: group-recovery success; polygon/size/density distributions; displacement magnitude and direction; whether A is always nadir; the true IoU-threshold reachability; runtime per epoch; the eventual score band.
