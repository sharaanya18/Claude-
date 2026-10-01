# Eris plan: T1 Field-capture burst midpoint recommendation

Status of evidence: no dataset was available for this run. Everything under "Data findings" is a diagnostic to run on TRAIN plus an UNVERIFIED hypothesis. Numbers marked (est.) are arithmetic or priors, not measurements. Only the three input files (agent file, CLAUDE.md, task description) were read.

## Contract & decision unit

**One valid answer.** For each test `id`, a JSON list of exactly 2 distinct candidate identifiers from {c0..c5}, best first, in column `ranked_candidates_json`; the file has `id, ranked_candidates_json`, one row per test id, in sample order. The exact JSON encoding (names `"c3"` vs integer indices vs bare list) must be copied from `sample_submission.csv` and enforced by an in-script validator (CLAUDE.md section 5). Invalid vs low-scoring: wrong count, duplicate candidates, unknown names, wrong row count or ids are invalid (dead last). Any valid pair is merely scored.

**Metric, term by term.** Per row: 1.0 if the gold candidate is rank 1, 0.2 if rank 2, else 0. Mean over rows. Rank-only: the metric sees only the ordering of the top 2, not calibration. Expected row score = p(rank1) + 0.2 p(rank2), where p is the model's posterior that a candidate is the gold. Sorting candidates by posterior is therefore exactly optimal, so no decode constants are needed. Random = 0.2 (1/6 + 0.2/6). No gating or hierarchical averaging; the row is the unit of scoring.

**True independent unit.** NOT the row. Bursts chain across rows (R of row i is L of row i+1), so the independent unit is a connected chain of rows (a contiguous capture session), and probably a coarser scene/site. No group column is provided: groups must be derived from image content (see Validation design). Each row is a decision problem: a set of 6 interchangeable candidates plus two anchors (L, R); the output is a 2-element ordered selection.

**Why the problem is hard to read at face value (UNVERIFIED, key structural point).** A burst has 8 frames: L = frame 0, R = frame 7, candidates = frames 1..6. The temporal midpoint 3.5 lies exactly between frames 3 and 4, which are equally close in index-time. So either (a) the gold is defined by true capture timestamps (irregular spacing breaks the tie, noisy label), or (b) the gold is a fixed convention (e.g. the lower-median frame 3). The 1.0/0.2 payoff looks designed to give partial credit for finding the "central pair". Implications:
- If the label is a coin flip within the central pair and the model finds the pair perfectly, the ceiling is 0.5*1.0 + 0.5*0.2 = **0.60** (est.). If the tie-break is learnable (convention or timestamp-driven visual cue), the ceiling approaches 1.0.
- This must be settled by train diagnostics before choosing whether L/R swap is a valid augmentation (D5 and the swap test below).

**Pipeline stages (diagnose separately).**
1. Candidate coverage: trivially 100% (the gold is always among c0..c5); the useful analogue is "gold within the central pair" recall.
2. Scoring: per-candidate posterior of being the midpoint frame, conditioned on L, R and the other candidates.
3. Decode: sort by posterior; optional permutation-marginal decode (see Metric-aware training & decode).

## Compliance regime

**Domain.** Computer vision, ranking, small data (~1,250 train rows, ~8,750 unique photos). CLAUDE.md section 6.3 applies, plus the "trained model must do the learning" rule (section 2.1).

**Allowed.** General-purpose pretrained backbone weights from HF/timm, genuinely fine-tuned in-script; hand-engineered pairwise features fed into the trained head; per-sample TTA (flip, crop views, and L/R swap only if the swap-symmetry test passes); fixed-count CV and ensembling inside the script; recombining real labelled units from train (chain-derived tasks), subject to reviewer confirmation.

**Banned or to avoid (explicit in CLAUDE.md or implied by the description).**
- No external data, no synthetic/generated frames (e.g. no diffusion/optical-flow interpolation to fabricate midpoints), no LLM-generated data.
- No test-set use beyond per-row inference. In particular, **do NOT chain test rows** (reconstructing that this row's R equals another test row's L to get longer context, or matching test photos to train photos) and do not run any near-duplicate/leak trick across rows. This is the "dedup/leak tricks using test rows" and whole-test aggregation ban. It costs the neighbour-context signal (unmeasurable, likely modest).
- No metadata leakage: EXIF timestamps, JPEG headers, zip member order, file names, file sizes, row/id order as features. These are audited (D1) but not used. Strip-the-ML view: if EXIF time solves the task, that is a rule-exploit, not ML. Raise to reviewer if found.
- No hand-written alignment/registration/optical-flow midpoint solver as the solution (rule-based; also defeated by independent random crops/flips/photometric changes).
- No pseudo-labels, no test-fitted normalisers, no ranking/z-scoring across the test file. Each test row's output is a function of its own 8 images plus train-fit weights.
- No wall-clock branching, no env-dependent fallbacks, fixed seeds, device "cuda" (CLAUDE.md section 3).

**Grey.** Frozen-embedding + small head: fine as a dev scaffold and cheap probe, NOT as the shipped model. Partial fine-tune of the backbone (last blocks) is genuine training; unfreezing depth is a dial to keep the "load-bearing training" argument obvious. TF-IDF-style/GBDT on embeddings: not used.

**Strip-the-ML test.** Remove the trained backbone/head: what remains is a slot prior (0.2) or a hand-made similarity heuristic (D6). Plan to report the best no-ML heuristic score; the trained model must clearly exceed it, which demonstrates the ML is load-bearing.

**Ambiguity needing the reviewer** (listed in Open questions): tie definition of the gold, chain-recombination labels, metadata.

## Data findings

All items are diagnostics on TRAIN only (test used only for schema, row count, id format, image-size stats for runtime/memory). All stated outcomes are hypotheses, UNVERIFIED.

D1. Schema and format. Columns of train.csv/test.csv (image path columns for L, R, c0..c5, gold column and its format: name vs index), sample_submission encoding of the JSON, zip layout and extraction path, row counts (~1,250 train, few hundred test), image-size and aspect-ratio distribution (min/median/max, per image role), JPEG quality, presence/absence of EXIF timestamps and any per-file naming pattern correlated with time (audit only; a finding goes to the reviewer, not into features). Hypothesis: sizes vary because of random crops; EXIF is stripped.

D2. Gold slot distribution over c0..c5 (counts, chi-square vs uniform). Hypothesis: uniform (candidates are shuffled), so the slot prior is worth 0.2. A skew would be a leak-like artefact: not used as a feature, and removed anyway by slot-permutation augmentation, but it would be reported.

D3. Chain structure. Embed every train photo with a frozen DINOv2 (train only), flip-invariant (max over original/flipped similarity). For each row's R, find the nearest L across all rows; look at the similarity histogram (hypothesis: sharply bimodal, so a gap threshold is data-derived). Output: fraction of rows with a successor, union-find components, chain-length distribution, whether row id order equals chain order (hypothesis: mostly yes, ids contiguous within chains), and whether any interior candidate also appears in other rows (hypothesis: no). Hypothesis on groups: a small number (tens) of long chains, perhaps one per capture session or site, making naive GroupKFold imbalanced and requiring block splits (see Validation design).

D4. Scene/site structure. Cluster chains by mean embedding (agglomerative, train only); count distinct scenes and rows per scene. Hypothesis: a handful of scenes (fixed field cameras) recur across many chains. This decides whether random-row, chain-block and scene-held-out CV give very different scores (a large gap signals scene-memorisation risk).

D5. Temporal geometry of the label (tie diagnostic, no model training). For every row compute the 8x8 embedding-distance matrix over (L, R, c0..c5), seriate the 6 candidates (spectral ordering, direction fixed by L->R), and record the seriation rank (1..6) of the gold. Hypotheses: gold lands at rank 3 or 4 in a clear majority of rows (est. 55-70%, noisy because the embeddings are crop/photometric-corrupted); the split between rank 3 and rank 4 tells us the tie structure: roughly 50/50 means a symmetric (timestamp or random) tie-break, a strong skew (e.g. 75/25) means a conventional lower/upper median. Also compute normalised positions d(L,g)/(d(L,g)+d(g,R)): mean near 0.5 with a wide spread is expected.

D6. No-ML reference scores on train: slot prior; argmin |cos(c,L) - cos(c,R)|; argmin of max(d(c,L), d(c,R)); colour-histogram balance. Expected (est.): 0.25-0.38. This is the bar for "ML is load-bearing" and also a pipeline bug detector.

D7. Corruption magnitude. Using matched duplicate photos from D3 (R_i vs L_{i+1} are the same underlying photo with independent crop/flip/photometric changes), measure the crop scale/offset (via patch matching), flip rate (expected ~50%), and brightness/contrast/colour shifts. Use these empirical ranges to set the training augmentation so that train-time views look like release-time views (train-derived statistics only).

D8. Spacing regularity. In reconstructed chains, test whether R_i is "the midpoint of L_i and R_{i+1}" in embedding terms (distance from R_i to L_i vs to R_{i+1}). Hypothesis: roughly equal if frames are equally spaced in time. This gates the chain-recombination augmentation (Structural signals, point 4).

D9. Information ceiling. From D5 compute the oracle "gold is within the seriation central pair" rate, giving a pair-recall ceiling for any embedding-based method, and the implied score with coin-flip order (est. 0.6 x pair recall + residual).

Hypothesis summary (unverified): per-row evidence is weak and noisy; signal sits in scene-level change (illumination direction, plant/vegetation growth, moving objects) that survives the independent photometric jitter only partially; the CV score will sit well below the 0.6 coin-flip ceiling.

## Validation design

**Reproducing the test split (inference from the description only).** Unknown whether the test is a random row split (neighbouring train/test rows share boundary photos), a block split (contiguous chains), or scene-held-out. The statement "Bursts continue across rows (one row's R photo can be the next row's L photo)" plus "no group column" suggests a row-level (random or sequential) split in which test rows may chain to train rows. Plan under both readings and pick the conservative one for model selection:
- **Primary CV (conservative): chain-block GroupKFold.** Build groups = connected components from D3. Because components may be very long, cut each chain into contiguous blocks of about 40-60 rows and drop a 1-row buffer at each cut (a cut shares one photo). Assign blocks to 5 folds balanced by row count. Hold out whole blocks.
- **Stress CV: scene-held-out.** Group by D4 scene clusters; report only for the final design and one or two key ablations (it measures the downside if test scenes are unseen). If primary and stress differ by more than 0.03, scene memorisation is the main risk and regularisation (shallower unfreezing, stronger augmentation) takes priority.
- Optional looser CV (random row KFold) only to show the optimism of leaky splits; never used for decisions.
- Repeats: 2 split seeds for cheap models (frozen-feature heads), 1 seed for fine-tuned runs (cost). Report mean and std across folds and seeds; use paired comparisons on identical folds.

**Noise level (est.).** Row score has mean ~0.45 and per-row std ~0.47 (e.g. P(1.0)=0.40, P(0.2)=0.20). Over ~1,250 OOF rows SE is ~0.013; paired comparisons on the same rows reduce this slightly. Test of "a few hundred rows" (say 300) has SE ~0.027, so differences under ~0.025 are unresolvable. Accept a change only if the paired gain exceeds ~1 SE (~0.015-0.02) and is directionally consistent across most folds. The public LB (a slice of the test) is weaker still; do not tune on it.

**Metric re-implementation and unit tests.** Exact scoring (1.0/0.2/0) re-implemented from the description; tests: perfect ranking = 1.0; gold forced to rank 2 = 0.2; gold excluded = 0.0; constant `[c0,c1]` and uniform-random ranking on shuffled-slot data = ~0.2; the slot-prior baseline (most frequent gold slot) = ~0.2 (a higher value implies the D2 artefact); reversed (worst) ranking = ~0 for most rows. Also test the submission validator against the sample.

**Post-hoc selections.** The only post-hoc choices are the ensemble blend (simple mean of per-row probabilities, no fitted weights, or at most one scalar) and optional boolean flags decided by an in-script cheap probe (see Fixed work plan). Any flag chosen on OOF gets cross-fitting: decide on 4 folds, confirm on the 5th.

**Overestimation warning.** A self-built proxy usually over-estimates the real score (group mis-specification and scene overlap). Plan for the real score to land 0.02-0.05 below primary CV.

## Overfit/underfit risks

**Overfitting (with mitigations).**
1. Few independent groups (chains/scenes) behind ~1,250 rows. Mitigation: block CV with buffers, stress CV, conservative selection.
2. Scene memorisation by a 86M-parameter backbone on ~8,750 photos: the model could learn "which scene" rather than "temporal position". Mitigation: unfreeze only the last 4 of 12 blocks, low LR (1e-5 backbone, 1e-3 head), layer-wise decay, weight decay 0.05, EMA of weights, fixed 10-12 epochs, per-image independent heavy augmentation (crop/flip/photometric with ranges from D7), candidate-slot permutation augmentation, dropout in the head.
3. Selection on OOF with many knobs. Mitigation: keep design knobs few and a priori (conventional LR, epochs); no extensive HPO; ablations judged by paired noise rule; the swap flag, resolution and recombination aux are the only pre-declared binary decisions.
4. Slot or order leakage (gold slot skew, id order, file order): removed by permutation augmentation and by not using any such feature.
5. CV optimism from near-duplicate photos across folds: mitigated by chain grouping (D3), buffer rows and the scene stress test.

**Underfitting.**
1. Too coarse a representation: 224 px DINOv2 may lose small cues (moving objects, small growth); Mitigation: test 224 vs 336-392 px on the cheap frozen probe first, choose by paired noise rule.
2. Pooling to one global vector per photo throws away spatial/correspondence information under independent crops. Mitigation: add patch-token correspondence statistics (soft nearest-neighbour match between a candidate's patches and L's / R's patches), which are crop- and flip-robust, as features to the head.
3. Loss mismatch: unsupervised relative position is only supervised by the gold. Mitigation: listwise CE over the 6 candidates (the same objective as the metric's rank-1 term), plus the chain-derived span-14 task if it passes the reviewer gate.
4. Gold-only supervision gives 1 positive per 6; with ~1,250 rows that is thin. Mitigation: set-level context head (the model sees all candidates), slot permutation (720x re-orderings) and L/R swap if symmetric.
5. Information ceiling: independent crops can remove the changing region from a given photo; cannot be fixed; accept.

## Recommended approach (primary + fallback)

**Primary: partial-fine-tuned DINOv2-B/14 + set-level listwise ranker with relative features.**
- Representation: `facebook/dinov2-base` via HF transformers (or `vit_base_patch14_dinov2` via timm), weights downloaded from the hub, revision pinned (record the commit hash at implementation time; I could not verify it). Input 224 px (test 336 later). Blocks 0-7 frozen (run under no_grad), blocks 8-11 plus final norm trainable. Each photo -> CLS + mean patch token (global descriptor) and the patch-token grid (for correspondence stats).
- Per-candidate relative features (computed per row from its own 8 images): cosine similarity of candidate to L and to R, their difference and sum, the projection coordinate t = <e_c - e_L, e_R - e_L> / ||e_R - e_L||^2 and its distance from 0.5, margin to the best other candidate, patch-correspondence stats (mean of max patch similarity of c to L, to R, and L-to-R reference), all within-row relative transforms (principle I).
- Head: small set transformer (2 layers, d=256-512, 4 heads, no slot positional encodings) over tokens [L, R, c0..c5] with role embeddings, per-token concat of backbone descriptor and relative features, output one logit per candidate. The head is permutation-equivariant over candidates.
- Loss: listwise cross-entropy over the 6 logits against the gold. Optional small auxiliary: symmetric margin so that gold is nearer to t = 0.5 than non-gold (only if the ablation helps).
- Augmentation: independent per-image random-resized crop, random flip, brightness/contrast/saturation/hue jitter with D7 ranges; random permutation of candidate slots; L/R swap only if the swap test passes (see Structural signals).
- Training: AdamW, cosine with 1 warm-up epoch, bf16 autocast, gradient clip 1.0, EMA, fixed 12 epochs, batch 8 rows (64 images), 5 grouped folds, fold models averaged at test (per-row mean of softmax probabilities, 2 flip-TTA views). No validation-triggered stopping.
- Compliance: clean (genuine fine-tuning; pretrained general backbone; per-row inference only).
- Why it fits this data: scene-level temporal drift is subtle and photo-level corruption is independent per image, so a strong self-supervised dense representation with crop-robust correspondence features and a set context is the leanest design that sees both global drift and local change.

**Fallback (lower risk / cheaper): DINOv2-S/14 (or ConvNeXt-Tiny from timm) with last 2 blocks/stages fine-tuned, global descriptor + relative features + 1-layer set head, listwise CE, 3 seeds x 5 folds.** Same validation, no correspondence features. Used if the primary fails the paired-noise rule against it, is unstable across folds, or the runtime estimate does not hold on the real A10G.

**Optional diversity member (only if it earns it): a different-assumption family** (CLIP ViT-B/16 or EVA02 from timm, language/contrastive pretraining) with the same head. Blend by mean probability only when its OOF alone is within ~0.02 of the primary and the blend gain exceeds 1 SE across seeds; otherwise spend the time on a second seed of the primary.

**Expected PRIVATE range (estimate, not a promise).** Heuristic no-ML ~0.28-0.38; primary 0.38-0.52 with central guess ~0.44; ceiling 0.60 if the tie is a coin flip. Reasoning: the signal is weak under random crops/photometric corruption and the midpoint is tied between two frames; the final number is expected 0.02-0.05 below CV.

## Rejected options

- **Frozen embeddings + GBDT / logistic head as the shipped model:** grey/likely rejected (no real training of the backbone); kept only as the dev probe.
- **Optical flow / registration / pixel-difference midpoint solver:** rule-based; independent crops, flips and photometric changes break alignment; fails strip-the-ML.
- **Chaining test rows or matching test photos to train photos for extra context:** banned (test cross-reference, whole-test aggregation).
- **EXIF/file-name/zip-order/id-order features:** leak-style metadata, not ML; audited only.
- **Pairwise-only Siamese scoring (each candidate scored independently with L, R):** loses the set context and the 2-before / 3-after count constraint; kept as an ablation.
- **Video transformers or large VLM zero-shot:** inference-only or no pretrained temporal model suited to 8 irregular-scene frames; slow; compliance grey.
- **Training from scratch:** ~1,250 rows is too few.
- **Synthetic interpolation frames (diffusion/flow):** synthetic data, banned.
- **Pseudo-labelling or test-time adaptation:** banned.
- **Heavy HPO:** CV SE (~0.013) cannot resolve fine optima; selects noise.
- **Label smoothing:** avoided on small data and unnecessary for a rank-only metric.
- **Per-row softmax temperature fitting on test:** banned (test statistics).

## Fixed work plan & runtime budget

All counts fixed, no clock in any condition, `device = "cuda"`, fixed seeds, fixed `num_workers`, fixed folds.

| Stage | Plan | A10G estimate (est.) |
|---|---|---|
| Unzip, JPEG decode, resize to a fixed 256 px short side, keep uint8 in RAM (~12.4k photos, ~2.4 GB) | fixed 4 workers, once | 1-2 min |
| Backbone download (HF hub, pinned revision) | once | <1 min |
| Chain/scene grouping with frozen embeddings (train only) | one pass over ~8.75k photos x2 flips | 1-2 min |
| Cheap in-script probe (frozen features + tiny head) to resolve the swap and resolution boolean flags | seconds per variant | 1-2 min |
| Primary fine-tune, per fold: 12 epochs x ~1,000 train rows x 8 images, last 4 blocks trainable, bf16, deterministic attention | ~20-30 s/epoch incl. aug (x1.3 for deterministic kernels) | 5-6 min per fold, 5 folds = 25-30 min |
| OOF + test inference (2 flip views, ~300 test rows x 8 images x 5 folds) | | 1-2 min |
| Validation, metric, write, re-read | | <1 min |
| **Total primary** | | **about 32-38 min** |

Headroom: stated budget ~1 h, so 38 min leaves ~37% headroom; do not add a second family unless the probe shows it adds >1 SE and the measured primary per-fold time leaves total <= 45 min. Memory: peak ~8-10 GB GPU at batch 64 images, bf16, frozen blocks under no_grad; ~2.4 GB RAM for cached images. Reduce the fold-time estimate by profiling one epoch before fixing the counts (CLAUDE.md section 3 rule 7); the plan above is an estimate until profiled.

Determinism notes: set `CUBLAS_WORKSPACE_CONFIG` before importing torch; `torch.use_deterministic_algorithms(True, warn_only=True)`; seeded `Generator` for data order and for GPU augmentation; prefer eager/math attention kernels if SDPA backward is non-deterministic; decode and ensembling are robust to residual nondeterminism (probability mean, stable argsort). No early stopping on validation (fixed epochs + EMA). Fold models are shipped as the ensemble (no extra full-data retrain): variance reduction from 5 models outweighs the ~20% data difference at this size; no in-script wall-clock fallbacks.

Input and output validation inside the script: assert required columns and image files exist; each image decodes to finite tensors; candidate names parsed from the sample/test file; after prediction, assert 2 distinct valid candidates per row, JSON round-trips, row order equals sample order, re-read the written CSV with `keep_default_na=False`. A constant fallback is not needed; if any validation fails the script raises loudly rather than writing a degraded file.

## Metric-aware training & decode

(i) Back-solve: the metric is 1.0 for the gold at rank 1, 0.2 at rank 2. The rank-1 term dominates (5x); a listwise cross-entropy over candidates targets exactly the rank-1 probability and, being a proper scoring rule, also orders the rest sensibly for the rank-2 term. No richer target can be reconstructed from the metric.

(ii) Weighting: all rows have equal weight in the metric; use equal row weights. Hierarchical averaging: none needed.

(iii) Ordinal cumulative heads: not applicable to gold-only labels. However, a time-coordinate scalar t in [0,1] is built into the relative features (distance of t from 0.5) as an inductive bias.

(iv) Expected-utility decode: expected score = p1 + 0.2 p2 for any ordered pair, so sorting by posterior p is optimal; no constants to fit and nothing to cross-fit. Ties are broken by a stable argsort by candidate index (deterministic). Posterior = per-row mean of fold-model softmax probabilities (and flip-TTA views).

(v) F-score thresholds: not applicable.

(vi) Ordering/ranking metrics: listwise rather than pointwise losses used; antisymmetric pairwise accumulation is not required for a top-2 rule.

(vii) Hard constraints at decode (optional, only after the base model is solid): the 6 candidates occupy ranks 1..6 in true time order; a gold at position 3 has exactly 2 candidates before it and 3 after (4 and 5 on the other tie-break). Enumerate the 720 permutations per row, score each with pairwise ordering potentials from the model's relative coordinate t (combined with model evidence, not a pure constraint search), and take the marginal posterior of each candidate sitting at rank 3 or 4. Enable only if OOF improves beyond 1 SE. It is a per-row operation (compliant).

(viii) Overdispersed counts, (ix) taxonomies: not applicable.

TTA: flip views and (only if symmetric) L/R swap view, per-row, averaged in log-prob space or prob space (choose a priori: probability mean).

## Structural signals

1. **Slot permutation invariance.** Candidates are shuffled by construction: the head is permutation-equivariant and each epoch applies a random permutation, removing slot artefacts and multiplying effective data.
2. **Independent per-image corruption.** The release applies crop, flip and photometric changes independently per photo; mimic this (D7 ranges) independently for each of the 8 images so the model learns invariance. Do not apply a shared geometric transform to the set; no geometry is tied across photos.
3. **Time-reversal symmetry (conditional).** Swapping L and R reverses time. If the gold is "closest to the midpoint by true time" (symmetric), the label is invariant to swapping L and R, giving a free 2x augmentation and a swap-TTA view. If the gold is a convention (e.g. lower median frame), swapping flips the central pair and the label would no longer be invariant, so swap must NOT be used. Decide with: (a) D5 rank-3 vs rank-4 split; (b) a CV comparison of swap-augmented vs not (paired); (c) the asymmetry probe: with the non-swap model, check how often top-1 changes when L and R are swapped at inference (symmetric label rule predicts agreement; convention predicts systematic disagreement). Default if undecided: no swap (compliant under every reading, with the measured cost reported).
4. **Chain structure in train (R_i = L_{i+1}).** Reconstruct chains from train photos (train only). If D8 supports equal spacing, build extra real labelled tasks: for rows i and i+1, the task (L_i, R_{i+1}) has R_i as the exact midpoint (frame 7 of 0..14), unambiguous with no tie; the candidates are R_i plus 5 real distractors drawn from the two rows' interiors. This is recombination of real labelled units (defensible), not fabricated blends; use at a small sampling weight and only if it beats noise and the reviewer agrees. These tasks need no information from test rows.
5. **Count constraint.** The 6 interior frames are ordered along a path from L to R; exactly 2 (or 3) lie before the gold. A set-level head and the optional permutation-marginal decode use it; an independent per-candidate scorer cannot.
6. **Anchors are fixed.** L is first and R last, so direction of time is known; no need to infer arrow of time from content (but the model must learn which direction a content change runs, via the role embeddings).
7. **Duplicate-photo pairs** (R_i, L_{i+1}) give a natural "same photo, different corruption" contrastive signal: can be used as an auxiliary consistency loss (descriptors of the same photo under different corruptions should match), strengthening photometric/crop invariance. Train only, optional.

## Experiment roadmap

Each step has a stop criterion; do not start step n+1 until n passes.
1. **Contract, metric and validator.** Parse train/test/sample, implement metric with the unit tests above, build the submission validator, write a trivial valid file (log it as a fallback only for pipeline testing). Stop when all tests pass and the format matches `sample_submission.csv`.
2. **Data diagnostics D1-D9 and fixed folds.** Produce chain groups, blocks with buffers, scene clusters, and save fold assignments (dev) / reproducible in-script. Stop when fold sizes are balanced (each fold ~20% rows, none dominated by one scene) and no photo appears across fold boundaries beyond buffers.
3. **No-ML heuristics + frozen-feature listwise head (dev scaffold).** Report OOF mean/std over 2 split seeds. Stop criterion: frozen head exceeds the best heuristic by >= 0.05; if not, check for bugs (candidate alignment, slot ordering, crop/flip handling) before anything else, and reconsider whether usable signal exists at this resolution.
4. **Primary fine-tune (single fold smoke test, then 5 folds).** Compare paired against the frozen head. Accept the primary if the paired gain > ~0.02 and consistent across folds. Record per-epoch curves to confirm the parameters moved and the loss decreases; verify the distribution of OOF top-1 probabilities.
5. **Structural ablations (cheap probe first, then confirmatory fine-tune on the winning one):** swap symmetry (decision rule above); resolution 224 vs 336-392; correspondence features on/off; set head vs pairwise head; unfreeze depth 2 vs 4 vs 6. One change at a time; accept only changes beating noise.
6. **Metric-aware decode:** permutation-marginal decode vs sorting. Accept if > 1 SE on OOF, cross-fitted.
7. **Chain-recombination span-14 auxiliary** (if the reviewer allows and D8 supports it): add at low weight; accept if it beats noise on the primary CV and on the stress CV.
8. **Diversity:** second family (CLIP/EVA02 or ConvNeXt) alone first, then blend only if within 0.02 and the blend gain > 1 SE across seeds; otherwise a second seed of the primary.
9. **Bounded HPO (optional):** none recommended; if added, a fixed small number of trials of LR and epochs inside the script with its own held-out check, and only if it clearly beats noise.
10. **Final fixed-plan run twice and diff:** run the exact command `python3 solution.py <public_dir> <submission_out>` from a clean `working/`; confirm top-1 agreement near 100% between two runs and identical output if the determinism holds; check the slot distribution of top-1 predictions is roughly uniform over c0..c5 and the confidence distribution resembles the OOF; record runtime and confirm >= 30% headroom.

Credits (6 per problem): baseline (frozen/ small model), best single, ensemble, final. Do not spend credits on tweaks below the noise floor (~0.025).

When two or three independent solvers would plausibly converge on a technique (here: a pretrained DINO-style backbone with listwise CE and slot permutation), implement and test it before any proxy-driven tweak.

## Compliance audit

CLAUDE.md section 7 and agent self-audits, answered against this plan:
- Test file read for anything other than one-row inference? No. The plan makes each prediction a function of that row's 8 images and train-fit weights. Chain/scene grouping, D7/D8 diagnostics, and normalisers are train-only. Test cross-row chaining and photo matching are explicitly excluded.
- Whole-test aggregation (rank/z-score across the test file, label-free prior from test lists, vocab/encoder on train+test)? None. Softmax is per-row; the ensemble is a per-row mean.
- Related-row ("sibling") leakage? In train, chain-mates share photos: handled by chain-block grouping and buffers; at test time sibling rows are not used. Within a row, candidates are siblings by design (the target is a relation among them); the features are within-row relative transforms of the row's own inputs, which is the legitimate task input, not another row's information.
- Time in conditionals, env-dependent fallbacks, cpu_count, import fallbacks? None planned; time for logs only; `device="cuda"`; fixed workers/seeds/folds/epochs; pinned revision.
- Hard-coded constants tuned offline? The only constants are conventional fixed hyperparameters (LR, epochs, augmentation ranges derived in-script from D7 on train); structural boolean decisions (swap, resolution, recombination) are decided by dev ablations and recorded; where cheap, decided in-script by a fixed-count probe. Flagged to the reviewer for the dev-ablation ones.
- External or synthetic data, self-hosted weights, non-allowed libraries? No: HF/timm general-purpose pretrained weights only; kornia/torchvision/transformers/timm assumed present (verify `timm`/`kornia` against the allowed list at implementation); chain recombination uses real train frames only (reviewer question).
- Strip-the-ML: without the trained backbone/head the best heuristic is ~0.3 (est.) vs expected model 0.4-0.5 (est.); model-heavy part dominates.
- Metadata (EXIF, names, order) used? No.
- Source readable, < 512 KB, comments explaining reasoning, no blobs? Planned yes.
- Compliance status of the primary: clean; fallback clean; second family clean; chain-recombination aux: grey-clean pending reviewer; frozen-feature probe: dev/diagnostic only.

## Open questions & assumptions

Questions for a reviewer (with the plan under each reading):
1. **Gold definition with 8 frames.** Is the gold derived from true timestamps (irregular spacing breaks the 3/4 tie) or a fixed convention (e.g. frame 3 or 4)? Under symmetric/timestamp reading: swap augmentation allowed, ceiling ~0.6 for index-tied frames. Under convention reading: no swap, direction-aware head, higher achievable score. Compliant default: no swap; cost measured by the paired swap/no-swap comparison.
2. **Chain-derived recombination tasks (span-14, midpoint = R_i)** built only from train labels and train photos: permissible as recombination of real labelled units, or "synthetic data"? Default: off unless approved.
3. **Metadata:** if EXIF timestamps or time-ordered file naming exist, do they count as forbidden leakage? Default: never used.
4. **Test rows chaining:** confirm that using the neighbouring test row's photo (R_i = L_{i+1}) is not allowed. Default: not used.
5. **Pretrained backbone:** is DINOv2 (self-supervised LVD-142M) acceptable as a "general-purpose pretrained weight"? Default: yes; fallback to a timm ImageNet backbone if not.
6. **Partial fine-tuning** (last 4 blocks) counts as genuine fine-tuning for "must train a model in-script"? Default: yes; unfreeze depth is a dial.
7. **Submission encoding** of `ranked_candidates_json` (names vs indices; spacing in the JSON): copy from `sample_submission.csv`; confirm quoting survives the CSV round trip.
8. **Test split mechanism and test-time corruption:** assumed identical to train (independent per-image crop/flip/photometric); if the test is scene-held-out, rely on the stress CV for model selection.

Assumptions (all UNVERIFIED): ~1,250 train / a few hundred test rows; ~8,750 unique train photos; A10G with 24 GB; HF hub reachable for the pinned backbone; images roughly 300-1000 px on the long side; frames equally spaced in time (D8 will test this); the gold is always among c0..c5; slot order is uniformly shuffled (D2).

What I could not verify: every data-dependent claim above (chain structure, gold slot uniformity, tie structure, image sizes, EXIF content, corruption ranges, heuristic and model scores); the DINOv2 revision hash; library availability on the platform image (`timm`, `kornia`); the actual A10G throughput of partial fine-tuning (runtime table is an estimate until one epoch is profiled).
