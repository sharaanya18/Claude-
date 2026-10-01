# U14 Nocturne Protocol Weave - Eris build plan

Status: planning only. No dataset files were available when this plan was written, so every statement under "Data findings" is a hypothesis to verify, marked UNVERIFIED. Facts taken from the challenge description are marked (desc). Estimates are labelled as estimates. Sources read: eris-strategist.md, CLAUDE.md, tasks/user/U14_nocturne_protocol_weave.md. Nothing else was opened and no internet was used.

---

## Contract & decision unit

**One valid answer.** `submission.csv` with exactly the columns `id,prediction`, every test id exactly once, in `sample_submission.csv` order. `prediction` is a finite, non-Boolean integer in {0,1,2,3}. Anything else is rejected (desc): extra columns, missing/duplicate/unknown ids, fractional values, NaN/inf, Boolean. Write `astype(np.int64)` and re-read the file with `keep_default_na=False` to check it.

**Invalid vs low-scoring.** Invalid means rejected, below zero. Low-scoring means a constant or chance prediction, which scores 0 after correction (desc).

**Metric, term by term (desc), to be re-implemented exactly:**
- BA_p is the mean recall over the 4 slots inside protocol p, for p = 0..6.
- c_p = clip((BA_p - 0.25)/0.75, 0, 1).
- g = (exp(mean_p log(0.05 + 0.95 c_p)) - 0.05)/0.95. This is a stabilised geometric mean over the 7 protocols, so the weakest protocol dominates.
- score = g^2.
- If all c_p are equal, score = c^2. BA 0.70 gives 0.36, BA 0.80 gives 0.54, BA 0.85 gives 0.64, BA 0.90 gives 0.75. Errors in score are roughly 2.7x the BA error at c near 0.7 (estimate, from d(score)/dBA = 2g/0.75).
- The metric is a pure hard-label metric. It is not calibration-sensitive, but the decision rule must be recall-balanced per slot inside each protocol.
- There is no gating beyond the clip at chance.

**True independent unit.** The camera site, not the event. Training has 298 cameras, about 19.6 events each (5834/298), and test has 131 unseen cameras (desc). Events from one camera share a background and probably come in bursts of near-duplicate frames. The effective sample size is therefore on the order of 300 camera clusters, not 5,834 rows. There is a second, latent unit: the 8 hidden identities.

**Latent structure (desc).**
- There are 8 hidden identities z in {0..7} and 7 protocols.
- Each protocol is a perfect matching of the 8 identities into 4 pairs, so the 7 matchings form a 1-factorisation of K8. All 28 pairs occur exactly once across protocols.
- Slot order is permuted per protocol. The slot label has no cross-protocol meaning.
- The target is y = m_p(z), where m_p maps an identity to its slot. This is 2-to-1, with exactly 2 identities per slot.
- The model must infer q(z | event) from indirect supervision, then apply the row's protocol map.

**Pipeline stages, diagnosed separately:**
1. Candidate coverage. This is trivial: the answer is always one of 4 slots, with a latent 8-state intermediate.
2. Scoring. This is the per-event posterior over the 8 identities, built from the 3 frames.
3. Decoding. Slot posterior pi_p(s | x) = sum_z q(z | x) M_p[z, s], then argmax with uniform slot prior.

Diagnose three things apart: identity-level separability of the frozen features, recovery of the pairing structure M_p, and slot accuracy given good q.

**Useful identity (derived from desc, to verify).**
- If an identity classifier has accuracy a and errors spread over the 7 other identities, slot accuracy is about a + (1-a)/7. In each protocol exactly one other identity shares your slot.
- Real confusions are not uniform. A protocol that pairs two visually similar identities in one slot gets an easy boost, and a protocol that separates them is harder. Per-protocol difficulty will therefore differ, and the geometric mean punishes the worst protocol.

---

## Compliance regime

**Domain.** Computer vision, camera-trap style, 3 grayscale frames per event, 8 latent classes with a structured label map. The description does not label the task "Fine-tuning" or "From-scratch". Assumption A1: it is a standard CV challenge, so CLAUDE.md section 6.3 applies. Do not feed raw pixels to a tabular model. Hand-crafted features go only alongside the image into a trained model.

**Explicit bans from the description. Treated as hard constraints:**
1. **External training data:** none.
2. **Reverse lookup of released samples, and "external matching, reverse search":**
   - No test-to-train image matching or retrieval.
   - No kNN-over-train as the decision rule. Parametric heads only.
   - No fingerprint, hash or near-duplicate tricks between test and train images.
3. **Task-specific pretrained models:** not permitted.
   - Avoid wildlife or camera-trap detectors and classifiers (MegaDetector, SpeciesNet, BioCLIP).
   - Avoid iNaturalist-trained backbones (for example timm `*.inat21` tags) and any weights fine-tuned on animals or species.
   - Use only generic vision weights: DINOv2 (LVD-142M), ImageNet-trained timm/HF models, generic CLIP/EVA image towers. No text prompts or zero-shot species names.
   - Source is HF Hub or timm only. Pin the model revision.
4. **Inferring targets from row order or serialization details:** ids are arbitrary.
   - Join `train.csv`, `train_folds.csv` and the npz by id, not position. Assert `ids` array equality.
   - Never use the id as a feature.
   - Shuffle with a seeded generator.
5. **Leaderboard probing:** none. Use CV as the source of truth.
6. **Predictions only from released event panels, protocols and training supervision.** Each test row's prediction is a function of its own 3 frames, its own protocol and the train-fit model only.

**CLAUDE.md bans that bind here, in addition to the above:**
- No test-set use beyond per-sample inference. In particular: no camera clustering of test events, no test-level normalisation, no balancing of predicted slot frequencies over the test set, no pseudo-labels, no fitting of any scaler, PCA or vocabulary on test or on train plus test.
- No wall-clock branching, no environment fallbacks, fixed seeds, fixed work plan.
- Source under 512 KB, plain readable.
- No external or synthetic training data. Planted-structure sanity data is allowed only as a dev-time unit test of the code and is never inside `solution.py`.

**Where the description is silent, and the assumptions made:**
- **Runtime.** It says only "one NVIDIA A10G; long pretraining not required". Assumption A2: use the CLAUDE.md default of at most about 50 min target, 1 h worst case.
- **Model-size cap.** None stated. Assumption A3: none applies. Staying at or below ViT-L/14 anyway.
- **TTA and ensembling.** Not mentioned. Assumption A4: allowed (CLAUDE.md allows per-sample TTA, and multi-model ensembling inside the script is allowed).
- **Use of `train_folds.csv`.** Assumption A5: it is shipped in `public_dir` at grading time, since it is listed as released data. The script asserts its presence loudly. There is no fallback split, because that would be an environment-dependent branch.
- **Which pretrained families count as "generic".** Assumption A6: DINOv2 / ImageNet / generic CLIP are generic. This is a reviewer question.
- **Frame differencing and per-image contrast normalisation across the event's own 3 frames.** Assumption A7: allowed. These are within-row operations feeding a trained model.
- **Test events from the same camera.** The description gives no site id and says nothing about grouping. Assumption A8: do not group test events. This is cross-row pooling of test and is banned by CLAUDE.md 2.3 #5.

**Compliance of the core idea.** The description itself lists the following as alternatives: "seven coupled heads with a shared bottleneck, a latent eight-state model whose protocol maps are learned jointly, or a consistency loss that exploits the complete pair-cover structure". The latent eight-state model is therefore endorsed in the text. The pair-cover regulariser is derived from the description's structure, not from tuned constants.

**Genuine-training layer (strip-the-ML test).** Removing the trained components leaves nothing. The 8-state head, the protocol maps and the optional LP-FT backbone blocks are all trained inside the script. See the audit section for how to make the training load-bearing and logged.

---

## Data findings

**None of these were run, because no dataset files were available. All are UNVERIFIED.** Run them on train only. Test is used only for schema, row count (2,434) and id format.

**Diagnostics to run, in order, with the hypothesis for each:**

1. **Schema and alignment.**
   - Shapes: images (5834,3,128,128) uint8 and (2434,3,128,128) uint8.
   - Check `ids` aligns to CSV order. Check pixel range, mean and std per frame index.
   - Check the CSV dtypes of `protocol` and `target`, and whether any ids are strings with a leading zero.
   - H: matches the description. `sample_submission.csv` row order equals the test CSV order (verify; the script should write in sample order, joined by id).

2. **Protocol and target balance.**
   - Table protocol x target, with exact counts. Also protocol x fold and target x fold.
   - H1: each protocol has about 833 rows (5834/7 = 833.4), slots balanced within about 2 percent.
   - **Derived implication.** If every protocol has exactly balanced slots, then for any pair a, b, P(a) + P(b) = 1/4. With K = 8 that forces every identity to 1/8. Check the deviation from exact balance. This justifies a uniform identity prior, a batch-level KL regulariser toward uniform, and plain argmax as the BA-optimal decision rule.

3. **Fold structure.** Rows per fold (H: about 1,167), and whether every fold contains all 7 protocols and all 4 slots. Check that the folds are really camera-disjoint using the frozen-embedding diagnostic in item 5.

4. **Image statistics.**
   - Fraction of frames with very low contrast or an almost-empty animal region, using the within-event frame difference |f_t - f_s| energy and the per-frame standard deviation.
   - Fraction of events where at least one frame is "informative" (desc says single frames can be empty, blurred or obstructed).
   - Fraction of events with near-identical frames.
   - Saturation fraction (infrared glare).
   - H: 15 to 40 percent of single frames are weak, and 3-frame pooling is essential. Frame differences isolate the moving animal because the background is static.

5. **Camera-appearance dominance (leak and shift tell).**
   - Extract frozen DINOv2 CLS embeddings. For a sample of train events, compute nearest neighbours within the same fold versus across folds.
   - H: within-fold neighbours are much more similar than across-fold ones (camera background memorisation). This confirms that grouped folds are mandatory and that the head needs heavy regularisation, augmentation or a background-suppressing cue.
   - Optionally cluster train events inside each fold by background similarity (union-find) to estimate the number of cameras per fold and the burst size. This is train-only and is used for diagnostics, not for splitting.

6. **Floor probes under the supplied folds** (the honest yardstick):
   - **(a)** Seven independent 4-way logistic probes, one per protocol, on frozen pooled features. Expect near the chance-corrected floor (desc says "seven unrelated constant rules ... near the floor").
   - **(b)** One shared 8-way probe via the latent model (the primary).
   - **(c)** A shared bottleneck with 7 x 4 coupled heads.
   - Report pooled-OOF score and per-protocol BA. Anything the primary does must beat (a) and (c) by more than noise.

7. **Pair structure recovery check.**
   - After fitting the latent model, harden M_p with Hungarian assignment (8 identities into 4 slots with capacity 2).
   - Verify: each slot has exactly 2 identities per protocol, and G = sum_p M_p M_p^T equals 6I + J (diagonal 7, off-diagonal 1).
   - Report the fraction of restarts that reach a valid factorisation, and the train NLL gap between valid and invalid restarts.
   - H: with 5.8k rows and a decent backbone, valid factorisations are recovered in most restarts.

8. **Slot-centroid identifiability probe.**
   - Compute the 28 slot-mean embeddings mu_{p,s}. In the ideal case each equals the mean of two identity means, so all 28 pairwise averages exist exactly once.
   - Fit an 8-centroid decomposition (alternating least squares) and report the residual. If it is small, the pairing structure is visually recoverable.
   - H: a clean decomposition is possible if identities are visually distinct.

9. **Ambiguity and information ceiling.**
   - Count near-duplicate train events (cosine similarity above a high threshold) that carry different slots under the same protocol. H: there are none except label noise from bursts straddling a camera boundary.
   - Under different protocols, near-duplicates (same event seen twice) would give direct cross-protocol identity linking. This is a legitimate training-time constraint, diagnostic only.
   - Ceiling: oracle on true identity gives 1.0. Realistic ceiling is limited by empty or occluded events. Estimate from the fraction of events whose best-frame evidence is weak.

**Hypotheses to carry into modelling (UNVERIFIED):**
- H-a: camera background dominates raw embeddings, so the head must resist background features.
- H-b: IR night images are far from ImageNet statistics, so a last-stage fine-tune may help over the frozen probe. Gate this on measurement.
- H-c: identities are species-like and a few pairs are visually close. These dominate the worst protocol.
- H-d: 128 px frames upsampled to 224 lose little, because the source resolution is 128. Resolution above 224 buys little (test at 336 if cheap).

---

## Validation design

**Scheme.** Use the supplied `train_folds.csv`, 5 camera-disjoint folds, unchanged. Do not re-split rows (a random split leaks camera appearance, per the description). Test is 131 unseen cameras, so camera-disjoint is the same axis of shift as deployment.

**Metric on OOF.** Compute the official score on the pooled OOF predictions (all 5,834 events scored once, like the real test), plus per-fold scores and per-protocol BA. Report mean and std of fold scores, but treat the pooled-OOF number as the headline because the official metric is on the full set. Fold-level scores are noisier: about 167 events per protocol and about 42 per slot per fold, so per-protocol BA standard error is about 3 percent per fold.

**Noise level (estimate).**
- Pooled OOF: about 833 events per protocol (about 208 per slot), BA standard error about 1.4 percent per protocol. Score standard error is about 0.02 to 0.03.
- So accept a change only if the pooled-OOF score improves by more than about 0.02 AND it is consistent on at least 4 of 5 folds in a paired comparison on the same folds.
- Add model-seed repeats (3 seeds of head initialisation and view sampling) to estimate training noise. Camera-level split repeats are not possible without camera ids. As an optional extension, derive pseudo-camera clusters from train backgrounds and do a second grouped split. This is not primary.

**Sanity fold.** Hold fold 4 out of every model-selection decision (backbone choice, L2 grid, pooling choice). Select on folds 0 to 3 (train on 3, validate on the fourth in an inner scheme, or simply on the 4-fold OOF of folds 0 to 3), then report fold 4 once at the end as the check against overfitting CV. The shipped script does the in-script L2 selection by CV on all 5 folds, and the dev-time sanity fold is a separate protocol.

**Metric unit tests (before any modelling).** Re-implement the metric and test:
- Perfect predictions give 1.0.
- Constant per-protocol prediction gives 0.0.
- Reversed or permuted slots give 0.0 (BA at or below 0.25 clipped).
- Uniformly random predictions give about 0.0.
- All-protocol c = 0.5 gives 0.25.
- One protocol at chance and six perfect gives g = (exp(6/7 log 1 + 1/7 log 0.05) - 0.05)/0.95 and score g^2. Check this against a hand computation. It shows how a single protocol at chance costs about 0.30 in g.
- A zero-protocol case: the 0.05 stabiliser keeps the log finite.

**Direction of proxy bias.**
- 5-fold training uses about 80 percent of the cameras (about 238 versus 298 in the final fit). Slightly pessimistic, by roughly 0.01 to 0.03 (estimate).
- Camera disjointness matches the test shift axis, so the proxy is not optimistic on that axis.
- Residual risk: test cameras are drawn from a different geography or sensor mix than the supplied folds (unknowable, since no metadata is released). If so, the real score lands below OOF. Expected private band is OOF minus 0.0 to 0.06 (estimate).
- Test sampling noise: about 348 events per protocol, about 87 per slot, so BA standard error is about 2 percent per protocol and about 0.03 in score (estimate). Rank differences below that are not meaningful.

**Nested checks for post-hoc selection.** L2 grid, epoch count, pooling choice, backbone choice and any decode offsets each get a cross-fitted or sanity-fold check. Offsets, if used, are fitted on 4 folds of OOF and applied to the 5th.

---

## Overfit/underfit risks

**Overfit risks and mitigations**

| Risk | Why here | Mitigation |
|---|---|---|
| Camera-background memorisation | 298 cameras, about 19.6 events each, static backgrounds in every frame | Camera-disjoint folds; strong dropout on frame features; random-resized-crop, brightness/gamma/contrast, blur and noise augmentation; horizontal flip; optional frame-difference "motion" view that suppresses static background; L2 grid with a strong end |
| Memorising groups with a large fine-tune | About 300 independent groups | Capacity ladder (below): stop at the lowest rung that wins under grouped CV; full fine-tune is rejected |
| Latent collapse or invalid structure | 8-state model with soft maps has permutation symmetry and local minima | Multiple fixed-seed restarts, pair-cover penalty, Sinkhorn maps, Hungarian hardening, selection by train-only NLL plus structure check |
| Selecting noise via the grid | Few knobs but a noisy metric (score SE about 0.02 to 0.03) | Keep the grid small (5 L2 values, 2 hidden sizes), plateau selection, sanity fold, paired comparison |
| Optimistic decode offsets | 21 free parameters | Optional; cross-fitted; adopt only if the gain exceeds noise on 4 of 5 folds, otherwise drop |
| Frame-order or id artefacts | desc warns about row order | Permutation-invariant frame pooling; id never a feature; join by id |

**Underfit risks and mitigations**

| Risk | Mitigation |
|---|---|
| Backbone too small or weak on IR night frames | Start from DINOv2 ViT-L/14 (strongest generic backbone that fits the budget); also test ViT-B/14 and one non-self-supervised family; LP-FT last-blocks rung if the frozen probe plateaus |
| Global features hide small animals | Rich pooling (CLS plus mean of patch tokens, flip-averaged); optional patch-grid MIL (score every location, LSE pool); frame-difference view |
| Single-frame blindness | Pool the 3 frames jointly (LSE over per-frame potentials), plus the motion view |
| Loss mismatched with metric | Exact marginal likelihood on slot, per-protocol-equal weighting, uniform identity prior; recall-balanced decode |
| Too little training of the maps | Joint training from the start with an annealed temperature, then hard maps with a short head re-fit |
| Information loss from resizing | Frames are 128 px; upsample bilinear to 224 (and test 336 if cheap). Do not downsample further |

**Capacity ladder (stop at the lowest rung that wins under grouped CV):**
1. Frozen strong features, richly pooled, with a linear latent head.
2. Few-parameter MLP head (hidden 256 to 512) with strong dropout.
3. LP-FT: warm-start the head from the probe, fine-tune the last 2 to 4 blocks with a tiny LR for 1 to 2 epochs. Log the parameter-change norm and the CV gain over the probe.
4. Partial then full fine-tune only if rungs 1 to 3 fail and rung 3 shows clear gains.

---

## Recommended approach (primary + fallback)

**Primary: frozen generic backbone, 3-frame MIL, 8-state latent model with learned protocol maps.**

*Representation.*
- Each frame (grayscale replicated to 3 channels, bilinear upsample 128 to 224, the backbone's own fixed ImageNet mean/std) goes through DINOv2 ViT-L/14 (pinned revision, bf16/fp16 autocast, fixed batch).
- Per-frame feature = [CLS, mean of patch tokens], 2048 dims, averaged over the original and horizontally flipped view (flip TTA, per-sample).
- Training data uses cached views: original, flip, and 2 fixed-seed augmented views per train frame (random-resized-crop scale 0.6 to 1, gamma/contrast jitter, light noise). Test uses original plus flip.

*Latent head.*
- Feature standardisation uses train-fold statistics only (fit on training rows of the fold, applied to validation and test). Dropout 0.3 to 0.5 on input.
- Per-frame potentials l_t(z) = MLP or linear (8 outputs, shared across frames).
- Event potential s(z) = log sum_t exp(l_t(z)) - log 3, so evidence from any single informative frame can carry the event. Compare against mean pooling and attention pooling in the roadmap.
- q(z | x) = softmax_z s(z).

*Protocol maps.*
- Learned A_p in R^(8x4) for p = 0..6 (224 parameters in total).
- M_p = Sinkhorn projection of exp(A_p/tau) onto row sums 1 and column sums 2, with tau annealed over a fixed schedule.
- Slot posterior pi_p(s | x) = sum_z q(z | x) M_p[z, s].

*Loss.*
- Exact marginal NLL: -log pi_p(y | x), averaged first within protocol and then across protocols (matches the metric's hierarchy).
- Pair-cover consistency penalty lambda_c * ||sum_p M_p M_p^T - (6I + J)||_F^2, ramped up over a fixed schedule.
- Batch-level KL(mean_x q || uniform) with a small weight (justified by the derived 1/8 identity balance, checked on train).
- AdamW with weight decay.

*Training and selection.*
- Fixed epochs, fixed LR schedule (cosine), mini-batches of 256 events.
- R = 5 restarts with fixed seeds. For each fold and for the final fit, rank restarts by train-only criterion (train NLL plus structure penalty) and keep the valid ones.
- After training, harden M_p with Hungarian assignment (8 identities x duplicated 4 slots), then re-fit the head for a short fixed schedule with the maps frozen. This enforces the exact structure and makes the shipped model use one consistent 8-state identity space.

*Ensembling.*
- Average in slot-posterior space, per protocol (permutation-invariant across restarts, unlike latent space).
- Final shipped predictions: refit on 100 percent of train, average over restarts. Fold models are used only for OOF numbers and calibration checks.

*Decode.* argmax_s pi_p(s | x) with uniform slot prior (BA-optimal under the balanced training prior). Optional cross-fitted per-(protocol, slot) log-offsets, adopted only if gains exceed noise.

**Upgrades, only after the lean design is measured (roadmap):**
- (U1) Frame-difference "motion" view as an extra pseudo-frame through the same backbone and head.
- (U2) Patch-grid MIL: 8x8 pooled patch tokens (about 3.3 GB fp16 for ViT-L at 25k frames), per-location potentials with LSE pooling over locations and frames, for a translation-equivariant head.
- (U3) A second backbone family (ImageNet-supervised or generic CLIP/EVA image tower) for diversity of assumption. Blend in slot-posterior space only if within about 0.02 of the first.
- (U4) LP-FT of the last 2 to 4 blocks of ViT-B/14 (see fixed plan). Adopt only on a measured, consistent gain.

**Fallback (if the latent maps are unstable):** shared bottleneck with 7 coupled 4-way heads on the same frame-pooled features, plus the pair-cover consistency regulariser applied to head-implied identity equivalence classes. A second fallback: exact structure enumeration. The 1-factorisations of K8 can be enumerated in-script as the discrete candidate set (6240 labelled ones from memory, UNVERIFIED, about 780 distinct up to identity relabelling given ordered protocols). Hard-EM over structure, with slot relabelling handled per protocol. This is heavier and only enters if the soft search fails.

**Expected score (estimate, not a promise).** The central guess for the pooled OOF is 0.30 to 0.55, with a plausible spread of 0.20 to 0.65. Reasoning: a night camera-trap 8-way identification from frozen generic features might give identity accuracy of 0.65 to 0.85, giving slot accuracy of about 0.70 to 0.87 and a score of about 0.36 to 0.67. Discount for IR domain shift, camera shift and imperfect map recovery. Private expectation is OOF minus 0 to 0.06. Unverifiable here.

---

## Rejected options

| Option | Reason |
|---|---|
| Seven independent per-protocol 4-way classifiers | The description says near the floor; each sees only about 833 rows and must learn a union of two visually unrelated identities. Kept only as the floor probe |
| Full fine-tune of ViT-L or ViT-B end to end | About 300 independent camera groups; memorises backgrounds; A10G cost too high for 5 folds |
| kNN or retrieval of test events against train images | "Reverse lookup" and "external matching" bans; also a lookup rather than a trained decision |
| BioCLIP, MegaDetector, SpeciesNet, iNat-trained backbones, CLIP text prompts with species names | "Task-specific pretrained models not permitted"; semantic names are not released |
| Clustering or normalising test events by camera (test-time adaptation, camera-level priors) | Whole-test aggregation, banned by CLAUDE.md 2.3 #5 |
| Pseudo-labelling or self-training on test | Banned |
| Gradient-reversal camera-invariance training | No camera ids released; pseudo-camera ids from clustering would add a fragile component |
| Forcing the predicted slot frequencies to be balanced over the test set | Whole-test aggregation, banned |
| Large Optuna HPO | Score SE of about 0.02 to 0.03 means many trials select noise; use a small in-script grid |
| GBDT on frozen embeddings | Grey under the playbook and no better than a trained head; reject |
| Argmax-identity-then-map decode | Worse than marginalising over q for slot accuracy; use the marginal |
| Exact structure enumeration as primary | Heavier; kept only as the second fallback |
| Wall-clock or deadline-driven control flow | Rejected by the determinism checker |

---

## Fixed work plan & runtime budget

All counts are hard-coded constants at the top of the file. All time figures are estimates to be profiled locally before hard-coding. Device is fixed to `cuda`. Only logging uses the clock.

| Stage | Fixed plan | A10G estimate |
|---|---|---|
| Load, asserts, id joins, folds | read csv/npz, assert alignment | under 1 min |
| Feature extraction, primary backbone (DINOv2 ViT-L/14, 224, fp16, batch 128) | train 17,502 frames x 4 views (orig, flip, 2 aug) = 70k; test 7,302 frames x 2 views = 14.6k; total about 85k forwards at about 250 img/s | about 6 min (range 4 to 10) |
| CV of latent head | 5 folds x 5 L2 values x 3 restarts = 75 fits, 60 epochs each, tiny head on cached features | about 6 to 10 min |
| Final refit and inference | 5 restarts on 100 percent, then hardening and short re-fit | about 1 to 2 min |
| Validation, write, re-read | assert schema | under 1 min |

**Config A (frozen, shipped by default):** about 15 to 22 min.

**Add-ons and their cost:**
- A second backbone for diversity: +6 min extraction, +6 to 8 min CV. Config A plus second backbone is about 30 to 35 min.
- Motion view: +about 2 min.
- Patch-grid MIL: +about 4 min for extraction and storage, plus a heavier head, about +8 min CV.
- **Config B (LP-FT rung, shipped only if it measurably wins):**
  - ViT-B/14, first 8 blocks frozen under no_grad, last 4 blocks trained, 4 epochs, about 17.5k frames per epoch at about 400 img/s, about 45 s per epoch, about 3 min per model.
  - 5 folds plus 1 full refit is about 18 min.
  - Config B total is about 40 to 45 min, so it replaces the second-backbone add-on. Headroom is then under 30 percent, so config B must be profiled on a clean run before it is accepted.

**Memory estimates.**
- Host: images about 0.4 GB; frame features about 0.4 GB (2048 dims x fp16 x 85k); optional patch-grid about 3.3 GB.
- GPU: ViT-L/14 batch 128 inference at fp16 about 6 to 8 GB; head training under 2 GB.
- Download: DINOv2-L about 1.2 GB (HF only).

**Determinism settings:** seeds for `random`, `numpy`, `torch`, `cuda`; `PYTHONHASHSEED`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch; `cudnn.deterministic=True`, `benchmark=False`; `use_deterministic_algorithms(True, warn_only=True)`; fixed DataLoader generator and `num_workers`; fp32 for the head; no `os.cpu_count`, no `cuda.is_available` branching; Sinkhorn iterations fixed. HF model revision pinned to a commit hash (look it up at dev time; not available here).

---

## Metric-aware training & decode

**Loss matches the metric structure.**
- The metric is hard-label BA per protocol, geometrically averaged. The training loss is the exact marginal log-likelihood of the observed slot, averaged within protocol and then across protocols, mirroring the hierarchical averaging.
- Uniform per-protocol weights at first. If the per-protocol OOF BA varies a lot, test up-weighting weak protocols by a smooth fixed rule (such as weights proportional to 1/(0.05 + 0.95 c_p) from the first-pass OOF, cross-fitted), because the geometric mean's gradient with respect to c_p is 0.95/(0.05 + 0.95 c_p). Adopt only if the gain exceeds noise.

**Decode.**
- BA-optimal decision under a uniform slot prior is argmax pi_p(s | x), where the posterior comes from the shared q.
- Marginalise over identities. Do not take the MAP identity and then map it.
- Calibration is not needed for argmax, but the posterior must be well scaled for slot-space averaging of restarts. Optionally apply a per-protocol temperature, fitted on OOF.
- Optional per-(protocol, slot) offsets: 21 free parameters, fitted on OOF with cross-fitting, bounded by the uniform prior (offsets near zero). Report cross-fitted gain. Drop unless gains are consistent over at least 4 of 5 folds.

**Hard constraints at decode.**
- The structure is enforced inside the model: hardened M_p with exactly 2 identities per slot and a valid pair cover. This is a constraint combined with model evidence, not a standalone search.

**Oracle checks (dev-time, not shipped).**
- Planted-structure unit test: generate a toy 1-factorisation of K8 with random identity labels and Gaussian features. Verify that the learner recovers M_p up to identity relabelling and that the oracle decode reproduces gold slots. This tests the code only. It is not training data and does not appear in `solution.py`.
- Feed gold slot one-hots through the decode and the metric: score must be 1.0.
- Report the frozen per-protocol probe under the same folds as the yardstick, and require the latent model to beat it by more than noise.

---

## Structural signals

Every invariant the description guarantees, and how each is used:

1. **8 identities, 7 protocols, each a perfect matching, all 28 pairs once.** Use: the latent 8-state model; Sinkhorn maps with row sums 1 and column sums 2; the pair-cover penalty G = sum_p M_p M_p^T = 6I + J; Hungarian hardening. This is the primary structural signal and carries most of the sample-efficiency gain: all 5.8k rows jointly supervise one 8-way identity space.
2. **Balanced slots in every protocol imply uniform identities (derived).** Use: uniform prior, batch KL, argmax decode. Verify on train.
3. **Slot labels are arbitrary per protocol.** Use: learned maps with random initialisation; ensemble in slot-posterior space, which is label-permutation-invariant. Never average latent posteriors across restarts.
4. **Three time-ordered frames, any can be empty.** Use: permutation-invariant LSE pooling over frames. Time reversal is label-invariant, so this is the right invariance. Optional frame-difference view as a within-event cue.
5. **Horizontal flip does not change identity.** Use: flip augmentation and flip TTA. No geometry-tied inputs exist (no coordinates, no boxes), so no other input needs transforming.
6. **Camera disjointness.** Use: supplied folds; background-suppressing augmentations; the motion view. The cameras appear in neither train-versus-validation nor train-versus-test.
7. **Protocol as input only.** The protocol selects M_p. It is not a visual feature. A shared q across protocols is exactly what makes the cross-protocol evidence useful ("evidence linking identities is distributed across other protocols and other cameras").
8. **Positional balance of slots.** There is no semantic meaning of slot index across protocols. Do not use slot index as a feature or prior beyond the within-protocol balance.
9. **Ids are arbitrary.** Never a feature; join by id.

Quantitative consequence to verify: if slot-centroid decomposition (finding 8) and hardened-map checks (finding 7) succeed, the model is effectively an 8-way species classifier trained with weak "union of two" supervision.

---

## Experiment roadmap

Each step has a stop criterion. Compare paired on the same folds. Fix the model seed list. Log id, change, pooled-OOF score, per-fold scores, per-protocol BA, estimated runtime.

1. **Contract, metric, validation (stop when all unit tests pass).** Implement the metric and the tests above, the id joins, the fold loader and the `validate_submission` function. Run the diagnostics in Data findings items 1 to 5 and record real numbers over the hypotheses.
2. **Floor and baseline (stop when end-to-end and valid).**
   - Extract frozen DINOv2-B and DINOv2-L features. Score the per-protocol probe floor and the shared coupled-head baseline.
   - Produce a valid `submission.csv` from the best baseline and run the full validity checklist. First credit candidate.
3. **Primary latent model (stop when it beats the floor by more than 0.02 pooled-OOF and in at least 4 of 5 folds).**
   - Implement the 8-state head, Sinkhorn maps, pair-cover penalty, KL and Hungarian hardening.
   - Check structure recovery rates across restarts (finding 7). If under about 60 percent of restarts reach a valid factorisation, fix the optimisation first (tau schedule, penalty ramp, initialisation). Only then consider the enumeration fallback.
4. **Representation and pooling (one change per experiment; stop at the first rung with no gain beyond noise).**
   - Backbone size and family (B versus L).
   - Pooling: LSE versus mean versus attention.
   - Rich pooling (CLS + mean-patch) versus CLS only.
   - Resolution 224 versus 336.
   - Motion view (U1).
   - Patch-grid MIL (U2).
5. **Regularisation (stop at a plateau).** L2 grid including the strong end, dropout, hidden size, number of views, epoch count. Pick the plateau centre, not the sharp optimum. Report on the sanity fold once.
6. **Capacity ladder, rung 3 (only if frozen performance plateaus).** LP-FT of the last 2 to 4 blocks. Record the parameter-change norm and the paired CV gain over the probe. Adopt only with a gain above noise and a runtime within budget.
7. **Metric-aware extras (only after 3 to 6 are solid).** Cross-fitted per-protocol weighting and slot offsets. Drop if not consistent.
8. **Diversity (only if members are within about 0.02 of each other).** A second backbone family in slot-posterior space. Check that the blend gains more than noise.
9. **Final fixed-plan run.** Run twice from a clean `working/`, diff the two outputs (expect near-identical), check the prediction distribution against OOF (per-protocol slot frequencies near uniform, no constant output), check runtime headroom, then run the compliance audit and upload. Use credits only for: floor/baseline, best single, ensemble if any, and final.

When two independent solvers would plausibly converge on a technique (frozen DINOv2 features plus a latent shared head), test it before any proxy-driven tweak.

---

## Compliance audit

CLAUDE.md section 7 and the strategist section B self-audits, against this plan.

| Check | Result |
|---|---|
| Any code path reads the test set for more than one-sample prediction (stats, vocab, scaler, clustering, dedup, rank-normalising, pseudo-labels)? | No. Scalers are fit on train folds only; test is transformed. Test rows are processed independently. |
| Wall-clock in a condition, loop, `break`, `min()` or library timeout? | No. Time is logging only. |
| `cuda.is_available`, `os.cpu_count`, import fallbacks, try/except that changes work? | No. Fixed `device="cuda"`, fixed worker and thread counts. |
| Hard-coded constants tuned offline? | The L2 grid, restarts and tau schedule are searched or fixed in-script. Architecture-level choices (backbone, pooling) are development-time design decisions and each is cross-validated. The only offline constants are structural facts from the description (K = 8, 7 protocols, 4 slots). Offsets and weights, if used, are fitted in-script on OOF. |
| External data, synthetic training data, self-hosted weights, GitHub models, non-allowed libs? | No. HF/timm generic weights only, pinned revision. The planted-structure unit test is dev-only and not in `solution.py`. |
| Strip-the-ML test (remove all trained components)? | Passes. The backbone provides generic features; the latent head and maps are trained from labels. Without them nothing predicts. There is no rule, regex or lookup. |
| Whole-test aggregation (normalisation, prior estimation, camera grouping across test rows)? | None. Slot priors are train-derived and uniform. |
| Sibling leakage? | Camera-disjoint supplied folds; train-only diagnostics for burst structure. |
| Related-row features cross-referencing group-mates? | None. Frame differencing uses the row's own 3 frames. |
| Reverse lookup / external matching / kNN against train? | None. Parametric heads only. |
| Task-specific pretrained weights? | None. DINOv2 (generic LVD-142M) or ImageNet / generic CLIP image towers; avoid iNat, BioCLIP, MegaDetector, SpeciesNet. |
| Inference-only or frozen-plus-small-head concerns? | Primary head is trained with a latent-structure objective, which the description lists as an intended approach. If a reviewer treats frozen features as grey, config B (LP-FT of the last blocks, with logged parameter-change norm and CV gain) is the compliance layer. |
| Source readable, under 512 KB, no blobs? | Plan: single plain-text file, well under the limit. |
| Seeds and determinism? | Fixed seeds, deterministic kernels (warn-only), fp32 head, fixed iterations and schedules. |
| Challenge-specific restrictions honoured? | Yes. Every ban in the description is listed in the compliance regime section. |
| Submission validity? | `validate_submission` against `sample_submission.csv` (columns, order, ids, row count) plus: dtype int64, values in {0,1,2,3}, not bool, reload with `keep_default_na=False`. |

---

## Open questions & assumptions

**Reviewer questions (plan under each reading):**
1. Is DINOv2 (LVD-142M self-supervised weights) accepted as a generic vision backbone under "generic vision weights permitted only when otherwise allowed by the platform"? If not, fall back to an ImageNet-supervised timm backbone (ConvNeXt or ViT); expect a modest score drop.
2. Is a frozen backbone with a trained latent head acceptable, or does the platform expect backbone fine-tuning? If fine-tuning is required, ship config B (LP-FT, with the change norm and CV gain logged).
3. Are iNaturalist-pretrained timm weights "task-specific"? Plan assumes yes and avoids them.
4. Is `train_folds.csv` guaranteed present in `public_dir` at grading? The plan assumes yes and asserts it. If not, the script would need an in-script grouped split from train-derived pseudo-cameras, which is a different design to be agreed first.
5. Are within-event frame differences and a derived "motion" view acceptable as inputs alongside the images? Assumed yes (within-row).
6. Any stated runtime limit for this challenge beyond the platform default? The description names only the A10G. Plan assumes the 50 min target and 1 h worst case.

**Assumptions made (A1 to A8 are listed under Compliance regime).** In addition:
- A9: the 8 identities are visually distinct classes (species-like), balanced at 1/8. To be checked on train.
- A10: test events have the same per-protocol composition as train. Not studied (only the 2,434 row count is used for runtime planning).
- A11: the three frames are time-ordered but the label is invariant to time order.

**What could not be verified here.**
- Every dataset statistic and every runtime figure (all estimated, to be profiled on the dev box).
- The DINOv2 HF revision hash and exact checkpoint availability.
- The count of K8 1-factorisations (6240 labelled, about 780 up to relabelling for ordered protocols), quoted from memory; only matters for the enumeration fallback.
- Whether Sinkhorn-map training reliably recovers a valid factorisation on real features; this is the main technical risk and the first thing to measure at roadmap step 3.
- Real vs estimated score; the expected range (central 0.30 to 0.55 pooled OOF, private OOF minus 0 to 0.06) is an estimate from first principles only.

**Three biggest risks:** (1) latent-structure recovery fails or lands in an invalid factorisation; (2) camera-appearance shift to 131 unseen sites (frozen features carry the background); (3) the weakest protocol dominates the geometric mean, so a single hard visual pair can cost more than average accuracy suggests.
