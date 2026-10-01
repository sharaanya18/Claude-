# Eris plan: T1 field-capture burst midpoint recommendation

Status of evidence: NO dataset was available. Everything under "Data findings" is a diagnostic to run and a hypothesis, all UNVERIFIED. Structural statements that come from the challenge text alone (8 frames, L first, R last, 6 shuffled interior candidates, 1.0 / 0.2 / 0 scoring) are marked as read from the description. Column names of train.csv / test.csv are not known; "the L / R / c0..c5 image references" is used for whatever they are called. Nothing here is a promised score.

## Contract & decision unit

**One valid answer.** For each test id, a JSON list of exactly 2 distinct candidate identifiers drawn from {c0..c5}, best first, in the column `ranked_candidates_json`, one row per test id, in `sample_submission.csv` order. The element format (strings "c3" vs ints 3, JSON spacing) must be copied from the gold-label column of train.csv and from sample_submission.csv (unverified; see D1). Invalid (scores below zero or fails validation): wrong columns, wrong/duplicate ids, a candidate outside the row's six, a duplicated candidate in the pair, unparseable JSON, a length other than 2. Merely low-scoring: valid pair with wrong order or wrong members.

**Metric, term by term (read from the description).** Per row: 1.0 if the gold candidate is ranked first, 0.2 if ranked second, 0 otherwise; mean over rows. Check: a uniformly random ordered pair gives 1/6*1 + 1/6*0.2 = 0.2, matching "random scores 0.2". The metric is rank-only (no calibration term), single gold per row, no gating, no hierarchical averaging. The only decode-relevant fact is the 1 : 0.2 weighting: expected score of an ordered pair (a, b) is p_a + 0.2 p_b, maximised by sorting candidates by posterior probability of being gold.

**True independent unit.** The row (one burst) is the decision unit, but rows are NOT independent: bursts chain (one row's R photo can be the next row's L photo), so scenes, backgrounds, camera, lighting and subjects are shared across consecutive rows. The independent unit for validation is the chain component (union-find over recovered R_i ~ L_(i+1) links), not the row. Effective sample size = number of chain components, unknown until D3 is run; the number of labels is 1,250 single-gold labels, each worth about log2(6) bits at most.

**Latent structure (read from the description).** A burst is 8 frames in time order f0..f7. L = f0 and R = f7 are given. The six candidates are f1..f6 in shuffled order. With evenly spaced frames the temporal midpoint is 3.5, so the gold is f3 or f4 (the two middle frames). The label is "the" gold midpoint candidate, so there is a tie-break rule or the timestamps are irregular; this is the single most important unknown (D6, Open question 1).

**Pipeline stages, to be diagnosed separately.**
1. Candidate coverage: gold is one of the six candidates by construction (verify 100% in train). Recall@2 of the pool is trivially 1; nothing to do.
2. Evidence quality: can the representation tell frames of the same burst apart in time order despite per-image independent crops, flips and photometric changes? Diagnostic: same-photo AUC (R_i vs L_(i+1)), and the similarity-vs-temporal-gap decay.
3. Ordering/scoring: given the evidence, the posterior over the 6! = 720 interior orderings, marginalised to P(candidate c is the gold slot). Diagnostic: rate at which gold is in the model's top-2 ("middle pair recall"), and top-1 rate.
4. Decode: sort by marginal posterior. Diagnostic: order accuracy r between the two middle candidates given both are in the top-2. Expected score is approximately q * (r + 0.2 (1 - r)) + small terms, where q is middle-pair recall.

## Compliance regime

**Domain.** Computer vision (CLAUDE.md 6.3) with a structured-decision output. Allowed per the description: general-purpose pretrained weights from HF/timm, one A10G, about 1 hour, the solution must train a model in-script, no external data.

**Allowed.** Pretrained backbone from HF (DINOv2 ViT-L/14, `facebook/dinov2-large`, pin the commit revision at first download and hard-code it), in-script training of a head and of the last backbone blocks, TTA by horizontal flip averaging (per-sample), per-row use of the row's own 8 images, reference banks built from TRAIN images, PCA fit on TRAIN patches only, augmentation of real train images (flips, crops, photometric jitter), recombining real labelled train units.

**Grey.** (a) Frozen embeddings plus a trained head is grey for CV (6.3). Mitigation: the learned metric/potentials are genuinely trained by exact marginal likelihood and a last-stage fine-tune (LP-FT) is part of the primary subject to a non-inferiority rule (see approach). (b) Within-row candidate-to-candidate similarity (seriation) is a relation among a row's own mates; it is the task's own input (the row gives all 8 images) and is a function of that row only, so it is kept, with one reviewer question. (c) Reference-bank normalisation uses train images only; fine.

**Banned or avoided, and why.**
- Any use of other TEST rows for a test row: linking test chains (R_i to L_(i+1)), neighbour-burst context, whole-test normalisation, test-fitted PCA/scalers/banks, de-duplication against test, pseudo-labels. These are test-set aggregation / leak tricks (CLAUDE.md 2.3 #5). The chain structure is used on TRAIN only (to build validation groups).
- Matching test images to train images to exploit overlap (train/test photo overlap or label lookup): banned as a leak trick; it also cannot be used to design validation (only schema, row count, id format and size statistics of test are allowed).
- Row order / id order / file names / zip member order / JPEG file size / EXIF timestamps / candidate position index c0..c5 as features. All are artefacts or metadata; audited in D1/D2/D11 and excluded.
- Raw pixels into a tabular model; time-based branching; env-dependent fallbacks; hard-coded offline constants (CLAUDE.md 3).

**Reading under every plausible interpretation.** Use only (i) the row's own 8 images, (ii) train-fit parameters, (iii) train-only banks and PCA. The measured cost of not using neighbouring test rows (cross-row context) is quantified on train grouped-CV for information only (roadmap step 5b) and is not shipped.

**Strip-the-ML test.** Remove the trained parts (learned gap-decay potentials, learned projection, slot weights, last-block fine-tune): what remains is a frozen-embedding betweenness heuristic (candidate closest to the L/R midpoint in embedding space). It is measured as the zero-learning baseline (D7) and must be clearly worse than the trained model; if the gap is within noise, the plan is too rule-based and the learned projection / LP-FT must be strengthened.

## Data findings

None of the following has been run. Each item gives the exact diagnostic on TRAIN, the expected outcome (hypothesis, UNVERIFIED) and what it changes.

**D1. Schema, label format, position prior.** Print columns, dtypes, row counts (expected about 1,250 train, a few hundred test), id format, how the L / R / c0..c5 references map to files in `train_images.zip`, the exact gold label column and its format (index vs name vs JSON), `sample_submission.csv` layout. Check gold in {c0..c5} for 100% of rows. Chi-square test of gold index uniformity over c0..c5. Hypothesis: uniform (candidates shuffled), random-pair baseline 0.200 +/- 0.012. If non-uniform: a generation artefact; do not use it as a feature (fragile), only log it. Consequence: fixes output format and the baseline sanity number.

**D2. Image statistics.** Width/height/aspect distribution per split (test: size statistics only), JPEG quality, colour space, EXIF presence (DateTimeOriginal, subsecond), file-name pattern, zip member order, file-size vs gold correlation, number of unique images. Hypotheses: crops give varying sizes and aspects within a single row; EXIF stripped by re-encoding; file-size/name/zip-order carry no label. Consequence: fixes the resize policy (square resize vs letterbox), the input resolution ladder (224 / 336 / 448), and the leak audit. If EXIF timestamps exist: do not use (metadata); ask a reviewer.

**D3. Chain recovery (train only).** Embed all unique train images with DINOv2 (flip-averaged CLS + mean patch). For every row's R, find the nearest L over all other rows; histogram top-1 cosine and the gap to the 2nd neighbour; mutual-NN links; choose the link threshold at the valley in the gap histogram; union-find into chain components; report component count, size distribution, fraction of rows with a linked successor, and whether id order agrees with link order (information only, never a feature). Hypotheses: most rows (about 80-95%) have a successor; chains are tens to a few hundred rows; a clear bimodal separation exists because the same photo recurs under independent perturbations. Consequence: defines the CV groups; if one giant component holds more than 50% of rows, switch to contiguous-block CV along the recovered chain order with a purge of +/-1 row.

**D4. Invariance ceiling and representation choice.** Same-photo pairs (R_i vs L_(i+1)) under independent crop/flip/photometric changes: cosine distribution vs different frames of the same burst vs frames from different chains. Compare CLS vs mean-patch vs concatenation, last vs penultimate layer, flip-averaged vs not, resolution 224/336/448, DINOv2-L vs a CLIP/SigLIP ViT-L. Metric: same-photo AUC and separation margin (this uses chain links, not gold). Hypotheses: flip-averaging and 448 px raise the AUC; DINOv2 beats CLIP on instance-level similarity; patch-level matching gives more separation than global cosine when crops differ. Consequence: picks backbone, pooling, resolution (final choice still validated on the gold metric).

**D5. Temporal decay structure.** For every train row compute the 8x8 similarity matrix (L, six candidates, R) under each channel. Without order labels, fit the permutation-marginal model (below) with gap-specific decay slopes a_delta, delta = 1..7, and inspect the fitted a_delta and the likelihood gain over a gap-free model. Hypotheses: similarity decays monotonically with temporal gap (Robinson-like structure); L-R similarity is the smallest in the row; adjacent-frame similarity is close to the same-photo ceiling from D4. If adjacent frames are near-identical (high-rate burst), the usable signal is subtle and local-patch channels and higher resolution matter; if frames differ a lot, global channels suffice. Consequence: decides whether local-correspondence channels are needed in the primary.

**D6. Tie-break diagnostic for the midpoint (decisive for the ceiling).** After fitting the model, per row identify the gold g and the other middle candidate m (the non-gold candidate with the highest posterior mass on slots {3,4}); record the fraction of rows in which g is closer to L than m. Also fit the slot-weight vector w over slots 1..6 and read its mass on slots 2, 3, 4, 5. Hypotheses: (H-a) gold is deterministically f3 (or deterministically f4): fraction near 1 (or 0), ceiling about 1.0; (H-b) random tie: fraction near 0.5, ceiling 0.6 (= 0.5 * 1 + 0.5 * 0.2 when both middle frames are in the top-2); (H-c) irregular timestamps: appreciable mass on slots 2 and 5. Consequence: sets what is achievable, whether the swap-L/R augmentation is label-consistent (needs symmetric w), and the expected range below.

**D7. Zero-learning baselines (floor, not shipped).** On train: random (expect 0.200), always c0/c1, the embedding betweenness heuristic (rank candidates by | d(c,L) - d(c,R) | and d(c,L) + d(c,R) - d(L,R)), spectral (Fiedler-vector) seriation of the 8 frames. Hypotheses: betweenness 0.30-0.40 if the global embedding carries the time signal; spectral seriation similar. These are the strip-the-ML reference and the first sanity gate (anything at or below 0.22 means a pipeline bug, not a hard problem).

**D8. Information ceiling estimates.** (i) Middle-pair recall q: fraction of train rows whose gold is in the model's top-2 (hypothesis 0.6-0.85 depending on D5). (ii) Order accuracy r given both middle frames in the top-2 (hypothesis 0.5-0.7 because f3 and f4 are one step apart and nearly equidistant from L and R by definition). (iii) Resulting ceiling about q * (r + 0.2 (1 - r)); under H-b with perfect middle-pair recall it is 0.60. Report the per-row score standard deviation (about 0.4-0.45 for scores in {0, 0.2, 1}) and the resulting standard error of a 1,250-row mean (about 0.012; paired deltas are tighter).

**D9. Duplicates and irreducible ambiguity.** Rows with identical or near-identical L and R (links in D3), exact duplicate candidates inside a row, rows sharing a photo set but different gold. Hypotheses: few or none; any pair of near-identical candidates in a row (static scene) is irreducible ambiguity and caps the score.

**D10. Photometric/low-level leak audit.** Predict gold from per-image brightness, contrast, sharpness, JPEG size, EXIF-free metadata with a tiny logistic model under grouped CV. Hypothesis: AUC about 0.5 because photometric changes are independent per image. Any signal above noise is an artefact, excluded from features.

**D11. Scene diversity and background dominance.** Within-chain vs across-chain similarity distributions; number of scene clusters; share of rows where background texture dominates similarity (vegetation, soil). Hypothesis: field photos share heavy common background, so raw cosine is dominated by it; reference-bank normalisation (below) helps. Consequence: justifies the bank channels (measure gain on gold metric, grouped CV).

**D12. Candidate sanity within rows.** Per row, the distribution of max candidate-candidate similarity vs L-R similarity (are some candidates nearer to each other than to L/R, confirming a linear path structure?). Hypothesis: the path structure holds in most rows; the rest are static or noisy rows.

## Validation design

**How the test split was probably made (unverified).** Either random by row or by chain; neither can be confirmed from test content (only schema, row count, id format and size statistics may be used). Choose the pessimistic, leak-free reading: hold out whole chain components. If test rows actually share chains with train, true scores will be a little higher than the grouped CV; if not, grouped CV matches. Either way, do not exploit overlap.

**Groups.** Chain components from D3 (union-find over R_i to L_(i+1) mutual-NN links above the valley threshold, plus any near-duplicate photo within any row). If the largest component exceeds 50% of rows, use contiguous blocks along the recovered chain order (purge +/-1 row). The derivation of groups is done in-script from train images only.

**Scheme.** GroupKFold, 5 folds, 3 different group-shuffle seeds (15 fits per config); report mean +/- std across folds and across seeds; compare designs paired on identical folds; a change is accepted only when the paired mean gain exceeds the larger of 1 standard error of paired fold differences and 0.01, and has the same sign in at least 12 of 15 fold-fits. A separate 20% sanity holdout of groups is carved out once, never used for selection, and scored a single time before the final run.

**Metric.** Re-implement in-script and unit-test: perfect ranking = 1.0; gold second = 0.2 for every row; gold absent = 0; reversed order of a perfect ranking = 0.2 on the pair when gold is second; constant ranking (c0, c1) approximately 0.2 on random-position data; average over all 720 candidate permutations = 0.2 exactly for a single row. Also log middle-pair recall q, top-1 rate and order accuracy r each run.

**Nested selection.** Every post-hoc selection (ridge strength grid, resolution, channel set, slot-weight prior strength, LP-FT inclusion) is run inside the outer fold on inner groups (nested) or cross-fitted; the reported OOF is the outer-fold score. Reference-bank construction for train rows excludes the row's own chain (leave-own-group-out) while the bank for a held-out or test row is the train bank; sizes are matched (fixed bank size, fixed seed) so train-time and test-time features have the same distribution.

**Expected optimism.** The self-built proxy will probably over-estimate the real private score by 0.02-0.05 (incomplete chain recovery leaves some leakage; selection effects); use it for relative comparisons only.

## Overfit/underfit risks

**Overfitting.**
1. Few independent groups, scenes shared across rows: a flexible head memorises scenes. Mitigation: grouped CV by chain; head with tens of parameters (gap-decay slopes) before any learned projection; projection is low-rank (rank <= 32) with a very wide ridge grid (1e-4 to 1e2, strong end included) selected per channel by the exact metric nested in the CV.
2. Weak supervision (one gold per row, about 1,250 labels): the latent ordering is identified only through the gold slot. Mitigation: the model has few latent parameters (a gap-decay vector per channel, 6 slot weights with a prior on slots 3 and 4), exact marginal likelihood, and a prior that ties the model to the description's structure.
3. LP-FT of a ViT-L on 1,250 rows can memorise chains. Mitigation: last 2 blocks only, LR 1e-5, 2 epochs, weight decay, parameter-change norm logged, accepted only by the non-inferiority rule on paired folds.
4. Many knobs on a noisy CV (standard error about 0.012). Mitigation: a small fixed grid, repeated split seeds, plateau choice, no validation-triggered stopping (fixed steps), nested checks.
5. Position artefacts (gold index, file order). Mitigation: not used (D1, D10).

**Underfitting.**
1. Too small or too global a representation: random crops and flips mean pixel alignment is impossible, and adjacent frames may differ by subtle local changes. Mitigation: DINOv2-L at 448 px with CLS plus mean patch pooling plus local patch-correspondence statistics, flip-averaged; ladder to a g-size backbone only if measured.
2. Background dominance in similarity. Mitigation: reference-bank normalised channels (several fixed banks, train images only).
3. Loss mismatched with the metric: use exact marginal NLL of the gold slot plus, optionally, a direct smooth expected-score term.
4. The unavoidable tie ceiling (D6, D8): not underfitting, but reported so the target is not mis-set.
5. Information thrown away by preprocessing (aspect distortion, down-sizing): decided by D2/D4 and resolution ablation.

## Recommended approach (primary + fallback)

### Primary: permutation-marginal seriation on strong frozen features, with learned metric and a last-stage fine-tune

**Idea.** Treat each row as a seriation problem. The unknown is the temporal order of the six candidates between the fixed ends L (slot 0) and R (slot 7). Score each of the 720 orderings pi with a small number of learned potentials, softmax-normalise over all 720, marginalise exactly to the quantity that is asked (which candidate sits in the middle slot), and train by exact marginal likelihood on the gold labels. Decode by sorting the marginal posterior.

**Representation.** DINOv2 ViT-L/14 (HF, pinned revision), bf16, input about 448 px (final choice from D4 and the resolution ablation), each image encoded unflipped and horizontally flipped. Per image keep: CLS, mean of patch tokens, and patch tokens reduced by a PCA to 64 dims fitted on TRAIN patches only. Global similarity between two images = average over the four (flip, flip) combinations (or max over the relative flip for the local channel, since flips are applied independently per image). Channels (about 6): global CLS cosine, mean-patch cosine, local correspondence score (mean of the top-k row maxima of the patch similarity matrix, fixed k, relative-flip max), plus reference-normalised versions: for image X let mu_X, sigma_X be the mean and std of its similarity to a fixed bank of train images from other chains; channel value z(A,B) = ((s(A,B) - mu_A) / sigma_A + (s(A,B) - mu_B) / sigma_B) / 2, averaged over several fixed banks. Distances d_ch = 1 - s or -z.

**Potentials (strip-test accountable, few parameters).** Score(pi) = - sum over channels ch, sum over all 28 frame pairs (i < j) of a_{ch, j-i} * d_ch(frame pi_i, frame pi_j), with frames 0 and 7 fixed to L and R. The parameters are the gap-specific slopes a_{ch,delta} (6 channels x 7 gaps = 42), constrained non-negative (softplus) with a second-difference smoothness penalty across delta. This encodes "similarity decays with temporal gap" (Robinson structure) without hard-coding any constants; everything is learned by the likelihood. An optional unary term uses the relative-position features t_hat = d(c,L) / (d(c,L) + d(c,R)) and the margin to the runner-up. Because L and R are fixed in place, the model is not invariant under reversal of the interior order, so orientation is identified; the model is exactly equivariant to the shuffled c-index.

**Likelihood.** P(gold = c) = sum over slots k in 1..6 of w_k * P(slot of c = k), where w = softmax(theta) over slots, initialised with equal mass on slots 3 and 4 (read from the description: 8 frames, even spacing) and a small floor on slots 2 and 5 with a ridge on theta, so irregular timestamps (D6, H-c) can be absorbed. Loss = - log P(gold) summed over rows (weights 1 per row since the metric is a plain mean), plus ridge penalties. Everything is enumerated exactly: 1,250 rows x 720 orderings x 28 pairs x 6 channels is a few hundred million multiply-adds per step on the GPU (milliseconds).

**Learned metric (rung 2 of the capacity ladder).** If the fixed-distance model leaves a measured gap, replace the fixed cosine channels by cosines of a low-rank linear projection W (1024 -> 32) of the frozen CLS/mean-patch features, initialised from the PCA basis, trained by the same marginal likelihood with weight decay toward the init, strength chosen by nested grouped CV over a very wide grid.

**Last-stage fine-tune (rung 3, compliance layer and possible gain).** Unfreeze the last 2 transformer blocks and the final norm of DINOv2-L only for the global channels (CLS + mean patch), warm-start the head from the probe (LP-FT), LR 1e-5 for the backbone and a separate small LR for the head, AdamW, weight decay 0.05, 2 epochs, batches of 4 rows (32 images) with per-image independent random resized crop (scale 0.6-1.0), horizontal flip and colour/brightness jitter, mirroring the release perturbations; the loss is the same marginal NLL through the permutation head. Log the parameter-change norm and the paired CV gain over the probe. Rule: ship the fine-tuned version if its paired gain on the same folds is at least -0.005 (non-inferior; it provides the genuine-training layer the platform asks for), otherwise ship the probe+learned-metric version and state the reviewer question about frozen features plus a trained head. The ship decision is made in dev; the final script contains one fixed code path.

**Refit and ship.** Fold models produce OOF scores and checks. The shipped predictions come from a refit on 100% of train rows with the same fixed number of steps (derived from CV), because the head has only tens to tens of thousands of parameters and a refit transfers (verify by comparing fold-model vs refit predictions on the sanity holdout).

**Decode.** Marginal posterior P(gold = c) per candidate; submit the top-2 in descending posterior, ties broken deterministically by candidate index.

**Why it fits this data.** The gold is defined by temporal position, which is unobserved; the description gives the generating structure (8 frames, fixed ends, shuffled interior). Learning a similarity-versus-time-gap law and inferring the ordering exactly uses all six candidates and both ends, handles independent crops/flips through flip-averaged, locally matched, bank-normalised evidence, and keeps the learned part small enough for 1,250 labels.

### Fallback: per-candidate listwise scorer on pairwise features

For each row and candidate c: the same channel distances to L, R, to each other candidate (sorted: nearest, second nearest), betweenness (d(c,L) + d(c,R) - d(L,R)), t_hat and |t_hat - 0.5|, within-row relative transforms (rank, z-score, margin to the runner-up of each quantity among the six), and a seriation position from a spectral order as a FEATURE (never a rule). Model: LightGBM LambdaRank/softmax-objective (query = row) with fixed seeds, shallow trees, strong regularisation, plus a diversity member using a second backbone (SigLIP/CLIP ViT-L) features; blend by train-reference rank average. Compliance: grey (GBDT on frozen embeddings); the primary's trained potentials and LP-FT are the answer to that. Expected to be a few points below the primary because it does not use the permutation structure; used if the primary is unstable or rejected on the latent-structure grounds.

## Rejected options

1. Cross-row context (neighbour bursts, linking test chains, trajectory extrapolation across bursts): banned as test-row pooling. It would likely help; the cost is measured on train only (roadmap 5b).
2. Matching test images to train images or to each other to inherit labels or neighbours: banned leak/dedup trick.
3. End-to-end full fine-tune or a set transformer over the 8 images: with about 1,250 single-gold labels and chain-shared scenes it memorises groups; rejected per the capacity ladder; kept only as a last roadmap rung if the lean design shows an underfit signature.
4. Pure embedding betweenness / spectral-seriation rule as the solution: fails the strip-the-ML test; kept as the baseline.
5. Photometric or file-level cues (brightness trends, JPEG size, EXIF, zip order, id order, candidate index): independent per-image photometric changes make them unreliable, and they are metadata/artefacts.
6. Pixel-space alignment, optical flow, SIFT-style hand matching as the core: crops and flips are independent per image; also rule-based.
7. Joint sequence decoders / sampled decoding / extra searches in the primary: the exact enumeration over 720 orderings already gives the exact marginals; no extra machinery (primary-design gate).
8. Synthetic burst construction (blending frames) or LLM-generated data: banned.
9. Long-span chain rows (L_i, R_(i+1), gold R_i, with distractors from the interiors) as extra training data: legitimate recombination of real train units and tie-free labels, but changes the slot geometry and adds machinery; deferred to roadmap step 5a, shipped only if the measured gain exceeds noise.
10. Larger backbones (DINOv2-g) at the start: a roadmap rung after the lean design is measured.

## Fixed work plan & runtime budget

All constants are fixed in the script (seeds, folds, epochs, steps, k, bank size, rank, resolution after the D4 ablation fixes it); no wall-clock branching; `device = "cuda"`; fixed `num_workers`; deterministic algorithms with warn_only; all HF weights downloaded at a pinned revision. Estimates are for one A10G and are NOT measured; profile one stage of each before freezing.

| Stage | Fixed plan | Est. time |
|---|---|---|
| 0. Read/validate inputs, unzip, build row table | assert schema, ids, gold in candidates, image count | 1 min |
| 1. Decode and resize about 11k unique images (about 8.8k train, about 2.4k test) | fixed square resize, 4 workers | 2 min |
| 2. DINOv2-L features at about 448 px, bf16, unflipped + flipped (about 22k forwards at about 50-60 img/s) | store CLS, mean patch, PCA-64 patch tokens (PCA fit on train only), fp16 | 8 min |
| 3. Chain recovery and groups (train), bank construction (train only, fixed seeds) | mutual NN + valley threshold, union-find | 1 min |
| 4. Pairwise channel tensors for all rows (28 pairs x 4 flip combos, local top-k, bank z-scores) | 8 banks x 256 images for local, 1024 for global | 3 min |
| 5. Probe head CV: 5 folds x 3 split seeds x ridge grid (about 8 values), 300 fixed Adam steps, full-batch exact marginal | seconds per fit | 4 min |
| 6. LP-FT paired check on 2 folds (2 epochs each at about 2.4 min/epoch) | logs parameter-change norm and paired gain | 10 min |
| 7. Final refit on 100% of train: probe/metric head, plus LP-FT 2 epochs | fixed steps | 6 min |
| 8. Test inference (about 0.3k rows x 8 images, flip-averaged, enumerate 720 orderings), write and re-read CSV, validator | | 2 min |
| Logging, slack | | 3 min |
| Total | | about 40 min estimate |

Headroom: against a 1 h worst case about 33%; against the 1.5 h ceiling much more. If profiling shows the LP-FT stages exceed budget, reduce to one paired fold and 1 epoch (fixed in the script, not branched on time). Memory: patch tokens 11k x 2 flips x 784 x 64 fp16 about 2.2 GB; ViT-L bf16 about 0.7 GB; LP-FT with gradients only on 2 blocks at 32 images x 448 px well under 24 GB.

**Input/output validation in the script.** Assert required files and columns; row counts and id format; every referenced image exists and decodes; features finite; each output list has 2 distinct candidates from the row's six; JSON element format equals the train gold format; ids and order equal `sample_submission.csv`; re-read the written CSV with `keep_default_na=False`; the CLAUDE.md `validate_submission` function plus a JSON-parse check on `ranked_candidates_json`. No constant fallback output is written (it would score 0.2, a silently degraded path); failures raise loudly. Run the full script twice and diff the predictions (expect identical or near-identical; LP-FT may differ slightly under non-deterministic kernels; decode constants are few and robust).

## Metric-aware training & decode

1. Back-solve the metric: it reduces to "which candidate is gold" with a fixed 1 / 0.2 / 0 payout; the richer target recoverable is the latent temporal order, learned through the gold slot marginal (nothing more is invertible).
2. Loss: exact marginal NLL of the gold, with per-row weight 1 (the metric is a plain row mean; no hierarchical averaging). Optionally add a smooth expected-score term E[score] = p_(1) + 0.2 p_(2) computed from the sorted marginals; adopted only if it beats NLL on paired folds.
3. Ordinal/tie handling: the slot weights w over slots 1..6 absorb tie-break asymmetry (D6) and irregular timing; ridge toward the 3 and 4 prior; learn the tie parameter a = P(slot 3 | gold) jointly.
4. Decode: expected-utility optimal ordering is descending posterior (maximises p_a + 0.2 p_b); no constants to tune, so nothing to over-fit. Posterior calibration only matters for the relative weighting of unary and gap-potential terms and for the L2 strengths, all set by nested CV. If a tiny per-slot recalibration is useful, fit a cross-fitted scalar temperature on OOF marginals and report it cross-fitted; it cannot change the order unless it differs per candidate, so it is not needed for the ranking itself.
5. Hard constraints at decode: the output is two distinct candidates from the row's six; guaranteed by construction and asserted.
6. Hierarchical/taxonomy items (ix) and beta-binomial counts (viii): not applicable.

## Structural signals

1. Fixed endpoints and 6! enumerable interior: exact latent ordering, no sampling.
2. Midpoint structure from the description (8 frames, L = f0, R = f7): gold in slots {3, 4}; verified by D6 through the learned slot weights; the model enforces it as a prior, not a rule.
3. Monotone similarity decay with temporal gap (Robinson structure): encoded by gap-specific slopes; verified by D5.
4. Per-image independent crop, flip and photometric changes: flip-averaged embeddings, relative-flip max for local matching, reference-normalised channels; LP-FT augmentation mirrors the release perturbations.
5. Shuffled candidate order: exact equivariance of the permutation model; the c-index is never a feature; augmentation by random reshuffle is a no-op by construction.
6. Chain structure (R_i = L_(i+1)): used on train for grouped CV and invariance diagnostics (same-photo pairs), optionally as positive pairs for the invariance auxiliary loss and for long-span rows in roadmap 5a. Never on test.
7. Time reversal (swap L and R and reverse the interior): the gap-potentials are symmetric under reversal, but the label rule might not be (slot 3 vs 4). Check with a train diagnostic: fit with and without swap augmentation and read the tie parameter a; use swap augmentation only if a is close to 0.5 or if the slot weights are mirrored consistently (w_k becomes w_(7-k) in the swapped row).
8. Static or near-duplicate candidates are irreducible ambiguity: measured in D9, not fought.

## Experiment roadmap

1. Contract, metric, validation. Implement the metric and its unit tests; chain recovery and groups (D3); grouped CV with 3 seeds. Stop criterion: unit tests pass; random baseline 0.200 +/- 0.012 on train; groups look sane (component count and sizes reported).
2. Cheapest end-to-end baseline, valid CSV. DINOv2-L CLS cosine betweenness heuristic plus the simplest learned head (a few-parameter listwise softmax over relative features). Write and validate a submission through the full checklist. Stop: valid file; OOF above the D7 floor by more than noise, else debug.
3. Representation and structure. D4 ablations (resolution, pooling, flip averaging, local channel, bank normalisation) judged by the gold metric on paired folds; implement the permutation-marginal model with fixed-distance channels; verify D5/D6; compare with the baseline. Stop: each component kept only if its paired gain exceeds noise; target is the primary's rung-1 model.
4. Metric-aware loss/decode and learned metric. Marginal NLL vs direct expected-score term; low-rank projection with nested ridge grid; slot-weight prior strength. Stop: gains exceed the larger of 1 SE and 0.01 across 12 of 15 fold-fits.
5. Diversity and extra signal. (a) Long-span chain-row augmentation (tie-free gold; deferred, only if step 4 plateaus). (b) Measure on train the cost of NOT using neighbour-burst context (information only, never shipped). (c) Second backbone (SigLIP/CLIP ViT-L) as a second family; score alone first, blend only if within noise of the primary and the blend wins on paired folds; prefer feeding its OOF marginal as a feature over a fixed-weight blend. (d) LP-FT paired check (non-inferiority rule), parameter-norm logging.
6. In-script bounded hyperparameter search: the small ridge/rank/k grids with fixed trial counts, each with its own nested held-out check. No offline pasted constants.
7. Final fixed-plan run: from a clean `working/`, run twice, diff; check the shipped prediction distribution against OOF (top-1 candidate rate by index near uniform; posterior sharpness similar); run the compliance audit; then spend credits in the order baseline, best single, final.

When two independent solvers would plausibly converge on a technique (flip-averaged DINOv2 similarity, betweenness features), test it in step 2-3 before any proxy-driven tweak.

## Compliance audit

CLAUDE.md 7 red items:
- Test file read only for one-row-at-a-time prediction: yes by design (a test row's output = function of its own 8 images, train-fit parameters, train bank, train PCA). No test linking, no test statistics, no test-fitted anything.
- Time in conditions: none (logging only).
- Environment fallbacks (`cuda.is_available`, `cpu_count`, try/except imports): none; fixed device and workers.
- Hard-coded offline-tuned constants: all constants (ridge strengths, k, rank, steps) are chosen by in-script nested CV from a fixed grid; fixed structural choices (slot prior centred on 3 and 4, resolution) come from the description or the in-script ablation; none from watching public-LB submissions.
- External or synthetic data / self-hosted weights / non-allowed libraries: none; HF weights at a pinned revision; libraries torch, transformers, numpy, pandas, scikit-learn, (lightgbm for the fallback).
- Strip-the-ML test: measured (D7); the shipped model's learned components (gap-decay vector, slot weights, projection, last-block fine-tune) are load-bearing if the trained version beats the zero-learning heuristic by more than noise; logged.
- Source readable, under 512 KB, no blobs: yes.
- Model-heavy part dominates: the ordering posterior is learned; no hand-written rule replaces it.
- Challenge-specific: general pretrained weights allowed; no external data; one A10G; about 1 hour; trains a model in-script (probe head trained by exact marginal likelihood plus LP-FT).

Section B self-audits:
- No whole-test aggregation: yes (PCA, banks, scalers all train-only; no rank/z-score across test rows).
- Sibling leakage: the candidate-to-candidate relation is the task's own inputs for one row; no cross-row pooling; flagged as reviewer question 3.
- Every constant derivable by an in-script train-only search: yes.
- Frozen features plus small head: addressed by LP-FT with parameter-norm logging and paired gain; the non-inferiority rule is stated honestly, including the case where it adds nothing.

## Open questions & assumptions

Reviewer questions:
1. Gold definition for an even-length burst: with L = f0 and R = f7 the midpoint 3.5 is a tie between f3 and f4. Is the gold the earlier, the later, a random choice, or the closest in actual (irregular) timestamp? If random, a perfect solver scores about 0.6, which changes what "good" means. Plan A (read as deterministic): the model learns the tie parameter from train and exploits it. Plan B (random/irregular): rank both middle candidates first; expected ceiling 0.6. The same code serves both.
2. Is the within-row use of the other five candidates (seriation) and of both ends acceptable? Assumed yes: they are the row's own inputs. Plan without it (each candidate scored only against L and R) is the fallback's feature subset and is measured on train grouped CV.
3. Does a trained exact-marginal-likelihood head over frozen features plus a last-stage fine-tune (last 2 blocks, 2 epochs) satisfy "train a model in-script"? Assumed yes; the fallback is a fuller fine-tune only if evidence shows it is needed and does not overfit.
4. Exact element format of `ranked_candidates_json` (candidate names vs indices; spacing). Assumed to mirror the train gold label format and the sample submission; must be verified from the files.
5. Do test rows come from the same chains as train rows? Cannot be checked from test content under the data rules; the plan is robust to both and never exploits overlap.
6. Is deriving extra training rows from adjacent train bursts (tie-free long-span rows) acceptable as recombination of real units? Assumed acceptable but deferred and optional.

Assumptions: crops/flips/photometric changes also apply at test time as in train (description); the interior frames are evenly spaced; the photos are in a continuous burst per row; chain links are recoverable by near-duplicate embedding matching.

What I could not verify: everything about the data (all D1-D12), the true tie rule, chain structure and split type, the speed numbers (about 50-60 img/s for ViT-L at 448 px on an A10G and 2.4 min/epoch for LP-FT are estimates), library revisions, and every numeric hypothesis above. Score expectation (an estimate, not a promise): random 0.20; zero-learning betweenness about 0.28-0.40; primary about 0.38-0.55 on private under the tie-free-ceiling reading, with a hard ceiling near 0.60 if the tie is random; I would expect the self-built grouped-CV proxy to read 0.02-0.05 above the private result.
