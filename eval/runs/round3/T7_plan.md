# Eris plan: T7 Allograph witness retrieval

Evidence base: only CLAUDE.md, the strategist procedure and the paraphrased challenge notes. No dataset was available, so every statement about the data is tagged [U] (unverified hypothesis) and every number about runtime is tagged [E] (estimate). Nothing here was measured.

## Contract & decision unit

**One valid answer per board (`case_id`)**
- `witness_pair_program`: `Q0:XY>Q1:XY>Q2:XY>Q3:XY>Q4:XY`. Each X<Y is a letter in A-T. All 10 letters across the 5 pairs are distinct. Exactly 10 of the 20 candidates are therefore left over as distractors.
- `preference_vector`: a JSON list of 5 values in {0,1,2}, e.g. `[2,1,2,0,1]`.
- Invalid versus low-scoring: a repeated letter, X>=Y, a missing query, a letter outside A-T or a non-JSON vector is invalid. I treat any distinctness violation as invalid even if the scorer would only penalise it. A valid but wrong program is merely low-scoring.

**Metric, term by term (exact O/E definitions come from the full description; unverified)**
- Score = 0.65*PairProgram + 0.20*Preference + 0.15*Joint.
- Per query: 0.25*O (letter overlap) + 0.30*P (exact pair) + 0.45*E (whole program exact). I assume P and O are averaged over the 5 queries and E is a per-board binary.
- Preference = 0.25*A (mean entry accuracy) + 0.75*V (whole vector exact).
- Joint = 1 only if the program AND the vector are both exact.
- There is no hierarchical averaging beyond the per-board mean, and the metric is not calibration-sensitive. It is an exact-match metric, so the answer must be a single decoded object.

**What the metric is worth (back-of-envelope, [E])**
- Preference: a per-entry accuracy of about 0.45-0.50 on a 17/41/42 prior gives A of about 0.5. V is about 0.5^5, which is 3% or less. Preference therefore contributes about 0.02-0.03 in total, and the exact gap between a constant vector and a good model is under about 0.02.
- Joint is about 0: it needs the whole vector exact, and the vector is only weakly learnable (see Data findings).
- Therefore about 95% of the score that can be influenced sits in the pair program (0.65 weight). All modelling effort goes to pair retrieval. The preference head is cheap, calibrated and honest.

**Decision unit versus independent unit**
- Decision unit: one board. The 5 queries and 20 candidates are coupled by the distinct-letter constraint.
- Independent unit for validation: the character class and its artifact (source object or source crop). It is not the board and not the query. Source crops recur across hundreds of train boards, and evaluation classes are all unseen. Effective sample size is the number of class/artifact components, not 2,000 boards or 10,000 queries.

**Pipeline stages, to be diagnosed separately**
1. Board parsing: panel/cell geometry and which crop is which letter and which query index.
2. Candidate coverage: oracle recall of both true witnesses inside the per-query top-K pool.
3. Scoring: calibrated query-candidate evidence plus a witness-witness coherence potential.
4. Decoding: the best globally valid board object, given good scores.
5. Preference head, separate.

## Compliance regime

**Domain.** Computer vision, metric learning and retrieval, labelled GPU-oriented fine-tuning. Treat it under CLAUDE.md 6.3, 6.4 and 6.7 as strictly as possible. Frozen embeddings plus a head alone are grey, so a genuine fine-tuning component must be load-bearing (see the LP-FT layer below).

**Explicit bans from the description, and how the plan respects each**

| Ban | Plan |
|---|---|
| Inference from case_id, row order, file path | Never read them. Ids are used only to write the output. |
| JPEG size or compression | Not used. Decode boards to arrays and discard file statistics. |
| Candidate-position frequencies | The model is permutation-equivariant over letters. Candidate order, letters and the label-box region are masked, and slot order is shuffled during training. Letter is never a feature. |
| Thresholds, prototypes, normalisation statistics, clusters, parameters fitted on eval boards | Every constant (temperature, none-logit, pair weight, K, preference calibration) is fitted by an in-script train-only search with held-out folds. Normalisation uses fixed constants and train-derived banks only. |
| Pseudo-labels | None. |
| Propagating character identity across eval cases or an eval-only dictionary | No test-test comparison. No dedup of test crops. Each test board is scored from its own 25 crops plus the trained model, streamed one board at a time. |
| Validation must hold out whole classes AND artifacts | See Validation design. |
| Synthetic data | None. Recombining real labelled train crops is flagged as a reviewer question (Open questions, Q3) and is not required by the primary. |

**Decode.** The distinct-letter assignment is a constraint of the row's own output space, so it is a within-row decision using only the row's inputs plus a train-fit model. Decode constants are train-OOF-fitted. This is the compliant mechanism for the "metric-aware" goal.

**Strip-the-ML test.** Remove the trained backbone blocks and matching head and nothing remains: there are no template, regex or lookup components. The decode is a generic assignment over learned scores. Compliant.

**Reading chosen.** The most conservative reading is a genuine fine-tune (last blocks plus heads) with a few-parameter decode. Its measured cost versus the frozen-only fallback is to be reported (roadmap step 3).

## Data findings

All items below are diagnostics to run on `train` only. Hypotheses are unverified.

**D1. Layout and parsing**
- Run: image size histogram. Median projection profiles of edge and border pixels per axis to find the 5 query panels and the candidate grid (2 rows x 10 columns) of candidate cells. Border colour per query panel. Position of the label tag and label box in each panel. Letterbox (white pad) extents per candidate.
- Hypothesis [U]: the layout is fixed per size (1280x760 in practice, 1700x1100 in the spec), so cell boxes follow from train-median profiles (a deterministic parser, with constants derived in-script from train). Letters are row-major A-J then K-T. Q-index equals left-to-right panel order, with the border colour a function of the index.
- If the order differs from my assumption, the plan changes (see Open questions Q1, Q2).
- Run: check that train labels are consistent with the assumed mapping, using an off-the-shelf frozen-feature nearest-neighbour baseline. A wrong letter/position or Q-index mapping would score at chance, and a right one clearly above it. This is a diagnostic for the mapping, not a modelling input.

**D2. Label structure**
- Run: check the 10 chosen letters are distinct in every train board. Check X<Y always. Count how often two queries in one board resolve to the same class. Count boards where the same crop (near-duplicate) is a witness and also a distractor.
- Hypothesis [U]: the five queries belong to five different classes, each with exactly two witnesses, and distractors belong to other classes (not hidden third witnesses).

**D3. Candidate-position and shortcut audit**
- Run: witness frequency per letter slot (chi-square versus uniform). Per-query-index witness frequency. Then a shortcut classifier on shortcut-only inputs, using only features that are not JPEG-derived: letterbox extents/aspect, border pixel statistics, label-box region, corner tag content. Report AUC for witness-vs-distractor and for query-slot matching.
- Hypothesis [U]: slot frequencies are near uniform, and letterbox/aspect carries a small amount of signal. If any shortcut AUC exceeds about 0.55, the model must be provably blind to it (masking) and the number is logged. If slot frequencies are non-uniform, they are deliberately NOT exploited (banned).

**D4. Recurrence and identity graph (train only)**
- Run: embed all train crops with a frozen backbone. Histogram the nearest-neighbour cosine similarity and choose the near-duplicate threshold from the gap in the histogram (bimodality). Count unique crops U versus 50,000 crop instances (2,000 boards x 25). Cluster-size histogram.
- Hypothesis [U]: "hundreds of thousands of near-duplicate pairs" implies roughly 10 instances per source crop, so U is perhaps 5-10k. This decides the cache size and backbone size (Fixed work plan).
- Run: build the class graph with an edge for each query-witness and witness-witness label link plus identity links. Report the connected-component size histogram. Purity check: two queries of the same board falling into one component indicates an over-merge. The near-duplicate threshold is raised until the conflict count is 0.
- Hypothesis [U]: there are a few hundred class components, possibly with a giant component if the threshold is loose, and boards are chained into one giant board-level component through shared classes (this drives the purged validation below).

**D5. Domain gap and invariances**
- Run: polarity (mean intensity of query vs candidate crops, ink on light vs dark). Stroke thickness. Crop sizes before and after pad removal.
- Hypothesis [U]: queries (rubbing/facsimile) are high-contrast, possibly inverse polarity, versus candidate photo crops. This is a cross-domain match.
- Run: symmetry check. Compare similarity of a query to its true witnesses with the query as-is, flipped horizontally and rotated by a few degrees, using a frozen backbone, on train only.
- Hypothesis [U]: flips REDUCE similarity (characters are not mirror-symmetric), so no flip augmentation or TTA unless the data say otherwise. Small rotation and scale jitter are presumed safe, to be confirmed by the same diagnostic.

**D6. Preference vector**
- Run: marginal distribution per entry and per query index (check 17/41/42). Joint distribution over the 243 vectors versus the product of marginals (entry correlation). Dependence on the answer: with the TRUE witnesses, compute cosine (frozen features) from query to witness 1, query to witness 2 and between witnesses. Compute the sorted difference and fit a small logistic and a small GBDT with grouped CV. Compare the accuracy to the majority-class baseline. Check ordinal structure (0<1<2).
- Hypothesis [U]: the label is a function of a distance on original crops that was lost in rendering. It will be weakly predictable (accuracy about 0.45-0.55 versus a 0.42 majority baseline) and the whole-vector hit rate will stay at a few percent. If it is no better than baseline, ship calibrated per-entry argmax (mostly the majority class) and say so.

**D7. Candidate coverage and information ceiling**
- Run, after the first baseline: per-query top-K recall for K=4,6,8,10, both witnesses inside the pool. Rank of the true witnesses. Fraction of boards where a pointwise argmax is invalid (repeated letter) and what the assignment fixes.
- Hypothesis [U]: top-6 holds both witnesses in at least 95% of queries on seen classes, and less on held-out classes.

## Validation design

**How the test split was made.** All evaluation classes are unseen, and artifacts are held out too. So the test is a class-and-artifact-disjoint split. A random board-level K-fold would leak the heavily recurring source crops and would badly overestimate.

**Grouping (train only)**
1. Crop identity: near-duplicate clusters (D4 threshold).
2. Class/artifact component: union-find over identity links plus label links (query-witness, witness-witness). A second grouping axis is a style cluster (k-means on a frozen style embedding of train crops) as a proxy for artifact if the dataset gives no artifact id. Hold out the union closure of both.
3. Folds: 5 folds over components, balanced by query count, fixed seed. Repeat with 3 different partition seeds. Report mean and std across folds and seeds.

**The giant-board-component problem and its mitigation**
- Because boards share classes, board-level closure is probably one component [U]. Do not split by board. Use PURGED folds instead.
- For fold k, the held-out component set H_k defines the validation. In training, every query and every candidate crop whose component is in H_k is masked out of the loss (queries dropped, columns removed from the softmax). Boards are not discarded, so most data survive.
- The validation is formed of episodes of held-out crops. Per validation board, use the held-out queries and their witnesses, and refill the candidate pool to 20 with crops from H_k (other validation boards' candidates, excluding the same class component), so every validation crop is unseen and the pool size matches the real 20. Report the metric on these episodes and, secondarily, on real validation boards with the fraction of "fully unseen" boards noted. Report how many training boards are partially masked.
- This mirrors the deployment shift (unseen classes, unseen artifacts) on the specific axis, not just leak-free. It over-estimates the real score if the real distractors are harder (look-alikes), so use it for relative comparisons only and plan for the real number to land below it.
- If the purge loses more than about 30% of crops in the largest folds, use fewer folds (3) rather than weakening the hold-out.

**Metric implementation.** Re-implement the exact metric from the full description and unit-test it on: a perfect program and vector (1.0); a program with every pair reversed or wrong (about 0 for P and E); a constant program (e.g. fixed letters); a base-rate vector (all 2s, then the best constant). The preference/joint terms are checked separately.

**Selection hygiene.** Hyperparameter choices (pair weight, K, temperature, none-logit, regularisation per head) are chosen on one set of seeds and re-checked cross-fitted on the other partition seeds. Paired comparison on the same folds. Fixed epochs, no validation-triggered stopping. Accept a change only if it beats the fold-to-fold noise.

## Overfit/underfit risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Memorising recurrent source crops (overfit) | Hundreds of thousands of near-duplicate pairs in train; test classes unseen | Purged class/artifact-disjoint CV; only last blocks plus small heads are trained; tiny LR; 2-3 epochs; fixed schedule; parameter-change norm logged |
| Few independent groups (overfit) | Number of class/artifact components is likely a few hundred [U] | Capacity ladder; heads kept low-rank; per-head regularisation chosen by exact-metric grouped CV, wide grid including the strong end |
| False negatives in contrastive loss | The same class recurs across boards; a cross-board "negative" may share the class | Mask same-component candidates from negatives; never use cross-board negatives without the component mask |
| Hubness / common-background similarity | Rubbings share texture; allographs vary | Reference-normalised score against a bank of TRAIN crops (CSLS-style), averaged over several fixed banks; dual-softmax inside the board |
| Shortcut learning (artifact style, letterbox, border, tag, slot) | The description names position and file/JPEG artefacts | Mask tag/label/border; strip pad; permutation shuffle of slots; shortcut audit D3; artifact-held-out CV |
| Underfit from downsizing | Glyph detail is small; 224 px may lose strokes | Measure 224 vs 336 px on frozen features under grouped CV; keep the larger if it wins |
| Underfit from small backbone | The most common loss versus winners | Prefer the larger frozen backbone (ViT-L) given the cache-split design, if the cache fits |
| Domain gap query-vs-candidate | Rubbing/facsimile versus photo crops | Asymmetric projection heads; polarity/contrast augmentation chosen by D5; train-only |
| Decode constants overfit | Few knobs | Only 3-4 constants, cross-fitted; fall back to plain MAP if the gain is inside noise |
| Optimistic proxy | Refilled pools may be easier than the real distractors | Relative comparisons only; expect the real score below the proxy |

## Recommended approach (primary + fallback)

**Primary: "cache-split LP-FT matcher plus board-level assignment decode".** One lean design, one trained scorer, one decode.

1. **Parse** each board into 5 query crops and 20 candidate crops using geometry derived in-script from train. Mask corner tags, label boxes and borders. Remove letterbox padding (within-sample). Grayscale plus per-crop contrast normalisation (within-sample, no test statistics). Whether to keep aspect as a scalar is decided by grouped CV (default: not used).
2. **Backbone**: DINOv2 (pinned HF revision), ViT-L/14 if the cache fits (U*V*~0.5 MB, about 12 GB or less), otherwise ViT-B/14. Input 224 px, or 336 px if D-level diagnostics say so. Pool richly: CLS plus mean of patch tokens, plus the 16x16 patch grid kept (avg-pooled to 8x8 for local matching). Frozen prefix, last 2 blocks trainable (LP-FT).
3. **Cache-split**: run the frozen prefix once per unique train crop for V=3 fixed views (1 clean, 2 augmented: scale 0.85-1.0, rotation up to about 8 degrees, contrast/polarity jitter; no flip unless D5 says it is safe), store the block-(N-2) tokens in fp16. Training then touches only the last 2 blocks plus heads, which makes it cheap. Test crops are pushed through the whole network once, per board, with no augmentation (one clean view).
4. **Heads** (few parameters, stated): separate query and candidate projections (e.g. 1024 to 256, low rank), one learned temperature, one learned "none" bias for distractors, one local-correspondence weight, and one witness-witness coherence bilinear (low rank). That is the entire learned decode vocabulary besides the backbone tail.
5. **Score** s(q,c) = scaled cosine of projected CLS/mean-patch plus a local Chamfer-style patch-correspondence term, minus a reference-bank normaliser from TRAIN crops (CSLS-style). Evidence is made board-relative by a dual-softmax: row-wise over 20 candidates, column-wise over 5 queries plus "none".
6. **Training loss** (exact structure, no latent mixture needed because train labels are observed): (a) per-query softmax over the 190 candidate pairs with the true pair as target, pair score = s(q,a) + s(q,b) + lambda*g(a,b), which trains the exact-pair quantity P directly; (b) per-candidate 6-way softmax (5 queries + none), which trains exclusivity and distractor rejection; (c) an auxiliary supervised-contrastive loss over crops with component labels, used only if D4 purity holds (zero conflicts). Randomly permute the slot order of candidates and queries each step.
7. **Decode**: exact within-board assignment (see Metric-aware training & decode).
8. **Preference head**: a separate small model on query-level features computed from the chosen pair (similarities, margins, column-softmax probability, none logit), grouped-CV regularised, calibrated per entry. Whether it uses ordinal heads is decided by D6.

**Fallback (fits if LP-FT does not beat the probe under purged CV).** Frozen DINOv2-L pooled features (CLS plus mean patch), a regularised low-rank bilinear similarity head trained with the same row/column/pair losses, the same decode and preference head. Compliance is greyer (frozen features), so the parameter-change evidence is not available; this is only the safety net for a measured failure of the primary.

**Third candidate (diversity of assumption, roadmap step 5 only).** A second backbone family (CLIP ViT-L/14 or supervised ConvNeXt) with the same head, blended via within-board z-scored scores, only if its stand-alone purged CV is close to the primary.

## Rejected options

- Template matching, SSIM, HOG/contour nearest neighbours or TF-IDF-like retrieval as the answer: rule-based, fails the strip-the-ML test.
- Classifying character classes: all evaluation classes are unseen, so it is open-set metric learning.
- Full fine-tune of the backbone: the recurrent near-duplicate crops would be memorised; cost is far higher; the ladder says stop at LP-FT.
- Training from scratch or a hand-built Siamese CNN: too little independent data; weaker than a pretrained backbone.
- OCR or VLM or zero-shot LLM reading of characters: inference-only and LLM-based; not allowed.
- Test-time clustering, test-test similarity, an eval-only character dictionary, cross-case identity propagation, test-set dedup: explicitly banned.
- Position, letter or slot priors, JPEG or layout statistics: banned.
- Random board-level CV: leaks recurring source crops.
- End-to-end board transformer predicting the program directly: heavy, memorises, harder to validate; the primary-design gate says no.
- Joint-sequence or sampling-based board decoders (beam over assignments with stochastic components): only after the lean decode is measured and shows a gain above noise.
- GBDT on hand features as the core scorer: not enough genuine learning here; allowed only inside the preference head.

## Fixed work plan & runtime budget

All counts are fixed constants: seeds, folds, epochs, views, K, batch size, workers. No wall-clock branching, no `cuda.is_available` or environment switches, device fixed to cuda. Estimates are for an A10G and are labelled [E]; the real per-stage numbers must come from a short profile before hardcoding.

| Stage | Fixed plan | Time [E] | Memory [E] |
|---|---|---|---|
| Parse and crop 2,500 boards (train 2,000 + test 500) | Decode, mask, strip pad, resize; one fixed pass | about 2 min | uint8 gray crops at 224 px about 3 GB |
| Identity graph and groups | Frozen embeddings, kNN, union-find, 5 folds x 3 partition seeds fixed | about 2-3 min (includes one frozen forward of unique crops) | embeddings for U crops |
| Cache prefix tokens | V=3 views of U unique train crops, ViT-L prefix | about 3-6 min if U about 5-10k | 12-16 GB fp16 on disk/RAM; if U x V x 0.5 MB exceeds about 14 GB use ViT-B (0.4 MB/crop) or V=2 |
| In-script purged CV for decode and preference calibration | 4 folds x 3 epochs, tail blocks plus heads, batch of 8 boards | about 12-16 min | GPU about 8-12 GB |
| Final fit on 100% of train | 3 epochs, same recipe | about 4 min | same |
| Test inference | 500 boards x 25 crops, one clean view, full forward, streamed | about 1-2 min | small |
| Decode plus preference plus validation plus write | Exact assignment per board, validator, re-read of the CSV | under 1 min | small |
| Total | | about 25-35 min | |

- Headroom: about 30% or more against a 45-50 min target [E]. If measured time overshoots, reduce V, folds or ViT size as a fixed constant in the script (decided in development, not at runtime).
- Determinism: seed everything, deterministic kernels with `warn_only`, fixed DataLoader generators, fixed workers, fixed thread counts, pinned pretrained revision, `n_trials` not used (no HPO beyond a fixed small grid on the head regularisation and decode constants).
- Output validation inside the script: row count, ids, program regex, letters in A-T, X<Y, 10 distinct letters, vector length 5 with values in {0,1,2}, JSON-parsable, re-read of the file with `keep_default_na=False`. Raise before writing a broken file.
- Fallback early write: write a valid but safe submission (valid program, base-rate vector) before the heavy stages, only as a loudly logged safety net, since a constant answer is not banned by the notes shown. Check against the full description before relying on this.

## Metric-aware training & decode

**Back-solving the metric**
- E (weight 0.45) is the largest single term and rewards the exact board-level assignment. P (0.30) rewards per-query exact pairs and O (0.25) rewards letters. The training loss (a) targets the exact per-query pair directly, which aligns with P, and the assignment decode targets E.

**Decode (first-order, lean)**
1. Per query, keep the top-K candidates by calibrated score, where K is fixed from the D7 oracle recall (target at least 99% pool recall on purged validation). Likely K around 7-8.
2. Enumerate legal pairs from the pool: for each query the pair score is s(q,a) + s(q,b) + lambda*g(a,b), each term log-probability-like after the dual-softmax and none-bias. Include the column "none" evidence: a candidate used by one query pays its competing-query evidence.
3. Find the maximum-score assignment of disjoint pairs across the 5 queries by an exact, deterministic search over the pools (bitmask or depth-first with fixed pruning), with a fixed tie-break by letter order. As an exact cross-check on a sample, compare with a Hungarian relaxation on pointwise scores (10 slots x 20 candidates) to measure how much the pair term and the pool restriction add.
4. Sort letters X<Y inside each pair and emit Q0..Q4 in order.
5. Optional second constant, only if cross-fitted OOF shows a gain above noise: blend MAP with per-letter marginals to favour O (weight 0.25). Default is plain MAP.

**Calibration**
- Fit the temperature and the none-bias on OOF (purged folds), cross-fitted. Report the OOF decode accuracy before and after. Use fixed constants thereafter. No test-set statistic enters any constant.

**Preference decode**
- Utility of a vector v = 0.01*sum_i marginal_i(v_i) + (0.15 + 0.15*P(program exact))*P(V=v), enumerated over the 243 vectors. With P(V) of order 1e-2 the first term dominates, so this is effectively per-entry argmax with a joint tie-break when entries are correlated (D6). Calibrate the per-entry probabilities with a tiny logistic recalibration on OOF, cross-fitted. If D6 shows no signal beyond the prior, ship the calibrated majority entries and state the cost.
- Ordinal targets: if D6 shows ordering, use cumulative at-least-k heads.

## Structural signals

- Exactly 2 witnesses per query, 5 distinct classes per board, 10 distinct letters, 10 distractors: hard decode constraints, and the distractor/"none" column in the loss.
- Witness-witness coherence: the two witnesses of a class are allographs, so their mutual similarity is a learned potential g(a,b), not assumed strong (its weight lambda is measured; it may be near zero).
- Permutation symmetry: slot order of candidates and queries is arbitrary. Randomly permute both every step, which also removes the letter/position shortcut. Assert that predictions are equivariant on a train sample.
- Recurrence across boards: the identity graph yields many additional real same-class positives per class (allograph variants). Use them only as the auxiliary supervised-contrastive loss under the purity check (a recombination of real labelled units), and for hard-negative masking.
- Symmetry augmentations are chosen from data: measure the flip effect (D5) before using any. Rotation/scale/contrast jitter applied identically within the sample pipeline. Queries and candidates may need different polarity treatment (D5).
- Reference-normalised evidence: score what a query and a candidate share relative to a bank of unrelated TRAIN crops, averaged over several fixed banks, plus local patch correspondence. Never a bank built from evaluation boards.
- Causal alignment: the label depends on the query and its own witnesses; training samples are defined per query-within-board, and sibling features (witness-witness) are used as the pair potential only.
- Preference: probably a function of distances between the query and its witnesses on the original crops [U]; give the head the learned-embedding analogues of those distances plus the ordering by letter (X<Y) so the head can learn any asymmetric structure.

## Experiment roadmap

| Step | Action | Stop / gate |
|---|---|---|
| 1 | Parser, mapping checks (letters, Q order), metric implementation and unit tests, identity graph and purged folds | Parsed boards visually and statistically consistent; metric tests pass (perfect 1, reversed/constant near 0); purge loss and component sizes reported |
| 2 | Strongest cheap baseline end-to-end and valid: frozen DINOv2 features, cosine score, pointwise argmax then Hungarian assignment; valid CSV and the shortcut audit | Valid file; first purged-CV number; pool recall by K |
| 3 | Representation and structure: backbone size and resolution on frozen features, richer pooling and local correspondence, reference-bank normaliser, dual-softmax. Then the LP-FT rung (last blocks) versus probe | Keep a rung only if it beats the previous under paired grouped CV above noise; log the parameter-change norm and its CV gain over the probe |
| 4 | Metric-aware training and decode: pair-softmax loss, none column, coherence potential, exact assignment, calibration; preference head | Each component must show a paired gain over noise; MAP versus marginal blend decided here |
| 5 | Diversity: second backbone family with z-scored within-board blend; supervised-contrastive auxiliary if purity holds | Only if stand-alone close to primary and blend beats noise |
| 6 | Bounded in-script search: fixed small grid on head regularisation and decode constants, with its own cross-fitted held-out check | No more than a handful of knobs; fixed counts |
| 7 | Final fixed-plan run twice from a clean working directory; diff the outputs; runtime profile on the A10G | Stable output; at least 30% headroom; validator passes |

Use submissions sparingly: baseline, best single model, final. When two independent solvers would plausibly converge on a technique (assignment decode, reference normalisation), test it before any proxy-driven tweak.

## Compliance audit

CLAUDE.md section 7 checks and the section B self-audits against this plan.

- Test file read only for per-board prediction? Yes: each board is scored from its own crops and a train-fit model, streamed. No test-set statistics, no dedup, no normalisation across the test file. Pass.
- No time-dependent branching? Pass by design (all counts fixed; time only logged).
- No environment-dependent fallbacks? Pass (device fixed; one code path; ViT-L vs ViT-B and V are fixed constants chosen in development).
- Hard-coded tuned constants? Every constant is an in-script train-only search with held-out checks; pretrained revision pinned; layout constants derived from train in-script. Pass, provided the layout derivation is in the script and not pasted. Flag in review.
- External data, synthetic data, own hosted weights? None. Only pretrained backbone weights from HF/timm. Recombined real crops used only as auxiliary or validation episodes (Open questions, Q3).
- Strip-the-ML: the backbone tail, projection heads and potentials are the model; removing them leaves nothing that works. Pass.
- Whole-test aggregation: none. The dual-softmax and board-relative normalisation use only the row's own 5 x 20 scores. The reference bank is built from train crops. Pass.
- Sibling leakage: the witness-witness potential is computed among a row's own candidates, and the target is a relation between mates, so this is flagged. Mitigation: it is part of the pair potential trained on the same labels (not a precomputed leak), and its measured gain is reported; drop it if inside noise.
- Position, letter, file, JPEG shortcuts: masked and audited (D3). Pass pending audit.
- Label-derived statistics: the identity graph and any train bank are train-derived. If they feed the loss (negative masking, supervised-contrastive), they are computed within the training part of each fold only (purged), with the same amount of data behind train and test time (the test scores use only the final model, no label statistics). Pass.
- Genuine training: LP-FT tail plus heads, with logged parameter-change norm and CV gain over the probe. Pass pending measurement.
- Source readable, below 512 KB, no blobs: yes by design.

## Open questions & assumptions

Reviewer questions (with the plan under each reading):
1. **Q-index mapping.** Is Q0..Q4 the left-to-right panel order, or is it given by the label tag or border colour? Reading A (positional): use position. Reading B (tag/colour): read the colour (a deterministic mapping learned from train, used only to identify the index, never as a matching feature). The D1 consistency test decides.
2. **Letter mapping.** Are letters A-T row-major over the 2x10 grid, or printed in the corner label box in shuffled order? Reading A: position gives the letter. Reading B: a tiny classifier trained on label-box crops (labels from train), only for letter identification.
3. **Is recombining real train crops into new episodes** (validation episodes and the auxiliary contrastive loss) acceptable as non-synthetic? The primary does not need the training use; validation use is internal.
4. **Is a train-built reference bank used as a score normaliser** compatible with "no prototypes/normalisation statistics fitted on evaluation boards"? It is train-only and used for hubness correction; if disallowed, replace it with within-board dual-softmax only and report the cost.
5. **Is the query-slot index allowed as a preference feature** (distinct from the banned candidate position)? Default: not used.
6. **Does a cache-split last-blocks fine-tune count as the "GPU-oriented fine-tuning"?** It trains real weights inside the script, with logged parameter change.
7. **Can distractors be hidden extra witnesses** (same class as a query)? If D2 shows they can, the loss uses component masks and the decode must not treat all non-chosen letters as strict negatives.
8. **Layout mismatch**: spec 1700x1100 versus 1280x760 actual. The parser uses train-derived layout. If test boards differ in size or layout, the parser must be re-derived from train only, since fitting layout on test is banned. Unknown.

Assumptions (all unverified): each board has 5 distinct classes, 2 witnesses each, and distractors are of other classes; X<Y holds always; the preference label is only weakly predictable; recurrence gives a few hundred class components; unique crops U is roughly 5-10k.

**Expected score (estimate, not a promise):** about 0.30 to 0.55 overall, central guess about 0.40, wide uncertainty. Reasoning: with per-query exact-pair accuracy p of 0.6, 0.8 and 0.9, the pair program is worth about 0.28, 0.42 and 0.51 once the 0.65 weight is applied, plus about 0.025-0.03 from preference and almost nothing from joint. The proxy validation will overestimate this, and the cross-domain rubbing-vs-photo gap is the main unknown. The preference head can move the total by under about 0.02.

**What could not be verified:** every data fact, the layout and mapping, the exact definitions of O and E, the near-duplicate structure and unique-crop count, the preference rule's learnability, the train/test shift beyond what the description states, and all runtime numbers. No code was written and no data were opened.
