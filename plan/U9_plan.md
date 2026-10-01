# U9 Quartet Lock: Eris build plan

Status: written WITHOUT dataset access (no files available to this run). Everything under "Data findings" is a diagnostic to run on TRAIN and a hypothesis, marked UNVERIFIED. Sources read: eris-strategist.md, CLAUDE.md, tasks/user/U9_quartet_lock.md only. No solution code here.

## Contract & decision unit

**One valid answer.** For each evaluation set (row), a JSON array holding 0,1,2,3 exactly once, most likely true record first. Positions index that row's own `candidates` list (the four (year, day_of_year, latitude, longitude) tuples shared, in the same order, by the four sets of a batch). Columns exactly `set_id`, `ranking`. Every row of `sample_submission.csv` must be present (non-evaluated ids still need a valid ranking). Invalid = wrong columns, blank/duplicate id, missing evaluated id, any ranking not a permutation of 0..3; validity is judged over the whole file, so a single bad row zeroes everything. Low-scoring but valid: any permutation.

**Metric, term by term.** S = 0.6*A + 0.4*R, where A = mean[ranking[0]==t] and R = mean 1/(1+pos). Per-row additive, every set weighted equally (quartets are equal-sized, so no hierarchical re-weighting). Random ordering: A=0.25, R=(1+1/2+1/3+1/4)/4=0.5208, S=0.3583 (matches the stated ~0.358). A perfect ranking is 1.0; a fully reversed one (truth last) is 0.4*0.25=0.10. A confident wrong first choice costs 0.6 immediately and R loses at most 0.75*0.4. Rank-only: no calibration is scored, but calibrated joint probabilities are what make the ranking optimal (see decode).

**Coupling.** Each batch hides one perfect matching (4 sets <-> 4 records). First choices that form a permutation are right for 0, 1, 2 or 4 sets, never 3. Because S is row-additive, the expected score of a row depends only on that row's marginal P(set i owns record j). Optimal output per row = sort candidates by the exact marginal under the model's posterior over the 24 matchings (proof: expected R is linear in positions with decreasing weights; expected A is the top marginal). No joint "assignment" object has to be output; the matching enters only through the marginals.

**Independent units.**
- Unit of prediction: the SET (3 unordered photographs).
- Unit of decoding: the BATCH (4 sets x 4 records, 24 matchings). Four rows = one puzzle.
- Unit of generalisation (what decides the private score): the COLLECTOR ("every evaluation set was gathered by a person who never appears in the training rows"). The number of independent collectors in train is unknown and is probably far below the number of sets. The effective sample size is that count, not rows.

**Pipeline stages (diagnose separately).**
1. Candidate coverage: trivial, always 4/4 contain the truth (oracle recall = 1 by construction). Verified in train by the diagnostics below.
2. Scoring: a learned compatibility potential s(set, record). Quality is measured by pairwise discrimination between the true and each false record of the same batch, split by the axis that separates them (date, place, year).
3. Decoding: exact matching posterior -> marginals -> sort. Measured with gold scores (must give S=1.0) and with the learned scores.

## Compliance regime

**Domain.** Cross-modal (CV x tabular numeric records). The description gives no category label (CV vs Fine-tuning vs from-scratch is silent). Assumed: CV playbook (CLAUDE.md 6.3) with a genuine trained component; fine-tuning-label strictness (6.7) is covered by the LP-FT step (see approach).

**Explicit bans in the description (hard constraints).**
1. Train genuinely inside the submitted script; a pipeline that still works with the learned model removed is invalid (strip-the-ML test).
2. No external datasets; do not manufacture training data.
3. Test set: inference only, ONE ROW or ONE PUBLISHED BATCH at a time. No pseudo-labelling, no test-time adaptation, no statistic computed across the whole evaluation set. Other rows of the SAME published batch are explicitly allowed; nothing outside the batch may be pooled.
4. Identifiers, row order and file metadata are not predictive signal.
5. Hand-labelling evaluation rows is prohibited.
6. CLAUDE.md bans that still apply: fit every scaler/encoder/vocabulary on train only; no wall-clock branching; no environment fallbacks; seeded; HF/timm weights only; < 512 KB plain source.

**What is allowed and used.** Batch-local joint decoding (matching marginals over the 24 permutations of one batch), batch-local relative record features (e.g. a candidate's day offset from the batch median), the three photos of a set pooled together, hand-measured physical size of the cutout (deterministic mask statistics fed to the trained model).

**Mechanism ban != goal ban.** "No statistic across the whole evaluation set" does not ban per-batch coupling; the description explicitly permits it. The batch posterior needs a single scalar temperature; it is fitted on TRAIN out-of-fold predictions only and shipped as a constant computed in-script, never estimated from test.

**Grey items and my position.**
- Frozen pretrained backbone + trained heads: allowed for CV; grey if the platform labels this "Fine-tuning". Compliance layer: LP-FT (last-stage fine-tune, tiny LR, 1-2 epochs) with parameter-change norm logged (see approach). Cost stated there.
- Re-pairing training sets with records from OTHER training batches to get extra negatives: this is standard in-batch/cross-batch contrastive negatives (no new labelled record is created) but the words "do not manufacture training data" make it a reviewer question. The primary design does NOT need it (quartet-only training + auxiliary regression heads). Gated, off by default.
- Domain-matched backbone via `open_clip` (BioCLIP-type): open_clip is not in the allowed-library list; ask a reviewer; primary uses a transformers/timm backbone.

**Strip-the-ML test on the plan.** Remove the learned compatibility model, aux heads and set encoder: what remains is a uniform prior and the decoder, which scores exactly random (0.358). The decoder cannot hide a rule solver because it contains no information about insect-record associations. Hand-measured size features are inputs, not rules; no threshold such as "bigger -> north" is coded.

## Data findings

ALL UNVERIFIED (no data available). Each line: diagnostic to run on TRAIN, then the expected result and why. The test files are used only for schema, row count and id format.

**D1. Schema and structure.** List train columns and dtypes; check that train has `batch_id`, `candidates`, a true-position or true-record label, and image reference(s) for 3 photos. Expect: batches of exactly 4 sets and 4 distinct records, a label giving the owner position, and `candidates` as a JSON string or list. Assert in train that within a batch all 4 rows have identical candidate lists. If train lacks the label as "position", derive it; if train has no batch structure, the whole matching design changes (flag as assumption failure; see Open questions).

**D2. Label geometry.** True position histogram (expect uniform, i.e. no positional prior; ids/order must not be used). Count distinct (year, doy, lat, lon) records versus sets: does the same record recur across sets/batches (same collection event, several sets)? Expect: some repeats if one event is split into several sets; this would define natural groups and could supply positive same-event pairs for an auxiliary metric loss.

**D3. Record ranges.** Min/max/missingness of year (spread; the batch constraint says 15 years max within a batch), day_of_year (1..366, check leap-day handling), latitude, longitude (degrees, sign conventions, any 0/NaN fills). Expect: clustered collection sites (few distinct localities), seasonal concentration in warm months, wide year range. Compute the within-batch spread distribution on each axis (confirm the 30-day / 15-year / 25-km / 10-year constraints hold: this validates my reading of the generator without hard-coding it).

**D4. Image canvas and scale.** Image size distribution (expect one fixed canvas size), the grey background value (expect a constant), foreground mask via |pixel - grey| > tolerance; bbox, area, major/minor axis (PCA of mask) in pixels. Expect: insects small relative to the canvas, wide spread of sizes, one insect per image. The canvas fraction occupied by the 99.9th-percentile bbox decides a FIXED scale-preserving crop/resize constant (derived from train, then hard-coded as a documented constant and re-derived in-script on train).

**D5. Metadata leakage audit (never used as a feature).** Do file name, file size, EXIF/JPEG quantization, decode time or image mode correlate with the true record's axes or with the collector cluster? Expect none or irrelevant; the audit exists so I strip metadata by decoding pixels only and so the description's "no signal" claim is not silently violated.

**D6. Duplicates and shared photographs.** Perceptual hash / feature cosine to find duplicate or near-duplicate photographs across sets. Expect few; any found are union-find edges for groups.

**D7. Collector structure (derived, since no person id is promised).** Build a graph on sets: edges for (a) identical record tuple, (b) the same locality within X km AND same year (X from the train distribution of the within-event gaps, not tuned on test), (c) near-duplicate photographs, (d) very high mean photo-feature cosine between sets with the same locality (a camera/background fingerprint). Union-find components = proxy collectors. Report component size distribution and the largest component share. Expect: a heavy-tailed size distribution; if one giant component spans > 30 percent of sets, the spatial edge is too loose; tighten and report.

**D8. Axis learnability (the key "what do the photos reveal" diagnostic).** Frozen DINOv2-class features of the set (mean of 3 photos, CLS + mean patch tokens), ridge heads, GroupKFold by D7 components, targets: sin/cos(doy), year, lat, lon (unit vector). Report grouped R^2 per axis. Expect (hypothesis): place and season are weakly to moderately readable (species ranges and phenology, plus size as a physical cue), year is hardest and may be near zero unless old specimens differ in condition or the species mix drifted. Whatever the result, it sets the realistic ceiling.

**D9. Pairwise discrimination by separating axis.** Using the D8 predictions as a Gaussian density (OOF), for every within-batch pair (true, false) of a set, compute AUC and split by which constraint made the pair distinct: >25 km at similar year, >10 years at similar place, doy gap bucket (0-5, 5-15, 15-30 days). Expect AUC near 0.5 for tiny gaps and above 0.5 for larger gaps on whichever axis carries signal. Tells which axes to invest in.

**D10. Size-only floor.** Train a tiny ridge/GBDT on mask-area/length/width statistics alone (grouped CV) and score the full S through the matching decoder. Expect a small but above-random S (physical size is the one explicitly flagged "true physical measurement"); this is the honest floor of the cheapest learned signal and the first ablation of the strip-the-ML test.

**D11. Frozen-probe floor under the same folds.** Per-set Gaussian-density probe (linear head on frozen features) through the same matching decode. Expect modest values (S roughly 0.40-0.50?, a guess). This is the yardstick any fine-tune must beat.

**D12. Ambiguity ceiling.** For train quartets, S of the oracle that knows ONLY one record axis (e.g. exact doy of the true record but nothing else, then argmax on that axis); not a model, a ceiling computation per axis. Expect: no single axis saturates; the batch generator is built so that coarse cues barely separate records.

**D13. Gold-score decoder check.** Feed one-hot gold potentials through the decoder -> S must equal 1.0 on every train quartet; feed uniform -> must equal the random S of the train label distribution; measure decoder runtime per batch (expect microseconds with 24 permutations).

**D14. Overfit tell.** Memorisation test: with random group labels versus derived collector groups, how much does held-out S differ with the same model? A large gap = collector fingerprints are exploitable (expected, since persons own camera/lighting quirks).

## Validation design

**How test was (likely) made.** Evaluation batches contain only collectors absent from training ("a person who never appears in the training rows"); each batch is four sets with four mutually confusable records. So the split is GROUPED BY COLLECTOR, with batch intact. Whether the test collectors share localities/years with train collectors is not stated, and I will not inspect test distributions to find out; the plan hedges with two schemes.

**Scheme V1 (primary): derived-collector grouped CV.** Groups = D7 components. Folds (5 folds, 3 split seeds; report mean and std over folds and over seeds): assign GROUPS to folds. A quartet is a validation quartet only if all four of its sets lie in the held-out groups; training uses only sets whose groups are in the training folds, and a training quartet is used only if all four sets are in training groups (otherwise its rows are dropped from both roles, which also keeps decode-time scores for validation honest). If the batch graph plus collector graph percolate into one giant component, break the batch edges (never the collector edges) and report how many quartets are dropped; fall back to V2 on the same counts. Bias direction: mildly OPTIMISTIC if derived groups under-merge true collectors (camera/background fingerprints still help the model on held-out sets of the same person, though D7(d) tries to merge them); mildly PESSIMISTIC from the lost training volume. Net: expect the private S to land at or somewhat below the V1 mean.

**Scheme V2 (pessimistic stress test): spatial-block grouped CV.** Group by coarse lat/lon blocks (a size chosen from the train locality spread, e.g. blocks containing the same number of sets) so place-specific species mixes are held out. Bias: PESSIMISTIC, since the real held-out collectors may well collect at already-seen places. Use for relative ranking of robustness, never as the headline number. The true private S should lie between V2 and V1.

**Metric re-implementation.** Exactly S = 0.6A + 0.4R with t(i) taken from the train label. Unit tests: perfect rankings = 1.0; reversed (truth last) = 0.10; constant ordering [0,1,2,3] gives the empirical fraction pattern of the train positions (≈0.358 if positions are uniform); uniform-random orderings over many draws ≈ 0.358; the coupled-row property (first choices forming a permutation -> #correct in {0,1,2,4}, never 3) checked by brute force over all 24 permutations x all 24 truths.

**Nested selection discipline.** Every post-hoc choice (head hyperparameters, aux weight, blend weights between families, decode temperature, LP-FT on/off, resolution) gets its own check: choose on inner folds of the training part of each outer fold, evaluate on the outer fold; or at minimum cross-fit temperature/blend weights (fit on the OOF of the other folds). Compare designs PAIRED on the same folds and the same split seeds. Accept a change only when it beats noise (more than 1 SE of paired fold differences, or consistent on most folds and all seeds). Fixed stop epochs (constants), not validation-triggered stopping.

**Final readout.** Because few collectors may exist, expect fold-to-fold std to be large; report both fold std and seed std. A self-built proxy usually over-estimates; use it for relative comparisons.

## Overfit/underfit risks

**Overfit (with mitigation).**
1. Collector fingerprint shortcut (camera, lighting, background halo of the cutout, specimen preparation) -> memorising the collector's usual records. Highest risk: held-out collectors are new. Mitigation: collector-grouped CV (V1) and spatial blocks (V2); mild photometric augmentation (brightness/contrast/gamma, small, label-agnostic); strongly regularised small head; compare a grayscale-input variant under V1; drop-one-photo augmentation; tiny capacity; frozen backbone by default. Optional later step only if measured: adversarial invariance to derived collector clusters (gradient reversal), not in the primary.
2. Few effective groups -> noisy model selection. Mitigation: wide-but-short fixed grid (<= 8 configs), repeated splits, paired comparison, prefer plateaus.
3. Tuned knobs: temperature, blend weights, aux weight, epochs. Mitigation: low-dimensional, cross-fitted, constants computed in-script from train OOF.
4. Record priors / location popularity bleeding into scoring. Mitigation: the matching likelihood cancels any record-only or set-only term (permutation invariance), so the model is trained only on the interaction; a per-row softmax would not cancel them, which is one reason the matching loss is the primary loss.
5. Memorising specific quartet pairings. Mitigation: random permutation of the three photos, random record-list order within a quartet, early fixed epoch count chosen from the train-vs-held-out matching NLL curve (log each epoch; fix the constant).

**Underfit (with mitigation).**
1. Down-sizing the canvas destroys fine morphology and the physical scale cue. Mitigation: scale-preserving FIXED crop and resize constants (D4), never per-image tight-crop-and-rescale; explicit mask-based size descriptors from the original pixels (area, major/minor axis, aspect, solidity, log transforms) concatenated to the embedding so scale cannot be lost by scale-invariant backbones.
2. Small/weak representation: start from the strongest backbone that fits the budget (DINOv2-L class) rather than a base model; richer pooling (CLS + mean patch tokens, flip-averaged, probably the layer before the last as well).
3. Fine-grained difference needed within a 30-day window: absolute Fourier encoding of day-of-year at low frequency cannot resolve it. Mitigation: also feed within-batch-relative record features (offset from batch median in days, years, km), plus high-frequency harmonics.
4. Weak loss mismatch: use the exact matching likelihood, not a plain row classification.
5. Signal may simply be weak (D8/D9): then a good decode and calibration are most of the available gain; do not add capacity to chase noise.

## Recommended approach (primary + fallback)

**Primary: "Frozen strong vision features + physical-size descriptors -> permutation-invariant set encoder -> learned compatibility potentials with the record encoder, trained by exact matching likelihood, decoded by exact batch marginals".**

*Representation of a set.*
- Per photo: frozen DINOv2-Large-class backbone (HF `facebook/dinov2-large` or similar, revision pinned), fixed scale-preserving crop/resize to a fixed resolution (constant derived from D4), features = CLS + mean of patch tokens, averaged over fixed flips (horizontal, vertical). Native physical size is carried by the explicit mask statistics (see underfit item 1).
- Per set: DeepSets/one-head attention over the three photo embeddings (permutation invariant) plus mean/max pooling; size descriptors (sorted by size so the order is canonical, plus set mean/min/max/std). Output h_S, d = 64..256 (grid).
- Training augmentation (physics-preserving): flips, 90-degree rotations, small brightness/contrast jitter; NO scale jitter, no random-resized crop (destroys the physical scale), drop-one-photo with small probability.

*Representation of a record.* Circular day-of-year (sin/cos with several harmonics, k=1..8), year (standardised plus several low-frequency terms), 3-D unit vector of lat/lon plus a few harmonics, products of season and latitude via a small MLP, and batch-relative features (offset from the batch median in days, years, east/north km, scaled by the batch spread). MLP -> e_r.

*Compatibility.* s(S, r) = sum_k phi_k(h_S) psi_k(e_r) with a few potentials (K ~ 8..32), plus a small interaction MLP on [h_S, e_r, h_S*e_r] (grid-selected). Learned parameters few; stated so the strip-the-ML argument is clean.

*Auxiliary supervision from every set (no negatives required).* From h_S predict the set's own record axes (sin/cos doy via binned soft classification with circular label smoothing, lat/lon cell classification, year bins, plus a Gaussian regression head). These provide gradient signal for each training set using only its true record; they also give family 2 below. Loss weight is a grid constant.

*Loss.* Exact matching likelihood over 24 permutations: -log [exp(sum_i s(i, pi*(i))/tau) / sum_pi exp(sum_i s(i, pi(i))/tau)] per training quartet, plus a small weight on row and column softmax cross-entropies (stabilises early learning). Quartets from the published training batches only. Weight decay, dropout on embeddings, EMA of head weights, fixed epochs.

*Family 2 inside the same script (diversity of assumption): predict-then-score.* Score a candidate by the sum over axes of the aux-head log-probabilities for the candidate's value (record-only terms cancel in the matching, so no density correction is needed). This assumes conditional independence of axes given the photos; the bilinear family does not. Combine log-potentials s = a*s_compat + b*s_density with (a, b) fitted on OOF, cross-fitted; keep the blend only if it beats the compat model alone by more than noise.

*LP-FT compliance layer.* Warm-start the head from the frozen-probe solution, then fine-tune only the last block (or the last 1-2 blocks) of the backbone for a fixed 1-2 epochs at a tiny LR (e.g. 1e-5, grid-selected from a very small list), log the parameter-change norm and its grouped-CV gain over the probe. Decision rule (in-script, data-determined, no clock): ship the LP-FT model unless it LOSES to the frozen model by more than one SE under V1; if it ties, ship LP-FT because genuine training in the backbone answers the "train genuinely" compliance question under every reading. State the cost: about +8-12 min of GPU time under the nominal sizing below, and a possibly small score loss if collectors are memorised.

*Decoding.* Exact marginals over the 24 matchings, calibrated temperature tau, row ranking = descending marginal (score as tie-breaker for strictness). Detailed below.

**Fallback (lowest rung of the capacity ladder, cheapest, still genuinely trained).** Frozen DINOv2 features + mask size descriptors -> ridge (strong end of a wide grid, selected per output by exact S under V1) to predict sin/cos(doy), year, lat/lon-vector; Gaussian predictive density with OOF residual variance per axis; candidate score = sum of per-axis log-densities; same matching decode. This is what ships if the primary fails a gate; it also is the roadmap's step-2 baseline.

**Candidate 3 (diversity, gated): train-reference kernel neighbours.** For a candidate record r, find training sets whose records are close to r in a space-time kernel and score the query set by its feature similarity to those neighbours' features, normalised against a bank of unrelated training sets (likelihood ratio against TRAIN references only; never the test set). Different assumption (nonparametric), but leave-own-collector-out is mandatory (otherwise it memorises), and it is lookup-like, so it enters only if it earns a measured gain over noise and as a minor member; strip-the-ML risk noted.

**Expected PRIVATE score (estimate with no data, labelled as a guess).** Anywhere from ~0.40 to ~0.55; a central guess near 0.45. Reasoning: with decode and calibration but weak per-pair signal the top choice accuracy could be 0.30-0.45 and the reciprocal rank 0.55-0.70 (e.g. A=0.40 with R=0.64 gives S=0.50). If the photos reveal place/season strongly, higher; if they mostly reveal collector fingerprints, the private score will fall to near random. The real number should land below V1.

## Rejected options

- **Full fine-tune of a ViT-L/B on the set-level objective:** with few collectors it memorises collector fingerprints (capacity ladder); high runtime; rejected as primary.
- **Per-row softmax over 4 candidates only (ignore the batch):** discards the coupling the description highlights; record priors do not cancel; expected to underperform by a measured amount (to be quantified in the roadmap's decode comparison). It is the fallback decode only if a reviewer ruled batch-joint decoding out (the text explicitly allows it).
- **Hungarian hard assignment as the final output:** optimal for exact-match count of the whole batch, but S is row-additive in marginals and R needs graded ranks; marginal sorting dominates in expectation. The MAP-first-choice-then-marginal-rest variant is evaluated as a paired comparison only.
- **Tight-crop-and-resize each insect to a fixed size:** destroys the physical scale that the description calls out as the shared measurement.
- **Random-resized-crop / scale jitter augmentation:** same reason.
- **Cross-batch re-pairing as extra negatives:** gated by the reviewer question (possible "manufacture training data" reading); not needed by the primary.
- **TF-IDF/regex/rule solvers or hard-coded relations (e.g. size vs latitude):** nothing textual here; banned by the strip test.
- **BioCLIP via open_clip:** not in allowed libs; reviewer question; if approved it would be the first backbone swap to measure (domain-matched).
- **Pseudo-labelling or test-time adaptation, whole-test normalisation of scores:** banned.
- **Large ensemble of near-identical seeds as the main lever:** variance reduction only; a handful of seeds is enough.

## Fixed work plan & runtime budget

**Assumptions (not stated in the description): single A10G 24 GB, runtime target <= 45 min with >= 30% headroom against CLAUDE.md's 50 min target (1.5 h official ceiling), nominal sizes: about 8k training sets (24k photos) and about 3k test sets (9k photos); unknown, so every stage below is quoted as a formula and the backbone/resolution constants are fixed after LOCAL profiling on the real counts, never by clock at run time.**

| Stage | Fixed work | Nominal estimate (A10G, fp16) |
|---|---|---|
| Load, asserts, group building (D7), mask size descriptors | one pass over images, CPU | ~3-5 min (parallel decode, fixed worker count) |
| Backbone feature extraction (train+test photos) | N_photos x 2 flips; ViT-L/14 at ~448 px, ~70-90 img/s | ~66k passes -> ~13-16 min; resolution/backbone constants fixed after profiling (a ViT-B/L at 336 px if N doubles) |
| Heads grid (<= 8 configs x 5 folds x 1 seed; fixed epochs; GPU, full quartets in memory) | ~3-5 s per fit | ~3-5 min |
| Final multi-seed refit on 100% of train (5 seeds x compat + aux; ridge fallback) | cheap | ~2 min |
| Temperature / blend fit on OOF | 1-3 scalars | seconds |
| LP-FT last block (1-2 epochs, one config, on all train; plus a 2-3 fold measured check) | image forward each epoch, bs fixed | ~8-12 min |
| Inference on test batches, decode, validation, write, re-read | trivial | ~1 min |
| Total | | ~30-40 min nominal; flagged as an assumption; cut resolution/LP-FT if the real counts push beyond 45 |

**Determinism.** Seeds: python/numpy/torch/cuda, `PYTHONHASHSEED`, DataLoader generators; `device="cuda"` hard-coded; `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`; fixed workers/threads; fixed fold assignment (seeded); fixed grid list (not Optuna time-based), no `elapsed()` in any condition, no env/cpu_count/`cuda.is_available()` branches, no import fallbacks. Pretrained revisions pinned. Fold models only for OOF; shipped predictions come from 100% refit; fixed epoch constants taken from the CV run (mean best epoch x 1.1, computed in-script from train CV).

**Memory.** Features in fp16: 24k+9k photos x 2 x 1024 dims is small. Fine-tune stage batch sized with headroom for 24 GB; no OOM at the largest batch (assert at the start with the real batch).

**Robustness.** Validate input (columns, 3 photos per set, 4 candidates per batch, identical candidate lists within a batch, finite year/doy/lat/lon, ids stripped consistently) and output (columns exactly `set_id`, `ranking`, row set equals the sample file's, each ranking a permutation of 0..3, JSON format matching the sample's spacing, written file re-read and re-checked). No silent fallback: any anomaly (e.g. a batch not having 4 rows) raises loudly, except that rows outside the evaluated batches get a valid default ranking from the model where possible (or `[0,1,2,3]` and a logged warning); the description says unscored rows only need validity.

## Metric-aware training & decode

**Back-solving the metric.** S = 0.6A + 0.4R is separable per row; the maximiser of expected S per row is the descending order of the posterior marginal P(i->j); no extra utility weights are needed.

**Posterior.** For a batch with potential matrix s (4x4), P(pi) proportional to exp( sum_i s[i, pi(i)] / tau ). Enumerate 24 permutations; marginals M[i, j]; row i ranks candidates by M[i, :] (tie-break by s). Temperature tau fitted by minimising the true-matching NLL on OOF predictions (one scalar, cross-fitted over folds), then fixed. Per the permutation structure the record-only and set-only terms cancel, so no prior correction is required.

**Training objective = the same likelihood** (exact marginal likelihood over matchings, per lesson F2.2), plus small row/column CE terms and aux axis heads. Loss weights per set are equal (the metric weights rows equally); no per-example weights to replicate.

**Decode comparisons (paired, same folds, same seeds), pre-registered:**
1. Marginal sort (primary).
2. Per-row softmax only, ignoring other rows (to measure the coupling gain).
3. MAP-matching first choice, remaining ordered by marginal.
Primary is kept unless (3) wins by more than noise, which is not expected under calibrated scores; the calibration of tau is the thing that matters (reliability of OOF top-1 marginal vs empirical accuracy is reported).

**Per-slot calibration.** Slots here are exchangeable (the true position is uniform, no positional prior to learn), so there is no per-slot recalibration; one scalar tau (+ blend weights) is the whole calibration, reported cross-fitted.

**Hard constraints.** Output validity enforced at write time; the decode itself is the only joint structure used.

## Structural signals

1. **Perfect matching per batch** -> matching likelihood, marginal decode; record-only and set-only potential terms cancel automatically.
2. **Exchangeability of the three photographs** -> permutation-invariant set encoder; random permutation and drop-one-photo augmentation.
3. **Physical scale is shared across all photographs** -> keep scale-preserving preprocessing; explicit size descriptors; no scale augmentation; flips/90-degree rotations are valid symmetries (insect orientation on canvas carries no signal; to be confirmed by a train diagnostic comparing orientation-augmented vs plain on the same folds, per lesson F2.6).
4. **Circular day-of-year** (Dec 31 adjacent to Jan 1; a 30-day window may straddle the year boundary) -> sin/cos harmonics and circular label smoothing; **sphere geometry** -> unit vector, great-circle km.
5. **Batch constraints** (within 30 calendar days, within 15 years, > 25 km or > 10 years apart) -> do NOT hard-code; use batch-relative record features so the model works at the right scale, and report the D3 diagnostic that these constraints hold in train.
6. **Within-batch symmetry:** all four records are positives for someone; hence no systematic offset of the true record versus decoys (unlike engineered-negatives tasks); matched negatives and positives share distributions, so no shift term is needed beyond the batch-relative features.
7. **Set-level auxiliary targets:** each train set's own record gives axis labels (no negatives required); co-event sets (same record tuple, if D2 finds repeats) give optional positive pairs for a supervised-contrastive auxiliary (leave-own-collector-out).
8. **Swap symmetry of candidate order:** the model's score for (set, record) must not depend on the position index; permuting candidate order and set order must leave marginals equivariant. Unit-tested.
9. **Uniform true position:** no positional prior (any position feature is ID/row-order leakage and banned).

## Experiment roadmap

1. **Contract, metric, decoder, validation correct and unit-tested.** Stop when: metric tests (perfect 1.0, reversed 0.10, random ~0.358, coupled-row property) pass; gold potentials through the decoder give 1.0 on every train quartet; derived-collector groups and V1/V2 splits built and their stats reported (D7). Candidate list identity within a batch asserted.
2. **Cheapest valid end-to-end baseline** (the fallback): size-only ridge + the frozen-probe Gaussian density through the matching decode -> valid `submission.csv` with exact format. Record V1 mean/std and V2. Stop criterion: a real number versus random 0.358; spend credit 1 only on this.
3. **Representation and structure:** compat model + matching loss, set encoder, batch-relative record features. Compare with the fallback paired on folds. Then backbone ladder (base vs large; resolution; CLS+mean patch; flips). Stop when two consecutive steps give gains within noise.
4. **Metric-aware loss/decode:** the three decode variants; tau calibration; reliability report. Aux heads on/off (paired). Photometric/grayscale variants for fingerprint robustness; compare under V1 and V2 together (a gain on V1 alone with a loss on V2 means fingerprint exploitation: reject).
5. **Compliance layer and diversity:** LP-FT last-block run with the change norm logged and CV gain; then family-2 blend; then candidate 3 (kernel neighbours) only if each alone is close in quality and the blend gains exceed noise (z-score/log-potential blend, fit cross-fitted). Best single -> credit 2; ensemble -> credit 3.
6. **In-script bounded HPO:** fixed list of <= 8 configs (d, K, weight decay, dropout, aux weight), fixed seeds, selected on inner folds of each outer training part; report the outer score. Prefer plateaus.
7. **Final fixed-plan run:** refit all, run end-to-end twice from a clean `working/`, diff the submission (identical or near-identical rankings), validator and compliance audit, then the final credit. Report CV (per fold + mean +/- std over seeds), fixed plan, runtime, audit, validator.

## Compliance audit

CLAUDE.md section 7:
- Test file read only for inference per batch (decode within a batch only): yes. No scaler/vocabulary/clustering/dedup/rank-normalisation across the test file; tau and blend weights come from train OOF: yes.
- Wall-clock in conditions: none planned (logging only).
- `cuda.is_available()` / `cpu_count()` / try-except import fallbacks: none planned.
- Hard-coded tuned constants: none by design; every constant (crop scale, epochs, tau, blend weights, LP-FT decision) is computed in-script from train-only search or a documented fixed grid; the only hard-coded values are grid lists and the fixed schedule (a flag for the implementer: re-derive crop constants in-script from train percentiles, do not paste values tuned from watching submissions).
- External data/synthetic data: none. Pretrained HF/timm weights only; no open_clip until a reviewer says yes.
- Strip-the-ML: passes (random 0.358 without the learned model).
- Hand-labelling: none.
- Source readability, < 512 KB, plain code: planned.
- Challenge-specific restrictions: each ban in the Compliance regime section mapped to a plan item.

Section B self-audits:
- *Strip-the-ML:* passes as argued.
- *No whole-test aggregation:* the decoder's only cross-row use is within one `batch_id` (4 rows), explicitly allowed; no test-wide rank/z-score or temperature.
- *Sibling leakage:* the relation "set owns record" is between a set and ONE of the four records; features computed from the set's batch-mates (batch-relative record offsets, other sets' scores in the marginals) are batch-local and explicitly allowed; I will still ablate the batch-relative features to make sure the gain is not an artefact of the generator (if relative features alone, without photos, beat random, that would show generator leakage and they would be removed).
- *Hard-coded constants derivable by an in-script train-only search:* yes (see audit item above).
- *Inference-only / frozen-features-plus-small-head:* addressed with LP-FT and the logged parameter-change norm; the trained set encoder, record encoder and compatibility potentials are the entire predictor.
- *Whole-train+test fits:* none (any normaliser over features is fit on train).
- *Label-derived statistics:* the only ones are the aux-head targets and (optionally) kernel-neighbour references, both leave-own-collector-out.

## Open questions & assumptions

**Assumptions where the description is silent (stated, to be confirmed):**
1. Compute: single A10G 24 GB (from CLAUDE.md). Runtime: not stated; planned to <= 45 min nominal with >= 30% headroom against 50 min; official ceiling 1.5 h.
2. Dataset sizes (train sets, test batches, canvas resolution, number of collectors): not stated; nominal 8k train sets / 3k test sets used for budgeting only; all stage times are formulas to be re-profiled locally. The 1,000,000-row cap is a generic limit and says nothing about the real test size.
3. File names and columns: assumed `train.csv`, `test.csv`, `sample_submission.csv` plus an image directory; columns `set_id`, `batch_id`, `candidates`, three photo references, and a true-owner label in train (position or record). Real names to be read from the data; the script must assert and loudly fail on mismatch.
4. Domain label (CV vs Fine-tuning vs from-scratch): not stated; assumed CV-like with pretrained weights allowed (the description does not say "from scratch"); HF/timm downloads allowed per CLAUDE.md.
5. Whether train contains a collector/person column: not promised ("none carries information" for ids); assumed absent, hence derived groups (D7). If a column exists, still union it with derived edges; do not use it as a feature.
6. Whether train rows come in the same quartet structure as test: assumed yes ("training and evaluation rows carry the same feature columns"; batch_id shared by four sets); if train has no quartet structure, the matching loss must be re-derived (e.g. training pairs only with the set's own batch, which would not exist) and the plan would be re-run.
7. Ranking JSON formatting: copy the format of the sample file (spacing) and assert it parses to a permutation.
8. Position index = index into that row's own `candidates` list (shared within the batch); asserted in-script; if lists ever differ across the four rows of a batch, map by record tuple.

**Reviewer questions (with a plan under each reading):**
1. Is a frozen backbone + trained set/record heads (+ LP-FT last block) acceptable as "train genuinely" for this challenge? Reading A (accepted): ship as primary. Reading B (genuine backbone fine-tune required): LP-FT is already in the plan and shipped by default; escalate to partial fine-tune only if V1 and V2 both show a measured gain over the probe.
2. Is exact matching marginalisation within one published batch OK (the text says yes)? Reading B (not allowed): per-row softmax decode; measured cost reported by decode comparison 2; expected to be sizeable because record priors no longer cancel.
3. Is re-pairing training sets with records of other training batches (extra negatives) "manufacturing training data"? Default: not used. If cleared, add as a gated roadmap experiment only if the quartet-only model underfits (train NLL high, held-out NLL equal).
4. Is `open_clip` / BioCLIP-type weights allowed (domain-matched backbone)? Default: DINOv2 via transformers.
5. Are hand-measured physical-size mask descriptors (deterministic image statistics fed to the trained model) acceptable as features? Default: yes (hand-engineered features fed into a real model are explicitly allowed in CLAUDE.md 2.2); a variant without them is kept as an ablation.

**What I could not verify.** Every dataset fact: schema, sizes, canvas, number of collectors, whether the quartet structure exists in train, which record axes the photos reveal, whether record-only labels recur, the true test split (grouped by collector is inferred only from the description), backbone availability on HF, throughput numbers, and the score band. No score is promised; the band 0.40-0.55 is an unvalidated estimate.
