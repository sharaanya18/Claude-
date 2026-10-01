# U21 Transit Stop Order Reconstruction - Eris plan

Status: written with NO dataset available. Everything under "Data findings" is a diagnostic to run plus a hypothesis, marked UNVERIFIED. Numbers quoted from the description are the description's own (measured on eval cases by the creator), not ours.

## Contract & decision unit

- One valid answer per case: a permutation of exactly the case's `stop_ids` (no omissions, no duplicates, no extra ids), written space-separated in `ordered_stop_ids`, with `first_stop_id` at position 1. Columns `case_id,ordered_stop_ids`; 4,822 rows; one row per `case_id` in `test.csv`; any row order; duplicated `case_id` is rejected; an empty cell scores 0 for that case.
- Invalid vs low-scoring: renamed columns or duplicated case_id = rejected. Omitted stops or empty cell = scored but charged as wrong against every pair. A complete permutation is therefore always at least as good as a partial one. Our validator must refuse to write anything that is not an exact permutation of the case's set with f first.
- Metric, term by term: `mean_ordering_agreement` = mean over cases of `max(0, (agreed - wrong) / C(n,2))`. For a complete permutation `agreed + wrong = C(n,2)`, so the case score is exactly Kendall tau (agreed minus discordant, over all pairs), floored at 0. Consequences:
  - Rewards global pairwise order, not adjacency. A local swap costs 1 pair; a wrongly placed corridor costs many.
  - The n-1 pairs involving f are free once f is first (fraction 2/n of all pairs, 0.10 at n=20). This is why "f at front, rest as given" scores 0.142 vs 0.07.
  - Per-case floor at 0 means reversal is not punished below 0. Direction is given by f, so this is irrelevant for us. The floor mildly favours gambles on hopeless cases (negative tau is free). We ignore this and maximise expected tau (first order).
  - Every case has weight 1 (no per-network macro average). Large networks (more cases) matter proportionally more. Loss weighting should be per case, equal weight, mean over that case's pairs.
- Sanity check of the metric reading (derivation, not data): random permutation tau has sd about sqrt(2(2n+5)/(9n(n-1))), about 0.16 at n=20; E[max(tau,0)] = 0.4 sd, about 0.065. This matches the stated random floor 0.070 and sample_submission 0.071, which confirms the "tau floored at 0" reading.
- Discrepancy flagged: the description says dropping one stop from a perfect 20-stop answer scores 0.760. By its own formula that is (171-19)/190 = 0.800 at n=20 (general: 1 - 4/n). It would be 0.760 only at n=16.7, so it is probably an average over the n distribution. UNVERIFIED, no effect on us because we always submit complete permutations. The unit test implements the stated formula literally and records this.
- Independent unit: the ROUTE (not the case, not the network). A route yields several patterns (two directions, short workings, variants), all on the same side of the split. Eval cases are from held-out routes inside the same 58 networks. Effective sample size is the number of routes, which is probably a few thousand (14,424 cases / several patterns per route; UNVERIFIED), not 14,424.
- Pipeline stages and how to diagnose each separately:
  1. Geometry-only ordering evidence (noisy points, rotation unknown per network).
  2. Network evidence (train patterns of other routes: stop-level progress, pairwise precedence, adjacency, directional corridors). Available for about 51.5% of stops and 37.2% of true consecutive pairs per the description.
  3. Decode (turn pairwise beliefs into one permutation that maximises expected tau).
  Diagnose each separately: geometry-only model; evidence-only on covered pairs; decode on gold pairwise scores (oracle, must give tau = 1).

## Compliance regime

Domain: "Other" (structured ordering/permutation). Compute: A10G. Labelled neither "Fine-tuning" nor "From scratch"; the description says there is no pretrained model for this task and anything must be fitted to the supplied training cases. So: train from scratch on the provided data only, no pretrained weights, no internet needed at all. This is the cleanest regime.

Explicit bans in the description (hard constraints) and how the plan respects each:
1. Hardcoding an order for specific test cases. We never do this; every output comes from a trained scorer plus a generic decoder.
2. Using `case_id`, `network` or `stop_id` strings as a prediction signal. No parsing, sorting, hashing-into-features or substring use of any id. `stop_id` and `network` are used only as equality join keys (to fetch the same network's training patterns / the same stop's coordinates). No learned embedding indexed by id and no per-network learned parameters. Network-level inputs are train-derived numeric descriptors (for example mean consecutive step) only.
3. Using presentation order in `stop_ids` or row order of any file. Internally, the first thing done to a case is to reorder its stops by a geometric canonical key (distance to f, then x, y), or shuffle with a seeded RNG, so no code path can see presentation order. The transformer has no positional encoding, so it is permutation-equivariant. Unit test: permuting `stop_ids` must leave predictions identical to 1e-5. Decode ties are broken by geometry, never by input or id order. Avoid anything that iterates over a Python `set` or `np.unique` of ids and lets that order leak.
4. Identifying real networks or consulting external sources (timetables, maps, geocoders, stop datasets). No external data and no network access at all in the script. Do not try to match coordinates to maps (they are rotated and rescaled anyway).
5. Private, role-gated or API models, external inference APIs, non-reproducible external weights. None used; no pretrained weights at all.

Where the description is silent, and what we assume:
- Runtime limit: silent. Assume CLAUDE.md: target <= 45 min on A10G, hard ceiling 1 h; no wall-clock branching (CLAUDE.md section 3).
- Use of the test set: the description allows "stop sets, coordinates, given first stop, and network structure learned from training". CLAUDE.md section 2.3 #5 additionally bans anything fitted on test. Conservative reading adopted:
  - A test case's prediction is a function of its own stops, their coordinates, f, and statistics fitted on TRAIN cases only.
  - No pooling of test cases: no joint decode across test cases, no counting how many test cases contain a stop, no consistency between test patterns of the same network, no pseudo-labels.
  - `stop_coordinates.csv` contains eval-only stops. We read coordinates only for the stops of the case being predicted. Any network-level geometric statistic (typical step, density reference, corridor graph) is fitted ONLY from stops that appear in train cases (train-only). The network centroid/scale are the organisers' preprocessing, not our fit; distance-from-centre is a per-stop attribute supplied in the file.
- "Network structure" feature family keyed by stop identity: explicitly invited by the description ("a stop you have to place may already appear in fifteen other patterns whose order you know"). Treated as allowed, with one reviewer question (see Open questions).
- Strip-the-ML test: without the trained model the best rule route is NN + 2-opt at 0.317 (stated). Our design uses 2-opt only as an input feature or fallback path to a trained scorer; the trained scorer must clearly beat it (target at least +0.10 over 0.317 under grouped CV), otherwise the solution is "rule-based with a wrapper" and should not be submitted. The model consumes learned evidence from train patterns and learns the deviations from geometry, which is the point of the task.
- TTA by co-rotating all geometry-tied inputs is allowed (per-sample inference trick).
- Determinism per CLAUDE.md section 3: fixed epochs, folds, seeds, batch, workers=0 or a fixed small number, `device="cuda"` hardcoded, no `is_available()` switches, no import fallbacks, time used for logging only, `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch.

## Data findings

NOTHING BELOW IS VERIFIED. Each item is a diagnostic to run on train only (test: schema, row count and size statistics only), with the expected result.

Diagnostics to run (D1-D14):
- D1 Structure and integrity. Shapes, dtypes, n_stops distribution (10-40; per-n counts), cases per network (58 networks; about 249 train and 83 test cases per network on average; min/max), stops per network (96,027 / 58, about 1,656 on average), join coverage of `stop_coordinates` onto every stop in train/test, duplicate coordinates, NaN. Assert every `ordered_stop_ids` is a permutation of `stop_ids`, with its first element equal to `first_stop_id`.
- D2 Metric unit tests and reproduction of the stated baselines on TRAIN (the description's numbers are on eval, so expect similar, not identical): given-order about 0.07, f-first about 0.14, dominant-axis about 0.24, NN-from-f + 2-opt about 0.32, mean relative position across train patterns about 0.22. Matching these validates the metric, the geometry loading and the NN/2-opt code. Also perfect = 1, reversed = 0 (floored), constant/random about 0.07.
- D3 Noise reading. Mean observed distance between consecutive stops in true order. Hypothesis H_noise: the description's 0.092 is the pre-noise step, since noise sd 0.16 per axis would otherwise be impossible (independent per-stop noise gives an observed consecutive distance of roughly 0.25-0.30, about 3x the step). If so, local order is mostly unrecoverable from geometry, only the corridor shape at scales >= several steps is, and tau is dominated by global structure. Also measure the observed-distance rank of the true successor among all unvisited stops (rank-1 frequency expected well below 50%) and oracle candidate recall of the true successor within K nearest unvisited for K = 3, 5, 10, 20, all. This decides whether any K-nearest candidate pruning is allowed (likely not; use all stops, n <= 40).
- D4 Ceiling probes. Tau of the true order after adding jitter (swap within windows of w = 2, 3, 5, 8 positions) to see how much a local-error-only solution would lose. Tau of "oracle knows each stop's rank within +-k" to bound what geometry plus smoothing can achieve.
- D5 Route groups. Within each network, build the overlap graph between train cases (containment = |A∩B| / min(|A|,|B|), plus Jaccard, plus reversed-order flag). Report the component size distribution at several thresholds, the share of networks collapsing into a giant component, the number of groups per network, and whether reverse-direction patterns (same set, reversed order) are present and merged. Targets: groups about equal to routes; no giant components.
- D6 Calibrate the grouping against the description's own held-out statistics. For a candidate group threshold, hold out 25% of groups per network and measure (a) the fraction of held-out-case stops that appear in the remaining train patterns (target 51.5%), (b) the fraction of held-out consecutive pairs occurring as a pair (either direction) in the remaining train patterns (target 37.2%), (c) the mean share of single-route stops (target 57.8%, route approximated by group). If our values are clearly higher, groups under-merge (sibling leakage) and the threshold must be loosened until they match. The grouping threshold is then chosen by an in-script, train-only search against these description constants.
- D7 Evidence strength by network. Per network: stop coverage, pair coverage, number of patterns per stop. Plot per-network CV tau against these (hypothesis: strong positive slope, dense urban networks gain a lot, sparse interurban ones about the geometry-only level).
- D8 Direction/corridor consistency. For stops in >= 2 train patterns of different groups: share where successor displacement vectors point the same way (cosine > 0.5); share of reverse-direction pairs where the stops are traversed in the opposite order. Decides whether directional evidence vectors are worth including and whether reversal augmentation is a valid symmetry (diagnostic: how often the reversed pattern, with last stop as new first, exists as a real train pattern).
- D9 Out-and-back / doubling-back. Fraction of train cases with pairs of stops closer than 1 observed-step but order gap > 5; fraction of cases where the route's start/end are near each other (loop). Tells how much a smooth-curve prior alone (principal curve) would fail.
- D10 First-stop prior. Rank of f along the case's principal axis, convex-hull-extremity of f, Spearman of true position vs distance from f, share of cases where the last stop is the farthest from f. Quantifies how much the f-relative frame carries.
- D11 Reflection symmetry. Compare the held-out score of the same model with and without mirror augmentation (all vectors reflected together). Reflection is not assumed to be a symmetry (driving side, one-way loops); keep it only if grouped CV improves by more than the noise.
- D12 Scale features. Relationship between n_stops, case spatial spread (largest PCA eigenvalue) and tau; typical step per network (train patterns) vs a per-case nearest-neighbour scale. Check that "network-step-normalised" and "case-normalised" distances behave consistently across networks.
- D13 Order/leak audits. Spearman of true rank with: presentation index, row index, case_id/stop_id string order. All expected about 0; the description says the pool-order and coordinate-only attacks sit at random. If any is not about 0 it is NOT a feature (banned) but would be reported.
- D14 Ambiguity. Duplicate stop sets within a network and their label agreement (expected: reverse directions agree up to reversal; if the same set appears with different true orders, that is irreducible ambiguity for the set-only part).

Hypotheses (UNVERIFIED), in order of expected impact:
- H1 Geometry gives the global shape; local order is mostly noise (H_noise). A model that denoises implicitly (attention over the whole point set) should beat NN + 2-opt (0.317) clearly, possibly 0.40-0.50 geometry-only.
- H2 Network evidence is the main upside but applies to only about half of stops and about 37% of true consecutive pairs; benefit is larger for pairs and stops with evidence, near zero for the 48.5% never seen. A learned model must tell "no evidence" (missing flag) from "evidence says unrelated".
- H3 Sibling leakage in naive random-case CV would inflate scores massively (patterns of a route are near-copies, including reverses); only route-grouped CV is trustworthy.
- H4 Reversal and mirror symmetries are not assumed.
- H5 Routes double back and run out-and-back; a path prior alone will mis-order those; evidence and the learned model must carry them.

## Validation design

Reproducing the split: test is a route-held-out split inside each network (about 25% of routes; 4,822 / (14,424 + 4,822)); all 58 networks appear on both sides; all patterns of a route on one side. Mirror it as follows.
1. Derive route groups within each network from TRAIN content only, by union-find on stop-set overlap (containment AND Jaccard thresholds) plus reversed-pattern merge. The threshold is selected by an in-script, train-only grid search so that the simulated held-out stop-coverage (51.5%) and consecutive-pair coverage (37.2%) match the description's eval statistics (D6). Guards: a cap on component size and a fallback to average-linkage clustering if single-linkage chains into a giant component. Over-merging is the safe error (stricter hold-out); under-merging leaks.
2. Folds: StratifiedGroupKFold(5), groups = derived route groups, stratified by network so every network is on both sides. Development: 2 split seeds (paired comparison on the same folds). In-script: the same group definition, 2 outer folds (see work plan) for calibration only.
3. Nested evidence features. For the outer validation fold, ALL evidence statistics (stop progress, pair precedence/adjacency counts, network step) are built only from the outer-train part. For outer-train cases, evidence is built leave-own-route-out (subtract the case's own group's contributions from the counters) so the features look like the test-time ones. Amount-of-data match: a test case sees all train routes (about 75% of all routes); a train case sees all train routes minus its own: near-identical. Strict nested check required: re-build features inside the fold and confirm that training-feature distributions (coverage fractions, count histograms) match validation-feature distributions; a leave-one-out statistic on a small base would otherwise inflate CV.
4. Metric: exactly the description's formula, floored per case; report mean +- std over folds and over seeds, per-network tau and per-n tau. Accept a change only if the paired fold difference exceeds 1 standard error and holds in most folds and both split seeds.
5. Holdout sanity: one group-disjoint 10% holdout (by group, stratified by network) untouched by any selection step (epochs/architecture/decode choice), scored once at the end.
6. Post-hoc selection steps (temperature scalar for pairwise probabilities, blend weight GBDT/transformer, decode variant) each get cross-fitted checks: fit on fold A OOF, evaluate on fold B.
7. Bias of the proxy (direction and rough size, UNVERIFIED):
   - Outer folds hold out 20% of train routes in addition to the real held-out 25%, so evidence is thinner than at test (60% vs 75% of routes): PESSIMISTIC for evidence-dependent gains, by roughly a few tau points in dense networks.
   - Under-merged groups: OPTIMISTIC (leak). Mitigated by D6 calibration.
   - The final model refit on 100% of train has the full evidence and should land slightly above the CV number on the evidence part. Expect real-test tau close to CV, plausibly a few points higher; do not promise.
8. Public LB is only a sanity check; do not tune on it.

## Overfit/underfit risks

Overfit (with mitigations):
- Sibling leakage: variants/reverse directions of the same route in both fold sides. Mitigation: derived groups, D6 calibration, strict nested evidence.
- Evidence leakage: a train case's own route (and its siblings) contributing to its own evidence features. Mitigation: leave-own-route-out counters, nested per outer fold.
- Train/test evidence mismatch (leave-one-out bias). Mitigation: as above; validation by comparing feature distributions.
- Few independent units (thousands of routes) vs a transformer with a few hundred thousand parameters, and absolute coordinates letting the net memorise (network, location) -> progress. Mitigation: input only coordinates relative to f and rotation-invariant quantities, co-rotated random rotation augmentation (all geometry-tied inputs rotate together), dropout, weight decay, fixed short schedule, EMA, multi-seed averaging, small model (d=128, 4 layers).
- Many tuned knobs. Mitigation: fixed a-priori design constants, adopt a change only if it beats noise; no deep HPO (small CV can only resolve large effects).
- Selection on reported OOF: decode/calibration scalars cross-fitted.

Underfit:
- Treating the problem as local (greedy NN) when the metric is global; mitigated by pairwise all-pairs loss and global attention.
- Throwing information away: normalising away absolute scale (the step length 0.092 vs noise 0.16 is meaningful) - keep both network-step-normalised and case-normalised distances; keep the missing-evidence flags; do not truncate n.
- Too little training or too small a model. Mitigation: profile epoch time and use the budget (about 40-60 epochs).
- Loss mismatched with the metric. Mitigation: all-pairs logistic loss with per-case mean (surrogate of tau).
- Ignoring evidence beyond direct pairs. Mitigation: roadmap step adds multi-hop reachability.

## Recommended approach (primary + fallback)

### Primary: relation-aware set transformer with all-pairs precedence head and Kemeny-style decode (lean design)

Representation (per case, canonical geometric ordering of stops; f flagged):
- Stop tokens. Coordinates relative to f in two normalisations (divided by the train-derived network step; divided by the case spread/median nearest-neighbour distance), distance to f and its within-case rank, distance from network centre, f-flag, n_stops, case anisotropy. Case-level geometry features are computed from the case's own stops only.
- Stop-level network evidence (train patterns, leave-route-out): known flag, log count of patterns containing the stop, mean/std of normalised position (rank fraction) in those patterns, counts as first/last stop, mean unit successor and predecessor displacement vectors. All vectors are expressed in the same frame as the coordinates and rotated together with them under augmentation.
- Pair-level relation features, used as additive attention biases and as inputs to the pair head: relative displacement (dx, dy) and distance (both normalisations), counts of "i directly before j" and "j directly before i" in train patterns, counts of "i before j anywhere" and "j before i anywhere" for co-occurring pairs, a co-occurrence flag, and a "no evidence" flag. This is a few learned channels, not a lookup table.
- Rotation: random rotation (and optionally reflection, only if D11 says so) of the whole case, including all evidence vectors, at train time; at test time average over a fixed set of 8 rotations (TTA).

Model: d=128, 4 heads, 4 layers, no positional encoding, attention logits = QK + learned per-head bias MLP(pair features). Output: per-stop progress score s_i (rank fraction regression, auxiliary) and antisymmetric pair logit `L_ij = g(h_i,h_j,e_ij) - g(h_j,h_i,e_ji)`; `P(i before j) = sigmoid(L_ij / T)`.

Loss: binary cross-entropy over all pairs (i,j) of non-f stops with the true order as the label; per-case mean over pairs, then mean over cases (the metric's own hierarchical averaging); plus a small weight on the progress regression. Pairs with f are masked from the loss (f is first by contract).

Training: AdamW, cosine with warmup, bf16 autocast, grad clip 1.0, EMA of weights, fixed epochs, batch of 64 cases padded to 40 with masks, no validation-triggered stopping.

Decode (metric-aware, first-order): the expected number of agreed pairs is the sum over ordered pairs of P(i before j) for the chosen order. 1. Start from sorting by averaged progress score with f forced first. 2. Run a fixed-pass adjacent-swap/insertion local search maximising `sum_{i before j} P_ij` (Kemeny objective on calibrated probabilities), with a fixed pass cap and geometric tie-breaking. 3. Temperature T is a single scalar fitted on OOF NLL (cross-fitted). Compare against plain sort-by-score on OOF; keep the Kemeny step only if it beats noise.

Shipped model: refit on 100% of the train cases, 3 seeds, average logits across seeds and TTA rotations. Fold models exist only to produce OOF numbers and the calibration scalar.

Why this fits: tau is a pairwise metric, so an all-pairs loss is the exact surrogate; global attention handles denoising of the route shape (noise 1.7 steps); relation features give a place to put the sparse, uneven network evidence; one forward pass and a cheap decode keep determinism and runtime safe.

### Fallback / diversity member: stop-level GBDT on engineered features

LightGBM (fixed seed, fixed rounds, fixed thread count) regressing the rank fraction `pos/(n-1)` of each stop, features: relative coordinates and distances to f, projection onto the case's principal axis (signed relative to f's side), position along the NN + 2-opt path from f (and its rank), position along a principal-curve fit (fixed 3 iterations of: order by projection, smoothing-spline fit of coordinates vs rank, re-project), within-case rank-normalised distances, local density, evidence features (leave-route-out mean rank fraction, counts, known flag, successor-vector projection onto the direction from f). Sort by prediction (f first). Compliant because the trained GBDT carries the model; the 2-opt and principal-curve routines are only feature generators. Its blend into the primary is done in logit space (`sigmoid(k * (s_i - s_j))`, k fitted on OOF) with one weight fitted on OOF, cross-fitted. Blend only if its standalone CV is within a few points of the primary.

### Roadmap candidates (enter only after the lean design is measured, gain > noise)
- Auxiliary successor head: softmax over the case's stops for each stop's true successor (plus an END token), trained by exact cross-entropy. Gives local path continuity; decode objective becomes precedence + lambda * successor log-probabilities.
- Multi-hop directed reachability on the train successor graph (hop distance i -> j).
- Autoregressive next-stop (pointer) head with fixed-width beam and minimum-Bayes-risk reranking on expected tau, as a diversity member.

## Rejected options

- NN + 2-opt tour alone (0.317): fails the strip-the-ML test; kept only as a feature and the floor to beat.
- Sorting by dominant axis or mean train position (0.236 / 0.216): same reason, and they are weak.
- 58 separate per-network models: about 250 cases each, thousands of free parameters; the task is one functional form (description) with network-level numeric descriptors as inputs.
- Learned stop-id embeddings or per-network embeddings: id-as-signal grey zone, memorise sibling routes, do not transfer to held-out routes.
- Random case-level K-fold or GroupKFold by `network`: the first leaks siblings; the second removes the very evidence (same-network patterns) available at test.
- Reversal augmentation as default: one-way systems break it; allowed only if D8/D11-style train diagnostics support it.
- Candidate pruning to K-nearest unvisited for the pointer decoder: noise makes the true successor often far in rank (D3 decides).
- Using test stops to build network density/corridor graphs, joint decode across test cases of a network, counting a stop's test occurrences: transductive, banned by CLAUDE.md section 2.3 #5.
- Exact TSP / Concorde / OR-tools style solvers as the main method: rule-based, wrong objective (route is not a shortest tour).
- Large pretrained models: none applicable ("nothing here can be solved by recalling text").
- Deep HPO (many Optuna trials): the CV cannot resolve small gains with thousands of routes; plateau-friendly fixed design instead.

## Fixed work plan & runtime budget

All counts are fixed in the script. The runtime numbers are ESTIMATES, to be replaced by local profiling (one epoch, one fold) before hardcoding the epoch count (CLAUDE.md section 3 rule 7). Assumed hardware: one A10G, `device="cuda"`.

| Stage | Fixed plan | Est. A10G time |
|---|---|---|
| 0. Load, assert integrity, canonical case ordering | pandas, vectorised | 1 min |
| 1. Route groups (train only), threshold search against 51.5% / 37.2% | grid of about 6 thresholds, union-find | 1-2 min |
| 2. Evidence counters (per outer fold and final), leave-route-out pair/stop tensors (float16, about 200 MB) | numpy | 3-4 min |
| 3. Fold models for calibration only: 2 outer folds x 1 seed | 40 epochs, batch 64, d=128, 4 layers (about 6 s/epoch est.) | 2 x 4 = 8 min |
| 4. GBDT fallback: 2 folds + full | 600 rounds, fixed lr, fixed threads | 4-5 min |
| 5. Final transformer: 100% train, 3 seeds | 40 epochs each | 3 x 5 = 15 min |
| 6. Inference: 3 seeds x 8 rotations x 4,822 cases, calibration T, blend, decode | fixed pass cap (e.g. 30), vectorised swaps | 3-4 min |
| 7. Validate and write | permutation check, re-read file | < 1 min |
| Total (estimate) | | about 35-38 min, about 25% headroom to 50 min; trim one seed or fold model if the profile says otherwise (static change to counts, never a runtime branch) |

Memory: padded batch 64 x 40 x 40 x C pair channels in bf16 is tiny; whole precomputed pair tensors stay on CPU (about 200 MB) and stream to the GPU. 24 GB is ample. No time-dependent control flow, no `torch.cuda.is_available()` switch, fixed `num_workers` (0 recommended: tensors are precomputed), seeds for `random`/`numpy`/`torch`/LightGBM, `PYTHONHASHSEED`, a seeded `torch.Generator` for shuffling.

Constants note: width, depth, epochs, LR are a-priori design choices, profiled for runtime only; the quantities fitted by data inside the script are the grouping threshold, the temperature T, the GBDT blend weight/scale and the decode variant (via the OOF held-out folds). No parameter pasted from an offline search.

## Metric-aware training & decode

- Back-solve of the metric: complete permutation => case score is Kendall tau floored at 0, so the target is per-pair order labels; no richer reconstruction is available beyond the full order itself (which we have).
- Training loss: all-pairs BCE, per-case mean then mean over cases (matches the per-case equal weighting); f-pairs masked. Auxiliary rank-fraction regression with small weight (progress score also gives the sort initialisation).
- Hierarchical averaging: case-level, not pair-level pooling (a 40-stop case has 780 pairs vs 45 for a 10-stop case; pair pooling would over-weight long cases relative to the metric).
- Decode: maximise sum of `P(i before j)` (expected agreed pairs), a Kemeny/feedback-arc-set objective, using calibrated probabilities; f forced first; local search with fixed pass cap. The floor at 0 makes expected-utility decode slightly gamble-friendly on hopeless cases; ignored (first order).
- Calibration: one temperature T on OOF pair logits (cross-fitted: fit on fold A, evaluate on fold B); per-structural-slot recalibration is not needed since the objective is ranking-based, but check the reliability of P for evidence-covered vs uncovered pairs (a two-bucket recalibration only if it helps across folds).
- Hard constraints at decode: f first; output is a permutation of the case's set (re-checked in the validator).
- Oracle check before modelling: feed the gold pair matrix (P = 1 for true order) through the decoder; it must reproduce every train order exactly (tau = 1), proving the decoder, parsing and formatting; also measure the decoder's runtime on 40-stop cases. Also feed the gold successor structure (if the roadmap successor head is added) to verify the path objective.
- Frozen/zero-shot yardstick under the same folds: NN + 2-opt from f (expected about 0.32) and the GBDT geometry-only variant.

## Structural signals

- f is first: mask f pairs, force at decode, include f-flag and f-relative frame (free 2/n of pairs).
- Presentation-order invariance: transformer without positional encoding; canonical geometric ordering; unit test with random permutation of `stop_ids`.
- Rotation invariance of geometry within a case; but evidence vectors live in the network frame, so the augmentation must co-rotate every geometry-tied input (coordinates, successor/predecessor vectors, pair displacements) by the same angle; assert on every training batch that the relative geometry of evidence vs coordinates is preserved (rotate a case and verify features rotate accordingly).
- Reflection: only if D11 shows a gain; if used, mirror all vectors together.
- Same stop = same displaced coordinates in every case (stated), so train patterns give true successor pairs whose observed positions are in the same noisy frame as the test coordinates; averaging does not denoise (stated), but consecutive-pair displacement statistics learned from other patterns directly encode the noise-corrupted corridor direction.
- Route variants and reverse directions: patterns of one route are near-copies; use them for route grouping (leak control) and, for evidence, remember that the reverse pattern of a held-out route is also held out. Evidence from other routes' reversals is legitimate corridor-direction evidence.
- Antisymmetry of the pair head by construction; transitivity is enforced only by the decode (a single permutation).
- Unseen stops (48.5% in eval): the model sees only the missing-evidence flag + geometry, so unseen stops are ordered through context stops that do have evidence; attention provides that coupling.
- Coupling check (structural couplings between outputs): progress score vs pair precedence are the same information; if the Kemeny decode and sort-by-score disagree materially, investigate the pair head's coherence.

## Experiment roadmap

1. Contract/metric/validation. Implement and unit-test the metric (D2), derive route groups with the D6 calibration against 51.5% / 37.2% / 57.8%, set up StratifiedGroupKFold(5) x 2 seeds, build the validator and permutation checks. Stop criterion: stated baselines reproduced approximately on train, group diagnostics match the description's coverage statistics.
2. Cheapest valid baseline end-to-end: NN + 2-opt from f (expected about 0.32 CV, shows the CSV path works) and f-first + sort by mean-position. Check the CSV format with the free upload check.
3. Geometry-only GBDT (stop-level rank-fraction regression): measure the gain over 2-opt; run oracle-candidate-recall and ceiling diagnostics (D3, D4). Stop when added features give no CV gain beyond noise.
4. Add evidence features (leave-route-out), nested per fold; check the train/validation evidence-distribution match; quantify gain by network density (D7). This step carries the evidence-design risk; do not proceed with a mismatch.
5. Primary transformer geometry-only vs geometry + evidence, all-pairs loss, rotation augmentation, TTA; ablations: reflection (D11), relation biases on/off, rank-regression aux on/off. Paired folds, 2 split seeds.
6. Decode: sort vs Kemeny local search; temperature fit; gold-matrix oracle check. Keep Kemeny only if it beats noise.
7. Roadmap extras one at a time only if steps 5-6 are solid: successor aux head, multi-hop reachability, pointer head, GBDT blend. Each must clear 1 standard error paired.
8. Freeze a final fixed plan (counts in section 8), refit on 100%, three seeds; run from clean `working/` twice and diff submissions (expect near-identical); the group-disjoint 10% holdout scored once; sanity check that predicted-order statistics (e.g. tau between submitted order and 2-opt order, distribution of first-few-step distances) resemble OOF.
9. Credits: baseline (step 2), best single model (step 5), final ensemble (step 8); do not spend credits on tweaks.

Estimated outcome (ESTIMATE, not a promise, no data seen): geometry-only learned ordering 0.38-0.48; with network evidence 0.45-0.62, centre about 0.52. Reasoning: the floor/no-learning anchor is 0.317; noise of 1.7 steps limits local order; evidence reaches about half of stops and about 37% of consecutive pairs and helps unevenly by network.

## Compliance audit

CLAUDE.md section 7 red items against the plan:
- Test file used only for per-case inference? Yes: the test case's stops, coordinates and f are the only test inputs; all statistics from train. No joint/pooled test processing. (pass)
- Time in any condition/argument? No; log only. (pass)
- `is_available()` / `cpu_count()` / import fallbacks / try-except changing work? None planned. (pass)
- Hardcoded offline-tuned constants? Architecture/epoch/LR are a-priori and profiled for runtime only; data-fitted scalars (group threshold, T, blend weight) are found in-script. (pass, with Q3)
- External data, synthetic training data, hosted weights, non-allowed libs? None; torch, numpy, pandas, scikit-learn, lightgbm, scipy only. (pass)
- Strip-the-ML test: removing the trained model leaves NN + 2-opt at 0.317; the trained model must add at least about 0.10 under grouped CV or the plan is not worth submitting. (conditional pass; measured in step 5)
- Source readable, < 512 KB, no blobs: plain Python with comments. (pass)
- Challenge-specific bans (section "Compliance regime" items 1-5): no id-string signal, no presentation/row-order signal, no external sources, no hardcoded orders, no private models. (pass; unit tests for permutation invariance and no id parsing)
- Sibling/related-row leakage: the target is a relation between mates of a group; features from other groups only (leave-own-route-out), checked by nested CV. (pass by design, verified in step 4)
- No whole-test aggregation: no per-network test statistics, no rank/z-score across test, no vocabulary or encoder fit on test. (pass)
- Hard-coded constants derivable by in-script train-only search: grouping threshold search against description constants; T and blend weight from OOF. (pass)
- Prompt-compliance wording: comments in the code must state "evidence is built from train patterns only; stop/network ids are join keys, never features". (to do at implementation)

## Open questions & assumptions

Reviewer questions:
- Q1 Is keying train-pattern evidence by `stop_id` identity (equality join within a network, never parsing the string) acceptable under "do not use stop_id strings as a prediction signal"? The description itself says stop identifiers are scoped to a network and invites learning from other patterns that contain the same stop. Plan: allowed; if not, the evidence family goes, with an expected cost of roughly 0.05-0.15 tau (UNVERIFIED).
- Q2 `stop_coordinates.csv` includes eval-only stops. We read only the case's own stops for per-case features and fit network-level geometric statistics on train-case stops only. Is that acceptable, or may network-wide geometry (for example corridor density from all listed stops) be used? Plan assumes the conservative reading.
- Q3 Are fixed architecture/epoch/LR constants chosen a priori and profiled locally for runtime (not tuned on a search) acceptable without an in-script HPO?
- Q4 Is deriving route groups from train stop-set overlap and using them for leave-route-out features acceptable? (It uses train data only.)

Assumptions where the description is silent:
- A1 Runtime: no limit stated; assume CLAUDE.md 45 min target, 1 h ceiling, A10G, no preemption.
- A2 No pretrained model or external data of any kind; training from scratch is compliant.
- A3 The 0.092 step is pre-noise (D3 to check); if post-noise, the noise description is inconsistent and geometry is more informative than assumed.
- A4 The "dropping one stop scores 0.760" example is an average over n (formula gives 0.800 at n=20); irrelevant to us.
- A5 Reflection and reversal symmetries are not assumed valid.
- A6 The description's coverage statistics (51.5%, 37.2%, 57.8%) are used only as calibration targets for our own train-only grouping, not as inputs to the model.

What could not be verified in this run: every data-dependent claim (route counts, group structure, noise reading, evidence coverage by network, baseline reproduction, runtime per epoch, any score). The plan must be revisited after the D1-D14 diagnostics; in particular, if D6 cannot be matched by any grouping threshold, or D3 shows that the successor is almost never within a modest neighbour rank, the candidate and decode design changes.
