# U18 plan: security-control related-pair recovery on boards

Status of evidence: NO dataset was available when this plan was written. Everything under "Data findings" is a diagnostic to run plus an UNVERIFIED hypothesis. Nothing below has been measured. Sources read: the eris-strategist agent file, CLAUDE.md, and the challenge text (tasks/user/U18_security_control_relations.md). Nothing else was opened, no internet.

## Contract & decision unit

**One valid answer.** For each test `item_id` (board) one row `item_id,edges`, where `edges` is a JSON list of `[i,j]` integer pairs over that board's local control indices (0 .. n_b-1). Format example from the text: `board_0170,"[[0, 3], [3, 7], [1, 7]]"`; `[]` is valid. Row order, ids and row count must equal `sample_submission.csv`.

**Invalid vs merely low-scoring.**
- Invalid (assume the whole file may be rejected): wrong columns/order, missing or extra board, non-JSON string, non-integer or out-of-range index, self-pair [i,i] (treat as invalid to be safe), NaN/empty cell. The writer will emit canonical pairs `[min,max]`, deduplicated, sorted, `json.dumps` with ints.
- Valid but low-scoring: empty list, all-pairs list, spurious pairs.

**Metric (as described).** Per board, F1 between predicted and true UNORDERED pair sets: P = |pred ∩ true| / |pred|, R = |pred ∩ true| / |true|, F1 = 2PR/(P+R) = 2|pred ∩ true| / (|pred| + |true|). Score = unweighted mean over boards (hierarchical: within-board F1, then mean across boards, so a board with 3 true edges weighs the same as one with 30). Partial credit is automatic. Not rank-based and calibration-sensitive: the number of predicted pairs k_b per board matters as much as the ordering. "Everything related" and "nothing related" are both explicitly weak. Edge cases the text does not define (unverified, assumed): pred empty and true nonempty gives 0; both empty gives 1 (probably never occurs, check in train that every board has >= 1 true edge).

**Independent unit.** The board is the prediction unit. The statistical unit that carries information is the CONTROL (and the relation between two controls), not the board: the split is control-disjoint, so independent units for validation are groups of boards that share controls (union-find, see Validation design). Rows inside a board are NOT independent: all pairs share the same n controls, and boards that share a control share its relations.

**Pipeline stages (diagnose separately).**
1. Candidate coverage: every board is small, so the candidate pool is ALL C(n_b,2) pairs. Oracle recall = 1.0 by construction (verify that all true edges satisfy i != j and i,j < n).
2. Scoring: p_ij = calibrated probability that pair (i,j) is a recorded relation, with node-role information (cluster member vs lexical trap).
3. Decoding: choose the pair set per board that maximises expected F1 given the p_ij (Metric-aware training & decode).

Diagnostics per stage: (a) ranking quality given reachability (per-board pair AUC / average precision of p_ij, and top-m precision using the TRUE m), (b) decode quality given gold scores (must give F1 = 1.0), (c) decode quality given model scores, with the true m as an oracle upper bound for the k-selection part.

## Compliance regime

**Domain.** Description silent on the label. Treated as NLP (text pair / link prediction over set-valued inputs), most plausibly "NLP" or "Fine-tuning". Assumption: pretrained HF encoders are allowed (CLAUDE.md default; the description does not ban them, and bans only external catalogs). Under the stricter "Fine-tuning" reading, a genuine fine-tune (LP-FT rung with logged parameter change) must be load-bearing. Under a "From scratch" reading all pretrained weights are banned; a from-scratch variant is specified under Recommended approach (Variant S). Reviewer question Q1 covers this.

**Explicit bans in the description (hard constraints).**
1. No external catalog, control-identifier list, or compliance mapping. Includes any NIST-style or framework knowledge brought in as data, rules, lookup tables, family lists, keyword-to-family maps, or hand-written "control A depends on control B" tables.
2. Do not re-identify controls against an outside catalog. Concretely: no prompting a generative LLM to guess control IDs/names/relations from its memory, no regex that reconstructs identifiers from residual text, no hand-built family classifier from outside knowledge. A learned notion of "similar topic" from train text is fine. (Residual unmasked identifiers, if the diagnostic finds any, are NOT to be used; they would be a way of naming relata.)
3. Solve from the supplied masked control text alone: only `title`, `text` per control (and the board's own set structure). Do not use `item_id` order/number as a feature.
4. Mask token `[CONTROL]` is part of the prose; using its count or position as a within-row text feature is allowed (it is text, not an identifier), but never try to fill it in.
5. "Predict a pair only for a genuine relationship": no padding output to hedge F1. Decode chooses k by expected F1 only.
6. Control-disjoint task: test controls never occur in train boards. Any train-control memorisation (identity features, per-control degree priors, lookup of nearest train control's neighbours) has zero transfer and must not be in the validation either (it would inflate CV).

**Inherited bans (CLAUDE.md).** No test-set statistics of any kind (no cross-board normalisation, no whole-test vocabularies/PCA/clustering, no cross-board consistency or transitive closure over test boards even though the same test control may appear in several test boards), no pseudo-labels, no synthetic training data (see rejected augmentations), no external data, no time-based control flow, no environment-dependent fallbacks, fixed seeds, source < 512 KB, HF/timm weights only, determinism.

**Where the description is silent, and what I assume.**
- Compute: assume one A10G 24 GB, and CLAUDE.md's <= 50 min target (<= 1 h worst case) since no limit is given.
- Runtime: no limit stated. Plan for <= 40 min estimated, >= 30% headroom.
- Sizes (boards, controls per board, text length): not stated. Assumed for budgeting only: train 500 to 3000 boards, 8 to 30 controls per board, 150 to 400 tokens per control, test 100 to 500 boards. All budget lines carry a scaling rule so they can be re-derived once real sizes are seen.
- Domain label: unknown (NLP vs fine-tuning vs from-scratch), see above.
- Whether pretrained weights are allowed: assumed yes.
- Board metadata: none besides `item_id`.

## Data findings

None verified. For each diagnostic: the exact train computation, then the hypothesis (UNVERIFIED, a guess from the description, to be confirmed or killed).

**D1. Schema and sizes.** `train.csv`: rows, columns, dtype of `controls`/`edges` (JSON strings), parse failures. n_b distribution (min/median/p90/max) for train; for test only n_b, row count, id format, text length statistics for budgeting (not distributions of content). Hypothesis H1: n_b in roughly 8 to 30; ids look like `board_0170`.

**D2. Edge statistics per board.** m_b (true edges), density m_b / C(n_b,2), distribution of m_b, share of boards with m_b = 0, self-pairs, duplicates, out-of-range, whether `[i,j]` and `[j,i]` both appear. Hypothesis H2: every board has m_b >= 1; density 5% to 30%; labels already canonical or need dedup.

**D3. Graph structure per board.** Degrees, isolated nodes (degree 0), number of non-trivial connected components, size of the largest one, number of triangles, clustering coefficient. Hypotheses H3: non-isolated nodes form ONE connected component in most boards (a walk over a connected neighbourhood); isolated nodes are the lexical traps; trap share 25% to 50% of nodes; trap-trap edges are absent. If several non-trivial components exist or trap-trap edges are found, the node-role factorisation below changes (use pair head without the role gate).

**D4. Information ceiling of the cheap baselines (train, board-wise F1).** (a) empty = 0 and all-pairs F1 = 2m/(P+m) averaged over boards, (b) predict only the top-k by cosine of a frozen encoder with k = true m (oracle k), (c) "oracle node set": all pairs among the true non-isolated nodes, (d) oracle node set + top-m by a pair scorer. Hypotheses H4: all-pairs F1 around 0.2 to 0.4 (this is the floor to beat); raw cosine top-m F1 is only mildly above, or below, all-pairs; (c) is far above (a) if clusters are dense, which would show node-role detection is the main lever.

**D5. The "similarity trap" claim.** For each true edge compute its within-board cosine percentile rank among the 2(n-1) candidate partners of its two endpoints; compare with non-edges. Per-board AUC of cosine for edge vs non-edge using 2 to 3 frozen encoders (e.g. general-purpose sentence embedding models). Share of boards where the nearest neighbour of a node is a non-partner. Hypotheses H5: AUC of raw cosine is near 0.5 to 0.65; the nearest neighbour is mostly a non-partner; the AUC conditional on "both nodes in the cluster" is higher than unconditional (traps drag raw similarity down). If AUC < 0.5, the sign flips for high-cosine pairs and the head must see board-relative cosine rank, which the design includes.

**D6. Node-role signal (trap vs cluster).** Node-level AUC for "degree > 0" from: mean cosine to the k nearest board-mates, max cosine to another board-mate, the size of the largest tight lexical clique the node sits in, token length, count of `[CONTROL]` mentions in its text, title length. Hypothesis H6: a tight lexical clique (seed's family-mates) exists in each board; its members except the seed are mostly traps; AUC 0.7 to 0.9 for degree>0 from "distance to the clique" features. If no such signal, the role head is dropped.

**D7. Control recurrence and group structure (decides validation).** Hash of normalised `title+text`; number of unique controls vs control instances; boards per control; union-find over boards that share any control; number and size of components; whether the same pair is labelled identically in every board where both occur (label consistency); whether a control's neighbour set differs across boards. Hypotheses H7: controls recur across train boards (boards are neighbourhood walks over a graph), possibly yielding one giant component. If giant: use control-fold restriction scheme (Validation design, scheme B). If many components: scheme A.

**D8. Within-train duplicates and near-duplicates.** Exact duplicates of text under different titles, near-duplicate pairs (cosine > 0.95) both inside and across boards; count of "enhancement-like" variants of the same base control (the same prose with one sentence added). Hypothesis H8: many near-duplicate siblings (same-family) exist; near-duplicates are almost never related to each other (they are the traps).

**D9. Text length and structure.** Token length of title, text (statement + guidance) under the candidate tokenizers; share above 256 / 384 / 512 tokens; whether text concatenates statement and guidance with a recognisable separator; count of `[CONTROL]` per text; residual identifier-like strings (a regex count over train prose, DIAGNOSTIC ONLY, never exploited). Hypotheses H9: 20% to 40% of controls exceed 256 tokens; `[CONTROL]` appears in many guidance passages; residual unmasked ids are rare.

**D10. Row-order / id leakage.** Correlation of node position with degree; correlation of `item_id` number with n_b, m_b, trap count. Hypothesis H10: none (shuffled), and `item_id` carries no signal; if it does, do NOT use it.

**D11. Reciprocity artefact.** The text says catalog lists are only partly reciprocal; labels are undirected. Check whether one-directional curation shows up as a feature (it cannot, directions are not given). Expect nothing.

**D12. Label balance.** Pair-level positive rate overall and by n_b; per-board positive rate; to set the base-rate bias init and class weighting.

**D13. Identical inputs, different labels.** Same control pair appearing in two boards with different labels (ambiguity floor); if it occurs, label noise ceiling.

**D14. Information ceiling of a text-only oracle.** Fit a very simple logistic regression on cosine + board-relative rank under grouped CV, and record the F1 with oracle k; this defines the "naive best" before the learned model.

## Validation design

**How the test was made (from the description, train structure only).** Controls were assigned individually, by a fixed random seed, to a train pool or a test pool; boards were assembled only inside a pool. So test boards contain controls never seen in train (control-disjoint), and train boards share controls only with other train boards. Axis of shift: new controls (new content and new relations), not new boards; no time axis; same board-construction generator.

**Scheme A (preferred when D7 shows many components).** GroupKFold(5) over boards where groups are union-find components over shared control hashes. Repeat with 3 different seeds of the group-to-fold assignment. Report mean and std over folds and seeds. Direction of bias: roughly neutral to slightly pessimistic (training shrinks by 1/5 relative to the real final fit, which uses 100% of train; the real test has full-pool training).

**Scheme B (giant component fallback).** Randomly assign each unique control to one of 5 folds (mirrors "split over individual controls, fixed seed"). For fold f: training boards are the train boards restricted to controls NOT in fold f (controls of fold f deleted, edges incident to them dropped); validation boards are the boards restricted to fold-f controls only. Restricted boards are smaller (about 1/5 of the nodes), shifting trap/cluster composition and lowering the number of candidate pairs. Direction of bias: optimistic on F1 magnitude (fewer distractors) but unreliable for k-selection; use it ONLY for relative comparisons and for fitting calibration constants with n-dependence explicit. Also hold out pure whole-board validation when D7 permits a subset of independent components (even a few large components can be used as a final sanity fold).

**Sanity fold.** One fold (or components covering ~15% of boards) is never used for any selection (HPO, calibration slope, k penalty); used once before the final run to catch CV overfitting.

**Metric implementation and unit tests (step 1 of roadmap).** Re-implement per-board F1 and the mean exactly. Tests: perfect predictions = 1; empty pred vs nonempty truth = 0; reversed pair order [j,i] equals [i,j]; duplicated pairs deduplicate; all-pairs equals 2m/(P+m); constant empty = 0; "base-rate" prediction (random k edges chosen from the all-pairs list) approximates the analytic expectation; both-empty case documented. Also test the writer: JSON round trip, canonical ordering, row count and id order against `sample_submission.csv`.

**Nested checks.** Every post-hoc selection gets its own held-out check: (i) regularisation/dropout/epoch grid (<= 12 configs) chosen on inner folds of the training part of each outer fold; (ii) calibration (a, b, c) and the expected-F1 denominator multiplier fitted on OOF of OTHER folds and applied to the held-out fold (cross-fitted); (iii) stacker weights likewise. Report the cross-fitted score, not the in-fold one.

**Noise and acceptance.** Compare designs paired on the same folds and seeds; accept a change only if the mean per-board F1 gain exceeds one standard error across fold means and is positive in most folds and seeds. Per-board F1 SD guess ~0.2, so with ~300 validation boards the standard error is ~0.012; gains below ~0.015 are not accepted without repetition. Expect the private score to land below the proxy for unknown reasons (new controls differ); proxy directions: scheme A mildly pessimistic (smaller train), scheme B optimistic in level but fine for ranking designs.

**Floors for yardsticks under the SAME folds.** (1) empty, (2) all-pairs, (3) frozen cosine top-k with oracle k, (4) frozen cosine + calibrated global-k. The fine-tuned system must beat (3) by a margin exceeding noise.

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Few unique controls (hypothesis: hundreds to a few thousand): the model memorises control identity and its neighbours. Mitigation: control-grouped CV, small head (~1 to 2 M parameters), dropout/weight decay, fixed short schedule, no identity-like features, token dropout on text, ladder rung chosen by grouped CV.
2. Boards share controls, so random-board CV would be badly optimistic. Mitigation: groups by union-find over control hashes (scheme A) or control-level folds (scheme B).
3. Selection on the reported OOF (calibration, k penalty, rung choice, blend weights). Mitigation: cross-fitted/nested checks, <= 12-config grid, 3 to 4 scalar decode constants.
4. Engineered negatives: traps are generated as same-family unrelated siblings; the model may learn "high lexical similarity to many board-mates means trap" which is partly generator-specific. If the real test boards are built by the same generator (stated), this transfers; the risk is on the "exploits data generation" reviewer flag (Q3), not on the score.
5. Sibling leakage (board-context features); see Compliance audit.
6. Fine-tuning top layers on ~k thousand unique controls memorises: mitigation is LP-FT (warm-start head, LR <= 2e-5, <= 2 epochs, <= top-2 layers), tracked held-out score each epoch.

**Underfit risks and mitigations.**
1. Frozen embeddings capture topical similarity, not functional dependency ("what this control needs"): expect a ceiling. Mitigation: two diverse frozen encoders (different training objectives), the set-level context model, LP-FT top layers, and a cross-encoder member in the roadmap.
2. Truncation: guidance prose may exceed the context. Mitigation: choose L from D9 to cover >= 95% of tokens (title + statement first, guidance tail truncated), or pool chunk embeddings by mean when D9 shows heavy tails (verify gain under CV).
3. Too small a backbone: use the largest encoder that fits in the budget (large-class embeddings) before adding machinery.
4. Loss/metric mismatch: BCE on pairs while the metric is per-board F1 with k selected. Mitigation: per-board weighting + calibration + expected-F1 decode.
5. Preprocessing loss: lower-casing/stripping `[CONTROL]` would throw away the signal that a control references others; keep the mask token as plain text.

## Recommended approach (primary + fallback)

### Primary: board set-model on pretrained control representations (LP-FT top layers), factorised node-role x pair-evidence, expected-F1 decode

Why this: relations are functional and cross-family, so (i) the per-control representation must carry function (what the control does and requires), (ii) the model must see the whole board because the signal "this node sits in the seed's tight lexical clique and is a trap" is only visible in context, and (iii) pair scoring must be symmetric. A set model over per-control vectors costs O(n) encoder passes per board rather than O(n^2), which fits the budget (a pair cross-encoder over all pairs does not).

**Representation.** Per control: text = `title + " [SEP] " + text` (the `[CONTROL]` mask kept verbatim), max length L from D9 (default 320 tokens). Two frozen strong pretrained encoders with different objectives (e.g. one retrieval-trained large embedding model and one other-family large embedding model, revisions pinned by commit hash at implementation time). Pooling: CLS plus mean over tokens, concatenated (flip-averaging does not apply to text). Compute unique-control embeddings once per encoder: train controls deduplicated by hash (train-side dedupe is fine; test controls are encoded board by board with no cross-board dedup, to avoid any appearance of test aggregation).

**Board model (trained, few parameters).**
- Project each control vector to d = 256 (LayerNorm, dropout 0.2 to 0.3).
- 2-layer Transformer encoder over the n control vectors of ONE board, no positional encoding (permutation-equivariant), 4 heads, pre-norm, dropout 0.1, plus broadcast of scale variables (log n_b) into each token.
- Node-role head: z_i = sigmoid(w . h_i + b), trained with BCE on the node target [degree_i > 0] (auxiliary and also part of the pair marginal).
- Pair head, symmetric by construction: q_ij = sigmoid(MLP([h_i + h_j, h_i * h_j, |h_i - h_j|, f_ij])) where f_ij = [raw cosine per encoder, board-relative rank and z-score of the cosine for i (among its board-mates) and for j, token-length ratio]. Within-board relative transforms use only that board's own controls.
- Pair marginal: p_ij = z_i * z_j * q_ij (exact marginal under role-independence; an explicit latent-structure potential: cluster membership is the unobserved quantity, edge evidence is conditional on both endpoints being in the cluster). Pair loss: weighted BCE on p_ij against the edge label, plus BCE on z_i. Initialise output biases at the base rates (D12).
- Weights: per-board normalisation so each board contributes equally, positives weighted 1 / m_b mildly (tested against uniform under grouped CV).
- Capacity ladder (F2.1): rung 0 = frozen features + this head (cheap, ~1 to 2 M parameters); rung 1 = LP-FT: unfreeze the top 2 transformer layers (and pooling) of each encoder, warm start from the rung-0 head, LR <= 2e-5 for the encoder, 1e-3 for the head, 1 to 2 epochs, fixed. To keep this affordable, the lower layers' token-level hidden states of each unique train control are computed once and cached; the top layers are fine-tuned on the cache (cost about top-2/24 of a full pass). Ship rung 1 if its grouped-CV gain over rung 0 exceeds noise; otherwise ship rung 0 plus the LP-FT run as logged evidence (norm of parameter change and CV delta recorded). Under a strict "Fine-tuning" label ship rung 1 whenever it is within noise of rung 0 (Q1).
- Regularisation per output chosen by the exact metric under grouped CV with a wide grid (weight decay, dropout, role-gate temperature), at most 12 configs, fixed schedule length, EMA of weights, no validation-triggered stopping.
- Refit on 100% of train for the shipped test predictions with a fixed epoch count, 3 seeds averaged; fold models only produce OOF predictions for calibration and decode constants.

**Compliance angle:** trained components are load-bearing (the role gate, the symmetric pair head, the set transformer, the LP-FT layers). Strip the ML: removing them leaves only cosine top-k, which the plan expects to be near the all-pairs floor.

### Fallback (also second member): pair cross-encoder fine-tune with board-relative post-features

DeBERTa-v3 (base, or large if budget allows) fine-tuned as a symmetric pair classifier on `[title_i text_i] [SEP] [title_j text_j]` with random order swap during training and swap-averaged at inference (symmetry). Training pairs: all positives plus hard negatives mined from the OUT-OF-FOLD scores of the primary (top-scoring non-edges per board) and random non-edges, with a fixed negative:positive ratio (e.g. 4:1 for the first epoch, refreshed once from OOF scores). Inference only on pairs that survive a candidate filter from the primary (top fixed fraction of pairs per board, with oracle recall of the filter measured in CV). Its probabilities enter a tiny stacker (logistic on [logit p_primary, logit p_cross, board-relative rank of each]) fitted on OOF (cross-fitted). Fully standard, clearly compliant (a trained pair model with genuine fine-tuning), runs in the budget only if the pair counts are modest. If the primary fails validation or gets a reviewer objection about set-context, this is the shipped design (with decode-only board context).

### Variant S (only if the label is "from scratch")
No pretrained weights. Train a tokenizer on train prose (or word-level vocabulary fit on TRAIN only), a small transformer/bi-GRU control encoder trained jointly with the same board set-model and pair head (contrastive plus pair BCE), same decode. Expected to be materially weaker (vocabulary of security prose is small but relations are semantic); note the loss as a cost.

## Rejected options

1. Cosine/TF-IDF/BM25 similarity thresholds (inference-only, rule-based, and the description states that the most similar control is usually an unrelated sibling): banned as the core, anti-informative as a lone score.
2. Zero-shot generative LLM prompting for "which controls are related": inference-only, memorised-catalog risk (violates "do not re-identify against an outside catalog"), not training.
3. Any lookup of external catalogs, identifier lists, family tables, compliance mappings: explicitly banned.
4. Per-control identity / train-neighbour lookup features (nearest train control's partners): useless under the control-disjoint split and inflating in CV.
5. Cross-board test aggregation (consistency of the same test control across test boards, graph transitivity across boards, global normalisation over test boards): banned (whole-test aggregation).
6. Full fine-tune of a large encoder end-to-end on pairs: O(n^2) passes per board, memorises controls on few unique controls (capacity ladder), over budget.
7. Pure GBDT/LambdaRank on engineered cosine features as the core: "hand-engineered features + off-the-shelf LambdaMART alone is not enough" (CLAUDE.md 6.4) and weak on functional relations. Allowed only as a minor stacker input if it helps beyond noise.
8. Board-mixing / synthetic board augmentation (combining controls from different boards): cross-board pairs would have unrecorded relations (label noise) and amounts to fabricated entities; not used. Only mild real-unit augmentation: random board-order permutation (a no-op for an equivariant model), token dropout on text, dropping a random subset of board controls together with their incident edges (tested under CV; remove if it hurts).
9. Directed-edge modelling: the target and metric are undirected and only partially reciprocal; direction would add parameters without reward.
10. Sampling-based joint decoders and latent mixtures in the primary: gated to later roadmap steps (primary-design gate).
11. Pseudo-labelling on test boards or test-time adaptation: banned.

## Fixed work plan & runtime budget

All counts below are fixed constants (no wall-clock branches, no `cuda.is_available`, `device="cuda"`, fixed workers). Times are ESTIMATES for an A10G under the assumed sizes (train 1500 boards x 15 controls = ~22k control instances, ~2.5k unique; test 300 boards x 15 = 4.5k controls), unverified. Scaling rules are given so they can be re-profiled; the real counts must be profiled locally and hard-coded.

| Stage | Fixed plan | Est. time (A10G) | Scaling rule |
|---|---|---|---|
| S0 load, parse, validate input, build hashes/groups | single pass | 1 min | linear in rows |
| S1 frozen embeddings, 2 large encoders, fp16, L=320, batch 64; also cache hidden states of layer (last-2) for unique train controls | 2.5k unique train + 4.5k test sequences | 4 min | seqs / ~200 per second per encoder |
| S2 rung-0 grouped CV: 5 folds x 2 split seeds, head training ~25 epochs each on cached vectors, ~12-config grid on inner folds | small head, cached | 8 min | folds x seeds x configs; head is tiny |
| S3 rung-1 (LP-FT top-2 layers) 5 folds x 1 seed x 2 epochs on the cached hidden states | cached top layers | 8 min | proportional to unique controls x 2 layers |
| S4 OOF calibration + decode constants (a, b, c, lambda) cross-fitted | closed-form/grid, 4 scalars | 1 min | trivial |
| S5 final refit on 100% train, 3 seeds, fixed epochs | rung chosen by S3 gate | 4 min | 3 x one fit |
| S6 test inference per board (set model + decode), swap-averaged | 300 boards | 1 min | linear |
| S7 write + validate + re-read | | < 1 min | |
| Optional S8 cross-encoder member (roadmap step 5 only): train 1 epoch on ~60k pairs (L_pair = 2 x 192 tokens) in the same fold structure for OOF; test inference on filtered pairs | +15 to 20 min if included | only if gain > noise |

Estimated total without S8: ~28 min; with S8: ~45 min (then reduce folds in S3). Target <= 40 min, >= 30% headroom against 1 h; memory estimate: embeddings 7k x 2 encoders x 2048 dims fp32 < 0.1 GB; hidden-state cache 2.5k x 320 x 1024 x 2 bytes ~ 1.6 GB per encoder (GPU or pinned host); training peak < 12 GB. If the real unique-control count is 10x larger the cache (16 GB) exceeds the plan: re-plan to top-1 layer or on-the-fly forward (stated as the scale-up path, decided offline now, not at runtime).

**Fixed constants to hard-code.** N_FOLDS=5, SPLIT_SEEDS=2, GRID<=12 configs, EPOCHS (rung 0) fixed after CV in-script by a plain argmax over fixed grid (not time), SEEDS=3, L, d=256, layers=2, heads=4, MC samples (only if the later sampling decoder is added) 256 with a fixed seed. The in-script grid selects among these; nothing is pasted from offline tuning.

## Metric-aware training & decode

**(i) Back-solve the metric.** Per-board F1 = 2 TP / (k + m_b). For a board the best predicted set under a probability model is the set maximising E[F1]. With independent pair probabilities (a ratio of expectations approximation), the best set of size k is the top-k by p, and the best k solves k* = argmax_k 2 S_k / (k + M), where S_k = sum of the top-k p_ij and M = sum over all pairs of p_ij (expected number of true edges). This is the first-order decode; no search over thresholds that could reach degenerate corners: the search is over k in [1, K_max] with K_max = n_b(n_b-1)/2 (and k = 0 only if every p is below a tiny floor; the train check on D2 settles whether to forbid k=0).

**(ii) Hierarchical averaging.** Training loss weights each board equally (per-board mean pair loss times 1/m_b-weighted positives tested). Nothing else is needed because the metric is a plain mean over boards.

**(iii) Calibration.** The expected-F1 decode is calibration-sensitive. Fit a tiny recalibration on OOF: logit p' = a * logit p + b + c * log n_b (3 scalars) by logistic regression, plus a multiplier lambda on M (to absorb a bias in the expected number of true edges), all cross-fitted (fit on other folds, apply to the held-out fold), then the report is the cross-fitted score.

**(iv) Composite structure at decode.** Structural constraints enter at decode combined with model evidence (a constraint search alone is not a predictor): (a) each predicted-cluster node should have at least one incident predicted edge; (b) isolated nodes by role z_i < tau get no edges. These are tested ONLY after the first-order decode is measured; their gain must exceed noise. The explicit joint role+edge sampling decoder (draw roles from z, edges from q, choose the set maximising Monte-Carlo expected F1, 256 draws, fixed seed) is a roadmap step 4b, not part of the lean primary.

**(v) Auxiliary losses that follow from the structure.** Node-role BCE (degree>0), any-edge per node, and an edge-count regression target (m_b / C(n_b,2)) at board level are cheap auxiliary heads; keep only if CV shows a gain.

**(vi) Oracle checks before modelling (lesson 15).** Feed gold scores (p = true labels) through the decode + writer + metric: the decoder must return exactly the gold edge set on every train board (F1 = 1.0), which proves canonical format, indexing and runtime. Measure the decode runtime. Report the frozen cosine probe under the same folds as the honest floor.

## Structural signals

Each invariant implied by the description, and how it is used. Mark each as hypothesis until the D-diagnostics confirm.
1. Permutation equivariance: boards are shuffled, position carries no signal. Use a position-free set transformer; random reshuffle each epoch is a no-op sanity check; assert that predictions are equivariant under permutation of a validation board.
2. Pair symmetry: relatedness is undirected. Symmetric pair head (sum, product, absolute difference); no letter-order inputs.
3. Board = connected related cluster + same-family traps with no relation to the cluster: nodes in the cluster have degree >= 1, traps degree 0 (D3). Encoded as a node-role head and the factorised marginal p_ij = z_i z_j q_ij (the latent role is explicit, per lesson 2). The head must also learn the generator's shape from scale variables (log n_b), not from hand-coded constants.
4. Traps are lexical siblings of the seed's family and form a tight clique (D5, D6): board-relative cosine rank and clique membership are features in the pair/role heads, learned weights (can be negative for pairs that are very similar).
5. Cross-family bias: relations link dissimilar prose; thus raw cosine is a weak or reversed signal, and per-encoder cosine enters only as one feature among learned ones.
6. Possible transitivity/triangle closure inside the cluster (D3): captured by the set transformer; explicit triangle potentials are a later decode step.
7. Hub controls (some controls relate to many; D3 degree distribution): captured by node features (functional text); no train-control identity used.
8. `[CONTROL]` mask count and context (the prose mentions other controls): a within-row text feature feeding the node-role head.
9. Control-disjointness: exploit it for validation design only. In training, near-duplicate controls (same family) across boards belong to the same group.
10. Partial reciprocity: undirected union labels, nothing to exploit.
11. Augmentation: only real-unit operations (token dropout, control dropping with incident edges, order shuffle). Symmetries needing conditioned transformations: none (no geometry-tied inputs).
12. Check claimed symmetries with train diagnostics: confirm that in training labels `[i,j]` is not systematically ordered (i<j) so that order carries no signal; check D10.

## Experiment roadmap

Each step has a stop criterion; do not start a later step until the earlier ones hold.
1. **Contract, metric, validation (stop: unit tests green).** Parser, canonical writer, F1 metric with unit tests (perfect, reversed, empty, all-pairs, constant, base-rate), gold-score decode returns F1 = 1.0 on all train boards, D1 to D14 diagnostics, folds chosen (scheme A vs B) per D7. Record the floors (empty, all-pairs, cosine with oracle k).
2. **Strongest cheap baseline, end to end and valid (stop: valid CSV, CV logged).** Frozen encoders + rung-0 board head, global-k decode; valid submission written; record CV mean +- std over 5 folds x 2 seeds. Expected: above all-pairs floor if the plan's hypothesis holds; if not, check D5/D6 before going on.
3. **Representation and structure (stop: gain > noise or revert).** Ablate in this order, paired on the same folds: (a) pair-only head without board context (the "context-free" reading of the sibling-leakage question, Q2), (b) + set transformer, (c) + node-role factorisation, (d) + second encoder, (e) + board-relative features, (f) LP-FT top-2 layers. Keep each only if it beats noise.
4. **Metric-aware loss and decode (stop: cross-fitted gain > noise).** Per-board weighting, calibration (a, b, c), expected-F1 k-selection with lambda; then structural decode constraints (4a/4b). Report decode quality with the TRUE m as the upper bound for k-selection.
5. **Diversity (stop: member within about 0.02 F1 of the primary alone and blend gain > noise).** Cross-encoder pair member (fallback design) on OOF-mined hard negatives + stacker on OOF; rank/z-score blend, not raw probability averaging; per-family score alone first.
6. **Bounded in-script HPO (stop: nested check).** <= 12 configs, fixed trials, seeded; its own outer-fold check; keep plateau centre not sharp optimum.
7. **Final fixed-plan run, twice, diff.** Full script from clean `working/`, exit 0, validator green, the two submission files identical or near-identical (edge-set Jaccard per board; investigate big deviations), runtime profile versus the estimate table above, then upload. Use credits only for baseline (step 2), best single (step 3/4), ensemble (step 5) and final (step 7); free CSV check in between.

When two or three independent solvers would plausibly converge on a technique (here: cross-encoder pair classification and set/context features are the likeliest convergence), test it before proxy-driven tweaks.

## Compliance audit

CLAUDE.md section 7 checklist against the plan:
- Test file read only for per-board inference: yes by design; decoding and normalisations use only the board's own controls and train-fit parameters; the test boards are never pooled; no cross-board dedup or consistency; no vocabularies or scalers fit on test.
- Time in conditions: none; time is logged only.
- `cuda.is_available`, cpu_count, try/except import fallbacks: none planned; `device="cuda"`.
- Hard-coded constants tuned offline: none permitted; every tuned constant (rung, grid choices, calibration a,b,c, lambda, k selection) is found in-script by train-only search with a fixed work plan; fixed design constants (d=256, layers=2, L from D9 rule) are architecture choices, to be reported.
- External data / synthetic data / outside weights / non-allowed libraries: HF-hosted pretrained weights only (revisions pinned), no catalog, no mapping, no identifiers; no synthetic boards.
- Strip-the-ML test: removing the trained board/pair/role heads and LP-FT layers leaves cosine top-k by frozen encoders, predicted to be near the all-pairs floor (D4) and far below the full system; the ML is load-bearing. The decode is a function of learned probabilities (a pure constraint search is not a predictor).
- Whole-test aggregation: none. A board's prediction depends on its own controls and a train-fit model. This also covers board-relative transforms (ranks/z-scores among the board's own controls) and the role head.
- Related-row (sibling) leakage: a board is the input row, and the board-mates are part of the same input; context features cross-reference mates while the target is a relation among mates (suspect per the agent rules). Mitigation: it is within-row, not cross-row, and the reading compliant under both interpretations is shipped as the pair-only head + decode-only context if the reviewer objects (Q2), with its measured cost known from roadmap step 3a. Leave-own-control-out label statistics are NOT used at all (no per-control priors).
- Hard-coded constants derivable in-script from train: yes (see above).
- Inference-only / frozen-features risk: rung-0 is frozen features + a trained head (grey in fine-tuning categories); rung-1 LP-FT top layers (logged parameter-change norm and CV delta) provides the genuine training layer; the cross-encoder fallback is a genuine fine-tune.
- Source < 512 KB, plain readable, comments on reasoning, seeds fixed, num_workers fixed: planned.
- Ban-language audit: no external catalog (checked), no identifier list (checked), no re-identification (no LLM generation, no regex to reconstruct IDs; residual-ID diagnostic is read-only and unused), solves from masked text only (title, text).

## Open questions & assumptions

**Reviewer questions.**
Q1. Domain label and pretrained weights: is this an NLP or a Fine-tuning challenge, and are pretrained text encoders (HF, pinned revisions) allowed? Plan under each reading: (a) allowed + fine-tuning label: ship LP-FT rung 1 as the compliance layer (primary); (b) allowed + NLP: ship the better of rung 0/1 by CV; (c) from scratch: Variant S, expected materially lower.
Q2. Is board-context (set-level) modelling acceptable when the target is a relation between board-mates, given the board is the input unit? Under reading "yes": primary as specified; under reading "no": pair-only head (symmetric) + decode-only use of the board's own probabilities, with the measured cost from roadmap step 3a. The conservative reading is compliant under both; its cost will be stated from CV.
Q3. Is learning the board generator's shape (traps are same-family, the cluster is connected) from TRAIN labels, via scale variables and node-role supervision, acceptable? Plan: yes with the learned (not hand-coded) form; flag once as an "exploits data generation" risk.
Q4. Is a pretrained embedding model that may have memorised public security catalogs acceptable under "do not re-identify controls against an outside catalog"? The plan makes no attempt at identification (no generation, no ID recall); an encoder's implicit knowledge is outside our control. Alternative if rejected: smaller/general-domain encoders; costs measured by CV.
Q5. Empty-versus-empty F1 convention and whether any test board has no true edges (affects whether k=0 is allowed in the decode).

**Assumptions (silent in description).**
- A1: Compute is one A10G; runtime budget <= 50 min per CLAUDE.md; plan estimate ~28 min (without S8).
- A2: Sizes are as in the budget table; budget re-derived once real sizes are profiled (offline) and hard-coded before submission.
- A3: Domain is NLP/fine-tuning with pretrained weights allowed.
- A4: Every board has at least one true edge; the cluster is connected; trap nodes are isolated (all UNVERIFIED, D2/D3).
- A5: The same control may appear in several train boards (D7); if not, scheme A reduces to GroupKFold over boards.
- A6: Both `train.csv` `controls` and `edges` are well-formed JSON, `edges` canonical or at least valid index pairs (D1/D2).
- A7: The test boards are generated by the same procedure as train boards (stated), so board-level structure (trap ratio, n_b, density) transfers.

**What could not be verified here.** All data-dependent statements: sizes, text lengths, edge density, group structure, whether raw similarity is anti-informative, whether node-role detection works, the F1 of every baseline, and all runtime numbers. **Expected private score range (ESTIMATE only, no data, no run):** mean edge-F1 roughly 0.30 to 0.55 for the primary, with a floor near the all-pairs baseline (guess 0.2 to 0.4); the proxy CV may be several points above or below because the proxy removes controls (scheme B optimistic, scheme A mildly pessimistic). No score is promised.
