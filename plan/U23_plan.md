# U23 plan: ranking backports by expected human-review workload (NDCG@20)

Status of evidence: NO dataset files were available to the strategist. Everything under "Data findings" is a diagnostic to run and a hypothesis, marked **[UNVERIFIED]**. Every statement about the description is quoted from `/home/user/Claude-/tasks/user/U23_backport_review_ranking.md` (the file may start mid-description at "Overview"; the compute/runtime section, if it existed above that, was not seen: see Open questions).

## Contract & decision unit

- **One valid answer:** `submission.csv` with columns `id,prediction` in that order, exactly 586 rows, every `id` from `test.csv` exactly once, `prediction` finite numeric. Only relative order matters (stable mergesort, so exact ties fall back to row order of the aligned frame; avoid ties by emitting continuous scores). The grader sorts both frames by id, so order of rows is not graded, but we keep the `sample_submission.csv` id order anyway.
- **Invalid vs merely low-scoring:** missing/duplicate/extra ids, non-finite or string predictions, wrong columns are invalid (dead last). Any ordering is valid.
- **Inputs:** `train.csv` (2,418 rows; columns `id`, `profile_text`; no target), `train_targets.csv` (`id`, `target` in {0,1,2}; joined by `id` only), `test.csv` (586 rows, projects absent from train), `sample_submission.csv`.
- **Metric (exact, from the description):** `NDCG@20` on ONE pooled list of all 586 test rows. Gain = `2**y - 1` -> quiet 0, light 1, active 3. DCG over the top 20 with `1/log2(rank+1)`; divided by the ideal DCG of the best 20 of the same list. Score 0.0 if no relevant row (not the case in holdout). Consequences:
  - Only the head of the list matters: roughly 20 of 586 rows decide the score. It is a tail-precision problem for actives (3x gain), with light rows as filler (1x) when actives run out. Quiet rows in the top 20 are pure loss; order below rank ~40 is irrelevant (weights < 0.19 each, and ties below are irrelevant).
  - Ranking by **expected gain** `E[gain|x] = P(y=1) + 3 P(y=2) = P(y>=1) + 2 P(y>=2)` maximises expected DCG for a fixed discount, so the target of learning is calibrated cumulative probabilities, then this scalar.
  - The ideal DCG is a function of the hidden labels only; it does not change our ranking.
  - Hierarchical/weighted averaging: none; a single list, single metric. The independent unit for **validation** is the project (group), not the row; the metric "slate" is the whole test pool.
- **Pipeline stages:** (1) parse profile string into features, (2) learned scorer(s) giving `p1=P(y>=1)`, `p2=P(y>=2)` per row, (3) decode = sort by `p1 + 2*p2` (no constraint set; no candidate-coverage stage, since every row is a candidate). Diagnose: how well features separate classes at all (probe), tail precision (P@20, NDCG@20 on fold slates), calibration of the two heads separately.

## Compliance regime

Domain: text-profile / small tabular ranking (NLP-labelled, non-image). The description neither says "fine-tuning" nor "from scratch". Treated as plain NLP/tabular: trained GBDT/linear/NN heads on parsed features are explicitly permitted ("any reproducible ranking, regression, NLP, text-profile, or multimodal model that trains only on the supplied public files"; "graded-relevance losses, pairwise ranking, calibration, ensembling").

Explicit bans in the description, treated as hard constraints, and the plan's handling:

| Ban (description) | Handling |
|---|---|
| Retrieve original PRs / reviews from a forge or any external service | No network at all except optional pinned HF weights for the optional encoder arm. Primary uses none. |
| Raw archive, raw repo URL, PR URL, PR number, author identity, **agent/human source indicator** as feature or lookup key | Profile text is said not to contain these. Diagnostic: list all field names / lexical-flag names in train vocabulary; any flag that explicitly encodes bot/agent/AI/human authorship or a source cohort is dropped (conservative). Reviewer question asked (see Open questions). |
| Row order, opaque identifier, opaque project hash as signal | `id` used only as join key and output key. Train rows are processed in a canonical, content-independent order; no feature or fold assignment derived from row position or id. Row-order/ID dependence is checked as a diagnostic only (not shipped) to prove it does not matter. If a project-hash column exists locally it is used only in dev scripts to test the quality of the content-derived groups, never in `solution.py` and never as a feature. |
| Synthetic examples or synthetic labels | None. No SMOTE, no mixup/blending of rows, no generated labels. Class re-weighting by sample weights is not synthetic data and is used only if CV-supported. |
| Do not interpret target as quality/importance; do not optimise a hidden/external queue | Nothing in the plan does; metric implemented exactly from the description. |
| "Train only from supplied public files", `train_targets.csv` only label source | Yes. |
| (CLAUDE.md 2.3 #5) no fitting on test | Vocabulary, parsers' band ordering, scalers, clusters, calibrators are all fit on train only; test is transformed and scored row by row. No rank/z-score normalisation across test, no test-based priors, no pseudo-labels. Model blending uses train-reference maps (OOF empirical CDFs), never test-wide ranks. |
| (CLAUDE.md 3) determinism | Fixed seeds, fixed rounds/epochs/trials, fixed thread count, no wall-clock branching, no `cuda.is_available()` or import fallbacks, no env-based recipe selection. |

Where the description is silent and what we assume:
- **Runtime/compute:** the supplied text states only "remain within the selected runtime budget" and mentions GPU encoders as allowed; no number appears in the part I could read. Assumed: CLAUDE.md contract, A10G 24 GB, worst case 1 h; plan targets <= ~30 min with the primary on CPU (see budget).
- **Model family / size caps:** none stated; assumed none.
- **Pretrained weights:** description permits "GPU-backed encoders or fine-tuned language models"; assumed HF-hosted pretrained encoders allowed (CLAUDE.md 1). Primary does not need them.
- **Is the challenge labelled fine-tuning (CLAUDE.md 6.7)?** Not in the text; assumed no. If it were, the encoder arm (see below) becomes mandatory-in-spirit, and the plan has it ready as step 5.
- **Group metadata for validation:** the description says "when legitimate group metadata is available during development"; the file description lists only `id` and `profile_text` for train, so assumed NO project column. Groups are derived from content.

Compliance status of the primary: clean. Strip-the-ML test: removing the trained models leaves only a parser; it cannot rank. Nothing in the pipeline is a hard-coded rule about review behaviour.

## Data findings

**[UNVERIFIED: no data. Diagnostics to run on train only, with the hypothesis expected for each.]**

Facts from the description (verified): train 2,418 rows, test 586 rows, project-disjoint, labels in {0,1,2}, profile contains "named count bands, lexical flags, branch category, file-change aggregates, and path-kind aggregates"; numerics are "profile tokens" (bands). Test/train ratio ~0.24.

Diagnostics (script `scratch/eda_u23.py`, not shipped):
1. Format discovery: print 10 raw `profile_text` strings; token delimiter; `name=value`-like structure; number of distinct field names; value cardinality per field; which values are numeric bands and how they are spelled (e.g. `b3`, `10-19`, `lt5`); token length distribution (for any encoder max_length; expect < 128 tokens at word level, 200-400 subword tokens if a tokenizer shreds band tokens). Hypothesis: key=value bands, 20-60 fields.
2. Label distribution exact %, per class. Hypothesis: quiet majority (50-70 %), light 10-25 %, active 15-30 %, but project-dependent. Compute the **ideal DCG** on random 586-row subsamples and the expected NDCG@20 of a random ranking (= mean gain / mean of top-20 gains approx.); hypothesis random NDCG@20 ~ 0.15-0.35 and the ideal top-20 saturates with actives (all 3's), so the metric ~ "active precision in top 20, position-weighted".
3. Missingness/sentinel tokens ("none", "unknown", "0"), constant and near-constant fields, duplicates: exact duplicate `profile_text` count and label disagreement among duplicates (irreducible ambiguity); near-duplicate pairs by token-set Jaccard >= 0.9. Hypothesis: several percent exact duplicates in bulk/automated backports; label disagreement among them gives a ceiling.
4. Per-field univariate signal: for each field/token, mean gain and P(active) by value with counts; Spearman of ordinal bands with target. Hypotheses [UNVERIFIED]: (a) larger file/line footprint -> more review but with non-monotone tail (huge mechanical backports get rubber-stamped); (b) lexical flags like conflict/security/revert/fix raise activity; "clean cherry-pick"-like flags lower it; (c) branch category (stable/release/maintenance/legacy) shifts base rate; (d) path-kind mix (tests, docs, config, generated/vendored) matters; (e) language mix mostly acts as project-type proxy.
5. **Project-structure recovery** (the key to validation): since no group id is expected, (i) exact/near-duplicate union-find, (ii) hierarchical clustering of "project-stable" fields (branch category, language/structure mix, path-kind aggregates) vs "PR-specific" fields (footprint, lexical flags). Evidence needed: within-cluster label mean variance beyond what random splitting gives (intraclass correlation of the target by cluster). Hypothesis: high ICC (project review norms dominate) which is exactly why the hidden split is project-disjoint.
6. **Leak/fingerprint tests:** (a) 1-NN and kNN label agreement under random stratified K-fold vs cluster-GroupKFold: a large drop means project-fingerprint memorisation; (b) LightGBM CV score under both schemes; (c) autocorrelation of target in file order and correlation of target with id hash prefix (expect none; if there is any, it is a trap and is ignored); (d) which tokens vanish when requiring support in >= m pseudo-groups.
7. Sanity of the metric: unit-test the exact `ndcg_at_k` on perfect, reversed, constant (stable-sort -> first 20 rows), base-rate, random predictions; confirm expected NDCG@20 of random on 586-row slates matches (2).
8. Test side (allowed: schema, row count, id format, size only): 586 rows, columns, id uniqueness, token length bound for runtime. No distribution comparison.

If a project-hash column does exist locally: dev-only, compute adjusted Rand index between it and my content clusters to choose the cluster count; do not ship.

## Validation design

Reproduce the split: the hidden split is **project-disjoint** (test projects absent from train); ~586 rows pooled in one list; train has ~2,418 rows. Random K-fold would let project norms leak through project-fingerprint fields and over-state skill (optimistic, size unknown, potentially large given hypothesised high ICC).

- **Groups (content-derived, in-script):** (1) union-find on exact duplicates and token-set Jaccard >= 0.9 (also protects against near-duplicate leakage); (2) agglomerative clustering (train-only, fixed seed, fixed linkage) on token features, preferring project-stable fields, into a fixed K clusters (pre-set grid K in {30, 60, 120}, each reported; primary K=60 ~ 40 rows/cluster). Bias: clusters coarser than real projects -> **pessimistic** (removes legitimately transferable neighbours, so train for each fold looks less like the holdout than real train looks like test); finer than projects -> optimistic. Report all three K and treat the spread as the uncertainty on project leakage. Rely on the scheme where gap vs random folds is largest as the conservative yardstick.
- **Folds:** `GroupKFold`/shuffled group split, 5 folds, **3 repeats** (different seeds of group-to-fold assignment AND cluster seed). Each outer fold has ~480 rows, close to the 586-row test pool, so per-fold NDCG@20 is an honest-size replica of the metric.
- **Metrics reported:** exact NDCG@20 per fold (mean +- std over 15 fold-evals and SE over repeats), plus smoother companions to judge changes: NDCG@20 on 200 bootstrap slates of 586 rows drawn from pooled OOF (reduces slate-sampling noise but not data noise), NDCG@50/@100, AUC(y=2 vs rest), AUC(y>=1), binary logloss of each head, Spearman of `E[gain]` with the label. Model selection uses the smooth ones (logloss, NDCG@100, bootstrap NDCG@20); NDCG@20 on folds is reported, not tuned on.
- **Acceptance rule:** accept a change only if the paired difference (same folds, same repeats) exceeds 1 SE across repeats and has the same sign in >= 11 of 15 fold-evals. Per-fold NDCG@20 SD is probably 0.05-0.10 [UNVERIFIED], so only gains >~ 0.02-0.03 are resolvable.
- **Nested check:** HPO (fixed 20 trials, seeded TPE, n_jobs=1) is run (a) once on all train to choose the shipped config, and (b) inside each outer fold of repeat 0 (5 x inner 3-fold) to report an honest nested NDCG@20. Blend weights/calibrators: equal weights or one-parameter; calibrators fit cross-fitted on OOF.
- **Holdout sanity fold:** before the final run, hold out ~20 % of clusters once; fit everything (including HPO) on the rest; evaluate once. Report it separately and do not retune.
- **Bias statement:** my proxy is (i) optimistic where pseudo-groups split real projects (the same project's rows in train and valid fold) and (ii) pessimistic where they over-merge. Net bias unknown sign; expected private score band is set from the group-CV result minus ~0.02-0.05 for selection effects, with +-0.07 slate noise on 586 rows.

## Overfit/underfit risks

Overfit (high):
1. **Project memorisation** through project-stable fields (branch category, language mix, path-kind mix). Mitigation: cluster-grouped CV; token support filter (a token enters the vocabulary only if present in >= 20 rows and in >= 5 pseudo-groups); shallow trees, high `min_child_samples`, strong L2; ablate project-stable field blocks and keep a block only if its grouped-CV gain exceeds noise.
2. Few effective groups (tens): effective sample size is groups not 2,418 rows; capacity ladder (below).
3. Top-20 metric is noisy; HPO on it selects noise. Mitigation: select on smooth surrogates, small search space, fixed rounds, nested check.
4. Duplicate/near-duplicate rows inflate CV; grouped by union-find.
5. Distribution shift in base rate across projects: calibration of `p1` vs `p2` shifts; ranking within a pool is mostly robust, but the blend coefficient between heads is theory-set (w=2) not tuned.

Underfit (moderate-low): tabular signal is mostly count/band structure; risk is over-regularising monotone effects (footprint) or discarding information when parsing bands to ordinals (keep both ordinal numeric and one-hot token views), or collapsing the 3-class label to binary (use cumulative heads). Encoders that tokenize band tokens into noisy subwords underfit numerics; mitigated by parsed numeric channel.

## Recommended approach (primary + fallback)

**Primary (lean, CPU, clean): parsed-profile features -> two cumulative ordinal heads -> expected-gain ranking, as a small ensemble of assumption-diverse learners.**
- Parse `profile_text` into (a) one-hot/count token features (vocabulary fit on train, support-filtered), (b) numeric ordinals for band tokens (parsed deterministically from the token value; band->midpoint number learned from train token spellings), (c) log/ratio features within the row only: lines added/deleted ratio, average changed lines per file, test/doc/config share of paths, branch-category flags crossed with footprint, count of lexical flags set, number of language groups. No test-wide or slate-relative statistics.
- Heads: `p1=P(y>=1)`, `p2=P(y>=2)` (cumulative at-least-k, enforce `p2<=p1`). Score `s = p1 + 2*p2` (exact expected gain).
- Members (each scored alone first on the same folds; blend only members within noise of the best):
  1. **LightGBM** binary heads, shallow (`num_leaves` 4-15), `min_child_samples` 30-100, `feature_fraction` 0.3-0.8, `bagging`, strong L2, fixed lr 0.03 and fixed rounds from a small grid; `deterministic=True`, `force_row_wise=True`, fixed threads. Optionally monotone constraints on footprint bands if grouped CV supports them (measured, not assumed).
  2. **Regularised logistic cumulative heads** (L2/elastic-net on standardised ordinals + one-hot tokens; C chosen in-script from a wide grid including strong end, per head). Different inductive assumption (additive, low capacity) = the first rung of the capacity ladder and a stability anchor across projects.
  3. **LightGBM LambdaRank arm** (listwise): pseudo-slates of ~500 rows formed from random partitions of the training part (re-drawn per seed), `label_gain=[0,1,3]`, `lambdarank_truncation_level=20`, `eval_at=[20]`. Diversity of loss assumption (directly optimises the head of the list). Its output enters the blend only via a train-reference map (empirical CDF of its OOF scores), never via test ranks.
  4. (Optional, step 5) **CatBoost** (CPU, ordered boosting, fixed seed/threads) as a different tree family if it beats noise.
- Blend: average of `s` from pointwise members (same scale), plus rank-CDF-mapped LambdaRank with an equal weight if it earns it. No free weights beyond 1-2.
- Shipped model: refit on 100 % of train with fixed counts (rounds chosen as the CV-derived value x 1.1 or from the HPO grid), 5-10 seeds averaged. Fold models are used for OOF/calibration numbers only, unless a measured transfer problem appears.

**Fallback / secondary arm (genuine NLP training, for diversity or if a reviewer wants an encoder): LP-FT style small encoder on the profile text.** Pinned HF checkpoint (e.g. `microsoft/deberta-v3-base` or a smaller BERT variant, revision pinned), input = the profile string (plus parsed numeric view as an extra head input), two sigmoid heads trained with BCE, order of operations by the capacity ladder: frozen mean+CLS pooled features -> linear head probe -> last-block fine-tune with tiny LR for 2-3 epochs, fixed schedule, no validation-triggered stopping; log parameter-change norm and its CV gain over the probe. Enters the blend only if grouped-CV gain over the GBDT/linear blend exceeds noise. Honest expectation: for synthetic band tokens it will probably match or trail the parsed-feature models; its value is diversity and as compliance insurance if "genuine NLP fine-tuning" is demanded.

Why this ranks first: effective sample size is a few dozen projects; the lowest-capacity rungs that win under grouped CV generalise best to unseen projects; expected-gain ranking is metric-exact; LambdaRank supplies a metric-shaped alternative.

## Rejected options

- **Full fine-tune of a base-size LM as primary:** ~2.4k rows, few dozen groups: memorises project fingerprints; subword-shredded band tokens; slower; not more compliant.
- **TF-IDF/BM25-style or kNN/kernel similarity as the core:** grey per CLAUDE.md 2.5 and a project-fingerprint magnet (neighbours are same-project rows, which don't exist at test).
- **Target encoding of branch category or any high-card field:** project-identifying leakage risk; branch category is low-cardinality, so it enters as plain one-hot/ordinal within the model, no label-derived encoding.
- **SMOTE / oversampling / mixup:** banned (synthetic examples).
- **Pseudo-labelling, test-time adaptation, test-wide rank/z-score normalisation, test-fit PCA/scaler/clusters:** banned (CLAUDE.md 2.3 #5).
- **Using id, row order, or any hash as signal or tie-breaker:** banned.
- **Binary-only "active vs rest" model:** discards light vs quiet ordering that the gain mapping rewards when actives run out; cumulative heads keep it.
- **Wall-clock-based HPO/early stopping:** rejected by the determinism checker.
- **Large multi-seed x multi-family ensembles beyond 3-4 members:** extra diversity of seed, not of assumption; not worth runtime or complexity.

## Fixed work plan & runtime budget

All counts hard-coded constants at top of file; no time-based branching; device fixed. GBDT and linear stages CPU with `N_THREADS=4` (constant); optional encoder arm `device="cuda"`. Estimates **[UNVERIFIED, profile locally]**:

| Stage | Plan | Est. |
|---|---|---|
| Load, parse, vocab, support filter, groups (union-find + clustering) | once | < 1 min |
| Smooth-surrogate grouped CV, 5 folds x 3 repeats, 3 members x 2 heads, fixed config | ~100 fits | 2-4 min |
| In-script HPO, Optuna TPE seeded, `n_trials=20` per member, inner 3-fold grouped, on full train | ~120-200 fits each on 2.4k rows, ~0.3-0.5 s | 5-10 min |
| Nested HPO check (repeat 0 only, 5 outer x 20 trials x 3 inner) | ~600 fits | 5-8 min |
| Final refit on all train: 3 members x 2 heads x 5 seeds | ~30 fits | < 1 min |
| Optional encoder arm: 5 folds x 3 epochs, 2.4k rows x <=256 tokens, fp16 autocast, batch 32, + final refit | ~25 s/epoch | 8-12 min |
| Validation, CSV write, reload check | | < 10 s |

Total ~15-25 min without the encoder arm, ~25-40 min with it; >= 30 % headroom vs 1 h. Memory: < 4 GB RAM, < 8 GB VRAM. Seeds: `PYTHONHASHSEED`, `random`, `numpy`, `torch`, LightGBM `seed`/`bagging_seed`/`feature_fraction_seed`/`deterministic`, `TPESampler(seed=...)`, `n_jobs=1`. Pin HF revision for any pretrained download. Source: single plain file < 100 KB.

## Metric-aware training & decode

- (i) Back-solve: target is ordinal {0,1,2} with gains {0,1,3}; cumulative heads give an exact `E[gain] = p1 + 2*p2`; no invertible transform to exploit.
- (ii) Loss weighting: metric weights are per-position, not per-row; a gain-weighted pointwise loss (active weight 3, light weight 1 on the corresponding head) is a CV-tested option; default is unweighted calibrated BCE because ranking by calibrated probability is optimal and weights hurt calibration.
- (iii) Ordinal: at-least-k cumulative heads, consistency `p2 = min(p2, p1)`.
- (iv) Utility decode: sort by expected gain (exactly the optimal rule for expected DCG with fixed discounts). There is no constraint set; no beam. A single constant, the weight 2 of `p2`, is theory-derived (3-1); sensitivity {1,2,3} is a diagnostic only, not tuned.
- (v) Listwise arm: LambdaRank with `label_gain=[0,1,3]`, truncation at 20, pseudo-slates of ~500 rows drawn from train only, so its training objective matches the 586-row list.
- Calibration: per-head Platt/isotonic on OOF predictions, cross-fitted and reported; only used if it changes ranking AND helps across folds (it matters only for the relative weight of the two heads and for blending across scales).
- Ties: continuous scores from multi-seed averages; no id-based tie-break.

## Structural signals

- Nested ordinal structure (quiet < light < active): cumulative heads and listwise loss.
- Monotone band semantics: footprint/count bands are ordered; expose ordinal numeric value so trees and linear heads can use order and not only one-hot; optional monotone constraints verified by grouped CV.
- Row-internal ratios: per-file churn, add/delete ratio, test/doc/config/generated share, flags count, branch-category x footprint interaction.
- Project-norm confound: review activity is project-driven; features that identify project type are legitimately predictive of norms but must transfer across projects, hence group-disjoint selection and block ablations (project-stable vs PR-specific fields).
- Duplicate/near-duplicate rows form groups first; their label disagreement bounds attainable accuracy.
- No symmetry augmentations apply (no geometry); no synthetic rows allowed.
- Backport-specific prior [UNVERIFIED]: a backport of an already reviewed change often needs only light review, so the active tail is probably driven by conflict/size/branch-category signals; confirm via diagnostic 4.

## Experiment roadmap

1. **Contract and validation:** implement exact `ndcg_at_k`, unit tests (perfect=1, reversed low, constant=first-20-rows value, random ~ expected), group derivation, 5x3 grouped CV harness, random-fold comparator. Stop when the harness reproduces the unit-test values and the random vs grouped gap is measured.
2. **Cheap baseline, end-to-end valid:** token features + L2 logistic cumulative heads; write and validate CSV; record per-fold NDCG@20, AUCs. Stop: valid CSV, CV table logged.
3. **Representation:** parsed ordinals + ratios; LightGBM cumulative heads; support-filter ablation; project-stable-block ablation. Stop when gains < 1 SE.
4. **Metric-aware loss/decode:** compare cumulative heads vs multiclass vs gain-regression vs LambdaRank pseudo-slates, all on identical folds; weight-w sensitivity diagnostic. Stop when the best single design is chosen.
5. **Diversity:** blend members within noise of best (LightGBM + logistic + LambdaRank; CatBoost; optional LP-FT encoder arm). Keep a member only if the paired grouped-CV gain exceeds noise.
6. **In-script HPO:** fixed 20 trials, narrow ranges; nested check; compare to fixed defaults; keep HPO only if nested score >= defaults.
7. **Final run:** clean `working/`, exact command `python3 solution.py ./dataset/public ./working/submission.csv`, run twice and diff; validator (CLAUDE.md 5); sanity: score distribution on test vs OOF (mean/std only, no tuning), no constant output, 586 unique ids. Credits: baseline, best single, blend, final.

## Compliance audit

CLAUDE.md section 7:
- Test file read only for one-row-at-a-time prediction: planned yes; transform uses train-fit vocabulary/parsers; no normalisation across test rows. Cluster/group fitting uses train only. PASS.
- Time inside conditions: none. Time used only in `log()`. PASS.
- `cuda.is_available()`, `os.cpu_count()`, import fallbacks: none; fixed constants. PASS.
- Hard-coded tuned constants: only the search ranges; chosen values come from in-script HPO; the `2` in `p1 + 2*p2` is derived from the metric's gains. PASS.
- External data / synthetic data / self-hosted weights / non-allowed libraries: none (numpy, pandas, scikit-learn, lightgbm, optional catboost, optuna; optional transformers HF weights, pinned). PASS.
- Strip-the-ML: parser alone cannot rank; the trained heads are the solution. PASS.
- Source readable, < 512 KB, commented. PASS.
- Challenge-specific restrictions honoured (table above).
Strategist self-audits:
- No whole-test aggregation: PASS (blend uses train-reference CDF; no test ranks, no test-fit anything).
- Sibling leakage: no cross-row features at all; train groups used in validation and for LambdaRank slates only.
- Hard-coded constants derivable by in-script train-only search: PASS.
- Bans on source indicators: vocabulary scan + drop (see above); residual risk reviewer-dependent.

## Open questions & assumptions

Reviewer questions:
1. Runtime budget: the text read says only "selected runtime budget"; confirm it is the standard 1.5 h on A10G, or a shorter figure.
2. If a lexical flag in the profile text correlates with automated/agent authorship (but is a content flag, not a source field), is it allowed? Plan: drop explicit source/agent indicators, keep generic content flags.
3. Is a GBDT/linear ensemble on parsed profile tokens (plus listwise loss) acceptable ML for this NLP-labelled challenge, or must a fine-tuned encoder carry weight? Plan ships the primary and holds an LP-FT encoder arm in reserve.
4. Is deriving pseudo-project groups by clustering train profiles (train only) acceptable as "grouped validation" given no project column? (Believed yes.)

Assumptions: no project column in train/test; train has only `id`, `profile_text`; labels join one-to-one on `id`; the "opaque project hash" ban is generic and refers to a field not present in the public view; metric is computed on one pooled list of all 586 rows.

Could not verify (no data): label distribution, profile format, token lengths, duplicate rate, ICC by project, whether any source/agent token exists, per-fold noise level, actual runtime. Estimated private NDCG@20 band **[UNVERIFIED estimate]**: ~0.35-0.65 (central ~0.50), with +-0.07 sampling noise on 586 rows and likely a group-CV optimism of a few hundredths; random-ranking floor guessed at 0.15-0.35. No score is promised.
