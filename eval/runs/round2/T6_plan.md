# T6 plan: cross-community next-interest recommendation (Eris Strategist, round 2)

Status of evidence: only the paraphrased description (`eval/tasks/T6_next_interest.md`), `CLAUDE.md` and the strategist spec were read. No dataset was available. Every statement about the data is therefore a hypothesis or a diagnostic to run, marked UNVERIFIED. Numbers that come from the description (1,666 train cases, 416 test cases, 8 candidates, 6 communities, history cap 30) are given as facts; numbers derived by arithmetic are shown with the arithmetic. Column names beyond `id, community, tag, rel_month, n_events, rank1, rank2, rank3` are not known (candidate columns are called "candidate slots" below).

## Contract & decision unit

**One valid answer.** One row per test case `id` with `rank1, rank2, rank3`: three distinct tags, each drawn from that case's own 8 candidates, most likely first. Validity assumptions (UNVERIFIED, confirm against `sample_submission.csv`): the three values are distinct, are members of the row's candidate set, are written in the same type/format as the candidate tags, no NaN or empty cell, ids in sample order. A duplicate or non-candidate value is probably an invalid row, not a low-scoring row. The in-script validator must therefore assert distinctness and membership per row, not just the generic CLAUDE.md section 5 checks.

**Metric, term by term (formula wording UNVERIFIED, see Open questions).**
- Per case: reciprocal rank at 3, RR3 = 1, 1/2, 1/3 if the answer is at rank 1, 2, 3 of my list, else 0.
- Chance normalisation: with 8 candidates and a uniformly random ordering, E[RR3] = (1 + 1/2 + 1/3) / 8 = 0.2292. The normalised score is presumably (MRR3 - 0.2292) / (1 - 0.2292) = (MRR3 - 0.2292) / 0.7708. So random = 0, perfect = 1, and a ranking worse than random is negative (down to -0.297 if never in the top 3), unless the platform clips.
- Final = 0.8 * (normalised MRR@3 over all cases) + 0.2 * min over the 6 answer-communities of the normalised score restricted to cases whose answer lies in that community. The answer community is hidden in test, but it is simply the community of the answer tag, so it is a function of the label and cannot be used as an input.
- This is a gated, hierarchical metric: the 0.2 term is determined by the single weakest community. Per-community noise: with about 416 / 6 = 69 test cases per community, the SD of per-case normalised RR3 is about 0.343 / 0.7708 = 0.445 under chance (and about the same at modest skill), so the SE of one community's score is about 0.445 / sqrt(69) = 0.054. The expected min of 6 equal-skill communities is about 1.27 * 0.054 = 0.07 below their mean, which costs about 0.2 * 0.07 = 0.014 of final score. The overall term has SE about 0.445 / sqrt(416) = 0.022 (weighted by 0.8: 0.017). Test-score differences smaller than about 0.02 are noise; plan the expectation accordingly.

**Decision unit.** The independent unit is the user (one case = one user; train and test users are disjoint). The decision is a within-case choice: a 3-element ordered subset of 8. The effective sample size is at most 1,666 users, fewer if near-duplicate users exist (diagnostic D8). Rows of candidates (1,666 x 8 = 13,328) are not independent.

**Pipeline stages, diagnosed separately.**
1. Candidate coverage: the answer is always among the 8 (verify 100% on train). No coverage loss is possible, the whole problem is ranking.
2. Ranking: calibrated posterior P(candidate c is the next adoption | history, c). This is where all the score is made or lost.
3. Decoding: for MRR@3 with a single labelled answer, ordering by posterior descending is optimal (the reward 1, 1/2, 1/3 is decreasing in rank and linear in the posterior, so sorting by p maximises expected RR3 by the rearrangement inequality). No expected-utility search is needed; decode is argsort of the posterior, top 3.

**What the engineered decoys imply (UNVERIFIED reading of the description).** Three of the seven decoys are matched to the answer on a co-occurrence score, two are popular in the answer's community, two are weak global fillers. Raw co-occurrence with the history is therefore neutralised by construction for 4 of 8 candidates (answer plus matched trio), and raw popularity is neutralised within the answer's community. Only the two weak fillers are easy to reject with generic scores. The ceiling of a co-occurrence-plus-popularity model is therefore roughly "answer uniformly among the 4 matched" after rejecting fillers (top-1 about 0.25, RR3 about (1 + 1/2 + 1/3) / 4 = 0.458, normalised about 0.30 at best), likely lower in practice (UNVERIFIED hypothesis H1). Any real gain must come from signal that the recipe did not equalise: order, timing, lags, community-to-community pathways, the per-event multiplicity `n_events`, and user-level latent interests.

## Compliance regime

**Domain.** Tabular / recommender (sequence of events, candidate ranking), CPU-only runner, about 1.5 h ceiling (plan to 1 h worst case per CLAUDE.md section 1). No pretrained backbone applies: tags are anonymised ids, so no pretrained text or embedding model carries meaning. Training from scratch inside the script on 1,666 cases is the only route, which makes it a genuine ML training solution (strip-the-ML test is satisfied as long as the learned scorers carry the ranking).

**Allowed.** Train-only count statistics fed to a trained model; small neural scorers trained in-script; gradient boosting or conditional logit ranker; cross-fitting; fixed seeds; in-script HPO with fixed trials.

**Explicit bans in the description (all apply to the plan).**
1. Test-set adaptation of any kind: no statistic, vocabulary or association fitted on the test rows' own features, even label-free. Consequences: tag vocabulary, tag-to-community map, popularity, co-occurrence, lag tables, normalisers and embeddings are all fit on train rows only. A test tag unseen in train maps to UNK plus the community given in the test row itself (a per-row function of its own inputs), never to a table built from test.
2. No pseudo-labelling, no test-based calibration or reweighting.
3. No training on synthetic generated data. This includes running the published decoy recipe to manufacture new labelled slates. I therefore do not generate synthetic slates.
4. Platform reviewers reject hand-derived exploitation of how the data was generated when the model does not learn it. This is the central compliance risk for this task: the decoy recipe is published, so anything that recomputes the recipe (the matched co-occurrence score, "popular in community" membership, or the slate's community composition) and uses it to spot decoys is exploitation of the generator. Decision (compliant under every plausible reading): **Tier A** features and models use only the user's history and per-candidate intrinsic statistics learned from training labels. **Tier B** (any feature that cross-references a candidate with its slate-mates: community plurality within the slate, rank of a candidate's popularity within the slate, count of candidates sharing the answer-like community) is built and measured in CV, but is NOT in the shipped primary. Its measured CV gain over Tier A is reported so a reviewer can decide (Open question Q1). If a reviewer explicitly allows learned listwise slate-context, it enters as a pre-planned variant, not as a hand-written rule.
5. Ambiguity that changes the approach: use of history prefixes as extra self-supervised training examples (Q2); candidate community availability in test (Q4). Plans under each reading are given in Open questions.

**Determinism and platform rules (CLAUDE.md section 3).** Fixed folds, epochs, seeds, thread counts (set explicitly), `device = "cpu"` hard-coded, no `elapsed()` in conditions, no import fallbacks, `n_jobs` fixed. Fixed-count early stopping or fixed epoch schedules only.

## Data findings

UNVERIFIED throughout. This section lists the exact train-only diagnostics to run and the hypotheses I expect them to confirm or refute. From the test files only schema, row count (416) and id format would be used, never feature distributions.

**D1. Schema and structure.** Number of distinct tags V; tags per community (6 communities, sizes and skew); is each tag in exactly one community (tag-to-community is a function)? History length distribution, share of histories at the cap of 30, whether the cap keeps the earliest or the most recent 30 events; `rel_month` range, sign, whether it is relative to the first event or to the answer time, number of ties (several events in one month, order unknown inside a month); `n_events` distribution (multiplicity or intensity); missingness; whether answer `rel_month` is given (assumed not). Hypothesis: histories are mostly short with a minority at the cap; ties are common, so within-month order is unusable and events must be treated as a set per month.

**D2. Candidate-slot artefacts.** Distribution of the answer's slot index (chi-square vs uniform over 8). Hypothesis: uniform. If not uniform, it is an ordering leak: never use slot position, and train with random permutation of candidate order so the model is permutation-equivariant.

**D3. Coverage and overlap.** Answer in the 8 candidates for 100% of train cases; fraction of candidates already present in the user's history; fraction of answers already in history (expect 0 or near 0). Hypothesis: if answers are never repeats, "already adopted" is a legitimate, learnable candidate filter (a learned feature, not a hard rule) and removes some decoys cheaply.

**D4. Baselines under the exact metric (leave-user-out statistics, so a user's own answer never counts toward its own score).** Report overall and per community, with bootstrap SE:
 (a) random (expect about 0 normalised); (b) global popularity of the tag as an answer; (c) community-conditional popularity; (d) PMI/co-occurrence of candidate with history tags (mean, max); (e) last-event community to candidate community transition; (f) recency-weighted association; (g) "candidate in the community with the most history events".
 Hypotheses: (d) beats chance only via rejecting the 2 fillers (small positive, about 0.05 to 0.15 normalised); (b) and (c) are near chance or negative because 2 decoys are popular in the answer's community; (e) and (g) carry the first real signal because they are not equalised by the recipe (UNVERIFIED).

**D5. Decomposition of failure.** For a trained model's OOF scores, measure P(answer outranks the 2 fillers), P(answer outranks the 2 popular decoys), P(answer is the top of the matched group). Decoy-type labels, recovered from the published recipe on train rows, are used for this DIAGNOSIS only, never as features, weights or loss terms. Hypothesis: separation from fillers is near 100%, popular decoys partly separable via order/lag/community signal, and the matched trio is where most of the remaining error is.

**D6. Matched-trio verification.** Within each slate compute the leave-user-out co-occurrence score of every candidate; check that the answer's rank within the slate is uniform among the answer-plus-matched group. Hypothesis: yes, uniform, so co-occurrence is information-free within the group.

**D7. Order, timing and community pathways.** (i) Order test: fit the same scorer with the history as an ordered sequence vs a shuffled bag (permutation null over events within user, ties kept); if order adds nothing, drop sequence machinery. (ii) Lag distribution between consecutive adoption events by community pair; lag between each history event and the next event within the history (this is the within-history analogue of the answer); autocorrelation of lags. (iii) Community transition matrix 6x6 of consecutive events and of last-history-event to answer, with counts per cell. (iv) Compare the process that generates the in-history transitions with the process that generates the answer: KL or total variation between the history-internal transition matrix and the history-to-answer matrix. Hypotheses: H3 the history-to-answer step has a different, more cross-community distribution than in-history transitions ("cross-community next interest"); lag buckets matter for pairs of different communities.

**D8. Duplicates and ambiguity.** Exact and near-duplicate histories (Jaccard of tag sets, identical tag sequences), users sharing the same answer tag and same candidate set; identical history with different answers (irreducible ambiguity); ceiling estimate via a nearest-neighbour oracle (answer of the most similar train user, leave-one-user-out) and via the naive best rule from D4. Hypothesis: a nontrivial share of users sit in near-duplicate clusters if the data is synthetic-looking; these must be grouped in CV.

**D9. Community imbalance.** Train answers per community (about 1,666 / 6 = 278 each if balanced), per-community baseline scores from D4 and tag-set size. Hypothesis: the weakest community (the one setting the 0.2 term) is the community with the largest tag set or the highest overlap with others; identify it from OOF, not guess.

**D10. Cold tags.** Distribution of train answer counts per tag; share of candidates with fewer than 5 train occurrences; whether the matched decoys favour well-covered tags (UNVERIFIED). Hypothesis: long tail; cold tags need community-level shrinkage.

**D11. Slate-composition ceiling (Tier B measurement, not for shipping).** Accuracy of "candidate belongs to the plurality community of the slate" and a simple classifier on slate composition. Hypothesis H4: because 2 decoys are popular in the answer's community, the answer's community is over-represented among candidates, so slate composition alone may beat chance by a large margin. This is a generation artefact: it is reported as a measured ceiling for the reviewer question, and a large value is treated as a warning that raw CV could be inflated if any Tier B signal leaks into Tier A features.

## Validation design

**Reproducing the split.** Train and test users are disjoint, and 416 / (1,666 + 416) = 20.0%, so the test split looks like a random 80/20 user hold-out, likely stratified or at least random over communities (UNVERIFIED; the answer community of test cases is hidden but presumably balanced like train). 5-fold CV matches this 80/20 ratio.

**Groups.** The decision unit is the user, so group by user at least. Derive groups by union-find on content overlap, not just a provided id: connect users with identical or near-identical histories (Jaccard of tag sets above a threshold chosen from the D8 distribution of nearest-neighbour similarities, by in-script rule), and users sharing the same answer tag AND the same set of candidates. `GroupKFold` over these clusters, stratified by answer community (grouped stratified assignment, fixed seed).

**Repeats.** 5 folds x 3 split seeds. Report mean and SD of the overall normalised MRR@3, each community's score, the min over communities, and the final composite 0.8 * overall + 0.2 * min. Because the same 1,666 cases are reused, repeating seeds does not reduce sampling noise of the case sample; use paired differences on identical folds. Accept a change only if the paired difference exceeds 1 SE of the paired fold differences and is positive in most folds and seeds. The OOF SE of the overall score is about 0.445 / sqrt(1,666) = 0.011; per community (about 278 cases) about 0.027.

**Test-like estimate of the min term.** Because the real min is over about 69 cases per community, also compute the composite by bootstrap-resampling OOF cases in a 416-case sample (stratified like test) many times: report mean and SD of the composite. This is a more honest number than the pooled OOF min (which is optimistically stable with 278 per community).

**Metric reimplementation and unit tests.** Implement the exact metric from the description and test: perfect ranking gives 1.0 overall and min 1.0 (composite 1.0); answer never in the top 3 gives -0.2292 / 0.7708 = -0.297; an answer-never-top-3 constant gives the same; uniformly random ordering averages about 0 (check by simulation, tolerance by SE); also check a swap where the answer is always rank 2 gives (0.5 - 0.2292) / 0.7708 = 0.351. Also unit-test the per-community split and the min over communities with a hand-built toy.

**Leak-proof stage structure.** Every label-derived statistic (popularity, PMI, lag-bucket association tables, community transitions) used as a feature on a TRAINING row is computed leave-user-out (subtract that user's own contribution from the counts, exact for count-based statistics) or from inner folds, so a user's answer never sits in its own feature. At validation and test time the features come from tables fit on all training users (in CV: all training-fold users). Without this, features look better in CV than in test because the answer is counted in its own association. Learned stage-1 neural scores used as stage-2 features are cross-fitted (OOF), with an honest nested number computed at dev time (outer 5 folds, inner 5 folds) to check the stacking optimism; the shipped script does a single-level cross-fit.

**Every post-hoc choice gets its own held-out check.** Regularisation strength, community weighting exponent, embedding size, number of mixture components, blend choices are chosen inside the inner folds and reported on outer-fold data; the final headline is the outer-fold number. A self-built proxy usually over-estimates the test score; the real number is expected to land below the OOF value (more so for any Tier B feature).

**Shift check.** The slate construction for train and test cases is assumed identical (same recipe). If the recipe's statistics were computed from the full dataset (train plus test), train slates may carry subtle statistical differences from test slates that CV cannot see; this is a reason to keep the shipped primary on history-based signal (Tier A).

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Few independent units (at most 1,666 users, fewer after de-duplication) against a tag embedding table of V x d parameters. Mitigation: capacity ladder, lowest rung that wins under grouped CV: (i) a handful-of-parameters conditional logit on count features; (ii) hierarchical low-rank scorer with tag embedding = community embedding + shrunken residual, tied between history and candidate use, small d (about 16 to 32), strong weight decay, dropout on history events; (iii) only if (ii) wins clearly, add the mixture/latent-interest head. Do not add a deeper transformer.
2. Self-inclusion leakage in count features (the answer counted in its own co-occurrence). Mitigation: leave-user-out statistics, nested check.
3. Near-duplicate users across folds. Mitigation: union-find groups.
4. Selection noise: many knobs on a noisy 0.011-SE CV. Mitigation: small fixed grids, nested evaluation, accept only gains larger than paired noise, plateau choices.
5. Min-over-communities noise and community-specific overfit. Mitigation: pool-level selection first, community weights as a single scalar exponent chosen by nested CV, not per-community tuning.
6. Decoy-recipe artefacts that exist in CV but not in test (or that reviewers reject). Mitigation: Tier A only in the primary; decoy-type labels never used as features or weights; CV gain of Tier B measured separately.
7. Distribution shift between prefix-derived examples and real cases (the final answer step may follow a different process than in-history steps, D7 iv). Mitigation: auxiliary prefix loss with a small weight selected by grouped CV (including weight 0), and a flag feature "answer-step vs history step" is not available at test, so the weight is the only control.

**Underfit risks and mitigations.**
1. Information thrown away by pooling: aggregating the history into bag-of-tags loses order, lags and multiplicity. Mitigation: lag-bucketed pair potentials and a recency-weighted pooled encoder that sees `rel_month` gaps and `n_events`.
2. Under-capacity in the community pathway: a single global bias per candidate community cannot express "users strong in community A next move into community C". Mitigation: an explicit community-level potential, a function of the user's 6-dim community mix and the last events' communities, with interaction by candidate community.
3. Heterogeneity across communities (one global model sacrifices the weakest one). Mitigation: community-specific scale and bias on the candidate side (about 12 parameters), community-balanced loss weights.
4. Truncated context if sequence length is capped too low. Mitigation: use all 30 events (the data cap), no further truncation.
5. Loss mismatch: the metric is MRR@3 but only the answer is labelled, so the correct training loss is the 8-way softmax cross-entropy (a proper score whose posterior ordering maximises expected RR3). A smooth-MRR surrogate adds nothing beyond that and is not planned.

## Recommended approach (primary + fallback)

### Primary: community-then-tag, listwise, lean (Tier A)

Idea. The next adoption is a function of the user's latent interests and the community pathway; explicit structure is placed in the model rather than asked of a generic classifier. Score each of the 8 candidates with two learned potentials, normalise by softmax over the 8 candidates of that case, and train by the 8-way softmax likelihood on the real labelled slates:

 log p(c | history) = softmax over the 8 candidates of  [ phi_comm(history, community(c)) + phi_tag(history, c) + b_comm(c) ]

1. **Community-level potential phi_comm.** Input: the user's 6-dim community profile (event counts and `n_events`-weighted shares, recency-weighted shares), the community of the last event and of the last 3 events, time span, history length, plus the 6x6 last-community to candidate-community interaction. Model: a few-parameter linear/softmax head, optionally a small GBDT in the fallback. This level has the most data per parameter (6 classes, 1,666 users) and directly controls the worst-community term.
2. **Tag-level potential phi_tag.** A hierarchical low-rank scorer: embedding of a tag = community embedding + shrunken tag residual (cold tags fall back to the community embedding), embeddings tied between history and candidate use. The user vector comes from a lean pooling of the history events weighted by learned recency/lag kernels (a small number of lag buckets, derived from train quantiles in-script) and `n_events`; score = user vector dot candidate embedding. Optional rung 3: mixture of K = 2 to 4 latent interest components, p(c) = sum_k pi_k(history) softmax_c(u_k . e_c), trained by exact marginal likelihood; kept only if it beats the single vector by more than the paired noise.
3. **Learned slate-independent priors fed as inputs, not rules.** Leave-user-out log popularity, community-conditional popularity (the conditional logit learns its sign: positive or negative), leave-user-out PMI/lag-bucketed association aggregates, "candidate already in history" flag, cold-tag indicator. These are features of a candidate and the history only.
4. **Auxiliary self-supervised loss (conditional on Q2).** Next-event prediction with a full-vocabulary softmax on prefixes of the train histories (real events, no generated decoys), small weight lambda from a small fixed grid including 0, chosen by nested grouped CV. Purpose: more gradient signal for the embeddings from about 1,666 users' histories; risk: it teaches popularity and co-occurrence, the very signals the decoys neutralise, hence the CV-selected small weight.
5. **Stage 2 stacking (one shot).** The stage-1 OOF logits, the count features and the user context go into a final conditional-logit (softmax over the 8) with strongly regularised L2 and community-specific scale/bias; penalty chosen per output by the exact metric under grouped CV with a very wide grid including the strong end. Refit on 100% of the train data for the shipped predictions (fold models only produce OOF numbers and calibration). Test features come from tables and models fit on all train users.

Why it fits. It encodes the factors the recipe leaves unequalised (community pathway, order, lag, multiplicity) with few parameters, fits the effective sample size, and keeps the component carrying the 0.2 term (community level) calibrated by construction. The final ranking is produced by trained components, so the strip-the-ML test fails the stripped version (remove stage 1 and 2 and nothing ranks).

Expected score (an estimate, not a promise, UNVERIFIED): overall normalised MRR@3 about 0.15 to 0.40 on private, composite about 0.12 to 0.37 after the min-term penalty (roughly 0.014 to 0.06 below the overall). Reasoning: neutralised co-occurrence and popularity cap a naive model near 0.3 even on a perfect filler rejection; any order/lag/pathway signal adds on top; uncertainty about whether such signal exists is large, hence the wide range. If D7 shows no order or pathway signal, the realistic range falls to 0.05 to 0.25.

### Fallback: gradient-boosted ranker on hand-built pair features (Tier A)

LightGBM lambdarank or binary logloss with grouped slates (small trees, depth 3, about 200 rounds, strong regularisation, fixed seeds, `num_threads` fixed), on the same Tier A feature set as primary item 3 plus the community profile and the lag-bucketed aggregates; no neural component. It is trained on real slates only, leave-user-out features, and refit on 100% data. It is cheaper, robust, and likely within a few hundredths of the primary. If the neural stage-1 does not beat the fallback by more than the paired noise (Experiment roadmap step 5), the fallback is shipped as primary and the neural scorer is dropped.

## Rejected options

1. Pure co-occurrence/PMI/item-item CF as the core: neutralised by the matched decoys (D6 expectation) and a grey rule-based statistic under CLAUDE.md section 2.5; kept only as input features.
2. Global or community popularity as a ranking rule: the 2 popular-in-community decoys make it near chance or harmful; it enters only as a learned-sign feature.
3. Recomputing the decoy recipe (matched-score, "popular in community" membership, community plurality of the slate) as a decoy detector: hand-derived exploitation of data generation, rejected by the platform; Tier B is measured, not shipped.
4. Generating synthetic slates by running the recipe on history prefixes: synthetic training data, banned.
5. Pseudo-labelling, test-fit vocabularies, test co-occurrence or popularity tables, transductive normalisation across the 416 test cases: explicitly banned.
6. Pretrained text or embedding models on tag ids: ids are anonymised, no semantics to exploit; no allowed external backbone helps.
7. A deeper from-scratch transformer or full sequence model over histories: far above the capacity ladder for at most 1,666 users; would memorise users. Revisit only if the lean scorer shows a measurable order effect (D7 i) and the capacity ladder says to climb.
8. Large ensembles of near-identical GBDTs: diversity of seed, not of assumption; the planned diversity is between the neural hierarchical scorer and the boosted pair-feature ranker.
9. Per-community separate models: 278 cases each, overfit; instead share parameters with community-specific scale/bias.

## Fixed work plan & runtime budget

CPU-only, `device = "cpu"` hard-coded, fixed thread counts (torch 4, LightGBM 4, BLAS fixed), fixed seeds for random/numpy/torch/LightGBM, deterministic algorithms where available. No wall-clock branching, no environment-dependent fallbacks. Runtime estimates are UNVERIFIED and must be profiled locally before hard-coding counts.

| Stage | Fixed plan | Estimate |
|---|---|---|
| Parse and validate input; build tag/community vocab from train only; lag-bucket edges from train quantiles | once | under 1 min |
| Group clustering (union-find on near-duplicates) and stratified grouped fold assignment | seed fixed, 5 folds | under 1 min |
| Leave-user-out count tables (popularity, PMI, lag-bucket association, community transitions) and feature build for 1,666 x 8 rows (about 13,328) and 416 x 8 rows | vectorised | 2 to 4 min |
| Stage 1 neural hierarchical scorer: 5 outer folds x 2 seeds plus the full-data fit x 2 seeds = 12 fits; about 40 epochs each, batch 128, d = 16 to 32, plus the prefix auxiliary (if allowed) | fixed epochs, fixed schedule, no validation-triggered stopping | about 10 to 14 min (about 50 s per fit, UNVERIFIED) |
| Stage 2: conditional-logit with fixed L2 grid via inner grouped CV; LightGBM fallback ranker and its CV (5 folds x 3 repeats, small grid) | fixed grid, fixed rounds | 5 to 8 min |
| Final blend selection among primary and fallback on OOF (single weight or none) | fixed | under 1 min |
| Refit on 100% train, predict 416 cases, validate the submission, re-read the file | once | 2 to 3 min |

Total about 22 to 32 minutes, against a worst-case 60 min target (headroom at least 45%) and the 1.5 h ceiling. Memory under 3 GB (13,328 candidate rows, small embedding table). The nested stacking check (outer 5 x inner 5 stage-1 fits, about 30 fits) is a DEV-TIME experiment run outside `solution.py` (never a shipped branch), estimated at 15 to 20 min.

Input and output validation inside the script: assert columns, id uniqueness and ids matching the sample order; per-row candidate set of size 8 and no NaN tags; check all values finite before and after scoring; after writing, re-read the CSV with `keep_default_na=False` and assert rank values are distinct and belong to that row's candidates, row count equals 416. If validation fails the script raises loudly (no silent fallback); a constant-output fallback is not planned (it is not banned but scores about 0 and hides failures). Every printed log line may show elapsed time as telemetry only.

## Metric-aware training & decode

1. **Loss.** Softmax cross-entropy over the 8 candidates of each case (the answer is the single labelled positive). This is a proper scoring rule for the posterior; ordering by posterior maximises expected RR3. Initialise the candidate output bias at the log of the base rate 1/8 (implicit under the softmax).
2. **Hierarchical metric averaging.** Weight each case by w_k = (1 / n_k)^gamma where n_k is the number of train cases whose answer is in community k (normalised to mean 1), with gamma from {0, 0.5, 1} chosen by nested CV on the composite objective 0.8 * overall + 0.2 * min (computed on test-sized bootstrap resamples to avoid the optimism of a stable pooled min). Because answer community is known on train rows, using it as a loss weight is legitimate; it is hidden only at test.
3. **Per-community calibration.** After OOF predictions, fit a tiny per-candidate-community scale and bias (12 parameters) on OOF logits, cross-fitted across folds and reported cross-fitted. Candidates from different communities compete in the same slate, so cross-community calibration directly changes the ranking, not just a probability value.
4. **Decode.** Rank the 8 candidates by calibrated posterior descending, output the top 3. For ties (extremely unlikely) break by a fixed deterministic key (stage-1 logit then lexicographic tag id), not by slot order. No constants are tuned on the test file.
5. **No threshold or decode constants** beyond the calibration in item 3 and the weight exponent gamma; both are fitted by in-script train-only search with their own held-out check.
6. **MRR@3 specifics.** Only the top-3 positions matter; do not spend capacity on ordering ranks 4 to 8. If the composite is sensitive to a weak community, prefer improving that community's community-level potential (item 1 of the primary) over per-community decode tricks, which cannot trade cases across communities without hurting those cases' own ranks.

## Structural signals

Invariants and structure to turn into modelling constraints (all to be verified on train, D-numbers refer to Data findings):
1. **Slate permutation invariance** (candidate order carries no information, D2): shuffle candidate order each epoch; the scorer is order-equivariant by design; slot index never a feature.
2. **tag to community is a function** (D1): hierarchical embedding (community + residual), cold-tag fallback to the community vector, candidate-community interaction potentials.
3. **Months-ties are sets** (D1): events sharing a `rel_month` are pooled as a set, order not modelled within a month.
4. **Time translation**: use lags relative to the last event/first event rather than absolute `rel_month` if `rel_month` is not absolute-meaningful (D1 decides).
5. **Causal alignment** (strategist lesson 4): the next adoption is caused by recent adoptions with lags; the training sample for stage 1 is "history up to the last event, plus the answer"; with `rel_month` of the answer unknown, the lag from the last event to the answer is marginalised, so the encoder uses recency-weighted pooling, and "months since adoption" are measured relative to the last event of the history. If D1 shows the answer's `rel_month` is available in train only, it is not used (cannot be used at test).
6. **Already-adopted filter**: a learned feature (D3), not a hard exclusion unless every train answer is a non-repeat and the reviewer accepts a learned zero weight.
7. **Event-drop augmentation** (randomly drop 10 to 20% of history events per epoch, recombination of real events only; no fabricated events): a regulariser that respects the claim "the answer depends on the rest of the history", tested for benefit by grouped CV.
8. **Reference-normalised evidence** (strategist lesson 3): association scores are likelihood ratios against the train background (PMI with shrinkage), averaged over community-specific banks, built from training users only.
9. **Per-event multiplicity**: `n_events` as a weight on an event's contribution; check that doubling it behaves like repeated adoption (D1).
10. **Community-conditional label structure**: the tag-level potential is conditioned on the candidate's community, so the model's tag-level ranking never has to compare tags across incompatible communities by itself.

## Experiment roadmap

Order by expected private gain per hour, with stop criteria. Credits are used for baseline, best single, ensemble and final only.

1. **Contract, metric and validation correct and unit-tested.** Implement the metric (perfect, reversed/never-top-3, rank-2, random-simulation tests), the grouped stratified folds and the per-community and test-sized-bootstrap reporting. Stop when all unit tests pass and the random baseline is within 2 SE of 0 normalised.
2. **Diagnostics D1 to D11.** Fill in this plan's Data findings with real numbers; decide gamma grid, lag buckets, whether order matters, whether prefix self-supervision is useful. Stop criterion: D7(i) tells whether order machinery is needed; if not, drop the sequence encoder and ship the fallback family.
3. **Cheapest valid baseline end-to-end.** Leave-user-out popularity, PMI and community-transition features into the conditional logit; write a valid `submission.csv` and re-validate. First credit. Record OOF composite with SE.
4. **Representation and structural insight.** Add the lag-bucketed potentials, community-level potential and the already-adopted flag (Tier A). Keep each only if the paired OOF gain exceeds noise. Stop climbing the capacity ladder at the lowest rung that wins.
5. **Neural hierarchical scorer (rung 2) and optional mixture (rung 3).** Test the neural scorer against the boosted fallback on the same folds; keep only if it beats the best simple model by more than 1 paired SE; stop here if not.
6. **Metric-aware loss and weighting.** Community-balanced weights (gamma grid), per-community calibration (cross-fitted), auxiliary prefix weight (if Q2 allows). One change per experiment.
7. **Diversity.** Blend the primary and the fallback (different assumptions: latent hierarchical embedding vs tree-based pair features) via z-score or train-reference rank average, or feed one family's OOF logit into the other as a feature; keep only when each member alone is close in quality and the blend beats both by more than noise.
8. **In-script bounded HPO.** Fixed small grids (L2, embedding size, gamma, lambda, mixture K) with fixed trials/seeds and its own nested check; no Optuna time limits.
9. **Tier B measurement (for the reviewer only).** Report the CV gain of slate-composition features; do not ship without reviewer approval.
10. **Final fixed-plan run.** Clean directory, one command, run twice and diff the submission (should be identical or near-identical); verify the test prediction distribution looks like OOF (class balance across communities, rank distribution); validator pass; third credit used for the ensemble if two or three earlier steps improved, the final credit for the frozen script.

Where two or three independent solvers would plausibly converge (a leave-user-out PMI/popularity baseline, community transition features, a lag-bucketed association), test them before any proxy-driven tweak.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits against the planned primary and fallback (answers are for the PLAN; they must be re-verified on the real script):

- Test file read for something other than one-sample prediction (stats, vocab, scaler fit, clustering, dedup, rank-normalisation across the test set, pseudo-labels)? No. Tag vocabulary, tag-to-community map, popularity, PMI, lag edges and embeddings are train-only; test rows are transformed one case at a time; the softmax and calibration act within a row only.
- Whole-test aggregation? None: a test row's prediction is a function of its own history and candidates plus train-fit tables and models. In particular no cross-test normalisation, no label-free prior estimation from test slates, no encoders fit on train plus test.
- Sibling (slate-mate) leakage? Candidates of the same slate are scored independently by Tier A. Slate-mate cross-referencing features (Tier B) are suspect because the decoys are constructed relative to the answer; they are excluded from the shipped primary.
- `time.time()` inside any condition? None planned; logging only.
- `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks, try-except that changes work? None; CPU hard-coded.
- Hard-coded constants tuned offline? Everything derivable by in-script train-only search: lag-bucket edges from train quantiles, L2, gamma, d, lambda by in-script nested CV with fixed grids and seeds. Do not paste values found from earlier real submissions; do not read the public leaderboard to choose constants.
- External data, synthetic data, hosted weights, non-allowed libraries? None. Libraries: numpy, pandas, scikit-learn or scipy for conditional logit if used, lightgbm, torch (CPU). No synthetic slates, no decoy generation.
- Strip-the-ML test: removing the learned stage-1/stage-2 components leaves only counting tables that rank near chance (by the D4/D6 expectation), so the trained model carries the ranking. Verify with the D4 baseline numbers.
- Is the model heavy part genuinely dominant (not a thin wrapper over a rule engine)? Yes, no hand-written decoy detection; the community-level and tag-level potentials are learned.
- Hand-derived exploitation of data generation? The published recipe is used only in diagnostics (D5, D11) and in the Tier B measurement; the shipped features do not recompute it. This remains a reviewer-discretion item (Q1, Q2).
- Source readable, under 512 KB, plain code, comments explaining reasoning? Planned.
- Challenge-specific restrictions honoured: CPU-only, in-script training, no external data, runtime well under the limit.
- Determinism: fixed folds, seeds, epochs, thread counts, fixed grids; no validation-triggered stopping used in the shipped path (fixed schedules only), deterministic tie-breaks.

## Open questions & assumptions

**Reviewer questions (with plans under each reading).**
- **Q1. Slate-context features.** Are learned listwise features that see the whole slate (peer-relative popularity rank, community plurality within the slate) allowed, or do they count as hand-derived exploitation of the published decoy recipe? Reading A (not allowed or unclear): ship Tier A only (the primary above); cost = the measured Tier B gain in CV (UNVERIFIED, could be large per D11 hypothesis H4). Reading B (allowed when learned by a trained listwise model): add a learned slate-context head (candidates attend to the slate's community histogram), selected by nested CV; expect a higher CV and a higher risk of train-versus-test slate-construction mismatch.
- **Q2. Prefix self-supervision.** May next-event prediction on prefixes of the real train histories (real events, no generated decoys) be used as an auxiliary training signal? Reading A (allowed): auxiliary loss with the CV-chosen weight. Reading B (not allowed): train stage 1 only on the 1,666 labelled slates, use histories as inputs only; cost is measured in step 6 of the roadmap (UNVERIFIED).
- **Q3. Exact metric.** Chance-normalisation constant (assumed 0.2292 for 8 uniformly random candidates), whether it is clipped at 0, whether per-community scores are normalised before the min, handling of duplicate or non-candidate rank values (assumed invalid), and whether the cap of 30 events truncates the earliest or the most recent events. The plan's unit tests and community weights are written to be insensitive to the first three.
- **Q4. Candidate community in test.** Is the community of each candidate tag given in the files (assumed derivable from train for known tags)? If a test candidate tag is unseen in train and its community is not provided per-row, it maps to UNK with no community potential (cost: unknown).
- **Q5. Source of the decoy statistics.** Were the co-occurrence and popularity statistics used to build decoys computed on train only, or on train plus test? If they used test rows, train and test slates differ systematically in a way CV cannot show; the plan then stays on history-based signal.
- **Q6. Reproducibility tolerance.** Are minor floating-point differences between two CPU runs acceptable for the deterministic check, or must the submission be byte-identical? The plan favours fixed seeds, single-thread-order-stable operations and deterministic tie-breaks, and ranks decoded from posteriors are robust to tiny numeric noise unless two candidates have near-identical scores.

**Assumptions (all UNVERIFIED).**
- A1. Answer community is the community of the answer tag and balanced roughly 1/6 per community.
- A2. Candidate list contains the answer exactly once and the answer is never already in the user's history.
- A3. `rel_month` is a within-user time axis and ties exist; the answer's own month is not provided.
- A4. The test split is a random user-level hold-out comparable to train (20% of 2,082 users).
- A5. The decoy recipe is identical for train and test.
- A6. Tag aliases are consistent, so a train-fit vocabulary covers nearly all test tags; unseen tags are rare and handled with UNK plus per-row community.

**What could not be verified in this run.** Everything about the actual data (all of Data findings), the exact metric formula, the file schema beyond the description, the candidate-slot column names, runtime estimates, the size of the Tier B gain, and the expected score range (a reasoned estimate only).
