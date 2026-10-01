# U6 Triage record reconstruction: build plan

Status of evidence: this plan was written WITHOUT any dataset files. Everything under "Data findings" is a diagnostic to run on `train.csv` plus an UNVERIFIED hypothesis. Facts marked [stated] come from the challenge text; everything else is an assumption or an estimate. Only three files were read: the agent procedure, CLAUDE.md, and `tasks/user/U6_triage_record.md`. No solution code is included.

## Contract & decision unit

**One valid answer.** A string `priority=<P>|breadth=<B>|delay=<D>|harm=<H>` with P in {0..9, unknown}, B in {0,1,2}, D in {0,1,2,3}, H in {1,2,3,4, unknown}, no spaces, one row per test id in `sample_submission.csv` order, columns `id,triage_record`, no duplicates, no empty strings. [stated]

**Invalid vs low-scoring.** Malformed or out-of-range fields lose credit for the affected field only [stated]. `priority` and `harm` are unknown together, always [stated], so the output must never contain exactly one of them as `unknown`. The script treats such a record as a bug and raises before writing. The legal output set is tiny and enumerable: 12 abstain records (`priority=unknown|breadth=b|delay=d|harm=unknown`) plus 10 x 3 x 4 x 4 = 480 reported records, 492 in all. The task is tagged Seq2Seq but needs no text generation. Predicting the four fields and rendering them loses no valid answer.

**Metric term by term** (composite, flat weights, no hierarchical averaging) [stated]:

| Term | Weight | Scored on | Notes |
|---|---|---|---|
| AWK(priority) | 0.35 | rows where TRUE priority is defined (~40%) | Asymmetric, under-triage costs more. Exact weights unpublished. |
| MacroF1(harm reported vs unknown) | 0.15 | all rows | Binary abstention. Always or never abstaining scores poorly. |
| QWK(harm severity) | 0.15 | rows where TRUE harm is reported | |
| QWK(breadth) | 0.15 | all rows | |
| QWK(delay) | 0.15 | all rows | |
| ExactRecordMatch | 0.05 | all rows | Low weight. Do not optimise jointly. |

All components are clipped to [0,1] and a constant or random solution scores about 0.

- 0.65 of the weight (priority, abstention, severity) rides on the harm judgement. Priority is driven "primarily" by harm.
- Breadth and delay carry 0.30 together, but delay is "a property of the reporting process", so it may carry little transferable text signal.
- Kappa is not additive per row and depends on the predicted marginal distribution. Rank-only output is not enough. Calibrated posteriors plus a decode that respects the kappa structure matter.

**Unspecified scoring semantics that change the decision** (AWK and the harm QWK are defined only on rows where truth is defined):

- (A) rows are dropped from AWK/QWK when we predict unknown;
- (B) such rows count as worst-case error;
- (C) unknown maps to some numeric value (for example 0, which is maximal under-triage).

Rows where truth is unknown but we predict a value are not scored in AWK or QWK, only in MacroF1 and Exact. So a false "reported" is cheap and a missed "reported" is expensive under B and C. The plan chooses the abstention threshold that is robust across A, B and C (see Metric-aware training & decode) and raises the point as a reviewer question.

**True independent unit.** The prediction unit is the report (a row's output depends only on its own text and a train-fit model). The generalisation unit is the reporting organisation (manufacturer). The test set holds only organisations absent from train [stated], but train has no organisation column, so groups must be reconstructed from text (see Validation design).

**Pipeline stages, diagnosed separately.**

1. Abstention: P(harm reported). Check AUC, F1 and its threshold.
2. Severity given reported: ordinal 1-4. Check QWK on true-reported rows.
3. Breadth and delay: ordinal.
4. Priority composition: a function of harm, breadth, delay and the report. Oracle check: feed gold components in and measure how deterministic priority is.
5. Decode: abstention threshold, ordinal cutpoints, asymmetric expected-loss decode.

Class coverage replaces "candidate recall". Check the minority classes (harm 4, priority 8-9, breadth 2, delay 3) separately.

## Compliance regime

**Domain.** Text to structured labels (NLP classification, tagged Seq2Seq). Compute is **CPU** [stated]. A few hundred thousand train rows [stated]. No runtime cap is stated beyond "Compute: CPU".

**Explicit bans in the description, treated as hard constraints:**

1. The test set is used only to produce one-row-at-a-time predictions. No training, fine-tuning, pseudo-labelling, self-training, threshold or hyper-parameter selection, calibration, feature fitting, vectoriser, normaliser or embedding fitting on test text, no exploitation of the test distribution, and no decision informed by test data. This includes the CLAUDE.md "check that the CSV distribution resembles OOF" habit. That check may be logged but must never gate or change anything.
2. No back-tracking to source records. No external database, corpus or mirror, no near-duplicate hashing against external data, no org-recovery.
3. No hard-coded answer tables, id-to-record dictionaries, filesystem probes or leaderboard probing. The public-leaderboard use CLAUDE.md allows as a "soft signal" is therefore limited. Do not choose between variants by leaderboard response. The free CSV upload is used for format checks only.
4. No hosted or closed-source LLM at any stage, including distillation or pseudo-labels. Only open-weights models and self-trained pipelines are permitted.
5. No rule-only pipelines. Predictions must come from a learned model that reads the text. No keyword-to-field lookup, and also no keyword override layered on a model (for example "if 'died' then harm=4").
6. Reconstruct from `report_text` alone. The only input feature is the text.

**Conflicts between the description and CLAUDE.md or the general guidebook** (the challenge wins, per CLAUDE.md 2.4):

- **CPU vs the A10G assumption.** CLAUDE.md sections 1, 3, 6.2 and the skeleton assume an A10G, `device="cuda"` and "default for text: fine-tune a transformer". The challenge says CPU, and a few hundred thousand reports cannot be fine-tuned through a transformer on CPU inside about 1 h. The plan hard-codes `device="cpu"` and fixed thread and process counts as constants. It uses no `cuda.is_available()` and no `os.cpu_count()`, which CLAUDE.md still forbids. The A10G runtime language does not apply. The grader's core count and RAM are unknown (open question).
- **TF-IDF and n-grams are "grey" in CLAUDE.md 2.5 and 6.2.** This description bans only rule-only pipelines and expressly permits "self-trained pipelines". The plan keeps a trainable neural text encoder (hashed n-gram EmbeddingBag network, trained from scratch) as a co-primary arm and does not rely on a TF-IDF-plus-linear model as the complete solution. A hashing-based sparse linear arm is kept as a diverse, cheap member and as the fallback. This is a reviewer question.
- **CLAUDE.md 4A "verify the produced CSV distribution resembles OOF" and "check CV vs public-LB agreement".** These touch test distribution or leaderboard. The first is logging-only here. The second is demoted to a format sanity check.
- **Boilerplate check.** The description has no "from scratch" or "fine-tuning" label. Pretrained HF weights remain allowed but are CPU-bound (see Rejected options).
- **Time.** CLAUDE.md section 3 bans wall-clock branching. All counts are fixed constants. Time is logged only.

**Grey or ambiguous items** (a reading that is compliant under every plausible interpretation is chosen in each case):

- Hashing or TF-IDF features as inputs to trained models. Compliant reading: stateless hashing, with IDF and group statistics fit on train only.
- Reconstructing pseudo-manufacturer groups by clustering train text. It uses train only and no test rows, and is used for validation and weighting.
- Hand-coded parsers of dates and numbers, if the diagnostics show dates in the text. Allowed only as features into a learned model, never as a rule.

## Data findings

These are the diagnostics to run on `train.csv`. All numbers are UNVERIFIED hypotheses unless marked [stated]. For test, use only schema, row count, id format and length statistics for runtime planning.

| # | Diagnostic (train only) | Hypothesis (unverified) | Decision it drives |
|---|---|---|---|
| D1 | Shapes, dtypes, nulls, empty or very short texts, duplicated ids, `TR` prefix format, id order vs labels or text (is the file sorted by filer, label or date?) | Few rows near-empty. Ids likely shuffled. If ordered, rows are sibling-leaky. | Group construction, row-order leak audit |
| D2 | Parse `triage_record` for 100% well-formed rows. Marginals of harm, breadth, delay, priority. Assert priority-unknown iff harm-unknown. | ~60% unknown [stated]. Harm 4 (death) rare, probably a few %. Breadth skewed to 0. Delay skewed, with mass in the 1-2 bins. | Class weights, minority-class handling, cumulative-head design |
| D3 | Crosstab priority x (harm, breadth, delay). Conditional entropy H(priority given the three), fraction explained by the modal mapping, spread of AWK/QWK of the modal mapping on train. | Priority is nearly a deterministic function of the three components, with ties plus some noise ("exact weighting unpublished"). | Whether to compose priority from component posteriors or predict it directly (or both) |
| D4 | Delay vs harm, delay vs breadth, breadth vs text length, priority gain when delay is short and harm is high | Serious cases are reported faster, so delay correlates with harm. Breadth correlates with text length and sentence count. | Cross-task features and multi-task sharing |
| D5 | Text length in chars and tokens (quantiles 5/50/95/99), share of text beyond 384 tokens, count of `(b)(4)` and `(b)(6)` tokens, digits and date-like strings per report, share of upper-case text | Terse reports with a heavy tail of long narratives. Redaction counts relate to patient presence. | Truncation policy (head plus tail), size features, whether a date feature is worth building |
| D6 | Exact and near-duplicate texts. Label agreement within exact-duplicate sets per target. | Heavy boilerplate duplication. Identical texts carry different labels at a measurable rate (irreducible ambiguity). | Noise ceiling, label-noise robustness, dedup of validation groups |
| D7 | Template structure. Share of reports whose first 10 or last 10 normalised tokens repeat at least 50 times, sentence-hash document frequency histogram, capped union-find components over mid-frequency boilerplate sentences, component size distribution. Per component: harm-unknown rate, mean severity, delay mix, length. | Large filer-specific template families. Per-filer label rates differ a lot (reporting style, not clinical truth). | Pseudo-manufacturer groups, group-capped weights, group-df feature filter |
| D8 | Filer-identification test: how well does a model on templates only (or cluster-id target encoding) predict each target under random vs cluster-grouped CV? | Large random-vs-grouped gap, especially for delay, breadth and abstention. | Size of the shift, which targets are safe to trust |
| D9 | Floors under three CV schemes: majority, length-only model, hashed unigram logistic. | Gap ladder: random CV > near-dup-grouped > template-cluster-grouped > coarse topic-grouped. | Calibrating the private band (bias direction) |
| D10 | Read 50 reports per harm class (read-only). Do "no injury" or "no patient involvement" phrases map to `unknown` or to harm=1? Where in the narrative do outcome statements sit (end of text?). | The unknown class is defined by absence of an outcome, but "no outcome" statements are ambiguous. Outcome statements sit anywhere, plausibly late. | Truncation policy, feature design, an honest ceiling for abstention |
| D11 | Date and sequence mentions (years, month names, "event date", "manufactured"). Does any report state both an event date and a received date? | Unknown. If present, delay has real transferable signal. If not, delay is mostly a filer artefact. | Whether a date-difference feature is worth a roadmap step |
| D12 | Top-weighted n-grams per target after a quick sparse fit (read-only, grouped CV). | Lexical cues (death, hospitalized, surgery, replaced, no injury) are the dominant signal for harm. | Sanity of the lexical arm |
| D13 | Oracle checks. Gold harm/breadth/delay through the composer then the metric. Gold probabilities through the decode must reproduce every gold record. | Perfect record reproduction, composed-priority AWK below 1 only by modal-mapping noise. | Decode and parser correctness, ceiling for composition |
| D14 | Metric unit tests: perfect=1, constant about 0, reversed < 0 (clipped), base-rate about 0, under-triage by 2 scores worse than over-triage by 2, each of readings A/B/C. | n/a | Metric correctness before any model work |

Expected size of the information ceiling (UNVERIFIED hypothesis): labels are derived from structured report fields, and the narrative only partly reflects them. The text-only ceiling for priority, breadth and delay is probably well below 1.0, and delay is probably the weakest.

## Validation design

**How the test split was made.** Group split by manufacturer, with test organisations absent from train [stated]. We have no organisation column, so we reconstruct pseudo-manufacturer groups from train text only.

**Group construction** (train only, never touches test; no external data):

1. Normalise text (lowercase, digits mapped to a placeholder).
2. Link reports that share a mid-frequency boilerplate sentence or a repeated opening or closing token window. Use union-find with a hard cap on component size (for example no component above about 3% of rows) so a generic sentence cannot create a giant component.
3. Residual singletons: assign to a coarse cluster (MiniBatchKMeans over an SVD of word-n-gram counts, K about 200, fit on train only) so every row has a group.
4. Group k-fold: balanced by group size (5 folds in the shipped script, 3 shuffled group-to-fold seeds during dev).
5. Report mean and std across folds and seeds. Accept a change only if it beats the paired fold noise and is consistent on at least 2 of 3 seeds. Bootstrap by group for confidence intervals.

**Scheme ladder** (each with the direction of its bias):

| Scheme | Expected bias vs real private score |
|---|---|
| Random k-fold | Optimistic. Filers recur, so templates leak. Likely a large gap on delay, breadth and abstention. |
| Near-duplicate-grouped | Still optimistic. Same filer, different wording. |
| Template-cluster-held-out (the target scheme) | Closest to deployment if clusters resemble manufacturers. Mildly optimistic if a manufacturer spans several clusters (leak across folds). |
| Coarse topic/device-held-out | Pessimistic by design. Removes whole device families that other manufacturers also make, and removes training data. |

Expected private band is between the third and fourth rows. State all four numbers in the final report and calibrate the band from them.

**Nested checks for every post-hoc step.**

- Stage-2 composer: second-level grouped CV on the OOF stage-1 posteriors.
- Abstention threshold, ordinal cutpoints, calibration maps: cross-fitted (estimate on 4 folds' OOF, apply to the fifth, report the stitched score).
- In-script alpha, weighting-power and group-df searches use a held-out inner fold that is excluded from the selection set.
- Dev only: a 10% group holdout fold untouched until the final run.

**Metric re-implementation.** Implement AWK as a family parametrised by an under-triage multiplier lambda (weights `w(p,t) = ((p-t)/(K-1))^2 * (1 + (lambda-1) * 1[p<t])`, lambda in {1, 2, 3, 5}), QWK, MacroF1, Exact, and the composite under readings A, B, C. Unit tests as in D14.

**What the proxy over- and under-states.** Our AWK is a proxy of an unpublished measure, so absolute values are approximate. Compare designs on the same folds only. Plan for the real number to land below the template-cluster CV because clusters probably under-separate manufacturers.

## Overfit/underfit risks

**Overfit risks.**

- *Filer-template memorisation.* High-cardinality n-grams (`(b)(4)`, boilerplate lines) let the model identify the filer and learn that filer's label rate, which will not transfer. Mitigations: (i) keep an n-gram only if it occurs in at least g distinct pseudo-clusters (g from {1, 3, 10}, chosen in-script); (ii) cluster-capped sample weights `w proportional to n_cluster^-p` with p from {0, 0.5}, chosen in-script; (iii) n-gram dropout (10-20%) in the neural arm as augmentation; (iv) strong L2 end included in the alpha grid.
- *CV leakage* (random or near-duplicate splits). Mitigation: group k-fold as above; treat random CV only as an upper reference.
- *Many tuned knobs* (alpha per head, threshold, cutpoints, lambda, weighting). Mitigation: low-dimensional, cross-fitted, fixed grids, nested held-out score reported.
- *Stacking on shared OOF.* Mitigation: second-level grouped CV and a dev-only holdout.
- *Unpublished AWK weights.* Decode lambda is a modelling assumption, not a fitted number. Choose it by worst-case regret across lambda_eval (see Metric-aware training & decode).
- *Fold-averaged vs refit test inputs for the stage-2 composer.* Use the fold-model average for stage-2 inputs because the composer was trained on 80%-data OOF posteriors. Compare against a 100% refit on the dev holdout and ship the refit only if it clearly wins.

**Underfit risks.**

- *Truncating long narratives.* The sparse arm sees the full text. The neural arm uses the first 384 plus last 128 tokens. D10 decides whether outcome statements are late.
- *Over-regularising the rare lethal cues* (harm 4, priority 8-9): cumulative at-least-k heads, per-head alpha chosen on the exact metric, not on loss.
- *Weak delay and breadth signal.* Accept that these may be near the floor. Choose strong regularisation by grouped CV rather than forcing capacity.
- *Information discarded by hashing collisions.* Use 2^21 buckets for word 1-2-grams. Check collision sensitivity once on dev.
- *Misspellings and abbreviations.* Word n-grams miss them. Roadmap step adds hashed character 4-grams per token in the neural arm only if memory and time allow.
- *Abstention hurts everything downstream.* A poor abstention threshold hurts MacroF1 plus AWK, harm QWK and Exact together. Tune it on the full composite under all readings.

## Recommended approach (primary + fallback)

**Primary (lean, two-arm, three-stage).**

1. **Representation (trained from scratch, CPU).**
   - Normalise: lowercase, digits mapped to a placeholder, redaction markers kept as tokens, whitespace collapsed.
   - Arm A (sparse): stateless hashing of word 1-2-grams into 2^21 buckets, sublinear TF with L2 norm per row, IDF and group-df filter fit on train only. No char n-grams in the primary because of memory (see budget).
   - Arm B (neural): multi-task EmbeddingBag network. Hashed word 1-3-grams (2^20 buckets, dim 96), LayerNorm, dropout 0.3 plus n-gram dropout, one hidden layer (256, GELU), 18 output logits (one for every cumulative head: abstain 1, severity 3, breadth 2, delay 3, priority 9). Loss is BCE per head with the metric's own weights (priority 0.35, others 0.15 each) and masking so severity and priority heads train only on rows with a reported outcome. Fixed schedule: 3 epochs, linear LR decay, SparseAdam for the embedding plus AdamW for the dense part.
2. **Stage 1 outputs (grouped 5-fold OOF, fold models averaged on test).**
   - Arm A: SGD logistic heads for abstain (1), severity (3 at-least-k, on reported rows only), breadth (2), delay (3). Priority is not modelled in Arm A, to save time.
   - Arm B: all 18 heads.
3. **Stage 2.**
   - Abstain, severity, breadth, delay: logit averaging of A and B with 2 weights per head, fit on OOF, then per-head cross-fitted calibration.
   - Priority: one small LightGBM multiclass model (10 classes, shallow trees, fixed rounds, `deterministic=True`, fixed thread count) over the calibrated stage-1 posteriors (A and B) plus a handful of size features (log length, sentence count, redaction count). Trained on reported rows only. Compare against a direct priority head from B and against a purely composed priority (D3), and keep whichever wins under nested grouped CV.
4. **Decode (OOF-fitted, cross-fitted).**
   - One abstention decision per row. If predicted unknown, emit `priority=unknown|...|harm=unknown`. Else decode severity, breadth, delay by cutpoints on expected value (QWK) and priority by asymmetric expected-loss minimisation over the stage-2 posterior.

**Why this fits the data.** Text signals are lexical and local (outcome phrases, number of failure descriptions). A hashed n-gram representation sees all 300k reports in minutes on CPU, a trainable embedding network adds nonlinearity and shared multi-task structure, and the group-aware filters aim at the real risk, which is filer shift rather than capacity.

**Fallback.** Arm A alone with stage 2 reduced to logistic regression and a composed priority. Use it if the neural arm fails to beat Arm A by more than the paired grouped-CV noise, or if the profiled runtime exceeds the budget. No runtime branching: the choice is made at dev time and the shipped file contains one code path.

**Optional third arm (only if profiling and CV justify it).** Fine-tune a tiny pretrained encoder (candidates such as `google/bert_uncased_L-2_H-128_A-2` or a similar mini model, revision pinned) on a fixed seeded subsample of 60-100k reports, max length 128 to 192, one epoch. It brings a genuine pretrained-fine-tuning component, which strengthens compliance, and it enters only if it earns a measured gain over noise and fits the budget. Throughput on the real CPU is unverified.

**Expected private composite (ESTIMATE, not a promise, no data seen).** About 0.25 to 0.50, central guess around 0.38. Reasoning: abstention MacroF1 roughly 0.65-0.78, harm QWK 0.35-0.6, priority AWK 0.3-0.55, breadth QWK 0.1-0.4, delay QWK 0.0-0.25, Exact about 0.05-0.2 (all hypotheses from the task description alone). Filer shift probably costs most on delay, breadth and abstention.

## Rejected options

- **Fine-tuning a transformer on all rows on CPU.** Back-of-envelope (unverified): a MiniLM-sized encoder (about 11M non-embedding parameters) at 128 tokens costs roughly 3 GFLOP per document forward only. 300k+ documents is on the order of 1e15 FLOP, which is hours on a handful of CPU cores. Rejected for the primary. Tiny encoders on a subsample are kept only as an optional arm.
- **Frozen sentence embeddings plus a linear head.** Same inference cost, and grey under CLAUDE.md for fine-tuning-style tasks.
- **Keyword/regex rules or a lexicon-to-field lookup, or a rule override on top of the model.** Explicitly banned ("rule-only pipelines"), and fails the strip-the-ML test.
- **Fitting any vectoriser, IDF, scaler, PCA or cluster on train+test.** Banned. Only train-fit statistics.
- **Pseudo-labelling or self-training on the test set.** Banned.
- **External MAUDE or any mirror, LLM-generated labels, closed LLM APIs, distillation from them.** Banned.
- **Joint 492-way record classifier.** Sparse classes, throws away ordinal structure, and Exact is only 5% of the score.
- **Character n-gram hashed matrices in the primary.** Estimated nnz per document is several thousand, which is multiple GB at 300k rows. Revisit only inside the neural arm and only with measured memory.
- **Char-CNN or BiGRU from scratch.** Too slow on CPU for 5 folds at this size. Roadmap only if the lean design leaves large headroom.
- **Per-filer features, filer target encoding, id or row-order features.** Fragile under the manufacturer split.
- **GBDT over raw TF-IDF.** Slow on CPU and grey.
- **Sibling or neighbour pooling across test reports (dedup, test-time clustering).** Banned.
- **Constant or always-abstain outputs for any component.** Score near 0 on that term.

## Fixed work plan & runtime budget

All counts fixed constants. Device `cpu`. Fixed worker and thread counts (assumed 4 processes or threads, hard-coded, unverified against the grader). Seeds fixed for `random`, `numpy`, `torch`, `PYTHONHASHSEED`, sklearn `random_state`, LightGBM `seed` plus `deterministic=True`, DataLoader `Generator`. Time is used for logging only.

**Estimates are unprofiled, assuming 4 vCPU, about 300k train rows, about 250 tokens per report, 16 GB RAM.** Profile on the dev machine, then hard-code the counts.

| Stage | Fixed plan | Est. minutes |
|---|---|---|
| 1. Load, normalise, parse labels | one pass | 1.5 |
| 2. Pseudo-groups | sentence/prefix/suffix hashing, capped union-find, K=200 residual clustering | 2 |
| 3. Hashing features, train and test | 2^21 word 1-2-gram sparse matrix, float32, about 1.2 GB; stateless | 3 |
| 4. Group-df filter and cluster weights | one sparse pass | 1 |
| 5. In-script search (alpha grid of 5, p in {0, 0.5}, g in {1, 3, 10}) | on a fixed seeded 100k subsample with inner held-out fold, Arm A only | 3 |
| 6. Arm A 5-fold | 9 heads x 5 folds, 4 parallel workers | 4 |
| 7. Arm B 5-fold | 3 epochs x 5 folds, about 30 s per epoch estimated | 8 |
| 8. Stage 2, calibration, decode search | LightGBM priority model with second-level 5-fold, 2-weight blends, cutpoints and abstention grid | 5 |
| 9. Test inference, validation, write | fold-average logits, validator, re-read CSV | 2 |
| Total | | about 30 min nominal |

Ceiling: 45 min worst case. This leaves roughly 40-50% headroom against a 50-60 min cap, and at least 30% against any stricter unstated limit. If the dev profile exceeds 35 minutes, cut in this order at build time (never at run time): char features, Arm B epochs, Arm B folds, Arm B entirely (fallback).

**Memory estimate.** Sparse matrix about 1.2 GB, Arm B embedding table 2^20 x 96 x 4 B = about 0.4 GB plus sparse optimiser state about 0.8 GB, LightGBM data small. Peak about 5-6 GB (unverified).

**Validation inside the script.** Parse and range-check every produced record against the regex-equivalent contract, check `priority` and `harm` unknown together, check the id list against `sample_submission.csv` order, check no NaN, row count and no empty string, then re-read the written file with `keep_default_na=False` and re-validate.

**Early fallback.** The script writes no early constant submission (constant output scores about 0 and hides failures). Failures must be loud.

**Dependencies.** numpy, pandas, scipy, scikit-learn, lightgbm, torch (all in the core stack). No try/except import fallbacks.

## Metric-aware training & decode

**Back-solving the metric.**

- Targets are ordinal, so every ordinal output uses cumulative at-least-k BCE heads. Expected value of the cumulative probabilities gives a continuous score for the cutpoint decode.
- The loss weights follow the metric's own weights: priority 0.35, abstention, severity, breadth, delay 0.15 each. Rows are masked to the rows the metric scores (severity and priority only on rows with a reported outcome).
- The metric has no per-example weights and no hierarchical averaging. The only "group" structure is filer shift, handled by weighting and filtering.

**Abstention.** P(reported) from stage 2, calibrated. Choose the threshold `t` on cross-fitted OOF to maximise the composite under each reading A, B, C and pick the `t` with minimum worst-case regret. The search is bounded by the label prior (about 40% reported) so it cannot reach degenerate corners (always or never abstain). Because false "reported" rows are cheap and misses are costly under B and C, expect a threshold below the pure-MacroF1 optimum. The measured cost of this choice versus the F1-optimal threshold is reported.

**Severity, breadth, delay.** QWK is maximised by decoding an expected value with ordered cutpoints tuned on grouped OOF (cross-fitted), not by argmax. Severity cutpoints use only true-reported rows.

**Priority (AWK).**

- Stage-2 posterior q over the 10 priority classes (reported rows).
- Decode is the class that minimises expected asymmetric squared loss `sum_t q_t * w(p,t)` for an under-triage multiplier `lambda_dec`.
- `lambda_dec` is not fittable (the formula is unpublished). In-script, evaluate each candidate `lambda_dec` in {1, 2, 3, 5} on OOF against each `lambda_eval` in {1, 2, 3, 5} and choose the `lambda_dec` with minimum worst-case regret. This keeps the constant derived by in-script search while staying robust to the unknown weights.
- Compare against a cutpoint decode on E[priority] shifted upward, and keep the better one under nested grouped CV.

**Calibration.** Per-head Platt or isotonic with very few parameters on grouped OOF, reported cross-fitted. Cumulative heads are made monotone after calibration.

**Hard constraints at decode.** `priority` and `harm` unknown together. Ordinal heads monotone. Breadth in 0-2 and delay in 0-3. Priority posterior restricted to 0-9.

**No joint decoder.** Exact is 5% and is handled by coherence of independently decoded fields. A joint-record decoder enters the roadmap only after a measured gain over noise (primary-design gate).

## Structural signals

- **Abstention gate.** `priority` and `harm` are unknown together, so the model has one decision (reported or not) and a severity and priority model conditional on reported. This couples three score terms and is enforced at decode.
- **Priority as a function of components.** Priority is driven by harm, raised by breadth and by fast reporting [stated]. D3 quantifies how deterministic this is. If it is nearly deterministic, the easier direction (components from text) feeds a small learned composer, and the harder one (priority from text directly) is the comparison arm. The mapping is learned by the stage-2 model on train labels, not hard-coded.
- **Ordinal cumulative structure.** All four outputs and priority use at-least-k heads and monotone calibration.
- **Multi-task sharing.** One encoder for all heads lets the harm signal support priority, and lets delay and breadth borrow lexical structure.
- **Delay and harm coupling.** Serious reports are expected to be filed faster. Stage 2 sees the stage-1 harm posterior when predicting delay and priority.
- **Breadth and length.** Breadth tracks the number of distinct failure descriptions. Size features (log length, sentence count) enter stage 2 as inputs to a trained model. Test their ablation under grouped CV because boilerplate length is filer-specific and can be shift-fragile.
- **Sibling and duplicate structure.** Duplicates and templates are groups first (validation and weights), never test-time pooling.
- **Redaction tokens.** `(b)(4)` and `(b)(6)` counts are features for the model, filtered by group-df so only filer-independent uses survive.
- **Augmentation.** n-gram dropout and random span dropout of the text are label-preserving augmentations of real rows (not synthetic data).
- **Possible date structure (conditional on D11).** If reports state event and received dates, a parsed date-difference feeds the delay head as one input to the learned model. It is not a rule.

## Experiment roadmap

1. **Contract, metric, validation (stop when all D13/D14 tests pass).** Implement the legal-record renderer, validator, AWK family, QWK, MacroF1, composite under readings A, B, C. Build pseudo-groups and the four-scheme ladder. Stop criterion: gold components reproduce gold records, perfect/constant/reversed/base-rate behave as specified, group fold sizes balanced with no giant group.
2. **Cheapest end-to-end baseline (valid submission).** Arm A only, default alpha, no filters, composed priority, naive decode. Record the ladder (random vs grouped). This is credit 1 (baseline).
3. **Representation and structure.** Add group-df filter, cluster weights, per-head alpha search, stage-2 priority model. Stop when additional steps do not beat paired grouped-CV noise on the composite or on the weighted terms.
4. **Metric-aware decode.** Abstention threshold under A/B/C, cutpoints, lambda_dec by worst-case regret. Cross-fit everything. Stop when the nested gain is within noise.
5. **Diversity of assumption.** Add Arm B (neural, multi-task). Score each arm alone first. Blend only if the two are close in quality. Then the optional tiny pretrained fine-tune arm only if D-checks say CPU throughput allows. Stop when a member does not add more than noise. This is credit 2 (best single) and credit 3 (ensemble).
6. **In-script bounded HPO** with fixed trial counts (alpha grid 5, p in 2, g in 3) and its own held-out check. Stop if the selection gap between inner and outer scores is larger than the gain.
7. **Final fixed-plan run from a clean `working/`, run twice and diff the outputs** (a determinism check is not test tuning). Credit 4 (final).

Never choose between variants by leaderboard score. Keep a dev-only group holdout that is untouched until the end.

## Compliance audit

CLAUDE.md section 7 and the agent self-audits, against the plan:

- Test read for anything but per-row prediction (stats, vocab, scaler fit, clustering, dedup, rank normalisation, pseudo-labels)? No. All fitting (IDF, group-df, clusters, weights, calibration, cutpoints) is train-only. Hashing is stateless.
- Wall-clock in any condition? No. Logging only.
- `cuda.is_available()`, `os.cpu_count()`, import fallbacks, try/except switching models? No. `device="cpu"` and thread counts are hard-coded.
- Hard-coded tuned constants? Alpha, weighting power, group-df, threshold, cutpoints and lambda_dec are chosen by in-script search on train-only OOF. `lambda_dec` candidates and evaluation grids are fixed design constants justified by the unpublished metric, not tuned from leaderboard feedback. Network sizes (96-dim, 256 hidden, 3 epochs) are fixed design choices and are noted as such.
- External data, synthetic data, hosted weights, closed APIs, non-allowed libraries? None.
- Strip-the-ML test: remove the trained models and nothing remains (no keyword tables, no rule overrides, no hand-written regex solving a field). Pass. The only hand-written parts are generic normalisation, group construction (train-only, used for validation and weighting) and format validation.
- Does the model-heavy part dominate? Yes. Two trained models plus a trained composer generate every field.
- Whole-test aggregation: none. Each row's output is a function of its own text and train-fit state. No rank or z-score normalisation across test rows.
- Sibling leakage: validation uses reconstructed groups, never random rows.
- Challenge-specific restrictions honoured: CPU, text-only input, no external lookups, open-weights only, no test use for selection or calibration, no leaderboard probing.
- Source under 512 KB, plain readable code, comments explaining reasoning, fixed seeds and workers: required in the build.
- The rest of the CLAUDE.md section 5 checklist (validator, clean-run command, columns and order) applies unchanged.

## Open questions & assumptions

**Reviewer questions (highest impact first).**

1. For AWK and harm QWK, what happens to rows where truth is defined but we predict `unknown` (and the reverse for the score terms that ignore truth-unknown rows)? Plan under each reading: (A) dropped, (B) worst-case, (C) mapped to 0. The plan picks the threshold with minimum worst-case regret across all three.
2. Is a hashed n-gram plus trained-embedding classifier (no transformer) accepted as the primary given CLAUDE.md marks TF-IDF and n-grams as grey, and given that the challenge is CPU-only and bans only rule-only pipelines?
3. What are the grader CPU core count, RAM and runtime cap? The description only says "Compute: CPU". The plan assumes 4 cores, 16 GB and CLAUDE.md's about 50 min target, 1 h worst case. If fewer cores, the fixed worker constants become a runtime risk.
4. May `priority` be numeric while `harm=unknown`? The plan assumes no (they are "unknown together, always") and never emits that.
5. Is it acceptable to use pretrained open-weights HF encoders (tiny, CPU) as an optional arm, with pinned revisions and internet access at run time?
6. Is a train-only clustering of report text to build pseudo-manufacturer groups for validation and sample weighting acceptable? (The plan believes yes: train only, no test rows.)

**Assumptions to verify at build time.**

- About 300k train rows, 250 tokens per report, 60% unknown, as stated. Row counts and lengths on test are used only for runtime planning.
- Priority is nearly a function of (harm, breadth, delay) (D3).
- Pseudo-groups approximate manufacturers well enough (D7, D8).
- Outcome statements are lexically detectable and not hidden only in filer-specific templates (D10, D12).
- Runtime estimates are unprofiled and must be measured before the counts are frozen.

**Could not verify in this run.** Everything in Data findings, all runtime and memory numbers, the performance band, the neural arm's throughput, the existence and revision of any pretrained mini encoder, and the exact AWK weights.
