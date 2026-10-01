# U10 plan: prior-art recommendation with contentless candidates

Scope of this plan: written from the challenge description (tasks/user/U10_prior_art_contentless.md), CLAUDE.md and the strategist procedure only. No dataset file was available, so every statement under "Data findings" is a diagnostic to run or an UNVERIFIED hypothesis. Numbers marked "derived" are arithmetic on figures stated in the description. No solution code is included.

## Contract & decision unit

**One valid answer.** For each of the 10,106 rows of `test_queries.parquet`, one `query_id` (string) and one `candidate_ids` string of at most 50 distinct pool ids separated by single spaces, best first. Always emit exactly 50: extra distinct ids can only add gain, and a short list can only lose it.

**Invalid vs low-scoring.**
- Invalid, so the file is rejected: a missing, duplicate or unknown `query_id`; wrong row count (10,106 plus header); wrong columns or names; NaN or empty cells.
- Silently wasted: repeated ids (ignored after the first) and ids outside the pool (never match).
- Low-scoring but valid: the constant most-cited list (0.0236 on the sealed test set).
- Id dtype trap: candidate and query ids are salted hashes. Read every csv and parquet id as `str` (`dtype=str`, `keep_default_na=False`), otherwise strings like `1e5...` can be parsed as floats. Re-read the written file the same way and assert that every id is in the pool.

**Metric, term by term.** For query q with n_q references inside the pool: DCG@50 = sum over references found at rank r<=50 of 1/log2(r+1), with binary gain. This is divided by IDCG = sum_{r=1..min(n_q,50)} 1/log2(r+1). The score is the plain mean over the 10,106 queries.
- A hit weighs 1/IDCG_q, so a query with 1 reference weighs its single hit far more than a query with 10 references. This is hierarchical averaging: each query counts equally, so positives of short-list queries carry more weight.
- It is rank-only and not calibration-sensitive across queries. Only the order inside a query matters.
- Nothing gates the score. Zero is possible per query. The strongest reference scores exactly 0 on 22.8% of queries, with a per-query median of 0.158.
- Rank-1 is worth 1.0, rank 2 about 0.63, rank 10 about 0.29, rank 50 about 0.18 per hit (the unnormalised discount).

**True independent unit.** The query (a citing work) is the unit for training, validation and scoring. Candidates are not independent units, because each is only "the company it keeps". Related queries (same authors, versions of the same study, series papers) form clusters, so validation groups must be derived (see Validation design). Test is explicitly per-query: the answer for one query is a function of its own text, its own field and train-fit artifacts only.

**Pipeline stages (diagnosed separately).**
1. Candidate coverage: the recall of the union candidate pool over the query's references, which is bounded by candidates with zero or very few train citations.
2. Scoring and ranking: how well the pooled candidates are ordered, given that the answer is reachable.
3. Decoding: sort by the marginal probability of being a reference. Under linear-gain NDCG with a per-query constant normaliser, this is exactly optimal. Diversification (MMR and similar) would lower expected NDCG.

## Compliance regime

**Domain.** Retrieval/recommendation with collaborative-filtering structure (contentless items) plus text. Under CLAUDE.md this maps to section 6.4 (RAG/retrieval/ranking) and the section 2.5 grey areas. The task is not labelled fine-tuning or from-scratch, but the model must demonstrably learn.

**Explicit bans and hard constraints from the description (all treated as hard).**
1. No identifying the underlying works and no matching published queries against OpenAlex, Crossref, a search engine or any other copy of the literature. No recovering a reference list by external lookup. In practice: no external data, no LLM guesses of references, no use of any external literature index.
2. Each query is answered on its own. The text of a query and what was learned from the training queries are the only inputs. NOT allowed: using other test queries collectively. This rules out:
   - kNN or graph links among test queries, and test-set clustering;
   - hubness correction, CSLS and similar normalisation computed over test queries;
   - rank or z-score normalisation of scores across the test file;
   - a TF-IDF, vocabulary, PCA/SVD or scaler fitted on train+test or on test;
   - test-set deduplication, sorting or batching by test-set statistics;
   - batch-statistic layers active at inference (use eval mode; padding must not change any query's output).
3. Do not fit, calibrate or select against the test queries as a set. Do not pool statistics over them. Do not derive labels from them and train on those. This also rules out pseudo-labels, self-training and transductive anything. Every threshold, K, temperature, epoch and round count is chosen by train-only search inside the script.
4. No network access at run time. Pretrained text encoders are allowed "when available offline and licensed for competition use"; fine-tuning them on training queries is allowed. This overrides CLAUDE.md section 1 (HF download at run time). The encoder must therefore already exist offline in the grading environment (see Open questions, Q1).
5. Runtime: the full script, from reading `./dataset/public/` to writing `./working/submission.csv`, within 90 minutes on one A10G. CLAUDE.md tightens the planning target to about 50 minutes or less with 30% or more headroom (see budget). The script still takes `sys.argv[1]` and `sys.argv[2]` per CLAUDE.md, defaulting to the two paths above.
6. Licensing: pick encoders with permissive licences (MIT or Apache style). Avoid non-commercial licences.
7. Inherited from CLAUDE.md: determinism (no wall-clock or environment-dependent branches), no `try/except` import fallbacks, fixed seeds, source under 512 KB, only allowed libraries.

**Where the description is silent, and my assumption.**
- Which encoders exist offline: unknown. Assumption: at least one general-purpose small/base English encoder is installed in a fixed local path. This must be confirmed (Q1). Because CLAUDE.md forbids a runtime fallback, the encoder choice is one constant set before submission, not a runtime switch.
- CPU core count and RAM: unknown. Assumption: 8 cores and 32 GB or more. All thread counts are hard-coded constants (no `os.cpu_count()`).
- Whether the query's `year` and `field` may be used as inputs: not banned. They are columns of the row's own input. Test years (2023, 2024) are outside the training range, so the raw year is never a feature. The relative gap feature (query year minus statistics cutoff) is optional and decided by measurement.
- Whether the label-derived statistics of candidates (citing-query centroid, field histogram, co-citation) are allowed: the description says candidates are placed "through the works that cited it in the training period", so yes. They are built from train only.
- Whether `train_citations` may be used: it is shipped data. For validation origins it must be recomputed from `train_references.csv` restricted to the cutoff, never read directly (see Validation design).

**Self-audits on the plan.**
- *Strip-the-ML test.* Remove the trained parts (two-tower head with learned candidate embeddings, LambdaRank reranker, any LP-FT layers). What remains is a kernel kNN vote and a TF-IDF/embedding centroid cosine, which is exactly the 0.20 reference class. Those non-parametric scorers are grey, so they enter only as inputs to trained scorers. The ablation must show the trained stack clearly above them under the same validation (target at least +0.015 NDCG, more than 3 paired standard errors); otherwise the compliance argument is weak.
- *No whole-test aggregation.* Every test feature is a pure function of the query row plus train artifacts (see the ban list above).
- *Sibling leakage.* The target is "is c a reference of q", not a relation between group-mates. The only within-row cross-candidate features (support from companion candidates in the row's own first-stage list) use a train-fit co-citation matrix. Label-derived statistics are computed out of time (cutoff) and with the query's own group excluded.
- *Hard-coded constants.* Every constant (kNN K, kernel temperature, embedding dims, weight decay, aux-loss weight, rounds, pool size) is searched in-script on train-only rolling-origin dev data with its own held-out year, or is a standard fixed default stated up front. Nothing is read off earlier submissions.
- *Genuine training load-bearing.* Report the parameter-change norm of any LP-FT layers and the CV gain over the frozen probe. The head is trained end-to-end (29.7k learned candidate vectors plus a query MLP). A frozen-encoder-plus-GBDT-only design is not proposed.

## Data findings

None of this is verified. All of it needs the actual train files. Hypotheses are marked (H). Arithmetic on figures from the description is marked "derived".

**Derived from the description.**
- Training reference rows are about 25,618 x 6.7, roughly 170k pairs (derived, assuming the quoted mean applies to training). The user-item matrix is 25.6k x 29.7k with density around 0.02%.
- Mean train citations per candidate is about 170k / 29.7k, roughly 5.8 (derived). With a heavy popularity tail, the median candidate probably has only about 3 to 5 training citers. This is the central difficulty: a candidate prototype is built from a handful of abstracts.
- Test: about 10.1k x 6.7, roughly 68k reference instances, so about 2.3 per candidate. Pool rule "five citations in the source corpus" suggests train count plus test count is at least 5 for every candidate (derived). Candidates with a low train count (including 0) must have been cited in 2023 or later. Inclusion in the pool is therefore informative about future citations. That is an artefact of how the pool was built; flag it once as an "exploits data generation" reviewer risk and let a trained model learn it (see Structural signals).

**Diagnostics to run on train, with the hypothesis for each (all UNVERIFIED).**
1. Counts and shapes: queries per `year` (2015-2022) and per `field`; share of empty `field`; share of empty or very short abstracts; title and abstract token-length distributions (p50/p90/p99) to fix `max_length` (H: abstracts about 150-250 words, so 256 tokens truncates some, 384 is safer). References per query: min, quantiles, share with exactly 1 reference, share above 50, number of training queries with 0 pool references (H: none or almost none).
2. Candidate citation histogram from `train_references.csv`: shares with 0, 1-4, 5-9, 10-49, 50+ train citations. Check that `candidates.train_citations` equals the count recomputed from `train_references.csv` (H: equal; if not, understand why before using it). H: a substantial minority (perhaps 20 to 35%) has fewer than 5 train citations; a small share (low single digits of percent) has 0 train citations. Those are unreachable by any text-based method.
3. Popularity concentration: share of reference mass in the top-50 / top-500 candidates; share of queries citing at least one top-50 candidate (H: a "large minority", consistent with the description). Reproduce on a temporal dev split the 0.0236 "most-cited" figure as a metric sanity check.
4. Reachability ceiling at the pseudo-test (see Validation design): for 2022 queries with statistics from queries up to 2021, the share of references with 0 or fewer than 3 citing queries. H: this share is an upper bound on how much of the 22.8% zero-score mass is unreachable. Also the share of dev queries where all references are cold (guaranteed zero for text-based methods).
5. Neighbour oracle: for each dev query, the best single training query's reference list scored by NDCG (an upper bound on a one-neighbour vote); the same for the best of the top-20 text neighbours. H: the oracle is much higher than the achieved score, so the gap is ranking quality, not coverage.
6. Co-citation structure: distribution of pairwise co-citation lift among a query's own references; share of reference pairs that co-occur in at least one other training query. H: strongly clustered by sub-field, with a few "hub" references (methods, software, surveys) that are cited across fields.
7. Duplicates and groups: near-duplicate abstracts (cosine above 0.95), identical or near-identical reference sets (Jaccard 0.8 or more), and clusters by shared rare references (candidates with train count at most 20; a union-find edge needs 3 or more shared rare references, so popular hubs do not collapse the graph). Report cluster-size distribution, cross-year links, and the share of 2022 queries with a near neighbour in earlier years. H: a visible self-citation / same-series effect that is also present in the real test and so is not removed. It is used only to forbid sibling leakage inside train.
8. Temporal drift: per-candidate first-cited year and last-cited year; per-year Spearman correlation of candidate citation counts; for 2022 queries, share of references first cited in 2021-2022. H: drift is real; recent candidates are over-represented in later queries. Rate of change of the top-50 list across years.
9. Field structure: for each candidate, the field histogram of its citers (purity). H: many candidates are field-pure and field is a strong gate, while hub candidates are cross-field. Also whether empty-`field` queries have different reference counts.
10. Leakage-column audit: confirm `query_id` and `candidate_id` order carry no year/field/popularity signal (salted hashes; check correlation of row order with year); confirm no query is also a candidate; confirm that each query's references exclude itself.
11. Irreducible ambiguity: near-identical texts (cosine above 0.97) with different reference sets, to bound how much text can ever predict; and share of references that no embedding neighbour supports.
12. Stage diagnostics at the dev pseudo-test (to be reported after each pipeline change): recall@50/100/300/1000 of each generator and of the union; NDCG of the union-ordered pool by the oracle ordering (coverage ceiling); NDCG given the first-stage order; NDCG after reranking.

**Hypotheses that shape the plan (UNVERIFIED).**
- H1: candidates are sparsely observed (a handful of citers), so free per-candidate vectors overfit, and parametrising each as a text-derived prototype plus a small residual is safer.
- H2: a candidate's citers can be multi-topic, so the max or top-m similarity to its citers (local evidence) beats a single centroid.
- H3: a global popularity nudge helps only slightly (the reference gains 0.0085), while popularity-relative features inside a learned reranker help more.
- H4: there is a measurable shift between train-era and test-era citation rates for low-count candidates (the pool rule), which a trained model can learn from `count` and `n_total` features.
- H5: zero-score queries are mostly "unreachable reference sets" (all or nearly all references cold or without text-supported neighbours) and cannot be rescued by modelling.

## Validation design

**How the test split was made.** Strict temporal: train is works published before 2023, test is works from 2023 (so 2023 and 2024). The pool is fixed across the split. The pool inclusion rule uses counts that include test-era citations. The dev protocol must mirror all three: temporal order, a gap of zero to one year, and the same fixed pool.

**Rolling-origin scheme (replaces plain K-fold; random K-fold on this data is badly optimistic because same-series and near-duplicate works leak across folds and because there is no drift).**
- Define origins c in {2017, 2018, 2019, 2020, 2021} (the number of origins is a fixed constant).
- At origin c, all label-derived statistics (counts, citing-query centroids, field/year histograms, co-citation matrix, kNN index, two-tower fit) use only training queries with year at most c. Rows to be scored at origin c are queries of year c+1 (gap 0) and c+2 (gap 1). This mirrors the real test: statistics from at most 2022, scored rows 2023 (gap 0) and 2024 (gap 1).
- LEAKAGE TRAP: never use the shipped `train_citations` for origins before the final one. It counts citations up to 2022 and would leak future labels. Recompute counts per origin from `train_references.csv` and `train_queries.year`. Use the shipped column only at the final origin, and only after checking that it equals the recomputed count.
- Level-2 (reranker) rows: for every origin, every scored query contributes a slate of pool candidates (see Fixed work plan) with features computed at that origin. A query appears in at most two origins (gap 0 and gap 1). Always split level-2 data by query (all of a query's rows go to the same side).

**Dev (honest) evaluation.** Reranker trained on rows from origins 2017-2019 (queries up to 2021), evaluated on 2022 queries scored from origin 2021 (gap 0) and origin 2020 (gap 1). This is the nested hold-out: no 2022 query label touches level-1 or level-2 fitting, and level-1 knobs (K, kernel temperature, dims, weight decay) are chosen from origins before the held-out year (a separate selection year, 2021 evaluated from origin 2020). Report mean and standard deviation over seeds (3 seeds for the neural head; GBDT seed averaging), and a paired query-level bootstrap standard error for every comparison.

**Final shipped run.** Level-1 statistics and models refit on 100% of training queries (origin 2022). Level-2 trained on all origins 2017-2021 rows. The number of boosting rounds and the number of two-tower epochs are the fixed values chosen on dev (rounds times 1.1 for the larger data), as constants decided by an in-script train-only dev search.

**Groups.** The query is the unit. Derive cluster ids by union-find on (a) near-duplicate text, (b) shared rare references (train count at most 20, 3 or more shared), and keep all members of a cluster on one side of any reranker split. Because the main split is temporal, clusters spanning the cutoff are real and are kept (the real test has them too). Report the share of dev queries with a same-cluster training neighbour and the dev score with and without those queries, so the effect of self-citation is visible.

**Metric re-implementation and unit tests.** Implement NDCG@50 exactly as written (binary gain, discount 1/log2(r+1), IDCG over min(n_q,50) references). Tests: perfect ranking gives 1.0; reversed ranking (references last) gives a low value; a constant list gives the same score for every query; a base-rate/random 50 gives about 0.0007 on dev (the quoted random score is 0.0007); the most-cited list scores in the 0.02-0.03 range on dev; duplicate ids and ids outside the pool contribute nothing. Also reproduce the two public reference solutions on the dev protocol (TF-IDF centroid cosine, and the same nudged by popularity). If dev and the quoted real-test numbers (0.2022 and 0.2107) differ, use the difference to calibrate the expected private band (the measured gap transfers, with caution).

**Direction of the proxy's bias (estimate).**
- Pessimistic: the dev origin has about one year less training data than the final run (smaller statistics, rarer candidates have fewer citers). Rough size: a relative drop of a few percent.
- Optimistic: the dev pool uses candidates admitted under the full-corpus threshold (same as test), so this is roughly neutral. Gap 0 and gap 1 rows mirror the test (2023 and 2024); 2024 is one year further from the training end than anything dev can show beyond gap 1, so drift on the real 2024 rows may be somewhat worse than dev suggests.
- Net expectation: dev within about plus or minus 0.01 of the real private score for the same pipeline, measured against the reference solutions on both sides. This is an estimate to be checked against the reference calibration above.

**Selection discipline.** Every post-hoc selection (K, temperature, dims, weight decay, aux-loss weight, GBDT rounds and leaves, pool size, any blend weights, decode constants) has its own held-out year: choose on the selection year, report on the held-out year. A change is accepted only if the paired gain exceeds 2 paired standard errors and is not reversed across the two gap settings.

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Free per-candidate vectors for 29.7k candidates with a median of about 3-5 positives: memorisation. Mitigation: candidate vector = linear map of the mean (and top-m prototypes) of its citers' frozen embeddings + a small residual vector with a strong L2/weight-decay penalty (residual norm and its CV gain logged); dropout on query embeddings; a fixed small number of epochs fixed by dev; logQ-corrected sampled softmax over all candidates (the full softmax over 29.7k is cheap).
2. Reranker trained on in-sample level-1 scores: catastrophically optimistic (the model would see the query's own references reflected in kNN/two-tower scores). Mitigation: all level-2 features come from the temporal origin that has not seen the query (rolling origin above); kNN excludes the query's own reference list by construction (it is later than the cutoff); two-tower is trained on queries at most c.
3. Count-scale mismatch between train rows and test rows (statistics from 80% or less versus 100% of data). Mitigation: use rates (count / number of queries at the origin), not raw counts; keep the same origin structure at test (c=2022) and verify that the feature distribution (not the labels) shifts smoothly across origins on train data only.
4. Many tuned knobs on a roughly 4k-query dev year (per-query NDCG standard deviation about 0.25, so about 0.004 standard error of the mean; paired differences about 0.002): selecting noise. Mitigation: small grids, plateaus over peaks, one selection year plus a separate held-out year, and report the number of knobs tried.
5. Reranker overfitting by memorising popular candidates: min_data_in_leaf 200 or more, strong feature and bagging fractions, fixed rounds, monotone constraints are not used (keep it simple), L2.
6. Sibling and same-series leakage within the labelled level-2 data. Mitigation: group-aware splits (the unit is the query, clusters are kept on one side), and the temporal origin structure.
7. Exploiting pool-construction artefacts (low train count means guaranteed future citations): the effect is real but is a learned reweighting from generic count features, not a hand-coded constant. Flag once for reviewers.

**Underfit risks and mitigations.**
1. Too weak a text representation or truncated text: use a strong general-purpose encoder with `max_length` set from the token-length quantiles (384 tokens if p90 exceeds 256), mean + CLS pooling, and two views (title only, title + abstract) as separate embeddings. Compare 1 versus 2 encoders only if each earns a measured gain.
2. Low-rank bottleneck: dimension too small for 29.7k candidates. Mitigation: dims 256 to 768 in the grid; compare with the kernel kNN vote as a non-parametric yardstick.
3. Popularity over-penalised: the reference nudge helps; always give the reranker `log(rate)` and relative-rate features and let it learn the trade-off.
4. Candidate pool too small: the union pool recall@300 versus recall@1000 gap tells whether to enlarge. Mitigation: pool size selected on dev (fixed constant afterwards).
5. Loss mismatch: sampled softmax is not NDCG. Mitigation: LambdaRank reranker (NDCG@50 objective) on top, per-query weighting for the true IDCG, see Metric-aware.
6. Cold candidates (0 or few citers): irreducible for text; only popularity-free features (the count and pool-inclusion artefact) help a little. Do not spend model capacity here.

## Recommended approach (primary + fallback)

**Primary: retrieve-and-rerank over frozen text embeddings, with a trained two-tower head and a LambdaRank reranker, rolling-origin level-2 data.**

1. *Text representation (fixed, frozen, label-free).* Encode title, abstract and "title + abstract" with one general-purpose pretrained encoder (an MIT/Apache base-size English embedding model that exists offline), fp16, `max_length` from the token-length diagnostic, mean and CLS pooling, flip-less (text has no TTA beyond averaging two views). Also a train-fit TF-IDF (word 1-2 grams, vocabulary fitted on training queries only, transform for test) followed by a truncated SVD fitted on training queries only, for a second view of a different nature (diversity of assumption). No label is used here, so the same embeddings serve every origin without leakage; this is important for the rolling-origin design (fine-tuning the encoder would need per-origin cross-fitting).
2. *Level-1 generators, each producing scores for every candidate for a query (all fitted with statistics at origin c):*
   - (a) Nadaraya-Watson kernel vote: P(c | q) = sum over training queries q' (top-K by cosine) of k(q,q') * 1[c in refs(q')] / sum k(q,q'), with k = exp(cos/tau). This directly estimates the marginal reference probability, is popularity-aware through the vote mass, and uses local (neighbour-level) evidence. K and tau on dev.
   - (b) Local-evidence candidate score: for candidate c, log-sum-exp (or max, top-2 mean) over cosine similarities between q and the citers of c, minus a reference term from a bank of unrelated training queries (reference-normalised evidence), averaged over the two views.
   - (c) Prototype cosine (the public baseline class): cosine of q to the citer centroid of c, for TF-IDF-SVD and dense views. Feature only, never the core.
   - (d) Two-tower head (the genuinely trained part): query tower is an MLP over the frozen embeddings (concatenated views), candidate vector e_c = W * [mean(citer embeddings), top-m prototypes] + delta_c with a strong penalty on delta_c; training loss is multi-positive sampled softmax over all 29.7k candidates (logQ-corrected by the origin's train frequency), per-query-normalised over its positives (so each query counts equally, matching the metric), plus an auxiliary co-citation contrastive loss between candidates cited by the same query (structure as free signal). Fixed epochs, fixed seeds, 3 seeds averaged.
   - (e) Popularity prior: log rate at the origin, to fill the pool.
3. *Candidate pool per query:* union of the top-150 of (a), top-100 of (d), top-100 of (b), top-50 of (c) and the top-30 most popular; about 300 candidates after deduplication (K to be confirmed by recall@K on dev). Zero-train-citation candidates are not specially retrieved.
4. *Level-2 reranker: LightGBM LambdaRank (NDCG@50), binary labels, fed with a broad but small feature family (about 40 features):*
   - Generator scores and their ranks, plus within-slate transforms (z-score within the query's own pool, gap to the top, margin to the runner-up) from (a)-(d).
   - Candidate statistics at the origin: log rate, count-based features, citer-field histogram match to the query's `field` (smoothed), citer first/last/mean year and recent-share (last two years of the origin) as trend features, count of citers, and the pool-artefact feature max(0, 5 - count_c) so the model can learn the low-count shift.
   - Local evidence: max, top-3 mean and mean similarity to citers (dense and TF-IDF views).
   - Companion support: sum over the row's own top-20 first-stage candidates c' != c of PPMI(c, c') * p1(c'), where PPMI is fit from training co-citation at the origin (a propagation step that uses only the row's own list and train-fit statistics).
   - Query descriptors: abstract length, field id (categorical), number of pool candidates with non-trivial scores (as a confidence scale), and optionally the gap (query year minus origin cutoff).
   Per-query sample weight set to the ratio of in-pool IDCG to true IDCG, so the weighting matches the metric even when some references are outside the pool (to be checked against the library's weight semantics for ranking; if per-group weights are unsupported, drop it and measure the loss).
5. *Decode:* sort the pool by the reranker score, break ties deterministically (higher origin rate, then ascending candidate id), write exactly the top 50 distinct ids.

**Why this fits this data.** It is the leanest design that encodes the structural facts: (i) a candidate exists only through its citers (prototype and local-evidence features, collaborative structure); (ii) co-citation is the only way to place candidates relative to each other (auxiliary loss, companion support); (iii) temporal drift and pool construction are replicated at every origin so the reranker learns them; (iv) the loss and decode follow the metric (per-query normalised loss; marginal-probability ordering; NDCG lambdas).

**Expected result (an ESTIMATE, not a promise, based on reasoning alone).** Private NDCG@50 in roughly 0.21 to 0.27, with a central value near 0.23. Reasoning: the public references are 0.2022 (TF-IDF centroid) and 0.2107 (with popularity); a dense encoder, a local kernel vote, a trained head and a reranker typically add around 10 to 25% relative on such sparse citation data, but cold and unreachable candidates cap the ceiling (the best reference already zeroes 22.8% of queries). With 10.1k test queries the sampling noise on the mean is about 0.0025. If no pretrained encoder is available offline (Q1), the same plan with a trained-on-train text tower would likely land lower, perhaps 0.20 to 0.24 (estimate).

**Fallback (lean, no GBDT).** Keep the kernel vote (a), the two-tower (d) and the popularity prior, blend them by within-query z-scores with 3 non-negative weights chosen on the selection year (no cross-test normalisation), and sort. Switch to the fallback if the reranker's paired gain over the fallback is under 2 standard errors, if LambdaRank training destabilises, or if a reviewer rejects the GBDT as the final scorer. A second fallback for the missing-encoder reading: replace the pretrained views with a text tower trained on the training queries only (embedding-bag over train-fit vocabulary or a small transformer trained on the same multi-positive loss), as a single constant fixed before submission.

## Rejected options

- Most-cited list or popularity-heavy ranking: 0.0236, identical for every query.
- TF-IDF centroid cosine as the complete solution (the 0.2022 reference): kept only as one input; also grey under CLAUDE.md section 2.5 as a core method.
- Cross-encoder or any model that needs candidate text: candidates have no text.
- End-to-end fine-tuning of the encoder with a 29.7k-way softmax as the primary: it breaks the clean rolling-origin features (the encoder would have seen the labels of the queries it later scores) unless refit per origin (5 x cost); kept only as a gated roadmap step with per-origin refit and the LP-FT rung (tiny LR, 1-2 epochs, head warm-started from the probe).
- Per-candidate free embeddings without a text-derived prototype: overfits on a median of a few citers.
- Pseudo-relevance feedback, test-test kNN, label propagation, test-time adaptation, transductive embedding alignment: banned (use of the test queries collectively).
- Hubness correction (CSLS) computed over the test queries: banned. The train-query bank version is allowed and is already covered by the reference normalisation in (b).
- MMR or other diversification: lowers expected NDCG under linear gain.
- Random K-fold over training queries: optimistic (series and near-duplicates, no drift).
- Using raw `year` as a feature: test years are outside the training range.
- Graph neural networks over the co-citation graph, mixture-of-softmax topic models, a neural set ranker: too heavy for the primary gate; MoS or a pretrain-init of candidate vectors from item-item PPMI-SVD enter the roadmap only after the lean design is measured and gain more than noise.
- SPECTER-family (citation-trained) encoders as the default: potentially stronger for citation tasks but a compliance question (Q2); default to a general-purpose encoder and treat it as a variant after reviewer approval.

## Fixed work plan & runtime budget

All counts are constants fixed before submission (CLAUDE.md section 3): one A10G, `device="cuda"`, fixed seeds, fixed `num_workers`, fixed thread counts (assumed 8), `deterministic=True` and fixed `num_threads` and `force_row_wise` for LightGBM, no time-dependent branches. Estimates are for an A10G, are NOT measured, and must be profiled on one origin before the counts are frozen.

| Stage | Work | Est. time |
|---|---|---|
| 1. Load, assert schema, ids as `str`, build pool/year indices | parquet + csv, integrity checks | 1 min |
| 2. Encode texts (about 25.6k train + 10.1k test, two views each) | base-size encoder, fp16, max_length 256-384, one view of title, one of title + abstract, fixed batch size | 4-7 min |
| 3. TF-IDF + SVD (train fit only) | 25.6k docs | 2 min |
| 4. Origins 2017-2021 and final 2022: kNN vote, local evidence, prototypes, co-citation matrix | sparse matmuls on GPU/CPU, six origins | 6-8 min |
| 5. Two-tower head (3 seeds) at six origins, small grid for 3-4 configs on the selection year only | head on precomputed embeddings, full 29.7k-way softmax | 10-14 min |
| 6. Pool building and features | about 15-20k level-2 query-rows x about 300 candidates (about 5M rows) plus 10.1k x 300 for test | 5-6 min |
| 7. LambdaRank training: dev fit (fixed rounds) + final fit, bagged over 3 seeds | about 5M rows x about 40 features | 8-12 min |
| 8. Test scoring, decode, write, re-read validation | 3M rows | 3 min |

Total about 40-55 minutes if no stage runs over; the target is 50 minutes or less. If profiling shows more, cut in this order: reduce the number of origins (keep 2018-2021 plus final), reduce pool to 200, then reduce seeds. These are decided offline from the profile, then fixed (never at run time). Against the stated 90-minute limit the headroom is at least 40%. Memory: embeddings 36k x (2-3 views x 768) floats fits in GPU and CPU RAM; the candidate matrix is sparse (about 170k nonzeros); level-2 feature table about 5M x 40 float32 is under 1 GB; two-tower logits batch 512 x 29.7k are small.

**Hyperparameters.** Standard conservative fixed defaults for LightGBM (about 63 leaves, learning rate 0.05, feature fraction 0.7, bagging 0.8, `min_data_in_leaf` at least 200, L2) with the number of rounds chosen by an in-script dev run (fixed grid of 3 values on the selection year, held-out year for the report). The two-tower grid is at most 4 configs. The total in-script search is bounded and has fixed trial counts, no timeout.

**In-script validation.** Input and output validation: schema, id uniqueness, row count 10,106, 50 distinct ids per row in the pool, no NaN, re-read the written file with `keep_default_na=False` and `dtype=str`, raise before writing a broken file. Per-query independence check on a fixed sample of 200 dev queries: scoring alone versus in a batch yields the same top-50 (fixes padding/batch contamination risk; also covers the "each query on its own" ban). The script uses no fallback paths; a failure is loud.

## Metric-aware training & decode

1. *Back-solve the metric.* Gain is binary and the discount depends only on rank, so a reference's contribution is 1/(IDCG_q * log2(r+1)). The per-query normaliser is constant across the candidates of one query, so the expected-utility-optimal output is the list sorted by the marginal probability that each candidate is a reference (derived; no Monte-Carlo or joint decode is needed). Joint or diversified decoding is therefore a negative; if a gain ever appears on dev it should be treated as noise first.
2. *Weight the loss with the metric's own weights, hierarchically.* In the two-tower loss, average over a query's positives and then over queries (each query counts equally). In LambdaRank, NDCG@50 truncation with the in-pool/true IDCG weight ratio; check the library semantics for ranking weights.
3. *Loss choice.* Sampled softmax (full softmax over 29.7k candidates is cheap), multi-positive; logQ correction for popularity bias; aux co-citation loss. This matches ranking by marginal probability. The reranker's LambdaRank is trained with NDCG@50 truncation, matching the metric.
4. *No calibration needed across queries.* Only within-query order matters. Where scores from different generators are combined in the fallback, use within-query z-scores (a within-row transform, allowed) with weights chosen on the selection year and reported on the held-out year.
5. *Decode constants.* None except pool size and tie-break. Tie-break is deterministic (origin rate, then candidate id).
6. *Gold-score oracle check before modelling.* Feed gold scores (1 if reference) through the pool, the decode and the metric: the result must equal the pool-recall-limited ceiling, which is computed independently from the labels; it also exercises the formatting and the 50-id cap.
7. *Frozen probe as the honest yardstick.* Report the kernel vote and the prototype cosine under the same dev origins; the trained stack must beat them.

## Structural signals

- **Candidate identity from company.** Each candidate is represented by (i) the dense and TF-IDF embeddings of its citers (centroid plus top-m prototypes), (ii) its citer field histogram, (iii) its citer year profile, (iv) its co-citation neighbours. Evidence is local, so local pooling (max, top-m, log-sum-exp) over citers is preferred to a single centroid.
- **Reference-normalised evidence.** Raw similarity is dominated by generic-topic and by hub candidates. Score each candidate relative to a bank of unrelated training queries (built from train only, several fixed banks averaged), never relative to test.
- **Co-citation as free supervision.** References of the same query are mutually related: add the auxiliary candidate-candidate loss, and the companion-support feature (within-row propagation through train-fit PPMI).
- **Exact marginal structure.** Reference sets are unordered, so the loss is permutation-invariant (multi-positive, normalised per query).
- **Temporal causal alignment.** The test queries cite older works from a pool whose counts include later citations: replicate this at each origin (origin-restricted statistics, gap 0/1 rows, pool fixed) so the reranker learns the drift and the low-count shift from data; do not hand-code any constant of the pool generator, and do not estimate anything from test slates.
- **Pool-construction artefact.** Inclusion needs 5 citations in the full corpus, so a candidate with fewer than 5 train citations must have been cited in the test period. Feed raw generic count features and let the model learn the shift. Note the "exploits data generation" risk once for reviewers.
- **Field gate.** The query's `field` is an own-row input; use field-conditional candidate rates as features (smoothed), not as a hard filter (empty field and cross-field hubs exist).
- **Invariants.** A candidate is never a query. A query never cites itself. No test-time use of other queries. Query-level (not row-level) splitting everywhere.
- **Augmentation on the text side.** Embedding-level dropout and the title-only/abstract-only view mix act as augmentations of real labelled units. No synthetic data is created.

## Experiment roadmap

1. *Contract, metric, validation, unit tests.* Implement NDCG@50 and the rolling-origin generator; pass the unit tests; reproduce the random, most-cited and the two reference baselines on dev; run the data diagnostics (all of the list above). Stop criterion: the dev reference numbers are in the same range as 0.0007 / 0.0236 / 0.2022 / 0.2107; if not, recheck the origin protocol before building anything.
2. *Cheapest valid end-to-end baseline.* Reference TF-IDF centroid plus popularity, writing a valid file in the exact format. Records the floor and checks formatting. This is the first credit.
3. *Representation and structural insight.* Dense versus TF-IDF kNN vote; title versus abstract views; `max_length`; K and tau; local evidence versus centroid. Stop when a further view or encoder does not beat noise (at most 2 encoders).
4. *Trained head and metric-aware loss.* Two-tower with the prototype-plus-residual parametrisation, aux loss, regularisation sweep; compare against the kernel vote under the same origins. Stop criterion: gain over the kernel vote above 2 paired standard errors, else keep the vote as the primary generator and use the head as a reranker feature.
5. *Reranker.* LambdaRank on the pool; add feature families one at a time (generator scores, candidate statistics, local evidence, companion support, drift features, the pool-artefact counts, gap); accept each only if the paired gain is above 2 standard errors and consistent across gap 0 and gap 1.
6. *Diversity.* A second encoder or the TF-IDF view as an extra generator; mixture-of-softmax or PPMI-SVD initialisation for the candidate vectors; LP-FT of the top layers per origin (gated by the cost in the budget). Score each member alone first; keep only those close in quality and that add a measured gain.
7. *Bounded in-script search.* Fixed small grids as in the work plan, with the selection year and the held-out year separated; log the number of knobs tried.
8. *Final run and stability.* Clean `working/`, the exact command, run twice and diff (the rankings should be identical or nearly so), verify the output distribution (popularity of the ranked candidates versus dev OOF, share of the top-50 that are cold), re-run the validator, then submit. Use credits only for: baseline, best single (kernel vote or head), full stack, and final.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits applied to this plan (all answers are intended answers for the plan, to be re-verified on the code):

- Does any code path read test data for something other than one-query-at-a-time inference? Intended no. Test queries are encoded and scored alone; no vocabulary, scaler, SVD, kNN, normalisation or dedup uses them. Verified by the per-query independence check and by code review.
- Time inside any conditional or argument? Intended no (time is used only in logging).
- `torch.cuda.is_available()`, `os.cpu_count()` or import fallbacks changing the work? Intended no. The encoder choice and thread counts are constants fixed before submission. This forces a pre-submission decision on Q1.
- Hard-coded constants tuned offline? Intended no. Every constant is found by an in-script train-only search with a separate held-out year, or is a standard default stated in the plan.
- External data, synthetic data, self-hosted weights, non-allowed libraries? Intended no. Only the challenge data and an offline pretrained general-purpose encoder; no online lookup; nothing derived from any copy of the literature.
- Would the solution still work with the ML removed? The non-parametric scorers alone approximate the 0.20 reference class, which is grey. Mitigation: the ablation (target gain of at least 0.015 over the best non-parametric member, more than 3 paired standard errors) and the trained two-tower plus reranker as the load-bearing final scorer. If the ablation fails, revisit the compliance posture.
- Raw-pixel tabular or inference-only? Not applicable; the head and the reranker are trained in-script.
- Source readable and below 512 KB, with explanatory comments? Planned.
- Did the plan honour every challenge-specific restriction? Yes: the collective-use ban, the no-network and offline-encoder condition, the 90-minute limit, the per-query independence, the no-external-lookup ban.
- Whole-test aggregation checklist: no test normalisation, no test-fit vocabularies or encoders, no hubness correction over test, no test pseudo-labels.
- Sibling leakage: label-derived statistics are temporal (origin-restricted) and exclude the query's own cluster; the companion-support feature uses only train co-citation and the row's own first-stage list.
- Label-derived statistics: candidate rates and prototypes are built from training queries only, out of time for every level-2 row; features are rates, so the same data-size scaling applies at train and test; a nested check (statistics built only from queries before the origin) is exactly the rolling-origin design.
- Known grey points for reviewers: non-parametric TF-IDF/kNN scorers as inputs; the pool-artefact count feature; frozen encoder plus trained heads rather than encoder fine-tuning; use of the query's field (and optionally gap).

## Open questions & assumptions

**Questions for a reviewer (each with a plan under each reading).**
- Q1. Which pretrained encoders are available offline in the grading environment, and is a Hugging Face download at run time (CLAUDE.md section 1) allowed despite "No network access"? Reading A (a named encoder is installed): primary as planned with that encoder. Reading B (none): use the trained-on-train text tower fallback (a single constant, not a runtime switch) and expect a lower score (estimate 0.02 to 0.04 NDCG lower). Reading C (HF download is allowed): the same as A with the encoder pinned to a fixed revision. The plan is safe under A and B; the cost of B is an estimate, not measured.
- Q2. Are citation-trained encoders (SPECTER-family) acceptable, given they may embed citation relationships of works near the test period? Default: general-purpose encoder only. If approved, test as a variant on dev (expected gain uncertain, perhaps 0.01 to 0.03).
- Q3. Is frozen encoder + trained head (candidate vectors + query MLP) + LambdaRank an acceptable "genuine training" design for this challenge (it is not labelled fine-tuning)? Plan under the strict reading: add the LP-FT rung (last layers, tiny LR, 1-2 epochs, refit per origin) with the parameter-change norm and the CV gain over the probe logged, if the budget allows.
- Q4. May the query's `year` appear only as a relative gap feature (query year minus statistics cutoff), or not at all? Default: measure with and without; drop it if the gain is below noise.
- Q5. Is the pool-inclusion count artefact (fewer than 5 train citations implies later citations) an acceptable signal? It uses only shipped train-derived numbers and is learned, not hard-coded, but could be read as exploiting data generation. Plan under the strict reading: drop the explicit max(0, 5 - count) feature (the model still sees generic count features) and report its measured cost.
- Q6. Is batching test queries for GPU encoding acceptable when per-query outputs are provably independent (padding masks, eval mode)? Assumed yes, with the per-query independence check in the script.

**Assumptions where the description is silent.**
- One A10G, 24 GB; at least 8 CPU cores and 32 GB RAM; a CUDA-capable LightGBM is not required (CPU LightGBM is used with a fixed thread count).
- The planning target is 50 minutes or less (CLAUDE.md), well below the stated 90 minutes.
- `train_citations` equals the count recomputed from `train_references.csv`; to be asserted.
- Training queries have at least one pool reference; queries with none (if any) are dropped from training and counted in the log.
- Per-year volumes are large enough in 2018-2022 to support rolling origins; if some years are tiny, merge adjacent years (decided from the diagnostics, then fixed).

**What could not be verified in this run.** Everything under Data findings (no data), encoder availability, package versions (LightGBM ranking-weight semantics), runtime estimates, the expected score band, and the strip-the-ML ablation margin. All of these are estimates or checks to run in step 1 and 2 of the roadmap.
