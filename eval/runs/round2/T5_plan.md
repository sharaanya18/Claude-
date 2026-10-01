# Eris plan: T5 Archival letter reference retrieval

Status of evidence: NO dataset was available to the strategist. Everything under "Data findings" is a diagnostic to run on train plus an UNVERIFIED hypothesis. Numbers marked "(guess)" are priors, not measurements. Nothing here was run.
Inputs read: the agent file, CLAUDE.md, and the T5 task description (paraphrased; exact column formats, sizes, sample_submission shape are therefore unknown and listed as open questions).

## Contract & decision unit

One valid answer, per query row (one mention inside one letter): an ordered list of exactly 10 DISTINCT letter ids, all present in `letters.csv`, none equal to the query's own `document_id` (the task says the referenced letter is a DIFFERENT letter). Shorter, duplicated, unknown-id or self-referencing lists are invalid or wasted slots (assume invalid until the submission schema is known).

Metric: MRR@10 = mean over queries of 1/r if the gold letter is at rank r <= 10, else 0 (rank 1 -> 1.0, rank 2 -> 0.5, rank 10 -> 0.1, rank 11+ -> 0). Per-query, unweighted (confirm there is no per-document or per-correspondence averaging; if the description turns out to average per document, replicate that hierarchy in the loss weights and in CV). The metric is rank-only, but the top few ranks dominate: a gold at rank 1 vs rank 3 is worth 0.67 of one query. Calibration across queries is irrelevant; calibration of the ordering within a query is everything. There is no gating that zeroes the score for a wrong subset; invalid ids/format would zero it through the validator.

Independent units (for effective sample size): not the query row. Several mentions come from the same source letter; several source letters can point to the same gold letter; letters of one correspondence form threads. Effective n is the number of connected components / correspondence clusters, probably far below the row count (unknown; diagnostic D5).

Pipeline stages (diagnosed separately):
1. Candidate coverage: from the archive (thousands of letters) build a per-query pool of fixed size N (default 200, set from diagnostic D3). Oracle recall@N of the gold in the pool is the hard ceiling on MRR.
2. Scoring: learned evidence about "is this archive letter the one the mention refers to": date evidence (the mention text vs the candidate's date, parsed and learned), correspondent/direction evidence (who wrote to whom), thread-position evidence (recency among the correspondent pair's letters), name evidence from the mention window, light lexical/topic overlap.
3. Decoding: sort by score, drop the self letter, take 10 distinct ids, deterministic tie-break (stable sort, then letter id). Because MRR is optimised by ordering on the posterior probability of the gold, no extra decode logic is needed. Pad from the stage-1 order if the pool has fewer than 10 after exclusions.

Reduction check: the generative-looking task ("find the letter") reduces to ranking a small enumerable legal set (pool of N archive ids). The reduction loses a valid answer only when the gold is outside the pool; that loss is measured by D3 and bounded by choosing N from the recall curve.

## Compliance regime

Domain: retrieval/ranking with structured date and name extraction over OCR text (CLAUDE.md 6.4 RAG/ranking, 6.2 NLP, 2.5 grey areas). Challenge text overrides generic rules: CPU-only, standard libraries, about 1.5 h, no pretrained weights needed, no external data, model trained in-script from the four CSVs. Note: CLAUDE.md prefers a target of <= 50 min and <= 1 h worst case; this plan targets <= 45 min on CPU, hard ceiling 60 min, well inside the stated 1.5 h.

Allowed here: numpy, pandas, scikit-learn (HashingVectorizer, LogisticRegression, HistGradientBoosting), scipy, standard library (datetime, re, difflib, unicodedata). LightGBM is in the Kaggle image and allowed by CLAUDE.md section 8 but "standard libraries" in the description is ambiguous; see Open questions Q1. No pretrained weights, no internet, no external lexicons downloaded. No `try/except import` fallbacks, no `os.cpu_count`, no clock-dependent branching (CLAUDE.md section 3): fixed `num_threads`, fixed seeds, `deterministic=True`.

Explicit risks from CLAUDE.md that apply:
- 6.4: "hand-engineered features + off-the-shelf LambdaMART alone is not enough". Mitigation: the load-bearing trained components are (a) supervised text heads trained on the mention windows (hashed char n-grams -> target attributes of the gold letter) and (b) a parametric listwise softmax model with learned potentials; the GBDT ranker is a stack on top, not the only learner. Report the ablation "regex-only features -> ranker" vs "with learned heads".
- 2.5 regex: regex is used only for deterministic number/token extraction and cleaning; the interpretation of an extracted token (which month a "7bre" is, whether "huj." means this month, whether 27 is a day of this or last month) is LEARNED from train gold pairs wherever data allows. A small seed lexicon of month/unit tokens used as FEATURES (not decisions) is grey; keep it small, comment it as a feature dictionary, and verify it adds a measured gain over the learned token->month table alone (if not, drop it).
- 2.3 #5 whole-test aggregation: no statistic is fit on `test.csv`. Per-query features are a function of that query row, the archive `letters.csv` (the retrieval index) and train-fit models. Corpus-level fits over `letters.csv` (IDF, vocabularies) are AVOIDED by design: use stateless `HashingVectorizer` (no vocabulary fit) and no IDF; the ranker learns the weights. Whether the archive may be used as a retrieval index including the text of test-source letters is flagged as Q2 (the archive is a task input, not the test file, but it contains the test query letters' text).
- Strip-the-ML test: remove the trained heads, listwise model and ranker and what remains is a date parser plus recency heuristic. Honest expectation: that would still score non-trivially (guess MRR 0.2-0.4), which is exactly why the interpretation steps are made learned and an ablation is logged. The ML is load-bearing in three places: reading of the mention (heads), combination and disambiguation among same-day/same-pair candidates (ranker), and weighting of evidence types.
- Hard-coded constants: pool size N, window width, hash dimension, regularisation strengths and tree sizes are chosen by in-script grouped CV over a fixed grid (data-determined, not clock-determined), never pasted from earlier submissions.

## Data findings

All items below are TO RUN on train (and `validation.csv`, which is labelled and therefore legitimately studied; and `letters.csv` for schema and archive structure). Hypotheses are UNVERIFIED. From `test.csv` use only schema, row count, id format and text-length statistics for planning.

D1 Schema and integrity. Shapes and dtypes of the four CSVs; null rates per column; `date` format(s) in `letters.csv` and in queries (ISO, partial year-only/year-month, free text, missing); do `mention_start`/`mention_end` index into `text` correctly (check text[start:end] length distribution, off-by-one, spans out of range, multiple mentions per document, overlapping spans); is `document_id` always in `letters.csv`; is the gold id always in `letters.csv`; is the gold ever equal to the document id; duplicate letter ids; duplicate (sender, recipient, date) tuples in the archive. Hypothesis: spans are valid for a large majority; a few percent of letters have missing/partial dates; gold is always in the archive.

D2 Mention anatomy. Distribution of mention length and a 100-sample reading of what the span covers (just "5th"? a whole phrase? names?). Share of mentions containing: a number 1-31; a month token (by language); a roman numeral; a "Xbre" abbreviation; a relative idiom (huj./ult./vorigen Monats/vor N Tagen); a weekday; a 2-digit year; a sender/recipient name. Language mix. Hypothesis (guess): 60-80% contain an explicit day number, 30-50% a month token, 10-25% relative idioms, small share weekdays; strong imbalance across languages (German probably dominant).

D3 Candidate coverage curve. For each train query compute the gold's rank under cheap stage-1 criteria (time gap and correspondent pair only) and the recall of the gold at pool sizes N in {25, 50, 100, 200, 400, 800}. Break down by the reason a gold is missed (different correspondent, later than the source letter, date missing, beyond the gap window). Hypothesis: gold is mostly an EARLIER letter of the same correspondents (A<->B), within a few weeks; recall@200 above 95% (guess); the missed tail is third-party letters and letters referenced FORWARD in time.

D4 Gap and direction structure. Distribution of (source date - gold date) in days, signed, with a quantile table and a log-bucket histogram; share of golds dated after the source letter; share with missing dates. Direction class of (gold sender, gold recipient) vs (source sender, source recipient): reverse (gold sent by the source's recipient to the source's sender, i.e. "your letter"), same direction ("my letter"), third party, unknown. Hypothesis: reverse direction is the plurality (guess 50-70%), same direction 15-30%, long right tail of the gap (median days to a few weeks, tail months). The direction can be predicted from the mention words ("your/Ihr/votre/vostra" vs "my/mein"); verify by a quick LR on mention window words.

D5 Group and leakage structure. Union-find components over {document_id, gold_id} (and separately over the unordered correspondent pair) on train; component size histogram; size of the largest component as a share of rows; number of distinct correspondent pairs; overlap of document_ids, gold ids, correspondent pairs and years between `train.csv` and `validation.csv` (this is the only available evidence for how `test.csv` was split, since validation was presumably cut the same way). Hypothesis: split is by random queries or random source documents; if train/validation share many gold ids or correspondent pairs, the test split is probably also leaky in that axis and a random-grouped CV is representative; if disjoint on documents only, group by document. A giant component (>30% of rows) would force coarser grouping (correspondent pair or time blocks).

D6 Ambiguity ceiling. For each train query count how many archive letters share the gold's (sender, recipient, date) tuple (same-day same-pair letters) and how many match the extracted day within the pair. Compute an oracle "naive best-possible" rule: among pool letters matching correspondent pair direction and parsed day, MRR if ordered by recency or randomly. Hypothesis: most queries have a unique match once direction + day (+ month) are known; a few percent have irreducible ties (duplicates, drafts, copies), capping MRR below 1.0 (guess ceiling 0.85-0.95).

D7 Date-reading ambiguity table (learned mapping evidence). Cross-tabulate extracted mention tokens (month names, "7bre"-style, roman numerals) against the GOLD letter's month, and mention day number against the gold's day; show how often (gold month - source month) is 0, -1, -2, and how often the day matches exactly; how often the match is off by 10, 11 or 12 days (Julian/Gregorian offsets between English and Continental correspondents) or year off by one (March-25 new-year). Hypothesis: month offset 0 or -1 dominates, exact day match in the large majority of cases with an explicit day; a few percent (concentrated in English/Russian-correspondent letters) show 10-12 day shifts; the year-start effect appears only in Jan-Mar letters of the 17th-18th century.

D8 OCR/name variants. Cardinality of `sender`/`recipient` strings; normalise (casefold, strip titles and particles von/van/de/zu/di, remove accents) and count cluster collapse; check whether sender/recipient look like canonical ids or noisy free text; how often a name appears in the mention window and matches the gold's sender/recipient. Hypothesis: names are noisy (initials, spelling variants), so fuzzy matching features help but exact id equality is a weak feature; at most 10-30% of mentions include a name.

D9 Text size/memory. Letter length distribution (chars, tokens), archive size, expected pool-feature rows = Q x N x F, to set the fixed budget. Hypothesis: OCR'd letters of a few hundred to a few thousand characters; archive comfortably fits in RAM; Q x N = a few million rows at most.

D10 Label-noise and conflicting keys. Queries with identical (document, mention) but different gold (should be none), gold equal to a letter whose date is after the source, golds with missing dates, language-specific failures. These define what the model cannot be expected to solve (loss analysis later).

## Validation design

Reproduce the test split from structure only: use the train-vs-validation relationship (labelled, so legitimately studied) from D5 to infer whether the split is random by query, by source document, by correspondence, or by time. `validation.csv` is the closest available image of the test split: treat it as an outer confirmatory holdout that never touches hyperparameter choice, grid selection or head fitting until the final model comparison. Final shipped models are refit on train + validation (100% of labelled data) with the CV-derived fixed counts.

Inner CV on train: GroupKFold, 5 folds, repeated with 3 split seeds (CLAUDE.md 4A); report mean and std of MRR@10. Groups = union-find components over {document_id, gold_id} (so a source letter's mentions and any letters sharing a gold stay together). If D5 shows a giant component, group by normalised unordered correspondent pair; if the pair groups are too few (< ~50), use blocked time slices (by year) as the group. If train/validation are disjoint on source documents but share correspondents, group by document only. Pick the grouping that mirrors the observed train/validation relation; and check that the CV score on grouped folds is within noise of the train-to-validation score (a gap far larger than the fold std means the grouping or the split mismatch needs revisiting).

Nested rules:
- The supervised text heads (day, month offset, gap bucket, direction, thread-rank) are fit on fold-train and produce OOF outputs for fold-validation rows; the ranker's TRAINING rows must receive OOF head outputs obtained by an inner cross-fit (heads trained on other folds), and test rows receive heads refit on all train rows. The ranker is never trained on in-sample head outputs.
- Regularisation per head (C of the logistic regression, hash dimension, window width) chosen by the exact metric proxy (the head's log loss and the downstream MRR) under grouped CV over a very wide grid including the strong-regularisation end.
- Pool size N, GBDT size (leaves, rounds) and learning rate chosen from a small fixed grid with their own held-out check (cross-fitted on the outer folds, then confirmed on validation.csv).

Metric implementation unit tests (exactly the formula above): perfect list (gold at rank 1) -> 1.0; gold at rank 10 -> 0.1; gold at rank 11 or absent -> 0; reversed order of a perfect ranking; constant scores (tie-break by id must be deterministic and not leak the gold); base-rate prediction (uniform random order over a pool of size N -> expected MRR@10 equals the closed-form (1/N)*sum_{r=1..10}(1/r)); a hand-built 3-query example (gold at ranks 1, 2 and 12) whose MRR@10 is (1 + 0.5 + 0)/3 = 0.5, checked by hand.

Additionally evaluate pool-restricted vs full-archive MRR separately (stage diagnosis: oracle recall, MRR conditional on gold in pool, MRR overall). A self-built proxy usually over-estimates the real score; use it for relative comparisons and expect the private number to land below the CV figure.

## Overfit/underfit risks

Overfit risks and mitigations:
- Few independent groups (components/correspondent pairs may number in the hundreds): constrain model capacity (capacity ladder below), strong-end regularisation, group CV with repeats, report std.
- High-cardinality identifiers: do NOT use sender/recipient strings, letter ids, or archive-level letter "popularity" (how often a letter appears as a gold in train) as features. Pair-level target statistics are not used. Only relational features (equality, fuzzy similarity, direction, rank within thread) which generalise across correspondents.
- Label leakage: golds of other train queries are used only through fold-excluded training; the heads' outputs fed to the ranker are cross-fitted; no feature uses the gold's mention back-links.
- Selection on the reported OOF: grids are small and fixed; the final check is validation.csv.
- Year/language skew: heads can latch onto era-specific vocabulary; check per-language/per-century MRR on the held-out groups, and keep the char n-gram window narrow and the hash dimension moderate.
- Calendar rules: do not hard-code Julian/Gregorian corrections; give signed day deltas and let trees learn the 10/11/12 patterns, check they hold across folds.

Underfit risks and mitigations:
- Truncated context: use mention +-K characters (K chosen from grid {60, 120, 240}) AND a separate whole-sentence bag; do not truncate the source letter's header (date/names).
- Information loss in preprocessing: keep raw digits, ordinal suffixes ("27ten", "5th", "5.") as tokens; do not lowercase away roman-numeral vs word distinctions in the char n-gram head; keep OCR variants (n-grams are robust).
- Too-narrow pool: N selected from the recall curve (D3); a separate third-party-letter generator branch (top matches by date alone) so cross-correspondent references remain reachable.
- Loss/metric mismatch: lambdarank truncation 10 (or softmax listwise log-likelihood of the gold), evaluated with the exact MRR@10.
- Missing latent structure: date reading is explicitly marginalised (see approach), not left to a flat classifier.

Capacity ladder (CLAUDE F2.1; stop at the lowest rung that wins under grouped CV): rung 1 factorised listwise softmax with a handful of learned potentials (about 30-60 parameters beyond the heads); rung 2 GBDT ranker with shallow trees over the same features plus the heads; rung 3 (only if rung 2 beats rung 1 by more than the repeated-CV noise and a diverse member is still worth the cost) a rank blend of both. No neural fine-tune is planned (CPU only, no pretrained weights).

## Recommended approach (primary + fallback)

### Primary: two-stage, latent-reading-aware ranker (stacked)

Stage 0, archive preparation (per run, from raw CSVs). Parse every archive letter's date into (year, month, day, precision flags) and weekday; normalise sender/recipient (casefold, accents, titles, particles). Build per-correspondent-pair sorted letter lists (unordered pair, with direction) so thread-position features are O(1) lookups. All derived from `letters.csv` only; no label use and no corpus-fit vocabulary.

Stage 1, candidate pool (recall-oriented, N fixed after D3 by an in-script grid). Union of: (a) letters of the same correspondent pair (either direction) dated before the source letter plus a small forward slack, ordered by time proximity, (b) letters in the signed time-gap window learned from train (quantiles of the gap distribution) from ANY correspondents, (c) letters matching the extracted day/month regardless of pair. Rank this union by a cheap logistic regression over a handful of features (log gap, pair match, day match) trained on train and keep the top N. Self letter always excluded. Report recall@N.

Stage 2a, supervised "attribute heads" trained on train mentions only (the genuinely trained text models, hashed stateless features: char n-grams 2-5 of the mention window, word 1-2-grams of the wider context, plus the raw extracted numeric tokens). Each head predicts an attribute of the GOLD letter relative to the source letter, from the mention text:
1. day-of-month of the gold (1-31 + "none", with the extracted integers as strong inputs);
2. month offset (gold month - source month, classes -12..+2, with "unknown");
3. year offset (0, -1, -2+, +1) to cover 2-digit years and March-25 years;
4. gap bucket (log-spaced day buckets, signed);
5. direction class (reverse, same, third party);
6. thread-rank class (the gold is the 1st, 2nd, 3rd+ most recent letter of the respective direction before the source letter);
7. weekday (7 + none) when a weekday cue exists.
Model: multinomial logistic regression on hashed sparse features with per-head strength chosen by grouped CV. These are the learned replacement for hand-written interpretation rules; they are cheap and CPU friendly, and they implement the latent date-reading factorisation (the reading of the mention, e.g. "27ten" = day 27 of this month or of last month, is a distribution over month offsets conditional on the text, trained from the gold dates, so the weak/ambiguous month is marginalised rather than fixed). For a candidate c, the log-posterior features are log P_head(attribute(c) | mention): log P(day = c.day), log P(month offset = c.month - source.month), log P(year offset), log P(gap bucket of c), log P(direction of c), log P(thread-rank of c), log P(weekday of c). Heads are cross-fitted (see Validation).

Stage 2b, deterministic extraction features (feed the ranker, grey regex scope): extracted integers 1-31 in the mention window and min |c.day - n|, exact-match flag, matches of c.day under +10/+11/+12 offsets; whether c.month equals the month that a seed month-token maps to (token->month also learned from train as a table, with the seed lexicon only as an additional feature); implied gap days when a number+unit idiom appears ("vor 8 Tagen") and |c.gap - implied|; candidate weekday vs mention weekday under both Gregorian and Julian reading of c.date; 2-digit year vs c.year mod 100.

Stage 2c, relational and structural features: direction of the candidate vs source (reverse/same/other) with fuzzy name similarity (difflib/char-trigram Jaccard) on normalised names; thread-recency rank among same-direction letters before the source date (rank 1 = most recent) and counts of competing same-day letters; gap in days and log gap; candidate date precision and missing flags; mention-window name overlap with candidate sender/recipient (fuzzy) and with the source's recipient; stateless hashed word-unigram cosine between the source letter's text and the candidate's text and between the mention window and the candidate's opening lines (sublinear tf, no IDF) as supporting evidence only; within-pool relative transforms (score minus pool max, rank within pool, margin to the runner-up) for choose-among-candidates behaviour.

Stage 3, ranker. LightGBM `lambdarank` (truncation level 10, groups = queries), shallow trees (num_leaves in {7, 15}), min_data_in_leaf large, lr 0.05, a fixed grid of at most 4 configs, rounds set by early stopping on fold validation MRR@10 inside CV, then refit on 100% of the labelled data with rounds = mean best iteration x 1.1 (fixed count). Train on the top-100 stage-1 candidates (hard negatives) plus a fixed random sample of 20 more; evaluate on the full pool. Seeds fixed, `deterministic=True`, `force_row_wise`, fixed `num_threads`.

Stage 4, decode: score all N pool candidates, sort descending with stable tie-break by letter id, drop the self letter, output the first 10, pad deterministically if needed, write the file, re-read it and validate against the schema.

How this meets "genuine training": (i) seven supervised heads on mention text, (ii) the stage-1 logistic regression, (iii) the ranker, all trained in-script; the ablation `regex-only -> ranker` vs `heads + ranker` is logged to show the learned components carry gain.

### Fallback (rung 1 of the ladder): factorised listwise softmax

Same stage 0-2 features, but replace Stage 3 with a parametric model: score(q, c) = sum_k w_k * f_k(q, c) with about 30-60 learned potentials (the head log-posteriors, the extraction-match indicators, the thread-rank and direction features), P(c | q) = softmax over the pool, trained by exact listwise log-likelihood of the gold (scipy L-BFGS, L2 chosen by grouped CV). It is the leanest design, the most defensible under the "model must learn" rule, and has no GBDT dependency (answers the "standard libraries" ambiguity: only numpy/scipy/sklearn). Ship it if the GBDT stack does not beat it by more than the repeated-CV noise, or if the reviewer rules LightGBM out (then substitute sklearn `HistGradientBoostingClassifier` pointwise over relative features as the rung-2 member, measured on the same folds).

Why this ranking (expected PRIVATE score order): primary exploits non-linear interactions (e.g. a day match only matters when direction and thread-rank agree; calendar offsets depend on era and language); fallback is more robust to few independent groups. If the two are close, a z-scored within-query blend of both is the diverse-assumption ensemble (different inductive bias: parametric potentials vs tree interactions).

Expected score (estimate, unverified, not a promise): MRR@10 about 0.45-0.70 on private, point guess about 0.55, with a ceiling set by D3 recall and D6 ambiguity. Reasoning: explicit day + direction + recency usually identifies the letter, but OCR, relative idioms, missing dates and same-day duplicates cap performance; the proxy likely over-estimates by several points.

## Rejected options

- Pure rule-based date matcher plus recency (strip-test fail; rejected as the solution, kept only as feature extraction).
- BM25/TF-IDF retrieval of the candidate by mention text as the core (the mention rarely contains topical text and the description lists it as grey; also IDF is a corpus fit, so only stateless hashed lexical overlap is kept as a minor feature).
- Neural cross-encoder or bi-encoder (CPU-only, 1.5 h, no pretrained weights; from-scratch transformers on thousands of letters would memorise groups; the capacity ladder stops at rungs 1-2).
- Test-set tricks: pseudo-labels, test-fit vocabularies, reciprocal-reference consistency among test queries (e.g. enforcing one gold per test letter), normalising scores across test queries (all banned by section 2.3 #5 / whole-test aggregation).
- Letter-id or sender-id target encodings (high-cardinality, group-specific artefact).
- Hand-written Easter/moveable-feast, saint-day calendars as rules (grey/brittle; reconsider only as a candidate-date feature if D2 shows a material share of "on St. Michael's day" mentions and the learned token->date table cannot capture them).
- Large GBDT with many leaves or many HPO trials (selects noise on few groups).

## Fixed work plan & runtime budget

CPU only; fixed `num_threads` (suggest 4; pick a constant, never derived from the machine); seeds fixed; no clock branches; the work count is data-determined only through fixed grids. Estimates are for an assumed (guess) Q = 5-20k labelled queries, archive 5-20k letters, N = 200, F ~ 80 features; recompute after D9.

| Stage | Fixed plan | Est. minutes (guess) | Memory |
|---|---|---|---|
| CSV load, date/name parse, pair lists | once | 1-2 | < 1 GB |
| Mention-window extraction + hashing (HashingVectorizer 2^20, stateless) | once | 1-2 | 1-2 GB sparse |
| Pool generation + stage-1 LR (grid over N in {50,100,200,400}, 1 split) | 5 folds only for the chosen N | 3-5 | 1-2 GB |
| Heads (7) cross-fit: outer 5 folds x inner 4 for the training-side OOF, wide C grid on one repeat then fixed | 5x4 fits x 7 heads | 6-10 | 2 GB |
| Pool feature matrix Q x N x F (float32, rows capped by hard negatives for training) | once per outer fold | 4-6 | up to 4-6 GB |
| Ranker CV: 5 folds x 3 split seeds x <= 4 configs, early stopping | ~60 fits (reduce configs to 2 after repeat 1) | 12-18 | 4 GB |
| Fallback softmax CV | 5 folds x 3 seeds | 2-4 | 2 GB |
| Final refit on train + validation, predict test, decode, validate | once | 4-6 | 4 GB |
| Total | | about 35-50 | peak <= 8 GB |

Target <= 45 min; the stated limit is 1.5 h, and CLAUDE.md asks for >= 30% headroom against the shorter planning value. If profiling shows a stage too slow, reduce fixed counts (N, number of grid configs, number of split seeds) permanently in the script, never conditionally. Local profile step (CLAUDE.md 3.7): time one head fit and one ranker fit on a subsample, extrapolate, then hard-code the counts. Run the full script twice and diff the two submissions (must be equal or near-identical); LightGBM determinism requires `deterministic=True` and fixed threads, and sorted iteration over dicts/sets in feature code.

Input/output validation inside the script: assert the four CSVs and required columns exist; assert spans valid or clip with a logged count; assert test ids unique and all letters resolvable; after decode assert 10 distinct archive ids per query, none equal to the document id, finite scores, row count equals the test row count; re-read the written file with `keep_default_na=False` and check the schema. Because the task description names four CSVs and no sample_submission, the validator uses the schema in Open question Q3.

## Metric-aware training & decode

- Back-solve: MRR@10 only needs the gold's rank; the target is binary per candidate with exactly one positive per query (confirm single gold). If a mention can legitimately have multiple valid letters (copies/duplicates), D6 will show it and the label becomes "any of the valid ids" with a listwise likelihood over the positive set.
- Loss: lambdarank with truncation 10 for the GBDT; listwise softmax negative log-likelihood of the gold for the fallback (its gradient equals the expected reciprocal-rank surrogate direction and is a proper scoring rule over the pool). Optionally compare `rank_xendcg`. Weight queries uniformly (matches the metric); if the metric turns out to average per document, weight by 1/(mentions per document).
- Expected utility: because the metric is a decreasing function of rank with one relevant item, ordering by posterior probability maximises expected MRR; no calibration across queries is needed. Do not fit per-query thresholds.
- Hierarchical averaging: none unless the description says so (Q4).
- Hard constraints at decode: exclude self; distinct ids; ids in the archive; pad deterministically from stage-1 order. Candidates whose date is clearly after the source letter are not removed by hard rule (forward references exist, D4); the learned gap head down-weights them.
- Per-slot calibration of head outputs: heads are trained with log loss; their log-probabilities are used as features and are not post-calibrated on test. Cross-fitted OOF head outputs feed the ranker; report the head log loss vs prior baseline per head.
- Decode constants: none beyond N (fixed by the in-script grid) and K (window).

## Structural signals

Invariants and free signals (each used as feature, auxiliary head, or constraint):
1. Reply structure: a letter that says "your letter" is usually addressed to the source's recipient reversed (direction head, pair match); used as feature and as the direction-class head target.
2. Thread-position: the referenced letter is usually the most recent letter in that direction before the source (recency rank within the correspondent pair); thread-rank head and rank feature.
3. Temporal causality: a reference in a letter dated d points to a letter dated at or before d except for forward references; the signed gap head learns the tail; month/year offsets handle "huj." (this month), "ult." (last month), "vor N Tagen" through learned conditional distributions.
4. Calendar structure: weekday is a deterministic function of the date; candidate weekday matches the mention weekday under both Julian and Gregorian readings; a 10/11/12 day shift and March-25 year start as signed deltas, learned by the ranker. All checked on every train case to see whether they hold (D7).
5. Symmetries: swapping which language the month names are in does not change the relationship; all heads receive language-agnostic char n-grams. No image geometry, so no augmentation of that sort. Recombining real labelled units is permitted: a mention can be trained against a pool built from any permutation of other letters; no synthetic letters are fabricated.
6. Coupling between outputs: the day-of-month and month offset together imply the date; the direction head and the pair match are coupled; test both directions: predict the date attribute from the mention and test whether conditioning the direction head on the date evidence helps (cheap train-only check, decide by CV).
7. Hidden taxonomy: month names and relative idiom classes can be discovered from TRAIN gold pairs (token -> month offset clusters); enforce no hidden taxonomy at decode, only through features; verify on every train case that tokens mapped by the learned table agree with the gold month at a high rate; report the residual disagreement as the noise floor.
8. Duplicates: identical or near-identical letters (copies, drafts) in the archive: treat as a group when building CV groups; at decode they are an irreducible tie; check D6 and keep the deterministic order.

## Experiment roadmap

1. Contract, metric and validation (stop when unit tests pass and the grouping choice is justified by D5): load data, run D1/D5, implement MRR@10 and the validator, build the train/validation grouped CV; verify the pool recall curve D3 and choose N.
2. Strongest cheap baseline, end-to-end and valid: stage-1 logistic regression over {pair match, direction, log gap, recency rank, day match} -> top-10. Record CV mean +- std and train-to-validation score; write a valid submission format-wise. Stop: valid file, metric reproducible.
3. Representation and structural insight: add heads (day, month offset, gap, direction, thread-rank) and the factorised listwise softmax (rung 1). Measure each head's marginal gain paired on the same folds with 3 split seeds; keep a head only if the gain exceeds noise. Report per-language and per-mention-type MRR.
4. Metric-aware ranker (rung 2): LightGBM lambdarank over the same features; compare to rung 1, paired, repeated seeds; ship the lower rung unless rung 2 wins by more than one std-error and more than half of the folds.
5. Diversity: z-scored within-query blend of rungs 1 and 2; keep only if it beats the best single beyond noise. Candidate additions in order of expected gain per hour: calendar offset features, weekday matching, name-in-mention features, thread competition features, hashed lexical overlap.
6. In-script bounded HPO: fixed grid (<= 4 ranker configs, per-head C grid, N, K), cross-fitted with its own held-out check; confirm on validation.csv once.
7. Final fixed-plan run: refit on train + validation with fixed counts, run twice, diff; report CV per fold, runtime, compliance audit result and validator output.

Credit use: baseline, best single, ensemble, final (CLAUDE.md 4A). Before any proxy-driven tweak, test the techniques most solvers would converge on first: pair+direction+recency, learned gap prior, and exact day match.

## Compliance audit

CLAUDE.md section 7 and agent self-audits, applied to the plan:
- Test file read only for one-sample prediction: queries are transformed row by row; the archive is the index; no statistic from test rows (vocabularies are stateless hashing; no IDF; no score normalisation across test queries). Flagged: archive text includes the test source letters (Q2). PASS with the caveat.
- No time conditionals, no cpu_count/cuda switches, no import fallbacks: fixed threads, fixed counts. PASS by design.
- Hard-coded tuned constants: N, K, C, tree sizes selected by an in-script grid cross-fitted over fixed lists; the seed month/unit lexicon is a feature dictionary, not tuned. PASS if the lexicon ablation is logged.
- External data / synthetic / hosted weights: none. PASS.
- Strip-the-ML test: partial risk acknowledged (rule features alone would score non-trivially); mitigated by learned heads and ranker plus a logged ablation. WATCH.
- 6.4 "LambdaMART alone not enough": mitigated by trained heads and the listwise softmax fallback. WATCH (reviewer question Q1).
- Whole-test aggregation: none; reciprocal/consistency decoding across test queries is explicitly excluded. PASS.
- Sibling leakage: thread-position features use only archive structure (not labels) of the source letter's correspondents, which is legitimate deployment information (the archive exists at query time); the gold's own thread links are never used as features for itself. PASS; confirm that no feature uses another letter's mention-to-gold annotation.
- Constants derivable in script: yes.
- Source < 512 KB and plain: keep lexicons small, no embedded blobs; comments explain reasoning.
- Determinism: stable sorts, fixed seeds, `deterministic=True`, no unordered set iteration in features.

## Open questions & assumptions

Reviewer questions:
- Q1: Is LightGBM acceptable under "standard libraries" for this CPU-only task, or only numpy/scipy/scikit-learn? Reading A (LightGBM ok): primary as planned. Reading B (sklearn only): the ranker is `HistGradientBoostingClassifier` pointwise on within-pool relative features; the fallback softmax is unchanged. Compliant under both: rung 1 (numpy/scipy/sklearn). Measured cost of B vs A: to be measured in roadmap step 4.
- Q2: May corpus-level processing of `letters.csv` (the shared archive, including text of letters that are test queries' source documents) be used for per-letter derived features (dates, names, pair lists, hashed vectors)? Reading A (archive = retrieval index, allowed): as planned. Reading B (anything touching a test source letter's text beyond answering that query is test aggregation): then restrict thread-position features to pair lists built from metadata only (sender/recipient/date) and drop hashed body-text overlap; cost measured by ablation. Plan is built so dropping body-text features costs little.
- Q3: Exact submission schema (column names; ranked ids as space-separated string, comma-separated, JSON, or ten columns), and whether a sample_submission exists. Assumption: one row per test query id with a single column of ten ids in the format the description gives; the validator will be written from the sample or the description. If no sample exists, treat the schema as spec text and ask.
- Q4: Is MRR averaged per query or per document? Assumption: per query.
- Q5: Can a mention have more than one valid gold? Assumption: single gold per query (D1/D6 will check).
- Q6: Are the date/name seed lexicons (months in five languages, roman numerals, "7bre" family, units) considered acceptable "feature dictionaries"? Reading A: acceptable; Reading B: banned hand-coding, then rely only on the learned token->month and token->gap tables from train (cost measured in step 5).

Assumptions that remain unverified (no data seen): sizes of train/validation/test; date formats; whether the gold is always in the archive; whether letters from multiple correspondents share names; the gap distribution; recall at N=200 above 95%; the share of mentions containing explicit day numbers; expected MRR range. Everything in "Data findings" is a to-do, not a result.
