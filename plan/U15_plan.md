# U15 - Catalogue-text tag ranking: build plan

Status of evidence. No dataset files were available when this plan was written. Everything under "Data findings" is either (a) stated in the challenge description (marked DESC) or (b) a hypothesis plus the exact train-only diagnostic that would test it (marked UNVERIFIED). Runtime numbers are unprofiled estimates. No score is promised.

---

## Contract & decision unit

**One valid answer.** For each of the 4,792 evaluation cases, one row `case_id,ranked_tags`, where `ranked_tags` is a space-separated ordering of the case's own 80 candidate codes. We always submit all 80 (shorter lists forfeit positives; top-10 only scores about 0.037). Codes outside the pool are ignored and repeats count once, so the validator should require a permutation of the pool.

**Invalid vs merely low-scoring (DESC).** Invalid: duplicated `case_id`, renamed columns, an index column, a missing case row. Empty `ranked_tags` scores zero. Anything else is valid but may score poorly.

**Metric, term by term (DESC).**
- Per case, AP = (1/3) * sum over the 3 positives of (i / r_i), where r_i is the 1-based rank of the i-th positive reached from the top.
- Mean over cases with equal weight, with no hierarchical institution averaging.
- It is order-only, with no threshold and no calibration sensitivity.
- Reference values: ranks (1,2,3) give 1.0. Ranks (1,5,10) give 0.567. Ranks (4,5,6) give 0.383. Ranks (1,2,80) give 0.679. Random is about 0.085.
- It is convex in the top ranks, so the top 5 positions carry most of the score. A confident wrong code at rank 1 is costly. Getting 2 of 3 positives into the top 3 matters more than a perfect tail.

**Independent unit.**
- The decision unit is the case (one title plus one pool).
- The generalisation unit is the HOLDING INSTITUTION (DESC). Cases from one institution share language, literalness and templates, so they are strongly correlated. Effective sample size is the number of institutions, not 15,003.
- The institution is not given as a column. It must be reconstructed from content (see Validation design).

**Pipeline stages, diagnosed separately.**
1. *Candidate coverage.* Positives are always inside the pool by construction, so pool recall is 100%. The real coverage question is representability: what share of held-out positives are codes with training support (their positive count in the training folds). Codes with no support are "cold". Report P(positive has >=1, >=5, >=20 training positives) under grouped CV. This is the information ceiling.
2. *Scoring.* P(code c is one of the 3 | title, pool). Report MAP of the scorer alone.
3. *Decoding.* Sort by score. The only structure is "exactly 3 correct, mutually coherent". Optional set-level re-rank (Structural signals).
- Oracle check: gold scores through the sorter must give MAP = 1.000 and a permutation of the pool. This also checks the parser and the writer.

---

## Compliance regime

**Domain.** NLP, short multilingual text. Task shape: extreme multi-label ranking restricted to a given 80-candidate pool. Labelled neither "from scratch" nor "fine-tuning" in the supplied text, but the pretrained-model policy explicitly permits fine-tuning a public text model.

**Explicit bans in the description, treated as hard constraints, and how the plan honours each.**

| Ban (DESC) | Plan consequence |
|---|---|
| Hardcoding a ranking for specific evaluation cases | None. Every score comes from a trained model or train-fit statistic applied to one case's own title and pool. |
| Using `case_id` strings as signal | `case_id` is read only to key the output. It is never a feature, a group key, or a sort key. Validation groups are built from content only. |
| Using row order or the order candidates are listed within a pool | Parse each pool, then sort its codes by code string into a canonical order. Training shuffles candidates with a seeded RNG. Models are permutation-invariant (no positional encodings, no index features). Final ties are broken by code string, never by supplied order, so the implementation must not use a stable argsort on supplied order. Row order is never used, including for CV folds. |
| Identifying the item and consulting its collection / deposit / mirror / archive | No lookups of any kind. No generative LLM asked "what does this artwork look like". Titles are only encoded. |
| Retrieving or reading the picture from any service | No image access. No network access except Hugging Face weight download. |
| Private, role-gated, API-key models or external inference APIs at prediction time | Only public Hugging Face checkpoints, downloaded in-script, with pinned revisions. |

**Rules from CLAUDE.md that also bind (the description is silent on them).**
- No external datasets, word lists, translation dictionaries or lexicons.
- No synthetic labelled data. Machine translation or back-translation augmentation of titles is rejected.
- No pseudo-labels, no test-time adaptation, no test-set statistics of any kind. A candidate-frequency count over evaluation pools is whole-test aggregation, even though the description lists it as a floor (0.064). It is never used.
- Every vocabulary, tokenizer, scaler, code-id table, PCA or cluster step is fit on train only.
- No wall-clock branching, no `try/except` import fallbacks, no `cuda.is_available()` switches, no `os.cpu_count()`-derived settings. A fixed work plan with seeds everywhere.
- Source file under 512 KB, plain readable code.

**Where the description is silent, and the assumption made.**
- Runtime limit: only "Compute: A10G" is given. Assume the CLAUDE.md regime: at most 1 h worst case, target about 35-45 min.
- Model size cap: none stated. Assume none beyond A10G 24 GB memory.
- External data: not mentioned. Assume banned (CLAUDE.md).
- Whether code-level statistics learned from train answers are allowed: DESC says "what is learned from training cases and answers" is a valid signal, so yes.
- Whether pool-level (set-level) signals other than order are allowed: DESC says "the candidate pool supplied with it" is a valid input and bans only the listed order. Assume the content of the pool is usable. This is reviewer question 2.
- Whether pretrained models' world knowledge of named works is acceptable: reviewer question 4.

**Frozen-features-plus-head status.** Frozen embeddings plus a small head would be grey. Genuine fine-tuning is load-bearing here: the LP-FT rung is the compliance layer. The script logs parameter-change norm and the OOF MAP gain of fine-tune over probe.

**Strip-the-ML test.** Remove every trained component and what remains is the pool, the title, and no rule. This scores about 0.085 (random). Even the "non-learned" 0.185 baseline is a train-learned association table. Nothing hand-written solves the task, so the strip test passes.

---

## Data findings

### Facts from the description (DESC)
- 15,003 train cases and 4,792 test cases, 80 candidates each, exactly 3 positives per pool. Test institutions are disjoint from train institutions.
- Titles: median 4 words, 1 to 54 words, several European languages, some descriptive, some inventory phrases, some proper names that say nothing about the depiction.
- Tags are opaque, globally consistent codes of the form `t` plus 12 hex characters.
- Each positive was given by at least 2 describers (a consensus word). Decoys are never-given words from the same commonness band, half from above and half from below each positive.
- Published floors on evaluation: supplied order 0.084, shuffled 0.086, reversed 0.085, code-sorted 0.086. Training-answer frequency descending 0.072, ascending 0.084. Evaluation-pool frequency 0.064. Never-a-training-answer first 0.088. Title-word association 0.185. Perfect 1.000.
- Inferences from the floors:
  - Pool order carries no signal.
  - Training-answer frequency is not neutral: the descending 0.072 is clearly below the 0.085 random level, so positives are somewhat LESS frequent as training answers than their decoys. A learned per-row model must learn this sign. It must not use frequency naively.
  - Never-a-training-answer first scoring about random (0.088) means "cold" codes are positives at roughly the same rate as they are decoys. A sizeable share of positives are therefore probably cold, with no title-to-code association learnable for them. That is a ceiling on MAP.

### Diagnostics to run on train (all UNVERIFIED; hypotheses attached)

1. **Schema and integrity.**
   - Assert that every pool has exactly 80 unique codes.
   - Assert that `train_labels` gives exactly 3 codes per case and all 3 lie in the pool.
   - Check no duplicate `case_id` and a 1:1 join between `train.csv` and `train_labels.csv`.
   - Hypothesis: all hold.
2. **Floor reproduction (also the metric unit test).** Compute MAP on train for: supplied order, shuffled, reversed, code-sorted, training-answer frequency descending and ascending, and never-a-training-answer first. The training-answer frequency must be computed leave-case-out. The first four should be about 0.085. Descending frequency should be below 0.085.
   - Hypothesis: if train supplied-order MAP is far from 0.084, the pool order leaks in train. It would still be unusable (banned), but it would signal a data difference.
3. **Title profile.**
   - Distributions of word count, character count and tokenizer token count (target: max_len covering the 54-word title, so about 96 tokens).
   - Share of non-Latin script, ALL-CAPS, trailing digits, inventory patterns, and punctuation templates.
   - Share of exact and normalised duplicate titles.
   - Hypothesis: heavy mass at 2-6 words, a long tail, and many repeated templated titles inside one institution.
4. **Ambiguity ceiling.** For identical normalised titles, compute the Jaccard overlap of positive sets across different cases, and an "oracle text-only" MAP: rank a case's pool by the other cases' positives that share its exact title (leave-case-out).
   - Hypothesis: this is only informative on templated titles and gives a rough text-only ceiling.
5. **Code vocabulary.**
   - Number of distinct codes appearing in pools and distinct positive codes.
   - Positive-count histogram. Hypothesis: Zipf-like with a long tail of 1-3 positives.
   - Pool appearances per code, and per-code positive rate = positives / pool appearances.
   - Share of cases whose 3 positives all have at least 5 other training positives.
   - The relation of positive rate to training-answer count.
   - Hypothesis: positive rate is roughly flat to mildly decreasing in commonness.
6. **Cold-positive rate under grouped CV (needs the groups from Validation design).** Report P(positive code has >=1, >=5, >=20 positives in the training folds). Hypothesis: around 10-30% of positives are cold. This is the information ceiling for the scoring stage.
7. **Pool coherence (the sibling signal).** For each train pool, using co-occurrence statistics from the OTHER folds, compute each candidate's mean PMI with the other 79 and with the top-5 by title score. Report the AUC of positives vs decoys.
   - Hypothesis: clearly above 0.5, because the 3 positives were chosen for the same picture while decoys are independent frequency-band draws. This is the plan's second signal after the title.
8. **Code language and title language.** Cluster codes by their title-word / title-language association profile and look for language blocks. Compare: in a pool, what fraction of decoys sit in a different code-language block from the title's language?
   - Hypothesis: positives often match the title's language block while decoys are drawn from all languages, making a cheap and strong filter.
9. **Frequency geometry within pools (feeds a compliance decision, not a feature yet).**
   - Within a pool, sort candidates by training pool-appearance count and look at where positives fall.
   - Test whether the pool looks like 3 sub-bands of about 26 around each positive.
   - Report the within-pool rank distribution of positives under training-answer frequency.
   - Hypothesis: there is residual structure (consistent with the 0.072 floor).
   - If it is large, the measured gain is a generator artefact and goes to the reviewer (open question 1).
10. **Group recovery (scratch EDA only, never in `solution.py`).** Check whether the content-derived style clusters form contiguous blocks in row order or in case_id, as a sanity check that they recover institutions. Row order and case_id are used only to audit the grouping, never as signal.
11. **Title-type strata.** Tag each title as descriptive, inventory or proper-name using simple surface cues, for REPORTING only. Hypothesis: MAP is much higher on descriptive titles and near the random floor on proper names.
12. **Candidate-pool pruning recall.** The fraction of OOF positives inside the top 30 and the top 40 of the stage-1 ranking. Hypothesis: about 75-90% in the top 30.

---

## Validation design

**Reproducing the split.** The test split holds out whole institutions. There is no institution column, so build pseudo-institutions from content, train only:
1. Normalise titles (Unicode NFKC, lowercase, collapse whitespace; digits mapped to a placeholder). This is cleaning only.
2. Union-find over near-duplicate and templated titles: exact normalised match, or character 3-gram Jaccard at least 0.8, or a shared rare token (document frequency between 2 and about 0.2% of cases, with a cap so no giant component forms). This keeps series and templated families together.
3. Cluster the components into G style groups. Features: character 1-4-gram TF-IDF (including punctuation and case) reduced by SVD to 64 dimensions, plus the group's positive-code profile. KMeans, fixed seed. Candidate G values: 24 and 40.
4. `GroupKFold` with 5 folds over the groups, so all cases of a group are on one side.

**Why this mirrors the test.** DESC says cataloguing convention travels with the institution (language, literalness, inventory vs descriptive phrasing). The style clustering is built on exactly those axes.

**Repeats.** In development, repeat with 2 cluster seeds and 2 values of G (4 splits total) and report mean +/- std over splits and fold std. The shipped script uses one fixed 5-fold split (runtime). A change is accepted only if it beats the paired fold-level noise (more than 1 standard error of paired fold differences, positive on most folds and on both G values).

**Direction and rough size of each proxy bias.**
- Random KFold: strongly optimistic. Series and templated titles sit on both sides, and house-style phrasing leaks. Never used for decisions. Expect +0.05 to +0.15 MAP over true held-out-institution.
- Style-group GroupKFold: probably slightly pessimistic if clusters are language-pure, because a fold removes a whole language that the real test (with several train institutions of that language) may still have in training. It is optimistic if clusters fail to merge an institution's cases. Expect within about +/-0.04 MAP of the real score, direction unknown.
- Calibration against the description: re-implement the "title-word association" baseline (per-code association with title words, learned on train folds) and run it under this grouped CV. The published value on the real test is 0.185. A reproduction near 0.185 supports the proxy. A reproduction far above suggests the proxy is too easy (house style leaking), far below suggests it is too hard. Use as a soft calibrator only, since the exact association formula used for 0.185 is unknown.

**Holdout sanity fold.** Pick one style-group fold (about 20% of groups) in advance and leave it out of all architecture, epoch, loss and stage-2 choices. Report it once, near the end, as an overfitting-to-CV check.

**Metric unit tests.** Implement the exact AP from the description. Test on: perfect order (1.0), reversed (about 0.085 on random pools), constant scores (tied, broken by code string, about 0.085), the base-rate (training-frequency) predictions (about 0.072 / 0.084), and the documented examples above (ranks 1,5,10 give 0.567). Also assert that removing the ranking truncation changes nothing when all 80 are submitted.

**Post-hoc selection discipline.** Stage-2 hyperparameters, blend weights and any decode scalar are chosen on the OOF of one set of folds and reported on cross-fitted folds. The fold-model-mean test predictions use the same score scale as OOF (see Fixed work plan).

---

## Overfit/underfit risks

### Overfit
| Risk | Mitigation |
|---|---|
| House-style memorisation (institution templates mapped to codes), with only tens of effective groups | Grouped CV; word dropout; capped duplicate-title weight in training; LP-FT with tiny LR; fixed short schedule; weight decay; code-dropout to UNK; no id or order features |
| Leaky label statistics in stage 2: positive counts and co-occurrence include the row's own triple | Cross-fit by fold: a row's features use statistics from the other folds only. Test cases are scored under each of the 5 fold-statistic sets and averaged, so train-time and test-time statistics are built on the same amount of data (CLAUDE.md 4A, strategist rule 10a) |
| Stage-2 selection noise (many features, 30-candidate rerank) | Small fixed LightGBM config, num rounds from mean CV best iteration times 1.1, ablation with paired folds, simple fallback to the stage-1 ranking |
| kNN and association members retrieving same-institution neighbours | Neighbour banks come only from training folds in CV; test uses fold banks averaged |
| Per-code embeddings for rare codes are noise | Prototype initialisation from frozen title embeddings of the code's positive titles; weight decay; shrink toward UNK; stage-2 sees a train-count feature so it can discount rare-code scores |
| Selecting the encoder, epochs and loss on CV | Few conventional options only (3-4 encoders in the frozen probe, 2 schedules); decisions logged; the holdout fold reports final |

### Underfit
| Risk | Mitigation |
|---|---|
| Too-small or monolingual representation (cross-lingual transfer is the crux for unseen institutions) | Start from a large multilingual encoder chosen by frozen-probe CV across candidates (see Recommended approach); larger beats smaller until the budget says otherwise |
| Truncating long titles | `max_len` about 96 tokens with dynamic padding; titles are short so cost is negligible |
| Information lost by normalisation | Keep case and accents for the cased encoder; normalisation (NFKC, digit placeholder) is used only for grouping and for the lexical member |
| Code semantics not learned for medium-rare codes | Prototype init; LP phase to convergence before unfreezing; codes also learn through the shared encoder |
| Loss/metric mismatch | Multi-positive listwise softmax per positive against negatives (AP-aligned); stage-2 lambdarank at pool level |
| Under-training | The LP phase has 3 epochs on cached frozen embeddings (cheap) and then 3 FT epochs; the probe-vs-FT gain is logged |

### Cold-code problem (both)
Codes never seen in training pools map to an UNK id. During training, 5% of candidate code ids are randomly replaced by UNK so the model learns a sensible UNK score and relies on title-coherence for it. In CV the vocabulary is rebuilt per fold from the training folds only, so the cold rate is simulated honestly.

---

## Recommended approach (primary + fallback)

### Primary: two-stage learned ranker with a fine-tuned multilingual title encoder

**Stage 1. Bi-encoder over opaque codes.**
- Title encoder: a public multilingual encoder, pooled by mean of last-layer tokens plus CLS, then a linear projection to 256 dimensions. The encoder is chosen in step 3 of the roadmap by frozen-probe grouped-CV MAP among a short fixed list (UNVERIFIED availability, to be pinned at dev time): `intfloat/multilingual-e5-large` (primary candidate; prefix `query: `), `BAAI/bge-m3`, `intfloat/multilingual-e5-base`, `sentence-transformers/LaBSE`. Larger and sentence-embedding-pretrained models are preferred because titles are tiny and the signal is cross-lingual semantics.
- Code table: one learned 256-d vector per training-vocabulary code plus one UNK vector. Score = cosine(projection(title), code) / learned temperature, plus a small regularised per-code bias. Initialise each code vector from the normalised mean of frozen title embeddings of the training titles where that code is positive (prototype init). No code has any prior semantics, so this association must be learned here, as the description says.
- Loss: for each positive p, `-log( exp(s_p) / (exp(s_p) + sum over negatives n of exp(s_n)) )`, averaged over the 3 positives. This ranks each positive against the 77 decoys and is aligned with AP. The pool is the training distribution (already frequency-matched), so no extra negatives are needed.
- Training recipe (the LP-FT rung of the capacity ladder):
  1. Compute frozen title embeddings once per fold.
  2. Train the projection plus code table on them for 3 epochs (LP).
  3. Unfreeze the encoder (all layers, layer-wise LR decay 0.9, LR about 1e-5 for the body, 1e-3 for the head and codes, warmup then cosine to 0) for 3 epochs, bf16 autocast, gradient clipping 1.0.
  4. The script logs the L2 norm of the parameter change and the OOF MAP of probe and fine-tune.
- Augmentation: word dropout 10%, random case and accent variants of the title, random shuffle of candidate order, 5% code-to-UNK. No translation, no synthetic titles.
- Output per fold: OOF pool scores and test scores. Test scores are the mean of the 5 fold models (z-scored within each pool before averaging).

**Stage 2. Pool-level reranker (cross-fitted).**
- Model: LightGBM LambdaRank, one group per case, reranking the top K=30 candidates of the stage-1 ordering (fixed K; the tail keeps its stage-1 order). Pruning is a "cheap robust selection, then richer model" design that cuts rows from 1.2M to about 450k. Verify OOF recall at 30 (diagnostic 12) before accepting.
- Features per (case, candidate):
  - Stage-1 score, within-pool z-score, rank, softmax probability, gap to the 3rd and 4th best, margin to runner-up, and pool-level entropy and max (lets the model learn how much to trust the title).
  - Pool coherence: each candidate's mean and max co-occurrence PMI (from training positive triples, cross-fitted, own case excluded) with the top 5 and the top 10 other candidates by stage-1 score.
  - Code statistics from training (cross-fitted, leave-case-out): log positive count, log pool-appearance count, smoothed positive rate, known/UNK flag, and their within-pool rank and z-score. These are "what is learned from training cases and answers" (see Open questions).
  - Title descriptors: word count, character count, non-ASCII fraction, digit presence, capitalised-token fraction (proper-name hint), script flag.
  - Not included: any feature built from supplied pool order, case_id, row index, or evaluation-pool statistics.
- Final ranking: stage-2 score for the top 30, then the rest in stage-1 order, ties broken by code string.

**Optional members (enter only after the roadmap gate "gain larger than noise").**
- L: a hashed word and character n-gram bag-of-embeddings scorer trained from scratch with the same listwise loss. This is a different assumption (lexical, no pretraining) and cheap. It joins stage 2 as an extra feature, not a fixed-weight blend.
- B: a second encoder family (for example LaBSE if the primary is e5-large). Used for diversity of pretraining objective.
- K: kNN vote over training titles with fine-tuned embeddings, neighbour banks from training folds only.

### Fallback
Stage 1 alone with the best encoder and the plain softmax ordering, no stage 2. If stage 2 does not beat stage 1 by more than the paired CV noise on the grouped folds, ship stage 1 (it has fewer moving parts and no stacking leakage risk).

### Why this fits THIS data
- The target is a word-level concept match between a short multilingual title and an opaque code. The model must learn the code semantics from 15,003 x 3 positives and 1.15M decoys. A shared encoder gives cross-lingual transfer to unseen institutions, and the per-code table gives memory.
- Pools are frequency-matched, so a pure popularity prior is neutral to slightly harmful. The listwise-per-pool loss learns the correct conditional behaviour instead of a global frequency.
- Three positives describe the same picture and decoys are independent band draws, so within-pool coherence is a built-in second signal beyond the title.
- Many titles are uninformative (proper names, inventory numbers). Giving stage 2 the stage-1 confidence and title descriptors lets it fall back on coherence and code statistics when the title is empty of content.

### Expected private MAP (estimate only, low confidence)
About 0.26 to 0.36, point guess around 0.30. Reasoning: the non-learned association baseline gives 0.185 on test. A multilingual fine-tuned scorer plausibly adds a clear margin on descriptive titles. Cold positives and proper-name titles hold the ceiling down. Coherence features add some more. I could not verify any of these numbers without the data.

---

## Rejected options

| Option | Reason |
|---|---|
| Title-word association / TF-IDF / PMI as the complete solution | It is the stated 0.185 baseline. Grey as a core under CLAUDE.md 2.5 and capped in quality. Kept only as the lexical member L and as the validation calibrator |
| Frozen embeddings + GBDT or logistic head only | Grey or likely rejected; fine-tuning must be load-bearing. The probe is the rung below, kept as the LP warm start and as the honest yardstick |
| Generative LLM (zero-shot or LoRA) describing the picture from the title | Inference-only and "identify the item"-adjacent; the description bans consulting the item's collection; also far over budget for 15k x 5 folds on an A10G within 1 h |
| Machine translation / back-translation of titles | Synthetic data (banned) and external model generation |
| Cross-encoder over (title, code text) | Codes are opaque; there is no code text to encode |
| Random KFold validation | Optimistic by construction; fails the held-out-institution split |
| Using pool order, case_id, row order or evaluation-pool frequencies | Explicitly banned or whole-test aggregation |
| Hand-coding the frequency-band geometry (reconstruct 3 sub-bands of about 26 and pick each band centre) | Hard-coding a data-generator artefact; fails the strip-the-ML spirit and the "exploits data generation" test. Only generic learned within-pool relative features are considered, behind an ablation and a reviewer question |
| Refit stage 1 on 100% of data, then feed stage 2 | Test features would come from a stronger model than the OOF features stage 2 was trained on. Fold-model averaging keeps the scales consistent |
| Pseudo-labelling, test-time adaptation, TTA across the test file | Banned (CLAUDE.md 2.3 #5) |
| A set-transformer over all 80 code tokens as the primary | Heavier than the primary-design gate allows; kept as a later roadmap candidate |
| Per-institution models / institution-id features | There is no institution id at test, and it would not transfer to unseen institutions |

---

## Fixed work plan & runtime budget

All numbers are unprofiled estimates for an A10G (24 GB), to be replaced by measured values before freezing counts. No branch depends on elapsed time; time is logged only.

**Fixed constants.**
- SEED = 42. Seed `random`, `numpy`, `torch`, CUDA, `PYTHONHASHSEED`, DataLoader `Generator`. `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch.
- N_FOLDS = 5 (style-group folds), G = 40 style clusters, `num_workers` fixed (title tokenisation is cheap; use 0 or 4 hardcoded), torch and LightGBM thread counts hardcoded (for example 8).
- Stage 1: 3 LP epochs on cached embeddings plus 3 FT epochs, batch 64 pools, `max_len` 96, bf16 autocast.
- Stage 2: K = 30, fixed LightGBM params (num_leaves 31, learning rate 0.05, min_data_in_leaf 100, feature_fraction 0.8, `deterministic=True`, `force_row_wise=True`, fixed seed), number of rounds = mean CV best iteration times 1.1 (a data-determined, in-script value).
- Pretrained revisions pinned by commit hash (recorded at dev time; UNVERIFIED here). Weights come only from Hugging Face. `HF_HOME` is redirected under the submission directory.

**Estimated runtime (A10G).**

| Stage | Work | Est. min |
|---|---|---|
| 0 | Load, validate, canonicalise pools, build groups (char-ngram SVD + KMeans + union-find), code vocab per fold | 2 |
| 1 | Download weights (about 2 GB per large model), frozen embeddings for about 20k titles | 3 |
| 2 | Cross-fitted co-occurrence and code statistics | 1 |
| 3 | Member A (large encoder), 5 folds x (LP about 0.5 + FT 3 epochs about 3 + test inference about 0.3) | 18 |
| 4 | Member B (base-size encoder), 5 folds | 6 |
| 5 | Member L (lexical hashed bag), 5 folds | 1.5 |
| 6 | Stage-2 feature build plus 5 LightGBM fits plus final fit (about 450k rows) | 4 |
| 7 | Build ranking, validator, write and reload CSV | 0.5 |
| | **Total** | **about 36** |

- Headroom against the 1 h worst case is about 40%. If measured runtime exceeds about 40 min, drop member B (or use the base encoder for A) as an offline decision and hardcode it.
- Memory: a large encoder, bf16 autocast with fp32 AdamW states, is roughly 9-12 GB at batch 64 and short sequences, within 24 GB. Free GPU memory between folds (`del model; torch.cuda.empty_cache()`). The code table (maybe 10-30k codes x 256) is negligible.
- Determinism caveat: CUDA atomics in embedding backward may leave tiny nondeterminism even with `warn_only`. The plan is robust to it (averaging over 5 fold models, rank-based stage 2) and run-twice agreement is a roadmap check.

**Input/output validation inside the script.**
- Before training: assert pool size 80, positives in pool, unique ids, required columns present.
- Before writing: the CLAUDE.md `validate_submission` adapted to `case_id,ranked_tags`, plus each row's ranked list must be a permutation of that case's pool (80 unique codes, no empties). Reload the written file with `keep_default_na=False` and re-check.
- No constant-output fallback is written early. A failed run should fail loudly.

---

## Metric-aware training & decode

- **Back-solve.** AP depends only on the ranks of the 3 positives. A positive moving from rank 4 to rank 1 gains much more than one moving from 40 to 20. Use top-heavy training signals: multi-positive softmax (each positive against all negatives) and, in stage 2, LambdaRank with a truncation level of about 20 and `eval_at` [3, 10, 30]. Compare LambdaRank against `rank_xendcg` and a binary objective on paired folds, and pick the best by OOF MAP.
- **Per-example weights.** Every case has the same structure (3 of 80), so equal case weights replicate the metric. No hierarchical (per-institution) averaging exists in the metric, so no institution reweighting is applied by default. An institution-balanced weighting (1/sqrt(group size)) is a roadmap experiment aimed at robustness, accepted only if it helps across folds.
- **Expected-utility decode.** With a calibrated posterior and exactly 3 positives, sorting by marginal probability is near-optimal for AP. A per-pool softmax scaled to sum 3 is monotone, so it does not change the ranking. No decode constants are tuned in the primary.
- **Hard constraint.** The output must be a permutation of the pool. A set-level re-rank only reorders within the pool.
- **Optional exact structure (roadmap step 5, after the lean design is measured).** Replace the coherence-feature stage with an exact 3-subset model: P(S) proportional to exp(sum of unary scores + sum of low-rank pairwise code potentials) over all C(80,3) = 82,160 triples per pool, trained by exact marginal log-likelihood of the gold triple, ranking by exact marginals. This is enumerable on GPU. It is the "explicit latent structure" version of the coherence signal. It enters only if it beats the GBDT coherence features by more than noise.
- **Calibration.** Stage 2 is a ranker, so no probability calibration is shipped. The cross-fitted OOF MAP is the reported number.

---

## Structural signals

Each invariant, and how it is used. Items marked "verify" need a train diagnostic first.

1. **Exactly 3 positives per pool** (DESC): listwise loss over the pool; enumerable 3-subset structure for the optional exact decoder. Verify the count on all train cases.
2. **Positives are mutually coherent; decoys are independent band draws** (UNVERIFIED, diagnostic 7): pool-coherence features and the optional triple model.
3. **Pool order carries no information** (DESC floors): permutation-invariant model, canonical code-string sort, seeded shuffle in training, tie-break by code string.
4. **Codes are globally consistent across institutions** (DESC): one shared code table. The code representation is the transferable part, the title phrasing is what shifts.
5. **Cross-lingual structure** (UNVERIFIED, diagnostic 8): a multilingual encoder maps title concepts into one space. If codes form language blocks and decoys ignore title language, the learned code table plus encoder will pick that up without hand-written rules.
6. **Cold codes**: UNK embedding with training-time UNK dropout. Stage 2 gets a known/UNK flag and the code training count.
7. **Frequency matching** (DESC): the listwise loss makes the per-code bias reflect P(positive | in pool), not raw popularity. The floors (descending 0.072) show a small residual relation. It enters only as generic cross-fitted within-pool relative features behind an ablation (open question 1). The plan does not reconstruct the sub-band geometry by hand.
8. **Consensus tags are concrete visible words**: expect learned codes with high per-code positive rates for common visible nouns. This is learned through the code bias and the stage-2 code statistics, not hard-coded.
9. **Title invariances**: case, accent and digit variation do not change the concept (augmentation); word order matters little (word dropout).
10. **Augmentation that is NOT used**: no flipping or translation of titles into other languages (synthetic data), no fabricating blended titles.

---

## Experiment roadmap

Each step has a stop criterion. One change per experiment, paired folds, repeated splits in development. Keep a log table (id, change, CV mean +/- std, per-fold, est. runtime, notes).

1. **Contract, metric, validation.** Implement exact AP and the unit tests. Reproduce the published floors on train (diagnostic 2). Build content-derived groups and the 5-fold split. Run the full set of train diagnostics. Stop when the floors reproduce and the groups look sane (group sizes, language purity, audit against row-order blocks in scratch only).
2. **Cheap valid baseline, end-to-end.** Title-word association scorer (train-learned PMI-style) under grouped CV and a valid submission. Compare with 0.185 to calibrate the proxy. Write a valid CSV through the validator. Stop when the pipeline runs from clean and the CSV validates.
3. **Representation ladder (stage 1).** For each candidate encoder: frozen probe (prototype codes + learned projection) under grouped CV. Pick the best by frozen MAP, then climb: LP-FT (last layers, then all layers). Stop at the lowest rung that beats the rung below by more than noise. Log parameter-change norm and probe-vs-FT gain. Decide `max_len`, LR and epoch count (conventional values, two options at most).
4. **Loss and training details.** Compare: per-positive-vs-negatives softmax, multi-positive full softmax, BCE; with and without code-dropout/UNK, word dropout, per-code bias. Keep only changes beating noise.
5. **Stage 2 and structure.** Add coherence features, code statistics, relative transforms, title descriptors. Ablate each family on paired folds (including the frequency family for the compliance decision). Compare LambdaRank / xendcg / binary. Then optionally test the exact 3-subset decoder against the GBDT coherence features.
6. **Diversity.** Add L, then B, then K, each scored alone first and added as stage-2 features only if the gain exceeds noise and its alone-score is close to the primary's. Prefer features into stage 2 over fixed-weight blends.
7. **In-script bounded selection.** If any selection stays (for example the stage-2 objective or rounds), make it a fixed small grid (at most 6 configs, `n_jobs=1`, seeded, no timeout) with its own cross-fitted check.
8. **Freeze and verify.** Hardcode the final counts. Run twice from a clean `working/` and diff the submissions (rank correlation of test scores about 1; same CV). Check the holdout sanity fold. Check the test ranking distribution against OOF (score scale, entropy distribution), as a bug catch only. Run the CLAUDE.md section 5 checklist.
9. **Credits.** Spend them on: the baseline (step 2), best single stage 1 (step 3-4), full two-stage (step 5-6), final (step 8). Do not spend credits on small tweaks.

---

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits, applied to the plan.

| Check | Result |
|---|---|
| Any code path reads the test file for more than one-case prediction (stats, vocab, scaler, clustering, dedup, rank-normalising across the file, pseudo-labels) | No. Vocab, tokenizer fit, groups, co-occurrence, code statistics, GBDT all fit on train. Within-pool z-scores use only that case's own pool. Test titles are encoded and scored one case at a time |
| Time in any `if`/loop/argument | No. Time is logged only |
| `cuda.is_available()`, `os.cpu_count()`, import fallbacks | None; device is `"cuda"`, threads and workers hardcoded |
| Hardcoded constants tuned offline | Architecture and schedule come from a few conventional defaults selected by grouped CV at development time and recorded in comments. The stage-2 rounds are derived in-script. Flagged as open question 6 |
| External data, synthetic data, self-hosted weights, non-allowed library | None. Public Hugging Face encoders only; torch, transformers, lightgbm, scikit-learn, numpy, pandas |
| Would it still work with the ML removed | No. About 0.085 (random) without the trained parts |
| Description bans (case_id, row order, pool order, item identification, picture retrieval, private or API models) | Each honoured (see Compliance regime table) |
| Sibling leakage: features cross-referencing group-mates when the target is a relation among them | Coherence is among the case's OWN candidates using train-fit statistics, with the own case excluded and statistics cross-fitted. It is a within-row signal using the pool the description permits. Flagged as open question 2 |
| Related-row leakage across cases | Neighbour, co-occurrence and count statistics for a row come only from other folds (groups), never its own group |
| Whole-test aggregation | None |
| Source readable, under 512 KB, no blobs | Planned as plain, commented code |
| Self-contained from raw data every run | Yes. Only the Hugging Face weights are downloaded |
| Genuine training load-bearing | Yes. Fine-tuned encoder, learned code table, trained GBDT |
| Frozen-features-only exposure | Mitigated by LP-FT as the compliance layer, parameter-change norm and probe-vs-FT gain logged |

---

## Open questions & assumptions

### Questions for a reviewer (with the plan under each reading)
1. **Code-level and within-pool relative frequency features.** The description says "how common a code is carries no information", but it also lists "what is learned from training cases and answers" as a valid signal, and the 0.072 floor shows a residual relation.
   - Reading A (allowed): ship cross-fitted code statistics and their within-pool relative transforms in stage 2.
   - Reading B (generator-artefact exploitation is disliked): ship stage 2 without the frequency family. The measured CV cost of dropping it is reported from the ablation.
   - Default: ship the version that is compliant under both readings (Reading B) unless the ablation shows the family is negligible, in which case it is dropped anyway. If it shows a large gain, treat that as a sign of exploiting the generator and ask before shipping.
2. **Pool coherence.** Is using co-occurrence between a case's own candidates (set-level evidence from the supplied pool, with train-fit statistics) acceptable? The description allows "the candidate pool supplied with it" and bans only its order.
   - If not allowed: ship stage 1 only (the fallback). The cost is the measured coherence gain.
3. **Frozen embeddings for prototype initialisation and the optional kNN member.** Acceptable given that the encoder is genuinely fine-tuned and the heads are trained?
4. **World knowledge in the pretrained encoder.** For proper-name titles, a public multilingual encoder may carry general knowledge about artists or named works. We assume this is allowed (the policy permits public pretrained text models) and that it is not "identifying the item and consulting its collection". Confirm.
5. **Runtime limit.** The description only says "Compute: A10G". Confirm that CLAUDE.md's 1 h worst case applies.
6. **Offline selection of encoder and schedule.** Does selecting among a few conventional configurations by grouped CV at development time count as "tuned constants"? The plan documents the options and keeps them few. If not, an in-script fixed-trial selection (at most 3 configurations) can replace it at a runtime cost.

### Assumptions
- One A10G, Hugging Face reachable for weights only; model names above exist and can be pinned by revision (UNVERIFIED).
- Pools are exactly 80 unique codes with exactly 3 positives, as described (to be asserted in the script).
- Institutions are not observable. Content-derived pseudo-institutions are a proxy whose bias direction is unknown (see Validation design).
- Cold positives (no training support) are a sizeable share, per the "never-a-training-answer first = 0.088" inference. No learned model can rank them well, so the realised ceiling is below 1.0.
- Runtime numbers, expected MAP band (0.26-0.36), group counts, vocabulary sizes, recall at 30, coherence AUC, and language-block structure are all UNVERIFIED hypotheses to be confirmed by the diagnostics above.

### What could not be verified in this run
No dataset was available, so no train statistics, no CV, no timing and no model availability checks were done. Every quantitative expectation in this plan is a hypothesis, not a measurement.
