# T8 plan: recovering OCR batch origin from character noise

Status: plan only. No dataset was available, so every data statement below is a HYPOTHESIS or a DIAGNOSTIC TO RUN, marked "(unverified)". Sources read: the eris-strategist agent file, CLAUDE.md, and the T8 challenge text. Nothing else was opened.

## Contract & decision unit

**One valid answer.** A CSV `submission.csv` with columns `bag_id, batch_id` and one row per test bag (110 rows, same order as `sample_submission.csv`). `batch_id` is a whitespace-separated list of integer labels, one per snippet in that bag, ordered by the INTEGER after `::` in the snippet id. The order must be numeric ("::10" after "::2"), never lexicographic. Labels are arbitrary symbols and only the partition matters.

**Invalid vs low-scoring.**
- Invalid, so the row or file is rejected: wrong label count for a bag, wrong column names, missing or extra bags, NaN or empty strings, wrong order.
- Valid but scores 0: all-one-group, all-singletons, or any partition with ARI <= 0.
- Because of this gate, the decoder always emits K in {2,3,4} clusters. The description says each bag mixes 2-4 batches, so K=1 and K=n are never emitted.

**Metric, term by term.**
- score = mean over the 110 bags of max(0, ARI(truth_b, pred_b)). Each bag has equal weight regardless of its size (16-50 snippets).
- ARI is permutation-invariant, depends only on the partition, and penalises merges and splits through pair counts.
- Large clusters dominate the ARI pair counts. Getting the big batches in a bag right matters more than isolating a 2-snippet batch.
- The clip at 0 makes the per-bag payoff convex near 0. A risky partition with E[ARI] = 0 costs nothing, while a safe partition cannot go below 0.
- Reference points from the challenge text: shipped reference 0.342, AI baseline 0.40, leaderboard high 0.44. Weak baselines: word TF-IDF 0.079, char TF-IDF 0.038, length-only 0.089, closed-set classifier to the nearest TRAIN batch 0.210, banned global-pooling shortcut 0.094.

**True independent unit.**
- Decision unit: the bag, partitioned on its own. Only that bag's rows may be used.
- Learning unit: the BATCH. Train has 19 batches, so the effective sample size for generalisation is about 19 groups, not the number of rows. The 110 test bags are drawn from 16 unseen batches, so test bags are not independent either. The standard error of the private mean is roughly sd(per-bag ARI) / sqrt(110), about 0.02 if sd is around 0.25 (unverified). Differences below about 0.02 are not resolvable.

**Pipeline stages, diagnosed separately.**
1. Representation or candidate coverage: do batch-discriminative, content-invariant noise statistics exist in the snippet at all? Diagnostic: leave-batches-out pair AUC of the features.
2. Scoring: a calibrated same-batch evidence score for each snippet pair (log-likelihood ratio).
3. Decoding: choose the number of clusters K and the partition inside the bag. Diagnostic: ARI when the decoder is given oracle pair scores built from true held-out batch statistics (diagnostic only, on train held-out batches). That separates scoring loss from decode loss.

## Compliance regime

**Domain.** Text, from-scratch regime (CLAUDE.md 6.8 plus the challenge's own limits). The challenge text overrides CLAUDE.md where they differ:
- CPU only (10 cores, 62 GB, 1.5 h). No A10G, no CUDA branch, no GPU assumptions. Use fixed thread counts as constants.
- No network, no pretrained weights, no pretrained tokenizers, no external data.
- Plan for at most about 30 min of the 90 min limit.

**Explicit bans from the description, and how the plan respects each.**
- Match test rows to a source archive: never done. No content lookups, no near-duplicate search against external text.
- Pooling or clustering the whole test set, and whole-test statistics: never done. No test-wide normalisation, vocabulary, PCA, scaler or prior. All fitted state comes from train only. Each bag is processed by a function of its own rows plus the train-fit model.
- Leaning on snippet length or language identity: handled by design, not just by hope.
  - Length features are excluded.
  - Pair sampling is length-matched.
  - Letter-identity features are restricted and ablated.
  - Training uses random-crop augmentation.
  - An extra length-matched evaluation is run (see Validation design).
- Strip-the-ML test: remove the learned pair metric, leaving raw clustering on the same features. The expected result is at or below the TF-IDF level (about 0.04-0.10, unverified). The decoder is a generic Bayesian clustering routine with no hard-coded batch knowledge, so the trained metric is load-bearing.

**Allowed or grey techniques.**
- Allowed: hand-built, content-agnostic OCR-noise features fed to a trained scorer. These are standard domain features and are not rule-based solving.
- Allowed: episodic training on bags simulated by recombining real labelled train snippets. This is recombination of real units, not synthetic data. It is listed as a reviewer question once.
- Grey: letter-level confusion statistics, which partly reveal language. Decision: tier them, ablate them, keep only if they win under leave-batches-out CV and are cleared by the reviewer.
- Grey: within-bag standardisation. The challenge allows using the bag's own rows, so it is allowed, but it is flagged.

**Genuine-training layer.** A closed-form two-covariance PLDA alone could be read as "just statistics". So the primary has a second stage: a discriminative fine-tune of a low-rank metric and calibration, trained by gradient descent for a fixed number of steps on simulated bags, initialised from the generative PLDA (the linear-probe-then-fine-tune ladder rung). Log the parameter-change norm and the CV gain over the PLDA init.

## Data findings

All items below are diagnostics to run on TRAIN. Expected outcomes are hypotheses (unverified). From test I use only: schema, row count (3,687), bag count (110), id format, snippets per bag (16-50) and snippet size statistics for runtime and memory. I do not study test content or feature distributions to choose features, models or thresholds.

1. **Structure and ids.**
   - Parse train ids, and check whether they carry a bag or batch component.
   - Compute exact snippets per batch (19 batches), the min, max and imbalance, the total row count, and the missing or NaN count.
   - Train rows are ordered by batch (stated). Row order and the integer after `::` are never features. The test index could correlate with batch if the generator sorted by batch. It must not be used as a prior, and this is stated as an assumption.
   - Expect: well-formed ids and no NaNs (unverified).
2. **Length profile.**
   - Per-batch distribution of characters and tokens, and the leave-batches-out pair AUC of length-only. The challenge's 0.089 length-only score implies some batch-length signal.
   - Expect: length is weakly informative and batch-confounded (unverified). Action: length-matched negatives and length-matched evaluation. Never a model feature.
3. **Character inventory.**
   - Per-batch Unicode category histogram, set of non-ASCII symbols, and symbols that occur in only 1-3 batches.
   - Hypothesis: a few rare symbols (long-s, ligatures, stray glyphs) mark batches in train but will not exist in unseen test batches. These would be memorised. Action: hash rare symbols into buckets by Unicode block and category, not by identity, and cap per-symbol weight.
4. **Noise-rate profile and transferability.**
   - Compute per-snippet noise rates: non-alphanumeric rate, punctuation-run rate, digit-in-word rate, isolated single-letter rate, mid-word uppercase rate, double-space rate, space-before-punctuation rate, hyphen and line-break artefacts, mean word length, share of tokens with unusual character-class transitions.
   - For each feature, compute the intraclass correlation (between-batch variance over total variance).
   - Split the 19 batches into two disjoint halves and check that the high-ICC features agree across halves, which tests transferability to unseen batches.
   - Expect: ICC in the 0.1-0.5 range for several features, with only partial agreement across halves (unverified). This is why the ceiling is low (reference around 0.34-0.44).
5. **Duplicates and shared content.**
   - Exact duplicates within and across batches.
   - Shingled near-duplicates (character 8-grams) to see whether the same underlying text appears under different batches.
   - If it does: paired noise differences are a very strong supervised signal. Use them as hard pair mining, not by matching to test.
   - If it does not: content is confounded with batch in train. This is the closed-set trap and explains word TF-IDF 0.079 versus closed-set 0.210.
   - Expect: little shared content (unverified).
6. **Content-versus-noise confounding.**
   - Compare a within-train random-split batch classifier on word features (high accuracy expected) against the same features under leave-batches-out clustering (near 0.08 expected).
   - This quantifies the generalisation gap that content features create, and sets what the noise-only model must beat.
7. **Within-batch heterogeneity.**
   - Check whether a batch is one noise population or a mixture (several scans, sources or dates).
   - Method: cluster each batch's features, check silhouette against a random split, and check bimodality of within-batch pair scores.
   - If heterogeneous, within-batch pair scores are bimodal and the single-Gaussian PLDA will under-merge. This is a risk to track.
8. **Small-snippet reliability.**
   - Distribution of characters per snippet, and the fraction of snippets too short for stable rates (under 40 characters).
   - Action: Hellinger or log transform of rates, shrinkage of rates toward the train mean, and learned or fixed precision (see open questions).
9. **Information ceiling.**
   - Oracle A: pair AUC and bag ARI when each held-out batch's true centroid and covariance are used (diagnostic only, on train held-out batches).
   - Oracle B: the closed-set classifier accuracy within train.
   - Oracle C: ARI of the best possible decoder given true pairwise same/different labels with noise.
   - These three give the ceiling and show how much is lost in scoring versus decoding.
10. **Relationship between batches.**
    - Cluster the 19 batch centroids. If batches form families (same engine, different resolution), a cohort-bank embedding (see Recommended approach) should help.
    - The closed-set 0.210 on unseen batches already suggests test batches resemble some train batches.

**Expected hypotheses (unverified).**
- Noise features alone give leave-batches-out pair AUC around 0.65-0.80.
- ARI is dominated by a few easy bags with 2 clearly different batches.
- Bags with 4 batches or unbalanced mixes are the hard ones.
- Decoding K wrongly costs more than the scorer's remaining slack.

**Test inputs used.** Schema, 3,687 rows, 110 bags, id format, snippet lengths for runtime. Nothing else.

## Validation design

**How the test split was made (inferred from the description and train structure only).** Held-out BATCHES: 16 batches never present in train. Bags are formed from 2-4 of those batches, with 16-50 snippets each. Validation mirrors this exactly: hold out whole batches, never random snippets.

**Primary scheme: repeated leave-batches-out episodic validation.**
- Make 5 folds of the 19 train batches (about 4 batches each, 3-4 in some). Repeat the partition with 3 split seeds, giving 15 held-out evaluations.
- Inside each held-out fold, simulate about 400 bags per fold per seed. Draw K uniformly from {2,3,4} and K distinct held-out batches. Draw size n uniformly from [16, 50]. Draw mix proportions from Dirichlet with alpha sampled from {1, 3, 10}, ensuring at least 2 snippets per batch when n allows. Draw snippets without replacement.
- For K=4 with only about 4 batches in a fold, the 4-batch bags reuse all held-out batches. The number of distinct 4-sets is small, so vary the folds and seeds.
- Model fitting uses only the other folds' batches. Train-on 14-15 batches, final fit on 19, so CV is slightly pessimistic.
- Report mean and std over fold-seed pairs, plus the bag-level standard error. Accept a change only when it beats noise on paired folds, in at least most fold-seed pairs.
- Also report ARI by K, by bag size and by mix balance.

**Own groups.** Batch id is the group. A second check for hidden grouping: if near-duplicate content exists (diagnostic 5), union-find snippets by shingle overlap and keep each connected component inside one side of a split.

**Simulation calibration check against the published reference numbers.** Implement the baselines the challenge states and run them on simulated bags. Compare orders of magnitude only, not as tuning targets:
- word TF-IDF near 0.08
- char TF-IDF near 0.04
- length-only near 0.09
- closed-set train-batch classifier near 0.21

If the simulation reproduces this ordering, it is test-like. If it does not, revisit the K prior, proportions, snippet sampling and which batches share content. This calibrates the harness without peeking at test data.

**Metric implementation.** Re-implement mean max(0, ARI) from the description and unit-test it on:
- perfect labels (1.0)
- relabelled permutation (1.0)
- all-one-group (0)
- all-singletons (0)
- random labels (about 0)
- reversed/anti-structured partitions (clipped to 0)

Also test the output formatter: numeric ordering of `::` ids, label count equals snippet count per bag, round-trip re-read of the written CSV.

**Post-hoc selection needs its own held-out check.** The hyperparameter grid (PCA dimension, shrinkage, within-bag centring, temperature, K penalty, fine-tune strength) is small (at most about 24 configs). Report (a) best-config CV, and (b) a nested estimate where the config is chosen on inner splits within each outer fold's training batches. If (a) exceeds (b) by more than about 0.01, choose the plateau centre, not the argmax. Calibration constants are cross-fitted: fit on OOF scores from other folds and applied to the held-out fold.

**Extra length-matched evaluation.** Re-evaluate the same models on bags resampled so that every batch in a bag has the same length-bin histogram. The ARI must not drop much. A large drop means the model leans on length, which violates the rules.

**Expected relation to the real score.** Self-built proxies usually over-estimate. The real number is expected below the CV mean.

## Overfit/underfit risks

**Overfit.**
1. 19 independent groups. Mitigation: low-capacity model (PCA to about 16-64 dims, rank-limited between-batch covariance, strong shrinkage), regularisation picked by leave-batches-out CV across a wide grid including the strong end.
2. Content memorised as batch identity (the 0.21 closed-set pattern). Mitigation: content-agnostic features only; hashed rare symbols; length-matched negatives; crop augmentation; evaluation on held-out batches only.
3. Rare-symbol memorisation. Mitigation: cap symbol weights, bucket by block and category.
4. Between-batch covariance estimated from 19 means has rank at most 18. Mitigation: shrink toward diagonal, cap its rank, and use the discriminative stage only with few free parameters.
5. Selection on the reported CV. Mitigation: nested check, fixed small grid, plateau choice.
6. Simulation prior mismatch (K distribution, mix, size). Mitigation: randomise alpha, evaluate sensitivity by K and balance, keep decode constants low-dimensional.
7. Calibrated temperature and K penalty overfit to simulated bags. Mitigation: cross-fitting, two scalars only.

**Underfit.**
1. Class-level features alone (letter, digit, punctuation classes) may throw away letter-specific OCR confusions. Mitigation: add tiered symbol-level and rare-bigram features, ablated under CV.
2. A linear Gaussian pair model cannot represent nonlinear noise interactions. Mitigation: discriminative fine-tune; fallback pair classifier; roadmap neural set encoder.
3. Single-Gaussian batch model under heterogeneous batches (diagnostic 7). Mitigation: allow a 2-component within-batch mixture if diagnostic 7 shows bimodality.
4. Too aggressive PCA or shrinkage. Mitigation: grid includes the weak end.
5. Short snippets give noisy rates. Mitigation: shrinkage and transforms, and crop augmentation so training sees short inputs.
6. Length-matched sampling could discard legitimate signal. Mitigation: use it only for pair negatives, and measure the cost.

## Recommended approach (primary + fallback)

**Primary: PLDA-style batch-verification model on content-agnostic OCR-noise features, followed by within-bag Bayesian agglomerative decoding.**

1. **Features (stateless or train-fit only).**
   - Count-based relative frequencies of OCR-noise statistics, transformed with sqrt (Hellinger) to stabilise variance. The families are:
     - Unicode-category and character-class unigrams and class-bigrams/trigrams.
     - Punctuation and symbol frequencies, hashed by code point into a fixed number of buckets (stateless, so no test-fit vocabulary).
     - Token-shape statistics: mixed digit-letter tokens, isolated single letters, mid-word capitals, punctuation runs, double spaces, space-before-punctuation, hyphen and line-break artefacts, word-length histogram.
     - Optional tier: a small set of rare-bigram or confusable-pair rates, subject to ablation and the language-identity question.
   - No raw length, no token identities, no word TF-IDF. Fit a standard scaler and PCA on TRAIN only, to a few dozen dimensions chosen by leave-batches-out CV.
2. **Generative stage.** Estimate a two-covariance PLDA on the training batches: between-batch covariance B (shrunk, rank-limited) and within-batch covariance W. Use simultaneous diagonalisation of W and B so the pair and group log-likelihoods are closed-form per dimension. Training snippets are augmented with random contiguous crops (50-100% of the snippet) so that W reflects realistic noise and length is not exploited.
3. **Discriminative stage (the genuine-training layer).**
   - Initialise a low-rank metric and bias from the PLDA, then train with a pairwise BCE and a bag-level loss on episodic bags simulated from train batches.
   - Use hard-batch mining at the batch level (batches nearest in train-centroid space), length-matched negatives, and a fixed number of epochs and steps with a fixed learning rate and weight decay.
   - Log parameter-change norms and the held-out gain over the PLDA init.
   - Fit per-pair calibration (temperature, bias) on cross-fitted OOF scores.
4. **Per-bag decode (see Metric-aware training & decode).** Compute the bag's pairwise log-odds matrix from its own rows only. Run Bayesian hierarchical agglomeration with the PLDA group marginal likelihood. Record the partitions with K = 4, 3, 2 and pick K by total marginal log-likelihood plus one calibrated K penalty. Refine with a few deterministic reassignment sweeps.
5. **Refit** on 100% of the 19 batches with the config chosen by CV, since a handful of scalars is cheap and more batches helps the between-batch covariance most.

**Why it fits THIS data.** The task is open-set verification: the test batches are never in train. Closed-set classification gets 0.21 and content features get under 0.10. PLDA-style models are the standard few-parameter answer for "same source?" with unseen sources, and the capacity ladder says to use a strongly regularised linear model when the independent-group count is about 19. The model is explicit about the latent structure: each snippet is a draw from an unobserved batch, and the group marginal likelihood sums that out exactly.

**Fallback: discriminative pair classifier plus correlation clustering.**
- Same features. Train LightGBM (CPU, fixed seed, fixed trees) on symmetric pair features: absolute difference, elementwise product, and within-bag relative features (rank of the pair distance among the row's own neighbours).
- Train on pairs from simulated episodic bags, with hard negatives from the nearest train batches (mined from train only). Fold-wise leave-batches-out evaluation. Calibrate per-pair probabilities and decode with correlation clustering at K in {2,3,4}.
- Reason to keep: diversity of assumption (nonlinear, discriminative, no Gaussian model), and robust if the Gaussian assumption breaks (diagnostic 7).

**Roadmap members, entering only if each beats noise under the same paired folds.**
- Cohort-bank embedding. Represent each snippet by its scores against held-out-style banks of train batch models. This is the reference-normalised evidence idea: at train time the bank excludes the batches in the bag, with fixed bank size (about 10) and several banks averaged at test time.
- Small neural set encoder over the bag's own snippet embeddings (attends only within the bag). This is allowed because it uses only the bag's rows. CPU-only, small.
- A small character-level CNN (from scratch) trained with supervised contrastive loss on episodes.

## Rejected options

- **Closed-set classifier to nearest train batch** (0.210 on unseen batches): it forces every snippet into one of the 19 known batches, which are not the test batches.
- **Word or character TF-IDF** (0.079 and 0.038): content-driven, banned as the complete solution, and confounded with batch in train.
- **Length-only or length as a feature** (0.089): explicitly banned.
- **Language-identity features**: explicitly banned. Letter-level features are tiered and ablated for this reason.
- **Test-wide clustering, global pooling or normalisation** (0.094 for the shortcut): explicitly banned.
- **Pretrained models or embeddings of any kind**: no pretrained weights allowed, CPU only, no network.
- **Matching snippets to a source archive or external text**: explicitly banned.
- **Large end-to-end neural clustering** (differentiable clustering, deep transformers): too many parameters for 19 groups, and violates the lean-primary gate. Later roadmap step only.
- **Full-data GBDT on raw snippet features with batch id as target**: closed-set again.
- **Using row order or the `::` index as a prior** (adjacent snippets same batch): an exploit of the data generation, likely rejected, and unverifiable on test.

## Fixed work plan & runtime budget

All counts are constants at the top of the script. No time-based branching and no environment-dependent fallbacks. Seeds fixed for `random`, `numpy` and `torch`. `torch.set_num_threads(10)` and fixed BLAS thread counts. CPU only, with no `cuda.is_available()` switches. Train N is unknown, so the estimates assume at most about 100k snippets of at most about 600 characters (unverified), which is the worst plausible case.

| Stage | Fixed plan | Est. CPU time (10 cores) |
|---|---|---|
| Read, parse ids, validate schema | single pass | under 1 min |
| Feature extraction (train + test) | vectorised counts, hashed buckets, crops: 3 crop draws per train snippet | 2-6 min |
| Scaler + PCA + generative PLDA (per fold, and final) | 5 folds x 3 seeds, closed form | 1-2 min |
| Episodic bag simulation | 400 bags per fold per seed for evaluation, plus training episodes | 1-2 min |
| Config grid (about 24 configs) with nested check | evaluate on simulated bags | 5-10 min |
| Discriminative stage (per fold and final) | fixed epochs, fixed episodes per epoch, small low-rank metric | 3-6 min |
| Decode on simulated bags (agglomeration, n at most 50, d at most 64) | per bag milliseconds | under 1 min |
| Final fit on 19 batches, predict 110 bags, validate, write | closed form + fixed-step fine-tune | under 2 min |
| **Total** | | **about 15-30 min** |

Headroom: more than 60% of the 90 min limit. Memory: at most about 6 GB (feature matrix of at most 100k x about 1,500 floats plus bag simulations). Fallback member adds about 5-8 min if kept.

Inside the script: schema assertion against `sample_submission.csv`, finite-value checks on all features and scores, one label per snippet per bag, K in {2,3,4} per bag, bag count 110, numeric ordering by the `::` index, re-read of the written file with `keep_default_na=False`. No constant-output fallback is written (a constant partition scores 0 and would hide a failure); any failure raises loudly.

## Metric-aware training & decode

1. **Back-solve the metric.** ARI is a function of pair agreement counts, so the learning target is the pairwise same-batch relation. A pairwise or bag-level loss matches it better than snippet classification. The clip at 0 means rare catastrophic bags cost nothing extra, so evaluate decode choices on clipped ARI.
2. **Hierarchical weighting.** Replicate the metric's averaging: the training and validation objective averages over BAGS first (each bag equal weight regardless of n), then across bags. Pair-level loss is reweighted by 1/(pairs in bag) so a 50-snippet bag does not dominate.
3. **Pair scoring.** The PLDA log-likelihood ratio gives the base score. A per-pair affine calibration (temperature, bias) is fitted on cross-fitted OOF scores with proper log-loss. Bias is initialised from the base rate of same-batch pairs in simulated bags.
4. **Decode, in-rules and per bag.**
   - Greedy Bayesian agglomeration using the group marginal likelihood (closed form after diagonalisation), recording partitions with K = 4, 3, 2.
   - Choose K by total marginal log-likelihood (scaled by the calibrated temperature) plus a single K penalty fitted on OOF simulated bags, cross-fitted. Allowed K are exactly {2,3,4}.
   - A few fixed reassignment sweeps (move a snippet to the cluster that most improves the objective) with deterministic tie-breaking by index.
5. **Test the alternatives before adding machinery.** Compare, under the same paired folds: always-K=3, argmax-marginal K, and calibrated-penalty K. Compare agglomerative against spectral clustering on the calibrated affinity. Keep the simplest that wins by more than noise.
6. **Expected-ARI decode** (choose the partition maximising expected ARI under posterior samples) enters the roadmap only if the lean decode is measured and a gap remains.
7. **Hard constraints.** The decoder never emits K=1 or K=n. Because it is a decode over the model's own evidence, it is not a pure constraint search.

## Structural signals

- **Partition structure.** Each snippet belongs to exactly one latent batch, and transitivity of "same batch" is enforced by the partition decode rather than by independent pair thresholds.
- **Permutation invariance.** The result must not depend on snippet order inside a bag. Shuffle bags during training and check that decode output is permutation-equivariant. Label-switching invariance is free with ARI.
- **Symmetry.** Pair features are symmetric by construction (absolute difference, product), and the pair score is symmetric.
- **Within-batch variability is content, not noise.** Different snippets in a batch carry different text but the same OCR process. Hence content-agnostic features, crop augmentation and length-matched negatives.
- **Bag size and K variation.** Simulate sizes 16-50 and K in {2,3,4}, with varied mix balance, so the model and decode constants see the full range.
- **Hard negatives.** Different batches with the nearest noise profiles are the pairs that drive ARI errors. Mine them at batch level from train centroids and oversample them.
- **Batch families (diagnostic 10).** If batches cluster into families, the cohort-bank member exploits it (leave-own-batches-out banks, as in the leave-own-group-out rule for label-derived statistics).
- **Within-bag relative context.** Rank-of-distance among the row's own bag-mates is a standard relative transform. This is flagged once (see open questions) because the target is a relation between bag-mates.
- **No use of row order or the `::` index** as evidence, only as the output ordering key.

## Experiment roadmap

1. **Contract, metric and validation.** Parse ids, implement ARI-based scoring and the output formatter, unit-test on perfect, permuted, constant, singleton and random inputs. Build leave-batches-out folds and the episodic bag simulator. Stop when the unit tests pass and the harness reproduces the reference-baseline ordering (word TF-IDF, char TF-IDF, length-only, closed-set) approximately. Run the data diagnostics 1-10 here.
2. **Cheapest end-to-end baseline.** Content-agnostic features, scaler, PCA, PLDA init with no fine-tune, agglomerative decode with K fixed at 3. Write a valid submission file. Record CV mean and std. Credit use: one.
3. **Representation and structure.** Add feature tiers one at a time (class n-grams, symbols, token shapes, optional rare-bigram tier). Add within-bag centring, crop augmentation and length-matched negatives. Run the length-matched evaluation. Keep a tier only when it wins by more than noise on paired folds.
4. **Metric-aware loss and decode.** Add discriminative fine-tune, calibration, K selection with penalty, and reassignment sweeps. Compare decode variants as in section 5 above. Stop when remaining gains are inside noise.
5. **Diversity.** Fallback LightGBM pair classifier alone first. Then cohort-bank member alone. Blend only members close in quality, preferring a feed-forward of OOF pair scores over fixed-weight averaging. Optional neural set encoder or character CNN last, only if the lean design is measured and a gap remains.
6. **Bounded in-script HPO.** The fixed small grid (at most about 24 configs) with the nested check. No offline-pasted constants.
7. **Final fixed-plan run.** Run twice from a clean working directory and diff the outputs. Run the CSV validator. Submit. Use credits only for baseline, best single, ensemble and final.

Where two or three independent solvers would plausibly converge (within-bag centring, K selection, length-matched negatives), test those before any proxy-driven tweak.

**Expected score (estimate, not a promise).** My expected private range is about 0.30-0.44, with the most likely value near 0.35-0.40, because the reference is 0.342, the AI baseline 0.40 and the leaderboard high 0.44, and the weak baselines show the noise signal is modest. This is reasoning from the challenge's reference numbers only. It is unverified until the leave-batches-out harness runs.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits, applied to the plan.

- Test file read for anything other than one-bag-at-a-time prediction: no. No test statistics, vocabularies, scalers, PCA, clustering or priors. Each bag is a function of its own rows plus train-fit parameters. Within-bag centring or relative features use the bag's own rows only, which the challenge permits.
- Wall-clock in any condition: no. Time is logging only.
- `cuda.is_available()`, `os.cpu_count()`, import fallbacks or try-except switching models: none. CPU only, fixed thread constants.
- Hard-coded tuned constants: none pasted offline. The grid, the temperature, the K penalty and the fine-tune strength are all found in-script by train-only search (cross-fitted). K range {2,3,4} comes from the description, not tuning.
- External data, synthetic data, pretrained weights, non-allowed libraries: none. Simulated bags recombine real labelled train snippets (flagged as a question). Crop augmentation transforms real snippets.
- Strip-the-ML test: passes, the learned metric is load-bearing (see Compliance regime).
- Model-heavy part dominates: yes, the decode is generic and the features carry no batch-specific rules. Hand-built features feed a trained scorer.
- Sibling-leakage audit: within-bag relative features cross-reference group-mates, and the target is a relation between mates. This is inherent to a bag-clustering task and uses only the bag's own rows, but it is listed as a reviewer question.
- No whole-test aggregation: confirmed. No rank or z-score across the test file, no test priors.
- Length and language: no length features, length-matched negatives and evaluation, letter-level features tiered and ablated.
- Source readable, under 512 KB, no encoded blobs: yes.
- Deterministic: fixed seeds, fixed thread counts, fixed episodes and epochs, deterministic tie-breaking in agglomeration, double run and diff before submission.
- Output ordering by the numeric `::` index: unit-tested.

## Open questions & assumptions

**Reviewer questions.**
1. Is per-bag standardisation, centring or within-bag relative features acceptable under "only that bag's rows may be used for its partition"? Default plan: yes. If the reviewer says no, drop centring and relative features and use the global train-fit scaler only. Measured cost of that reading is a CV ablation.
2. Are letter-level OCR confusion statistics (rare bigram rates) "language identity"? Default: tiered, off unless it wins under leave-batches-out CV and is cleared. The compliant-under-every-reading choice is class-level and symbol-bucket features only.
3. Is a train-fit out-of-vocabulary rate against a train-built word list acceptable, given its confound with language? Default: excluded.
4. May snippet length be used as a measurement-precision weight (not as an identity feature)? Default: not used at all. The measured cost comes from the length-matched evaluation versus the shrinkage-only variant.
5. Is a closed-form PLDA plus a fixed-step discriminative fine-tune accepted as genuine training? Default: yes, with logged parameter movement and CV gain over the init.
6. Is episodic bag simulation from real labelled train snippets acceptable (recombining real labelled units, not fabricated entities)? Default: yes.
7. Is a cohort bank of train-batch models (train labels, leave-own-batches-out, fixed bank size) acceptable? Roadmap member only.

**Assumptions.**
- The number of batches per bag K is uniform on {2,3,4}, mix proportions range from balanced to skewed, and bag size is uniform on [16,50]. The true generator is unknown. Validation reports by K and balance and tests sensitivity.
- The integer after `::` is used only as the output ordering key and may or may not correlate with batch in test. It is never used as evidence.
- Train N is unknown. Runtime estimates assume at most about 100k snippets.
- A train batch is one homogeneous noise population (diagnostic 7 checks this).
- Snippet text in train and test is non-empty. Empty or very short snippets get shrinkage-dominated features and are handled without crashing.

**Could not verify (no dataset).** Every number about the data, feature informativeness, ICC, pair AUC, expected CV level, runtime and memory, and whether the reference-baseline ordering reproduces in simulation. These are hypotheses until the diagnostics in Data findings are run.
