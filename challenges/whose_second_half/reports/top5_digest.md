# Whose Second Half Is It: digest of the five ranked private-LB solutions

Source: five `solution.py` files supplied by the user (rank 1 to 5 on the private LB). **Scores were not supplied, only ranks.**
Everything below is read from the code, own words, nothing re-run. Gains are therefore *not measured by us*.
Baseline for comparison: our own `solution.py` (public 0.54 per the user; local grouped CV not yet measured, learned-kernel OOF alone 0.429).

## 0. One-paragraph read

All five solve the same problem the same way: **learn artist-to-artist affinity from the ~23k full days, refit that learning without the held-out fold's prefix sessions, turn it into per-(prefix, candidate) pair features, then train a small row-aware head on only the 1,081 labelled rows, and decode each row jointly.** Nobody trained on extra constructed rows. The ranks differ mostly in (a) how many independent affinity views feed the head, (b) how well the head sees the competing pairs of the row, (c) how many members are averaged. Rank 1 is the only one that works on a *token-level* interaction tensor instead of pooled scalars.

## 1. Per-solution digest

### Rank 1 (token-interaction head, 8 embedding views, feature-bagged ensemble)
- **Affinity views (all refit per fold, 6 pools x 8 views = 48 skip-gram models, 10 processes, single-thread each):** six artist views with different "sentences" (distinct artists in first-occurrence order, windows 50; the day cut into *sittings* at 600 / 1800 / 7200 s gaps; plain play order window 5 with consecutive repeats collapsed; sparse vs dense subsampling) plus release-level and recording-level views (min_count 2). dim 256, 20-30 epochs, negatives 5-10.
- **Both vector tables kept:** for each view in.in, in.out and out.in cosines (skip-gram's own co-play logit).
- **First-order count matrices (7):** full-day co-occurrence cosine and lift; **first-half artist -> artist that is NEW in the second half** (cosine, lift, and a rank-weighted 1/(1+rank) version), i.e. exactly the target the task defines; transition cosine at distance 1 and <=3; PPMI-normalised neighbour overlap.
- **Head input:** for every prefix, its up to 96 most recently played distinct artists (token), against each of the 12 continuation artists, with ~34 channels per (token, candidate artist) + 5 token meta features (log count, order, recency, popularity, has-embedding). Plus 20 group features per (prefix, candidate artist) (prefix-centroid similarity per view, popularity, release overlap).
- **Head:** token MLP -> value + attention logits; attention / mean / max pool over the prefix tokens (10 % token dropout); pair MLP; the two continuation artists are summed; then one context layer mixing the pair with its row mean, column mean and bundle mean. Loss = row cross-entropy + column cross-entropy (not the 720-permutation likelihood).
- **Ensemble:** 10 members for epoch selection, 20 for the final. Each member sees a random 70 % of channels and 80 % of group features (feature bagging). Per-member logits z-scored per row and summed, Hungarian at the end. Epoch count (50 or 60) picked on held-out fold 0.
- **Test features** come from the pool of all non-test sessions (not fold pools).

### Rank 2 (heavy feature bank + LightGBM + set network, two selected-in-run scalars)
- **Corpus = complete non-test sessions; all of it refit per fold.** Test rows are scored under *each* fold corpus and the stacker outputs averaged, so test features come from exactly the regime the stackers were trained on.
- **Learned components:** multinomial denoising autoencoder over artists (first half -> second half and second half -> first half, softmax over vocabulary, 50 % input dropout, 20 epochs); shifted-PPMI SVD embedding (192d, window 10, 1/distance weights, context smoothing .75, recency-weighted prefix vector); idf cosine session-kNN with several popularity exponents, top-50 neighbour counts, recency-weighted prefixes; **windowed session-kNN** (corpus days cut into overlapping 25- / 50-play windows, artist + release + recording level, one round of query expansion from the 300 best windows plus embedding similarity).
- **Listener-fingerprint profile features:** share of never-seen / rare recordings, missing-release share, release overlap, recording "hit" level vs artist level; these enter the stacker **double-centred inside the bundle**.
- **Stackers:** LightGBM (binary pair rows, 3 seeds, 500 rounds, + within-prefix and within-candidate z-scores and ranks) and a permutation-equivariant bundle network (20 seeds, row + column softmax loss). Network logits added to LightGBM log-odds with weight chosen from {0.5..2}; decode temperature chosen from {0.25..2}; decode = exact 720-permutation marginals, then Hungarian on the marginals.
- Reports 49 min of 90 on a laptop, with headroom statement in the docstring.

### Rank 3 (exact permutation likelihood in both GBM and row network)
- **Views per pool:** item2vec x4 (artist window 20 two seeds, window 5, release window 20), PPMI-SVD (day-level), LSA (SVD of idf day x artist), item-item co-occurrence (cosine, conditional, PPMI; sum and max), session-kNN votes at top-100/200 against *all* artists and against *new-in-second-half* artists, release-level kNN, IN.OUT soft-OR affinities, release overlap, binomial log-likelihood of the empty-release habit.
- **LightGBM with a custom objective = log-likelihood of the true assignment among the 720** (gradient = exact pair marginal minus label, bounded hessian). 36 pair rows per bundle stay together.
- **RowNet:** attention over the prefix artists using 11 per-artist inputs (log count, rarity, recency in plays and in seconds, was it played in the last sitting, **how many other prefixes of the same row share the artist**, its similarity to the other prefixes' centroids); every embedding view pooled and scored against the two candidate artists, plus a candidate-conditional soft-max attention and a linear term over all pair scalars; trained on the exact 720 permutation likelihood.
- **In-script selection:** 5-fold refit of both, blend weight in {0..0.5} and temperature in {1..2.5} picked on OOF accuracy. Final decode = argmax of exact marginals.
- Test features from the full non-test pool.

### Rank 4 (smallest feature set, best self-diagnostics)
- **Hand-written skip-gram in torch** (3 views: wide window 10, near window 2, day-level window 0 = random same-day pairs; 3 seeds each; dim 128), PPMI-SVD, popularity-weighted session co-listening, decayed transition-in counts, top-3 and recency-pooled similarity, release sharing.
- **"How, not what" style features** of the prefix and candidate: repetition, run length, album share, median gap, quick-gap share, span, mean track popularity, candidate track / release / artist-track popularity and its distance to the prefix's level. In-run ablation: listening-style evidence on top of taste evidence.
- **Set matcher** (permutation-equivariant, 0-2 layers) over the 6x6 grid with raw + double-centred features, trained on the **exact 720-permutation NLL or** row+col CE (chosen by Optuna), **24 seeded TPE trials**, 5 seeds.
- **Decoder chosen in-run**: per-prefix argmax of marginals ("expected", allows repeats) vs Hungarian, picked on the HPO folds.
- **Test rows scored under each of the five fold spaces, marginals averaged.**
- Prints: per-fold mean/std/min, folds never seen by HPO, raw-similarity-only baseline, single-seed spread, tercile slices (prefix size, true-continuation popularity), a train-probe overfit gap with a verdict. This is the best template for our report.

### Rank 5 (wide feature bank, supervised column selection, CatBoost + linear permutation models)
- **Five families (3-fold refit):** count affinities with several length normalisations + transitions + kNN votes + release/recording entity affinities + 128d SVD; PPMI-SVD x6 variants; CBOW + skip-gram word2vec; "all-graph" SVD over the top 20k artists. ~650 columns per family, **selected by supervised effect size after double-centring, with correlation de-duplication**, and augmented with row-max and column-max normalised copies.
- **Models:** CatBoost (depths 4 and 6, binary logloss), a CatBoost on a separate structured subset, and **two linear models trained on the exact 720-permutation likelihood** on log-signed, double-centred inputs (3 seeds, snapshot averaging at epochs 8/12/18). Weights .25 / .375 / .375, temperature .5, Hungarian on marginals.
- **Flags to NOT copy:** (i) the embedding corpus appends *all* rows of `continuations.csv` as 2-artist sentences, which includes the candidates of test rows (transductive over test candidates; the rules only ban fitting on test *first halves*, but a reviewer may read it as test use); (ii) column selection uses labels of all OOF rows before the final fit (selection leak, no in-script validation reported); (iii) train rows are sorted by `id` before folding (id order as a processing key); (iv) 3-fold feature refit vs full pool for test creates a small corpus-size mismatch.

## 2. What all five share (the transferable core)

| # | Pattern | Who | Our baseline |
|---|---|---|---|
| 1 | Every statistic / embedding refit **without the held-out fold's prefix sessions** (and without all test sessions) | 5/5 | partially (Item2Vec yes; LK / release graph use `fit_all`; 20 feature folds vs 5 model folds not aligned) |
| 2 | Learn **first-half -> new second-half artist** affinity from the full days | 1, 2, 3 explicitly; 4, 5 via whole-day co-listening | no: targets are all plays (`Ca`) |
| 3 | **Several embedding views**, different windows / sentence definitions / levels (artist, release, recording) | 1 (8), 2 (3+), 3 (6), 4 (4), 5 (7) | one Word2Vec window 20 |
| 4 | Keep **skip-gram IN and OUT** tables, score in.out | 1, 3 | no (cosine of normalised input vectors only) |
| 5 | Head sees **competing pairs** (row / column / bundle means, double-centring, z-scores and ranks within prefix and candidate) | 5/5 | rowwise features; MLP is per-pair and independent |
| 6 | **Exact 720-permutation likelihood** or row + column softmax as the loss | 1,2: row+col CE; 3,4,5: exact NLL | exact NLL (good) |
| 7 | Listener **fingerprint** features: empty-release rate, recording rarity, repetition, gaps | 2, 3, 4 | partial (traits) |
| 8 | **Large seed / member ensemble** with randomised members | 1 (20, feature-bagged), 2 (20), 3-5 (3-5) | 3 seeds |
| 9 | In-script selection of **blend weight / temperature / decoder** on OOF from a small grid | 2, 3, 4 | temperature only |
| 10 | Test features computed under **each fold corpus and averaged** (train and test same regime) | 2, 4 | no (full pool for test, 20-fold pools for train) |

## 3. What looks like the real separator between ranks (hypothesis, not measured)
1. **Resolution of the evidence:** rank 1 keeps a token x candidate tensor and learns the pooling; others hand-pool. This is the only structurally different idea in the five.
2. **Number of independent views x seeds**: rank 1 and rank 2 average 20 members; both also have the most views.
3. **Matching train and test feature regime** (rank 2, 4) vs mixed regimes (ours).
4. **Row context for the head** (rank 3's "how many other prefixes share this artist", rank 1/2/4's equivariant layers).

## 4. Ranked next steps for our solution (expected gain per hour, all still need grouped-CV confirmation)
1. Re-target kNN / Naive-Bayes / transition affinities to **new-in-second-half artists** (pattern 2). Cheapest, directly the task's definition.
2. Add **in.out skip-gram** features and 2-3 more Word2Vec views (window 5, sitting-cut, release level) (patterns 3, 4).
3. Give the head **row context**: equivariant layer or at least double-centring of every feature and the shared-artist count (patterns 5, 7).
4. Align feature folds with model folds and **score test under each fold corpus then average** (pattern 10); add grouped CV by shared prefix sessions.
5. More members (feature-bagged, 10-20) and in-script selection of blend weight / temperature / decoder (patterns 8, 9).
6. Only then consider a token-level head (rank 1 style): most expensive, highest ceiling.

## 5. Compliance reading (CLAUDE.md 2.3, 3)
- Rank 1, 2, 3, 4: all fitted statistics exclude test sessions; test rows read per row. Within-row joint decoding only. Look clean by our standard. Rank 2 calls `use_deterministic_algorithms(True)` without `warn_only` (runtime-error risk). Rank 1 uses forked multiprocessing pools (fixed seeds, single-thread workers; keep determinism check).
- Rank 5: see flags in its section; do not copy items (i)-(iii).
- None of the five constructs extra labelled rows from full days. That idea stays a reviewer-risk question (could read as "solver-generated synthetic training data"); the top-5 evidence says it is not needed.
