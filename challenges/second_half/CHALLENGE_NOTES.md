# Challenge notes: second_half

## Status
- [x] contract  - [x] data audit  - [x] plan  - [x] validation (5-fold by row, held-out sessions removed from fit)  - [x] baseline/final v1  - [~] experiments (plateau at 0.44)
- [ ] review (compliance, runtime, red-team)  - [ ] presubmit  - [ ] submitted  - [ ] closed (lessons written)

## Contract (decision unit, valid answer, metric terms, constraints, bans, compute, runtime)
- Unit: one row = 6 prefixes x 6 shuffled candidate continuations; output 6 ints (a permutation is the valid-by-construction answer). Metric (A-1/6)/(5/6); chance 0.
- CPU only (10 cores/62GB), 90 min limit -> hardcode device cpu, thread caps; fixed work plan, no clock branches.
- Fit ONLY on: train.csv, continuations.csv (train rows), plays of sessions NOT in test.csv. Never fit on test first halves, other test rows, or global stats over test candidates/continuations.
- Biggest risk: leakage in own CV (fitting affinity on the held-out rows' second halves) and an accidental global statistic over test rows. Both avoided by excluding held-out sessions from every fit.

## Data audit (shapes, groups, duplicates, label distribution, structure, ceiling diagnostics)
- listens: 1,150,550 plays, 23,990 sessions, 148,729 artists, 3.9% empty release.
- Sessions: 6,486 in train rows (full day, median prefix 24 plays), 3,582 in test rows (first half only, median 25), 13,922 in NO row (full days, short: median 11 plays, <16 so ineligible for rows) = extra fit material. Total full sessions usable = 20,408.
- Each session appears in at most one row; train/test sessions and train/test continuations are disjoint. 10,068 continuations, all with exactly 2 artists.
- Continuation rebuild from the second half reproduces 96% on train (misses = 3-session threshold counted over test first halves too) -> definition confirmed.
- Continuation artists: 99.8% occur in some non-test first half, median 13-14 sessions, ~8% have fewer than 3 -> the item side is mostly well covered, the long tail is thin.
- Release overlap: true continuation's first-play release occurs in own prefix for 1.9% of pairs vs 0.14% for wrong in-row pairs (~13x lift, tiny coverage); recording overlap 0% -> release is a small auxiliary feature, recording is useless.
- Candidate popularity does not differ between true and in-row mean (rows are matched on popularity) -> popularity is not a signal, only a nuisance column effect.
- Raw kNN session-similarity baseline (5-fold by row, held-out sessions removed from fit): argmax 0.16-0.20, with Hungarian 0.30-0.36 (target = whole day of neighbours beat second half only; gamma 1 beat 3). Joint assignment alone is worth about +0.15, so column/row normalisation and calibrated scoring are the main levers.

## Validation design (mirror of the hidden split, groups, bias direction of the proxy)

## Plan (primary, fallback, rejected options, fixed work plan)
Primary (not yet built): several affinity scores (user-kNN with whole-day target, item-item PMI/embedding, last-N-plays recency weighting, release-level) -> LightGBM pair scorer trained on train-row pairs with nested session exclusion -> row-wise log-softmax/Sinkhorn -> Hungarian. Next: per-row normalisation experiment, then pair scorer.
Rejected: popularity, ids/order, anything using test first halves for fitting.

## Experiment log (id, hypothesis, change, CV mean +- std, per-fold, runtime, kept?, notes)
All scores: chance-corrected metric, 5-fold by row, held-out rows' sessions removed from every fit, Hungarian decode unless noted.
| id | change | CV | note |
|---|---|---|---|
| e1 | user-kNN cosine (idf^.5), target whole day | 0.36 (argmax 0.20) | joint assignment worth ~+0.15 |
| e3 | log(S+eps) before Hungarian | 0.41 | nonlinear transform matters, row/col shifts do not |
| e4 | directed transitions W<=10, recency weights | 0.28 | recency HURTS; signal is whole-prefix taste, not last plays |
| e5 | kernel ridge / EASE-style first->second half | 0.30 (2 folds) | worse than kNN, slow; too little data for decorrelation |
| e6 | exact recording-sequence (radio/playlist) linkage | 8% hit | no deterministic leak |
| e7 | background-standardised scores (colz) | 0.43 (log 0.43) | no gain over plain log |
| e8 | + release / recording tokens in kNN | art 0.398, art+rel 0.407 | tiny gain |
| e10 | LightGBM pair scorer on 35 CF features | 0.43 (argmax 0.38) | calibrated but same Hungarian level |
| e12 | listener traits: recording obscurity r=.53 true vs .05 wrong, empty-release r=.34 | traits alone 0.15; + CF 0.45 | real but overlaps with CF |
| e15 | neural bag->second-half net (DAE), fold 0 | 0.13 / 0.24 with augmentation vs kNN 0.42 | too little data |
| e16 | kNN hyper grid: tf log +0.04, gamma .5 +0.03, top-k hurts, fit size 25%/50%/100% = 0.24/0.31/0.35 | best single 0.42 | accuracy scales steeply with fit-set size |
| e17 | SVD artist embeddings k=64/128/256 | 0.24/0.26/0.28 | not better than kNN |
| e18 | session-level naive-Bayes PMI item-item (alpha 20, mean) alone | 0.4165 | comparable to kNN, different flavour |
| e19 | prefix scalars (length, span, gaps) x candidate popularity into the stack | 0.451 vs 0.451 | no gain |
| v2 | + NB-PMI (alpha 20 and 3) features in the stack | 0.4670 +- 0.024 | +0.03, positive in all 5 folds: diversity of base scorers pays |
| v2b | + release / recording-level NB | 0.4628 | no gain, removed |
| v3 | + row-exclusive prefix bags (artists unique to that person within the row), NB and kNN | 0.4717 | small |
| v3b | + candidate-pair AND co-occurrence kNN | 0.4715 | no gain, kept (harmless) |
| v4 | + learned histogram kernel (OOF 0.429 alone) and release-graph links as stack features | LightGBM stack 0.481 | +0.01 |
| v5 | listwise permutation-NLL MLP stacker (3 seeds, hid 32, 250 steps) replaces LightGBM | **0.5147 +- 0.0133** | +0.035, every fold; row/col-equivariant variant gave no gain (0.514 at 150 steps, overfits beyond) |
| v6 | joint end-to-end training: histogram-kernel heads inside the network, trained together with the stacker on the permutation likelihood (exp28) | 1 head 0.495, 4 heads 0.500, 4 heads/150 steps 0.504, 8 heads/l2 .1 0.508 | NOT better than separate training (0.515): rejected |
| (earlier final v3b) | solution.py: 3 kNN variants + NB-PMI x3 + transitions + traits + exclusive-bag + pair-AND -> LightGBM(3 seeds) -> permutation-posterior decode | **0.4715 +- 0.0140** (folds .453 .456 .477 .485 .486); argmax .445, Hungarian .468 | runtime ~2.8 min on 4 cores; two runs byte-identical |

## Error analysis

## Honest ceiling statement
LEADERBOARD CONTEXT (screenshot 2026-10-05): AI baseline 0.6181, current top public score 0.5098 (1 solver), 6/6 credits. The owner's 0.7/0.8 figures were wrong. Best honest OOF here is 0.47, so below the AI baseline (which is the bar for the prize pool) and just under the current top public score. Every representation tried (kNN, item transitions, ridge, SVD, neural, trait features) plateaus at 0.40-0.45, and accuracy grows steeply with the amount of fit data (the one lever that would help, the test first halves, is banned by the rules). If 0.7+ is attainable there is a signal I have not found; candidate directions not yet tried: more trait families, row-synthesis from fit sessions to enlarge the stacker training set (grey: counts as solver-built training data), many more kNN variants.

## Honest ceiling statement (update 2)
After the listwise permutation-likelihood MLP stacker (+0.035 over LightGBM on the same features), the final local OOF is 0.5147 +- 0.0133 and the platform's pre-submission CSV check scored 0.5514 (public subset), above the top public score seen earlier (0.5098) but below the AI baseline 0.6181.

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)

## Platform pre-submission checks (screenshot, comment-free solution.py, 2026-10-05)
- CSV Score Validation: 0.5514450867052024 (public subset) vs local OOF 0.5147 +- 0.0133: gap +0.037, same direction as a lucky/easier public subset; local CV not contradicted.
- Prompt Compliance: "may need review", Medium confidence; the ONLY finding was prompt_runtime (solution.py:18 N_FOLDS = 5): 90-minute limit cannot be verified statically. Assessment otherwise: CPU-only, threads capped, fits only non-test sessions, genuine training, per-row decode. No rule violation flagged. Local runtime ~9 min on 4 cores.
- Held-out Answer Ingestion: passed (High). Deterministic Execution: passed (High).
- "All checks passed. Ready to submit." Submitting spends a credit; still below the AI baseline 0.6181 so not prize-pool eligible.

## v7 (penalised kNN + 20 feature folds), Kaggle CPU run (2026-10-05)
- Kaggle private CPU kernel (4 cores), comment-free solution.py: finished in 976 s; internal OOF Hungarian 0.5227, posterior 0.5210, per-fold mean 0.5208 +- 0.0221 (local run 0.5221 / 0.5208).
- Kaggle submission.csv vs local: 99.92% of entries identical (1 of 597 rows differs): last-digit CPU differences, valid format.
- upload/ now holds v7 (solution.py + Kaggle-run submission.csv). upload_v5_checked/ keeps the earlier version that the platform's pre-submission check scored 0.5514 public (checks passed).
- Dataset was uploaded to a private Kaggle dataset (sharanya1805/second-half-data) at the owner's request; platform data terms not checked.

## Round: next-play features + reference-style blend (2026-10-05)
- Structure found: rank-1 continuation artist is the very next play after the prefix in 31% of rows, within 3 plays in 58%; rank-1 = first new artist of the second half in 60%.
- Next-play transition features (artist/release/recording, last 1-5 plays, leave-row-out): discriminative (true pairs 29% nonzero vs 16% wrong for last-5 artist transition) but weak alone (0.15 Hungarian) and NO gain in the stack: MLP 0.5221 -> 0.5219 (marginal decode 0.5208 -> 0.5241, noise).
- Reference-solution blend (MLP majority + linear listwise + hist-GBDT, probability-space, weight grid with MLP >= others): linear 0.507, GBDT 0.491; best blend = MLP alone (0.5241). Rejected.
- Plateau: every lever since v5 moves CV by <= 0.01. Expected public for v7 ~0.555-0.56 (v5: CV 0.515 -> public 0.5514). AI baseline 0.6181 NOT reached.

## Decision (owner, 2026-10-05)
- Owner: "If the description bans it, don't do it." Pseudo-rows built from unused training days are dropped: the description bans solver-generated synthetic training data, and solver-assembled rows fall under the conservative reading.
- Final candidate: v7 in upload/ (comment-free solution.py + Kaggle-run submission.csv), CV 0.521, expected public ~0.555-0.56. Fallback: upload_v5_checked/ (public 0.5514, all platform checks passed).

## BREAKTHROUGH: item2vec artist embeddings (2026-10-05)
- Leaderboard update: 5 solvers above AI baseline (0.6231-0.6647), 5 more at 0.59-0.61 -> 0.6+ reachable; my "only banned data gets there" hypothesis was wrong.
- Diagnostics that ruled things out: stacker learning curve flat (25%/50%/75%/100% rows: .490/.510/.518/.522); errors uniform across candidate popularity / prefix coverage; adding held-out first halves (emulating the banned test-first-half fitting) only +0.006 for item-item cosine; drift matrix (first->new second half) 0.22; recency weighting +0.01 at most; item-item cosine alpha .5 = 0.461 alone.
- exp40: gensim skip-gram Word2Vec on fit-day artist sequences (consecutive repeats collapsed), prefix = mean of unit artist vectors, score = cosine with mean candidate vector: w5 d64 0.5317, w20 d128 0.5445, w50 d128 0.5314 ALONE (5-fold, 80% fit) vs whole v7 stack 0.522.
- Determinism plan for the script: workers=1 and a fixed hashfxn (Python's str hash is randomised per process).
