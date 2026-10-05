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
| **final** | solution.py: 3 kNN variants + transitions + traits -> LightGBM(3 seeds) -> permutation-posterior decode | **0.4365 +- 0.0167** (folds .432 .408 .458 .440 .445); argmax .399, Hungarian .433 | runtime ~2.5 min on 4 cores |

## Error analysis

## Honest ceiling statement
The target of 0.8 (AI baseline 0.7, per the owner) was NOT reached: best honest OOF is 0.44. Every representation tried (kNN, item transitions, ridge, SVD, neural, trait features) plateaus at 0.40-0.45, and accuracy grows steeply with the amount of fit data (the one lever that would help, the test first halves, is banned by the rules). If 0.7+ is attainable there is a signal I have not found; candidate directions not yet tried: more trait families, row-synthesis from fit sessions to enlarge the stacker training set (grey: counts as solver-built training data), many more kNN variants.

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
