# Challenge notes: second_half

## Status
- [x] contract  - [~] data audit (first pass: reports/audit1.py, exp1.py, exp2.py)  - [ ] strategist plan  - [ ] validation  - [ ] baseline  - [ ] experiments
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

## Error analysis

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
