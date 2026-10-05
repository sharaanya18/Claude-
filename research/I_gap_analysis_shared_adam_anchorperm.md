# Gap analysis: my solutions versus the ten supplied top solutions (5 Shared Adam, 5 AnchorPerm)

Nothing below was tested. Every improvement is a hypothesis with a compliance tag and an expected effort. Problem statements are stored verbatim in
challenges/shared_adam_repair/CHALLENGE.md and challenges/anchorperm/CHALLENGE.md; digests are in G_ and H_ files.

## 1. Shared Adam Memory Repair

My solution (public 0.589): per-request linear-Gaussian posterior over the two first-moment buffers (prior = span of the 52 per-image gradients at the public weights),
LightGBM estimate of the common second moment v, 4096 pooled candidate orders on the posterior mean, top-128 re-scored under 8 posterior draws with random v noise,
grouped validation by capture sequence, holdout-fold selection of one noise level.

| # | Gap versus the top solutions | What they do | Hypothesis for my next version | Compliance | Effort |
|---|---|---|---|---|---|
| 1 | Search space | prefix-tree enumeration of all 40,320 orders on GPU (ranks 1, 3, 5); pool plus swap local search (2, 4) | add a swap local search around the best pool orders, or enumerate with a prefix tree; keep the fixed plan | fine | medium |
| 2 | Gradient basis | gradients at public weights AND at weights moved along a common-step trajectory (rank 5: steps 0/4/8/12; rank 3: perturbed point, 114 columns) | build the trajectory from a predicted common step with the right magnitude (about 0.002 per update), not a large guessed shift; then re-test a basis at several path points | fine | small |
| 3 | Common vs difference | model (A+B)/2 and (A-B)/2 separately with their own scales; rank 1 trains with an explicit difference loss | give the common and difference parts separate prior scales; add a differential loss if a learned estimator is used | fine | medium |
| 4 | Moment prior | learned per-parameter MLP for m/sqrt(v) as well as log v (all five) | my m prior is a zero-mean gradient span; a learned mean (and a learned v with pixel x unit factorisation, rank 3) is missing | fine (trained on train moments) | medium |
| 5 | Inversion | exact Gauss-Newton / LM / LBFGS on the real one-step equations, iterated; rank 1 also conditions on v sensitivity | I used one linearisation; iterate the exact forward model (my small test of this was worth only about +0.01 at low noise) | fine | small |
| 6 | Uncertainty | block-wise spreads measured on held-out capture sequences, used for draws or Laplace samples (ranks 1, 2, 5) | replace my hand-set draw noise with spreads measured on grouped out-of-fold residuals | fine | medium |
| 7 | Hypothesis diversity | 8 states per request (rank 1), 8 prior-strength fits (rank 3), seeds x folds ensembles (rank 2) | add a few prior-strength variants and require feasibility under each | fine | small |
| 8 | Decision rule | choose the rule inside the script on held-out sequences with the official metric (ranks 2, 4); tail-quality rule using the limit as a reference-draw count (rank 2) | compare mean-log, mean-quality and tail-quality on held-out sequences; estimate the "best of N random orders" equivalent from train | fine | medium |
| 9 | End-to-end training | rank 2 trains the estimator through the unrolled simulator with an outcome loss | optional; high effort, only if GPU time allows | fine | high |
| 10 | Validation noise | rank 2 and 4 select strategies on several held-out sequences | I validated on a small set per scene; use more requests and report per-scene scores | fine | small |

Ordering by expected gain per effort (untested guess): 2, 1, 6, 8, 3, 7, 5, 4, 10, 9.
Compliance note: ranks 2 and 4 contain a cuda/cpu switch or a wall-clock alarm; I would not copy those.

## 2. AnchorPerm

My v3 (public about 0.58 vs baseline 0.5616): row-local model, no use of other test rows.
My blocked v4: pooled-anchor adaptation (left out because no written approval exists).

| # | Gap versus the top solutions | What they do | Hypothesis | Compliance | Effort |
|---|---|---|---|---|---|
| 1 | Pooled regime adaptation | all five fit an anchor map/adapter on the pooled anchors of a discovered regime (see H_ digest) | the large lever, but it is test-set use | DO NOT ADOPT without written approval (CLAUDE.md 2.3A) | n/a |
| 2 | Second-order invariant evidence | within-role cosine geometry, item-in-row statistics, anchor profiles, QAP consistency feed a gradient-boosted pair model (rank 5) or a learned assignment net (rank 1) | add these row-local invariants as features; I tried QAP/Sinkhorn heads on top of a first-order model, but not a pair classifier over invariant item statistics | fine | medium |
| 3 | Row-local dequantisation | MLE of the quantiser scale per row, conditional-mean latents (rank 1) | cheap normalisation that makes the model scale-invariant | fine | small |
| 4 | Shift simulation breadth | rotation, partial coordinate permutation, finer requantisation, offsets, noise; a regime bank whose anchor-estimated maps come from other TRAIN rows | my augmentation used rotations and noise; add signed partial permutations and finer requantisation, and train a model that consumes an anchor-estimated map as an input | fine (train-only) | medium |
| 5 | Learned graduated assignment | Sinkhorn + QAP messages with deep supervision (rank 1) | a learned decoder instead of Hungarian on a fixed matrix | fine | high |
| 6 | Ensembling | 10-12 seeds, average Sinkhorn marginals, Hungarian on marginals | I used fewer members | fine | small |
| 7 | Confidence | isotonic on held-out-anchor or out-of-fold rows; leave-one-anchor-out recovery; bootstrap agreement; confidence slope per regime | use the leave-one-anchor-out recovery within a row as a feature; isotonic on out-of-fold rows from shift-simulated views | fine | small |
| 8 | Validation | simulated-regime views scored as separate strata, with the worst-stratum term (rank 5) | report familiar_sparse / familiar_rich / shifted_sparse / shifted_rich on simulated views | fine | small |

Honest reading: without pooling I expect the compliant ceiling to stay close to what I had; the invariant-feature levers (2, 3, 4, 6, 7) are the plausible small gains.
Do not hunt for loopholes to close the gap (CLAUDE.md 2.3A).

## 3. Cross-challenge lessons (rubric for future blind plans)
1. Check whether the hidden quantity is low-dimensional in the response space (Shared Adam) before modelling anything else.
2. Measure how much a candidate lever could add on train, using an oracle (min-norm state with exact responses gave 0.93 vs 0.34 for random orders).
3. Use the metric inside decoding (expected-quality over draws; expected-correct Hungarian).
4. Calibrate confidence on out-of-fold or simulated-shift rows, with stratum-aware validation.
5. When every leader breaks a rule, record it, do not adopt it, and ask for a written approval (L013).
