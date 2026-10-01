# Metric-aware training and decoding (the strongest recurring lever)

Treat "model outputs → submitted answer" as its own optimised stage. Fit every constant of that stage on
out-of-fold predictions, with its own nested check (see `validation-recipes.md` V3). All of this is *training-time* or
*per-row* decoding and is compliant unless the challenge text bans metric-aware decoding (then see §9).

## 1. Re-implement the metric first, then attack it
- Copy the formula exactly (tie rules, empty cases, clipping, averaging order). Unit-test hand-computed extremes:
  perfect → best, reversed/all-wrong → worst, constant/prior → documented baseline, empty/abstain case.
- **Oracle decode check**: feed the *gold* local scores through your decoder; it must reproduce ~100% of train
  targets. If not, the decoder (not the model) is the bottleneck, or your metric implementation is wrong.
- Print the metric's best constant / prior baseline under your CV: it is the floor every model must beat.

## 2. Loss design from the metric
| Metric shape | Training move |
|---|---|
| Invertible/saturating transform of a latent (pair weights `min(|Δ|/c,1)`) | Back-solve the latent scale per group, regress it (A1); use an asymmetric hinge for censored rows where the target is only a bound (A6) |
| Per-example weights | Pass them as sample weights (A2); also divide by each nesting level's denominator so macro-by-group metrics are replicated exactly (A7) |
| Pairwise ordering / Kendall / ranking accuracy | Weighted RankNet loss `w·softplus(-y·(s_i−s_j))` (A5), antisymmetric accumulation `score[i]+=p_ij; score[j]-=p_ij` at decode (D3) |
| Ordinal small-K | K−1 cumulative binary heads ("at least k"), decode by expected gain `Σ P(y≥k)` or utility matrix (A8) |
| Brier / Brier skill | Add squared-error to the loss, recalibrate per output slot on OOF (A17); small isotonic/Platt per slot, cross-fitted |
| F1 / exact-span F | Score (start,end) pairs (GlobalPointer/biaffine) when exact boundaries dominate (A13); closed-form emit threshold (A12) |
| IoU / Dice / Jaccard | Expected-IoU-optimal set per instance (A16, §4 below); expected-Dice-optimal family under a latent posterior |
| NDCG / MAP / MRR | Listwise softmax over candidates + hard negatives refreshed from the model's own scores (G14); LambdaRank head on a candidate pool; rank-based decode |
| Count/proportion aggregates (Bray-Curtis, KL) | Treat as learning from label proportions (LLP); mean-pooled linear is the right baseline; Beta-Binomial MC decode (A9) |
| Composite (match + packet + abstention) | Expected utility over *every* valid joint output under the model posterior (A3) |
| Spans/runs credited contiguously | Add same-label persistence potential to the Viterbi decode and test the "off" setting as a real candidate (A11) |

## 3. Expected-utility (Bayes-risk / MBR) decode
`y* = argmax_{y∈Valid} Σ_s p(s)·U(y, s)`. Enumerate the valid set when small (≤ 10^5), else sample candidates from
beam/top-k then re-rank by MBR with the exact metric as the utility. Sources of `p(s)`: calibrated softmax,
temperature fitted on OOF, shrinkage toward a train-only prior. Include "abstain/empty" as a candidate when the metric
scores emptiness (empty-vs-empty = 1). Bound any fitted decode grid by the known label prior so it cannot drift into a
degenerate region (G6).

## 4. Recipes worth memorising
- **Expected-IoU mask**: sort pixels by `p`, keep top-k maximising `Σ_{i≤k}p_i / (k + Σp − Σ_{i≤k}p_i)`.
- **Closed-form per-label threshold** for count-based F-scores: differentiate the metric w.r.t. "emit one more
  span"; typical form τ = L / (a + bρ); iterate to a fixed point on OOF (lower variance than grid search).
- **Exact latent-order marginal** (A18): ≤ 8 items → enumerate all orders, score each with learned pairwise
  potentials (`cost = pairwise.flatten(1) @ INC.T` with a constant 0/1 incidence matrix: deterministic and fast),
  softmax over orders, marginalise to the queried position (midpoint, next, first).
- **Exact assignment likelihood**: for ≤ 8! matchings or ≤ 10^4 joint assignments, normalise exactly, train by
  NLL, decode by posterior-mean marginals (valid by construction) or Hungarian/Sinkhorn on log-potentials.
- **Decoy-slate / candidate-set structure** (A14, C16, D12): recover role/taxonomy structure from train answers,
  use relabelling-invariant membership-cell descriptors, enforce hard constraints (one answer per role, 10 distinct
  letters) via assignment decode. Audit near-perfect scores on decoy tasks as generator tells (compliance, Q5).
- **Beta-Binomial posterior** (A9) when group members are correlated; per-class concentration by grid MLE on OOF.
- **Discriminative credit head** (A10): small softmax scorer over decode-candidate descriptors, trained to maximise
  realised OOF metric; use only when the plain expected-utility decode has plateaued.
- **Two-pass self-conditioned decode** (A15): write the pass-1 answer into the auxiliary slots, re-score, average.
- **Cross-task coupling** (A4): when output Y is (nearly) determined by output X under the task's design, train a
  small classifier for the easy direction `(context, option)→X` and force the decode of Y from the confident X.
- **Gate for emptiness** (C7): separate "is anything there?" head when empty answers are common and separately
  rewarded; route only gate-positive rows to the localiser.
- **Per-slot OOF calibration**: one scalar or isotonic map per structurally different output slot (e.g. per
  position, per class, per protocol), fitted cross-fitted; ship only if the cross-fitted gain is positive.
- **Reconstruct more signal than the labels show** (A1): hidden continuous quantities behind bounded transforms,
  deterministic relations between columns, summary labels decomposable into per-item targets.

## 5. Weighting and balancing
Macro-over-groups metrics: weight rows by 1/(group count) and by metric weights at every nesting level. Macro-over-
labels: class-balanced loss, and check per-class F1 (a head can collapse to the majority class: 75% accuracy, 0 F1).
Rare classes with few positive groups: balance folds on the rare labels (V6).

## 6. Calibrate decode parameters honestly
Fit decode constants on OOF of models trained on a reduced split (or K-fold), then transfer the constants to models
refit on 100% (B9). Report the cross-fitted number (fit on half A, score on half B, swap) beside the in-sample
one. Count the constants: more than ~6 free decode parameters on a few hundred OOF rows is overfitting.

## 7. Rank / score normalisation without test statistics
Normalise against the **train** reference distribution (fitted once), never across the test batch: map each test
score through the train empirical CDF, or use fixed z-score constants from OOF. Rank-averaging *over the test set*
is whole-test aggregation and is banned (compliance Q7).

## 8. Sequence/segmentation specifics
Auxiliary geometric channels (distance transform, offset-to-centroid) decoded by watershed for touching instances
(J1); un-transform directional channels under TTA (J2); soft-marginalise over near-tied candidates before the hard
decision (J4); transport held-out data into the target scale/sensor before tuning the decoder (B10).

## 9. When the rulebook bans metric-aware decoding
Move the same effect into a published training-time technique: class-balanced or logit-adjusted cross-entropy,
focal/asymmetric loss, per-slot loss weights, prior correction by fixed train-derived logit offsets. Ship plain argmax.
State in comments that decoding is plain argmax and the rebalancing is in the loss. Check the description wording
before using any tuned decode (an editorial task banned it while a span task rewarded it).
