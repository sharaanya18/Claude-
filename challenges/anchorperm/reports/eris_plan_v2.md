# AnchorPerm: strategist plan v2 (transfer-row lift)

Scope: this plan looks for approaches not yet tried that could lift accuracy on transfer rows while staying compliant. Every number marked "measured" comes from this session on TRAIN rows only (split: `dev_common.py`, fit = 2,929 rows, va = 700 rows; subsets of 300 to 500 va rows). Every experiment script is in `reports/v2_scratch/`, and the shift simulators are in `reports/v2_scratch/sims.py`. I did not read or analyse test feature distributions. The statements about test rows below are the caller's diagnostics, quoted as given.

**Bottom line (estimate).** None of the row-local mechanisms I measured gives +0.05 transfer accuracy under any shift family that can be simulated from train. The best compliant stack is worth about +0.01 to +0.03 on strongly shifted rows, about 0 on mildly shifted rows, plus about 0.002 to 0.006 score from confidence. The +0.13 transfer gap to the top public score matches what a solver gets by pooling test anchors across rows (measured below). That technique is banned under CLAUDE.md 2.3.5 and the description's "row-local anchor conditioning" reading. It belongs as a reviewer question, not in the plan.

## Contract & decision unit
- Unit: one row (8 to 12 Q items, the same number of A items, 32-dim codes in {-2..2}, 2 to 5 anchors). The output is a permutation of the m = n - k hidden answers (train mean m = 6.83; chance accuracy 0.153 measured) plus one confidence value c.
- Row loss = 0.85 L_seq + 0.15 (c - R)^2. Score = 0.75 mean + 0.25 worst of {familiar, transfer} x {sparse k = 2-3, rich k = 4-5}. Lower is better.
- Sensitivity (estimate): if about 80% of test rows are transfer rows (caller's diagnostic), +0.01 transfer accuracy is worth about 0.85 x 0.01 x (0.75 x 0.8 + 0.25) = 0.007 score. Beating the AI baseline (0.5616) from v3 (about 0.58) needs about +0.025 transfer accuracy, or about +0.02 accuracy plus a confidence gain.
- Public LB noise: row-loss SD is about 0.26, so with roughly 450 public rows the SE is about 0.012. The v3-vs-baseline gap (0.018) is only about 1.5 SE. Do not read sub-0.015 public differences.
- Pipeline stages: (1) local evidence (pair logits); (2) decode (Hungarian, which is valid by construction); (3) confidence = E[R | row features]. Coverage is complete because the legal set is all permutations. The decode is not the bottleneck: Sinkhorn/expected-utility decode was already tried with zero gain.

## Compliance regime
- Classification: structured prediction from scratch (no pretrained weights are needed). CLAUDE.md 2.3.5 governs. The description explicitly allows row-local anchor conditioning, relational pair scoring, coherent one-to-one decoding, confidence calibration and training-only robustness augmentation. It prohibits id lookups and order channels.
- Clean (row-local): every mechanism in this plan computes a row's prediction from that row's own Q/A/anchors plus parameters fit on train. That includes per-row EM over the latent assignment and a row-specific map deviation. Wording for reviewers: say "row-local latent-variable inference", not "test-time adaptation". Nothing is shared or updated across test rows, and model weights are untouched at test time.
- Banned: pooling test anchors or test codes across rows to fit a transfer map, covariance, PCA basis or calibrator (2.3.5). Synthetic rows sampled from a fitted generative model (for example a PCCA generator with a new gain) are 2.3.6 synthetic data. Use them for offline diagnostics only, never for training in solution.py. Id/row-order channels are banned by the description (also measured useless: corr(q index, answer index) = -0.006; fixed-point rate 0.100 vs 0.102 expected).
- Grey (low): shift simulators whose family was chosen partly in view of the caller's test diagnostics. Mitigation: each simulator is a generic family (noise, partial rotation, coordinate mixing, full rotation) justified by the description's "held-out representation regimes"/"role-specific mappings". Sample its parameters from wide ranges and never fit them to test statistics.
- Strip-the-ML test: every new component is either a trained network feature or a ridge layer with a learned prior on top of trained towers. Without the towers, the invariant/structure channel alone gives 0.26 to 0.28 accuracy (measured), so the ML is load-bearing.

## Data findings
New, measured on train:
1. The cross-role map is dense, not a signed coordinate permutation. The largest |corr(q_c, a_c')| over true pairs is 0.244. Each Q coordinate's best partner reaches only 0.08 to 0.24, so the signal is spread across a rotation-like map.
2. Hidden regimes probably exist in train. A mixture of row-level linear maps (a = W_k q, within-row centred, ridge 50, fit 2,900 / held out 729 rows) scores:

   | K | held-out log-lik per row | W correlation between components | mean max responsibility |
   |---|---|---|---|
   | 1 | -244.75 | n/a | n/a |
   | 2 | -244.22 | 0.73 | n/a |
   | 3 | -243.86 | 0.66-0.73 | 0.94 |
   | 4 | -243.90 | n/a | n/a |

   - Components are stable across EM restarts (matched W correlation 0.91 to 0.93).
   - The per-regime within-row A covariance differs between regimes: matrix correlation 0.93 to 0.975, against a split-half reference of 0.997. The Q covariance is 0.96 to 0.985.
   - Row-mean GMMs show no clusters (BIC prefers K = 1). Regimes therefore differ in the map, not in offsets.
   - Value is small. An oracle regime choice using all of the row's labels gives linear accuracy +0.028 (0.497 to 0.525), and that number is optimistic. A regime posterior from anchors only gives +0.000.
3. Mapping-invariant evidence is weak.
   - Structure-only seeded graph matching (row-centred cosine Gram matrices, anchor term plus 10 Sinkhorn graph-matching iterations, no training) reaches 0.264 clean. It is identical under full rotation, 0.228 at prot1.0, 0.206 at coordmix0.5 and 0.186 at noise0.5.
   - A rotation-invariant first-order signal exists: an item's norm about its row centroid correlates 0.279 with its partner's (within-row Spearman 0.217). Adding it to graph matching: +0.014 clean/rotfull, +0.024 coordmix.
   - The invariant channel caps at about 0.28 accuracy.
4. Anisotropic gain (amplifying the top-8 train PCs by 1.8, then requantising) costs the v3-config NN -0.04 (0.571 to 0.529). Augmenting with the same family (random or PC subspaces, gain up to 2) does not recover it (0.516 to 0.523). The loss is clipping/rounding information loss, not a robustness gap.
5. Per-row EM map adaptation is the only row-local mechanism that helps more as the shift grows (linear proxy, 300 to 400 va rows). Setup: W_row = W_global + Delta, full-rank ridge lambda = 30, 8 EM steps, anchors plus Sinkhorn-soft hidden pairs.

   | shift | global linear | with EM adaptation | gain |
   |---|---|---|---|
   | clean | 0.515 | 0.516 to 0.520 | about 0 |
   | prot0.7 | 0.320 | 0.337 | +0.017 |
   | prot1.0 | 0.243 | 0.284 | +0.041 |
   | coordmix0.5 | 0.223 | 0.262 | +0.039 |

   Low-rank Delta (top-6/12 PCs) was worse than full-rank ridge.
6. Leaderboard diagnosis. A linear map fit only on anchor pairs (about 3.2 per row) from 300 / 900 rows gives hidden accuracy 0.452 / 0.484 on clean va. A solver pooling the test anchors of about 740 transfer rows would therefore land at about 0.43 to 0.48 transfer accuracy, which is exactly the caller's implied 0.43 for the top public score. The most likely explanation of the top score is cross-row pooling of test anchors, which is banned here (see Open questions).

## Validation design
- Hold-out unit: the whole row. No duplicate or near-duplicate structure was found, and ids are row-local. Keep `dev_common` va/fit for dev, and KFold(5) in solution.py.
- Transfer proxy (relative comparisons only). Fixed simulators applied to va rows; seeds fixed as in `sims.py`/`dev_noise.py`:

  | name | how it is built | seed |
  |---|---|---|
  | clean | unchanged | n/a |
  | noise0.5 | `noisy(va, 0.5, 3)`: add N(0, 0.5), round, clip | 3 |
  | prot0.7, prot1.0 | `amp(va, 1.0, 8, 4, theta)`: partial rotation exp(theta S) of both roles, then requantise | 4 |
  | coordmix0.5 | `coordmix(va, 0.5, 5)`: 16 of 32 coordinates per role replaced by a random signed unit mixture of 3 other coordinates, then requantise | 5 |
  | rotfull | `rot(va, 2)`: independent Haar rotation per role, no requant | 2 |

- Proxy score: assign each va row a stratum by k (sparse/rich) and a "transfer" tag. Proxy = 0.18 x clean + 0.82 x the mean over {noise0.5, prot0.7, prot1.0, coordmix0.5}, computed with the exact metric, OOF confidence and the 0.75/0.25 rule. Report rotfull separately as the alien extreme. The 0.18/0.82 weighting is the caller's diagnostic; it is an evaluation weighting only and is never used in training.
- Bias of the proxy, with reasons:
  - Probably optimistic on familiar rows: random split, and the generator is identical.
  - Unknown sign on transfer rows: the simulators are guesses. The real transfer may be higher-SNR (caller: larger spread, more concentrated spectrum), which would make the EM-adaptation and structure gains larger on test than in the proxy.
  - Previous simulators matched the platform within 0.01, so treat the proxy as rank-ordering, with about ±0.015 absolute uncertainty.
- Acceptance rule: compare paired on the same rows with 2 training seeds each (train-seed noise measured at about 0.005 to 0.009 accuracy). Accept a change only if proxy transfer accuracy gains at least +0.012 averaged over seeds, it improves at least 3 of the 4 shift sets, and clean drops by no more than 0.005.
- Every new constant (lambda, EM steps, Sinkhorn tau, blend/gate weights) is selected inside solution.py on folds that exclude the rows used for its reported OOF number (nested), or is fixed a priori.

## Overfit/underfit risks
- Overfit to the simulators (largest risk). Tuning lambda or gates to prot/coordmix could fail on the real regime. Mitigation: sample simulator parameters from wide ranges during training, require gains on at least 3 of 4 families, and prefer mechanisms with a principled prior (EM-ridge) over learned black-box gates.
- Clean cost of robustness. Rotation-type augmentation costs clean accuracy 0.01 to 0.02 (measured: rot0.3, 0.571 to 0.550). Mitigation: gate per row, and keep the familiar expert untouched.
- Calibrator extrapolation. Test rows with spread beyond the stress copies get flat HGB extrapolation. Mitigation: broaden the stress mix (below) and use monotone-sensible features (expert agreement).
- Underfit: none on the familiar side. Clean is at 95% of the Monte-Carlo ceiling and the learning curve is saturated, so do not add capacity.

## Recommended approach (primary + fallback)
**Primary: v3 plus three row-local add-ons, each gated by the acceptance rule.**

R1. Row-local EM map-adaptation expert (idea c/a). Mechanism: per row, solve min_Delta sum_p w_p ||a_p - q_p (W + Delta)||^2 + lambda ||Delta||^2 over anchors (w = 1) plus soft hidden pairs (Sinkhorn rows of the current score, weight = max prob), with T EM steps. This is Bayesian inference of the row's own map deviation under a Gaussian prior fit on train.
- Why it can survive the shift: it is the only measured mechanism whose gain grows with the size of the map change, from 0 clean to +0.04 at prot1.0/coordmix. It also uses more information on higher-SNR rows automatically.
- Two implementations, in order:
  - (R1a, cheap) Input-space linear expert (W from train ridge, lambda = 30, T = 8, tau = 1) as a second scorer. Its z-scored cost enters the NN head as an extra pair feature, together with the row diagnostic ||Delta||_F and the anchor residual before and after adaptation. Retrain the head so the network learns when to trust the expert.
  - (R1b, if R1a passes) A differentiable ridge head in a 16-dim projection of the tower embeddings (R2-D2/MetaOptNet style): Delta solved in closed form per row with batched `torch.linalg.solve`, T = 3 unrolled EM steps, lambda a learned log-parameter, trained with the existing listwise CE on episodes where 50% of rows get a random input-space shift (prot theta ~ U(0.3, 1.2) on one or both roles, or coordmix frac ~ U(0.2, 0.6)).
- Measured so far: a fixed-weight blend of the NN with R1a at w = 0.25 gave +0.009 prot0.7, +0.020 prot1.0, +0.007 coordmix and -0.011 clean. The head feature (R1a) is meant to remove that clean cost.
- Compliance: clean.

R2. Expert-agreement gating and confidence (idea d plus reliability). Train two members per seed: F (v3 config) and R (rot_prob 0.3).
- Measured mean F/R decode agreement is a strong label-free per-row shift signal: clean 0.74, prot0.7 0.61, prot1.0 0.50, coordmix 0.47, rotfull 0.32.
- Plain logit averaging of F and R (measured):

  | shift | change vs F |
  |---|---|
  | clean | -0.003 |
  | prot0.7 | -0.015 |
  | prot1.0 | +0.003 |
  | coordmix | +0.010 |
  | rotfull | +0.057 |

- The per-row oracle choice has +0.06 to +0.07 headroom, which is optimistic.
- Use a tiny logistic gate g(row) on [agreement, rel, n, k, ||Delta||, Q/A spread], fit on OOF clean plus stressed copies (cross-fitted), and combine S = z(S_F) + g z(S_R).
- Separately, add agreement, ||Delta||, F-vs-structure agreement and tower-anchor z-score to the confidence features.
- Expected (estimate): accuracy +0.0 to +0.015 depending on how alien the transfer is; confidence -0.002 to -0.006 score. Compliance: clean (row-local features; calibrator fit on train OOF).

R3. Broader stress mix for the calibrator and gate. STRESS_COPIES = 5 kinds:
- gain+noise and noise (as now);
- partial rotation theta ~ U(0.4, 1.2) (as now);
- coordmix frac ~ U(0.2, 0.6);
- one role fully rotated.

Each copy has a fixed seed. This prices the calibrator over a wider shift family. Expected (estimate): -0.002 to -0.004 score. Compliance: clean ("training-only robustness augmentation").

**Fallback:** v3 unchanged plus R3 only (calibration-only change). It cannot hurt accuracy and has the smallest risk.

**Expected platform range (estimate, not a promise):** primary 0.560 to 0.575, fallback 0.572 to 0.580. Reasoning: proxy transfer gain of +0.01 to +0.03 accuracy at 0.007 score per 0.01, plus 0.002 to 0.006 from confidence, starting from 0.58 with ±0.012 public noise. Beating 0.5616 has roughly even odds; reaching 0.496 is not plausible compliantly.

## Rejected options
- Pooling test anchors or test codes across rows to fit the transfer map: +0.13 transfer accuracy expected (diagnostic above), but banned by 2.3.5 (whole-test aggregation / transductive fitting).
- Synthetic rows from a fitted PCCA generator with altered gain/SNR, for training: banned by 2.3.6. Offline diagnostics only.
- Set-transformer in-context meta-learner as a black box: context encoders already gave 0. The information budget (2 to 5 anchors at canonical correlation of about 0.5, roughly 1k map parameters) makes explicit EM-ridge (R1b) the principled, lower-variance version.
- Invariant channel as a separate head feature (hidden-hidden graph-matching term plus norm-about-centroid). Measured: without rotation episodes it was neutral (clean 0.563, prot0.7 0.354, rotfull 0.167). With rot_prob 0.2 it was the same as rotation augmentation alone (rotfull 0.236 vs 0.228; clean -0.01). Keep only as an input to R2's gate if cheap.
- Anisotropic-gain augmentation: measured 0 to -0.01.
- Regime mixture-of-experts over the K = 3 familiar maps: anchor-posterior gain 0.000 (measured). Optional diversity only; see roadmap step 6.
- Low-rank Delta in R1: worse than full-rank ridge (measured).
- Sinkhorn/expected-utility decode, two-pass pseudo-anchors, QAP refine, more capacity: already measured at 0 by the caller.

## Fixed work plan & runtime budget
- Keep the v3 skeleton: 5 folds x N_SEEDS models for OOF/calibration plus N_SEEDS full-data models.
- Changes:
  - R2 doubles the member count (F and R) unless N_SEEDS is reduced from 5 to 3 per family. Do the latter: 3 F + 3 R = 6 models per fold, about the same cost.
  - R1a adds about 1 to 2 ms per row per EM run. With about 3.6k OOF rows x 6 variants plus 907 test rows, that is about 40 s on CPU.
  - R1b adds a 16 x 16 batched solve per row per step: under 10% of training time.
- Estimate: v3 wall time x about 1.1. The caller has v3's measured runtime; keep at least 30% headroom against 60 min.
- All counts fixed (folds, seeds, epochs = 40, EM steps T, Sinkhorn iterations, stress kinds/seeds). No wall-clock branches, device fixed as in v3.

## Metric-aware training & decode
- Keep the listwise row+column CE over the hidden block with fresh anchors every epoch. Keep Hungarian on symmetric log-probabilities.
- Confidence: regress R with squared loss (matches the (c - R)^2 term). The worst-stratum term does not change per-row decoding, because rows decode independently and c = E[R] is optimal under any positive row weight. So there is no metric trick beyond better E[R].
- Use the agreement and ||Delta|| features, since R on shifted rows is what drives transfer_sparse, the likely worst stratum.
- Report proxy strata separately: transfer_sparse vs transfer_rich (k = 2-3 vs 4-5).

## Structural signals
Already used by the model: row-local normalisation; anchors as inputs; the hidden one-to-one constraint (Hungarian); row/column symmetric CE.

New:
- (a) Expert agreement as a free, label-free alienness signal (R2).
- (b) Row-level map deviation inferred by EM within the row (R1).
- (c) Rotation-invariant norm-about-centroid (corr 0.279) and Gram-matrix structure (0.26 accuracy alone): weak, gate input only.
- (d) Train contains about 3 map regimes (held-out +0.9 nats/row). They could supply real-data "map-change" episodes for R1b training: take a row from regime j and add q (W_k - W_j) to its answer codes (the A side), with k != j. This augments real rows, but is unmeasured.

## Experiment roadmap
Each step takes at most 8 minutes on CPU. Run 2 training seeds, paired on the same 500 va rows, and use the acceptance rule above.
1. **R1a head feature (top experiment 1).**
   - Extend PairScorer with 3 features: z-scored EM-adapted linear cost, ||Delta||_F broadcast, and anchor residual drop. W and lambda = 30, T = 8 are fit on `fit` rows only.
   - Train the v3 config. Evaluate {clean, noise0.5, prot0.7, prot1.0, coordmix0.5, rotfull}.
   - Expected (estimate): clean ±0.005, prot1.0/coordmix +0.01 to +0.03, prot0.7 +0.005 to +0.015.
   - Stop if the proxy gain is below +0.012.
2. **R2 gate plus agreement confidence (top experiment 2).**
   - Train F and R on `fit`. On `fit` OOF (inner 3-fold) build gate data over clean plus 4 stress kinds, fit a logistic g, and evaluate gated vs F on va.
   - Expected (estimate): clean -0.002 to 0, rotfull +0.04, prot1.0/coordmix +0.005 to +0.015, prot0.7 about 0.
   - Then add the agreement features to the confidence calibrator and compare proxy score (not accuracy). Expected confidence gain: -0.002 to -0.006.
3. **R3 stress mix (top experiment 3, cheapest).**
   - Rerun v3's OOF calibration with 5 stress kinds. Compare cross-fitted proxy score on held-out copies of each kind (leave-one-kind-out: fit the calibrator without kind X, score on X) to measure extrapolation to unseen shifts.
   - Expected (estimate): -0.002 to -0.004.
4. If step 1 passes: R1b (differentiable ridge head, shift episodes 50%). Stop unless it beats R1a by at least +0.01 on the proxy.
5. Assemble the passed steps into solution.py. Run the full fixed plan twice and diff. The compliance scan must pass.
6. Optional diversity: R1b trained with regime-swap episodes (Structural signals d). Only if steps 1 to 4 left time.
7. Credits (4 left): submit one combined version only if the proxy score improves by at least 0.012 over v3. Keep at least 2 credits in reserve, and do not use credits to probe single add-ons.

## Compliance audit
- Test file use: parsing, then per-row prediction only. EM, Sinkhorn, gate and calibrator inputs are each computed from a single row. No statistic spans test rows. The W/PC bases and stress simulators are fit on train only.
- No wall-clock logic; seeds fixed; stress copies seeded; no hardware branches. Remember the platform has an A10G, but v3 runs on CPU with fixed threads. Keep it that way, set explicitly as a constant.
- Strip-the-ML: removing the towers leaves at most 0.28 accuracy against 0.56, so ML is load-bearing. R1's ridge prior is trained on train.
- Hard-coded constants: lambda, T and tau must be either learned (R1b) or chosen by an in-script grid on train folds, nested. Gate and calibrator are fit in-script on OOF.
- Synthetic data: only perturbations of real train rows (allowed augmentation). No generator samples.
- Reviewer wording: put a docstring requirements map in solution.py. Describe R1 explicitly as "row-local latent-variable inference with a train-fitted prior; no parameters are shared or updated across test rows".

## Open questions & assumptions
1. Reviewer question (potentially decisive): "May the model pool the revealed anchor pairs of several test rows (no labels) to estimate a transfer-regime mapping?" The description says only "row-local anchor conditioning" is allowed and CLAUDE.md 2.3.5 bans it, so this plan assumes NO. Under YES the gain would be about +0.1 transfer accuracy (diagnostic above). Ask before the competition closes, and do not implement it without a written yes.
2. Assumption: the top public score (0.496) relies on cross-row test pooling and may not survive review. This is evidence-based (anchor-only maps from 900 rows give 0.48), not verified.
3. Not verified:
   - the true transfer generator; the simulators are guesses;
   - whether R1a's gain carries into the NN head (only the fixed-weight blend was measured);
   - v3's exact runtime on the platform;
   - the share of transfer rows on the private split.
4. Higher-SNR transfer (if real) favours R1 and the structure channel more than the proxy shows. Lower-SNR transfer (pure noise) makes every lever about 0 (noise0.5: the best blend gained only +0.006).
