# Shared Adam Memory Repair: digest of five top solutions (user-supplied, own words, no code copied)

My solution: public 0.589 on the platform (gradient-span Gaussian posterior over the two first-moment buffers, LightGBM estimate of v, 4096-order pool,
top-128 re-scored under 8 posterior draws). The five solutions below were supplied for learning; their private scores are not known to me.

## What all five share
1. **Latent-state estimation, not order ranking.** Every one infers the two hidden Adam states, then picks the order by simulating the disclosed learner
   (the same direction my own analysis found: only the response of the buffers on the request images matters).
2. **Per-parameter learned priors** (an MLP shared across the 2,368 parameters) for log v and for m (as m/sqrt(v) or m/gradient-scale), fed by
   gradient statistics of the request's own images (per-image / per-batch / per-class squared gradients, row/column statistics, weights, layer one-hots).
3. **Exact diagnostic inversion** with the real one-step Adam equations: Gauss-Newton / Levenberg-Marquardt or LBFGS fits of the state to the 4 probe outputs,
   with a ridge or Gaussian prior (my version used one linearisation).
4. **Whole order space or near it:** prefix-tree enumeration of all 40,320 orders on the GPU (ranks 1, 3, 5) or a pool plus swap local search (ranks 2, 4).
5. **Several state hypotheses per request**, averaged in the objective (mean log divergence, or mean quality limit/D with the cap check).

## Levers that separate them from mine
- **Better gradient bases** (rank 3, rank 5): per-image gradients taken not only at the public weights but at weights moved along a *common-step trajectory*
  (rank 5 uses steps 0/4/8/12) or at a perturbed point (rank 3). The hidden buffers are EMAs of gradients taken along a moving path, so a basis at w0 alone
  misses part of them (I saw R2 0.4-0.9 in u-space). My single guessed perturbation did not help; the trajectory built from a *predicted* common step is the cleaner version.
- **Common / difference decomposition** (ranks 1, 5): model (m_A+m_B)/2 and (m_A-m_B)/2 separately, with their own coefficients, scales and a loss that
  targets the A-B difference (rank 1 trains with an explicit differential loss). The order effect lives in the difference.
- **Learned uncertainty from out-of-fold residuals** (ranks 1, 2, 5): block-wise spreads measured on held-out capture sequences, then used for posterior draws
  or Laplace samples. My draws used hand-set noise.
- **End-to-end training through the unrolled simulator** (rank 2): loss = simulated divergence signature versus the one produced by the true moments.
- **Using the public divergence limit as calibration** (rank 2): the limit behaves like "best of N random orders"; N is estimated from training data and used in a
  tail-quality decision rule. (I noticed the same fact but did not use it.)
- **Decision rule chosen in-script on held-out capture sequences with the official metric** (ranks 2, 4): variants (screen, mean log, tail quality, draw temperature)
  are scored on training sequences never used to fit that estimator; canonical/reverse orders stay in the comparison as sanity baselines.
- **Prior strengths varied across an ensemble** (rank 3): eight MAP fits with different noise/prior scales, first fit ranks all orders, the rest re-score the
  top 4,000; a candidate must be feasible under every member.

## Compliance observations (for our stricter rules)
- Rank 4 arms a wall-clock `signal.alarm` deadline; rank 1 states a "user override" run limit; rank 2 picks `cuda` or `cpu` with `is_available`; rank 3 accepts
  optional CLI flags. CLAUDE.md section 3 flags all of these patterns; whether they were accepted by reviewers is unknown to me. Keep our stricter reading.
- Rank 5 raises when CUDA is absent (no fallback), which fits the contract better than a silent CPU switch.
