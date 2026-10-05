# AnchorPerm: digest of five top solutions (user-supplied, own words, no code copied)

Context: my own v3 (row-local only) scored about 0.58 on the platform against an AI baseline of 0.5616; my pooled-anchor v4 was blocked
because CLAUDE.md 2.3A requires written reviewer approval for any use of other test rows. These five solutions are the competitors' approaches.

## The one thing all five share (and it is the thing I had blocked)
Every one of them estimates the held-out representation regime from the revealed anchors of MANY test rows and then uses that estimate on each row:
- rank 1: GMM/BIC over row statistics of the test file -> per-regime orthogonal adapters fitted on that regime's revealed anchors, one round of self-training with the
  ensemble's confident hidden pairs, refinement on held-out-anchor episodes, identity kept if a cross-fitted held-out-anchor check shows no gain;
- rank 2: familiar-versus-shifted split by extreme-code share, k-means on covariance signatures, regime maps from pooled anchors plus decoded pairs, each member adapted
  on the shifted rows' revealed anchors (cross-fitted halves);
- rank 3: k-means on row statistics, anchor-only regularised CCA per regime, cross-fitted so a row never sees its own anchors;
- rank 4: GMM over train+test row statistics, bidirectional ridge fitted on the regime's anchors, 8 rounds of self-training on its own decoded pairs;
- rank 5: a ridge map pooled over all other evaluation rows' anchors (own row left out), fed to the pair model as a second view.
This is test-set use beyond one-row inference (CLAUDE.md 2.3 #5 / 2.3A). The challenge text lists "row-local anchor conditioning" as allowed and is silent on pooling.
No written approval exists in this workspace, so these mechanisms stay out of our solutions. Recording them is for understanding, not for adoption.
Whether the platform's post-competition review accepts them is unknown to me; the five files ranked, so the automated checks did not stop them.

## Patterns that do NOT touch other test rows (compliant, adoptable)
1. **Regime-invariant second-order evidence:** within-role cosine geometry (question-question vs answer-answer), anchor-profile similarity, norms/centrality of an item
   inside its own row, quadratic-assignment consistency of the hidden items. These do not depend on the code mapping, so they survive representation shift.
2. **Row-local dequantisation:** maximum-likelihood estimate of the row's quantiser scale, then conditional-mean latent values (rank 1).
3. **Train-only shift simulation:** random partial rotations / signed partial coordinate permutations / re-quantisation on a finer grid / offsets, with a "regime bank" whose
   anchor-estimated maps are simulated from other TRAIN rows, so the model learns to weigh an anchor-fitted map against the train map (ranks 1, 2, 5).
4. **Learned graduated assignment:** Sinkhorn plus quadratic-assignment messages inside the network with deep supervision at every round (rank 1); a transformer over
   question and answer items with anchor-link embeddings and gated relational channels (rank 2); two-stage gradient-boosted pair classifier over first-order
   out-of-fold tower scores plus second-order features (rank 5).
5. **Decoding:** average Sinkhorn marginals over 10-12 seeds, then Hungarian on the marginals (maximises expected correct positions); pairwise-swap local search on a
   graph-matching objective (rank 4).
6. **Confidence:** isotonic map from the mean selected marginal to realised row accuracy, fitted on held-out-anchor positions (rank 1) or on out-of-fold / simulated rows
   (ranks 4, 5); leave-one-anchor-out recovery as a per-row signal (ranks 3, 5); bootstrap agreement of the whole assignment (rank 4). Because the metric is
   0.85 * L_seq + 0.15 * (c - R)^2, a calibrated expected accuracy is the right target.
7. **Stratum-aware validation:** score simulated-regime views as their own strata (familiar/shifted x sparse/rich) and report the worst stratum, since the
   metric adds 0.25 * worst-stratum loss (rank 5).

## Reading for future work
- My earlier ceiling analysis (L001/L003) was right: row-local models could not recover a changed code mapping; the large lever was pooling anchors across a regime.
- If a future challenge states pooling is allowed, or a reviewer gives WRITTEN approval, the rank-1 recipe is the template (discover regimes, fit adapters on revealed
  anchors only, cross-fitted held-out-anchor safety check, identity fallback). Otherwise use items 1-7 above, which are compliant.
