# Phase 5 — Reference-approach analysis

## 0. Status of the input (read this first)

The brief promised three inputs: the platform repository, the dataset, and **the #1 private-LB
solution**. Only the dataset archive (`5897cb66-public.zip`) was attached to this session. No
solution code, write-up or score table for a #1 private-LB entry was supplied, and I did not go
looking for one: other solvers' Eris solutions must not be read or copied (`platform-facts.md`,
research note), and the challenge forbids any derivative that could encode the held-back session.

So **Phase 5 as specified could not be executed**, and nothing below is a reading of a #1
solution. What follows is the next most useful thing: a reverse-engineering of the *reference
ladder the challenge description publishes*, which is a real, scored, documented pipeline, plus
the twelve-point analysis applied to it and to my own approach so the comparison the brief asks
for can still be made. **If you paste the #1 solution, I will redo this phase properly** — the
analysis slots in without disturbing anything else.

## 1. What the published reference ladder actually tells us

| rung | score (262 held-out) | what it isolates |
|---|---|---|
| 0.5 everywhere | 0.0000 | the metric's zero point |
| "no-fit rule, whole response is measurement noise" | 0.2756 | blow-sampling noise alone, response assumed nil |
| training average for everyone | 0.4705 | the population prior; FDS pinned at 0.5 |
| GBDT on the feature table alone, never opening a trace | 0.5168 | demographics + blow counts |
| **GBDT on standard indices recomputed from the traces** | **0.6017** | the trace indices and their within-session scatter |
| the same + normalised flow-volume loop shape + its spread | 0.5962 | loop shape *costs* 0.0055 |
| oracle given the true response magnitude, still facing blow noise | 0.9116 | the ceiling |

Five things are deducible, and all five were load-bearing for my design:

1. **Deriving the indices is worth +0.0852 over the trace-blind table** (bootstrap SE 0.0174, 4.9
   SE, positive in 100% of 4,000 resamples). So accurate FEV1/FVC/PEF reconstruction is the
   single largest *measured* lever in the description.
2. **Loop shape is a measured negative.** Two independent blocks (shape and shape-spread) were
   added and the score fell. The description itself warns against over-reading it, but the
   direction is informative: a nine-target model does not have the capacity to spend on 24 extra
   correlated columns at n=1,300.
3. **Within-session scatter is inside the winning rung, not the losing one.** The description says
   so explicitly ("what does pay, already inside the indices rung, is how much the session's
   FEV1, FVC and PEF disagree between blows, because the answer's before-side is a draw from
   exactly those blows").
4. **The latent is a scalar per arm.** The oracle is defined by handing the model the "true
   response magnitude" and nothing else, and it reaches 0.9116; a model additionally handed the
   hidden session's spread *and* blow count reaches 0.6003, i.e. no better than 0.6017. So the
   post-session's own scatter carries no usable information once the response size is known or
   unknown respectively.
5. **The reference is not fragility-aware.** It is described as "gradient boosting on a few dozen
   derived features" — i.e. nine (or one multi-output) regressions on the published probabilities.
   Nothing in that construction targets the ordering of `4p(1-p)`, and the geometric mean makes
   that half of the score as valuable as the Brier half.

## 2. The twelve-point analysis

Applied to the published reference pipeline (the only reference available), with my own approach
beside it.

**1. Feature engineering.** Reference: "a few dozen derived features" — standard spirometric
indices per blow, aggregated per session, plus the shipped demographics and blow counts. Mine:
205 per-participant features — FEV1, FEV0.5, FEV3, FEV6, FVC, PEF, FEF25/50/75, FEF25-75,
FEV1/FVC, FEV1/FEV6, time-to-peak, volume-at-peak, forced expiratory time, back-extrapolated
volume and its fraction, end-of-test and plateau flow, artefact counts — each summarised across
the session by max/min/median/mean/sd/CV/range/IQR/MAD, best-minus-second-best,
best-minus-median, plus all-blow versus acceptable-only contrasts, ATS 150 mL repeatability
flags, blow-order trends, and **normative reference equations fitted on the shipped pool** giving
percent-predicted FEV1/FVC/PEF/FEF25-75. That last block is the single most important feature:
`pp_fev1_max` is the top gain feature for the FEV1-arm latent.

**2. Trace processing.** Identical in principle — decode, back-extrapolate, integrate. The one
place to get right is casting to float before subtracting 62; verified against the published byte
statistics (min 18, 37,544/37,548 traces with a byte below 62).

**3. Participant-level aggregation.** Reference: summary statistics fed to the model. Mine: the
same statistics as *features*, **plus the within-session scatter handled exactly rather than
learned** — the pre-side bootstrap over triples is enumerated in closed form (≤ C(12,3)=220
multisets) and is part of the decode, not part of the feature vector.

**4. Probability modelling.** Reference: regress the nine published probabilities directly. Mine:
regress the *conditional survival function of the latent response* with an ordinal ladder of
"at least this much response" heads, then push it through the disclosed bootstrap. The nine
probabilities are never regressed.

**5. Calibration.** Reference: whatever L2 regression on probabilities gives, possibly clipped.
Mine: structural. `p = E_latent[P_bootstrap(event | latent)]` is a calibrated probability by
construction if the latent posterior is calibrated, so calibration reduces to getting one
conditional distribution right — checked by PIT uniformity — rather than nine isotonic patches.
Monotonicity within an arm is automatic, which the description notes the truth always satisfies.

**6. ATS prediction.** Reference: a tenth regression. Mine: the ATS rule is evaluated *exactly*
inside the decode, including its 200 mL absolute arm and its 12% relative arm, over the joint
distribution of the two latents coupled by a one-parameter Gaussian copula whose correlation is
estimated from out-of-fold PIT values (0.53, against the 0.436 target-level correlation the
description publishes).

**7. Fragility ranking.** Reference: nothing. Mine: the decisive observation is that FDS depends
**only on the ordering of |p−0.5|**, so any symmetric monotone transform of `ats` leaves it
unchanged and the only way to move it is a different fragility signal. The Brier-optimal
`p = E[t|x]` has fragility `4p(1−p)`, but the quantity the grader compares against is
`E[4t(1−t)|x] = 4p(1−p) − 4Var(t|x)`; where the latent posterior is wide the two orderings
disagree badly (a participant who is *certainly* positive-or-negative but we don't know which
gets `p ≈ 0.5`, hence maximal implied fragility, and minimal true fragility). Computing
`φ = E[4t(1−t)|x]` by sampling the latent posterior and transferring **its ordering** onto the
submitted `ats` by weighted isotonic regression lifts FDS from 0.653 to 0.682 at an RCS cost of
0.005 — measured +0.011 on the final score. A direct GBDT trained on the true fragility is
*worse* (Somers' D 0.292 vs 0.367): the physics beats the black box here.

**8. Multi-task structure.** Reference: nine effectively independent heads (the monotone
structure and the shared response are left for the model to rediscover from 1,300 rows). Mine:
two latent dimensions, hard-coupled to all nine outputs by the published rule. That is the
data-efficiency argument — 2 conditional distributions instead of 9 noisy regressions.

**9. Validation.** Reference: unknown; the description reports held-out numbers and bootstrap
SEs. Mine: participant-disjoint by construction, stratified on response band × acceptable-blow
count exactly as the description's validation tip recommends, 5 folds × repeated seeds, the exact
official metric, every post-hoc constant cross-fitted, and the harness calibrated by reproducing
two published rungs (0.5974 against 0.6017; 0.9133 against 0.9116).

**10. Ensembling.** Reference: a single GBDT. Mine: three deliberately different GBDT
configurations averaged on the logit scale for the latent survival, plus the direct nine-target
model kept as a structurally different blend member with a cross-fitted weight.

**11. Hidden weaknesses — of the reference.** (a) FDS left entirely to chance; a constant-ish
`ats` caps the score at `sqrt(RCS·0.5)`. (b) Non-monotone, non-coherent outputs across the five
FEV1 thresholds, which the description says "only costs you". (c) No mechanism to separate "we
don't know the response" from "the response is borderline" — exactly the distinction FDS pays
for. (d) Nine heads × 1,300 rows is a lot of variance to spend on correlated targets.

**Hidden weaknesses — of mine**, stated plainly: (a) the whole approach rests on the assumption
that the hidden session's blows look like the pre-session's rescaled, which is the assumption
*the organisers themselves used to compute the published ceiling* but is still an assumption, and
the description warns to "treat it as a guide"; (b) the latent targets are inverted from the
published train probabilities, so any systematic error in my index conventions propagates into
the targets — mitigated by the oracle check landing on 0.9133; (c) the isotonic fragility
transfer is a flexible post-hoc step on 1,300 rows and needs the cross-fitted number, not the
in-sample one; (d) 80 participants are left-censored on the FVC arm, handled by the ordinal
formulation but never positively identified.

**12. Generalisable components.** Transferable beyond this challenge: *(i)* when the label is a
documented function of a low-dimensional latent, invert for the latent and learn **its
conditional distribution**, then re-run the documented function — calibration and output
coherence come for free; *(ii)* enumerate small discrete nuisance distributions exactly instead
of learning around them; *(iii)* when a metric term depends only on an *ordering*, work out what
statistic that ordering should follow and transfer it with isotonic regression at minimal cost to
the other term; *(iv)* fit normative reference equations on an unlabelled shipped pool to turn an
absolute measurement into a percent-predicted one. Dataset-specific: the exact index conventions,
the 62-byte offset, the 200 mL/12% rule, the three-blow triple.
