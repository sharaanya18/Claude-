# Metric spec: Marginal Role Coverage@2 (complementary_entity_evidence)

Files: `metric.py` (stdlib only), `tests_metric.py` (27 tests, all pass; includes differential fuzz against a verbatim copy of the description's `grade()`).

## 1. Formula
Per slate (rows sharing the middle id key = `query_id`): `selected` = top-2 by `(-score, id)` (ascending string id breaks ties).
`slate_score = |roles[s1] U roles[s2]| / max_{pairs} |roles[a] U roles[b]|`. Metric = plain mean over slates (each slate weight 1; slate sizes 2..8, so candidates in big slates do not weigh more).
Roles are the candidate's incident role inventory minus the seed's known roles (hidden for test). Range [0,1], 1 is attained by any optimal pair.

## 2. Invalid / rejected outputs (copied from grade())
Columns exactly `id,prediction`; non-empty, no NaN; unique ids; id set equals the test set; every prediction float-coercible and finite (else exception, not a penalty); each slate complete (`len(members) == slate_size`, all members agree, `type is int`, 2 <= size <= 8); optimum 0 raises (test slates are pre-filtered so it never occurs). Finite wrong rankings just receive their actual utility. A prediction string beginning with `{` is read as the oracle JSON and uses `oracle_rank` (implemented in `_numeric`).

## 3. Term analysis
| Aspect | Consequence |
|---|---|
| Output used | Only the top-2 set per slate matters. Scores of ranks 3..8 are irrelevant; only the pair choice is scored. |
| Pair, not item | The value of a candidate depends on the partner: redundancy is penalised (test `redundancy_case`: top-2 by individual size A,B gives 2/3 while A+C gives 1). |
| Denominator | Per-slate best pair union (>=1). Slates whose optimum is 1 or 2 are the most sensitive: one missing role changes the score by 50% or 33%. Scores are in multiples of 1/opt. |
| Ties | Constant scores = "two smallest ids" = id-order baseline; id order is opaque, so this is a random-pair baseline (toy: ~0.55). Emit distinct scores so the tie rule is never used. |
| Equal-optimal pairs | Any optimal pair gives 1; `oracle_rank` is one tie choice only. Train on role sets, not on the oracle pair. |
| Weight | Equal per slate; 164 test slates / 59 cohorts, about a third public. Per-slate SD is large (scores like 0.5, 0.67, 0.75, 1.0), so expect LB SE of roughly 0.03. |
| Saturation | Not invertible. Perfect role prediction = 1.0 (oracle decode check below). |

## 4. Training-move map and decode plan
1. Train a per-candidate, per-role presence model (multi-label BCE over a closed role vocabulary of direction x category; known roles masked in loss and decode). It is the quantity the metric is a function of, so it gives dense supervision (every role of every candidate) instead of the single oracle pair. Optionally add a pairwise or listwise head (pair label = union size / optimum, softmax over the <=28 pairs) as a second ensemble member.
2. Calibrate p per role on OOF (one temperature or Platt scale, cross-fitted), since the decode consumes probabilities.
3. Decode candidates, ordered by expected value / risk:
   - A. `choose_pair_expected_union` (closed form, E|union| under independent Bernoulli roles, exact for that model, no randomness). Default.
   - B. `choose_pair_mc_ratio` (expected ratio |union|/opt, MC, fixed seed 20240601, common random numbers across pairs, n_draws=2000; conditioning on opt>0 only rescales the table and never changes the argmax, fallback to A when every draw has opt 0). Use >= 5000 draws if enabled.
   - C. Pointwise top-2 by expected count (ablation floor; ignores redundancy).
   Ban check: the description does not forbid metric-aware decoding (only "no hardcoding of oracle choices", no test-set statistics); the decode is per slate from that slate's own outputs, so it is per-sample inference. Re-check at review (see questions).
4. `pair_to_scores` turns the pair into distinct scores: chosen pair first (larger own expected count first), remaining sorted by marginal expected new roles given the pair, ties by id. `decode_all` groups by slate key only.

## 5. Oracle decode check
Gold role sets as 0/1 probabilities give score exactly 1.0 for both decoders on 400 random toy slates (min = mean = 1.0), and through the dict-level `decode_all` + `marginal_role_coverage` path on 50 slates. Metric and decoder are consistent.

## 6. Closed-form vs MC decode on toy slates (calibrated uncertainty, 600 slates, mean size 5.7)
Toy slates: independent roles, calibrated Beta(0.35,2.0)-scaled probabilities, Zipf role prior, retained when optimum>0 and pair utilities differ (like the real preparation). `informative` shrinks p toward the prior.
| informative | same pair CF vs MC(2k) | MC seed A vs seed B | gold score MC / CF / top-2-individual / random / id-order |
|---|---|---|---|
| 1.0 | 93.8% | 93.2% | 0.734 / 0.739 / 0.724 / 0.542 / 0.554 |
| 0.7 | 91.5% | 90.7% | 0.731 / 0.734 / 0.724 / 0.542 / 0.554 |
| 0.4 | 86.0% | 82.5% | 0.727 / 0.726 / 0.724 / 0.542 / 0.554 |
Paired MC - CF = -0.005 / -0.003 / +0.001 (SE ~0.005): no detectable difference. With a converged MC (30k draws, 250 slates) CF picks the same pair in 97.6% of slates; the mean loss of CF in expected ratio is 8e-5 (max 0.007). Disagreements are near-ties, and most of the 6-14% MC-vs-CF disagreement at 2k draws is MC noise (seed-vs-seed agreement is about as low). Conclusion: closed form is the primary decode; MC at 2000 draws adds noise, and with >=5000 draws it is at best a tie-breaker. The big gain is redundancy-awareness (+0.01 over pointwise top-2 on toys here; the gap is larger when roles overlap strongly, so re-measure on real OOF) and above all the role model quality. Toy numbers are not predictions for the real task (role independence and calibration are assumed).

## 7. Free decode constants (count: <= 3, limit 6)
(1) probability temperature/Platt scale (cross-fitted on OOF by cohort groups), (2) optional blend weight between role-head p and pairwise-head pair score if both exist, (3) n_draws (fixed 2000-5000, not fitted; seed fixed). Role independence is assumed; an overlap-correlation correction is rejected for now (extra constants).

## 8. Open questions for a reviewer
- Direction of `oracle_rank`: `decode_answers` feeds it as `prediction` where larger = selected first, so the optimal pair presumably has the largest values; unverified until train_targets are available (check on train that `grade(decode_answers)` = 1.0 with `marginal_role_coverage`).
- Is the within-slate role independence reasonable (roles of one sentence co-occur by relation structure)? Check OOF calibration of union probabilities once data exist.
- Is metric-aware decoding of probabilities acceptable under the platform's "model must do the learning" rule? It is a deterministic post-processing of trained model outputs, not a rule engine, but flag it.
- Prose says "Cohorts need at least five sentences" while slate sizes run 2..8; slates may have fewer than 8 candidates; the grader enforces completeness through `slate_size` only.
