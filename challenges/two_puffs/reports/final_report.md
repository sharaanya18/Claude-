# Two Puffs — final report

## Dataset

| | |
|---|---|
| train participants | **1,300** |
| test participants | **262** (= all of `sample_submission.csv`, ids in the same order) |
| graded pre-inhaler blows | **7,547** (`blows.jsonl`, 1,562 participants) |
| pool participants / blows | **6,045 / 30,001** (`pool.jsonl`, no targets) |
| traces decoded per run | **37,548** in ~71 s |
| acceptable blows (graded) | 7,206 of 7,547 = **95.5 %**; pool 92.2 % |
| `n_acceptable` per participant | mean 4.62, range 2–10 (train); mean 4.57, range 2–8 (test) |
| acceptable-blow indices | FEV1 2.411 ± 0.896 L, FVC 3.576 ± 1.331 L, PEF 6.553 ± 2.855 L/s, FEV1/FVC 0.681 ± 0.089 |
| unacceptable blows | shorter and weaker (FEV1 2.038, PEF 4.679, 842 vs 1,152 samples) |
| right-censored traces | 387 of 37,548 = 1.03 % at the 2,044-sample recording limit |
| demographics (train) | age 39.1 ± 22.7 (6–79), height 164.9 ± 16.0 cm, weight 72.5 ± 24.5 kg, BMI 25.9 ± 6.4, 63.2 % M |
| train vs test | matched on every margin (test age 39.5, height 164.6, BMI 25.5, 63.0 % M) — no covariate shift |

Verified against the description: byte min 18, 37,544/37,548 traces with a byte below 62, all
targets exact multiples of 1/4000, monotone within each arm for all 1,300, `corr(fev1_10, fvc_05)`
0.4359 (vs 0.436), `ats > 0.5` 22.69 % (vs 22.7 %), fragility bands 40.69/50.15/9.15 %
(vs 40.7/50.2/9.2). Table `n_blows`/`n_acceptable` match the traces for all 1,562 participants.

## Best validation (the shipped script's own numbers)

Stratified (response band × acceptable-blow count) participant-disjoint 5-fold, exact official
metric, every post-hoc constant cross-fitted.

| | score | RCS | FDS |
|---|---|---|---|
| **out-of-fold, final pipeline** | **0.6240** | 0.5672 | 0.6865 |
| per-fold mean ± sd | **0.6251 ± 0.0097** | | |
| worst fold | **0.6110** | | |
| 3 split seeds (42 / 7 / 2024) | 0.6240 / 0.6241 / 0.6267 → **0.6249 ± 0.0015** | | |
| untouched 20 % sanity holdout | **0.6246** (rest 0.6237) → no selection optimism | | |

Mean Brier 0.07870. Per-target Brier: fev1_00 0.0584, fev1_05 0.1221, fev1_10 0.1053,
fev1_15 0.0585, fev1_20 0.0279, fvc_00 0.1342, fvc_05 0.0725, fvc_10 0.0331, ats 0.0962.

ATS ranking: Somers' D of the submitted fragility against the truth **0.3788**, against 0.2944 for
the fragility implied by the Brier-optimal probability.

Reference rungs reproduced in the same harness: 0.5 everywhere **0.0000**; constant train mean
**0.4828**; nine-target GBDT on recomputed indices **0.5917–0.5974** (documented 0.6017 held-out);
oracle given the fitted latent **0.9133** margin-only / 0.9397 ats-refined (documented 0.9116).

## Best model

* **Model type** — two ordinal "at least this much response" gradient-boosting ladders (3 LightGBM
  configurations each, averaged on the logit scale) learning the *conditional survival function* of
  a latent response scalar per arm; plus a direct nine-target LightGBM as a minor blend member; plus
  a one-parameter Gaussian copula and a weighted isotonic fragility map. No pretrained weights, no
  neural nets, no GPU.
* **Feature set** — 220 columns: 205 participant features computed from the raw traces (FEV0.5/1/3/6,
  FVC, PEF, FEF25/50/75, FEF25-75, ratios, timings, back-extrapolated volume, limb concavity,
  artefact counts), each summarised across the session by level **and** scatter statistics
  (sd/CV/range/IQR/MAD, best-minus-second-best, best-minus-median, ATS 150 mL repeatability),
  all-blow-vs-acceptable contrasts, blow-order trends, ethnicity dummies, and **pool-fitted
  percent-predicted** values.
* **Target formulation** — the nine probabilities are never regressed. The published bootstrap is
  enumerated exactly (≤ C(12,3)=220 triples) and every event becomes one threshold on a latent
  `delta` with `post = pre * (1 + delta)`; `LAM = 1` chosen by CV over {0, 0.5, 1}.
* **Calibration** — structural: `p = E_delta[P_bootstrap(event | delta)]`. PIT of the latent is near
  uniform; a global shift/spread recalibration found spread = 1.00 optimal, i.e. nothing left to fix.
  All 262 test rows are monotone in both arms by construction.
* **Pool usage** — normative reference equations only (unsupervised, no targets). Never as labelled data.
* **Ensemble** — 3 LightGBM configs × 2 arms, + direct nine-target member at weight 0.15–0.20 fitted
  out-of-fold on a bounded grid, all refit on 100 % of train with fixed counts.

## What raised performance above the ~0.65-class baselines

Against the published best reference (0.6017) the gains are, in measured order:

1. **+0.086 — replacing nine probability regressions with one conditional distribution over a latent
   response, decoded through the exact bootstrap** (0.527 → 0.613 once the distribution was made
   conditional; see the failure note below). This is where calibration and output coherence come from.
2. **+0.017 — the fragility half.** FDS depends *only* on the ordering of `|p − 0.5|`, so the only
   lever is a better fragility signal. `E[4t(1−t)|x]` from the latent posterior orders the truth far
   better than the implied `4p(1−p)` (D 0.36 vs 0.29), and the **Borda / expected-marginal-rank**
   statistic beats even that (D 0.379). Transferring its *ordering* by weighted isotonic regression,
   rather than blending its values, converts almost all of it into score at an RCS cost of 0.0002.
3. **+0.003 — three diverse boosting configurations** instead of one.
4. **+0.001 — the direct nine-target member** as a structurally different blend partner.
5. The pool-derived **percent-predicted** features are the top-ranked learned features for the FEV1
   arm (`pp_fef2575_max`, `pp_fev1_max`, `pp_ratio_med`, `pp_pef_max`) — the more obstructed relative
   to an unselected person of the same build, the larger the response.

The instructive failure: a first version used one *global* residual law instead of a conditional one
and scored **0.527**, worse than the direct baseline, because the empirical residual law is
heavy-tailed (kurtosis 30) and far too wide (`fvc_10` mean 0.223 against a truth of 0.083).

## Reference comparison

The #1 private-LB solution was **not supplied** in this session (only the dataset archive was
attached), so Phase 5 was executed against the reference ladder the description publishes. Details
and the full twelve-point analysis are in `private_lb_reference_analysis.md`. Borrowed
conceptually: recomputing indices from the traces, and keeping within-session scatter. Improved:
the forward model (exact bootstrap instead of learning around it), the target (a latent
distribution instead of nine regressions), calibration (structural instead of none), and the
fragility half (which the reference ladder does not address at all).

## Generalisation argument

1. The decode is the challenge's **own published label construction**, not a fitted artefact, so it
   cannot overfit.
2. The learned part is a 2-dimensional conditional distribution, not 9 free functions — far less
   capacity to overfit on 1,300 participants.
3. Only three post-hoc constants exist (one copula correlation, one blend weight on a 9-point
   bounded grid, one monotone isotonic map), all cross-fitted.
4. Stability is measured, not assumed: 3 split seeds give 0.6249 ± 0.0015; the untouched 20 %
   holdout scores 0.6246 against 0.6237 on the rest; the worst fold is 0.6110.
5. Train and test match on every demographic margin, so the fold design mirrors the hidden split.
6. The OOF number is mildly **pessimistic**: fold models see 80 % of train while the shipped model
   is refit on 100 % (the reference rung reproduced 0.5974 OOF against 0.6017 held-out).

**Expected private band: ≈ 0.60–0.66**, centred slightly above the OOF 0.624, with FDS (a pairwise
concordance on 262 participants) the noisier half.

## Honest note on the 0.75 target

0.75 is not reachable from the pre-inhaler session. The task reduces exactly to predicting one
scalar per arm — that reduction is the organisers' own, and reproducing their published oracle to
within 0.002 confirms it. The achievable out-of-fold R² of that scalar is **0.18** (FEV1) and
**0.11** (FVC). Feeding the real pipeline an artificially improved latent gives 0.18 → 0.613,
0.44 → 0.668, 0.52 → 0.724, 0.58 → 0.782, so 0.75 needs R² ≈ 0.55–0.60. The description
corroborates this independently. Full reasoning in `final_approach.md` §15.

## Compliance and engineering checks

| check | result |
|---|---|
| `python3 solution.py <public_dir> <submission_out>` | exits 0, **296 s** (~5 min), far inside the 1.5 h ceiling |
| determinism | two independent full runs produce a **byte-for-byte identical** submission |
| `compliance_scan.py` | **0 errors** (one unavoidable warning: `base64.b64decode`, which is the challenge's own trace format) |
| `validate_submission.py` | **PASS** — 262 rows, columns/ids/order match `sample_submission.csv`, JSON parses, all nine keys, all values finite in [0,1] |
| monotonicity | FEV1 arm 262/262, FVC arm 262/262 (not required by the grader, but the truth always is) |
| `assert` statements in the shipped script | **0** — all checks are explicit `if ... raise ValueError` and are unreachable by construction |
| `try`/`except` in the shipped script | **0** (Guidebook §3.3 bans fallbacks) |
| seeds | `random`, `numpy`, every LightGBM seed argument by name, the fold shuffler, the posterior lattice |
| clock / machine dependence | none; time is used for log lines only; `N_THREADS` is a constant |
| test-set use | none beyond one-participant-at-a-time inference; the Borda reference CDF and every model and map are fitted on train and reused |
| pretrained weights / network | none; no library outside numpy, pandas, scipy, scikit-learn, LightGBM |
| strip-the-ML control (run in-script) | 0.5369 without the learned distribution vs 0.6240 with it — the models supply **56.7 %** of the skill above the constant-prediction floor |

## Final submission

`submissions/submission_v1.csv` (identical to `run3/working/submission.csv` and
`run4/working/submission.csv`): 262 rows, `id,target_json`, all nine keys per row, every value
finite and in [0,1], ids exactly the test ids in `sample_submission.csv` order, no duplicates, no
missing or unknown ids. Test prediction means track the train target means closely
(e.g. ats 0.2665 vs 0.2506; fev1_10 0.332 vs 0.340).
