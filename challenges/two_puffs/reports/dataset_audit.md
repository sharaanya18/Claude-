# Phase 1 — Complete dataset audit

All numbers below were computed from the shipped archive, not assumed. Scripts:
`work/audit.py`, `work/decode_all.py`, `work/invert2.py`.

## 1. Files and sizes

| file | rows | content |
|---|---|---|
| `train.csv` | 1,300 | id + 8 feature columns + `target_json` |
| `test.csv` | 262 | same, no `target_json` |
| `sample_submission.csv` | 262 | `id,target_json`, 0.5 everywhere |
| `blows.jsonl` | 1,562 participants | **7,547** pre-inhaler blows for train+test |
| `pool.jsonl` | 6,045 participants | **30,001** blows, demographics inline, no targets |

1,300 + 262 = 1,562 = the `blows.jsonl` line count exactly: every graded participant has a
record and no extras. Total traces decoded: **37,548**.

## 2. Trace decoding — verified against the specification

```
raw      = np.frombuffer(buf, np.uint8).astype(np.float64)   # cast BEFORE subtracting
delta_ml = raw - 62
volume_l = np.cumsum(delta_ml) * btps / 1000.0
flow_ls  = delta_ml * btps / 10.0
```

Decoded byte distribution over all 37,548 traces: **min 18, max 254**; **37,544** traces contain
at least one byte below 62. Both match the description exactly, which confirms the decode path
(and confirms that subtracting on the unsigned array would wrap those samples to ≥194 and blow the
volume curve up to the documented ~15 L failure mode).

**387 traces (1.03%) reach `n_points = 2044`** — the recording limit, right-censoring forced
expiratory time. Matches the stated 1.2%-ish figure. A `censored` flag and a censored fraction are
carried as features.

Quality checks on the 7,206 acceptable graded blows: no non-positive FVC, no FVC above 8 L, and
`FEV1 <= FVC` for every single blow. FEV1/FVC mean 0.681 — a clinically obstructed population,
consistent with "everyone was selected for the second test because their first session already
showed obstruction".

## 3. Index conventions (Phase 2) — implemented as specified

- `ipk = argmax(flow)`, `PEF = flow[ipk]`.
- ATS back-extrapolation: the line through the peak-flow sample with slope PEF, intersected with
  zero volume, gives `i0 = round(ipk - volume[ipk] / (PEF * 0.01))`, clamped to ≥ 0.
- `FEV1 = volume[i0+100] - volume[i0]` (last sample if the trace is shorter).
- `FVC = max(volume) - volume[i0]`.
- Best FEV1 and best FVC within a draw are each the max over that draw's three blows, so they may
  come from different blows but share the same triple.

Observed: `i0` median 46 samples (0.46 s), mean 45.4, `i0 == 0` for 0.03% of blows — i.e. the
traces carry a short zero-flow lead-in and the back-extrapolation lands at the start of the forced
effort, as it should. Six blows (four acceptable) have their flow maximum after sample 200, where
back-extrapolation is physically meaningless; the documented convention is followed anyway, since
matching the answer generator beats "fixing" 4 blows in 7,206.

**Verification that these conventions are the right ones is in §7** — they reproduce the published
oracle score, which they could not do if FEV1/FVC were computed differently.

## 4. Participant-level structure

| | train | test |
|---|---|---|
| participants | 1,300 | 262 |
| age (years) | mean 39.1, sd 22.7, range 6–79 | mean 39.5 |
| height (cm) | mean 164.9, sd 16.0, range 112.9–202.7 | mean 164.6 |
| weight (kg) | mean 72.5, sd 24.5, range 17.8–186.9 | mean 71.3 |
| BMI | mean 25.9, sd 6.35, range 13.2–60.0 | mean 25.5 |
| sex | 63.2% M / 36.8% F | 63.0% M / 37.0% F |
| n_blows | mean 4.85, range 2–10 | mean 4.76 |
| n_acceptable | mean 4.62, range 2–10 | mean 4.57 |

Ethnicity, train vs test: white 48.9/50.8, black 20.4/19.1, Mexican-American 15.3/14.5,
other-Hispanic 8.8/7.3, other 6.6/8.4. **Train and test are drawn from the same distribution on
every margin** — there is no visible covariate shift to rehearse, so plain stratified
participant-disjoint folds are the right mirror of the hidden split.

`n_acceptable` distribution (train): 2→5, 3→384, 4→331, 5→249, 6→134, 7→107, 8→87, 9→2, 10→1.
Only 5 participants hold two acceptable blows; sampling three *with replacement* is what keeps the
design well defined for them.

## 5. Blow-level structure

| | acceptable (graded) | unacceptable (graded) | pool |
|---|---|---|---|
| n | 7,206 | 341 | 30,001 |
| FEV1 (L) | 2.411 ± 0.896 | 2.038 ± 0.750 | 2.609 ± 0.985 |
| FVC (L) | 3.576 ± 1.331 | 3.005 ± 1.114 | 3.241 ± 1.212 |
| PEF (L/s) | 6.553 ± 2.855 | 4.679 ± 2.460 | 6.811 ± 3.046 |
| FEV1/FVC | 0.681 ± 0.089 | — | 0.812 |
| samples | 1,152 ± 468 | 842 ± 483 | — |

Unacceptable blows are shorter, weaker and lower-flow — consistent with coughs, false starts and
early terminations. 95.5% of graded blows are acceptable (92.2% in the pool).

Confirmed quirks, so no capacity is spent on them: `effort` and `acceptable` are perfectly
collinear (every `D` is the unacceptable one — so `effort` carries no extra information);
`position` is 7,518 standing vs 29 seated in the graded file; recording stops at 2,044 samples.

**The pool is a different population**: FEV1/FVC 0.812 versus 0.681 in the graded file. It is
"less obstructed by construction" exactly as described. That makes it *valuable as a normative
reference* (what a comparable-but-unselected person blows) and *misleading as extra labelled
data*. Pool anthropometrics have a little missingness (height 31, weight 29, BMI 35 of 6,045),
carried as NaN.

## 6. Target structure

| target | mean | sd | % exactly 0 | % exactly 1 |
|---|---|---|---|---|
| fev1_00 | 0.8698 | 0.251 | 0.6 | 45.1 |
| fev1_05 | 0.6223 | 0.386 | 5.5 | 19.5 |
| fev1_10 | 0.3403 | 0.386 | 18.8 | 7.1 |
| fev1_15 | 0.1673 | 0.297 | 38.6 | 2.2 |
| fev1_20 | 0.0787 | 0.205 | 54.4 | 1.2 |
| fvc_00 | 0.5336 | 0.384 | 6.0 | 11.0 |
| fvc_05 | 0.1954 | 0.307 | 28.2 | 2.7 |
| fvc_10 | 0.0832 | 0.208 | 49.6 | 0.9 |
| ats | 0.2506 | 0.347 | 23.6 | 4.3 |

Checks that passed:
- **Every value is an exact multiple of 1/4000** → the 4,000-draw construction is confirmed.
- Monotone non-increasing within each arm for all 1,300 participants, as stated.
- `corr(fev1_10, fvc_05) = 0.4359` against the documented 0.436.
- `ats > 0.5` for 22.69% of participants; documented "the usual rule calls 22.7 percent responsive".
- Fragility bands: **40.69%** strictly between 0.05 and 0.95, **50.15%** ≤ 0.05, **9.15%** ≥ 0.95,
  against the documented 40.7 / 50.2 / 9.2.

These four independent agreements confirm the target parsing, the metric implementation and the
population at once.

## 7. Metric reproduction and reference rungs (Phase 3)

`work/spiro.py` implements RCS, FDS and `sqrt(RCS*FDS)` from the description. Hand-checked
extremes: perfect → 1.0; 0.5 everywhere → 0.0 (RCS 0, FDS 0.5); replacing `ats` by `1-ats` leaves
FDS at exactly 1.0 (the documented fragility symmetry); a constant `ats` gives FDS exactly 0.5
however good the other eight are. Somers' D is normalised by pairs untied **in the truth**, which
is what `scipy.stats.somersd(true_fragility, predicted_fragility)` does.

| rung | documented (262 held-out) | reproduced here |
|---|---|---|
| 0.5 everywhere | 0.0000 | 0.0000 |
| training average for everyone | 0.4705 | 0.4828 (in-sample on train) |
| GBDT on recomputed trace indices | 0.6017 | **0.5974** (5-fold OOF on train) |
| oracle given the true response magnitude | 0.9116 | **0.9133** |

The oracle row is the important one. It was reproduced by *fitting* a per-participant scalar
response multiplier per arm to the published probabilities and pushing it back through the exact
bootstrap. It could not land within 0.002 of the documented value if the decoding, the
back-extrapolation convention, the triple-bootstrap rule or the metric were wrong. The 0.5974 row
confirms the CV harness is calibrated against the published board.

## 8. The latent structure the audit uncovered

Inverting the published targets for a per-arm response multiplier (`work/invert2.py`) gives:

- `R_fev1`: median 1.069, IQR 1.034–1.118, 5–95% 0.975–1.208.
- `R_fvc`: median 1.001, IQR 0.975–1.036, 5–95% 0.544–1.113.
- `corr(log R_fev1, log R_fvc) = 0.429` — close to the documented 0.436 between `fev1_10` and
  `fvc_05`, i.e. the two arms are genuinely different dimensions and both must be modelled.
- Identification is sharp: the median width of the interval of multipliers consistent with the
  published probabilities is 0.003 log units. 80 participants are left-censored on the FVC arm
  (all FVC probabilities ≈ 0, so only an upper bound is identified) and 9 on the FEV1 arm; these
  are handled by the ordinal formulation, which only ever needs the 0/1 side of a threshold.

The within-session scatter the description points at is *not* a feature to be learned here — it is
an exact, enumerable part of the decode. With at most 10 acceptable blows there are at most
C(12,3) = 220 distinct triples, so the full pre-side joint distribution of (best FEV1, best FVC) is
a weighted list of at most 220 entries, computed in closed form with multinomial weights.

## 9. Information ceiling

- Published practical ceiling: 0.9116 (reproduced 0.9133). Constant-prediction floor: 0.4705.
- Competitive band therefore ≈ 0.441 wide; the published best reference claims 29.7% of it.
- The missing information is **response size**, not observation noise: a model additionally handed
  the hidden session's spread and blow count scores 0.6003, no better than 0.6017.
- Measured predictability of the latent from the pre-session (5-fold OOF, LightGBM):
  `log R_fev1` **R² = 0.183** (corr 0.435), `log R_fvc` **R² = 0.108** (corr 0.356). The single
  strongest feature is the pool-derived percent-predicted FEV1 — the more obstructed the
  participant relative to an unselected person of the same age/sex/height/ethnicity, the larger the
  response. That is the clinically expected direction and it is the engine of the whole solution.
