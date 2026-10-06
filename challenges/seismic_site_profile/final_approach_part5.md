## 4. Performance

All figures are the exact challenge metric on **station-held-out** out-of-fold
predictions: the supplied `folds.csv` split, plus a second split generated the
same way (whole stations, stratified on the station's profile). Conversion:
`cell_score = 0.25 + 0.75 * score`, because all four bands are present in all
four cells so the blind reference is 0.25.

### The shipped configuration
Equal-weight average of **ordgb + ord + et** on the 355-column feature set,
decoded with the training profile prior at lambda = 1.0:

| split | score | per-cell macro-F1 (0-5, 5-10, 10-20, 20-30 m) | cell_score |
|---|---|---|---|
| supplied `folds.csv` | **0.2721** | 0.416  0.475  0.480  0.446 | 0.4541 |
| second station split | **0.2803** | 0.407  0.495  0.482  0.456 | 0.4602 |
| **mean** | **0.2762** | 0.412  0.485  0.481  0.451 | 0.457 |

Fold spread within a split is 0.014-0.027 (standard deviation over the five
folds); no accepted decision rested on a single fold.

### How it was built up
| step | score | gain |
|---|---|---|
| H/V curve alone (24 columns), boosted trees | 0.1528 | - |
| + normalised component spectra, peak structure, shape stats, polarisation, window stability, time domain | 0.2190 | +0.066 |
| + trend-removed spectra and cepstral lags | 0.2159 | -0.003 |
| + coda-window H/V (group K) | 0.2264 | +0.011 |
| + long-window low-frequency H/V (group M) | 0.2470 | +0.021 |
| best single family: ordinal cumulative-link boosting (ordgb) | 0.2470 | - |
| equal-weight blend of three diverse families | 0.2619 | +0.015 |
| **+ structured joint-profile decode (lambda = 1.0)** | **0.2762** | **+0.014** |

### Against the published references
| | score | mean per-cell macro-F1 |
|---|---|---|
| constant profile / most common band / `sample_submission.csv` | 0.001 | 0.25 |
| random draw at training band frequencies | 0.006 | 0.25 |
| 2 hand-built spectral scalars + classifier | 0.0678 | 0.301 |
| 5-NN in a hand-built log-spectral space | 0.1721 | 0.379 |
| **best published reference** (log-spectral features, LR + GB per cell) | **0.2165** | 0.412 |
| CNN from scratch (53 of 90 min) | 0.1993 | 0.399 |
| **this solution** | **0.2762** | **0.457** |
| stated AI baseline (from the solver prompt, not the brief) | 0.28 | 0.460 |
| target | 0.32 | 0.490 |

**This is +0.060 on the best published reference, a 28% relative improvement,
and it lands essentially at parity with the stated 0.28 AI baseline
(0.2762 against 0.280, a difference of 0.004 in mean per-cell macro-F1).**

### Did selection inflate these numbers?
The ensemble subset and lambda were chosen by looking at cross-validated
scores, so the honest figure must include the cost of having chosen. Choosing
on one split and scoring on the other, both ways round:

| | score |
|---|---|
| chose top3, lambda=1.3 on the second split -> scored on the supplied split | 0.2715 |
| chose top3, lambda=1.0 on the supplied split -> scored on the second split | 0.2803 |
| **honest mean** | **0.2759** |

That is within 0.0003 of the reported 0.2762, because the choice is stable:
`top3` beats every other subset **at every value of lambda** on both splits,
and the lambda curve is flat from 0.7 to 1.3 (0.2736, 0.2762, 0.2761), so
lambda = 1.0 sits in the middle of a plateau rather than on a peak. An earlier
version of the pipeline, before the booster settings were made portable, showed
a real selection cost of about 0.010 (optimistic 0.2749 against honest 0.2647);
that gap closed when the configuration became stable.

### Honest assessment against the 0.32 target
**The evidence does not support a claim of 0.32 on unseen data, and I am not
going to make one.** Station-held-out cross-validation puts this solution at
0.276 +- 0.004 across splits. On top of that, the brief states the private
board has a standard deviation of about 0.019 for a solver near 0.20, so a
realistic private-board expectation is roughly **0.26-0.30**, centred a little
below 0.28. Reaching 0.32 would need mean per-cell macro-F1 of 0.490 against
the 0.457 measured here.

Three measurements say why the remaining gap is hard rather than merely
unworked:
1. **Removing the earthquake noise entirely is worth only +0.032.** Replacing a
   validation record's features with the mean over its own station's five
   records - which pools across evaluation records and is therefore forbidden
   in a submission, and was run once purely as a diagnostic - moved the score
   from 0.2118 to 0.2440. So the remaining error is not averaging noise; it is
   information the representation does not contain.
2. **Errors are overwhelmingly adjacent-band confusions**, which is exactly the
   non-uniqueness the brief describes: quite different profiles produce almost
   the same surface response.
3. **The two middle bands are intrinsically harder than the extremes**
   (per-band F1 roughly 0.28-0.42 against 0.44-0.66) and macro-F1 weights them
   equally, so a large share of what is left is unavailable from one
   amplitude-normalised record.

What I would try next, in order of expected value: a genuine parametric
inversion (fit a layered transfer model per record and use its fitted
velocities and thicknesses as features, rather than summary statistics of the
H/V curve); a multi-task neural model trained from scratch on the log
spectrogram with the four cells as correlated outputs and heavy augmentation,
ensembled with these feature models rather than replacing them; and a
per-cell-specialised feature search, since the 0-5 m cell behaves differently
from the other three and the one family built specifically for it failed.
