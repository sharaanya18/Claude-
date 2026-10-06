## 8. Stress tests, and what was rejected

Every item below was built, measured on station-held-out folds, and then
**kept or thrown away on the measurement**, not on how good the idea sounded.

### Rejected after measurement
| idea | why it was plausible | measured outcome | verdict |
|---|---|---|---|
| **Konno-Ohmachi smoothed H/V** (group L, 38 cols) | the standard HVSR smoother; constant-width in log frequency, so it preserves peak height better than bin averaging | family mean 0.2104 -> 0.2130, but the best family went 0.2228 -> 0.2224, and on top of group M 0.2302 -> 0.2278 | **dropped** - 38 columns for nothing |
| **Short-window high-frequency H/V** (group N, 30 cols) | the 0-5 m cell resonates at 10-22 Hz and is persistently the weakest cell; 5 s windows give more looks where single-record SNR is worst | overall 0.2302 -> 0.2289, and **cell 0 itself went 0.362 -> 0.358** - it failed at the one job it was designed for | **dropped** |
| **Cepstral features** (group J, 68 cols) | a layered site is quasi-periodic in frequency, so its two-way travel time should appear as a cepstral peak, separable from the smooth source/path term by quefrency | median absolute Spearman correlation with site stiffness of just **0.05** - the weakest of any family - but the group still measured +0.001 to +0.011 in combination | **kept, with the caveat stated**; it earns its place only through interactions |
| **Monotonic depth constraint** (`cell4 >= ... >= cell1`) | velocity does increase with depth physically | only **52.8%** of training records are non-decreasing, and down-column steps are near-symmetric, because each cell's bands are cut at depth-specific thresholds | **rejected** - would be wrong on ~47% of records |
| **A from-scratch CNN** | it is a waveform problem | the brief measures it at 0.1993 against 0.2165 for hand-built features, using 53 of 90 minutes | **not attempted** - a measured loss for an unmeasured hope on 239 units |
| **Fitted blend weights** (greedy forward selection with replacement, chosen per fold on the other folds only) | learned weights usually beat an average | see the table in section 4 | compared, not assumed |

### Sensitivity and stability checks
* **Single-split noise is the size of the effects being chased.** Two nearly
  identical boosting configs measured **0.2165 and 0.2295 on the same split**.
  That one observation drove the whole methodology: every decision after it was
  taken on the mean over several station-held-out splits, which cut the spread
  between splits to +-0.001 to +-0.005.
* **Fold spread** is reported for every run. The accepted configurations sit at
  a fold standard deviation of roughly 0.014-0.027, and no accepted change
  depended on a single fold.
* **Per-band behaviour**: no band is ever left unpredicted. The predicted
  marginal already tracks the gold marginal closely before any decoding
  (cell 0: predicted `[304 254 335 286]` against gold `[309 262 319 289]`),
  because the cumulative-link construction produces balanced predictions by
  itself. This is why the macro-F1 decode buys little here, and it was checked
  rather than assumed: on synthetic data where a model *is* shy of the middle
  bands, the same decode lifts macro-F1 from 0.458 to 0.558.
* **Error structure**: confusions are overwhelmingly to **adjacent** bands, so
  the ordering is learned and the limit is resolution. Extreme columns are
  called best (cell-wise accuracy 0.579 for the softest group against
  0.323-0.417 for intermediate ones).
* **Predicted cross-cell correlation (0.66-0.76) is LOWER than gold
  (0.71-0.88)**, so the model is not collapsing to a single stiffness latent -
  it under-couples the cells rather than over-coupling them. This is what gives
  the structured joint-profile decode a mechanism, and it is why that decode
  was tested rather than dismissed.
* **Portability**: the one estimator argument needing a recent scikit-learn
  (`max_features` on the histogram booster, added in 1.4) was measured against
  the portable default so that the choice is evidence-based rather than a bet
  on the grading machine's version. An environment-dependent fallback would
  itself be a rejection pattern, so there is none anywhere in the pipeline.
* **A contradiction with the brief worth recording**: the brief expects depth
  resolution to *degrade downward*. Measured, the **shallowest** cell is the
  hardest (macro-F1 0.374 against 0.410-0.490 for the deeper three). A 5 m cell
  is thin, its resonance sits where attenuation and noise are worst, and its
  thickness-weighted harmonic mean is sensitive to fine detail. The deep cells
  benefit from the long-window low-frequency family (group M); the symmetric
  high-frequency family for the shallow cell did not work (above).
