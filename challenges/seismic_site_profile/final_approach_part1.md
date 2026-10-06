# Final approach - Seismic Site Profile

## 1. Final representation

Each record is one `(3, 3000)` array: east, north and vertical acceleration,
60 s at 50 Hz, scaled so the largest absolute sample is 1. Trailing zero
padding (present in 10.6% of records, up to 24 s) is stripped before anything
else, otherwise the padded records get a different effective spectral
resolution and a false envelope.

The representation is built around one fact: a single record is
*source x path x site*, and only the site term is wanted. Two transformations
cancel most of the rest without needing a reference station, which is not
available here:

* **component ratios** - the three components of one record share the source
  and very nearly the path, so the horizontal-to-vertical spectral ratio
  (H/V, the standard Nakamura estimator) largely divides them out and leaves
  the site response;
* **shape relative to a record's own smooth trend** - magnitude, distance and
  high-frequency attenuation set a smooth spectral envelope, so subtracting a
  low-order polynomial fit in log-frequency keeps the resonance structure and
  discards the broad source shape.

Power spectra come from Hann-tapered, demeaned 20.5 s windows (1024 samples)
stepped by 5.1 s, so a full record gives 8 looks and a padded one 4, and the
per-window spectra are combined with a **median** rather than a mean so a
single noisy window cannot carry a feature.

Twelve families are used (355 columns). They were chosen before modelling from
the physics, then kept or dropped on station-held-out cross-validation:

| group | columns | what it measures | why it should carry site information |
|---|---|---|---|
| A | 24 | log H/V in 24 log-spaced bins, 0.44-22 Hz | the classic site-response observable; peak near f0 ~ Vs/(4H) |
| B | 24 | normalised horizontal log-spectral shape | amplification band, with overall level removed |
| C | 24 | normalised vertical log-spectral shape | the source/path reference the ratio divides by |
| D | 13 | H/V peak structure: f0, peak amplitude and width, second peak, band means | f0 fixes the two-way travel time; amplitude fixes the impedance contrast |
| E | 30 | spectral centroid/spread/entropy/rolloff/slopes and six band-energy ratios per component | stiff rock keeps high-frequency energy that soft ground attenuates |
| F | 12 | horizontal polarisation per band, **eigenvalues only** | rotation-invariant: E/N orientation is arbitrary and must not be learned |
| G | 8 | stability of H/V and of f0 across windows | separates a steady site resonance from a transient arrival |
| H | 32 | time-domain shape: duration, coda decay, crest, kurtosis, zero crossings, H/V RMS ratio | soft sites ring longer; the H/V RMS ratio survives amplitude normalisation |
| I | 48 | trend-removed log H and log H/V | resonance with the smooth source/path envelope regressed out |
| J | 68 | cepstrum of the trend-removed log spectrum, lags 0.02-0.60 s | a layered site is quasi-periodic in frequency, so its travel time appears as a cepstral lag |
| K | 40 | H/V of the **late coda** plus the early-to-late drift | the late coda approaches a diffuse field, where H/V is tied most directly to the site |
| M | 32 | **41 s windows**, H/V and horizontal shape over 0.26-4.8 Hz | the 20-30 m cell resonates lowest, and a 20 s window barely resolves it |

Everything is computed from one record in isolation. Nothing uses the id, the
row order, a record's position in a file, or any other record.

Two families were built, measured and **dropped**:
* **L** (32-bin Konno-Ohmachi-smoothed H/V): the textbook HVSR smoother, but on
  three splits it was worth +0.003 on the mean over three families and
  **-0.002** for the best family, so it added 38 columns for nothing.
* **N** (short-window 7.5-24 Hz H/V, aimed at the 0-5 m cell, which is
  persistently the weakest): see section 4 for the measured result.

Group **J** earns its place only in combination: on its own its features have
a median absolute Spearman correlation with site stiffness of just 0.05.

## 3. Validation

`folds.csv` is used exactly as supplied and is the only split used for model
selection reporting. It holds out **whole stations**: 239 stations over 1,179
records, 3-5 records per station, 47-48 stations per fold, and **no station
spans two folds**. Every record at a station carries an identical profile, so
the effective labelled sample is **239, not 1,179** - the number that governs
how much model capacity the data can support.

A record-level split would be worthless here: it would put four of a station's
five records in training and the fifth in validation with the same answer.
Measured directly, that mistake inflates the score far above anything the
leaderboard would return.

Because one 5-fold split over 239 stations leaves an out-of-fold standard error
of roughly the size of the gains being chased (two near-identical configs
measured 0.2165 and 0.2295 on the same split), **every decision in this work
was taken on the mean over several station-held-out splits**. The extra splits
(`repsplits.py`) are generated the same way as the supplied one: whole stations
held out, stratified on the station's own profile so all four bands appear in
every fold of every cell, with balanced fold sizes. Repeated CV brought the
spread between splits down to +-0.001 to +-0.005.

Scoring uses `metric.py`, an exact reimplementation: macro-F1 over the bands
present in gold within each depth cell, averaged over the four cells, expressed
as skill over the record-blind reference. It reproduces every published
reference point, including the brief's `1.0000 -> 0.9965` figure for one
unrecognisable row, in closed form (that figure pins both the
macro-over-gold-bands reading and K = 4; the evaluation cell in question simply
has a 48-member band). All four bands are present in all four cells, so
**blind = 0.25** and `cell_score = 0.25 + 0.75 * score`. The published
references therefore correspond to mean per-cell macro-F1 of 0.412 (0.2165),
0.460 (the stated 0.28 baseline) and 0.490 (the 0.32 target).

Blend weights and the decode weights are fitted **only on training-fold
out-of-fold material**: for every held-out fold they are re-fitted on the other
folds, so no held-out station's label ever informs its own prediction.

## 7. Rule compliance

| rule | how it is honoured |
|---|---|
| from scratch, no pretrained weights of any kind | only numpy/pandas/scikit-learn estimators, each constructed untrained and fitted on the supplied records inside `solution.py`. No checkpoint, embedding, tokenizer or downloaded artifact exists anywhere in the pipeline. |
| no external data or seismological corpora | the only inputs opened are `train.csv`, `test.csv`, `sample_submission.csv`, `folds.csv` and `waveforms.npz` from the supplied directory. |
| no looking up evaluation records externally | no network call of any kind; the profile is inferred from the waveform. |
| no id, row order or file position as a signal | ids are used only as `waveforms.npz` keys and to write the output column. No feature depends on a record's position, and the pipeline is invariant to row order. |
| predict each evaluation record on its own | every feature comes from one record in isolation; the multiple views of a record are sub-windows of that same record. No scaler, PCA, quantile, cluster or threshold is fitted on evaluation data - the band weights come from training out-of-fold predictions and are a fixed 4x4 constant at inference. Verified mechanically with the half-rows independence test. |
| no pooling/voting/grouping across evaluation records | none exists. The site-mean experiment that would violate this was run once as a labelled diagnostic to size the noise ceiling and is not part of the solution. |
| no pseudo-labelling the evaluation set | no evaluation prediction is ever fed back into training. |
| CPU only, 10 cores, 62 GB, 90 min | no GPU code path, no torch; BLAS thread counts pinned to 10 before numpy is imported; peak memory is a few hundred MB. |
| no packages beyond the preinstalled environment | numpy, pandas, scikit-learn. |
| submission grammar | exactly `id,profile`, all 333 test ids in `test.csv` order, four lowercase bands joined by `|`; `validate_submission()` raises before any file is written. |
| determinism | all seeds fixed, threads pinned, fixed number of folds/views/seeds/trees/iterations. No branch anywhere reads the clock, the hardware or what happens to be importable; elapsed time is printed and never read back. |

One judgement call is worth stating plainly. The brief forbids forcing a
monotonic depth profile only implicitly, but the training data settles it:
**only 52.8% of records are non-decreasing with depth**, and down-column steps
are near-symmetric (cell0 -> cell1: 22.5% up, 54.1% flat, 23.4% down), because
each cell's bands are cut at depth-specific velocity thresholds. A
`cell4 >= cell3 >= cell2 >= cell1` constraint would be wrong on roughly 47% of
records and was rejected. The cross-cell structure is used only through the
correlation (0.71-0.88) and, optionally, the empirical joint distribution over
the 68 observed profiles - both taken from training folds only.
