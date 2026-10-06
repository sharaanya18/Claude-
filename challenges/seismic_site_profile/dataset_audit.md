# Phase 2-3 - Dataset forensic audit

All numbers below are produced by `audit.py`, `audit_wf.py`, `icc.py` and
`adversarial.py` in this directory.

## Files and schema
| File | Rows | Verified |
|---|---|---|
| `train.csv` | 1,179 | columns `id,profile`; no duplicate ids |
| `test.csv` | 333 | column `id`; no duplicates; **zero overlap with train ids** |
| `sample_submission.csv` | 333 | columns `id,profile`; id order **identical** to `test.csv` |
| `folds.csv` | 1,179 | columns `id,site,fold`; id set exactly equals train's; no duplicates |
| `waveforms.npz` | 1,512 keys | key set exactly equals train ids + test ids |

Every profile parses to exactly four cells, every token is in `{v1,v2,v3,v4}`,
no malformed rows.

## The decisive structural fact: 239 stations, not 1,179 records
`folds.csv` gives 239 sites over 1,179 records (3-5 records each, median 5,
mean 4.93). **Every record at a station carries an identical profile** (0 of
239 sites has more than one distinct profile). The effective labelled sample
size is therefore **239**, not 1,179 - the single most important number for
model capacity and for why regularisation dominates everything here.

Each site lies entirely within one fold (0 sites span folds). Folds hold 47-48
sites / 232-238 records. So the supplied folds reproduce the evaluation
condition exactly: validation stations are never seen in training.

A random record-level split would put 4 of a station's 5 records in training
and 1 in validation *with the same answer*, which is the trap the brief warns
about. Site-held-out CV is used everywhere in this work.

## Label distribution and structure
Bands are close to balanced in every cell (records):

| cell | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 0-5 m | 309 | 262 | 319 | 289 |
| 5-10 m | 294 | 288 | 308 | 289 |
| 10-20 m | 299 | 285 | 291 | 304 |
| 20-30 m | 299 | 277 | 307 | 296 |

All four bands appear in all four cells in every fold, at both record and site
level, so **K = 4 everywhere and the blind reference is 0.25**. Useful
conversion: `cell_score = 0.25 + 0.75 * score`, so the published references map
to mean per-cell macro-F1 of 0.412 (0.2165), 0.460 (0.28 baseline) and
**0.490 for the 0.32 target**.

68 of the 256 possible profiles occur. The distribution is dominated by flat
columns: `v1|v1|v1|v1` (159 records / 32 sites) and `v4|v4|v4|v4` (134 / 27).
Adjacent cells correlate strongly on the numeric band: 0.76 (c0,c1), 0.85
(c1,c2), 0.88 (c2,c3), 0.71 (c0,c3). There is a dominant "overall column
stiffness" factor with per-cell deviations on top.

**The profile is NOT monotonic with depth** - only 52.8% of records are
non-decreasing, and down-column steps are near-symmetric (c0->c1: 22.5% up,
54.1% flat, 23.4% down). This is expected because each cell's bands are cut at
depth-specific velocity thresholds, so a uniform column lands in the same band
at every depth. **Any forced `cell4 >= ... >= cell1` constraint would be wrong
on ~47% of records** and is rejected. The usable structure is the correlation,
not an ordering.

## Waveforms
1,512 entries, each `(3, 3000)` float32, E/N/Z at 50 Hz for 60 s. No NaN, no
Inf, no zero or near-zero traces. `max|x| == 1.0` for every record, confirming
per-record amplitude normalisation, so absolute amplification is unavailable by
construction. The maximum sits on a horizontal component in 95.2% of records.
Mean per-component RMS is 0.128 / 0.129 / 0.083 (E/N/Z): the horizontals carry
~1.78x the vertical RMS, and that ratio survives normalisation, so
inter-component ratios are legitimate information.

**161 records (10.6%) carry trailing zero padding**, up to 1,207 samples
(24 s). Feature extraction strips it before windowing; otherwise the padded
records get a different effective spectral resolution and a spurious envelope.

No clipping (median 1 sample within 0.1% of unity - the normalising sample
itself). Records are essentially zero-mean (|mean|/RMS 0.024-0.051); windows
are demeaned anyway.

## Which features are site signature and which are earthquake?
`icc.py` decomposes each feature's variance into between-station and
within-station (i.e. across-earthquake) parts on train only. Median ICC and
median |Spearman| against site stiffness, by family:

| group | what it is | ICC | abs rho |
|---|---|---|---|
| A | H/V spectral ratio curve, 24 log bins | **0.72** | 0.28 |
| D | H/V peak structure scalars | **0.72** | 0.31 |
| K | coda-window H/V + early-late drift | 0.54 | 0.19 |
| I | trend-removed log spectra | 0.48 | 0.13 |
| B | normalised horizontal log spectrum | 0.46 | 0.31 |
| C | normalised vertical log spectrum | 0.37 | 0.13 |
| E | spectral shape stats + band ratios | 0.37 | 0.17 |
| J | cepstral coefficients | 0.35 | **0.05** |
| G | multi-window stability | 0.27 | 0.32 |
| H | time-domain shape / duration | 0.24 | 0.11 |
| F | horizontal polarisation (rotation-invariant) | 0.22 | 0.12 |

The site signal is concentrated in **H/V amplitude at low frequency**: the
strongest individual features are `hv_std`, `hv_lo` (mean log H/V below 2 Hz),
`hv_range`, `hv_a0` (peak amplitude) and the 0.44-2.4 Hz H/V bins. Time-domain
and polarisation families are dominated by the earthquake, as expected. The
cepstral family is site-stable but almost unrelated to the label on its own -
it must earn its place in combination or be dropped.

## Distribution shift (Phase 24)
A train-vs-test logistic discriminator on all 323 features reaches
**AUC 0.618**. The honest reference is a discriminator separating two disjoint
*sets of training stations* of the same size, which reaches **0.634** (mean of
3 draws). The evaluation records are therefore **no more distinguishable from
train than one group of training stations is from another** - there is no
detectable evaluation-set shift, and the station-held-out CV should track the
private board up to shard noise. No test-adaptive countermeasure is warranted
(and none would be permitted).

## Consequences for modelling
1. Capacity must be set for ~239 independent units. Strong regularisation and
   variance reduction (multi-seed, multi-view, ensembling) over tuned capacity.
2. Representation must suppress source/path: ratios between components and
   shape relative to a record's own smooth trend, not absolute spectra.
3. No monotonic depth constraint; exploit cross-cell correlation instead.
4. Model selection on site-held-out folds only, with fold spread reported.

## How much of the error is earthquake noise? (diagnostic, not a method)
A labelled diagnostic, run once and never part of a submission: predict each
validation record normally, then predict it again with its features replaced by
the mean over the ~5 records of its own station. That averaging pools across
evaluation records and is forbidden in a submission (and impossible there - the
evaluation records carry no station key); it is run only to size the headroom.

| validation features | OOF score | per-cell macro-F1 |
|---|---|---|
| one record (honest) | 0.2118 | 0.373 0.436 0.430 0.395 |
| mean over the station's records (diagnostic) | **0.2440** | 0.427 0.430 0.453 0.422 |

**Removing the single-record earthquake noise entirely is worth only +0.032.**
That reframes the whole problem: source/path contamination is *not* the
dominant limitation, the representation is. It follows that variance reduction
(multi-seed averaging, several views of one record, ensembling) can win at most
a few points and in practice much less, because sub-windows of one record share
the same earthquake. Effort therefore went into information content - new
feature families and model families with a better inductive bias - rather than
into noise averaging. This diagnostic is the main reason the work is shaped the
way it is.
