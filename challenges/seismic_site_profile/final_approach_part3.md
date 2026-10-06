## 6. Runtime and memory

Measured on the development box, which has **4 cores and 15 GB** - the target
is 10 cores and 62 GB, so these timings carry roughly a 2x margin in my favour.

| stage | time (4 cores) |
|---|---|
| feature extraction, 1,179 training records | 8.6 s |
| feature extraction, 333 evaluation records | 2.0 s |
| ordgb, five fold models (each also scoring the evaluation records) | 110 s |
| hgb | 117 s |
| et | 39 s |
| lr | 22 s |
| ldak | 3 s |
| ord | 1 s |
| pls | <1 s |
| blend, band-weight fitting, validation, write | <1 s |
| **total** | **RUNTIME_PLACEHOLDER** |

Peak resident memory is a few hundred MB: the whole waveform corpus is
54 MB of float32 and is read key by key, and the feature matrix is
1,512 x 355 doubles.

The work plan is fixed and cannot change with the machine or the clock: a
constant number of folds, views, seeds, trees and boosting iterations, thread
counts pinned to 10 before numpy is imported, and no branch anywhere that reads
elapsed time, hardware, or what happens to be importable. Elapsed time is
printed for logging and never read back.

Against the 90-minute limit this leaves a very large margin. That is a
deliberate choice rather than an accident: the noise-ceiling diagnostic showed
that the expensive routes to a better score (many seeds, many views) cannot buy
much here, so the budget was spent on representation and model structure
instead, and the leftover time is headroom against a slower machine.
