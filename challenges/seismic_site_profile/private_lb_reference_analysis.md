# Phase 5 - Reference analysis

## The supplied artifact does not exist

Phase 5 of the brief asks me to inspect "the separate private-LB #1 reference
solution/artifact supplied with the workspace". **No such artifact was
supplied.** What was provided is:

* the task prompt (`reports/solver_prompt_verbatim.txt`), and
* one Google Drive archive, `public (5).zip` (36,075,312 bytes), which
  contains exactly five files: `train.csv`, `test.csv`,
  `sample_submission.csv`, `waveforms.npz`, `folds.csv`.

The workspace at `/home/user/Claude-` contains six unrelated Eris challenge
workspaces and no seismic solver, write-up, notebook or score artifact. I
searched the repository, the uploads directory and the extracted archive.

I am not going to invent a forensic analysis of a file I do not have. What
follows is what can legitimately be read off the **reference scores quoted in
the challenge brief itself**, which is the only reference information actually
available.

## What the quoted reference scores tell us

All on the real grader, evaluation set, whole stations held out, nothing tuned
on it. Converted to mean per-cell macro-F1 via `cell = 0.25 + 0.75 * score`:

| Reference approach | score | mean per-cell macro-F1 |
|---|---|---|
| 2 hand-built spectral scalars (centroid, H/V peak freq) + classifier | 0.0678 | 0.301 |
| 5-NN copy in hand-built log-spectral feature space | 0.1721 | 0.379 |
| same features, logistic regression + gradient boosting per cell | **0.2165** | 0.412 |
| CNN from scratch on log spectrograms + log H/V, 53 of 90 min | 0.1993 | 0.399 |
| constant profile / most-common band / sample_submission | 0.001 | 0.25 |
| random draw at training band frequencies (40 seeds) | 0.006 | 0.25 |
| **"current AI baseline"** (from the solver prompt, not the brief) | ~0.28 | 0.460 |
| **target** | 0.32 | 0.490 |

Five things follow, and they shaped this solution:

1. **Two scalars are worth 0.068, a full log-spectral curve is worth 0.17-0.22.**
   The site information is distributed across the whole frequency axis, not in
   the peak frequency alone. A binned curve, not a handful of summary scalars,
   must be the backbone. My ICC analysis independently confirms this: the
   strongest single features are H/V *amplitude* measures across the
   0.44-2.4 Hz band (`hv_std`, `hv_lo`, `hv_range`, `hv_a0`), not `hv_f0`.

2. **5-NN at 0.1721 is remarkably close to fitted models at 0.2165.** A pure
   similarity lookup in the right space captures most of the available signal,
   which says the decision surface is smooth and low-complexity. That argues
   for strong regularisation and against high-capacity models - consistent
   with 239 independent labelled stations.

3. **A from-scratch CNN (0.1993) loses to hand-built features (0.2165)** and
   burns 59% of the runtime doing it. With 239 effective units this is the
   expected outcome, and the brief says so explicitly. A deep model is
   therefore not the route; it is at best an ensemble member, and on a CPU-only
   90-minute budget it is not worth its cost.

4. **The gap from 0.2165 to the ~0.28 baseline is about 0.048 in score, i.e.
   0.036 in mean per-cell macro-F1.** That is not a different paradigm - it is
   the size of gain that better source/path cancellation, variance reduction
   (multi-seed, multi-view, ensembling) and a macro-F1-aware decode plausibly
   deliver on top of the same representation family.

5. **The floor structure confirms the metric reading.** That a constant profile
   and the most-common band both land at exactly 0.001 while a
   frequency-matched random draw lands at 0.006 is only consistent with
   macro-F1 averaged over the bands present in gold, with K = 4 in all four
   cells. `metric.py` reproduces all of these, plus the brief's
   1.0000 -> 0.9965 single-bad-row figure in closed form.

## What is NOT inferable

The brief names no feature list, model hyperparameters, window scheme,
smoothing, ensemble structure or post-processing for any reference. Anything I
asserted about those would be fabrication. In particular I cannot say which
parts of a top solution exploit training-specific patterns, because I have not
seen one.

## Consequence for this work

Phase 5 is answered as "no reference artifact available"; the reproduction plan
in `reference_reproduction_plan.md` is therefore built from the *tiers* above
rather than from reference code, and every design choice is justified by my own
station-held-out CV rather than by imitation. The three reference tiers were
re-implemented independently and land in the right order and roughly the right
place (see `baseline_results.csv`), which is the only sense in which the
reference has been "reproduced".
