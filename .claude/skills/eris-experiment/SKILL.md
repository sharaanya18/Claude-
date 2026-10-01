---
name: eris-experiment
description: Run and log ONE controlled experiment on an Eris challenge: state the hypothesis, change one thing, evaluate on identical grouped splits with the exact metric, apply the noise-aware keep/revert rule, and update the experiment log and plateau detector. Use for every change after the baseline.
---

# /eris-experiment

1. **Hypothesis** (one sentence, with the mechanism and the evidence that motivates it). One change only.
2. **Run** on the *same* splits as the current best (same seeds). Capture: OOF mean ± std, per-fold scores, key slices, wall time, peak memory, and the paired per-fold difference versus the best.
3. **Decision rule** (noise-aware, replaces fixed deltas): keep if the paired gain exceeds ~1 standard error of the fold differences **or** is positive on most folds and split seeds, and the cost fits the runtime plan.
   Neutral (|gain| < 1 SE): keep only if it simplifies the code or reduces variance; otherwise revert. Negative: revert and log the likely reason tied to the data.
   Anything that touches a selection step (hyperparameters, blend, decode constants) additionally reports the nested/cross-fitted number.
4. **Log** a row in `working/experiment_log.csv`: `exp_id, hypothesis, change, cv_mean, cv_std, per_fold, nested_or_holdout, wall_s, kept, notes`.
5. **Leakage red flags**: CV near-perfect on a noisy task; CV std near zero; CV moves opposite to the public LB; gain disappears under grouped folds. Stop and audit.
6. **Plateau**: three consecutive experiments with gains below noise → `/eris-plateau`.
7. After each kept change, run the compliance scan; keep `solution.py` the single source of truth (no `solution_v2.py`).
