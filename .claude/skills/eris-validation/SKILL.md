---
name: eris-validation
description: Design and verify the CV harness for an Eris challenge so it predicts the PRIVATE score: mirror the hidden split, derive groups, exact metric, nested checks for every selection step, bias direction. Delegates to eris-validation-architect. Use before any modelling and whenever CV and the public leaderboard disagree.
---

# /eris-validation

Launch `eris-validation-architect` with the contract, data audit and metric spec. Required outputs: `validate.py` (or equivalent), `reports/split_audit.md`.
Golden rules: all preprocessing inside each fold; fit scalers/vocab/TF-IDF/encoders on the training fold only; ≥ 5 folds (2–3 split seeds when noisy); group by derived units; the CV metric is the exact
challenge metric; every post-hoc step has its own nested check; state the bias direction; one sanity holdout untouched until the end.
When CV disagrees with the public LB: first look for a split mismatch (unit, magnitude of shift, fold size) and selection double-dipping; trust CV unless a mismatch is found. Re-run
`make_groups.py --audit-fold` on the fold column.
