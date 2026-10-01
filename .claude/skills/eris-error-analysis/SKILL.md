---
name: eris-error-analysis
description: After an experiment, analyse real out-of-fold predictions for evidence-backed failure patterns by slice and by pipeline stage (coverage, ranking, decode) and return exactly three ranked hypotheses. Delegates to eris-error-analyst. Use when choosing the next experiment.
---

# /eris-error-analysis

Save OOF predictions per experiment (`reports/oof/<exp_id>.*`). Launch `eris-error-analyst` with the OOF path, `reports/split_audit.md`, the metric (`metric.py`) and the current plan.
Required output: slices with numbers and sizes, the coverage/ranking/decode split for structured pipelines, oracle analysis, and exactly three ranked hypotheses
(problem observed → concrete change → expected gain → time). Reject advice that is not tied to a measured row of evidence. Append to `CHALLENGE_NOTES.md § Error analysis`, then `/eris-implement` the top one.
