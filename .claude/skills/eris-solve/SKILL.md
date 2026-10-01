---
name: eris-solve
description: End-to-end S-tier pipeline for ANY Eris/Shipd challenge on unseen data. Use when the user gives a new challenge description (and dataset) and wants a compliant solution.py with the best private-leaderboard generalisation. Orchestrates contract, data audit, strategist plan, metric and validation, baseline, disciplined experiments, reviews, presubmit and lessons.
---

# /eris-solve — full pipeline

Goal: a **compliant**, **deterministic**, **correctly formatted** `solution.py` with the best *private*-leaderboard score the budget allows. Compliance and validity outrank score
(CLAUDE.md priority list). Never branch on the clock; never use test statistics; trust grouped CV over the public leaderboard.

## Phase 0 — Workspace
`bash .claude/scripts/new_challenge.sh <slug>` (or work in the cwd if it already holds `CHALLENGE.md` and `dataset/public/`). Paste the **verbatim** description into `CHALLENGE.md`. If data are not
present, continue with the contract and plan only and say what would be diagnosed. Keep `CHALLENGE_NOTES.md` current: it is the memory that survives context loss.

## Phase 1 — Contract (`/eris-contract`)
Decision unit, valid answer, invalid outputs, metric terms, bans, silences, compute, runtime, submission grammar. Name the single biggest risk in plain words.

## Phase 2 — Evidence (parallel)
- Agent `eris-data-auditor` → `reports/data_audit.md`.
- Agent `eris-metric-engineer` → `metric.py`, tests, `reports/metric_spec.md` (oracle decode check).
Run both in one message. Read their reports; resolve discrepancies (metric prose vs formula) explicitly.

## Phase 3 — Strategy
Agent `eris-strategist` (gives it contract + audit + metric spec paths) → `plan/eris_plan.md`. Challenge the plan: is the primary lean? Is there a frozen-probe yardstick? Is every lever tied to a
pattern id and checked against the bans? Pick primary + fallback + rejected options.

## Phase 4 — Validation
Agent `eris-validation-architect` → `validate.py`, `reports/split_audit.md` (hidden-split mirror, groups, nested helper, bias direction). Do not model before this exists.

## Phase 5 — Baseline (`/eris-baseline`)
Strong, honest, end-to-end baseline from `.claude/scripts/solution_template.py` with in-script validation and a single validated write at the end. Record exact-metric OOF mean ± std and slice scores. Run
`/eris-presubmit`'s cheap checks once so the pipeline is proven before optimisation.

## Phase 6 — Iterate (`/eris-experiment`, `/eris-error-analysis`, `/eris-implement`, `/eris-plateau`)
One change per experiment, on identical splits, logged. Order by expected private gain per hour: (1) metric/validation/ceiling → (2) coupling/structure/metric-aware target → (3) strongest affordable
representation → (4) loss/decode/augmentation to the metric and physics → (5) diversity ensemble → (6) seeds/snapshots and full refit → (7) small in-script HPO → (8) micro post-processing. Use
`eris-error-analyst` after each notable experiment; use `eris-red-team` once the approach stabilises.

## Phase 7 — Freeze
Fixed counts from CV (mean best epoch × 1.1), refit on 100%, determinism check, profile runtime. Then `/eris-review` (compliance + runtime + red-team in parallel).

## Phase 8 — Presubmit and submission (`/eris-presubmit`, `/eris-postsubmit`)
Free CSV check before spending a credit; submit the script; log the CV vs public gap; do not chase the public number.

## Phase 9 — Close (`/eris-close`, `/eris-pattern`)
Write the lessons that would have changed a decision; add transferable patterns to the library.

## Final answer format
Report: contract summary; validation design and bias direction; CV mean ± std (per fold/slice); fixed work plan and estimated A10G/CPU runtime; compliance self-audit; validator and determinism output;
what was NOT verified. Never present an LLM-rubric or CV number as a leaderboard score.
