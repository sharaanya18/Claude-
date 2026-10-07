# Phase 1 — Platform / repository reconnaissance

## What the supplied repository is
`sharaanya18/Claude-` is **not** a code library for this challenge. It is a *solver playbook and
process harness* for Project Eris / Shipd challenges:

| Piece | Content | Verdict |
|---|---|---|
| `CLAUDE.md` (41 KB) | The binding rulebook: submission contract, compliance rules, determinism rules, domain playbooks, pre-submission checklist | **REUSE** — it is the contract this solution must satisfy |
| `.claude/skills/eris-*` | Workflow skills (`/eris-solve`, `/eris-contract`, `/eris-baseline`, …) | REUSE as process |
| `.claude/skills/eris-playbook/references/` | Distilled technique notes by problem family, incl. `families/retrieval-ranking-slates.md` | REUSE as hypotheses only |
| `.claude/scripts/` | `compliance_scan.py`, `validate_submission.py`, `determinism_check.py`, `local_run.py`, `make_groups.py` | **REUSE** — used as gates here |
| `.claude/agents/` | Reviewer agents (compliance, runtime, red-team) | REUSE as review |
| `challenges/anchorperm/`, `…/es_gl_matching/` etc. | Six **unrelated** past challenges (permutation/assignment, matching, forecasting) | **DO NOT REUSE** — different domains; their `solution.py` files solve other problems |

## Reusable patterns actually taken
- The argv contract and the `validate_submission` shape from `CLAUDE.md` §1/§5 (adapted: this
  challenge's targets are a JSON list and free text, so the validator also parses the JSON and
  asserts non-empty strings, since an empty cell reloads as `NaN`).
- The fixed-work-plan / no-clock-branching discipline from §3 — directly shaping the design
  (fixed folds, rounds, seeds, thresholds).
- Grouped CV that mirrors the hidden split (§4A) — here grouped **by Act**.
- The "model must do the learning" test (§2.1) — the reason the ranking and the edit decisions are
  gradient-boosted models trained in-script rather than hand-tuned rules.

## Explicitly NOT reused
- Any `solution.py` or `dev_*.py` from `challenges/*` — those are other tasks' pipelines
  (QAP solvers, Sinkhorn balancing, ridge blends) with nothing transferable here.
- `kaggle_gpu_run.py` — no GPU is needed; this solution is CPU-only.
- The guidebook's wall-clock training safeguard — `CLAUDE.md` §3 records that the platform's
  Deterministic Execution check rejects exactly that pattern. **Not used.**

## Determinism and runtime facts taken from the harness
- Fixed seeds, fixed rounds/folds, fixed thread counts; time only in log lines.
- No `try: import X except`, no `cuda.is_available()`, no `os.cpu_count()`.
- Source must stay plain, readable and < 512,000 bytes (this solution: ~70 KB).

## Leakage risks the harness warns about, checked here
- Fitting anything on test rows → avoided: both boosters, the IDF table and the two thresholds are
  fitted on the 512 training queries and the supplied corpus only.
- Index building over the corpus is **not** test fitting: the corpus is the shared reference the
  task supplies, identical for train and test, and every test row is scored independently of the
  other test rows.
