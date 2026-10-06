# Phase 1 - Platform reconnaissance

The "repository" supplied for this task is the Eris/Shipd solver workspace at
`/home/user/Claude-`, not a seismology project. It contains no Seismic Site
Profile solver. Six unrelated challenge workspaces exist
(`anchorperm`, `complementary_entity_evidence`, `concession_forecast`,
`es_gl_matching`, `grounded_ip_formulation`, `shared_adam_repair`); none is a
waveform task, so nothing challenge-specific transfers.

## 1. Platform architecture
`CLAUDE.md` (41 KB) is the operative contract. A solution is a single
self-contained `solution.py` invoked as
`python3 solution.py <public_dir> <submission_out>`, which must train from raw
data every run and write one CSV. Four automated pre-submission checks gate it:
CSV Score Validation, Prompt Compliance, Held-out Answer Ingestion and
Deterministic Execution. Each challenge workspace is
`challenges/<slug>/{dataset/public, working, reports}` plus `solution.py`,
`metric.py` and report files - the layout mirrors the grader.

## 2. Reusable components
| Component | Why it applies here |
|---|---|
| `.claude/scripts/compliance_scan.py` | AST scan for the rejection patterns: wall-clock branching, env fallbacks, test-set fitting, hardcoded tuned constants. Runs automatically on every `solution*.py` edit via the PostToolUse hook. |
| `.claude/scripts/validate_submission.py` | Enforces header/order/row-count/id-set/no-empty-cell against `sample_submission.csv`, with `--allowed-col`. Covers the outright-rejection list in this brief. |
| `.claude/scripts/determinism_check.py` | Runs the platform command twice and byte-diffs the CSV. The determinism requirement here is explicit. |
| `.claude/scripts/half_rows_test.py` | **The most relevant utility.** Re-runs on a subset of test rows and checks predictions are unchanged - a direct test of this challenge's "predict each evaluation record on its own; do not pool across evaluation records" rule. |
| `.claude/scripts/local_run.py`, `eris_check.sh` | One-shot gate: platform-form run + validation + determinism + independence. |
| `.claude/scripts/solution_template.py` | The compliant scaffold: argv contract, seeding, requirements map in the docstring, fixed work plan. |

## 3. Relevant competition patterns (methodology only)
From `CLAUDE.md` §4A and the other workspaces' structure, the transferable
discipline is: mirror the hidden split in CV; implement the official metric
exactly and optimise it rather than a proxy loss; one change per experiment
with a logged CV table; accept a change only when the gain exceeds fold noise;
prefer regularised diverse ensembles over one tuned model; fit blend weights
and any decode threshold on out-of-fold predictions only; keep a fixed work
plan with no clock branching.

## 4. Potentially useful utilities
`cv_driver_template.py` and `make_groups.py` (grouped-CV scaffolding - here the
groups are supplied in `folds.csv`, so only the driver shape is useful).
`test_schema_audit.py` for the submission grammar.

## 5. Irrelevant components
Everything GPU-shaped: `kaggle_gpu_run.py`, the A10G/AMP/bf16 guidance, the
transformer and timm recipes in `CLAUDE.md` §6.2/§6.3, HF weight downloading
and `HF_HOME` plumbing. This challenge is CPU-only and from-scratch, and
`CLAUDE.md` §2.4 states the description overrides those defaults. The other
six challenges' feature code is domain-specific and not reused.

## 6. Leakage risks specific to this challenge
1. **Record-level CV would be catastrophic.** Each station contributes ~5
   records that share one answer, so a random split puts a station's answer in
   both halves. `folds.csv` holds out whole stations; it is mandatory.
2. **Any statistic fitted on test rows** (scaler, PCA, vocabulary, quantiles)
   is banned - fit on train, `.transform` test.
3. **Cross-evaluation-record pooling** is banned by the brief even though the
   data would permit it: no clustering test records into stations, no group
   votes, no rank-normalising over the test set. Thresholds must be absolute
   values derived from training.
4. `id`, row order and file position are banned as signals.
5. Decode/threshold tuning on the same OOF used for selection double-dips;
   needs a nested check.

## 7. Reproducibility requirements
Seed `random`, `numpy` and every estimator's `random_state`; set
`PYTHONHASHSEED`; pin BLAS thread counts *before* importing numpy; fixed
numbers of folds, trees, iterations and crops; never branch on elapsed time
(`CLAUDE.md` §3 - this is the check that has actually blocked submissions, and
it overrides the guidebook's own time-safeguard advice).

## 8. CPU / runtime considerations
Target 10 cores / 62 GB / 90 min, and the brief forbids GPU. Development here
runs on 4 cores / 15 GB, so local timings are conservative - roughly a 2x
margin in my favour. Waveform feature extraction is 6 ms/record (~9 s for all
1,512), so the budget is dominated by model fitting, which must stay well
inside the limit with >=30% headroom.

## 9. Recommended platform components for this challenge
`solution_template.py` as the scaffold; `compliance_scan.py` +
`validate_submission.py` + `determinism_check.py` + `half_rows_test.py` as the
pre-submission gate; the §4A experiment discipline as the working method.

## REUSE
- The `solution.py` argv/output contract and the compliant scaffold.
- `compliance_scan.py`, `validate_submission.py`, `determinism_check.py`,
  `half_rows_test.py`, `local_run.py`, `eris_check.sh`.
- The challenge-workspace layout and report skeletons.
- The §4A validation/experiment discipline and the §3 determinism rules.
- The exact-metric-first principle.

## DO NOT REUSE
- Any other challenge's features, models or solution code.
- All GPU/A10G/AMP guidance and HF/timm pretrained-weight plumbing (forbidden:
  from-scratch, CPU-only).
- `CLAUDE.md`'s default "assume one A10G" and `device="cuda"` (overridden by
  the description per §2.4).
- The guidebook's wall-clock training safeguard (rejected by the determinism
  checker).
- Record-level or random CV of any kind.
