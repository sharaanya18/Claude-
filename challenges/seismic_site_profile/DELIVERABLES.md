# Deliverables index - Seismic Site Profile

Workspace: `challenges/seismic_site_profile/`
Run the solver exactly as the platform does:

    python3 solution.py ./dataset/public ./working/submission.csv

## Required documents
| file | phase | contents |
|---|---|---|
| `platform_analysis.md` | 1 | reconnaissance of the Eris solver workspace; REUSE / DO NOT REUSE |
| `dataset_audit.md` | 2-3 | schema, the 239-station structure, label structure, waveform integrity, feature site-consistency (ICC), adversarial validation, the noise-ceiling diagnostic |
| `private_lb_reference_analysis.md` | 5 | **states that no reference artifact was supplied** and analyses the published reference scores instead |
| `reference_reproduction_plan.md` | 5b | tier reproduction and the four improvement axes |
| `metric.py` | 4 | exact grader, self-tested in closed form and against sklearn |
| `baseline_results.csv` | 6 | tier reproduction and the single-split ablation ladder |
| `experiment_results.csv` | 7-27 | every repeated-CV decision with per-split scores and the keep/reject call |
| `final_approach.md` | 32 | representation, models, validation, performance, generalisation, runtime, rule compliance |
| `submission_validation_report.txt` | 29 | row/id/profile checks and predicted band distribution |
| `working/gate/submission.csv` | 29 | the submission |
| `solution.py` | 28-30 | the single self-contained solver |

## Supporting code
| file | role |
|---|---|
| `features.py` | the twelve feature families (groups A-N; L and N measured and dropped) |
| `models.py` | model families used in experiments |
| `decode.py` | macro-F1 decode: prior matching, direct F1 coordinate ascent, structured joint-profile decode |
| `cv.py` | station-held-out CV harness, multi-view runner, repeated-split runner |
| `repsplits.py` | extra station-held-out splits, stratified like the supplied one |
| `icc.py` | between-station vs between-earthquake variance decomposition |
| `adversarial.py` | train-vs-test discriminator, calibrated against a within-train station split |
| `error_analysis.py` | per-band F1, confusion, per-station consistency, cross-cell correlation |
| `audit.py`, `audit_wf.py` | label/fold audit and waveform integrity audit |
| `exp0*.py`, `exp1*.py` | the experiments, in order |
| `build_solution.py` | assembles `solution.py` from the reviewed parts (development tool; the shipped file is plain source) |
| `gate.sh` | one-shot pre-submission gate: compliance scan, timed run, both validators, determinism, per-record independence |
| `make_validation_report.py` | writes `submission_validation_report.txt` |

## Reproducing the experiments
`cache_feats.py` and `cache_crops.py` write feature caches used by the
experiment scripts (the shipped `solution.py` does not read any cache - it
recomputes everything from the raw waveforms every run).
