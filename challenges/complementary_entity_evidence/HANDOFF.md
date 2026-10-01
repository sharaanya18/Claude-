# Handoff: Complementary Entity Evidence Selection (read this first in a new session)

Branch: `ccr-88f20c83-nwrhm0`. Read `/home/user/Claude-/CLAUDE.md` (rules, §1A platform facts, §12 system map), then this file.

## State
- `CHALLENGE.md` (verbatim description incl. grader), `reports/data_audit.md` (real-data audit), `reports/metric_spec.md`, `reports/eris_plan.md` (blind plan).
- `metric.py` + `tests_metric.py` (27 tests pass): exact metric, expected-union decode (closed form), MC decode, gold-oracle check.
- `solution.py` (draft): DeBERTa-v3-large role tagger, masked BCE over 73 roles, seeds as extra labelled sentences, labels merged across a sentence's appearances, cohort-grouped 5-fold CV chooses epochs + temperature, refit on 100% (2 seeds), expected-union pair decode, validated write. Runs end to end on the real data on CPU with a tiny model. MODEL_REVISION is still "main": pin a commit sha before submitting.
- `dev_run.py`: dev harness (not part of the submission): `python3 dev_run.py EXP KEY=VAL ...` runs the CV with patched constants and saves OOF logits to `reports/oof/EXP.npz`.
- Dataset is NOT in git. The user's zip must be attached again in the new session: unzip into `dataset/public/{train.csv,train_targets.csv,test.csv,sample_submission.csv}`.
- Key data facts: random pair 0.513 (type priors ≈ random); `oracle_rank` larger = first; 44% of candidates have no new role; 0 label contradictions across repeated sentences; 256 cohorts; sentences ≈ 11 words.
- CPU dev CV (electra-small, 5 folds, 8 epochs, lr 2e-4) was running when this session ended; result not recorded.

## Next steps
1. Start of session: check `echo ${KAGGLE_API_TOKEN:+set}` (never print it). If set: `pip install -U kaggle` (needs CLI >= 1.8), then `python3 .claude/scripts/kaggle_gpu_run.py data challenges/complementary_entity_evidence` (private dataset; confirm data terms with the user first), then `... run challenges/complementary_entity_evidence dev_run.py cvlarge MODEL_NAME=microsoft/deberta-v3-large DEVICE=cuda N_FOLDS=5 MAX_EPOCHS=4 BATCH_SIZE=16` and `... fetch ...` (outputs in `kaggle_out/reports/oof/`). The runner is untested against the live API: expect small fixes.
2. Experiments (one change each, identical cohort folds, exact metric, paired fold differences): large vs base encoder; anchor masking ablation (typed mask) to curb anchor-name memorisation; with/without seed sentences as labelled examples; per-role bias; second encoder family for diversity; pairwise redundancy term only if error analysis shows near-paraphrase pairs mispicked.
3. Error analysis on OOF with `eris-error-analyst` (coverage = roles reachable, ranking, decode). Compare closed-form vs pointwise top-2 decode on real OOF.
4. Freeze: fixed epochs/seeds from CV, pin revisions, profile A10G runtime (budget 90 min; target ≤ 50), `bash .claude/scripts/eris_check.sh challenges/complementary_entity_evidence` (use `--help`-free defaults; it needs a GPU for the real model), then the three reviewer agents, presubmit, submit.
5. Remember: the platform runs on an A10G; Kaggle T4/P100 numbers are for relative comparisons only.

## Reviewer questions to settle (from the plan)
Anchor masking (Q4), conditioning of the decode on the retention rule (Q5), whether expected-union decoding counts as acceptable metric-aware decoding (the description does not forbid it).
