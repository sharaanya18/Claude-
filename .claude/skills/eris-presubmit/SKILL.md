---
name: eris-presubmit
description: The gate immediately before spending a submission credit on an Eris challenge: fresh clean run, validator, local score sanity (the platform check costs a credit), credit and progression check. Use after /eris-review passes.
---

# /eris-presubmit

1. Run the whole gate in one go: `bash .claude/scripts/eris_check.sh challenges/<slug>` (scaffold version, compliance scan, platform-command run + CSV validation, two-run determinism diff, half-rows independence test). Logs and OOF live in `reports/`, so a clean `working/` never loses them; confirm the submission was produced by the current `solution.py`.
2. `python3 .claude/scripts/validate_submission.py working/submission.csv dataset/public/sample_submission.csv [--test ... --json-cols ... --prob-cols ...]` passes (reload without NA coercion).
3. Prediction sanity: no constant outputs, class balance near train priors where expected, ranges valid, distribution similar to OOF (a bug check, never tuning). A shift of a few points versus the train priors is expected when the test has few independent groups; do not retune on it, only investigate if it is extreme (e.g. one class vanishes).
4. Local score sanity only (there is no free platform probe; every upload costs a credit): CV score plausible (not 0, not suspiciously perfect) and consistent with the previous accepted submission; do not tune on the public number.
5. Credits: per-challenge (6, +1 per 4 h) and global daily; is this the best use now, or would more iteration help? Is it meaningfully better than the last submission, with a documented reason?
6. Checklist: runs from a clean dir with no manual steps; no hardcoded absolute paths; no `input()`/debug traps; comments explain reasoning; source < 512 KB; only allowed imports.
Verdict line: "Ready to submit" or "Not ready: <list>". If anything fails, fix and restart this skill from step 1.
