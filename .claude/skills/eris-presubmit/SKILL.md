---
name: eris-presubmit
description: The gate immediately before spending a submission credit on an Eris challenge: fresh clean run, validator, score sanity via the free CSV check, credit and progression check. Use after /eris-review passes.
---

# /eris-presubmit

1. Clean run: `rm -rf working && python3 .claude/scripts/local_run.py challenges/<slug>`; confirm the submission was produced by the current `solution.py`.
2. `python3 .claude/scripts/validate_submission.py working/submission.csv dataset/public/sample_submission.csv [--test ... --json-cols ... --prob-cols ...]` passes (reload without NA coercion).
3. Prediction sanity: no constant outputs, class balance near train priors where expected, ranges valid, distribution similar to OOF (a bug check, never tuning).
4. Free CSV upload first (no credit): score plausible (not 0, not suspiciously perfect); compare to the previous accepted submission; do not tune on it.
5. Credits: per-challenge (6, +1 per 4 h) and global daily; is this the best use now, or would more iteration help? Is it meaningfully better than the last submission, with a documented reason?
6. Checklist: runs from a clean dir with no manual steps; no hardcoded absolute paths; no `input()`/debug traps; comments explain reasoning; source < 512 KB; only allowed imports.
Verdict line: "Ready to submit" or "Not ready: <list>". If anything fails, fix and restart this skill from step 1.
