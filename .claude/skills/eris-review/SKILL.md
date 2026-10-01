---
name: eris-review
description: Final pre-freeze review of an Eris solution by three independent agents in parallel (compliance, runtime/determinism, private-LB red team) plus the scripted checks. Use before presubmit and before spending a credit.
---

# /eris-review

Run the scripted gates first:
1. `python3 .claude/scripts/compliance_scan.py challenges/<slug>/solution.py` (no ERROR).
2. `python3 .claude/scripts/local_run.py challenges/<slug>` (runs the platform command, validates the CSV, reports runtime).
3. `python3 .claude/scripts/determinism_check.py challenges/<slug>/solution.py challenges/<slug>/dataset/public` (two runs, byte-identical) when runtime allows.

Then launch **in one message**: `eris-compliance-reviewer`, `eris-runtime-reviewer`, `eris-red-team`, each with `CHALLENGE.md`, `reports/contract.md`, `solution.py`, and the logs. Collect their reports.
Rules: any BLOCK or "NOT READY" stops the pipeline; fix and re-run the **whole** review (do not patch one item and assume the rest holds). Findings with no clear fix become reviewer questions with a plan under each
reading. Summarise PASS/FLAG/BLOCK counts and the red team's top three risks in `CHALLENGE_NOTES.md`.
