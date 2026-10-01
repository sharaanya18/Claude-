---
name: eris-start
description: Intake for a new Eris challenge: scaffold the workspace, store the verbatim description, extract the contract, audit the data, name the single biggest risk. Use when the user pastes a challenge description and says to start it, or wants intake only without the full pipeline.
---

# /eris-start — intake

1. Read `CLAUDE.md`. Scaffold with `bash .claude/scripts/new_challenge.sh <slug>` unless the folder exists. Save the description **verbatim** to `CHALLENGE.md` (never summarise).
2. Run `/eris-contract` and fill `CHALLENGE_NOTES.md § Contract`.
3. If data exist, launch `eris-data-auditor` (read-only) and fill `§ Data audit`; if not, list the diagnostics that would be run.
4. State the **single biggest risk**, grounded in what the contract and audit just showed (e.g. "private split is project-disjoint but train has no project column → derive groups from content",
   "99th-percentile length exceeds the encoder limit", "the metric's worst-group term dominates"), and the plan for it, in one plain paragraph.
5. Hand off: `/eris-solve` for the full pipeline, or `/eris-baseline` if the plan is already clear.
