---
name: eris-implement
description: Implement one specific, concrete improvement in solution.py (from error analysis, plateau advice or the plan): edit in place, keep the scaffold, rerun the compliance scan and the grouped CV, decide keep or revert with the noise-aware rule, and log it. Use for each hypothesis.
---

# /eris-implement

- Edit `solution.py` in place; no parallel versions. Keep seeds, I/O, fixed-plan constants and validators unchanged; touch model/feature/decode logic only.
- The change must be self-contained in the script, original and commented (why, not what). No hardcoded tuned constants: derive them in-script from train-only evidence.
- Run `python3 .claude/scripts/compliance_scan.py solution.py`; fix every ERROR and justify every WARN in a comment or reviewer question.
- Run the grouped CV (`validate.py`) on identical splits; apply `/eris-experiment`'s decision rule; revert if not kept so `solution.py` stays at the best known state.
- Log the row and update `CHALLENGE_NOTES.md § Experiment log`.
