#!/usr/bin/env python3
"""SessionStart hook: print a compact orientation (stdout becomes session context)."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
lines = ["Eris/Shipd workspace. Read CLAUDE.md first. Start a challenge with /eris-solve (full pipeline) or "
         "/eris-start (intake only). Scripts: .claude/scripts (compliance_scan, validate_submission, make_groups, "
         "determinism_check, local_run, new_challenge.sh). Top rules: argv contract, real in-script training, "
         "no test statistics, fixed work plan (never branch on the clock), mirror the hidden split in CV, "
         "decode for the exact metric."]
ch = root / "challenges"
if ch.is_dir():
    rows = []
    for d in sorted(p for p in ch.iterdir() if p.is_dir()):
        notes = d / "CHALLENGE_NOTES.md"
        if notes.exists():
            txt = notes.read_text(encoding="utf-8", errors="replace")
            done = len(re.findall(r"- \[x\]", txt, re.I))
            todo = len(re.findall(r"- \[ \]", txt))
            rows.append(f"  - {d.name}: {done} done / {todo} open status items")
    if rows:
        lines.append("Challenges in workspace:\n" + "\n".join(rows) + "\nUse /eris-resume <slug> to continue one.")
print("\n".join(lines))
