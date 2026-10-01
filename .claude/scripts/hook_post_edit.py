#!/usr/bin/env python3
"""PostToolUse hook: after Edit/Write/MultiEdit of a solution file, run the compliance scan.

Reads the hook JSON from stdin. Exit 2 with the findings on stderr when the scan has ERRORs, which feeds
them back to Claude; exit 0 (silent) otherwise. Never blocks edits to other files.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    path = (payload.get("tool_input") or {}).get("file_path", "")
    name = Path(path).name
    if not path.endswith(".py") or not (name == "solution.py" or name.startswith("solution_") or "/challenges/" in path and name.startswith("solution")):
        return 0
    if not Path(path).exists():
        return 0
    r = subprocess.run([sys.executable, str(HERE / "compliance_scan.py"), path], capture_output=True, text=True)
    if r.returncode == 2:
        sys.stderr.write("compliance_scan found ERRORs in the file you just edited. Fix them before continuing "
                         "(rules: CLAUDE.md §2-§3):\n" + r.stdout)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
