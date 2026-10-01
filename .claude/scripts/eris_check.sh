#!/usr/bin/env bash
# One-shot pre-review gate for a challenge workspace:  bash .claude/scripts/eris_check.sh challenges/<slug>
# Runs: scaffold version check, compliance scan, platform-style run + CSV validation, determinism (2 runs), half-rows independence test.
set -uo pipefail
dir="$(cd "${1:?usage: eris_check.sh challenges/<slug>}" && pwd)"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sol="$dir/solution.py"; pub="$dir/dataset/public"
status=0
step() { echo; echo "=== $1"; }
step "scaffold version"
have=$(grep -m1 -o 'eris-template-version: [0-9]*' "$sol" | grep -o '[0-9]*$' || true)
want=$(grep -m1 -o 'eris-template-version: [0-9]*' "$S/solution_template.py" | grep -o '[0-9]*$')
if [ "${have:-0}" -lt "$want" ]; then echo "WARN: solution.py scaffold is version ${have:-none} < template $want. Diff against .claude/scripts/solution_template.py (placeholder writes removed, requirements map added)."; fi
step "compliance scan";            python3 "$S/compliance_scan.py" "$sol" || status=1
step "run + validate (platform command)"; python3 "$S/local_run.py" "$dir" || status=1
step "determinism (two runs, byte diff)"; python3 "$S/determinism_check.py" "$sol" "$pub" || status=1
step "half-rows independence";     python3 "$S/half_rows_test.py" "$sol" "$pub" || status=1
echo; [ $status -eq 0 ] && echo "ALL GATES PASSED (still run the three reviewer agents)" || echo "SOME GATES FAILED"
exit $status
