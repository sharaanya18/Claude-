#!/usr/bin/env bash
# Scaffold a challenge workspace:  bash .claude/scripts/new_challenge.sh <slug>
# Creates challenges/<slug>/{CHALLENGE.md,CHALLENGE_NOTES.md,solution.py,dataset/public,working,reports}.
set -euo pipefail
slug="${1:?usage: new_challenge.sh <slug>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dir="$root/challenges/$slug"
if [ -e "$dir" ]; then echo "challenges/$slug already exists" >&2; exit 1; fi
mkdir -p "$dir/dataset/public" "$dir/working" "$dir/reports"
cp "$root/.claude/scripts/solution_template.py" "$dir/solution.py"
cat > "$dir/CHALLENGE.md" <<'EOF'
Paste the full challenge description here, verbatim (do not summarise).
EOF
cat > "$dir/CHALLENGE_NOTES.md" <<'EOF'
# Challenge notes: SLUG

## Status
- [ ] contract  - [ ] data audit  - [ ] strategist plan  - [ ] validation  - [ ] baseline  - [ ] experiments
- [ ] review (compliance, runtime, red-team)  - [ ] presubmit  - [ ] submitted  - [ ] closed (lessons written)

## Contract (decision unit, valid answer, metric terms, constraints, bans, compute, runtime)

## Data audit (shapes, groups, duplicates, label distribution, structure, ceiling diagnostics)

## Validation design (mirror of the hidden split, groups, bias direction of the proxy)

## Plan (primary, fallback, rejected options, fixed work plan)

## Experiment log (id, hypothesis, change, CV mean +- std, per-fold, runtime, kept?, notes)

## Error analysis

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
EOF
sed -i "s/SLUG/$slug/" "$dir/CHALLENGE_NOTES.md"
echo "created $dir"
echo "next: put data in $dir/dataset/public, paste the description into $dir/CHALLENGE.md, then run /eris-solve"
