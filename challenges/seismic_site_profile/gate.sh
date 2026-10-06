#!/usr/bin/env bash
# Full pre-submission gate for this challenge.
#   bash gate.sh            run everything
#   bash gate.sh quick      skip the two repeat runs (determinism, half-rows)
set -uo pipefail
cd "$(dirname "$0")"
S=../../.claude/scripts
status=0
step() { echo; echo "=== $1"; }

step "static compliance scan"
python3 "$S/compliance_scan.py" solution.py || status=1

step "platform-form run (timed)"
rm -rf working/gate && mkdir -p working/gate
start=$(date +%s)
python3 -u solution.py ./dataset/public ./working/gate/submission.csv > logs/gate_run.log 2>&1
rc=$?
end=$(date +%s)
secs=$((end-start))
echo "exit code $rc, wall time ${secs}s ($((secs/60))m)"
[ $rc -eq 0 ] || { echo "RUN FAILED"; tail -25 logs/gate_run.log; status=1; }
grep -E "OOF|band weights" logs/gate_run.log | head -20

step "submission validation (repo validator)"
python3 "$S/validate_submission.py" working/gate/submission.csv dataset/public/sample_submission.csv \
    --test dataset/public/test.csv --id-col id || status=1

step "submission validation report (challenge-specific)"
python3 make_validation_report.py working/gate/submission.csv dataset/public \
    submission_validation_report.txt "${secs}s on 4 cores (dev box)" || status=1

if [ "${1:-full}" != "quick" ]; then
  step "determinism: two full runs, byte-compare"
  python3 "$S/determinism_check.py" solution.py ./dataset/public --runs 2 || status=1
  step "per-record independence: half the evaluation rows"
  python3 "$S/half_rows_test.py" solution.py ./dataset/public || status=1
fi

echo; [ $status -eq 0 ] && echo "ALL GATES PASSED" || echo "SOME GATES FAILED"
exit $status
