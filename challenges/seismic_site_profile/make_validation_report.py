"""Phase 29: independent validator + report for a produced submission."""
import sys, csv, time
from collections import Counter
from pathlib import Path

sub_p = Path(sys.argv[1]); pub = Path(sys.argv[2])
out_p = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("submission_validation_report.txt")
runtime = sys.argv[4] if len(sys.argv) > 4 else "not measured in this invocation"

rows = list(csv.reader(open(sub_p, newline="", encoding="utf-8")))
hdr, body = rows[0], rows[1:]
test_ids = [r[0] for r in list(csv.reader(open(pub / "test.csv", newline="")))[1:]]
samp = list(csv.reader(open(pub / "sample_submission.csv", newline="")))
BANDS = {"v1", "v2", "v3", "v4"}

ids = [r[0] for r in body]
dup = [k for k, v in Counter(ids).items() if v > 1]
missing = [i for i in test_ids if i not in set(ids)]
extra = [i for i in ids if i not in set(test_ids)]
bad, cells = [], [Counter() for _ in range(4)]
for r in body:
    p = r[1] if len(r) > 1 else ""
    parts = p.split("|")
    if len(parts) != 4 or any(b not in BANDS for b in parts) or p != p.strip():
        bad.append((r[0], p)); continue
    for i, b in enumerate(parts):
        cells[i].update([b])

L = []
A = L.append
A("SUBMISSION VALIDATION REPORT")
A(f"generated            : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
A(f"file                 : {sub_p.resolve()}")
A(f"runtime of producing run : {runtime}")
A("")
A(f"header               : {hdr}  (required exactly ['id', 'profile'] in that order)")
A(f"header correct       : {hdr == ['id', 'profile']}")
A(f"row count            : {len(body)}   (required 333)")
A(f"sample_submission rows: {len(samp) - 1}")
A(f"duplicate ids        : {len(dup)}  {dup[:5]}")
A(f"missing test ids     : {len(missing)}  {missing[:5]}")
A(f"extra ids not in test: {len(extra)}  {extra[:5]}")
A(f"id order equals test.csv : {ids == test_ids}")
A(f"invalid/malformed profiles : {len(bad)}  {bad[:5]}")
A(f"empty or NaN cells   : {sum(1 for r in body if len(r) < 2 or r[1].strip() == '')}")
A("")
A("predicted band distribution per depth cell")
for i, c in enumerate(cells):
    tot = sum(c.values()) or 1
    A(f"  cell {i} ({['0-5 m','5-10 m','10-20 m','20-30 m'][i]:7s}): " +
      "  ".join(f"{b}={c.get(b,0):3d} ({100*c.get(b,0)/tot:4.1f}%)" for b in ("v1","v2","v3","v4")))
A("")
ok = (hdr == ["id", "profile"] and len(body) == 333 and not dup and not missing
      and not extra and ids == test_ids and not bad)
A(f"VERDICT: {'PASS - submittable' if ok else 'FAIL - do not submit'}")
txt = "\n".join(L)
out_p.write_text(txt + "\n")
print(txt)
sys.exit(0 if ok else 1)
