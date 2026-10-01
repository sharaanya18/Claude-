#!/usr/bin/env python3
"""Corpus helper for the /eris-learn skill (stdlib only).

  python3 .claude/scripts/corpus.py new <slug>        scaffold corpus/<slug>/ (CHALLENGE.md, outcome.md, solutions/, digests/, blind/, gap_report.md)
  python3 .claude/scripts/corpus.py index             rebuild corpus/INDEX.md from every corpus/*/outcome.md front matter
  python3 .claude/scripts/corpus.py check <slug>      verify the blind-first protocol: blind plan written before any digest, no solution text in blind/

Layout of corpus/<slug>/:
  CHALLENGE.md      verbatim description (never summarised)
  outcome.md        front matter (family, ranks seen, scores, baseline, what is verified) + notes
  solutions/rankN.* raw competitor files, kept LOCAL (git-ignored): distilled digests are committed, raw code is not
  blind/            the strategist plan written from CHALLENGE.md alone, BEFORE reading any solution
  digests/rankN.md  structured digest of each solution (own words, no pasted code)
  gap_report.md     blind plan vs top solutions: rubric scores, gaps, library updates made
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"

OUTCOME = """---
slug: {slug}
family: TODO            # one of the eris-playbook families, or several separated by commas
added: TODO             # YYYY-MM-DD
solutions_seen: 0       # number of top solutions supplied
ranks: TODO             # e.g. "private 1-5" or "public only"
scores: TODO            # numbers exactly as given by the user; mark unverified ones
ai_baseline: TODO
board_high: TODO
review_status: unknown  # accepted / rejected / unknown for each solution if known
blind_plan_score: TODO  # rubric agreement of the blind plan with the top solutions, 0-1
---
# Outcome notes
What is verified (from the user / platform) and what is hearsay. Keep the distinction explicit.
"""


def new(slug):
    d = CORPUS / slug
    if d.exists():
        sys.exit(f"corpus/{slug} already exists")
    for sub in ("solutions", "digests", "blind"):
        (d / sub).mkdir(parents=True)
    (d / "CHALLENGE.md").write_text("Paste the challenge description here, verbatim.\n", encoding="utf-8")
    (d / "outcome.md").write_text(OUTCOME.format(slug=slug), encoding="utf-8")
    (d / "gap_report.md").write_text("# Gap report: " + slug + "\n(see .claude/templates/gap_report.md)\n", encoding="utf-8")
    print(f"created {d}\nnext: write the description to CHALLENGE.md, run the BLIND strategist first (blind/eris_plan.md), only then save solutions to solutions/")


def front(path):
    m = re.match(r"---\n(.*?)\n---", path.read_text(encoding="utf-8"), re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.split("#")[0].strip()
    return out


def index():
    rows = []
    for d in sorted(p for p in CORPUS.iterdir() if p.is_dir()) if CORPUS.exists() else []:
        o = d / "outcome.md"
        if o.exists():
            f = front(o)
            rows.append((d.name, f.get("family", ""), f.get("solutions_seen", ""), f.get("ranks", ""), f.get("blind_plan_score", ""), f.get("added", "")))
    lines = ["# Corpus index (auto-generated: python3 .claude/scripts/corpus.py index)", "",
             "| Challenge | Family | Solutions | Ranks | Blind-plan agreement | Added |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    nums = []
    for r in rows:
        try:
            nums.append(float(r[4]))
        except ValueError:
            pass
    if nums:
        lines += ["", f"Blind-plan agreement over {len(nums)} scored challenges: mean {sum(nums) / len(nums):.2f}, min {min(nums):.2f}, max {max(nums):.2f}. "
                  "Track this over time: it measures how well the system predicts what top solvers did on tasks it had not seen (agreement, not a leaderboard score)."]
    CORPUS.mkdir(exist_ok=True)
    (CORPUS / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote corpus/INDEX.md with", len(rows), "entries")


def check(slug):
    d = CORPUS / slug
    problems = []
    plan = d / "blind" / "eris_plan.md"
    if not plan.exists():
        problems.append("blind/eris_plan.md missing: run the strategist on CHALLENGE.md BEFORE reading solutions")
    else:
        digests = list((d / "digests").glob("*.md"))
        for dg in digests:
            if dg.stat().st_mtime < plan.stat().st_mtime:
                problems.append(f"{dg.name} is older than the blind plan: the blind plan was not written first (protocol broken)")
        text = plan.read_text(encoding="utf-8", errors="replace").lower()
        for sol in (d / "solutions").glob("*"):
            if sol.is_file() and sol.suffix in (".py", ".ipynb", ".md", ".txt"):
                toks = set(re.findall(r"[a-z_]{12,}", sol.read_text(encoding="utf-8", errors="replace").lower()))
                hits = [t for t in toks if t in text and t not in {"sample_submission", "compliance_scan"}]
                if len(hits) > 25:
                    problems.append(f"blind plan shares {len(hits)} long identifiers with {sol.name}: possible contamination, re-run blind")
    if problems:
        print("FAIL:\n  " + "\n  ".join(problems))
        return 1
    print("PASS: blind-first protocol holds for", slug)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(64)
    cmd = sys.argv[1]
    if cmd == "new" and len(sys.argv) > 2:
        new(sys.argv[2])
    elif cmd == "index":
        index()
    elif cmd == "check" and len(sys.argv) > 2:
        sys.exit(check(sys.argv[2]))
    else:
        print(__doc__)
        sys.exit(64)
