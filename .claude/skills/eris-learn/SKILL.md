---
name: eris-learn
description: Ingest a NEW Eris challenge together with its top leaderboard solution(s) supplied by the user, and turn them into reusable knowledge. Runs a blind strategist plan first, distils each solution independently, grades the blind plan against what the top solvers did, classifies the gaps, updates the pattern library and corpus index, and commits. Use whenever the user pastes a challenge description with winning or top-ranked solutions or write-ups.
---

# /eris-learn <slug> — learn from a challenge and its top solutions

Purpose: every challenge+solutions pair the user supplies should make the next unseen challenge easier. The loop is a train/test loop: **predict blind, then learn from the answer**, so the system's quality is measured honestly each time.

## Inputs (from the user's message)
1. The challenge description (verbatim). If it is missing, ask once; solutions without their task text teach little.
2. One or more top solutions (code, notebooks, or write-ups), ideally with rank, score, whether the board was private, the AI baseline and the board high.
3. Any known review outcome (accepted/rejected) per solution. Mark what the user states as verified and everything else as unverified.

## Protocol (order matters: the blind plan must exist before any solution is saved or read by a subagent)
**Step 0 — Scaffold and store the task.** `python3 .claude/scripts/corpus.py new <slug>`. Write the description **verbatim** to `corpus/<slug>/CHALLENGE.md`. Fill the facts you were given in `corpus/<slug>/outcome.md` (family, ranks, scores, baseline, verified vs unverified). **Do not save the solutions yet.**

**Step 1 — Blind plan.** Launch the `eris-strategist` agent with only `corpus/<slug>/CHALLENGE.md` (plus CLAUDE.md and the playbook) and write its plan to `corpus/<slug>/blind/eris_plan.md`. Tell it explicitly: no dataset is available unless the user gave one; do not open any other path under `corpus/<slug>/`; do not search the web; list the diagnostics you would run and mark data claims unverified. If no subagent can be spawned, state that the blind condition is compromised (you have already read the solutions) and record `blind_plan_score: unavailable` instead of fabricating one.

**Step 2 — Save solutions.** Write each solution verbatim to `corpus/<slug>/solutions/rank<N>.<ext>` (raw files are git-ignored; only distilled knowledge is committed).

**Step 3 — Independent digests.** Launch one `eris-solution-analyst` per solution **in one message** (parallel, independent). Each saves `corpus/<slug>/digests/rank<N>.md` from `.claude/templates/solution_digest.md`.
Then `python3 .claude/scripts/corpus.py check <slug>` (blind-first protocol holds, no contamination).

**Step 4 — Grade the blind plan.** Launch `eris-gap-grader` with the digests and the blind plan. It writes the rubric from the digests *first*, scores the plan (1 / 0.5 / 0), classifies misses (KNOWLEDGE / PROCESS / RULE / JUDGEMENT) and proposes the smallest updates → `corpus/<slug>/gap_report.md`. Put the agreement score in `outcome.md` (`blind_plan_score`).

**Step 5 — Apply the updates yourself, minimally and verifiably.**
- KNOWLEDGE gaps → add entries to `.claude/skills/eris-playbook/references/learned-patterns.md` (format below). If a lever is already in a curated file, extend that entry with "independently confirmed again by <slug>" instead of duplicating; when a lever reaches ≥ 2 independent challenges, promote it into the right curated file (metric-and-decoding, validation-recipes, features-and-representations, ensembling-and-training, a family file) and mark `[N×]`.
- Add a worked example line to the matching family file (task shape → what top solvers did → what the library should recommend next time).
- PROCESS gaps → one checklist line in the relevant agent or skill (strategist, validation architect, metric engineer, baseline). JUDGEMENT gaps → a principle or margin note in `00-s-tier-principles.md`. Do not rewrite structure from a single challenge; recurring misses across `corpus/INDEX.md` justify bigger changes.
- RULE conflicts (a top solution used a wall-clock budget, fallback, test statistics, hand rules, etc.) → keep the *idea* only if a compliant implementation exists; mark the implementation DO NOT ADOPT with the reason; never add a lever that the current platform rules would reject.
- Every new entry: **When** (data signal) → **What** (mechanism, enough to re-implement originally) → **Gain** (author-reported vs measured; say "unverified") → **Cost/Risk** (compute, compliance grey zones) → **Evidence** (slug, rank, number of independent confirmations). No pasted code.

**Step 6 — Index and sanity checks.** `python3 .claude/scripts/corpus.py index`. Re-run `python3 -m unittest discover -s .claude/scripts/tests`. If agents or skills changed, re-read them for contradictions with CLAUDE.md §2-§3.

**Step 7 — Commit and push** (digests, gap report, outcome, library updates, INDEX; not `solutions/`).

## What to report back (short)
Blind-plan agreement score and the trend in `corpus/INDEX.md`; the top three levers the system missed; which library files changed; which solutions looked non-compliant under current rules and why; what remains unverified. Never present agreement as a leaderboard score.

## Rules
- Use only solutions the user supplied. Do not search for other competitors' Eris solutions or leaked answers (rejection-worthy and off-limits).
- Treat author-reported scores as claims. A top-5 rank does not imply the solution was accepted by reviewers.
- Raw competitor code is reference material, never pasted into `solution.py`; the library holds distilled mechanisms and triggers.
- Keep the library lean: prefer extending, tag confirmations, delete or demote entries that later challenges contradict (note why).
