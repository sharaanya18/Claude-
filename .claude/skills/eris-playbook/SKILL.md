---
name: eris-playbook
description: Router to the S-tier pattern library for Eris/Shipd challenges. Use after reading a new challenge description to classify the problem family (tabular, text, retrieval/slates, vision, audio, structured assignment, partition/bags, generation/programs, multimodal) and load only the reference files that apply (principles, metric/decoding, validation, features, ensembling/training, engineering/compliance).
---

# /eris-playbook — pick the right references, then apply them

The library distills 13 challenges, 60+ top leaderboard solutions, 6 reviewer rejections and 25 planned unseen challenges.
It is **hypothesis material, not a recipe**: every item must be validated on grouped CV for the challenge at hand and
must pass the challenge's own bans. Read `00-s-tier-principles.md` always; then only what the family needs.

## Step 1 — classify (a challenge can have more than one family)

| Cues in the description | Family file (in `references/families/`) |
|---|---|
| Rows of numeric/categorical features, molecules/proteins, sensor aggregates, forecasting, aggregate-label counts, reference-bank calibration | `tabular-and-scientific.md` |
| Text in → label/span/sequence/headline out, cloze/slot filling, OCR'd or multilingual text | `text-nlp.md` |
| Rank candidates for a query, archives, recommenders, decoy slates, NDCG/MAP/MRR | `retrieval-ranking-slates.md` |
| Images/frames/boards, segmentation, detection, geometry, multi-view, frame timing | `vision.md` |
| Waveforms, speech, arrays/sensors, tactile/force streams | `audio-signal.md` |
| Permutation / matching / ordering / relations / tensors with marginals / distinct selections / joint outputs | `structured-assignment.md` |
| Partition each bag/case on its own, unseen source classes, ARI-type metrics, per-row independence rules | `partition-and-bags.md` |
| Programs, pipelines, JSON, restoration/repair, unseen compositions, free-form generation | `generation-and-programs.md` |
| Several views/streams per case, robotics, cross-embodiment alignment | `multimodal-robotics.md` |

## Step 2 — cross-cutting references (read what the plan needs)

| Need | File |
|---|---|
| Always | `00-s-tier-principles.md` |
| Loss, decode, thresholds, oracle checks, per-slot calibration | `metric-and-decoding.md` |
| Split design, groups, nested selection, bias direction, leakage tells | `validation-recipes.md` |
| Feature ideas by task shape; domain-theory architectures | `features-and-representations.md` |
| Ensembling, pretrained usage, training recipes, efficiency | `ensembling-and-training.md` |
| Determinism, I/O contract, what reviewers reject, self-audit | `engineering-and-compliance.md` |

## Step 3 — use them

1. Write the contract (decision unit, valid output, metric terms, bans, compute) — `/eris-contract`.
2. For each lever you consider, note its source id (e.g. A3, B9, C16), the compute it needs, the ban it might hit, and the
   cheap experiment that would falsify it. Prefer levers that **independent solvers converged on**.
3. Rank by expected private gain per hour (principles §2), choose a lean primary + a fallback, list rejected options
   with reasons.
4. Re-check the final plan against `engineering-and-compliance.md` §3–§4.

## Rules of use
- A pattern that conflicts with Deterministic Execution is marked DO NOT ADOPT in the references; follow CLAUDE.md §3.
- The challenge description overrides every default here (hardware, weights, runtime, TTA, per-row independence).
- Do not copy code from the references; derive and write original, commented code in `solution.py`.
