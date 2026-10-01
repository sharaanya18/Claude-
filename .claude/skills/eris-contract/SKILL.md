---
name: eris-contract
description: Extract the exact contract of an Eris challenge before any code: decision unit, valid and invalid outputs, metric terms, submission grammar, data files, explicit bans and silences, compute and runtime. Use first on every new challenge and whenever the description changes.
---

# /eris-contract — what exactly is being asked

Read the description twice. Fill every field; mark unknown ones `UNKNOWN: ask a reviewer` and write the assumption you will use.

```
CHALLENGE CONTRACT
Name / domain family (see eris-playbook) / Difficulty label (NLP, CV, RAG, Fine-tuning, From-scratch, Tabular, Bio/Chem)
Decision unit: what is predicted jointly (row, case, bag, slate, episode) and what is independent
One valid answer: ...
Invalid answers (score below zero or rejected): ...
Metric: exact formula, direction, per-example weights, nesting (macro over what), empty cases, ties, clipping, reference scores
Submission: file, columns in order, id column, row count, cell grammar (JSON/prefix/fixed decimals/sorted pairs), allowed value sets
Data files: shapes, group columns, anything unusual (opaque ids, sidecar files, tensors)
Hidden split: how test differs from train (disjoint units, time, new classes) and the held-out unit
Compute / runtime: stated hardware (A10G default, CPU-only if stated), time limit (shorter than 1 h overrides), memory
Allowed / required: pretrained weights? from-scratch? model families? fine-tuning required? TTA?
Explicit bans (hard constraints): copy each sentence from "What not to use" verbatim
Silences and assumptions: what the text does not say and the safest reading
Per-row independence: may rows/bags be pooled? (default: no whole-test statistics ever)
Boilerplate to ignore: "no internet" (HF/timm weights are allowed), "rule-based OK" (ML training is required)
Description-vs-formula discrepancies and reviewer questions
```
Red flags: non-standard metric (paste the formula; implement and test it), probabilistic vs hard outputs, multiple targets, "from scratch", a limit under 1 h, restricted model families, a metric
that bans metric-aware decoding, published decoy recipes, hints that test comes from unseen units.
Write the result to `CHALLENGE_NOTES.md § Contract` and `reports/contract.md`.
