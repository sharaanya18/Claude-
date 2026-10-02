---
name: eris-data-audit
description: Audit TRAIN data for a new Eris challenge (shapes, targets, groups, duplicates, ordering/id leakage, slices, information ceiling) by delegating to the eris-data-auditor agent. Use after the contract and before choosing validation or models.
---

# /eris-data-audit

Launch the `eris-data-auditor` agent with: the path to `CHALLENGE.md`, `reports/contract.md`, and `dataset/public/`. Ask for `reports/data_audit.md`.
Then read it yourself and extract: number of *independent* groups, giant-group warnings, identical-input/different-label rate, candidate-pool oracle recall, constant-baseline score, slices to
report, leakage tells, synthetic-data signals (record, never hard-code). Copy the 10-line summary into `CHALLENGE_NOTES.md § Data audit`.
Rule: train drives decisions; from test files only schema/row count/id format/size are allowed.

Added checks (2026-10): information ceiling by simulation where the generator is guessable (learned-patterns L001); label-shuffle run of the whole pipeline (must score at chance); id/order/size/metadata-only model under grouped CV; random-CV vs group-CV gap; nearest-neighbour distance random vs group folds; learning curve over number of GROUPS. Test files: schema, row count and length statistics only (CLAUDE.md 2.3A).
