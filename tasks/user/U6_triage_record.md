# U6 — Triage record reconstruction from device-failure reports (user-supplied, verbatim; Domain: Seq2Seq, Medium, CPU, tags medical/text)

## Overview
Each report is a single free-text medical-device failure narrative; the required output is a structured record: one string encoding four interlocking judgments. Two judgments may have no determinable answer, and deciding that is itself scored: abstain by emitting `unknown` rather than guessing patient severity. Scoring is cost-asymmetric - under-stating urgency is penalised more than over-stating - and every test report comes from an organisation absent from training.

## Background
Reports are real, messy, terse, abbreviated, redacted ((b)(4)), spanning infusion pumps, implantable leads, staplers, glucose meters, catheters and hundreds of other device types. Three properties shape difficulty: (1) Evidence scarcity and abstention: ~6 in 10 reports never state a patient outcome; those carry harm=unknown and priority=unknown; distinguishing "no outcome reported" from "minor outcome reported" is scored as its own component and cannot be gamed (always/never abstaining both score poorly). (2) Asymmetric cost: priority is scored with an agreement measure that penalises under-statement more heavily than over-statement. (3) Distribution shift: evaluation is entirely on manufacturers held out of training, so filer-specific phrasing and boilerplate templates cannot be memorised.

## Task
For each report produce report_text -> triage_record: `priority=<0-9 or unknown>|breadth=<0-2>|delay=<0-3>|harm=<1-4 or unknown>`
Fields: harm (1 minor/other consequence, 2 required intervention, 3 hospitalization/disability/life-threatening, 4 death, or unknown = no patient outcome stated, not a claim that no harm occurred); breadth (distinct failure modes: 0 one, 1 two, 2 three or more); delay (regulatory reporting delay: 0 within a week, 1 a week to a month, 2 one to six months, 3 over six months - a property of the reporting process); priority (overall triage tier 0-9 or unknown; driven primarily by how serious the reported patient consequence was, raised further when the failure involved more than one mode or when a serious case reached the regulator quickly; exact weighting unpublished; priority and harm are unknown together, always).

## Evaluation metric (composite in [0,1], higher is better)
score = 0.35*AWK(priority, rows where true priority defined) + 0.15*MacroF1(harm reported vs unknown) + 0.15*QWK(harm severity, rows where true harm reported) + 0.15*QWK(breadth) + 0.15*QWK(delay) + 0.05*ExactRecordMatch
AWK = asymmetric weighted kappa (quadratic-weighted-like, under-triage penalised more heavily). QWK = quadratic-weighted Cohen's kappa. ExactRecordMatch = fraction with all four fields correct. Components clipped to [0,1]; chance/constant/random score near 0.

## Data
id (TR / TS prefixed, splits share no ids), report_text (only input feature), triage_record (train only target). Train is a few hundred thousand rows; ~60% have harm=unknown/priority=unknown. Test is split by manufacturer; reporting organisations in test do not appear in training.
Files: dataset/public/train.csv, test.csv, sample_submission.csv. Output ./working/submission.csv with columns id,triage_record; exactly one row per test id, header, no empty records, no duplicates. Out-of-range or malformed records get no credit for affected fields.
Compute: CPU.

## What not to use
Reconstruct from report_text alone. Prohibited: using the test set for anything but producing predictions (no training, fine-tuning, pseudo-labelling, self-training, threshold/hyper-parameter selection, calibration, feature fitting, fitting vectorisers/normalisers/embeddings on test text, exploiting test distribution; no decision informed by test data); back-tracking to source records (no querying/downloading/scraping/matching against any external database, corpus or mirror to recover fields, dates, outcomes or organisation); hard-coded answer tables or grader probing (no id->record dictionaries, no near-duplicate hashing against external data, no filesystem probes, no leaderboard probing); hosted/closed-source LLM APIs at any stage (including distillation/pseudo-labelling from such teachers; only open-weights models and self-trained pipelines permitted); rule-only pipelines (predictions must come from a learned model that reads the text, not a hand-written keyword->field lookup).
