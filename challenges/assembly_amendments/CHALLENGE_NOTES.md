# Challenge notes: assembly_amendments

## Status
- [x] contract  - [x] data audit  - [x] strategist plan  - [ ] validation  - [ ] baseline  - [ ] experiments
- [ ] review (compliance, runtime, red-team)  - [ ] presubmit  - [ ] submitted  - [ ] closed (lessons written)

## Contract (decision unit, valid answer, metric terms, constraints, bans, compute, runtime)
Full contract: `reports/contract.md`. Summary:
- Domain: structured assignment/partition over a bag of items (amendments), with French legal-text NLP features feeding both sub-tasks.
- Decision unit: one board (article-in-body). Predict per-item `fates` (5-way, must match board's `counts` exactly) and `joint` (partition of items) jointly within the board; boards independent of each other.
- Metric: weighted mean per board of `carried` (0.35, chance-corrected binary adopted-vs-not, skipped if a==0), `disposal` (0.35, chance-corrected 4-way among non-adopted), `joint` (0.30, ARI, skipped if no true pair shares a label — 227/622 test boards). Malformed fates -> -1 on carried+disposal; malformed joint -> 0 on joint. Non-standard metric, must be hand-implemented+tested (eris-metric-engineer) before modelling.
- Hidden split: GroupKFold on `bill_id` is mandatory (bills disjoint train/test, confirmed 0 overlap). Test further split into 2 unnamed slices by bill.
- Compute: no challenge-specific override -> CLAUDE.md default (~50min A10G target, 1h ceiling).
- Allowed: HF pretrained French LM fine-tuned (CamemBERT/FlauBERT/mDeBERTa) + GBDT ensemble member. Banned: external AN records/lookups of real outcomes, external model trained on these records, test-set statistics beyond one-board inference.
- `item_no` carries no order signal (randomized) — do not use as feature; sort by it only to recover canonical per-board item order from the shuffled items-file rows.
- Biggest risk: `joint` partition is the causal backbone for `disposal`'s `fell` fate — must be modelled early/first-class (pairwise same-discussion classifier -> clustering), not bolted on. Second risk: GroupKFold leakage.

## Data audit (shapes, groups, duplicates, label distribution, structure, ceiling diagnostics)
Full report: `reports/data_audit.md`. Key points:
- Train clean (0 missingness/mismatches). Fate imbalance 5.96x; adopted share drops 44%->16.5% as n_amendments grows.
- bill_id fully disjoint train/test (209 vs 111 bills). GroupKFold(bill_id) mandatory; bill sizes extremely skewed (max 472/4392 boards in one bill) -> use size-aware greedy fold assignment, not plain sklearn GroupKFold.
- 55% of bills have cross-board exact-duplicate dispositif text (re-tabled across readings/stages) -> confirms grouping necessity.
- Within-board exact dispositif dup = 0 (pre-deduped by construction). Near-dup text (Jaccard>0.8) -> 25x lift for shared joint label, 68% precision. Dup exposé text -> 15x lift, 41% precision. Feed as features, never hardcode.
- "Fell" only 16.7% explained by same-recorded-joint-group adoption; 55% of fell items are in singleton groups. Board-wide "Supprimer"(delete)-adopted -> 2.5x lift in others' fell rate (36.0% vs 14.2%) — real mechanic partly operates at whole-board level beyond the published joint field. Strong candidate feature, not a rule.
- 46.3% of exact-duplicate-text items (across different boards) have conflicting fates -> hard ceiling on text-only modelling; board context features (counts, bill_kind, examining_body, division, n_amendments) are structurally necessary.
- item_no confirmed pure noise (flat adopted-rate across quartiles) — do not use as ordered feature; still must sort by it explicitly (file row order happens to match on train but not guaranteed).
- Train board-size dist differs sharply from test (57.7% of train boards have n<4; test floor is 4) — restrict to n>=4 subset when matching test's regime for diagnostics (singleton-only joint rate: 70.7% all-train vs ~41% at n>=4, vs test's stated 36.5%).
- Dispositif length p99~850 words, max 14444 — fixed truncation (512-768 tokens) will cut tail; add log-length as feature.

## Validation design (mirror of the hidden split, groups, bias direction of the proxy)
Full plan: `reports/eris_plan.md`. Summary:
- Primary approach: fine-tuned French encoder (almanach/camembertav2-base, pinned revision) with 3 heads (fate, kill-power, grouped) -> pair LightGBM for `joint` -> item LightGBM stacker for `fates` (cascade/noisy-OR features) -> exact linear-assignment decode for fates (Bayes-optimal given counts-fixed chance baselines) -> average-linkage clustering decode for joint (OOF-selected cut tau). Fallback: smaller camembert-base if primary stage-1 profiles >25min.
- 5 bill-grouped size-aware folds (bills also merged when sharing >=3 identical edits per the dossier rule). Headline CV = metric.py score on the "test-like" subset (n in [4,80], >=2 nonzero counts; 1652 boards/144 bills) — raw train is ~27x inflated per metric_spec, never compare directly.
- Decode uses only each board's own `counts` (a given input column) — not test-set fitting; verified compliant (contract Q3).
- Cascade mechanic (deletion/rewrite-adopted -> 97% others fall) must be LEARNED (kill-power head + noisy-OR features), never hardcoded.
- Rejected: end-to-end set-transformer as primary (roadmap step 5 only), pairwise cross-encoder for joint (10x cost), large encoders (runtime), GBDT-only no-encoder (grey/lower ceiling, dev yardstick only), hardcoded cascade/Jaccard rules, counts fed into encoder (reserved ablation).
- Runtime estimate: ~28-34 min primary / ~18-22 min fallback on A10G vs 60min ceiling.
- Expected test-like OOF range 38-52 (non-trained yardstick = 18.8, counts-respecting random ~1.1). Estimates only, not measured (no GPU in dev sandbox).
- Open reviewer questions logged in eris_plan.md: Q1 (general French web-pretrained encoder vs "model trained on them" ban), Q2 (hand-parsed deterministic features), Q3 (expected-utility decode using board's own counts).

## Plan (primary, fallback, rejected options, fixed work plan)

## Experiment log (id, hypothesis, change, CV mean +- std, per-fold, runtime, kept?, notes)

## Error analysis

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
