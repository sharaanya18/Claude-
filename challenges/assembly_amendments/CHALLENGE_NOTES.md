# Challenge notes: assembly_amendments

## Status
- [x] contract  - [x] data audit  - [x] strategist plan  - [x] validation  - [x] baseline  - [ ] experiments
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
(see Validation design section above for the plan summary.)

## Validation harness (implemented)
`validate.py` (importable: `build_folds`, `build_merged_bill_groups`, `filter_test_like`/`test_like_mask`, `score_cv`, `bootstrap_ci`, `nested_select`, `run_oracle_checks`, `random_shuffle_baseline`, `parse_division_type`, `n_bucket`) + `reports/split_audit.md`. Verified end-to-end (`python3 validate.py` reruns everything from train CSVs only): gold-fed-back scores exactly 100.0, random-shuffle baseline ~0.26 on test-like subset (near 0 as expected), fold assignment deterministic, no bill_id spans >1 fold.
**Load-bearing fix the agent found**: the plan's literal "merge bills sharing >=3 identical texts" rule chains 72/209 bills into ONE group covering 85% of the data (driven by generic one-liners like "Supprimer cet article", 1553 occurrences) — unusable for 5-fold CV. Fixed with a length floor (>=40 normalized chars) before counting shared texts: gives 203 groups (6 real merge pairs, largest group 3 bills), stable for any floor in [40,100] chars, and the 6 survivors are genuine budget-cycle re-tabled text (300+ chars, cites specific code articles). Any future code touching bill-grouping must keep this floor.
Fold balance: test-like items balanced to within 0.1% across 5 folds. CMP/deuxième lecture readings absent from test-like per-fold table — expected (bill-clustered + rarely have >=2 nonzero counts), not a bug.
Bias direction: ~1-2pt optimism expected from in-script decode-constant selection (use `nested_select` to quantify) + stacking smoothing; net plan is private-LB ≈ test-like OOF minus ~1-2 points, use `bootstrap_ci` (bill-resampled, 1000 resamples) for the real interval on model OOF.
solution.py must: call `build_folds` once, use `filter_test_like`/`test_like_mask` for stage-2 training population + headline reporting, call `score_cv` after assembling OOF, use `nested_select` for every decode-constant choice (tau, Sinkhorn on/off, pair weighting), call `bootstrap_ci` on final OOF for the reported CI.

## Experiment log (id, hypothesis, change, CV mean +- std, per-fold, runtime, kept?, notes)

### exp0: baseline solution.py (implements eris_plan.md Primary A)
Implements: fine-tuned camembertav2-base (3 heads: fate/kill-power/grouped) -> pairwise LightGBM for joint -> item LightGBM stacker for fates (cascade/noisy-OR + margin + board features) -> exact linear-assignment decode (Bayes-optimal given counts) + average-linkage clustering decode for joint (tau selected via inlined nested_select on OOF ARI). Self-contained (metric.py/validate.py logic ported/duplicated into solution.py since the platform only receives solution.py itself — this is deliberate, documented in the file's docstring, not an oversight).

Known baseline simplifications vs the full plan (documented in solution.py docstring, candidates for /eris-experiment): Sinkhorn/IPF projection omitted (fixed off); LLRD simplified to 2 param groups (backbone/head LR) instead of per-layer decay; pair/stacker feature lists implement the plan's core signals, not every variant enumerated.

**Local verification (no GPU in this dev sandbox)**:
- `compliance_scan.py`: 0 errors, 2 WARN (heuristic false-positives on `test_items.groupby("item_id")` — manually verified + commented in code: both are per-board iteration over test's own existing partition, never cross-board aggregation; see lines flagged), 3 INFO (pretrained-weights confirmation, expected).
- Bugs found and fixed during smoke-testing: (1) dead/wrong conditional `if n in train_pair_mats` in the OOF joint-decode loop (checked int n against a dict keyed by item_id strings — always false, fell through to the correct branch by luck; fixed to `if bid in train_pair_mats`). (2) O(n^2) pandas `.iloc`-per-pair in `build_pairs` was a real runtime risk at the ~250k-pair scale this challenge needs; rewrote to pull per-board fields into plain arrays once, loop with lightweight indexing. (3) quadratic re-`.set_index()` of train_targets inside a dict comprehension; hoisted out. (4) added a defensive `_assert_item_no_order` check (merges reset pandas index repeatedly; a silent reordering bug here would corrupt which fate/joint lands on which item_no with no shape-check catching it).
- Ran the full pipeline end-to-end twice via a throwaway CPU-patched copy (tiny-random-bert, 2 folds, 1 epoch, DEVICE=cpu) against two local smoke datasets built from real train rows: (a) 40 boards/280 items/24 bills, all n<=15; (b) 83 boards/1174 items/47 bills with n up to 65 (stress-tests pair-matrix/decode at near-test-max board size). Both ran without crashing, produced valid, counts-respecting, reload-checked submissions. Confirms: decode validity, fold/feature-engineering correctness, and that stages 2a/2b/decode are fast (~8s for 1174 items) — stage-1 encoder training dominates runtime as expected.
- Verified independently that `transformers` loads `almanach/camembertav2-base` at the pinned revision cleanly (resolves the plan's open question about deberta-v2/tokenizer.json loading).

**Real GPU run**: kicked off via `.claude/scripts/kaggle_gpu_run.py` (data uploaded as private Kaggle dataset `sharanya1805/assembly-amendments-data`, kernel `sharanya1805/assembly-amendments-run`, T4/P100). First push was auto-cancelled by Kaggle ~60s after queueing (transient — no concurrency/quota issue found: dataset was `ready`, no other kernel was RUNNING on the account at the time); second push is RUNNING. Results to be added here once it completes.

## Error analysis

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
