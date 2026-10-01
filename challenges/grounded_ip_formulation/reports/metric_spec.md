# Metric spec: grounded_ip_formulation

Tools: `formulation.py` (parser + HiGHS MILP via scipy.optimize.milp, mip_rel_gap=1e-9), `metric.py` (three-term score), `tests_metric.py` (30 tests, run `python3 tests_metric.py`).

## Verified on train data
- All 248 seed formulations parse and reproduce `optimal_value` within 1e-4 (248/248; worst rel err 2.3e-10).
- Solve time per case: mean 7.5 ms, p95 15 ms, max 0.31 s. Seeds average 5.8 variables (max 21), 6.2 rows; 210/248 have int/bin variables.
- `metric.case_score` ~60 ms/case, so it is usable inside OOF scoring.

## Case score = (value + counterfactual + structure) / 3
- Value: optimum within rel 1e-4 of the reference optimum.
- Counterfactual: share of 3 variants (every number times its own factor in [0.55,1.45] excl. (0.95,1.05); equal values share a factor; years 1990-2030 fixed) with an equal optimum at rel 1e-6.
- Structure: F1 of canonical rows/objective/int-bin item under ONE best variable renaming.
- Invalid output (unparsable, nonlinear, unknown reference, over limits) scores 0 on everything. A parseable program with no finite optimum still earns structure credit (up to 1/3).

## Approximations (hidden grader unavailable)
1. Counterfactual factors are my own seeded draws, so the term is noisy with 3 variants.
2. Cf fallback to the value term for ~3% of references that are infeasible on every random variant (typed-in then scores 0.677, not 0.667).
3. Renaming: colour refinement (3 rounds) + pairwise-exchange hill climb.
4. Equality rows compared up to overall sign.
5. The int/bin item is emitted only when the model has an int/bin variable (`metric.INT_ITEM_ALWAYS` switches it).
6. Numbers compared at 9 significant digits.

## Decode / training implications
- Inference ranking among the model's own candidates: (1) parses and references exist, (2) finite optimum, (3) plausibility and agreement across samples, (4) model likelihood. No hand repair of outputs.
- Training-time verifier: `formulation.matches_optimum(text, numbers, optimal_value, 1e-4)` is exactly the value term. It cannot detect slack or coincidental rows, so prefer samples using many distinct references with no typed-in numbers.

## Open reviewer questions
int/bin item when none declared; `x4cake_a` parsed as a variable name; keyword case and `==`; whether declared-only variables count toward the 60 limit; grader MILP gap.
