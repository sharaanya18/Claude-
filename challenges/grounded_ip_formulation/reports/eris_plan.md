# Eris plan: Weakly Supervised Grounded Integer Program Formulation

Author: eris-strategist. Date 2026-10-01. Inputs: `CHALLENGE.md` (verbatim description), the coordinator's notes, light
read-only profiling of `dataset/public/` (train-side statistics; for test only schema, row count and length statistics).
Scratch parser and solver used for the oracle check (not a deliverable, re-write originally in `solution.py`):
`/tmp/claude-0/-home-user-Claude-/7a1c0fd8-3c89-5b5b-b0a2-4491386772f7/scratchpad/fp.py`.

Tags: [VERIFIED] = measured on the local files today; [UNVERIFIED] = reasoning or estimate, to be measured on GPU.
No GPU in the sandbox: **every runtime number below is an estimate from FLOP/overhead arithmetic. Profile it before you
freeze the plan.**

---

## Contract & decision unit

- **Decision unit:** one word problem (row of `test.csv`). The output is one string, a linear MILP in the challenge grammar.
  - Statements are separated by `;`.
  - The first statement is `max|min|maximize|minimize[:] <linear expr>`.
  - Each further statement is a constraint `<expr> (<=|>=|=) <expr>`, or a declaration `int a, b` / `bin c`.
  - Expressions use refs `x<k>`, plain numbers (commas allowed; trailing `%` means /100), variable names
    (`[A-Za-z_][A-Za-z0-9_]*`, not of the form `x<k>` or `q<k>`), `+ - * /` and parentheses. A product or quotient
    may have at most one variable side.
  - All variables are nonnegative. At most 60 variables and 150 constraints.
- **Submission file:** `case_id,formulation`, one row per test id, any order. Mirror `sample_submission.csv` anyway: same
  ids, same order. [VERIFIED] test has **345** rows and train **2,477** (the description says 347 / 2,479). Train the
  validator on the files, not on the prose.
- **Invalid vs low-scoring.** File level: a duplicated or missing id or column means rejection. Row level: a formulation
  that does not parse, is non-linear, references a missing `x<k>` or has no finite optimum scores 0 on the value and
  counterfactual terms. It probably also loses structure credit if it does not parse.
- **Metric per case:** `(V + C + S) / 3`.
  - **V** = 1 if our optimum on the real numbers equals the reference optimum within relative tolerance 1e-4.
  - **C** = share of 3 hidden variants on which our optimum equals the reference optimum, relative tolerance 1e-6.
    - Each variant multiplies every distinct number by its own factor in [0.55, 0.95] ∪ [1.05, 1.45].
    - Equal values share one factor. Years 1990 to 2030 stay fixed.
    - Typed literals are never perturbed (only refs move).
  - **S** = F1 over items: rows (normalised sense, (var, coef) set, rhs, all at real numbers), the objective (direction
    and pairs), and one integrality item. Items are compared after ONE model-wide one-to-one variable renaming. The
    grader finds that renaming by 3-round neighbourhood refinement plus pairwise exchanges.
  - Case scores are averaged equally over the 345 cases. No hierarchical weights.
- **What each term rewards.**
  - V: the right decision value.
  - C: grounding. Every coefficient must be a ref to the number that supplies it, and slack-but-wrong constraints are
    punished. Since `numbers_json` dedupes values, **a value maps to exactly one ref**, so "the right value under the
    wrong ref" cannot happen. C failures come from wrong or missing constraints that happen not to bind, and from a
    literal `1`/`100` versus a ref to a text number that equals 1/100.
  - S: imitation of the *reference program's style*, including its helper variables and its imperfections.
- **Published yardsticks (on test):** sample 0.0088; most common seed 0.0059; another case's formulation 0.0086; typed
  optimum (V only) 0.3368; reference with numbers typed in (V + S, no C) 0.6667; perfect 1.0. The AI baseline is not
  printed: ask (see Open questions).
- **Pipeline stages, each diagnosed separately (S-tier §3):**
  1. Candidate generation by a fine-tuned LM. Coverage = pass@K (any candidate with V = 1).
  2. Selection among the model's own candidates. Ranking = whether the chosen candidate is V-correct when one exists.
  3. Decode validity: parse + linear + finite optimum.

## Compliance regime

**Domain:** NLP / seq2seq, effectively a "Fine-tuning" regime. Pretrained weights must be *fitted on the supplied
training cases*.

**Explicit bans, and how the plan respects each.**

| Ban (description) | Plan |
|---|---|
| Hardcoding / hand-editing / lookup of formulations keyed by `case_id` | Every submitted string is verbatim LM output. Ids are used only to join. |
| Untrained prompting, in-context examples, retrieval into the prompt | The LM is fine-tuned in-script. The prompt holds only the case's own text. No retrieved train cases. |
| Hand-written parsing rules or templates that *produce or repair* formulations (keyword tables, regex fixes, positional defaults) | No output edits at all. Hand rules appear in exactly two allowed places: (a) **building inputs** (inline ref annotation, ¶→newline); (b) **parsing model outputs to check them / choose among them**. The parser never rewrites a string. |
| Eval-time search for formulations | Fixed K samples plus greedy per test case. One selection pass among them. No resampling loops conditioned on validity, no grammar enumeration, no solver feedback into generation. |
| `case_id` / row order as signal | Never used. Length-sorted batching only orders compute and does not change any per-case input. |
| Outside copies of the collection / models trained on its formulations | Only a general base LM (Qwen2.5 base) from HF, pinned. No OR-specialised checkpoints (ORLM, LLMOPT, OptiMUS, etc.). |
| Fitting on evaluation cases (text, pseudo-labels, TTA) | Test text is used only as the per-case prompt. No tokenizer or vocab fitting, no continued pretraining on test text. |
| API / private / gated models | None. Qwen2.5 is Apache-2.0 and not gated [VERIFIED via HF API]. |

**Intended weak supervision.** Train-side rejection sampling against `optimal_value` with HiGHS (`scipy.optimize.milp`)
is "the intended use" and is fully clean. Train-side self-made perturbations used to choose among *accepted* training
candidates are also clean: they touch only training cases, no reference is needed, and they only select training
targets.

**Self-audits.**
- **Strip-the-ML:** remove the fine-tuned LM and nothing produces a formulation. The inline annotator and the parser
  cannot output a program. Passes by construction.
- **No whole-test aggregation:** each test row's output depends only on its own prompt, the frozen model and its own
  K samples. Selection uses only that row's candidates. The half-rows test must pass, with batch-composition numeric
  noise as the only caveat (see Fixed work plan).
- **Sibling leakage:** groups are used only for validation folds. No test-time feature compares a row with other rows.
- **Hard-coded constants:** fixed-plan counts (K, epochs, LR, views) are principled defaults justified by profiling,
  not by submission feedback. The decode selection rule has no fitted weights: majority/MBR with a logprob tie-break.
  The MILP option choice is derived in-script from seeds (see Metric-aware training & decode).

**Grey items** (each also listed in Open questions):
1. At test time, solving the model's own candidates on *self-perturbed* versions of the test numbers to vote on
   counterfactual agreement. It is selection among outputs, but it resembles "doing for evaluation cases what is
   allowed for training cases". **Off in the primary.** Enable only after reviewer approval *and* a dev gain greater
   than noise (required margin: ≥ +0.02 case score on the dev test-like slice, consistent across ≥ 4/5 folds).
2. Contamination of general pretrained LMs. Qwen2.5 base may have seen the published collection on the web. The
   description permits "an open-weight decoder model" and "general knowledge of optimization modelling". State in the
   docstring that no OR-formulation-tuned checkpoint is used.
3. Uploading the public data to Kaggle (private dataset) for GPU dev runs. The rules forbid downloading/searching
   copies, not private dev compute, but confirm before using `kaggle_gpu_run.py data`.

## Data findings

All [VERIFIED] unless marked otherwise.

**Shapes.**
- Train: 2,477 rows `case_id, problem, numbers_json`.
- Labels: 2,477 rows. `optimal_value` median 1,880, range −3.0e6 to 5.75e6. 1.9% are negative; 12.6% are non-integer.
- Seeds: 248 formulations.
- Test: 345 rows, same schema.
- No missing columns.

**Text.**
- Train problem length: median 1,080 chars (p25 909, max 2,420). Test: median 1,154, max 2,162 (schema-level size only).
- Table share (Markdown `|---|`): train 50.9%, test 50.4%.
- With the Qwen2.5 tokenizer, prompt (text + numbers list) is median 370 / p99 630 / max 818 tokens on train; test max
  673.

**Numbers.**
- Refs per problem: median 12, range 3 to 30. Refs are always `x0..x(N-1)` in a per-problem random order (100% of rows).
- With the description's number definition (digits, thousands commas, optional decimals, % → fraction),
  **100% of the 42,877 number tokens in train text map to a ref, and every ref occurs in its text.** Inline annotation is
  therefore lossless and unambiguous.

**Seed formulations (248).**
- All 248 parse with a grammar parser and **solve with `scipy.optimize.milp` (HiGHS) to their `optimal_value`
  (248/248, 1.9 s total).** This is the oracle check (lever 15): grammar, parser and solver agree with the labels.
- Variable counts: 4: 101, 5: 73, 6: 31, 7: 6, 8: 2, 9: 15, 10: 3, 12: 8, 15: 5, 20: 3, 21: 1. **Only 74 / 248 (30%)
  have ≥ 6 variables.** Test has all ≥ 6 (median 7, up to 18).
- Constraints: median 6, max 20.
- Objective: max 189 / min 59. `int` declared in 83%, `bin` in 3%.
- Equality constraints in 17%. Parentheses in 23% (e.g. `(x8 + (-x6))*croissants`). Division in 2%.
- Seeds use 93.5% of their refs on average.
- Typed literals: `1` ×32, `0` ×8, `100` ×2. The literal-1-versus-ref choice is a learnable grounding decision that
  only C sees.
- Target length (Qwen tokens): median 112, p90 196, p99 366, max 463. About 22 tokens per variable.
- **References are noisy** (coordinator's note, confirmed). Examples:
  - Helper variables such as `labor_hours = x7*croissants + x9*muffins`.
  - Self-referential rows such as `rm2_a >= rm1_a + rm2_a`.
  - A 3×4 transport seed with only ONE demand constraint.

  A text-faithful model will sometimes miss V and S on such cases. Imitating the reference style (seeds plus accepted
  self-samples) is the metric-optimal behaviour.

**Family mix.**
- Transport/logistics keyword share in train is 20.6% (description: ~1/5; test ~1/2). Blend and invest are rare (< 1%).
- Among seeds, transport-keyword problems have mean 8.7 variables (median 6) against 5.1 for the rest.
- **Test-like data lives in the transport slice and the ≥ 6-variable seeds.**
- Number count is a weak proxy for size (corr(n_refs, n_vars) = 0.19 on seeds). Do not use it as the test-like slice
  definition.

**Scenario groups.**
- Union-find on (lowercased first sentence) OR (sorted value set) gives **342 groups** for 2,477 rows: mean 7.2, median
  4, max 92 (3.7% of rows), 19 singletons.
- 313 groups contain both table and prose versions.
- 222 groups contain ≥ 2 distinct number sets.
- Seeds cover 136 groups. 1,434 non-seed rows share a group with a seed.
- Number-masked text is almost always unique (2,464 masked templates; only 25 rows share one). Group-mates are
  *different problems on a shared opening*, not number-variants of one template. So **real same-template
  counterfactuals are NOT available in train**. That proxy is dropped.
- Effective sample size is ~342 groups, not 2,477 rows.

**Other checks.**
- Leakage suspects: none in the columns. Ids are opaque and must not be used.
- Irreducible ambiguity: many programs share an optimum (the core weak-supervision problem), and some references are
  imperfect.

## Validation design

**Hidden split, reproduced.**
- The split is scenario-disjoint (same opening sentence or same number set), and the test is shifted to ≥ 6 variables
  and ~50% transport.
- Mirror: **5 group folds** over the 342 union-find groups, assigned greedily so that each fold gets about:
  - 20% of rows;
  - ~50 seed rows, of which ~15 have ≥ 6 variables;
  - ~20% transport-keyword rows.
- The keyword is used only to stratify and slice validation, never as a model input or rule. Cap: no fold holds more
  than one of the top-3 largest groups.

**Dev protocol.**
- A full pipeline run is fit on the other 4 folds: seeds and non-seed rows of those groups only. The EI candidate pool
  also comes from training-fold rows only.
- Report per fold:
  - **M1** = V on all held-out rows (~495 rows; labels known for all). SE ≈ 0.02.
  - **M2** = full case score (V + C + S) on held-out seed rows (~50) with the seed as reference. C uses our own 3
    factor draws per case that follow the stated recipe. Skip and replace variants where the reference has no finite
    optimum.
  - **M3 (test-like)** = M2 on seed rows with ≥ 6 variables (~15 per fold, 74 over 5 folds), plus V on transport-keyword
    held-out rows (~100 per fold).
- Headline proxy = M3 pooled over folds.
- Stage diagnostics on held-out rows:
  - pass@K coverage;
  - selection hit rate (the chosen candidate is V-correct, given one exists);
  - validity rate;
  - EI acceptance yield;
  - **acceptance precision**: among held-out seed rows, the share of V-accepted candidates whose C = 1 against the seed.
    This measures how often "matches the optimum" is a false positive.

**Metric unit tests on held-out seeds** (reproduce the published yardsticks in order of magnitude):
- reference vs itself = 1.0;
- `max v; v <= x0` ≈ 0.01;
- typed optimum `max v; v <= <opt>` ≈ 0.33 to 0.34;
- reference with numbers typed in ≈ 0.667 (C ≈ 0);
- a reference with two coefficients swapped between variables loses those rows in S;
- shuffled row order and renamed variables give S = 1.

**Structure term implementation (approximate the grader).**
- Normalise rows: `>=` becomes `<=` by negation; an equality is scaled so its first coefficient is positive. "First"
  is ambiguous, so use the canonical variable order after renaming and test both readings.
- Compare coefficients at relative tolerance 1e-9.
- Variable signatures: 3 rounds of refinement over (row, coef) incidence, then Hungarian assignment on signature
  agreement, then a fixed number of pairwise-swap passes.
- F1 = 2·matched / (|ours| + |ref|) over rows + objective + integrality item.
- Bias: our matcher may be weaker than the grader's, which makes our S slightly pessimistic.

**Selection steps and their held-out checks.**
- Model choice (0.5B vs 1.5B vs Coder), number of EI rounds, K, the EI target-selection rule and the decode rule are
  each compared **paired on the same folds** (playbook lever 8, V4).
- Accept a change only if the paired gain in M3 (or M1 when M3 is too noisy) exceeds 1 SE of the paired fold
  differences, or holds in ≥ 4/5 folds.
- The decode rule has no fitted constants, so it needs no cross-fitting.
- Keep fold 5 as the **sanity holdout**: it touches no selection until the final config is frozen (V4).

**Bias direction of the proxy.**
- M1: **optimistic** by roughly +0.10 to +0.20 on V vs test. 70% of held-out rows are 4 to 5-variable,
  production-mix problems, which are easier than the test mix.
- M3: closer to test-like, but still **mildly optimistic**. Seeds with ≥ 6 variables are mostly ≤ 12 variables; test
  goes up to 18 and has 50% transport against ~60% of the large seeds.
- Training on 80% of groups is mildly **pessimistic** (−0.01 to −0.03) against the 100% refit.
- The self-drawn counterfactuals are unbiased in expectation.
- Net: expect the private score **0.03 to 0.10 below pooled M3**. Use the published 0.3368 / 0.6667 numbers as anchors.

**Cost.** One fold run of the cheap config (0.5B, SFT on seeds only, no EI) is ~6 min on an A10G [UNVERIFIED]. A full
EI config is ~40 min. Ablate on folds 1 to 2 with the cheap config. Run the full config on all 5 folds only for the
final candidates.

## Overfit/underfit risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Memorising scenarios (342 groups; 1,434 non-seed rows near seed scenarios) | Group stats above | Group folds. Fixed, short epoch counts (3 on seeds, 2 on the union). Ref-permutation augmentation stops the model memorising ref indices. Restart from base weights for each SFT round, STaR-style, so seed epochs do not compound. |
| **False-positive EI targets** (wrong program, right optimum: slack constraint, missing `int` with an integral LP optimum, a coincidence) teach C failures | The core issue named by the description | Accept only on exact V (rel 1e-4) **and** finite optima under 3 self-drawn perturbations of the train numbers. Among multiple accepted candidates for a row, keep the one in the largest perturbation-agreement cluster (tie: highest mean token logprob). Measure acceptance precision on held-out seeds before and after this filter. Keep at most 1 target per row. |
| Size/family shift (train mostly 4 to 5 variables, test 6 to 18, half transport) | 30% of seeds have ≥ 6 variables | Up-weight accepted and seed targets with ≥ 6 variables (4 permuted views against 2). EI coverage on transport rows is the key diagnostic. Allow long outputs (`max_new_tokens` 768 at test). Watch for truncation: assert EOS was reached and log the count. |
| Under-capacity (0.5B on 12 to 18-variable tables) | Principle "under-sizing loses" | Capacity rung experiment (Qwen2.5-1.5B, embeddings frozen) as roadmap step 3, gated on dev gain and runtime. |
| Ref-copy errors | Refs are random per problem | Inline annotation (`0.05 [x7]`) puts the ref next to its number. Random ref permutation (4 views) forces copying rather than index memorisation. |
| Imitating noisy references | Helper-variable seeds | This is metric-optimal for S and V. Do not "clean" seeds by hand: that would be hand-editing the target distribution. |
| Selection-on-OOF optimism | Many dev comparisons | Paired folds, sanity holdout, no fitted decode weights. |
| fp16 overflow in Qwen activations | Known for some Qwen sizes [UNVERIFIED for 0.5B] | Assert finite loss/logits each step and log max-abs activations once. If they overflow, switch dev and ship together to bf16 (A10G native). |

## Recommended approach (primary + fallback)

### Primary (clean): Qwen2.5-0.5B full fine-tune → 2 rounds of expert iteration (rejection sampling vs optimal_value) → fixed-K sampling + solver-checked MBR selection

**Model.**
- `Qwen/Qwen2.5-0.5B` (base, not Instruct), pinned `revision="060db6499f32faf8b98477b0a26969ef7d8b9987"`.
- Apache-2.0, ~1 GB, fp32 master weights fit easily on 24 GB.
- The tokenizer has `<=`, `>=`, `;`, single digits and byte-level ¶ [VERIFIED]. T5/flan-T5 lack `<` (coordinator), so
  they would need an output-mapping layer, which is grey "repair". Rejected.
- Coder variant `Qwen/Qwen2.5-Coder-0.5B` @ `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301` is a roadmap A/B.
- Why 0.5B as primary:
  - it is the only rung that runs both EI rounds inside the budget with ≥ 30% headroom;
  - it fits the Kaggle T4 dev GPUs with the *same* precision path (fp32 master + fp16 autocast + GradScaler), which
    makes dev numbers transfer (lever 16);
  - 1.5B full FT needs bf16 parameters and ~18 GB, so it is A10G-only and has one EI round.

**Input serialisation** (hand-built *inputs*, explicitly allowed):
1. Replace ¶ with `\n` so Markdown tables look like pretraining text.
2. After every number token in the text (description's number definition; % gives a fraction), append its ref in
   brackets: `$2 [x8]`, `0.05 [x7] hours`, `25% [x3]`. Lossless: 100% coverage verified.
3. Append a `Numbers:` line `x0=0.2; x1=100; ...` in the given order. This is redundant; the dev A/B tests "inline
   only".
4. Prompt template: `### Problem:\n{annotated}\n### Formulation:\n`. Target is the formulation + EOS. Loss on target
   tokens only.

**Augmentation** (real labelled units, isomorphic relabelling, not synthetic):
- Random permutation π of `0..N-1` applied consistently to the inline annotation, the numbers line and every `x<k>` in
  the target.
- Views per example: seeds 4; accepted EI targets 2, or 4 if the target has ≥ 6 variables.
- View 0 is the identity.
- Permutations come from a seeded `numpy.random.Generator`.

**Stage 1 (SFT on seeds).**
- 248 seeds × 4 views, 3 epochs.
- AdamW, LR 5e-5, cosine, 3% warmup, weight decay 0.0, grad clip 1.0.
- Effective batch 16 sequences (micro 8 × accumulation 2), max length 1,280 tokens.
- Assert no truncation. Assert the parameter-change norm is > 0 after step 1 (lever 1).

**Stage 2 (EI round 1).**
- For all 2,229 non-seed train rows, sample K = 4 per row (temperature 0.8, top-p 0.95, `max_new_tokens` 512).
- One fixed seed per batch, `torch.manual_seed(SEED + batch_idx)` before each `generate`.
- Parse and solve each candidate with HiGHS.
- **Accept** if parseable, linear, within the size caps, and |opt − label| ≤ 1e-4·max(1, |label|).
- **Then filter:** draw 3 factor sets for the *train* row (same recipe as the grader, seeded per row). Keep only
  candidates whose optimum is finite on all 3.
- If several distinct accepted programs remain, cluster by the vector of 3 perturbed optima (rel 1e-6). Keep one
  representative of the largest cluster; tie-break by mean token logprob, then candidate index.
- Seed rows keep their seed. Never replace a seed by a sample.

**Stage 3 (SFT round 2).** Restart from the base weights. Train on seeds ×4 views + accepted targets ×2/×4 views,
2 epochs, same optimiser.

**Stage 4 (EI round 2).** Sample K = 4 only for non-seed rows still without an accepted target. Same acceptance and
filter. Rows accepted in round 1 keep their round-1 target.

**Stage 5 (final SFT).** Restart from base. Seeds ×4 + all accepted ×2/×4, 2 epochs. This is the shipped model,
trained on 100% of train.

**Stage 6 (test decode).** Per test row: 1 greedy + 8 samples (temperature 0.7, top-p 0.95, `max_new_tokens` 768).

**Stage 7 (selection among the model's own outputs, per row).**
1. Parse each candidate. Keep those that are valid with a finite real-number optimum.
2. Score each valid candidate by the expected-utility proxy:

   `U(c) = mean over other valid c' of [ 1{opt(c) ≈ opt(c') at rel 1e-4} + S(c, c') ] / 2`

   This is MBR with the model's own sample distribution as the posterior (A3; S-tier §1.1).
3. Pick argmax U; tie-break by mean token logprob, then by candidate index (greedy first).
4. If no candidate is valid, submit the greedy output **verbatim** (logged). No repair and no constant fallback.

### Fallback (clean)

Same pipeline with **one** EI round (stages 4 and 5 removed; the stage 3 model is shipped). Test decode K = 4.
Estimated ~30 min. Use it if profiling shows the primary above ~55 min on an A10G, or if round 2 shows no paired gain.

### Capacity alternative (roadmap step 3; becomes primary if it wins)

- `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323`, full fine-tune with the tied token embeddings
  frozen.
- Memory: bf16 params + fp32 Adam states ≈ 16 GB; 233M frozen embedding params save ~2.8 GB.
- One EI round with K = 3, test K = 6. Estimated ~60 min [UNVERIFIED].
- Ship only if paired M3 gain ≥ +0.03 (beyond noise) and the measured A10G runtime stays ≤ 63 min (30% headroom on
  90 min).
- LoRA is a weaker fallback for this rung (E6: full FT > LoRA on small data).

### Compliance-adjusted ranking

All three candidates are equally clean. They rank by expected private score × runtime safety. The grey test-time
perturbation vote is the only grey lever; it needs reviewer approval plus a ≥ +0.02 M3 margin.

## Rejected options

- **T5 / flan-T5:** no `<` in the vocabulary. Mapping `<=` through placeholder tokens is post-hoc output rewriting
  (grey). Its sentencepiece also handles digits and tables poorly. BART and CodeT5+ (byte-level BPE) are possible
  diversity arms but weaker pretrained reasoning; not planned.
- **Prompting / few-shot / retrieval-in-prompt:** banned.
- **Grammar-constrained decoding** (masking tokens to the formulation grammar, or restricting refs to existing ones):
  arguably a "hand-written parsing rule producing formulations". Grey with little gain, because validity filtering
  among K samples recovers most of it. Ask before adopting.
- **Template / slot-filling pipeline** (classify constraint types, fill slots from tables): needs hand templates, and
  strip-the-ML risk. Banned in spirit.
- **Hand-cleaning noisy seeds:** hand-editing targets. It also hurts S against noisy test references.
- **Canonical renumbering of refs in the prompt with an inverse map applied to the output:** a deterministic bijection,
  but it is post-processing of model output. Inline annotation with original refs plus permutation augmentation gives
  the same benefit with zero output edits.
- **Adaptive test-time resampling** ("sample until a finite optimum"): eval-time search. Fixed K only.
- **Same-template sibling counterfactuals as an EI filter:** only 25 rows share a masked template [VERIFIED]. Not
  available.
- **RL with policy gradients (GRPO/PPO) on V reward:** does not fit the budget beside EI. EI is its cheap, stable,
  deterministic-plan special case.
- **Up-sampling transport rows by detecting the keyword at test time / per-family models:** keyword routing at test is
  a hand rule. Train-side weighting by *target size* is used instead.
- **Typed optimum, or typed literal numbers:** impossible at test (no optimum), and the latter forfeits C.
- **In-script K-fold CV of the whole LM pipeline:** 5× the runtime. Dev-time folds only. The shipped script trains once
  on 100% of train.

## Fixed work plan & runtime budget

**Constants (top of file, no branching):**
- `SEED = 42`
- `MODEL_ID = "Qwen/Qwen2.5-0.5B"`, `REVISION = "060db649..."`
- `VIEWS_SEED = 4`, `VIEWS_EI = 2`, `VIEWS_EI_LARGE = 4`
- `EPOCHS_S1 = 3`, `EPOCHS_S3 = 2`, `EPOCHS_S5 = 2`
- `LR = 5e-5`, `MICRO_BS = 8`, `ACCUM = 2`, `MAX_LEN = 1280`
- `K_EI = 4`, `T_EI = 0.8`, `TOP_P = 0.95`, `MAXNEW_EI = 512`, `GEN_PROMPTS_PER_BATCH = 64` (→ 256 sequences)
- `K_TEST = 8`, `T_TEST = 0.7`, `MAXNEW_TEST = 768`, `TEST_PROMPTS_PER_BATCH = 32`
- `N_PERTURB = 3`, `MILP_NODE_LIMIT = 50000`
- `NUM_WORKERS = 2`, `torch.set_num_threads(4)`, solver pool fixed at 4 processes with ordered `map`
- `device = "cuda"` hardcoded

**Determinism rules.**
- No `time` in conditions. No `cuda.is_available` switches. No try/except import fallbacks.
- No library `time_limit`. HiGHS `node_limit` is deterministic; log how many solves hit it.
- `cudnn.deterministic = True`; `use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG` set
  before importing torch.
- Left padding with an attention mask. Batches are length-sorted deterministically.

**Estimated A10G runtime, 0.5B primary [UNVERIFIED].** Basis: ~0.045 s per 520-token training sequence at ~35 TFLOPS;
~40 to 45 ms per decode step at 256 sequences.

| Stage | Work | Est. |
|---|---|---|
| 0 Load, annotate, seed oracle check (assert 248/248 solve to labels) | download ~1 GB | 1.5 min |
| 1 SFT seeds | 992 seq × 3 ep | 4 min |
| 2 EI-1 sampling | 2,229 × 4 = 8,916 seq, ~35 batches × ~300 steps | 9 min |
| 2b solve + 3 perturbations | ~9k + ~10k small MILPs | 3 min |
| 3 SFT round 2 (from base) | ~3,000 seq × 2 ep | 6 min |
| 4 EI-2 sampling + solve | ~1,200 × 4 = 4,800 seq | 6.5 min |
| 5 final SFT (from base) | ~3,800 seq × 2 ep | 7.5 min |
| 6 test decode | 345 × 9 = 3,105 seq, ≤ 768 new tokens | 6 min |
| 7 MBR selection | ~3.1k solves + ~12k pairwise S matchings | 2.5 min |
| **Total** | | **≈ 46 min** (≈ 49% headroom vs 90 min; within CLAUDE.md's 50-min target) |

**Memory.** Training 0.5B with fp32 master and Adam: 8 GB + activations ≈ 12 GB at micro-batch 8 × 1,280 (enable
gradient checkpointing if needed; it is fixed, not conditional). Generation: KV cache ~12 KB/token × 256 × 1,300 ≈
4 GB.

**What to cut if the A10G profile is over ~55 min.** Hardcode the cuts; never branch on time. In order:
1. `K_TEST` 8 → 4 (−3 min).
2. Remove EI round 2 (stage 4) and stage 5 by reusing the stage 3 model (−13 min). This is the fallback.
3. `VIEWS_SEED` 4 → 3 and `EPOCHS_S1` 3 → 2 (−2 min).
4. `K_EI` 4 → 3 (−2 min).

Never cut the oracle check, the validator or the seed SFT.

**Validation inside the script.**
- Inputs: schema, ref coverage = 1.0, refs 0..N−1.
- Seed oracle: 248/248 solve to labels, else raise.
- Outputs: ids equal `sample_submission` ids, no duplicates, no empty strings (reload with
  `keep_default_na=False`).
- Log the counts of candidates that are valid, finite and truncated, and the "no valid candidate" rows.
- Atomic write.

## Metric-aware training & decode

1. **Loss.** Token cross-entropy on the target only: the standard proper likelihood for a generative model. The case
   is the metric unit; a sum over tokens upweights long programs, which suits the ≥ 6-variable test. Views give size
   weighting: ×4 for targets with ≥ 6 variables, against the description's stated shift.
2. **Weak supervision → targets (EI / STaR-style rejection sampling).** V-exact acceptance is the training-side "value
   term" filter. The perturbation-finiteness and perturbation-agreement filter is a reference-free proxy for the C term
   (lever 10b: fold the model's own verified outputs back into training).
3. **MILP solver settings.** Use scipy `milp` defaults for `mip_rel_gap` unless seeds show disagreement. In-script, solve
   all seeds under the default gap and under 1e-9. Use the setting whose optima match labels at rel 1e-6 on more seeds
   (train-only evidence; no hand-picked constant). The scratch check with 1e-9 matched 248/248 at 1e-4.
4. **Decode.** Fixed K; validity filter; MBR under a utility built from the metric's own terms: V agreement plus the S
   matcher among candidates. C agreement via self-perturbation is grey and off by default. No fitted constants, so no
   calibration step or cross-fit is needed.
5. **Grammar the selector accepts must be at least as strict as the grader's:** objective first; one relational
   operator per constraint; `int`/`bin` lists; ≤ 60 variables and ≤ 150 constraints; no `q<k>` / `x<k>`-shaped variable
   names; at most one variable side per product or quotient. The description's example `x4cake_a` (no `*`) is
   ambiguous. Seeds always use `*`, so the model will too. The validator flags any candidate containing `x\d+[A-Za-z_]`
   as invalid.
6. **Oracle checks.**
   - Gold seeds through parser + solver reproduce labels (done: 248/248).
   - Gold seeds through the dev metric give case score 1.0.
   - The MBR selector, given the gold seed among 8 random wrong candidates, must not systematically reject it. Report
     the rate at which it picks gold.

## Structural signals

- **Ref naming is arbitrary** (random per problem) → permutation augmentation. Verify on seeds that permuting refs in
  input and target preserves the solved optimum (assert in dev).
- **Value → ref bijection** (`numbers_json` dedupes) → inline annotation makes grounding a local copy decision.
  Literal 0/1/−1/100 are the only legal non-refs (description) → the model learns when to type 1/100 (seeds show 42
  such literals).
- **Variable naming and row order do not matter** to the metric → no need to canonicalise outputs. Do not shuffle row
  order in targets by default: text-order generation is easier to learn. Shuffled order is a dev A/B only.
- **Every reference has a finite optimum**, and test references keep it on ≥ 3 counterfactual variants → finiteness
  under perturbation is a property of good programs. Use it as an EI filter (train) and as a validity filter
  (real-number finiteness only, at test).
- **Table vs prose twins inside groups** (313 groups) → the model sees the same structure in both modalities. No extra
  handling, but keep twins in the same fold.
- **Integrality:** 83% of seeds declare `int`. An LP optimum that happens to be integral makes a missing `int`
  invisible to V but visible to S. The perturbation filter catches some of these on train. Acceptance precision on dev
  seeds measures the rest.

## Experiment roadmap

Each step: paired on folds 1 to 2 (cheap) and then 1 to 5 (final candidates). Log to `reports/experiment_log.csv`.

1. **Contract and metric (no GPU).**
   - Parser, solver, perturbation generator, S matcher.
   - Unit tests: yardsticks reproduced in order on held-out seeds.
   - Seed oracle 248/248.
   - Fold builder with stratification printout.

   Stop when all tests pass.
2. **Baseline:** 0.5B, Stage 1 only (seeds SFT, inline annotation) + greedy decode.
   - Report validity, M1, M2, M3.
   - Expected: validity ≥ 85%, M1 V ~0.2 to 0.35, M3 ~0.2 to 0.3 [UNVERIFIED].

   Stop if validity < 70%: fix the input/format first.
3. **Representation A/Bs, cheap config.**
   - (a) inline annotation vs numbers list only;
   - (b) ref-permutation views 1 vs 4;
   - (c) Qwen2.5-0.5B vs Coder-0.5B;
   - (d) **capacity:** 1.5B (frozen embeddings), on an A10G-class GPU only.

   Keep winners by the paired rule.
4. **Expert iteration.**
   - Round 1, then round 2 (full config).
   - Report coverage (pass@4), acceptance yield by slice (transport, ≥ 6-var proxy), and acceptance precision on
     held-out seeds with and without the perturbation filter.

   Keep a round only if M3 gains beyond noise.
5. **Decode.**
   - Greedy vs K = 8 + V-vote vs K = 8 + MBR (V + S).
   - Report pass@K versus selected-hit rate (ranking gap).
   - The perturbation-vote MBR is gated on reviewer approval.
6. **Size-shift levers.** ×4 views for large targets vs uniform; `max_new_tokens` / truncation audit on long transport
   rows.
7. **Final fixed plan:**
   - full run on A10G (or the closest available) with a timed profile;
   - run twice and diff (expect identical or near-identical strings; report the % of rows identical);
   - half-rows test;
   - compliance scan, validator, reviewers.

Credits: baseline (step 2 shipped as a valid end-to-end script), best single (after step 4), final (after step 7).

## Compliance audit

| Check (CLAUDE.md §7 + challenge) | Status |
|---|---|
| Test rows used only for per-row inference | Yes. Prompt per row; selection among that row's own candidates only. |
| Clock / hardware / environment branching | None by design. Time is logged only; HiGHS node_limit is fixed, with no time_limit. |
| Hard-coded tuned constants | Only fixed-plan counts and LR defaults. MILP gap chosen in-script from seeds. No decode weights. |
| External data / weights | HF Qwen2.5 base pinned. No OR-tuned checkpoints, no web lookups. |
| Strip-the-ML | Passes: without the LM there is no formulation. |
| Output repair / templates | None: outputs are submitted verbatim; the parser only reads. |
| `case_id` / row order as signal | No. |
| Training genuinely load-bearing | Yes: 3 SFT stages; parameter-change norm logged; EI improves over seeds-only (measured). |
| Docstring requirements map | Required: one line per ban above → where it is satisfied. |
| Source size / readability | Plain Python, est. 1,000 to 1,500 lines, ≪ 512 KB. |

## Open questions & assumptions

1. **Runtime limit.** The description says only "Compute: A10G"; the coordinator states 90 min. The plan targets
   ≈ 46 min. Confirm the limit.
2. **AI baseline score** is not printed in the description. Needed to judge payout odds.
3. **Test-time self-perturbation vote** (grey): is solving the model's own candidates on randomly perturbed copies of
   an evaluation case's numbers, to choose among them, allowed? Off by default; also ask the reviewer.
4. **Grammar-constrained decoding:** "hand-written parsing rule producing formulations" or a legitimate decoder
   constraint? Assumed banned; not used.
5. **Pretraining contamination:** is Qwen2.5 base acceptable given that the published collection may be in web
   corpora? Assumed yes (the description names open-weight decoders).
6. **Uploading public data to Kaggle** for dev GPU runs: confirm it is not considered copying the data elsewhere.
7. **`x4cake_a` in the description example:** implicit multiplication, or a typo? Assumed `*` is required. The model
   follows the seeds' `*`.
8. **Grader's S details** ("first coefficient positive" ordering; coefficient tolerance) and its HiGHS options: unknown.
   Our proxy is approximate (slightly pessimistic).

**Expected score (ESTIMATE, not a promise).**
- Primary 0.5B: private **0.28 to 0.45**, central ~0.36.
  - Reasoning: V on test-like problems ~0.30 to 0.45; C ≈ 0.85 × V; S ≈ 0.40 to 0.55, since partial rows count even
    when V misses.
  - It clears the 0.0088 floor by a wide margin. Beating the 0.3368 value-only yardstick is roughly a coin flip for
    0.5B.
- The 1.5B rung, if it fits, is estimated +0.03 to +0.08.
- The 0.6667 typed-numbers level needs near-perfect structure and is out of reach for this budget.

**Could not verify:** any GPU runtime, fp16 stability of Qwen2.5-0.5B, EI acceptance yield and precision, the grader's
exact S matcher and solver options, and HF download speed on the grading host.
