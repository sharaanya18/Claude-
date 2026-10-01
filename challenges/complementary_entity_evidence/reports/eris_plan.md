# Eris plan: Complementary Entity Evidence Selection (Marginal Role Coverage@2)

Status: written before any data was available (`dataset/public/` is empty). Every number below that does not come
straight from CHALLENGE.md is a **hypothesis marked UNVERIFIED** and has a named diagnostic that confirms or
refutes it. Families: retrieval-ranking-slates + structured-assignment (choose a pair) on a text-nlp core
(anchor-conditioned relation-role tagging).

---

## Contract & decision unit

**One valid answer.** `submission.csv` with columns exactly `id,prediction` (that order), 1,197 rows, every test id
once, no extra columns, no NaN, every prediction a finite float. A JSON payload is also parsed by `_numeric`, but we
always write plain floats. Scale is irrelevant. Only the **top-2 per slate** matters; positions 3..n never count.

**Invalid vs low.** Invalid (rejected): wrong columns/order, missing, duplicate or unknown ids, NaN, non-finite. Low but
valid: any finite ranking. Exact score ties are broken by ascending id inside the grader. We always emit strictly
distinct scores, so the id tie-break never decides anything (we must not rely on it, ban: "deterministic candidate
order").

**Metric, term by term.** For slate q with candidates C_q (|C_q| = slate_size in 2..8), hidden new-role set
R_c = Inc(c) \ Known(q), where Inc(c) is the candidate's incident directed relation-role inventory for the anchor
mention and Known(q) is the seed's full inventory (`known_roles`). The selected pair is (a,b) = the top-2 by score.
Score_q = |R_a ∪ R_b| / max_{x<y} |R_x ∪ R_y|. The final score is the unweighted mean over slates (every slate weighs
the same whatever its size or optimum). Properties:
- It depends only on rank: no calibration in the output, but the *internal* pair choice needs calibrated per-role
  probabilities (see Metric-aware training & decode).
- The denominator is constant within a slate, so the decision is "pick the best unordered pair out of C(n,2) <= 28".
- Repeated roles count once, so complementarity matters: two candidates that both add `out:X` are worth one role.
- Slates with n = 2 score 1.0 for every submission. Slates with optimum 1 only need one candidate with any new role.
- Retention rule: every slate has optimum > 0 and non-constant pair utilities, so the random-pair expectation is < 1.

**True independent unit.** The normalised anchor/type **cohort** (256 train, 59 test). Slates (772 / 164) share cohorts
(about 3.0 slates per cohort in train, 2.8 in test) and candidate sentences recur across slates of the same cohort
(5,696 rows / 3,431 distinct sentences = 1.66 appearances per sentence; test 1,197 / 743 = 1.61). The hidden split and the
public/private split are both cohort-level. Public ≈ 1/3 of 59 cohorts ≈ 20 cohorts ≈ 55 slates; private ≈ 39 cohorts ≈ 109
slates. The private noise is therefore large (see Validation design).

**Pipeline stages, diagnosed separately.**
1. *Candidate coverage*: trivially 100% (every legal answer is a pair from the slate). Reducing the task to "choose an
   unordered pair" loses no valid answer (verify: the gold EU decode reproduces 1.0 on every train slate).
2. *Scoring*: per-candidate, per-role probability p_{c,r} = P(r ∈ Inc(c) | candidate text, anchor span, anchor_type),
   then hard mask r ∈ Known(q) to 0. Measured by masked OOF log-loss / per-role AUC.
3. *Decoding*: pair choice by expected union (exact, closed form), measured by the gap "decode with gold sets"
   (must be 1.0) vs "decode with OOF probabilities".

**The key factorisation (verify on train, D4).** By definition R_c depends on the seed only through the mask Known(q).
Inc(c) is a property of the candidate sentence plus its anchor mention, identical in every slate where it appears. So
the learnable object is an **anchor-conditioned multi-label role tagger at sentence level**. The seed text carries no
further information about R_c beyond `known_roles` (and an entity-level prior, handled in roadmap step 4b). This is
why "lexical similarity to the seed" is a weak, possibly negative signal, as the description warns.

---

## Compliance regime

**Domain / regime.** NLP + learned ranking. The description explicitly allows "pretrained general-purpose language
representations and learned ranking/role models" and requires that "the complete pipeline must perform meaningful
training and fit the declared budget" (A10G, **90 min training + inference**). Default acceptance (platform-facts §5): a
fine-tuned transformer is clean; TF-IDF / n-grams / regex and hand features with an off-the-shelf ranker are grey; an
inference-only pipeline is banned.

**Explicit bans in the description, and how the plan honours each:**

| Ban (quoted/paraphrased) | Plan |
|---|---|
| "Train using the public training data. Use test candidates only for inference." | Test rows are read only for: ids, the slate key (middle id part, explicitly allowed "for grouping and output alignment"), per-row inference inputs (`candidate_text`, `anchor_start/end`, `anchor_type`, `known_roles` as the mask). **Test seeds (`query_text` + `known_roles`) are NOT used as training examples**, even though they look labelled. No tokenizer/vocab/scaler fitted on test. |
| No generated sentences, synthetic labels, pseudo-labels, external examples | Training examples are the real train candidate sentences and the real train seed sentences only. No pseudo-labels on test. No entity-swap augmentation that writes new sentences (see Open questions Q4). |
| "Do not reverse-map opaque identifiers or treat the deterministic candidate order as target evidence." | Ids are used only to split off the slate key and to align output. No id features. Candidate key is not used to link appearances (we link by exact `candidate_text` + span instead). Row order is canonicalised away (candidates sorted by `candidate_text` inside a slate before decoding; a unit test shuffles rows and asserts the same output). |
| Do not hardcode example scores or the training `oracle_rank` tie choices | `oracle_rank` is never a training target. It is read only in the metric unit test (the oracle pair must score 1.0). |
| No web, annotation mirrors, external label matching, leaderboard probing | Closed data. HF weights only (pinned revision). No constants derived from LB feedback. |
| "Must perform meaningful training", 90-min A10G | Full fine-tune of a pretrained encoder (load-bearing; the strip-the-ML test below gives the random baseline without it). Fixed plan ≈ 40 min. |

**CLAUDE.md bans also apply:** no wall-clock branching, no env/hardware fallbacks, no test fitting, seeds fixed, HPO
(here: epoch-count selection and calibration) done in-script on train only.

**Self-audits on this plan.**
- *Strip-the-ML test.* Remove the trained tagger: all candidates in a slate share `anchor`, `anchor_type`,
  `query_text`, `known_roles`. Without a model nothing distinguishes them, so the only output is a constant, which
  falls back to the id tie-break, i.e. a random-like pair. The mask and the EU decode alone score nothing above random.
  **Passes clearly.** Report the number (D3: random-pair expectation and id-order constant baseline).
- *No whole-test aggregation.* A slate's prediction is a function of its own rows plus the frozen train-fit model. No
  cohort-level pooling over test slates (e.g. pooling the `known_roles` of all test seeds of one anchor would be a strong
  but **banned** entity prior). No test-batch normalisation. Half-rows test at slate granularity (`half_rows_test.py`):
  drop half the slates and the kept slates' predictions must be identical. The MC variant (roadmap) seeds its random
  numbers from slate content, not from the slate's position in the file.
- *Sibling leakage.* The target *is* a relation between slate-mates (pair union). The only cross-candidate computation
  is the expected-union decode on the model's own per-candidate probabilities, i.e. the metric itself under the model
  posterior. No feature reconstructs a candidate's label from its mates' labels (none are visible at test time anyway).
  Label aggregation across *train* appearances of one sentence uses train labels only and is grouped inside one cohort,
  hence inside one fold.
- *Hard-coded constants.* Recipe defaults (LR, batch, warmup, max length from a train-only length percentile) are
  documented literature defaults. The epoch count and the calibration parameters are selected in-script on OOF of
  cohort folds. Nothing comes from LB feedback.
- *Training is load-bearing.* Log the parameter-change norm of the encoder after fine-tuning, the frozen-probe yardstick
  vs the fine-tune on the same folds, and per-role learned head biases.

---

## Data findings

**Nothing below is verified: the dataset directory is empty.** These are the counts in the description, plus the
diagnostics to run first (train only; scratch notebook, not in `solution.py`), each with its expected outcome.

Given (description): train 5,696 rows / 772 slates (mean size 7.38) / 3,431 distinct sentences / 256 cohorts; test
1,197 / 164 (mean 7.30) / 743 / 59. Columns: `id, anchor, anchor_type, query_text, known_roles, candidate_text,
anchor_start, anchor_end`; `train_targets.csv`: `id,target` with JSON `{query_id, roles, slate_size, oracle_rank}`.

| # | Diagnostic (train only) | Hypothesis (UNVERIFIED) | Why it matters |
|---|---|---|---|
| D1 | Schema; dtypes; `id` regex `^ev_[0-9a-f]+_[0-9a-f]+$`; middle key ↔ `target.query_id` 1:1; rows per slate == `slate_size`; slate-size histogram | Mostly 8-candidate slates, a tail of 2–7 from small cohorts | Grouping key at test must be the id middle key (the grader groups by `query_id`) |
| D2 | Role vocabulary from train `known_roles` ∪ train `roles`; K; per-role frequency in targets vs in known; direction format (e.g. `in:`/`out:` + category) | Small closed vocabulary (≈ 8–40 directed roles); heavy-tailed, with "rare roles" (description) | Head size; rare-role calibration; test known roles outside the vocab are simply ignored |
| D3 | Per slate: optimum distribution; exact random-pair expectation mean_{pairs}(U/opt); id-order constant baseline; **count-greedy oracle** (top-2 by gold \|R\|, ignoring overlap); worst-pair score | Optimum mostly 1–3; random ≈ 0.55–0.70; count-greedy oracle ≈ 0.9–0.97 (< 1 because of overlap) | Floor; size of the complementarity effect; decode value |
| D4 | Group rows by exact (`candidate_text`, `anchor_start`, `anchor_end`): appearances per sentence; does each sentence always carry the same `anchor`/type (one cohort)? **Contradictions**: r ∈ R in one appearance and r ∉ Known ∪ R in another | 1.66 appearances on average; 0 contradictions; each sentence in exactly one cohort | Validates the factorisation Inc(c) = seed-independent; enables label aggregation |
| D5 | Censoring: fraction of (sentence, role) cells whose label is unknown after aggregation (r ∈ Known in *every* appearance), per role | High for the cohort's dominant roles (e.g. a person anchor's most common role) | Masked loss is mandatory; seeds as fully labelled examples recover the censored roles |
| D6 | Seeds: distinct `query_text` count; seeds per cohort; seed ∩ candidate texts = ∅ (description says disjoint); can the anchor be located in `query_text` (exact, then casefold) and how often more than once | ≈ 772 distinct seeds, ~3 per cohort; anchor found in > 95% | ~+20% fully labelled training sentences (labels = `known_roles`, nothing censored) |
| D7 | Cohort key = `NFKC(anchor).casefold().strip()` + "\|" + `anchor_type`; count == 256? Slates and rows per cohort; largest cohort share | 256 confirmed; a few large cohorts (common anchors) | Fold balance; giant-cohort cap is not needed if < 10% |
| D8 | `candidate_text[anchor_start:anchor_end] == anchor` (exact / casefold); anchor at first occurrence? Multiple occurrences? | ~100% match; mostly the first occurrence | Marker insertion; seed anchor-location rule must imitate the candidate rule |
| D9 | Token lengths with the chosen tokenizer (p50/p95/p99/max) of `candidate_text` with markers | Short sentences (read speech; p99 < 96 tokens) | MAX_LEN (no truncation of the anchor context); runtime |
| D10 | Role × `anchor_type` contingency on train labels | Strong type → role compatibility (some roles never occur for some types) | The model must *learn* it via the type marker. **Do not hard-mask by type** |
| D11 | Near-duplicate sentences across cohorts: `make_groups.py train.csv --keys cohort --text candidate_text,query_text --jaccard 0.8` (cohort column added in scratch) | Few cross-cohort near-dups; no giant component | If many: a stricter fold variant, reported beside the cohort folds |
| D12 | Shortcut controls: within-slate Spearman of gold \|R\| vs candidate length, vs frozen-embedding cosine to the seed, vs anchor position | Length weakly positive; similarity to the seed ~0 or negative (description) | Yardsticks; make sure the model is not just a length scorer |
| D13 | Role co-occurrence within Inc (from seeds, fully labelled): pairwise lift | Some roles co-occur (both directions of one event) | Justifies a shared multi-label head; possible factorised head later |

Irreducible ambiguity: exact duplicate sentences were removed upstream, so identical-input/different-label pairs should be
0 (D4 checks this). The information ceiling is the role tagger's accuracy on unseen anchors plus annotation omissions
(description: "annotation omissions … limit coverage"), which cap achievable precision.

---

## Validation design

**Hidden split reconstruction.** Whole anchor/type cohorts go to train or test, and no sentence crosses. The test has 59
cohorts / 164 slates. Mirror: **GroupKFold by cohort, 5 folds**. Each fold holds about 51 cohorts / 154 slates, close to
the test size (so fold-level noise ≈ test noise). Stratify cohorts by `anchor_type` (StratifiedGroupKFold, seed fixed) and
balance slate counts. All appearances of a sentence and its seeds are in the same cohort, hence the same fold. The
label aggregation (D4) is computed *inside* each training split only (validation cohorts never contribute labels to
training targets).

**Groups beyond the given key.** Union-find of cohort with cross-cohort near-duplicates (D11) as a *stricter variant*. If
it changes fold scores by more than noise, report both; ship decisions on the cohort split (it is what the test uses).

**Repeats and noise rule.** Offline decisions: 5 folds × 3 split seeds, paired on identical folds. Accept a change only if
the mean paired gain > 1 SE of the paired fold differences and it is positive in ≥ 2 of the 3 seeds. In-script: 1 split
seed (OOF for epoch selection, calibration and the reported CV). Expected noise (UNVERIFIED, assuming per-slate SD ≈ 0.2
and a design effect of ≈ 1.5 from cohort clustering): CV over 772 slates SE ≈ 0.011; private (~109 slates) SE ≈ 0.025;
public (~55 slates) SE ≈ 0.035. **Do not chase the public LB.**

**Metric re-implementation.** Paste the official `grade` code verbatim. Build OOF "answers" from `train_targets.csv`
and score OOF predictions with it. Unit tests: (1) gold-set EU decode → exactly 1.0 on all 772 slates; (2) oracle_rank as
prediction (−rank) → 1.0; (3) worst-pair → reported minimum; (4) constant 0 → id-order baseline; (5) 200 random
permutations → mean equals the analytic random-pair expectation to 1e-3; (6) shuffling rows inside a slate leaves our
decode unchanged.

**Nested / cross-fitted selection.** Three selection steps, each with its own check:
1. *Epoch count e\** chosen in-script by mean masked OOF BCE over the 5 folds (evaluated at end of epochs 2..6 of one
   fixed 6-epoch schedule). Report the slate metric at e\* and at fixed epoch 4 side by side. Selection optimism is
   expected ≤ 0.005.
2. *Calibration* (K per-role biases + 1 temperature) fitted on OOF. Cross-fit: fit on folds {1,2,3} → score {4,5} and
   swap; print the cross-fitted slate score next to the in-sample one. Applied only if the cross-fitted masked log-loss
   improves on all splits (a data-determined, deterministic decision).
3. *Any blend weight* (roadmap step 5): same cross-fit.

**Sanity holdout (offline dev only).** Hold out about 15% of cohorts (≈ 38 cohorts) for the whole of development. Look at
it once before the final freeze and once to compare "full refit" vs "fold-average" shipping.

**Bias direction of each proxy.**
- Cohort-grouped CV: right axis. Slightly **pessimistic**, because it trains on 80% of cohorts vs 100% for the refit
  (≈ −0.005).
- In-sample calibration / epoch selection: **optimistic** by ≤ 0.01. Cross-fitting removes it.
- Row-level or slate-level random CV: strongly **optimistic** (the same sentence appears in train and validation, and
  the anchor's role prior is memorised). Never use it.
- Overall expected private ≈ cohort-CV ± 0.03 (noise dominates any bias).

---

## Overfit/underfit risks

| Risk | Kind | Evidence / size | Mitigation |
|---|---|---|---|
| Memorising anchor names / cohort role priors (test anchors are unseen) | overfit | 256 cohorts, ~13 sentences each | Cohort CV. Typed entity markers so the type, not the name, carries the prior. Variant (roadmap 3c): typed anchor *mask* replacing the name in train **and** test (a deterministic representation, not augmentation). Keep it if paired CV ≥ +1 SE |
| Few independent groups vs a 435M-param encoder | overfit | 256 cohorts, ~4.2k sentences | Fixed short schedule (≤ 6 epochs), LLRD 0.85, dropout 0.1, weight decay 0.01, e\* selected in-script, 2-seed refit average. The base model is the fallback rung |
| Censoring bias (labels of the cohort's dominant roles are missing more often) | bias | D5 | Masked BCE only on observed cells. Train seeds as fully labelled examples. Aggregate labels across appearances |
| Repeated sentences over-weighted | overfit | 1.66 appearances / sentence | One training example per distinct (sentence, span). Optional sqrt(appearances) weight as a measured variant |
| Many tuned knobs on 772 OOF slates | overfit | — | Only e\* plus K+1 calibration scalars. No decode constants in the primary (the EU decode is parameter-free) |
| Independence assumption in the EU decode (near-paraphrase candidates are correlated) | decode error | D11/D12 error analysis | Roadmap 4c: learned pair-redundancy term, only if error analysis shows duplicate-pair failures |
| Too-small representation | underfit | Literature: large encoders +2–4 F1 on sentence-level RE | DeBERTa-v3-**large** primary; base only if profiling breaks the budget |
| Losing anchor identity/position (mean pooling over the sentence) | underfit | — | Typed entity markers + marker-token pooling (concatenated with CLS) |
| Truncation | underfit | D9 | MAX_LEN from train p99 + margin, asserted; never truncate the anchor window |
| Loss mismatched to the metric | underfit | — | Proper scoring rule (BCE) → calibrated p → exact expected-union decode. Not a listwise loss on 772 slates in the primary |
| Rare roles | underfit | D2 | Base-rate bias init; no pos_weight (it would break calibration); per-role calibration bias |

---

## Recommended approach (primary + fallback)

### Primary (clean): fine-tuned anchor-marked role tagger + expected-union pair decode

1. **Training examples (train only).**
   (a) One example per distinct train candidate (`candidate_text`, `anchor_start`, `anchor_end`). Label vector over the
   K train roles with three states, aggregated over its appearances j: positive if r ∈ ∪_j roles_j; negative if
   r ∉ Known_j and r ∉ roles_j for some j; unknown (masked) otherwise. Assert 0 contradictions (D4).
   (b) One example per distinct train **seed** sentence: anchor located in `query_text` by exact substring (casefold
   fallback, first occurrence, same rule D8 confirms for candidates). Label = `known_roles` (positive), all other roles
   negative: fully observed. Seeds whose anchor cannot be located are dropped (count logged).
   This is A1-style "reconstruct more signal than the labels show" plus censored-target masking (A6-type one-sided
   supervision).
2. **Representation (E-B, P5).** `microsoft/deberta-v3-large` (pinned revision SHA). Input: the sentence with a typed
   entity marker in punctuation form around the anchor span, `… @ * {anchor_type} * {anchor} @ …` (no new vocabulary,
   which suits small data). Pooling: concat(hidden at the first `@` token, CLS) → dropout 0.1 → Linear(2H, K). Head
   biases initialised to logit(observed positive rate per role) (lesson 7).
3. **Loss.** Masked BCE averaged over observed cells, equal weight per example. This is a proper scoring rule, so the
   probabilities stay calibrated for the decode.
4. **Training (fixed plan).** AdamW, LR 1.5e-5 (encoder top layer) with layer-wise decay 0.85, head LR 1e-3, weight decay
   0.01, warmup 10% then linear decay over a fixed 6-epoch schedule, batch 16, grad-clip 1.0, fp16 autocast + GradScaler,
   length-bucketed batches from a seeded Generator, num_workers 2.
   5-fold cohort CV: evaluate masked OOF BCE at the end of epochs 2..6 → choose e\* in-script → OOF probabilities at e\*.
5. **Calibration.** Per-role bias + shared temperature on OOF logits (masked BCE, L-BFGS, fixed 100 iterations),
   cross-fitted check (Validation design, step 2).
6. **Final fit (V9).** Refit on 100% of the training examples with the same 6-epoch schedule stopped at e\*, 2 seeds.
   Average logits → calibration → sigmoid.
7. **Decode per test slate (its own rows only).** p_{c,r} = 0 for r ∈ Known(q) (hard mask from the target definition).
   EU(a,b) = Σ_{r∉Known} [1 − (1−p_{a,r})(1−p_{b,r})]. Pick (a\*, b\*) = argmax over the C(n,2) pairs. Candidates are
   canonically sorted by `candidate_text`; ties are broken by the larger Σ_r p, then text order. Write prediction
   = n+1 for a\*, n for b\*, and the remaining candidates n−1, n−2, … ordered by expected new-role count Σ_r p_{c,r}.
   All scores are finite and distinct.
8. **Diagnostics logged in-script.** Per-fold slate metric (official `grade`), mean ± std; per-role OOF AUC/log-loss;
   count-greedy vs EU decode on OOF; encoder parameter-change norm; frozen-probe yardstick (optional, ~1 min).

Why it fits THIS data: the target is literally a per-sentence role inventory minus a supplied mask. Learning Inc(c) at
sentence level uses every observed (sentence, role) cell (about 4.2k × K labels) rather than 772 slate-level choices,
transfers to unseen anchors (the role semantics live in the context words, not the name), and makes the pair decode
exact and parameter-free.

**Expected score (ESTIMATE, not a promise).** Random pair ≈ 0.55–0.70 (hypothesis). A good role tagger on short
sentences, cohort CV ≈ 0.82–0.92. Private ≈ CV ± 0.03. Reasoning: the count-greedy gold oracle is < 1 (overlap), annotation
omissions cap precision, and private has ~109 slates. The AI baseline is not printed in the description, so the payout
gate is unknown.

### Fallback (clean): same pipeline with `microsoft/deberta-v3-base`
Use it if profiling shows the large model's fold time > 7 min on A10G, or if large is not ≥ 1 SE better than base on
paired cohort CV (3 seeds). Base is ~2.5× faster, so the plan becomes 3 refit seeds + 5 fold models. Decide offline from
profiling and paired CV, then hardcode. **No runtime switching.**

### Compliance-adjusted ranking and the grey-area margin
1. Fine-tuned DeBERTa-v3 tagger + EU decode: clean, highest expected score.
2. Same with base: clean, slightly lower.
3. Frozen DeBERTa features + per-role logistic heads: grey-light. The description allows learned role models on
   pretrained representations, but "meaningful training" is weaker. Yardstick only.
4. Char-n-gram/TF-IDF linear tagger, GBDT/LambdaMART over hand features: grey (NLP regime: reviewer discretion).
   Yardstick only.

**Required margin to prefer any grey option as primary:** ≥ +0.02 absolute AND ≥ 2 SE of paired fold differences on 5×3
repeated cohort CV, positive in all 3 split seeds. Below that it may enter only as a minor member (logit-space weight
≤ 0.3, cross-fitted) and only if it adds ≥ 1 SE paired gain. I expect neither to clear the bar.

---

## Rejected options

| Option | Reason |
|---|---|
| Listwise / pairwise cross-encoder over (seed, candidate) trained on slate utilities | Ignores the exact factorisation. Learns seed similarity, which the description says is misleading. Slate-level supervision (772 decisions) instead of ~4.2k×K labelled cells. Kept only as roadmap 5b (a pair-softmax fine-tune *on top of* the tagger) if it beats noise |
| Lexical / embedding similarity to the seed as the ranking signal | Weak or negative by design. Also grey (lexical as core). Logged as a yardstick only (D12) |
| GBDT / LambdaMART on hand features (length, overlap, verbs) | Grey ("hand features + off-the-shelf ranker alone is not enough"), and those features do not see role semantics |
| Training on test seeds (`query_text` + `known_roles`) | **Banned**: uses test rows beyond inference ("use test candidates only for inference"). Transductive |
| Pooling test seeds' `known_roles` per anchor across test slates as an entity prior | **Banned**: whole-test aggregation (cross-row pooling of test rows) |
| Linking appearances through the candidate key or any id parsing beyond the slate key | **Banned**: id reverse-mapping. Text equality is used for train aggregation only |
| Imitating `oracle_rank` | Banned as hidden-tie imitation, and inferior: role sets are richer supervision |
| Entity-swap augmentation (replace the anchor with another train anchor) | Writes new sentences, which risks "generated sentences". Replaced by the deterministic typed-mask representation variant (roadmap 3c) |
| Continued MLM pretraining on test text | Banned (test fitting). On train text only: allowed but low expected value for short sentences, so roadmap-last |
| NLI-verbalised role entailment (hand-written role hypotheses) | Hand templates per role; K× inference cost; grey-ish wording. Roadmap 5c only as a diversity member |
| Hard type → role masks, keyword lexicons, regex relation patterns | Hand rules that replace learning (strip-the-ML / V-type rejection). The model learns type compatibility from the type marker |
| Class-weighted / focal BCE | Breaks calibration that the EU decode relies on. The metric counts roles equally without per-role weights, so calibration matters more than recall |
| Fold-average shipping only | The refit on 100% is preferred (V9). Adding the fold models to the test ensemble is tested on the sanity holdout (roadmap 6) |

---

## Fixed work plan & runtime budget

Constants (top of `solution.py`, no environment or clock dependence): `SEED=42`, `N_FOLDS=5`, `MAX_EPOCHS=6`,
`EVAL_EPOCHS=(2,3,4,5,6)`, `BATCH=16`, `MAX_LEN` = train p99 + 16 (computed in-script from **train** text only, capped
at 192), `LR_ENC=1.5e-5`, `LLRD=0.85`, `LR_HEAD=1e-3`, `WD=0.01`, `WARMUP=0.1`, `REFIT_SEEDS=(42,43)`, `NUM_WORKERS=2`,
`CALIB_ITERS=100`, `device="cuda"`, `MODEL="microsoft/deberta-v3-large"` + `REVISION=<sha>`. Determinism:
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch, cudnn deterministic, TF32 off,
`use_deterministic_algorithms(True, warn_only=True)`, seeded DataLoader generator, fixed thread counts.

| Stage | Work | A10G estimate (UNVERIFIED, profile first) |
|---|---|---|
| Load, parse JSON, aggregate labels, build folds, metric self-tests | CPU | 1 min |
| Download deberta-v3-large (~0.9 GB) + tokenize | — | 1–2 min |
| 5-fold CV: ~3.4k examples × 6 epochs per fold, eval each epoch | ≈ 80–100 samples/s at ~40 tokens | 5 × 4.5 = 22 min |
| Calibration fit + cross-fit check + OOF report | CPU | < 1 min |
| Refit 2 seeds × ~4.2k × e\* (≤ 6) epochs | — | 2 × 5 = 10 min |
| Test inference (743 distinct sentences × 2 models) + decode + validate + write | — | < 1 min |
| **Total** | | **≈ 36–40 min** (≥ 55% headroom vs the 90-min limit; ≥ 30% even if throughput is half the estimate) |

Memory: large + fp16 autocast, batch 16, len ≤ 192 → < 12 GB of 24 GB. Free GPU memory between folds.
In-script validation: input schema; id regex; rows per slate == count of the slate key in test; every test
`anchor_start/end` slices to the anchor; probabilities finite; output finite, unique ids in sample order, 1,197 rows;
reload with `keep_default_na=False`; atomic write. **No placeholder or fallback file**; any failure raises.

---

## Metric-aware training & decode

- **Back-solved target (G-i).** The metric's hidden sets are Inc(c) \ Known(q). We back-solve Inc(c) at sentence level and
  treat roles in Known as censored (masked, never negative).
- **Loss (G-ii).** Masked BCE per (sentence, role). The metric weights slates equally, and a sentence's influence ∝ its
  appearances. Variant: weight sentences by appearances^0.5 (roadmap 4d, adopt only on paired gain).
- **Decode (G-iv, G-vii).** Hard constraint: p = 0 on Known(q) roles. Choose the pair by expected utility. The primary uses
  E[|A∪B|] in closed form (exact under within-slate independence, zero constants). Variant 4a: maximise E[|A∪B|/opt]
  by Monte-Carlo over Bernoulli draws of the whole slate (S = 2,000 common random numbers from a fixed generator, indexed
  by the canonical candidate order, so content-seeded and independent of other slates). Variant 4a′: also condition the
  draws on the published retention rule (opt > 0 and non-constant pair utilities). Reviewer question Q5. Adopt each only
  if the paired gain is > 1 SE.
- **Calibration (lesson 7).** Base-rate bias init + K per-role biases + 1 temperature fitted on OOF, cross-fitted.
  Stage-2 (roadmap 4b): per-role logistic recalibration with inputs [logit_r, Known multi-hot, anchor_type one-hot,
  |Known|], fitted row-level on OOF (labels: R for r ∉ Known), L2-strong, cross-fitted. This is the learned "entity prior
  from the seed" and the only use of seed information beyond the mask.
- **Oracle checks (lesson 15).** Gold sets through the decoder → 1.0 on every train slate (proves grouping, masking,
  pair enumeration and output writing). Count-greedy with gold counts → the value of complementarity. Frozen-probe
  yardstick on the same folds.
- **Three-stage diagnosis on OOF.** Ranking quality: per-role AUC, within-slate Spearman of expected vs gold |R|. Decode
  quality: EU decode on OOF p vs count-greedy on OOF p (expected +0.01 to +0.04 for EU, UNVERIFIED).

---

## Structural signals

| Invariant guaranteed by the data process | Use |
|---|---|
| R_c = Inc(c) \ Known(q) (definition) | Hard mask at decode. Masked (censored) loss in training. Seed-independent sentence-level target |
| The same sentence has one Inc across its appearances (one cohort per sentence, exact-dup dedup) | Label aggregation across train appearances; verified by 0 contradictions (D4) |
| Seeds are fully labelled (`known_roles` = seed's inventory); seeds ∩ candidates = ∅ | Train seeds as extra fully observed examples (train only) |
| All slate members share anchor/type/seed | Within-slate discrimination comes only from candidate text: the model must read context. No slate-constant features in the tagger |
| Union counts a role once | Complementarity via the expected-union pair decode (not count-greedy) |
| Candidates sampled independently of roles (no engineered decoys) | No decoy-generator modelling needed (no Q5/Q6 exposure). Slate composition carries no label signal and is not featurised |
| Retention: opt > 0 and non-constant utilities | Optional posterior conditioning (4a′), reviewer question |
| Roles = direction × relation category (D2) | Roadmap 3d: factorised head (shared relation embedding × direction) for rare roles. The parse of the role string is metadata, not a rule |
| Row order inside a slate is meaningless | Canonical sort + permutation-invariance unit test |

**How the model discovers structure itself (and how we show it).** Learned, not coded: which context words express
which relation role and in which direction; type → role compatibility (from the type marker; D10 contingency is
reproduced by the model's predicted probabilities, log it); role co-occurrence (shared multi-label head); the seed-based
entity prior (stage-2 learned weights, logged); redundancy between candidates (through the probabilities feeding the
union). Coded but contractual (the definition of the target and the metric, not data patterns): the known-role mask and
the pair-union decode. No keyword lists, no regex relation patterns, no type → role tables, no id/order features, no
generator constants. Comment each of these in the code.

---

## Experiment roadmap

Each step: one change, paired 5×3 cohort CV (offline), keep if > 1 SE.

1. **Contract and validation.** Run D1–D13. Paste `grade` verbatim. Unit tests 1–6 pass. Folds (StratifiedGroupKFold by
   cohort). Write the random / id-order / count-greedy-gold / worst numbers into CHALLENGE_NOTES. *Stop if* D4 shows
   contradictions > 1% (then the factorisation is wrong: switch to row-level masked targets with seed-conditioned input,
   and re-plan).
2. **Cheapest honest baseline end-to-end.** Frozen deberta-v3-base marker-pooled features + per-role logistic, EU decode,
   valid CSV through the full script path. It gives the yardstick and the first free CSV check. Then fine-tuned
   deberta-v3-base (primary pipeline) → first credit (baseline submission).
3. **Representation.** (a) large vs base (profile fold time on A10G, decide and hardcode). (b) Seeds as extra training
   examples on/off. (c) Typed anchor mask vs name-visible markers. (d) Factorised direction × relation head. Expected
   largest lifts: 3a and 3b.
4. **Metric-aware.** (a) MC E[U/opt] decode vs closed-form E[U]; (a′) retention conditioning (only after a reviewer
   answer). (b) Stage-2 seed/type recalibration. (c) Pair-redundancy correction, only if OOF error analysis shows
   failures on near-paraphrase pairs. (d) Appearance weighting. Second credit: best single model.
5. **Diversity** (only after 3–4 have settled). (a) A roberta-large tagger (different tokenizer) averaged in logit space
   after per-model calibration. (b) A pair-softmax fine-tune stage: loss = −Σ_pairs softmax(EU/τ)·U/opt per train slate,
   starting from the tagger, 1 epoch. (c) An NLI-style member. Keep a member only with ≥ 1 SE paired gain; blend weights
   cross-fitted, ≤ 2 free weights. Third credit: ensemble.
6. **Bounded in-script selection.** e\* already in-script. Optionally LR ∈ {1e-5, 2e-5} as an in-script choice on fold
   OOF, only if runtime allows (it doubles CV cost; probably not). On the sanity holdout, compare refit-only vs
   refit + fold models.
7. **Final.** Freeze constants. Clean run twice and diff (`determinism_check.py`). `compliance_scan.py`,
   `validate_submission.py`, `half_rows_test.py` at slate granularity. Profile per-stage time. Final credit.

When independent solvers would converge on something, it is "fine-tune an encoder per candidate and pick a
complementary pair". That is the primary, so it is tested first.

---

## Compliance audit

| CLAUDE.md §7 / plan audit | Status in plan |
|---|---|
| Test rows used beyond one-sample (here: one-slate) inference? | No. Ids, slate key and per-row inputs only. Test seeds are not trained on. No cross-slate pooling. Half-rows test planned |
| Clock in any condition/argument? | No. Time is logged only |
| Env/hardware fallbacks, try/except model switching, cpu_count? | No. `device="cuda"` is hardcoded and the model is fixed after offline profiling |
| Constants tuned offline? | Recipe defaults documented as literature defaults. e\* and calibration are fitted in-script on OOF. No LB-derived numbers |
| External / synthetic / pseudo data, self-hosted weights? | No. HF pinned revision only. No augmentation that writes new sentences |
| Strip-the-ML? | Without the tagger the output is constant (random-like). The ML is load-bearing |
| Id reverse mapping / candidate order as evidence? | No. Canonical sort plus a permutation-invariance test |
| Hardcoding example scores / oracle ties? | No. `oracle_rank` is used in a unit test only |
| Source legible, < 512 KB, requirements map in the docstring | Yes. Fill the template's requirements map: "A10G, 90 min → fixed ≈ 40 min plan", "meaningful training → full fine-tune, change-norm logged", each "What Not To Use" line → where avoided |
| Seeds fixed, workers fixed, deterministic flags | Yes. Run twice and diff |

Residual reviewer-facing risks: (i) train label aggregation across appearances and train seeds as examples; (ii) the
pair decode being "metric-aware" (not banned here); (iii) the optional retention conditioning.

---

## Open questions & assumptions

- **Q1 (assumption, low risk):** Using train seed sentences with their `known_roles` as fully labelled training examples is
  "training on the supplied examples". If a reviewer disagrees, drop 3b. The cost is measured in roadmap 3b.
- **Q2 (low risk):** Aggregating a sentence's labels across its train appearances (exact text match, train only) is
  legitimate label processing, not id reverse-mapping.
- **Q3 (low risk):** A per-slate expected-utility pair decode over the slate's own candidates is a per-sample decision (the
  slate is the unit; grouping by slate key is explicitly allowed).
- **Q4 (ask before using):** Is a typed anchor mask (the name replaced by its type, identically for train and test) a
  representation choice and therefore allowed? Is random anchor-name dropout in training a "generated sentence"? The plan
  ships only the deterministic version and only if it wins.
- **Q5 (ask before using):** Is conditioning the decode posterior on the published retention rule (opt > 0, non-constant
  utilities) acceptable, or is it "exploiting data generation"? It is excluded from the primary.
- **Assumptions to verify first:** D1 slate key ↔ `query_id`; D4 zero contradictions (the foundation of the factorisation);
  D6 anchor locatable in seeds; D9 short sentences; the role vocabulary is small and closed (D2). Test known roles outside
  the train vocab are ignored (mask-only).
- **Unknowns:** the AI baseline score (not in the description); real A10G throughput of deberta-v3-large on this text
  (no GPU profiling done); whether the dev sandbox has a GPU. Every runtime figure above is an estimate until profiled.
