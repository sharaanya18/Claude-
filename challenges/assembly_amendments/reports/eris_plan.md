# Eris build plan — assembly_amendments

Author: eris-strategist, 2026-10-08. Inputs: `CHALLENGE.md`, `reports/contract.md`, `reports/data_audit.md`,
`reports/metric_spec.md`, `metric.py` (the scoring ground truth, used as-is). Playbook ids cited in brackets
(families: structured-assignment + partition-and-bags + text-nlp; cross-cutting: metric-and-decoding,
validation-recipes, engineering-and-compliance).

New train-only profiling done for this plan (scratch scripts, read-only, no training; numbers quoted below as
"[P]"):
- Test-like train subset (n_amendments in [4,80] AND >=2 nonzero entries in `counts`): **1652 boards, 144 bills,
  21,257 items**. Of these 9.8% have a==0 and 39.9% are joint-singleton-only (test: 36.5%). Item fate shares
  25.1/36.5/23.6/6.7/8.0%.
- `fell > 0` essentially requires `adopted > 0`: only 41 of 1059 train boards with f>0 have a==0 (24 in test-like).
- Recorded multi-member joint groups (test-like): 1361 have 0 adopted members, 693 exactly 1, 33 two or more.
  "At most one adopted per joint group" is a strong (98%) but not absolute coupling.
- **Article-level cascade (the dominant `fell` mechanism)**: dispositifs opening "Supprimer cet article" (673 items)
  or "Rédiger ainsi cet article" (419) in test-like boards. On boards where such an item was ADOPTED, the other
  non-adopted items fell **97.1%** of the time (2905/2992) vs **16.0%** (2117/12934) on other boards. An
  article-deletion item itself never falls (0.0%) — it is called first. This is the latent calling order surfacing.
- Same referenced "alinéa N" (55.7% of dispositifs cite one) → shared joint label 15.5% vs 3.6% base (4.3x lift).
- Exposé < 60 chars (rapporteur "Rédactionnel"/"Précision", 2522 items) → 77.4% adopted; exposé mentioning
  "[groupe]" (8560 items) → 15.7% adopted / 58.3% rejected; dispositif with a revenue "gage" clause (4079) →
  18.1% not-moved (2.2x). Author identity is masked but drafting style still carries it; a fine-tuned encoder
  will learn these, hand features only assist.
- Non-trained diagnostic yardstick (NOT to ship; it is a rule pipeline): rank adopted by short exposé / no
  "[groupe]", random disposal, average-link word-Jaccard joint at 0.6 → test-like mean board 0.188 (=18.8 points;
  carried 0.21, disposal 0.11, joint 0.32). Counts-respecting random → 0.011. These bracket the floor.
- Training pairs for the joint model: 251,257 within-board pairs on boards with n<=80; 222,590 on the 1249 such
  boards that have >=1 true joint pair.
- HF availability checked: `almanach/camembertav2-base` (DeBERTa-v2 arch, 12L/768, vocab 32,768,
  sha `54cc91d6ac45a540c7e0faeb677f4dd8201d3d61`, has tokenizer.json + safetensors) and `almanach/camembert-base`
  (sha `a75967561c78f2aa81cc41045378d3b4ee25af9e`) both resolve.

---

## Contract & decision unit

- **Decision unit = one board** (one test.csv row, 4..80 items in test). Items within a board are predicted
  jointly; boards are predicted independently from a train-fit model. One valid answer per board:
  `fates` = JSON int list of length n in `item_no` order whose multiset equals `counts` exactly, and `joint` =
  JSON int list of length n (only the induced partition matters). 622 rows, columns `item_id,fates,joint`, ids and
  order from `sample_submission.csv`.
- **Invalid vs low**: wrong-multiplicity/length/unparseable `fates` → carried = disposal = -1 (catastrophic);
  bad `joint` → 0. The decoder below is valid by construction and the script asserts validity per board before
  writing.
- **What the metric rewards** (exact, `metric.py`):
  - carried (0.35, defined iff a>0): chance-corrected accuracy of adopted/not over n items,
    `e_c = (a²+(n-a)²)/n²`.
  - disposal (0.35, defined iff a<n): chance-corrected accuracy over the n-a truly non-adopted items, placing one
    as adopted = wrong, `e_d = (r²+f²+w²+m²)/(n(n-a))`.
  - joint (0.30, defined iff truth has >=1 shared label): ARI.
  - Board score = defined-weight-renormalised mean; final = 100 × mean board score (floored at 0 once).
  - Because a, r, f, w, m are GIVEN per board, the definedness of carried/disposal and both chance baselines are
    known constants at inference time. Only joint-definedness is unknown — and crucially, **joint is only scored on
    boards where some pair truly shares a label**, so joint predictions only matter conditional on that event.
- **Pipeline stages, diagnosed separately** [principles §3]:
  1. evidence: per-item fate probabilities given text + board context (ranking quality: per-board top-a hit rate,
     within-board AUC per code);
  2. decode-fates: counts-respecting assignment (decode quality: feed GOLD one-hot probabilities → must give
     carried = disposal = 1 on every board);
  3. evidence-joint: pairwise same-group probabilities (pair AUC / AP on boards with >=1 true pair);
  4. decode-joint: partition from pair probabilities (feed GOLD pair matrix → ARI = 1 on every board).
  The "coverage" stage is trivial here: the valid output space is enumerable and the decoder covers all of it.

## Compliance regime

- Domain: **NLP (French legal text) + structured assignment/partition**. Regime: fine-tuned transformer =
  accepted; GBDT stacker over fine-tuned outputs + hand features = accepted as members; TF-IDF/regex as the core =
  grey/banned. No "from scratch", no "fine-tuning" label, no model restriction, no shorter runtime stated →
  CLAUDE.md defaults (one A10G, <=50 min target, HF weights allowed).
- **Explicit ban (verbatim)**: "Do not use any external copy of these amendments, of their outcomes, of the records
  or reports of the sittings, or any model trained on them, and do not look up the withheld authors or outcomes."
  → no external data, no lookups, no model fine-tuned on AN records. A general French web-pretrained encoder
  (CamemBERTa-v2 / CamemBERT) is a general-purpose backbone and is used; reviewer question Q1 below because such
  corpora may incidentally contain public AN web pages.
- **Allowed in-rules mechanisms** (each checked):
  - Using the board's own `counts` at inference (feature + decode constraint): `counts` is a test.csv input column,
    and the description itself says "Each code must be used exactly as many times as the board's counts say".
    Count-matching decode is required by the output grammar; it is per-board, not test-set fitting.
  - Expected-utility (metric-aware) decode: the description does not ban decoding tricks; it uses only one board's
    own model outputs and counts.
  - Within-board cross-item features (pair similarities, relative ranks, "mass of likely-adopted competitors"):
    the board is the decision unit and the description defines the task as reconstructing interactions among a
    board's amendments, so board-mates are explicitly visible and legitimate (not "sibling" leakage, rejection S).
  - Training-set filtering to boards whose `counts` have >=2 nonzero codes: derived from the stated test population
    and computed from `counts` (an input), not from test data.
- **Banned / avoided**:
  - Any cross-BOARD test feature: e.g. matching a plenary board's texts to the committee board of the same test bill,
    bill-level aggregates over test boards, normalising scores across the test file, fitting a vocabulary/TF-IDF/
    scaler on test text. Each test board's output must be a function of that board's rows + train-fit models only
    (half-rows test must pass: drop half the test boards, kept boards' outputs byte-identical).
  - `bill_id`, `item_id`, `item_no`, file row order as features (opaque/noise; `item_no` used only to order outputs).
  - `bill_title` as a feature (bill-specific topic memorisation; test bills unseen). Possible roadmap ablation only.
  - Hand rules that set fates/groups ("if Supprimer adopted then the rest fall") — the 97% cascade must be LEARNED:
    it enters only as features into trained models and via a learned per-item "kill-power" head.
  - Time guards, hardware/env branches, try/except import fallbacks (CLAUDE.md §3).
- **Self-audits on the plan**:
  - Strip-the-ML: remove the transformer and both GBDTs → the decoder receives no scores → counts-respecting
    arbitrary assignment and all-singleton partition ≈ 0 points (measured random = 1.1). Pass. The regex parsers
    alone produce nothing; the non-trained yardstick (18.8) is dev-only and never in `solution.py`.
  - No whole-test aggregation: all stats per board; train-only fitting. Pass by design; verified by half-rows test.
  - Sibling leakage: within-board features are the task's own structure (see above); cross-board train features
    (none planned) would be built leave-own-bill-out.
  - Every constant is either a principled default or chosen in-script on train OOF (cluster cut τ, Sinkhorn on/off,
    GBDT rounds). Nothing copied from submission feedback.
  - Load-bearing training: the fine-tuned encoder is the main evidence source; logged ablation "stage 2 without
    encoder features" must lose clearly (expected >= 5 points), and the parameter-change norm is asserted > 0.

## Data findings

(Full audit in `reports/data_audit.md`; only modelling-relevant facts here.)
- 4392 train boards / 31,566 items / 209 bills; test 622 boards / 8851 items / 111 bills; 0 bill overlap. Top bill
  = 10.7% of train boards.
- Train size distribution is not test-like: 57.7% of boards have n<4; 45% of boards hand out a free degenerate
  term (a==n, or a==0 with one non-adopted fate). Test-like subset defined above: 1652 boards / 144 bills.
- Fates (all train items) 25.8/39.8/19.6/6.7/8.1%; adopted share falls from 44% (n<=2) to 16.5% (n>=31).
- Joint: 70.7% of all boards singleton-only, 39.9% of test-like boards; one train board has a 100-member group.
- Within-board exact duplicate dispositif = 0 pairs (deduplicated). Word-Jaccard > 0.8 → 68% shared label (25x),
  identical exposé → 41% (15x), Jaccard <= 0.2 → 0.41%. Grouped items ~2x longer than singletons.
- Fell: 55% of fell items are singletons in the recorded grouping; the article-level deletion/rewrite cascade
  explains most of the rest ([P]: 97% of non-adopted fall when an article-level item is adopted).
- 46.3% of dispositifs repeated across boards carry different fates → board context is required.
- Dispositif chars p50 292 / p99 5064 / max 85k; exposé p50 1044 / p99 4691 chars; one exposé sentinel in train.
- `reading`: CMP boards 97% adopted, deuxième lecture 0% adopted (93 boards total, bill-clustered).
- Categoricals: `bill_kind` (10), `reading` (4), `text_examined` (2), `examining_body` (10), `division` free text →
  parse to {numbered article, additionnel après, additionnel avant, other} + "bis/ter..." suffix flag.

## Validation design

- **Split requirement (validation-architect implements)**: K=5 folds grouped by `bill_id`, size-aware (balance the
  number of test-like boards and items per fold, not just bills), and bills should additionally be union-merged when
  they share >=3 exact dispositif texts (mirrors the description's own dossier rule; 376 texts span 2+ bills). One
  fixed fold assignment, seeded, computed inside `solution.py` from train only. All three learned stages (encoder,
  pair GBDT, item GBDT) and every decode-constant search use the SAME folds.
- **Headline number = test-like OOF score**: `metric.score_submission` restricted to the 1652 test-like boards
  (n in [4,80], >=2 nonzero counts). Also report: per-term means (carried / disposal / joint over boards where
  defined), raw all-train score (labelled "inflated, do not compare"), per-fold scores, and slices by
  `examining_body == "Séance publique"` vs committee, `division` type, n bucket (4-9, 10-29, 30-80), and
  `bill_kind`. Bill-level bootstrap (1000 resamples of bills) for a 90% CI on the headline.
- **Noise rule**: accept a change only if the paired per-fold difference passes the Nadeau-Bengio corrected
  test (V16) or is positive in >= 4/5 folds with mean gain > 1 corrected SE. Stage-1 encoder runs are expensive,
  so encoder-level comparisons use one split; stage-2/decode comparisons (cheap, on cached OOF in dev) add 2 extra
  stage-2 split seeds.
- **Nested selection** (V3): the decode constants (cluster cut τ, Sinkhorn on/off, pair-loss weighting choice) are
  selected in-script on OOF; report the cross-fitted value (select on folds {0,1} + half of fold 2's bills, score
  the rest, swap, average) next to the in-sample value.
- **Oracle checks (unit tests, run every dev iteration)**: gold one-hot fate probabilities → decode → carried =
  disposal = 1.0 on all boards; gold pair matrix → decode → ARI = 1.0 on all boards with pairs; decoded fates always
  pass `metric.parse_fates` against counts (all 4392 train boards, including n=1..368).
- **Bias direction of the proxy**: bill-grouped folds mirror the split unit exactly; the test-like filter matches
  the stated test population. Remaining biases: (1) stacking with early stopping on the held-out fold and in-sample
  decode selection → optimistic by ~1-2 points; (2) the private slice is ~half of 111 bills, so the private number
  has a sampling SD of roughly ±2.5 points around the true mean (per-board SD ~0.4 / sqrt(311)) plus bill
  clustering; (3) test-time stage-1 features are 5-fold averages (smoother than single-model OOF) → slight
  calibration shift, sign uncertain. Plan for private ≈ test-like OOF minus ~2 points, ±4.

## Overfit/underfit risks

| Risk | Kind | Evidence | Mitigation |
|---|---|---|---|
| Few independent units (209 bills, 144 test-like) | overfit | top bill 10.7% of boards | bill-grouped folds; fixed short schedule (2 epochs); GBDT min_data_in_leaf >= 50, feature_fraction 0.8; no in-script HPO beyond 3 discrete decode choices |
| Encoder memorises bill vocabulary | overfit | 55% of bills re-table identical text across stages | no `bill_title`/ids; low LR 2e-5 with LLRD 0.9; weight decay 0.01; OOF-only stacking |
| Stacking optimism | overfit | stage 2 trained on OOF from stage-1 models that saw stage-2 val rows of other folds | same folds for all stages; report cross-fitted decode numbers |
| Giant boards dominate losses | misfit | n>80 boards hold 16% of items / 60% of pairs | item loss weight 1/sqrt(n_b) in stage 1, 1/n_b in stage 2 (board-equal, mirrors metric); pair model trained on boards n<=80 only |
| Degenerate train boards dilute stage 2 | misfit | 45% of boards fully determined by counts | stage 2 and decode selection trained/evaluated on boards with >=2 nonzero counts only |
| Truncation loses evidence | underfit | dispositif p99 850 words | dispositif head <=160 tokens + exposé head to fill 320; log-length features in stage 2 |
| Under-sized representation | underfit | text carries author style (short exposé → 77% adopted) | strongest affordable French encoder (CamemBERTa-v2 base); large models rejected only on runtime (see Rejected) |
| Cascade not learned by per-item model | underfit | 97% fell rate after article-level adoption | learned kill-power head + p_adopt-weighted cascade features in stage 2 (see Structural signals) |
| Joint threshold drifts to degenerate corner | overfit | ARI gives 0 to all-singletons/all-one | τ grid bounded to [0.2, 0.8]; selection only on boards with >=1 true pair (the scored population) |

## Recommended approach (primary + fallback)

### Primary (A): fine-tuned French encoder → pair GBDT (joint) → item GBDT stacker → exact expected-utility decode

Order of computation (each stage OOF over the same 5 bill-grouped folds; test = average/refit as stated):

**Stage 1 — per-item multi-head encoder** [text-nlp default build; B-capacity: full FT, items are many]
- Backbone `almanach/camembertav2-base`, revision pinned to `54cc91d6ac45a540c7e0faeb677f4dd8201d3d61`.
- Input text (one sequence per item): `"{examining_body} | {reading} | {text_examined} | {division}"` [SEP]
  dispositif (head, <=160 tokens) [SEP] exposé (head, fill to max_len 320; sentinel "(pas d'exposé sommaire)"
  replaced by a fixed short marker "aucun exposé"). Implementer measures the real token-length distribution on train
  once and keeps 320 unless p90 of the combined text is far lower. No positional/item_no info; items are encoded
  independently, so the stage is permutation-invariant over the board by construction.
- Pooling: mean of last hidden layer (masked) → dropout 0.1 → heads:
  1. fate head: 5-way softmax, CE (no label smoothing), item weight 1/sqrt(n_board);
  2. kill-power head (learned cascade potential): sigmoid, BCE on soft target
     `k_i = (#other items on the board that fell) / (n-1)`, masked to items that were ADOPTED on boards with n>=2;
  3. grouped head: sigmoid, BCE on "item has >=1 joint partner" (boards n>=2).
  Loss = CE + 0.5·BCE_kill + 0.5·BCE_grouped (fixed weights, principled default; ablate once).
  Counts are NOT fed to the encoder: it learns text-propensity evidence; counts enter in stage 2 and the decode.
- Training: AdamW lr 2e-5 (head 1e-3), LLRD 0.9, weight decay 0.01, linear warmup 6% then linear decay, 2 epochs,
  batch 32, grad-clip 1.0, bf16 autocast, length-bucketed batches with a seeded generator, fixed num_workers=2.
  All 31,566 items are used for training (all board sizes; text evidence transfers), excluding the held-out fold.
- Outputs per item: 5 fate logits, kill logit, grouped logit, and the 768-d pooled embedding (fp16) for pair
  cosines. Train rows: OOF from the fold model; test rows: average of the 5 fold models' outputs (logits averaged;
  embedding cosines computed per fold model and averaged).
- Assert the backbone parameter-change norm > 0 after the first optimizer step and log it per fold.

**Stage 2a — pair model for `joint`** [partition-and-bags default build; structured-assignment "pair classifiers"]
- Rows: every unordered within-board item pair (i<j), only boards with n in [2,80]. Swap-symmetric features only
  (min/max/abs-diff/product of per-item quantities; no ordering by item_no):
  - lexical, no fitted vocabulary: word-set Jaccard and containment (dispositif), char-5-gram-set Jaccard
    (dispositif), exposé exact-equality flag and word-Jaccard, Jaccard of the sets of quoted strings « … » (the
    passage anchors), shared referenced alinéa numbers (count, any), both "cet article" scope flags, opening-verb
    category of each item (min/max code), log-length min/max/abs-diff;
  - learned: cosine of stage-1 embeddings; min/max of stage-1 grouped probability; min/max and product of stage-1
    p_adopt and p_fell; |p_rejected_i - p_rejected_j|;
  - board context: log n, counts fractions, division type, examining_body, reading, bill_kind (LightGBM
    categoricals).
- Model: LightGBM binary, `deterministic=True`, `num_threads=4`, `seed`, lr 0.05, num_leaves 31,
  min_data_in_leaf 50, feature_fraction 0.8, bagging off, early stopping on held-out fold logloss (patience 100,
  max 2000) → final refit on 100% with mean best iteration × 1.1.
- **Training population = boards that have >=1 true joint pair** (1249 boards, 222,590 pairs). Rationale
  (metric-aware): the joint term is only scored on such boards, so the probability we need at decode is
  P(same | board has a pair). Ablation: all boards n in [2,80]. Pair weight: uniform within board × 1/sqrt(#pairs
  of board) (chosen between {uniform, 1/sqrt} in-script on OOF ARI; one binary choice).

**Stage 2b — item stacker for `fates`** [metric-and-decoding §2; F2.9 relative transforms; F2.2 latent structure]
- Rows: items of boards with n in [2,80] and >=2 nonzero counts (≈ 23,005 items). Multiclass LightGBM (5 classes),
  same deterministic settings, sample weight 1/n_board.
- Features:
  - stage-1 OOF: 5 fate log-probs, kill logit, grouped logit;
  - within-board relative transforms of p_adopt and of each p_k: rank/n, (p − board mean), z within board, and the
    counts-aware cutoff margin `p_k(i) − (c_k-th largest p_k on the board)` and `c_k/n` for each code k;
  - cascade features (learned latent structure, mean-field form): `S_kill(i) = 1 − Π_{j≠i}(1 − p_adopt(j)·kill(j))`
    (noisy-OR probability that some OTHER item that kills the article was adopted), `max_{j≠i} p_adopt(j)·kill(j)`,
    `Σ_{j≠i} p_joint(i,j)·p_adopt(j)` (adopted competitor mass in i's predicted joint discussion),
    `Σ_j p_joint(i,j)` (expected group size), own kill logit × S_kill;
  - board: log n, a/n, r/n, f/n, w/n, m/n, n−a, division type + suffix flag, examining_body, reading, bill_kind,
    text_examined;
  - item hand features (deterministic extraction, fed to the model): log char length of dispositif/exposé, exposé
    < 60 chars flag, "[groupe]" count in exposé, revenue-gage clause flag, opening-verb category (Supprimer /
    Rédiger / Substituer / Compléter / Insérer / Après / À / I. / other), "cet article" scope flag, number of cited
    alinéas, number of quoted strings.
- p_joint for train rows comes from the stage-2a OOF; for test from the stage-2a refit (or fold-average; pick one,
  fixed: refit).
- Final: refit on 100% with fixed rounds (mean best iteration × 1.1). Test stage-2b features use the averaged
  stage-1 outputs.

**Decode — fates (exact for the two fate terms)**
1. Per board, take stage-2b probabilities Q (n×5); clip at 1e-6; zero the columns of codes with c_k = 0.
2. Optional Sinkhorn/IPF projection (fixed 50 iterations): alternate row-normalise to 1 and column-normalise to
   `counts`, giving count-consistent marginals. On/off selected in-script on OOF (one binary choice).
3. Utility of giving item i code c:
   `U(i,0) = W_c·q_i0`,  `U(i,c≠0) = W_c·(1−q_i0) + W_d·q_ic`, with
   `W_c = 0.35 / (n·(1−e_c))` if 0<a<n else 0, `W_d = 0.35 / ((n−a)·(1−e_d))` if a<n and 1−e_d > 1e-9 else 0.
   (Joint weight only changes the per-board normaliser, which is identical for every fate assignment, so it is
   omitted.) Because carried and disposal are linear in per-item hit indicators with count-fixed chance baselines,
   maximising Σ_i U(i, c_i) is the **exact Bayes-optimal decode for the expected fate terms** given the marginals.
4. Solve as a linear assignment: expand columns into `counts[k]` slots of code k (n×n matrix, n<=80 in test, <=368
   in train CV), `scipy.optimize.linear_sum_assignment` on −U. Deterministic. Assert multiset == counts; raise on
   failure (never write an invalid row).
   Edge cases are automatic: a=0 (no code-0 slots), single nonzero count (one valid assignment), n=4.

**Decode — joint**
- Symmetric pair matrix P (n×n) from stage 2a; distance 1−P; `scipy.cluster.hierarchy.linkage(method="average")`,
  cut at distance 1−τ, labels from `fcluster` → JSON list of ints in item_no order. τ chosen in-script from
  {0.20, 0.25, …, 0.80} maximising mean OOF ARI over test-like boards that have a true pair (the metric excludes
  the others automatically). Linkage {average, single} is a dev-time choice (fixed in code after the dev
  experiment, not searched in-script) to keep in-script selections at three.
- Oracle: gold P (1 for same group, 0 else) must give ARI 1 for every τ in the grid.

**Why A fits this data**: fates are a within-board ranking problem under known counts (decode exact, evidence from
text style), `fell` is mostly a cascade from an adopted article-level item (learned kill-power + noisy-OR features),
joint is a pairwise relation dominated by passage anchors and near-duplicate edits (pair GBDT with swap-symmetric
features + learned cosine). The encoder is load-bearing for the adopted/rejected/withdrawn/not-moved evidence that no
structure feature carries. Lean: one encoder run, two small GBDTs, two deterministic decoders.

### Fallback (B): same pipeline, smaller/faster encoder
- `almanach/camembert-base` (RoBERTa arch, revision `a75967561c78f2aa81cc41045378d3b4ee25af9e`), max_len 256,
  2 epochs. Use if profiling shows primary stage 1 > 25 min on A10G, or if CamemBERTa-v2 under-performs it on
  the paired fold check. The choice is made in dev and hardcoded (no runtime switch).
- Emergency fallback if GPU fine-tuning is impossible to fit: 1 epoch of camembert-base at max_len 192 (~7 min).

### Expected range (estimate, not a promise)
Test-like OOF headline 38–52 (the non-trained yardstick gives 18.8; a learned joint should reach ARI 0.45–0.6 vs
0.32; carried 0.40–0.55 vs 0.21; disposal 0.35–0.5 vs 0.11 with the cascade). Private LB estimate 35–50 given ~2
points of optimism and ±2.5–4 points slice noise. The printed AI baseline is not stated in the description.

## Rejected options

- **End-to-end board set-transformer as PRIMARY** (encoder + permutation-equivariant attention over the board's
  items + bilinear joint head + Sinkhorn-in-loss): highest ceiling in principle, but heavier training (board-packed
  batches, up to 368 items/board), more knobs, and the playbook gate (F2.13) says measure the lean design first.
  Kept as roadmap step 5 (light version on cached stage-1 embeddings).
- **Pairwise cross-encoder for joint** (concatenate both items into the transformer): 250k train pairs × 5 folds
  is ~10x the stage-1 cost; out of budget.
- **Large encoders (camembert-large, xlm-roberta-large)**: ~3x the base cost → 5 folds × 2 epochs ≈ 60+ min.
  Rejected on runtime; could re-enter only as 3-fold/1-epoch if the profile shows >20 min spare (unlikely).
- **mDeBERTa-v3-base / multilingual models**: weaker French tokenisation than a French-native vocabulary; possible
  diversity member in roadmap step 5 only if runtime allows.
- **GBDT-only on TF-IDF / hand features (no fine-tuned encoder)**: grey under the NLP regime ("TF-IDF/n-grams …
  reviewer's discretion") and lower ceiling; used only as a dev yardstick (V11), never shipped.
- **Per-item argmax or rule-based fate assignment**: argmax violates counts → −1/−1; rules fail strip-the-ML.
- **Hard-coded cascade ("if an article deletion is adopted, everything else falls") or Jaccard-threshold joint
  rule**: banned by CLAUDE.md §2.1 / rejection V; the same information enters as learned features.
- **Fitted TF-IDF vocabulary on train+test, or any cross-board test features** (same-bill committee↔plenary
  matching in test): banned (§2.3 #5).
- **Feeding counts into the encoder** in the primary: tempting shortcut; reserved as a roadmap ablation because it
  can make stage-1 logits mostly re-encode counts and weaken the text evidence the stacker needs.
- **LLM/generative formulation**: unnecessary; the valid output space is a counts-constrained assignment plus a
  partition, solved exactly/deterministically.

## Fixed work plan & runtime budget

Constants (to be finalised by the baseline writer after profiling one epoch on an A10G-class GPU):

| Constant | Value |
|---|---|
| SEED | 42 (python, numpy, torch, cuda, DataLoader generator, LightGBM `seed`) |
| N_FOLDS | 5 (bill-grouped, size-aware, built from train only) |
| Encoder | `almanach/camembertav2-base` @ `54cc91d6…` (fallback `almanach/camembert-base` @ `a7596756…`) |
| MAX_LEN / dispositif budget | 320 / 160 tokens |
| EPOCHS / BATCH / LR / LLRD | 2 / 32 / 2e-5 (head 1e-3) / 0.9 |
| Precision | bf16 autocast; `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)` |
| NUM_WORKERS / torch threads | 2 / 4 |
| LightGBM | lr 0.05, leaves 31, min_data_in_leaf 50, ff 0.8, `deterministic=True`, `force_row_wise=True`, `num_threads=4`, ES patience 100, max 2000; refit rounds = mean best × 1.1 |
| Sinkhorn iters | 50 |
| τ grid | 0.20…0.80 step 0.05 (13 values) |
| In-script selections | τ; Sinkhorn on/off; pair weighting {uniform, 1/sqrt} |

Runtime estimate on one A10G (to be verified by profiling; ≥30% headroom vs 60 min):

| Stage | Estimate |
|---|---|
| Load, parse, sort by (item_id, item_no), tokenise 40.4k items (fast tokenizer) | 1–1.5 min |
| Stage 1 train: 5 folds × 2 epochs × ~25.3k items ≈ 253k samples at ~170–200/s (DeBERTa-v2 base, bf16, ~300 tok) | 21–25 min |
| Stage 1 inference: OOF 31.6k + test 8.85k × 5 = 76k at ~700/s | ~2 min |
| Pair features (≈251k train + ≈150–250k test pairs, vectorised per board) | 1–2 min |
| Pair GBDT 5 folds + refit (~220k × ~45 features) | 1–2 min |
| Item GBDT 5 folds + refit (~23k × ~80 features) | <1 min |
| Decode: OOF τ/Sinkhorn search on 1652 boards + test decode 622 boards | <1 min |
| Validation + write + re-read | seconds |
| **Total** | **~28–34 min** (fallback encoder: ~18–22 min) |

Memory: base encoder, batch 32 × 320 tokens bf16 ≈ 8–10 GB on 24 GB; test embeddings 5 × 8851 × 768 fp16 ≈ 68 MB;
OOF embeddings 31.6k × 768 fp16 ≈ 49 MB. HF cache under `WORK_DIR/hf_cache`.

Script-level validation: assert input schema, n_amendments == item rows per board, item_no is a permutation of
1..n, counts sum to n; after decode assert per-board multiset == counts and len(joint) == n; build the frame from
`sample_submission.csv` ids/order; `validate_submission`; write; re-read with `keep_default_na=False` and
`json.loads` every cell. No fallback or placeholder writes; a failed assertion raises.

## Metric-aware training & decode

- Loss/target alignment: proper scoring rules (CE/BCE) so stage-2b probabilities are usable marginals for the
  expected-utility decode; board-equal weights (1/n_b) mirror the metric's per-board averaging (G-ii). Chance
  correction makes per-item weights 1/(n(1−e)) inside a board, which only rescales whole boards — the 1/n_b
  weight captures the main effect; exact `W_c+W_d` weights are a roadmap ablation.
- Counts as evidence: counts fractions and count-aware cutoff margins in stage 2b; Sinkhorn makes marginals
  consistent with counts; the assignment decode enforces them exactly (constraint inside the valid space, combined
  with model evidence — G-vii).
- Decode is exact for E[carried] + E[disposal] given marginals (linearity); the joint partition is an approximate
  ARI maximiser (average linkage with an OOF-chosen cut), trained on the scored population only.
- Fates↔joint coupling (A4): joint informs fates via `Σ p_joint·p_adopt` and expected-group-size features; fates
  inform joint via stage-1 fate probabilities in pair features. Train diagnostics to print: share of predicted
  joint groups with >1 decoded adopted member (truth: 33/726 = 4.5% of groups with an adoption).
- Calibration: report OOF reliability of p_adopt per n-bucket; if the cross-fitted gain is positive, fit a single
  temperature on stage-2b logits (one scalar) — roadmap step 4, not primary.
- Bounded selections: τ grid inside [0.2,0.8] so the cut cannot collapse to all-singletons/all-one.

## Structural signals

| Invariant / structure (verified on train) | How it is used |
|---|---|
| Fate multiset = counts (4392/4392) | assignment decode; counts features; Sinkhorn |
| item_no random, items exchangeable | per-item encoding, swap-symmetric pair features, permutation-equivariant aggregates; outputs reordered by item_no only |
| a==0 ⇒ (almost) no fell (41/1059 exceptions) | counts features let stage 2b learn it; no hard rule |
| Article-level deletion/rewrite adoption ⇒ others fall (97%); deletion never falls | learned kill-power head + noisy-OR cascade features; opening-verb/scope features |
| ≤1 adopted per joint group (98% of groups) | Σ p_joint·p_adopt feature; roadmap: decode-time penalty |
| Joint = transitive closure of "called together" | partition via linkage (transitive by construction) |
| Joint scored only when truth has a pair | pair model trained on that population; τ selected on it |
| Identical edits deduplicated within a board | no exact-match feature; near-duplicate Jaccard + quoted-anchor overlap instead |
| Board context changes fates of identical texts (46%) | board fields in the encoder prefix and stage 2 |
| Test boards: n in [4,80], >=2 distinct fates | headline CV filter; stage-2 training filter; pair model on n<=80 |

## Experiment roadmap

Dev iteration caches stage-1 OOF/test outputs to disk for stage-2/decode experiments (dev only; `solution.py`
always recomputes from raw data). Use `.claude/scripts/kaggle_gpu_run.py` for encoder runs.

1. **Contract + metric + validation** (no GPU): fold builder (validation-architect), oracle decode tests (fates
   and joint), counts-respecting random = ~0 on test-like boards, half-rows test harness. Stop when all oracles
   pass on all 4392 boards.
2. **Cheap end-to-end baseline (CPU, dev yardstick only)**: stage 2a + 2b + decode with hand/lexical features
   only (no encoder) → valid CSV, test-like OOF. Expect ≥ 22 (above the 18.8 rule yardstick). Not a submission
   candidate (grey under NLP regime) — it is the floor the encoder must beat.
3. **Primary with encoder** (camembertav2-base, 2 epochs): full pipeline. Gate to spend credit #1: valid file,
   all oracles pass, test-like OOF ≥ 30 and ≥ +5 over step 2, each term positive in every fold, runtime ≤ 40 min.
   Compare camembert-base fallback on the same folds (paired) once.
4. **Metric-aware refinements** (cheap, cached): Sinkhorn on/off; pair-model population (scored-only vs all);
   exact metric weights; one temperature; kill/grouped heads ablation (requires one encoder rerun); counts in
   encoder head (one encoder rerun). Keep each only if it passes the noise rule.
5. **Structure/diversity** (only after 3–4 are measured): (a) light set-transformer on cached stage-1 embeddings
   + board context (2 layers, no positions) producing 5-way logits and pair logits, trained per board with the
   same folds, blended into stage 2b as an OOF feature (not fixed-weight averaging); (b) directed cascade pair
   model "P(i fell | j adopted)" trained with gold adopted j, aggregated by noisy-OR with predicted p_adopt;
   (c) two-pass self-conditioned features (decoded adopted set → recompute cascade features). Stop when two
   consecutive levers fail the noise rule.
6. **Second encoder seed or family** only if runtime headroom ≥ 15 min after step 5; average logits.
7. **Final fixed plan**: clean run twice on A10G, diff CSVs (must be identical or near-identical; test-pred
   correlation ≈ 1), runtime with ≥30% headroom, compliance scan, half-rows test, presubmit reviewers. Credits:
   baseline (step 3) → best single (step 4) → final (step 5/6).

## Compliance audit

CLAUDE.md §7 against this plan:
- Test file read only for per-board inference (items of that board + its counts/meta); no stats, vocab, scaler,
  clustering or normalisation across test boards. Lexical similarities need no fitted vocabulary. ✓
- No `time` in control flow; no `cuda.is_available()` / `cpu_count()` branches; no try/except fallbacks; fixed
  epochs/folds/rounds/threads/workers; seeds everywhere; LightGBM deterministic. ✓
- Constants: principled defaults or in-script OOF selections (τ, Sinkhorn, pair weighting, GBDT rounds via
  validation early stopping). Dev-time choices (encoder, linkage, max_len) are architecture choices justified in
  comments, not tuned thresholds. ✓
- External data: none; weights only from HF (pinned revisions). ✓
- Strip-the-ML: ≈0 without trained components. ✓ Model-heavy part dominates (encoder + 2 GBDTs drive every score;
  regex only extracts deterministic categories/numbers as features). ✓
- Readable single file < 512 KB, docstring with a requirements map (counts decode, per-board independence,
  train-only fitting, no external data, fixed plan). ✓
- Challenge-specific: no outcome lookups, no model trained on AN records, item_no not used, ids not used. ✓

## Open questions & assumptions

Reviewer questions:
- **Q1** General French encoders (CamemBERTa-v2: CulturaX/HAL/Wikipedia; CamemBERT: OSCAR/CCNet) are web-pretrained
  and may incidentally contain public assemblee-nationale.fr pages. We read "any model trained on them" as models
  trained on these amendment/outcome records specifically; a general-purpose LM is allowed. If a reviewer reads it
  more strictly, the pipeline would need a from-scratch encoder (much weaker; no plan B prepared beyond noting it).
- **Q2** Hand-parsed deterministic categories (opening verb, "cet article" scope, cited alinéa numbers, quoted
  anchors) fed into trained models: we treat them as standard domain features (CLAUDE.md §2.2); the cascade itself
  is learned. Low risk, but it is the one spot a reviewer could call "extracting the dataset's structure".
- **Q3** Expected-utility assignment decode using the board's own counts: required by the output grammar; we
  assume metric-aware decoding is acceptable since the description does not ban it.

Assumptions / could not verify:
- Throughput numbers are estimates (no GPU or torch in this sandbox); profiling one epoch on A10G-class hardware
  decides camembertav2-base vs camembert-base and max_len.
- `transformers` in the Kaggle image must load `deberta-v2` with the model's `tokenizer.json` (no sentencepiece
  file in the repo); verify in the first GPU run. If it fails, switch to the fallback in code (dev decision).
- scikit-learn/scipy are present in this sandbox now (the audit reported otherwise); LightGBM and torch are not —
  stage-2/decode dev on CPU needs them installed in the dev environment or run on Kaggle.
- The validation-architect owns the exact fold algorithm (size-aware, bill union by ≥3 shared exact edits).
- Test slice composition (two evaluation slices by bill) unknown; private noise ±2.5–4 points is an estimate.
- Expected score bands are estimates from a rule yardstick and term-wise reasoning, not measurements.
