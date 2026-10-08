# Eris plan: Whose Second Half Is It (BLIND strategist run)

Inputs read: `corpus/whose_second_half/CHALLENGE.md`, which is itself a reconstruction and not verbatim platform text, plus
CLAUDE.md and the eris-playbook references. **No dataset was available.** Every number about rows, columns, distributions,
hardware or runtime below is marked *assumption* or *unverified* and must be confirmed by the first audit step (roadmap
step 0) before any code is frozen. Pattern ids cited are from the playbook (A*, C*, D*, E*, G*, V*, Q*, L*).

---

## Contract & decision unit

- **Decision unit = one row of `test.csv`**: six prefix sessions (first halves) and six shuffled continuation tokens.
  The rows are independent of each other. Within a row the output is a **bijection** (a permutation of 1..6):
  `match_k` = the 1-indexed position (in the row's candidate list) of the continuation that belongs to prefix k.
- **Valid answer**: an id column plus `match_1..match_6`, integers in 1..6, one row per test row, in sample order.
  *Unverified*: exact column names, the id column name, whether the six candidates are one list-valued column or six
  columns, and whether a non-permutation answer is legal. The script reads all of these from `sample_submission.csv` and
  `test.csv` and asserts them. A permutation is always legal, so we always ship one.
- **Metric (assumed from the reconstruction)**: mean over rows of (correct slots / 6) = per-slot accuracy. Chance = 1/6
  for any fixed guess, because the candidates are shuffled. The chance-corrected `(acc-1/6)/(5/6)` is a diagnostic only.
  *Alternative reading to keep ready*: exact-permutation accuracy (all 6 right). The decoder differs by reading
  (see Metric-aware decode), so the script computes both and the description's formula picks one.
- **What the metric rewards**: the expected number of correctly matched slots. Under a posterior over permutations, the
  Bayes-optimal *valid* answer maximises sum_i P(prefix i -> candidate pi(i)): this is Hungarian on the posterior
  **marginals**, not on raw scores (A3 / exact assignment likelihood). If non-permutations were legal, a per-slot argmax of
  the marginals would be marginally better in expectation. We still ship a permutation (safe, and in practice equal).
- **Pipeline stages, diagnosed separately (three-stage diagnosis)**: (1) *coverage*: trivially 100% because the answer is
  always among the 6 (closed world); (2) *pair scoring*: how well a learned s(prefix, continuation) ranks the true
  continuation for each prefix (per-prefix top-1 and MRR before decoding); (3) *decode*: the gain of the joint bijection
  decode over the per-prefix argmax, and the gap to the oracle decode.
- **Structural insight that shapes the model**: under a bijection, any score offset that depends only on the prefix or
  only on the candidate cancels in every permutation's total. Candidate popularity (a popular artist "fits"
  everyone) and prefix diversity are therefore absorbed by the joint decode. Raw per-pair scores still need row/column
  normalisation before a per-prefix argmax, but the permutation likelihood is invariant to them. The learned scorer only
  has to be right up to row and column offsets, which is why exact permutation NLL (below) is the right training loss.

## Compliance regime

- **Domain bucket: unknown** (*assumption*: recommender/sequence data, filed as Tabular or RAG/ranking; "From-scratch"
  is possible because all known solvers trained embeddings from scratch). The plan holds under the strictest of these:
  **no pretrained weights at all**, everything learned from the released listens, CPU-feasible, plus a genuinely trained
  neural sequence model whose scores feed a trained GBDT. That is compliant in the Tabular, RAG/ranking and From-scratch regimes.
  The trained ranker is load-bearing (RAG rule: "hand features + LambdaMART alone" is grey). The neural sequence model is
  trained in-script, and its scores alone are reported as a member.
- **Hard rules from CLAUDE.md that apply directly**:
  1. **Test half-2 guard.** If `listens.csv` contains `half==2` rows for sessions referenced by `test.csv`, they are the
     held-out answer (matching them is a lookup). The script drops every `half==2` row of every test-referenced session at
     load, asserts that none remain, and comments why. Never read `private/` or answer files.
  2. **No fitting on test inputs** (§2.3 #5). The vocabularies, embeddings, co-occurrence and transition statistics and the
     sequence model are fitted only on *bank sessions*: sessions referenced by train rows (both halves, out of fold) plus
     any listens sessions referenced by neither train nor test (*unverified whether such sessions exist*). Test prefixes
     (first halves of test sessions) are **not** used as unlabelled training text. This is the main reviewer question
     (below). Its cost is measured offline by a simulation (roadmap step 3c), not by touching test.
  3. **Row-local inference**: each test row's answer is a function of its own 6 prefixes, 6 continuation tokens and the
     train-fit models. There is no normalisation, popularity count or clustering across test rows. Popularity comes from bank
     sessions only. Half-rows test: drop half the test rows and the kept rows' outputs must be identical.
  4. **No id/order features**: session ids, continuation token strings, row order and candidate position are never
     features. The id-only probe (V19b) must score at chance under grouped CV.
  5. Fixed work plan: fixed folds, epochs, rounds, trials, threads; time used only for logging; hardcoded `DEVICE="cpu"`
     (*assumption*: CPU-only with about 10 cores. If the real description promises an A10G, change the one constant and the
     epoch counts and re-profile; there is no runtime branch).
- **Sibling question**: features compare a prefix with the *other candidates in its own row*. These are visible inputs
  of the same row and the task is defined over them, so they are row-local and legitimate (C3 relative transforms). There is
  no cross-referencing of hidden siblings in other rows.
- **Strip-the-ML test**: exact-overlap features (same artist as the last prefix play) plus Hungarian would score well
  above chance on their own. The script never contains such a decision path: every feature enters the trained LightGBM,
  and the decoder consumes only model marginals. The log reports the learned feature importances and the sequence model's
  stand-alone accuracy, so the learning is visible (engineering §5). Text normalisation (lower-casing, stripping
  "feat."/"(Remastered)" for name-token overlap) is cleaning, not answer-deciding regex.
- **Recombining real units** (sampling extra 6-slates from real bank sessions, re-cutting real sessions at other
  points to create extra (history, next plays) pairs) is defensible augmentation (L: resampling which slice is observed).
  No fabricated sessions.

## Data findings

*No data was available: everything here is either from CHALLENGE.md or an audit to run first.*

Known from the reconstruction:
- `listens.csv`: per session, ordered plays with `relative_position`, `artist`, `release`, `recording`, `half` (1/2) and
  `seconds` (time within the day).
- `continuations.csv`: per continuation token, up to 2 entries (`rank` 1-2) of (artist, release, recording). That is the
  first one or two plays of a second half. No timestamp is mentioned (*unverified*). If seconds are present, gap features
  (`cont_start - prefix_end >= 0`, gap size) become first-order and must be added.
- `train.csv`: `prefixes` (6 session ids), `candidates` (6 tokens), `match_1..match_6`. `test.csv` has the same without
  matches.

Audit battery to run before modelling (train and bank only; outputs go into this section):
1. Shapes: number of train/test rows, sessions, plays per session (half-1 length distribution, p50/p95 for the
   truncation length L), unique artists/releases/recordings, share of continuations with 1 vs 2 entries.
2. **Bijection guarantee**: every train row's `match_*` is a permutation of 1..6, and every row has exactly 6 and 6.
   Do the same session ids or tokens recur across rows (train-train and, by id only, train-test)? Recurrence implies union-find
   groups and a possible leak to audit.
3. **Which sessions have half 2 in listens.csv**: train-referenced, test-referenced, unreferenced. This decides guard 1 and
   whether extra unlabelled bank sessions exist.
4. **Information ceiling slices** (train, label-aware, analysis only): share of continuations whose rank-1 artist equals
   the last prefix artist; appears anywhere in the prefix; whose rank-1 recording appears in the prefix (replays);
   same release as the last prefix play; no overlap of any kind with the prefix ("cold continuation"). Share of rows where
   two prefixes end with the same artist, or two candidates have identical content (irreducible ambiguity: the
   expected per-slot loss is computable). The cold share bounds what overlap features can do. The rest needs learned
   taste similarity.
5. **How the halves were cut**: is the split at a play count midpoint, at a clock time, or at the largest pause? Compare the
   `seconds` gap at the boundary against the within-half gaps. If it is a pause, the continuation is a "new listening
   episode", and the sequence model needs a boundary token and boundary-specific fine-tuning.
6. **How rows were assembled**: are the 6 sessions in a row random, or similar (same day/time-of-day, shared artists,
   same user)? Compare within-row prefix-prefix artist overlap with random pairs. If similar, decoys are hard and the
   validation slates must be built the same way (do not resample random 6-slates for training without matching this).
7. Is there a user or date field (*unverified*)? If dates exist, check whether test sessions are later than train
   (by session metadata only, not feature distributions). That would switch validation to time-ordered folds.
8. Leakage audit: id-only and order-only probes at chance; label-shuffle run at chance (V19a/b).

## Validation design

- **Hidden split (assumption)**: test rows are built the same way as train rows from disjoint sessions; random at the row
  level. If audit 7 shows a time ordering, use blocked time folds instead.
- **Groups**: union-find over train rows sharing any session id or continuation token (`make_groups.py`-style). If a
  user id exists, group by user as well and report both (if users span train/test, the user-level split is pessimistic).
- **Folds**: GroupKFold, K=5, two split seeds for the comparisons that matter (fixed in-script). The same fold assignment
  defines the **bank exclusion**: for train rows in fold k, every statistic and the sequence model come from
  bank_k = all bank sessions except fold-k sessions (the prefix halves and the second halves of fold-k sessions are both
  excluded, which mirrors test, where the test sessions are absent from the bank). This is the critical leak: the pair
  (last prefix play -> first continuation play) of a train session is literally in a transition table built from all train
  listens.
- **Matched statistic sizes (V8)**: train-row features come from 80% banks. Test features are computed under each of the
  5 bank_k (also 80%), the stage-2 model is applied to each, and the 5 marginal matrices are averaged. Train-time and
  test-time features then have the same distribution. One extra free benefit: a 5-bank ensemble.
- **Stage-2 OOF**: LightGBM is cross-validated on the same outer folds. Bias: fold-j features contain counts from
  fold-k second halves (standard stacking), which is mildly optimistic. Measure it once with a strict nested check for the
  count features only (banks excluding both j and k: 20 cheap count banks) and report the gap.
- **Metric re-implementation and unit tests**: perfect = 1.0; a derangement = 0.0; identity guess on shuffled
  train labels = about 1/6; uniformly random permutations = 1/6 in expectation; exact-permutation variant tested likewise.
- **Oracle decode check**: one-hot gold scores through the 720-permutation marginal + Hungarian decoder must reproduce
  100% of `match_*`. Measure the decoder runtime per 10k rows.
- **Yardsticks under the same folds**: (a) chance 0.1667; (b) a single learned-free signal ranked by a trained logistic
  regression (last-artist match only) as the floor; (c) the sequence model alone; (d) full stage-2.
- **Noise rule**: accept a change only if the paired per-fold gain passes the corrected resampled t (V16) and is
  sign-consistent across folds and both seeds. Sanity holdout: 15% of groups that are untouched by any selection
  until the final run.
- **Bias statements**: (i) 80% banks for OOF vs 80% banks for test = matched, unbiased by size. (ii) If test sessions
  share users with train sessions and our groups do not split users, CV is test-like. If test users are new
  (*unknown*), CV is optimistic by the "user memory" share. Report CV on a user/artist-cluster grouped split as a
  pessimistic bound if no user id exists (cluster bank sessions by their artist sets). (iii) The selection of T, blend
  weights and HPO is cross-fitted, so it is reported unbiased.

## Overfit/underfit risks

| Risk | Where | Mitigation |
|---|---|---|
| Transition-table leakage (own session's boundary pair in the counts) | count features, sequence model | bank_k exclusion, asserted by a unit test: a fold-k session id never appears in bank_k |
| Train/test feature-distribution mismatch (100% vs 80% banks) | stage-2 | 5 fold banks for test, averaged |
| Memorising rare recordings/ids | sequence model vocab | min count 2 in bank_k (else OOV bucket per level); recording embedding backed off to release and artist embeddings (sum of three levels) |
| Over-tuned GBDT on correlated pair rows | stage-2 | grouped folds by row; moderate leaves (31-63), min_data_in_leaf >= 100, feature_fraction 0.8, fixed rounds = 1.1 x mean best iter |
| Too many decode knobs | decode | one scalar temperature T only, cross-fitted; Hungarian has no knobs |
| Underfitting taste for cold continuations (artist never in prefix) | features | learned sequence model + item-item co-occurrence PMI/SVD similarity; this is where most of the remaining accuracy lives |
| Information lost to truncation | sequence model | L = p95 of half-1 length (*from audit*, guess 128), most-recent plays kept; exact-overlap features use the full prefix |
| Loss mismatched with metric | stage-2 | exact permutation NLL for the neural members; binary logloss for GBDT followed by a perm-NLL temperature fit on OOF |
| Determinism noise on CPU threads | sequence model | fixed `torch.set_num_threads(10)`, deterministic algorithms, seeded generators; double-run diff |

## Recommended approach (primary + fallback)

### Primary: two-stage "learned evidence -> exact bijection decode"

**Stage 1, per fold bank bank_k (k=1..5), all fitted on bank sessions only:**

1a. *Count statistics (cheap, deterministic)*, all smoothed with backoff:
   - transition counts and conditional probabilities at three levels: recording->recording, release->release,
     artist->artist, plus a two-hop variant (a->x->b, which catches skipped tracks) and the "same release, next track"
     signal learned from successor counts. It is never hand-coded from track numbers.
   - item popularity (bank counts) for each level, used as a feature (the model learns to discount popular matches) and
     for PMI normalisation (C19 reference-normalised evidence).
   - item-item co-occurrence within bank sessions -> PPMI -> truncated SVD (scipy `svds`, fixed `v0`) at the artist
     level (dim 64). This gives cosine similarity between the prefix artist profile (recency-weighted mean) and the
     continuation artists.
1b. *Learned sequence model ("next-play model"), trained from scratch with PyTorch on CPU*:
   - each play = E_artist + E_release + E_recording (OOV buckets per level) + position-from-end embedding; a 1-layer
     GRU (d=128) over the last L plays of the session.
   - objective: next-play prediction at **every position** of every bank session (dense signal), sampled softmax over
     recordings with 512 shared negatives drawn by popularity and a logQ correction, plus auxiliary artist and release
     heads. Then a **boundary fine-tune**: half-1 -> first play of half-2 pairs from bank sessions, with a boundary
     token (if audit 5 shows the cut is at a pause, this step matters most).
   - pair score: log P(c1 | prefix) + 0.5 * log P(c2 | prefix, c1) at each level (artist, release, recording log-probs
     as separate features). Under the bijection the popularity baseline cancels (see Contract), but per-pair PMI versions
     (minus the unconditional log P(c)) are also given to stage 2.
   - fixed 3 epochs over all positions, then 1 boundary epoch; AdamW lr 2e-3, batch 256 sessions, seed fixed.

1c. *Row-local exact-overlap features* (no bank needed): for each (prefix p, candidate c) and for c's rank-1 and rank-2
   entries separately: artist == last prefix artist; release == last prefix release; recording in prefix (replay);
   count and recency (plays since the last occurrence) of the artist, release and recording in the prefix; share of the
   prefix that is this artist; the last-k (k=1,3,10) window versions; normalised artist-name token Jaccard (catches
   "A feat. B"); prefix length, distinct-artist ratio (predictability), hour of prefix end (`seconds`/3600), and whether c
   has 1 or 2 entries.

**Stage 2: trained pair scorer.** LightGBM (binary objective, label = 1 for the true pair; 36 pairs per row) on 1a-1c
plus **row-relative transforms** (C3) of the 8 strongest scores: value minus the max over the 6 candidates for this
prefix, minus the max over the 6 prefixes for this candidate, rank in both directions, and a 10-iteration Sinkhorn-normalised
version of the sequence-model score (row-local, so it is legal). OOF on the outer folds; final model refit on all train
rows at fixed rounds. HPO: defaults first, then optionally 20 seeded TPE trials (roadmap step 6).

**Decode (row-local, exact):** S = stage-2 logits (6x6) / T. Enumerate the 720 permutations with a constant 720x36
incidence matrix: perm score = INC @ vec(S), softmax, marginals M = INC^T p reshaped 6x6 (A18 / exact assignment
likelihood). Average M over the 5 test banks. `scipy.optimize.linear_sum_assignment(M, maximize=True)` -> `match_*`.
T is a single scalar fitted by minimising the permutation NLL of the true permutation on stage-2 OOF, cross-fitted.

**Why this fits this data**: the evidence is local (the last plays predict the next ones: album runs, same-artist runs,
replays), together with global taste similarity for cold continuations. Counts capture the first and the learned sequence
model captures both. The output is a tiny enumerable structure, so exact inference is free. All five known solvers
trained embeddings from scratch, which is consistent with stage 1b.

Expected score (estimate, not a promise): per-slot accuracy **0.60-0.80** (chance 0.167). Reasoning: in
personal listening logs consecutive plays share the artist roughly 40-60% of the time and albums are often played in
order. The bijection then lifts ambiguous prefixes once the confident ones are assigned. An overlap-only GBDT is
probably 0.50-0.65. The lower half of the range applies if rows are built from deliberately similar sessions (audit 6)
or if cold continuations dominate (audit 4).

### Fallback (if the sequence model is unstable, too slow on the real CPU, or adds less than the noise)

The same stage 1a + 1c + stage 2 + decode without 1b, replacing the GRU with a **from-scratch skip-gram-style item
embedding trained in torch** (one fixed-epoch pass, deterministic) at the artist and release levels. This keeps a genuinely
trained neural representation in the pipeline for the RAG/From-scratch regimes. Expected about 0.02-0.05 lower than the primary
(estimate). Compliance-adjusted: equally clean.

## Rejected options

- **Rule decoder** (last-artist match / overlap count -> Hungarian, no trained model): fails strip-the-ML (Q1); it is used
  only as a *feature*, with a logistic regression as the yardstick.
- **Using test prefixes (half 1 of test sessions) as unlabelled training data** for embeddings/co-occurrence: likely
  some extra lift on cold items, but it fits on test inputs (§2.3 #5, Q2). Not built without written reviewer approval
  saved in `reports/reviewer_approval.md` (CLAUDE.md 2.3A). The cost is measured by simulation instead.
- **Any cross-test-row statistic** (candidate frequency over test slates, test-wide normalisation, EM over test rows):
  rejection class R.
- **Pretrained text encoders on artist/release names** (e.g., multilingual MiniLM): might help cold artists, but they
  violate a possible from-scratch rule (the reconstruction says no solver used one), and names carry little taste signal.
  Rejected under the strict reading. If the real description explicitly allows HF weights, re-evaluate as a member.
- **Large transformer sequence model (SASRec-size) or joint 6x6 set-transformer as the primary**: heavier, CPU-risky,
  and it is not shown to beat the lean GRU + GBDT. The set-transformer enters the roadmap as a diversity member only
  (step 5).
- **Fold-averaged stage-2 models instead of a full refit**: the refit on 100% of rows is preferred (V9). The banks stay 80% by
  design (matched sizes).
- **Per-slot independent argmax**: violates the bijection, which costs accuracy. Hungarian on marginals dominates it.

## Fixed work plan & runtime budget

Assumed hardware: CPU, 10 threads, about 60 GB RAM, limit unknown (target <= 40 min, >= 30% headroom against 1 h).
*Assumed sizes*: about 50k-150k sessions, about 5-15M plays (unverified; profile first and scale L, epochs and embedding
dims as **fixed constants**, never at runtime).

| Stage | Work | Estimate |
|---|---|---|
| Load + guards + parsing | read csvs, drop test half-2, build session arrays | 1-2 min |
| 1a counts x5 banks | groupby transitions, popularity, PPMI-SVD (artist) | 3-5 min |
| 1b GRU x5 banks | 3 epochs all positions + 1 boundary epoch, CPU, d=128 | 12-18 min (largest risk; profile one epoch) |
| Pair features | train rows OOF (1 bank each) + test rows x5 banks, vectorised | 3-5 min |
| Stage 2 | LightGBM 5-fold OOF (early stopping on the fold metric, fixed patience) + full refit at fixed rounds | 3-5 min |
| Decode | 720-perm marginals for test x5 + Hungarian | < 1 min |
| Validate + write | `validate_submission` + reload with `keep_default_na=False` | < 0.5 min |
| **Total** | | **about 25-35 min** |

Fixed constants: `N_FOLDS=5, SPLIT_SEED=0, L=128, GRU_DIM=128, EPOCHS_ALL=3, EPOCHS_BOUNDARY=1, BATCH=256,
N_NEG=512, LGB_MAX_ROUNDS=3000, LGB_PATIENCE=100, LGB_THREADS=10, TORCH_THREADS=10, DEVICE="cpu"`. Seeds:
`random`, `numpy`, `torch`, `PYTHONHASHSEED`, the LightGBM `seed` with `deterministic=True, force_row_wise=True`.
If the step-0 profiling shows 1b over budget, reduce `EPOCHS_ALL` or `L` as a hardcoded constant, never adaptively.
Memory: the GRU embeddings (about 200k recordings x 128 x 4 bytes, about 100 MB) and the count tables are small. The 5 bank copies
are processed sequentially and freed.

## Metric-aware training & decode

- **Loss and metric alignment**: the per-slot accuracy of a bijection equals the expected number of correct
  assignments, so decode = Hungarian on the exact permutation marginals (A3/A18). If the description's metric is exact-row
  accuracy, decode = the MAP permutation (Hungarian on S). Both are implemented, and the description decides.
- **Training loss**: stage 2 uses binary logloss on 36 pairs. Then a single temperature T is fitted by **exact
  permutation NLL** on OOF: loss = logsumexp(INC @ vec(S/T)) - sum of the true-pair S/T. This is cross-fitted on group
  halves and reported beside the in-sample value. The step-5 neural row member trains directly on this exact NLL (lesson 2).
- **Hierarchical averaging**: the metric averages slots within rows and then rows. Every row has 6 slots, so no reweighting
  is needed. All rows weigh equally in the LightGBM groups (36 pairs each).
- **Decoder oracle**: gold one-hot -> 100% reproduction, asserted in a unit test (not in the shipped path).
- **Calibration diagnostics**: the reliability of the marginals M per slot rank (the most confident vs the least
  confident prefix in the row), reported on OOF only. No extra decode constants beyond T.
- **No metric gaming that needs test statistics**: everything stays per row.

## Structural signals

1. **Bijection** (guaranteed per the reconstruction; verify on all train rows): exact 720-permutation inference and
   Hungarian decode. Prefix-only and candidate-only offsets cancel, so the scorer is relieved of popularity calibration.
2. **Causal alignment (lesson 4)**: the last plays of half 1 cause the first play of half 2, and c1 causes c2. Features
   are built around the boundary (last-k windows, recency); the sequence model scores c2 conditioned on c1.
3. **Multi-level item hierarchy** (recording ⊂ release ⊂ artist): every statistic and embedding at all three levels
   with backoff. This gives robustness for rare recordings.
4. **Replays and album order**: learned from bank successor counts and the sequence model. Not hand-coded from track
   numbers.
5. **Symmetry of the row**: prefix order and candidate order carry no meaning. Features are permutation-equivariant (row-relative
   transforms are computed over sets). The step-5 neural member uses no positional encoding over the 6x6 grid. Training
   augmentation shuffles the candidate order (labels permuted accordingly).
6. **Dense next-play supervision**: every consecutive pair inside bank sessions is a real training example of the
   same kind of event as the label (lesson 10b: labels of the same kind as the inputs, cross-fitted by bank exclusion).
7. **Extra slates from real sessions** (roadmap step 5, only if audit 6 shows the rows are random): resample 6-slates from
   bank sessions to train the neural row member. These are recombinations of real labelled units, not synthetic data.

## Experiment roadmap

0. **Audit** (Data findings 1-8) and profiling of one GRU epoch on the real data. Fill the assumed numbers into this plan.
   Stop if guard 1 finds that test half-2 is present (it must be dropped; document it).
1. **Contract**: metric implementation + unit tests; oracle decode; submission validator against `sample_submission.csv`;
   chance and last-artist yardsticks on 5 grouped folds x 2 seeds.
2. **Baseline end-to-end**: 1a + 1c features -> LightGBM -> Hungarian on sigmoid scores. This is the first valid CSV and
   the first credit (baseline).
3. **Representation**: (a) add 1b sequence-model scores; (b) compare boundary fine-tuning on vs off; (c) offline-only
   simulation of the transductive cost: on fold k, add fold-k half-1 listens to the bank and measure the gain (this
   quantifies what the compliant choice gives up, for the reviewer question; it is never shipped). Stop when the
   paired gain is below noise.
4. **Metric-aware decode**: temperature + exact marginals vs raw Hungarian (expected small but free); row-relative
   features; the strict nested check of stacking optimism. Credit 2: best single.
5. **Diversity**: neural row member, a small permutation-equivariant 6x6 axial-attention network over the pair-feature
   tensor, trained with exact permutation NLL (assumption diversity: it learns the competition effects jointly). Blend
   in log-marginal space with one weight, cross-fitted; keep it only if it beats the corrected-t bar. Second option:
   a recording-level vs artist-level-only sequence model. Credit 3: ensemble.
6. **Bounded HPO**: 20 seeded TPE trials on the LightGBM leaves/min_data/learning_rate, evaluated on inner folds of the
   training part only. Accept the result only if it beats the defaults on the sanity holdout.
7. **Final fixed-plan run** from a clean `working/`, run twice and diffed, half-rows test, compliance scan, runtime
   log. Credit 4: final. Keep 2 credits spare.

## Compliance audit

- Test reads: only per-row features of each test row's own prefixes and candidates. Test half-2 listens are dropped
  with an assert. No vocab, statistic or model is fitted on test prefixes. **PASS by design** (verify with the half-rows test).
- Clock/hardware/env branches: none. `DEVICE` is a constant, threads are constants, time is used only in `log()`. **PASS**.
- Hardcoded tuned constants: only principled defaults and fixed budgets. T, the blend weight and the HPO choice are
  fitted in-script on OOF. **PASS**.
- External data/weights: none (no pretrained models at all). Libraries: numpy, pandas, scipy, scikit-learn, lightgbm,
  torch (all core). **PASS**.
- Strip-the-ML: no rule decision path. Features feed the trained LightGBM. The neural sequence model is trained
  in-script and is load-bearing (its stand-alone OOF is logged). **PASS with one reviewer note** (overlap features are strong).
- Siblings: within-row comparisons are the task itself. No cross-row label lookups. Bank statistics exclude the row's own
  fold. **PASS**.
- Determinism: seeded everything, deterministic LightGBM, fixed CPU threads, `use_deterministic_algorithms(True,
  warn_only=True)`. Double-run diff required before upload.
- Output: validator asserts the columns/order/ids/no NaN, that the values are in 1..6, and that each row is a permutation.
  Written once at the end, never in an `except`.

## Open questions & assumptions

Reviewer questions (ask before the deadline; the plan works under both answers):
1. *May the first halves of test sessions (released in `listens.csv` as inputs) be used as unlabelled sequences to train
   embeddings/co-occurrence?* Plan: **no** (CLAUDE.md §2.3 #5) unless a written approval exists. The step-3c simulation
   quantifies the cost. If the description explicitly lists `listens.csv` as an unlabelled corpus for all sessions,
   ask anyway: the strict reading is the shipped one.
2. *Is the challenge bucketed as From-scratch / no-pretrained?* The plan uses no pretrained weights, so it is safe either
   way.
3. *Must the submission be a permutation?* We always ship one.

Assumptions to verify in step 0: the metric formula (per-slot accuracy), the submission schema (`match_1..match_6`, id
column name), CPU-only hardware and the runtime limit, data sizes, the existence of unreferenced bank sessions, whether
`continuations.csv` carries timestamps, whether user or date fields exist, and how the halves and rows were constructed.

Not verifiable in this blind run: anything about real data distributions, the AI baseline value, the true row counts,
and therefore the runtime estimate and the expected score range (both are labelled estimates).
