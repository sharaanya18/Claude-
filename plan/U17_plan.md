# U17 plan: unseen-pair shell pipeline synthesis (CPU, from scratch)

Status of evidence: NO dataset files were available. Everything under "Data findings" is a diagnostic to run plus an UNVERIFIED hypothesis. Nothing below was measured. All score numbers are estimates.

## Contract & decision unit

**One valid answer.** One row per test id, columns exactly `id,output`; `output` is a single non-empty string, at most 4,096 chars, no NUL, balanced POSIX quoting under `shlex.split` (lexical check only), CSV-escaped, ids unpadded, unique, same set as `test.csv` (160 rows).
Invalid (whole submission rejected): bad columns, missing/extra/duplicate/blank/padded ids, empty or over-length output, unbalanced quotes, NUL, non-string. Merely low-scoring: anything else, including commands that would not run.

**Metric (re-implement exactly, unit-test it).** Tokenize with Unicode word runs plus each non-whitespace punctuation char (assumed `re.findall(r"\w+|[^\w\s]", s)`; `-rf` -> `-`,`rf`; `$2` -> `$`,`2`; `*.txt` -> `*`,`.`,`txt`). Per case: `0.5*(1 - D/max(len P, len T, 1)) + 0.5*[P == T]`, D = token Levenshtein; final = plain mean over the 160 cases. Consequences:
- Whitespace and quote style cost nothing except via the quote characters themselves, which ARE tokens (`'` vs `"` matters); whitespace never matters. So the model can work on metric tokens and I only need whitespace for the shlex validity check.
- Half the credit is exact token-sequence match; the other half is graded, so a "central" prediction (right head skeleton, right flags, right literals) earns about 0.3-0.45 even when not exact. Expected-utility (MBR) decoding under THIS metric is directly justified.
- Exact match is capped by reference ambiguity (quote style, flag order, `-name` vs `-iname`, alternative idioms "not exhaustively annotated"). The ceiling is below 1 and must be estimated from train (see diagnostics).

**Independent unit / decision unit.** The test row is one command (1-3 outer pipeline stages), and since every evaluation command holds at least one withheld ordered head-head adjacency, every test case has >= 2 stages. The independent unit for validation is NOT the row but the **withheld-adjacency group** (all programs containing a given ordered head pair, including as a subpath) plus canonical-output duplicates and 5-word-span description neighbours.

**Pipeline stages to diagnose separately.**
1. Skeleton (candidate coverage): which ordered list of heads (n = 2-3) does the program use. Oracle: top-K recall of the gold skeleton.
2. Stage realisation (scoring): given a head, its stage slot and the description, produce flags, literal args, quoting. Oracle: score with the GOLD skeleton.
3. Decode: pool candidates, choose by expected metric (MBR). Oracle: best-in-pool score (upper bound for any reranker).

## Compliance regime

Domain: NLP-to-code seq2seq, **"From scratch" regime (CLAUDE.md 6.8) on CPU**. Challenge text overrides CLAUDE.md where they differ.

Explicit bans in the description (hard constraints):
| Ban | How the plan honours it |
|---|---|
| Pretrained models, embeddings, weights | None. No `transformers`, no timm, no HF downloads, no network at all. Embeddings (word + hashed char n-gram) and tokenizer/vocab are trained/built from train only. |
| External labeled corpora | None. |
| Original-source lookup, recovered source ids, hidden outputs, hardcoded test answers | No memorised knowledge of the source corpus is encoded; no man-page/flag lexicon typed from memory (flags/heads/vocab come only from train outputs). |
| Case ids as predictive features | `id` is only used to align the output rows. |
| Train from supplied public files only | Yes. |
| Whole-command retrieval as the solution | kNN retrieval exists ONLY as a dev control (floor), never shipped. |

Other binding rules (CLAUDE.md): fit every vocab/encoder on train only; per-row inference only (no test-wide statistics, no pseudo-labels, no test-time adaptation, no batch-dependent outputs); fixed work plan, no wall-clock branching; no `cuda`/`cpu_count` switches; seeds; source < 512 KB, plain readable; no synthetic labelled data; HPO only inside the script with fixed trial count.

**Conflicts resolved**
- CLAUDE.md says `device="cuda"` / A10G. This challenge says CPU for training, features and inference, so hard-code `device="cpu"`, fixed `torch.set_num_threads(4)` and `set_num_interop_threads(1)`, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set before importing torch. No `os.cpu_count()`.
- CLAUDE.md validator asserts `len(sub) == len(sample)` and ids equal to the sample. The description says "one row per test ID" and the sample is described as format examples (the quoted example row has id `example_case`). So the validator must check ids/rows against `test.csv` and columns against `sample_submission.csv`, and only check the sample's row count if it equals the test row count. UNVERIFIED which holds; inspect the sample first.
- The sample's outputs are training outputs for format only: never used as data or as a baseline.

Silent in the description (assumptions): runtime limit (assume <= 1 h, target <= 30 min at 4 threads, see budget); core count of the grader (assume 4, plan 2x headroom); allowed libraries beyond the stack (assume numpy, scipy, torch, scikit-learn available; use torch + numpy only in the shipped script); exact grader tokenization regex (assume above; the model is insensitive to small differences); whether stage-factored training counts as acceptable preprocessing (see Open questions).

## Data findings

All items are diagnostics to run on `train.csv` (4,107 rows, `id,input,output`) and hypotheses; **none verified**. Test files: only schema, 160 rows, id format and length statistics for runtime planning.

**D1. Integrity.** Shape, dtypes, empty/duplicate inputs, duplicate or conflicting outputs, trailing whitespace/backslash/newlines, non-ASCII. Hypothesis: clean; a few near-duplicate descriptions with different quoting.

**D2. Quote-aware stage parse (preprocessing only).** Split each output on top-level `|` (pipes inside `'..'`, `".."`, `\|` are not separators), take head = first shell word of each stage. Round-trip check: `" | ".join(stages)` re-tokenises to the original token list for 100% of rows; `shlex.split` succeeds on 100% of train outputs (if not, learn the exceptions before choosing the quote-balance guard). Report: stage-count histogram (hypothesis: 1-stage majority, roughly 55-75%; 2-stage most of the rest; 3-stage under 10%); number of distinct heads H (hypothesis 60-150 with a Zipf tail); heads with < 20 programs; env-assignment or wrapper-leading stages (`LC_ALL=C sort`).

**D3. Adjacency structure.** Count ordered head pairs (adjacent in a program) and subpaths of length 3; matrix of counts; which pairs have 25-150 programs (candidate pseudo-held-out pairs); for each head its slot distribution (first/middle/last). Hypothesis: heads fall into roles (sources `ls/find/cat/ps`; filters `grep/awk/sed`; transformers `sort/uniq/cut/tr`; sinks `wc/head/tail/xargs`); PMI matrix of adjacencies has low effective rank (SVD: rank <= 8 captures > 85% of variance). This supports the low-rank role transition model.

**D4. Piped vs standalone argument behaviour.** For each head with >= 20 occurrences, fraction of occurrences with no file operand at slot >= 2 vs at slot 1. Hypothesis: strong effect for `grep/sort/wc/head/tail/cut/uniq` (file operand dropped when piped). This is the "data flow" the model must learn and it is learnable from non-withheld adjacencies, since each evaluated head has >= 20 training programs.

**D5. Copy statistics.** Fraction of output metric tokens (a) in the output vocab at freq >= 3, (b) not in it but present verbatim among description tokens (copyable literal), (c) neither (must be generated from sub-pieces). Quoted literals found verbatim in the description. Hypothesis: (b) is large (names, patterns, numbers), (c) under 5%. Share of `'` vs `"` quoting (ambiguity source), share containing `$`, braces (awk/sed programs), regex metacharacters.

**D6. Lengths.** Description tokens (hypothesis median 12-20, p95 about 45), output tokens per stage (median 6-12) and per program (median 10-18, p95 about 50). Sets max source/target lengths and CPU cost.

**D7. Description-side ambiguity ceilings.** Group rows by shared 3-/4-word spans; among near-duplicate descriptions with different groups, how often outputs are token-identical, and the mean similarity. This estimates the exact-match ceiling (hypothesis 35-55% exact on genuinely new compositions) and the similarity ceiling.

**D8. Control floors under the pseudo-split (see Validation).** kNN-retrieval control (TF-IDF over descriptions, train-fit only, dev-only): hypothesis score 0.15-0.25 with exact about 0. Majority-skeleton control. These floors confirm the split behaves as described.

**D9. Head determinability.** Skeleton and stage-head accuracy from a bag-of-n-gram classifier trained on stage-1 train rows only (dev diagnostic): how much of the head set is recoverable from the description, and how often a description names a head's function in non-obvious words ("count" -> `wc`, "unique" -> `sort -u` vs `uniq`). Hypothesis: 80-90% of stage heads recoverable; remaining errors are semantic aliases.

**D10. Oracle/round-trip checks (mandatory before modelling).** (i) Metric unit tests: identical -> 1.0; same tokens different whitespace -> 1.0; reversed token order -> exact 0 and similarity from Levenshtein; empty/constant `ls` -> low; (ii) tokens -> string (using the word-boundary flag below) -> `shlex.split` ok -> metric tokens equal for 100% of train; (iii) feeding GOLD skeletons and GOLD stage tokens through the decoder/MBR/join path reproduces every train output exactly (proves constraints, join and formatting lose no valid answer).

**D11. Sample/test schema.** Row counts and ids of `sample_submission.csv` vs `test.csv`; id patterns for any order or length correlation with outputs (do not exploit; ids are opaque); test description length stats only (for runtime).

**D12. Leakage suspects.** Rows whose description contains the full output verbatim (trivial copies); any column other than `input` informative about `output`. Expect none beyond the opaque id.

## Validation design

**How the real split was made.** Seven ordered head-head adjacencies removed from train entirely (training programs containing them, even as subpaths, are dropped); evaluation = programs containing a withheld pair, at most 50 per pair, six pairs; groups by canonical tokenized output never cross; train requests sharing any contiguous 5-word span with an eval request are dropped; evaluation commands are distinct canonical commands.

**Local mimic (the "withheld-adjacency CV"), built from train only.**
1. Parse stages (D2), count adjacencies (D3).
2. Select pseudo-withheld ordered pairs deterministically: pairs with train program count in [25, 150], both heads having >= 40 programs not containing the pair; ordered by (count, lexicographic), taking every k-th so counts spread over the range. Group them into folds of 2 pairs: `F = 3` folds for the in-script run (6 pseudo pairs), `R = 2-3` alternative pair selections (split seeds) for dev experiments, plus a **dev-only fourth fold, "sanity"**, whose pairs are never used for any selection.
3. Fold f: eval set `E_f` = all train programs containing any pair of fold f as an adjacency (including subpaths), de-duplicated to distinct canonical commands, capped at 50 per pair. Train set `T_f` = train programs containing NONE of the fold's pairs, minus programs whose request shares a contiguous 5-word span with any request in `E_f`, minus canonical-output duplicates of `E_f`. Union-find over (canonical output, shared 5-word span) builds groups so no group straddles `T_f`/`E_f`. A row-level random split must never be used (it would reward memorising near-duplicates, which the real split forbids).
4. Report the exact official metric on `E_f`: mean +/- std over folds and over split seeds, per-pair mean, and the decomposition (similarity part, exact part, skeleton accuracy, oracle-in-pool). Paired comparisons on identical folds and seeds; accept a change only if it wins on most folds and the mean gain exceeds roughly one standard error (expect SE about 0.02-0.03 given 150-300 eval rows per fold set).

**Direction of bias (estimate with reasoning).**
- Pessimistic: `T_f` loses the fold's adjacency programs plus 5-gram neighbours on top of the 7 already withheld (maybe 8-15% fewer rows), and the model sees one extra unseen head-pair family.
- Optimistic: pseudo pairs are drawn from pairs frequent enough to have 25-150 programs, so their heads have richer argument variety in train than some real pair; real eval pairs were chosen by the benchmark builders and might be semantically harder.
- Net: roughly unbiased to mildly pessimistic (0 to -4 points). The real private score is additionally dominated by sampling noise on 160 cases (SE about +/- 0.03 for a per-case std of about 0.35). Do not read public-LB differences under about 0.04 as signal.

**Post-hoc selections need their own held-out check.** The decode constants (lambda, T, alpha, see below) are chosen by a coarse grid on the pooled OOF of folds; report them cross-fitted (choose on 2 folds, score on the third, rotate). If the cross-fitted gain over neutral defaults is not clearly positive on most folds, ship the defaults.

## Overfit/underfit risks

| Risk | Evidence/why here | Mitigation |
|---|---|---|
| Memorising training programs/adjacency absence | The model sees ZERO programs with the 7 withheld adjacencies; a full-history decoder learns "after head A never comes head B" as a strong prior. | Stage-factored realiser (no inter-stage history); skeleton ranker with a LOW-RANK, L2-regularised transition term and a calibrated weight lambda fit on pseudo-withheld folds (where adjacencies really are unseen). |
| Few effective independent units | Head pairs and heads are few (H of order 100); 4,107 rows heavily clustered by description templates. | Group-aware CV; small models; weight decay; dropout; EMA; fixed epochs. |
| Literal memorisation | Rare file names/patterns appear once. | Pointer-generator copy; vocabulary min-freq cut; source-token dropout restricted to tokens not copied into the target; hashed char n-grams so OOV literals get a usable embedding. |
| Tuning constants on a 3-fold OOF | About 300-450 eval rows, 3 decode knobs. | Coarse grids (3 values each), cross-fitted check, neutral default if no clear gain. |
| Public-LB overfit | 160 test rows. | Ignore public LB; trust withheld-adjacency CV. |
| Underfit: model too small/regularised | From-scratch 128-d transformer on about 7k stage examples. | Track train vs held-out metric per epoch in dev; fixed epoch count chosen from dev curves; ablate dropout 0.1/0.25/0.4. |
| Underfit: representation loses literals | Tokenizing to metric tokens could split `*.txt`. | Pointer copy works token by token; round-trip check D10. |
| Compute underfit | CPU, unknown cores. | Small architecture; profile first; fixed epochs set to what fits at 2x headroom. |
| Quote imbalance (rejects the whole file) | Open-string generation of quote characters. | Decode-time quote-state constraint; candidate filter by `shlex`; deterministic closing-quote repair; assert at the end. |

## Recommended approach (primary + fallback)

### Primary: "skeleton ranker + stage-factored pointer-generator realiser", CPU, from scratch, fold-ensemble, MBR decode

Principle: the unseen thing is an adjacency; make the model NOT condition on adjacency identity except through a small, low-rank, calibrated term, and make each stage's realisation depend only on (description, head, slot).

**Preprocessing (deterministic, train-fit only).** Metric-tokenize descriptions and outputs. Output tokens carry a "starts a new whitespace-delimited word" boolean (a second tiny output head), so the string can be rebuilt with correct spacing and quoting; the metric ignores it. Vocabulary: source words (lower-cased, min freq 2) and target tokens (min freq 3), sorted by (-freq, token) for determinism; stable hashing for char n-grams via `zlib.crc32` (never `hash()`). Stage parse as in D2; each training program yields its skeleton (list of heads) and one stage target per stage.

**Component A: skeleton ranker (few parameters, exact).**
- Encoder: embeddings (word + hashed char-trigram buckets, e.g. 2,048) -> 1-layer small transformer or BiGRU, d=128 -> pooled vector.
- Score of a head sequence `h_1..h_n`, `n in {1,2,3}`: `s(n|x) + sum_k u_k(h_k | x) + lambda * sum_k t(h_{k-1}, h_k)`, where `u_k` are slot-specific linear heads sharing a base head logit (`u_k = u_base + slot_bias_k`), and `t = phi(h_prev)^T psi(h_next)` has **rank r = 4-8** with weight decay.
- Train by exact marginal likelihood: softmax over ALL enumerable sequences (about H + H^2 + H^3; with H capped at about 60 named heads + `<UNK>` this is about 220k sequences per row, cheap in batches of 32-64 with factored tensors); exact top-K by enumeration (K = 5 at inference). Heads with fewer than 3 train occurrences collapse into `<UNK>`, which the realiser then writes itself.
- Task-statement constraint at decode: `n in {2, 3}` (every evaluation command contains an adjacency). Flagged in Open questions as low risk; measurable on `E_f`.
- Auxiliary loss (free): bag-of-heads BCE from the pooled vector (position-independent, adjacency-free evidence).

**Component B: stage realiser (pointer-generator transformer).**
- Input: description tokens, then a short control prefix: `<HEAD=h>`, slot id (1/2/3), `is_last`, `n_stages`. Optional previous-head token with dropout (ablation, see roadmap); default OFF.
- Architecture: encoder-decoder transformer d=128, 4 heads, 3 encoder + 2 decoder layers, FFN 512, dropout 0.25, learned positional embeddings, tied target embeddings; copy distribution via attention over source tokens with a `p_gen` gate; word-start flag head; the head token is forced as the first decoded token(s) (teacher-forced at train and test).
- Training examples: one per stage of every training program, with the full description as source (each program contributes n stage examples). Loss: token CE (mean per stage), weight 1.0; ablate program-normalised weights (1/n) and a x2 upweight of slot >= 2 stages, since evaluation contains only multi-stage programs.
- Fixed schedule: AdamW lr 1e-3 peak, 3-epoch warmup, cosine, wd 0.01, grad-clip 1.0, batch 64, 40 epochs, EMA 0.999 weights used for decode. Label smoothing off by default (ablate 0.05). No validation-triggered stopping.
- Decode: constrained beam (width 5) per stage: no top-level `|`, `;`, `&`, backtick, `$(`; quote-state tracked so EOS is illegal inside an open quote; max 60 tokens/stage; head forced.

**Joining and output.** Candidate program = top skeletons (K=5) x per-stage beam hypotheses (top 3 per stage, product capped at 32 candidates per row by joint score), joined with ` | `. The score is `alpha * logP_A(skeleton) + sum_k logP_B(stage_k | head_k, slot_k, x)` averaged over ensemble members (all members rescore the pooled candidates by teacher forcing). Final choice by MBR with the exact metric utility (below).

**Ensemble and refit policy.** Three fold models (each trained on `T_f`, about 85-90% of train) serve three roles at once: (i) OOF predictions on `E_f` to fit/check the decode constants, (ii) ensemble members for test, (iii) the honest CV report. No 100% refit: it cannot be validated against the withheld-adjacency proxy and costs runtime. If, in dev, a 100%-data member measurably helps on a fresh sanity fold built the same way, revisit.

### Fallback (and roadmap step 2 baseline): single whole-program pointer-generator transformer + MBR
Same tokenization, vocab, copy, constraints, MBR; source = description, target = full pipeline with `|` tokens; add "head-history dropout" (replace previous-stage head tokens with `<H>` with p = 0.5) to weaken the learned negative prior on unseen adjacencies; 3 seeds. Simple, clearly compliant, and the yardstick the primary must beat on the withheld-adjacency CV. Expected to underperform the primary on exact match because it must implicitly infer the skeleton and cannot be told which adjacency is plausible.

## Rejected options

- **Whole-command retrieval / kNN / BM25 as the solution:** explicit partial-credit control with zero exact capability; fails the strip-the-ML test; kept only as a dev floor.
- **Fine-tuning any pretrained model / pretrained embeddings or tokenizers:** banned.
- **Synthetic recombination (stitch two training stages, concatenate descriptions, create new labelled pairs for unseen adjacencies):** would likely give the largest compositional gain (GECA-style) but is "synthetic labelled data" under CLAUDE.md 2.3.6 and targets the withheld structure. Rejected; listed as a reviewer question.
- **Literal-swap augmentation (replace a literal in description and target with another train literal):** creates modified labelled pairs; borderline; rejected by default (pointer-copy teaches the same invariance without new data). Reviewer question.
- **Steering decode toward adjacencies that never occur in train (novelty feature/constraint):** exploits split construction; grey. Default OFF. A learned novelty weight fit on pseudo-withheld folds would probably help, so measure it in dev and ask the reviewer before enabling.
- **Hand-written role lists, flag tables, regex command templates, "if description says X emit Y" rules:** violates the prime directive and the strip test.
- **Large from-scratch transformer (> 10M params), char-level seq2seq:** CPU cost and tiny data; no gain expected over a small copy model.
- **Full H x H learned transition matrix:** drives never-seen pairs to -infinity by maximum likelihood, exactly the failure mode being fought.
- **Wall-clock-based stopping, dynamic beam shrinking, env-dependent worker/thread choices:** banned (determinism check).
- **Test-time adaptation, self-training/pseudo-labels, test-fit vocab, rank-normalising over the test set:** banned.
- **Time-boxed Optuna HPO:** not used; no wide HPO at all (CV cannot resolve it with 3 folds).

## Fixed work plan & runtime budget

All counts hard-coded constants; time used only in `print` logging. Device `cpu`, 4 threads, seeds `1000+fold`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `PYTHONHASHSEED=0`, `DataLoader`-free manual batching with a seeded `numpy.random.Generator`.

Estimates are UNVERIFIED guesses for 4 CPU threads; step 1 of the roadmap must profile one epoch and then fix the counts.

| Stage | Plan | Est. time (4 threads) |
|---|---|---|
| Load, tokenize, stage-parse, vocab, adjacency counts, 3 withheld-adjacency folds (union-find, 5-gram index) | deterministic | 0.5 min |
| Skeleton ranker per fold | d=128, 60 epochs, enumerative softmax | 1-1.5 min each |
| Stage realiser per fold | d=128, 3+2 layers, about 2M params, about 7k stage examples, batch 64 (about 115 steps/epoch), 40 epochs, about 70 ms/step | 5-6 min each |
| OOF decode on `E_f` (beam 5, pooled rescoring, 3 members only for test) | single member for OOF | about 1.5 min per fold |
| Test decode: 160 rows x 3 members, K=5 skeletons x 3 stage hyps, rescoring | | about 2 min |
| Decode-constant grid (lambda in {0,0.5,1}, T in {0.5,1,2}, alpha in {0.5,1,2}) with MBR over cached candidate pools | no retraining | about 1 min |
| Validation, write, re-read | | seconds |
| **Total** | | **about 25-30 min at 4 threads; up to about 50 min if the grader is 2x slower** |

Memory under 3 GB. Headroom against a 1 h ceiling about 40% at 4 threads; if profiling gives more than 30 min, cut realiser epochs to 30 or rank layers to 2+2 before touching anything else.

## Metric-aware training & decode

- **Back-solve the metric.** `score = 0.5*sim + 0.5*exact`. Train with token CE (proper scoring); the exact term is handled at decode.
- **MBR decode with the official utility.** For each test row, candidate set C (<= 32 programs) with weights `w_i = softmax_i(score_i / T)`; choose `argmax_c sum_i w_i * U(c, c_i)`, `U = 0.5*(1 - D/max(len)) + 0.5*[equal]`. Per row, no cross-row information. When the posterior is peaked this returns the mode; when it is diffuse it returns the central candidate, which is the right behaviour for partial credit.
- **Constants fit in-script on OOF only:** lambda (transition weight, 0 means no adjacency prior), T (MBR temperature), alpha (skeleton vs realiser weight). Coarse grids, report cross-fitted (rotate folds); fall back to neutral defaults (lambda=1, T=1, alpha=1) if the cross-fitted gain is not positive on most folds.
- **Hard constraints inside the valid output space:** forced head tokens; no top-level `|`/`;`/`&`/backtick/`$(`; EOS illegal inside open quotes; max lengths; every candidate must pass `shlex.split`; final deterministic closing-quote repair (counted and logged, expected 0) and an assert.
- **Calibration of probabilities:** optional per-slot temperature on the skeleton softmax, fit on OOF (one scalar), only if reliability on OOF shows over-confidence.
- **Slot prior:** upweighting slot >= 2 stage examples and `n in {2,3}` match the evaluation population stated in the description (not test statistics).

## Structural signals

Each invariant is turned into a model input, constraint or auxiliary loss, never a rule that outputs the answer.
- **Pipe structure:** program = ` | `-joined stages; pipes in quotes are not separators (parse and decode constraint); each stage is independent given (x, head, slot).
- **Head role geometry:** low-rank transition embeddings generalise from seen adjacencies to unseen ones through shared roles (source / filter / transformer / sink).
- **Data flow:** the operand-omission behaviour at slot >= 2 (D4) is learned from `slot`, `is_last`, head embeddings; every head has >= 20 training programs, so each stage's piped form is seen even though the adjacency is not.
- **Copy:** literal names and patterns in the description are part of the program (stated in the task): pointer-generator copy plus hashed char n-grams for OOV literals.
- **Quoting grammar:** balanced quotes; awk/sed braces; decode constraint plus the word-start flag so rebuilt strings tokenise identically.
- **Description-to-stage alignment:** the full description is visible to every stage; the head embedding in the control prefix lets cross-attention find the clause that corresponds to that head (learned, no explicit alignment labels).
- **Bag-of-heads auxiliary loss:** adjacency-free evidence of which commands the description needs.
- **Head set closure:** evaluated heads all appear >= 20 times in train; `<UNK>` head path exists for the tail.
- **Symmetries:** none exploitable for augmentation (descriptions are free text; pipe order is meaningful). Do not invent swap augmentations.

## Experiment roadmap

Each step has a stop criterion; keep a table (id, change, CV mean +/- std over folds x split seeds, per-pair, runtime).

1. **Contract and measurement.** Parse, round-trip, metric tests (D10), withheld-adjacency folds with 2-3 split seeds + the dev sanity fold, `validate_submission` against `test.csv`. Stop when all round-trip and unit tests pass and fold sizes look sane (at least 100 eval rows per fold set). Also run controls: kNN retrieval (expect about 0.15-0.25) and majority skeleton.
2. **Baseline end-to-end (fallback model).** Whole-program pointer-generator, greedy decode, valid CSV. Record CV. Profile one epoch to fix epoch counts.
3. **Representation/structure.** Add: stage-factored realiser with gold skeletons (oracle upper bound for B), then skeleton ranker alone (top-1/top-5 recall under withheld adjacency), then A+B joined. Variants: (a) no previous-head info [default], (b) previous head visible with dropout p in {0.5, 0.9}, (c) full history (equals fallback). Keep the simplest variant within noise of the best. Stop if the joined primary does not beat the fallback by more than one SE; then ship the fallback + MBR.
4. **Metric-aware decode.** Beam + MBR (K, T). Expected +0.02-0.04; keep only if positive on most folds. Then lambda for the low-rank transition (including lambda=0), cross-fitted.
5. **Diversity.** Add ensemble members across folds (already built in); test a second assumption family only if each alone is within about 0.03 of the best: a GRU copy-seq2seq or a skeleton-conditioned variant with previous-head visibility, blended via pooled-candidate rescoring.
6. **Bounded in-script search.** Only the decode-constant grids above, with cross-fitted evaluation. No architecture HPO in the shipped script; architecture choices (d, layers, dropout) come from a few dev ablations and must be coarse, not tuned to a decimal.
7. **Reviewer-gated options (only after answers):** novelty feature on adjacency counts; literal-swap or stage-recombination augmentation. Report their CV deltas separately so the decision is informed.
8. **Final fixed-plan run.** Clean `working/`, exact command `python3 solution.py <public_dir> <submission_out>`, run twice, diff the CSVs (expect identical; if not, find the nondeterminism before anything else), validator, runtime log versus table, compliance self-audit. Report per-fold CV (mean +/- std), the fixed plan, runtime estimate and audit.

## Compliance audit

CLAUDE.md section 7 and agent self-audits, answered for the PLAN (re-answer on the code):
- Test file read for anything beyond per-row prediction? **No.** Vocab, hashes, adjacency counts, folds, constants all from train; test rows only enter encoders/decoders one at a time. Batched padding is masked so outputs do not depend on batch composition.
- `time.time()` in any condition? **No**; logging only.
- `cuda`/`cpu_count`/import-fallback/try-except changing work? **No**; fixed `cpu`, 4 threads.
- Hardcoded tuned constants? Only structural choices (d, layers, epochs, grid values); decode constants found by in-script cross-fitted search. Nothing pasted from earlier submissions.
- External data/synthetic data/self-hosted weights/non-allowed library? **No.** The stage parse yields targets from real labels (not new data); literal swap and recombination are excluded pending a reviewer answer.
- Strip-the-ML test: remove A and B -> no output at all (no rules, no templates, no lookup). The parse and constraint code only structure and validate. **Pass.** Retrieval control is dev-only and not in `solution.py`.
- Model-heavy part dominates? Yes: trained skeleton ranker + trained realiser + learned decode scores; MBR is a decision rule over model posteriors.
- Challenge restrictions honoured: from scratch, CPU, no pretrained anything, no ids as features, no source lookup. **Yes.**
- Source readable, < 512 KB, no blobs; comments explain reasoning; seeds fixed; threads fixed. Planned yes.
- Sibling leakage: validation groups by canonical output and 5-word spans; stage targets share a description across stages of the same program but programs are never split across train/eval.
- Held-out adjacency list never inferred as a model input; the optional novelty feature is OFF by default.
- Remaining risks to re-audit on the code: `n in {2,3}` constraint; stage-factored preprocessing; quote repair path being deterministic.

## Open questions & assumptions

**Reviewer questions (plan under each reading):**
1. Is stage-factored training (each real program -> one real training example per stage, parse by a deterministic shell lexer) acceptable preprocessing rather than "synthetic data" or a "rule-based" component? Assumed YES. If NO: use the fallback whole-program model (head-history dropout + MBR), expect a lower exact rate.
2. Is stage-level or literal-swap recombination allowed as augmentation? Assumed NO (not used).
3. May the decoder use a feature or constraint that favours head adjacencies never observed in training (exploiting how the split was made)? Assumed NO (OFF); would likely help; the plan measures it on pseudo-withheld folds anyway.
4. Is restricting the output to 2-3 stages (since every evaluation program contains an adjacency) acceptable? Assumed yes (task statement, not test statistics); negligible risk, but ask.
5. Runtime limit and core count are not stated. Assumed <= 1 h, 4 threads; ask for the real number.
6. Exact grader tokenization regex (underscore and digits inside word runs, Unicode classes). Assumed `\w+|[^\w\s]`; the model barely depends on it.
7. Does `sample_submission.csv` hold all 160 test ids? Assumed to check at run time; the validator uses `test.csv` ids.

**Assumptions (silent in the text):** torch/numpy available on the CPU grader; no internet used; 4 CPU threads; the "from scratch" rule also forbids any pretrained tokenizer (own tokenization only); training-time code generated nothing at runtime.

**What could not be verified (no data):** every diagnostic in "Data findings"; stage-count and head distributions; whether withheld-pair frequencies allow the 25-150 pseudo-pair rule; copy rate; runtime figures (all are estimates until profiled); the expected score.

**Expected score (estimate, not a promise).** Retrieval control about 0.15-0.25; fallback whole-program pointer-generator about 0.30-0.40; primary (skeleton + stage realiser + MBR) about 0.35-0.50 with exact-match about 15-30%. Reasoning: partial-credit similarity about 0.6-0.7 plus a minority of exact hits, limited by quoting/flag ambiguity. Real private score could land +/- 0.03 from sampling noise on 160 cases alone.
