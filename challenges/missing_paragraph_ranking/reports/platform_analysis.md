# Phase 0 — Platform analysis (Project Eris / Shipd)

Sources inspected: `Project Eris — Solver Guidebook` (updated, 15 pp., supplied this session),
the solver workspace repository (`CLAUDE.md`, `.claude/skills/eris-playbook/**`,
`.claude/scripts/**`, six prior challenge workspaces), and the challenge description itself.

> **Note on supplied material.** The brief promised (1) a GitHub repository for the
> competition/platform and (2) a reference approach that took #1 on the private leaderboard.
> Only the guidebook PDF and `public.zip` arrived. The platform analysis below is therefore
> built from the guidebook plus the distilled pattern library already in this repository
> (which encodes 13 earlier challenges and 60+ leaderboard solutions). **`private_lb_reference_analysis.md`
> is not written** — there is no reference solution to analyse. Send it and I will do Phase 5.

## 1. Platform structure

| Mechanism | Fact |
|---|---|
| Entry contract | `python3 solution.py <public_dir> <submission_out>`; both from `sys.argv`. |
| Execution | One self-contained script, run end to end on an **NVIDIA A10G**, fresh from raw data. No cached artefacts, no attached files, no previous-run outputs. |
| Runtime | Expected max **1.5 h**, up to +30 min grace for a genuinely good solution. Plan ≤ 50 min. |
| Libraries | Exactly the Kaggle Docker image. Nothing else, no approval route, no `pip install`. |
| Internet | Only for pretrained backbone weights (HF/timm) — **irrelevant here, this challenge bans pretrained weights entirely**. |
| Submission | A CSV is uploaded *and* the script; only the script's own run counts for the leaderboard. |
| Credits | 6 per challenge (+1/4 h), global ~15–25/day. **The script run always burns the credit**, even on a silent bug. There is no free score probe. |
| Scoring | Public LB visible during the challenge; **private LB decides prizes** and is revealed only at the end. |
| Review | Every solution is reviewed *after* the competition. Review is **relative**: solutions are rated for appropriateness and compared rank-by-rank; a rank-1 solution much less appropriate than rank 2 is removed. |

### The four pre-submission checks
1. **CSV grading** — the uploaded CSV must grade without error.
2. **Prompt compliance** — an LLM reads the script against the challenge overview and every rule in it.
3. **Determinism** — an LLM reads the script for unseeded randomness, clock-conditioned control
   flow, and machine-conditioned branches. This is a *code review*, not a re-run.
4. **Script run** — the real end-to-end execution. Only this one consumes a credit.

Failures 1–3 refund the credit (5-minute cooldown). Failure 4 never does. A malformed or
incomplete `submission.csv` scores **below zero** (dead last) and still costs the credit.

### What the determinism check rejects (verbatim from the guidebook)
Missing seeds; runtime-conditioned code (time budgets, "stop after N minutes", epochs chosen
from how fast the first one ran); test-environment-conditioned code (`os.cpu_count()`,
`cuda if available`, batch size from free VRAM); `try/except` fallbacks that switch method;
unused switches and branches that *could* choose between methods. Explicitly fine: in-script
HPO with a fixed trial count, early stopping on a validation score, AMP, a seeded DataLoader
with a fixed worker count.

**This directly contradicts the older guidebook's "stop training at ~3000 s" advice.** The
checker wins: no elapsed-time branch of any kind. Time may be printed, never compared.

## 2. Relevant prior solutions and transferable principles

From `.claude/skills/eris-playbook/` (distilled, not copied) the items that bear on a ranking
challenge of this shape:

- **Generate → score → decode as three separable systems**, diagnosed separately (coverage /
  ranking / decode). Here coverage is trivially 100 % — the answer is always in the pool of 20 —
  so *all* the work is in ranking. Any effort spent on candidate generation is wasted.
- **Put the metric in the loss.** The metric is mean normalised rank over a fixed pool of 20, so
  the natural loss is listwise softmax cross-entropy over the 20 candidates, not pairwise
  binary classification. Every step of rank improvement counts equally (unlike MRR), so the
  loss should care about the whole ordering, not only the top.
- **Mirror the hidden split, then be paranoid.** The split holds out *abstracts*; evaluation
  pools are built only from evaluation abstracts. Validation pools must therefore be rebuilt
  from held-out abstracts only — using the supplied training pools as validation pools would
  mix seen and unseen text and be optimistic.
- **Nested/cross-fit every selection step** (V3): hyperparameters, blend weights, epoch counts.
- **Representation first, machinery second** (6× confirmed): the single biggest lever is almost
  always a better representation, not a bigger model. Confirmed again here — spaced-seed
  n-grams beat every architectural change tried.
- **Anti-pattern, 6× recorded:** a complex architecture ranking below the simple winner.
- **L003/L011 (AnchorPerm):** when the leaders' big lever is pooling information across
  evaluation rows, that lever is banned; compete with row-local methods and accept the ceiling.
  Directly relevant: this challenge *explicitly* bans reasoning across evaluation queries.

## 3. Transferable ranking techniques, screened against this challenge's rules

| Technique | Verdict here |
|---|---|
| Listwise softmax over the slate | **Core.** Matches the metric and the fixed pool of 20. |
| Pairwise logistic / hinge / BPR | Usable, strictly weaker than listwise given a complete pool. |
| Hard-negative mining from an out-of-fold scorer | Allowed (train-side only). Must never see validation/eval rows. |
| In-batch negatives | Allowed, and should be **same-subfield** to match the real pool. |
| Cross-encoder rerank | Allowed if trained from scratch. |
| Bi-encoder / dual tower | Allowed if trained from scratch. |
| Pool-relative features (rank, z-score, margin) | **Allowed** — the pool is part of one query's input. Not cross-query. |
| LambdaMART / LightGBM ranker on hand features | **Grey, and guidebook §6.3 calls it out by name**: "feature engineering plus an off-the-shelf ranking algorithm isn't enough". Not as the primary. |
| TF-IDF / BM25 similarity | Permitted **as a model feature only**; may not decide the ranking. |
| Pretrained encoders / embeddings / spell-checkers | **Banned outright** (from-scratch challenge). |
| Pseudo-labelling, test-time adaptation, SSL on eval text | **Banned** (both the challenge text and guidebook §5.2.5). |
| Candidate frequency / ids / row order / pool order | **Banned explicitly**, and measured to be at chance anyway (0.5032–0.5073). |
| Assigning each candidate to at most one query | **Banned** (cross-query reasoning), though the data would permit it. |

## 4. Leakage mechanisms specific to this challenge

1. **Cross-query matching.** 412 queries draw from 578 candidates; a global assignment would
   measurably help. Explicitly forbidden — each query must be scored from its own pool alone.
2. **Candidate frequency.** In training every candidate appears in exactly 20 pools; in
   evaluation 13–15 times, deliberately decorrelated from being an answer. Banned and useless.
3. **Self-supervision on evaluation text.** Tempting (578 extra snippets) and banned.
4. **Fitting any statistic on evaluation text** — idf, vocabulary, buckets, normalisers. All
   must be fitted on training text and only `transform`ed onto evaluation text.
5. **Validation leakage through pools.** Using the supplied training pool for a held-out query
   puts training abstracts in a validation pool. Rebuild pools from held-out abstracts only.
6. **Validation leakage through idf.** Statistics fitted on the whole training set make
   training pairs look different from held-out pairs; any model calibrated on that mismatch
   transfers badly. Fixed here with an inner-train/inner-val split.

## 5. Rules that matter most for this challenge

- From-scratch: **every weight randomly initialised**, trained only on `public/`.
- No outside text of any kind, including dictionaries and word lists.
- Overlap scores may be features; **the trained model must decide the ranking**, and a solution
  that keeps most of its score with the model removed is rejected as rule-based.
- No cross-query reasoning, no id/frequency/order signals, no hand-edited or hardcoded rankings.
- Determinism: fixed epochs/folds/seeds, no clock, no hardware branches.

## 6. Recommended research direction (what the analysis implies)

1. The pool is complete, so **ranking is the whole task** — spend everything on the scorer.
2. The metric is pool-normalised rank, so train **listwise over the 20**, and build validation
   pools exactly like evaluation pools.
3. Corruption rises from 20 % to 30 % between training and evaluation. The corruption process
   is published, so **re-corrupting training text toward and past 30 %** is legitimate
   augmentation and directly rehearses the shift. Select models at the 30 % level, not 20 %.
4. Representation is the dominant lever. Contiguous n-grams are the wrong unit under letter
   noise; **spaced seeds** (gapped n-grams) are the principled fix and were confirmed to help.
5. Build the learned model so that the lexical evidence enters as a *decomposed profile* the
   model weighs, never as a single pre-computed similarity that the model only rescales.
