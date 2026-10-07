# Phase 1 — Dataset audit

All figures measured from `public/` (`dev/audit.py`, `dev/structure.py`, `dev/diag_*.py`).
Test text is summarised only for shape and charset checks; no evaluation statistic enters the
pipeline.

## 1. Shapes and identifiers — all clean

| File | Rows | Notes |
|---|---|---|
| `train.csv` | 2,440 | `query_id`, `snippet_a`, `candidates` (20 ids) |
| `test.csv` | 412 | same three columns |
| `candidates.csv` | 3,018 | `candidate_id`, `snippet_b` |
| `train_labels.csv` | 2,440 | one target per training query |
| `sample_submission.csv` | 412 | ranking = the pool in its given order |

- Query ids unique in both splits; candidate ids unique; labels cover every training query.
- Every pool has exactly 20 **distinct** ids. No duplicates anywhere.
- Every training target lies inside its own pool.
- `sample_submission` ids match `test.csv` **in the same order**, and each ranking is exactly
  that query's pool in the given order (so it scores at chance, 0.5037 as published).
- **Train and evaluation candidate sets are disjoint**: 2,440 train-pool candidates,
  578 evaluation-pool candidates, intersection 0, union = all 3,018 rows. The abstract-level
  hold-out is real and complete.

## 2. Pool design and the recovered subfield structure

- Training: each of the 2,440 candidates appears in **exactly 20** pools; every training
  abstract is both a query and a candidate, and `train_labels.target_id` is a bijection onto
  the training candidate set.
- Evaluation: 578 candidates over 412 queries, each appearing **13–15** times (mean 14.26).
  So 166 evaluation abstracts are distractors only and never an answer.
- Union-find over "appeared in the same training pool" returns **exactly 6 components**
  (595 / 430 / 427 / 369 / 367 / 252), every pool lies inside one component, and each query's
  subfield equals its pool's. These are the six research subfields. The same construction on
  the evaluation pools also returns exactly 6 components (137 / 124 / 93 / 91 / 68 / 65).
- **Use made of this:** training-side subfields drive (a) honest validation pools — negatives
  drawn only from held-out abstracts of the query's own subfield — and (b) same-subfield
  in-batch negatives during training. The recovery is done on training pools only and no
  subfield label is ever a prediction feature.

## 3. Text characteristics

- Every snippet is **exactly 100 characters**, train and test, query and candidate. No empties.
- 65 distinct characters; **no character appears in test that is absent from train**, so a
  train-fitted character vocabulary covers evaluation completely.
- The corpus is entirely **lowercase** (0 uppercase characters), pre-tokenised arXiv-style:
  punctuation is spaced out, 17.2 % of characters are spaces, 0.77 % digits, 2.2 % other
  punctuation, 79.8 % lowercase letters. ~17 tokens per snippet, mean token length 4.92.
- 23 % of snippets contain a digit. Structural markers (`*`, `$`, `\`) are rare (< 1 %).
- One exact duplicate snippet across all 5,458 texts; otherwise no duplicate text, no
  duplicate query text, no near-duplicate families.
- Train and test char-class profiles are indistinguishable (lower 0.7984 vs 0.7968,
  space 0.1724 vs 0.1735, digit 0.0077 vs 0.0080): no formatting shift, only a noise shift.

## 4. The corruption model, measured

Letters are replaced by a uniform random letter of the same case; spaces, digits and
punctuation are untouched. Under `observed = (1-p)·clean + p·uniform`, the total-variation
distance of the letter histogram from uniform scales as `(1-p)`:

- TV(train, uniform) = 0.2793, TV(test, uniform) = 0.2453, ratio **0.8781**.
- With p_train = 0.20 this gives **p_test = 0.2975**, matching the stated 20 % / 30 % exactly.
- The ratio also discriminates the replacement model: uniform over all 26 letters predicts
  0.875, uniform over the other 25 predicts 0.869. The data favour **uniform over all 26**.

This measurement is a *confirmation* of the published rates, not an input: the pipeline uses
the description's 20 % / 30 % only. Composing corruption gives the augmentation constant
`q = 1 − (1−0.30)/(1−0.20) = 0.125` to carry a training snippet to the evaluation noise level.

## 5. Where the signal actually is (fold-0 diagnostics at 30 % noise)

AUC = true continuation vs its 19 same-subfield decoys.

| Signal | AUC | Reading |
|---|---|---|
| char 2-grams, tf-idf cosine | 0.6042 | |
| char 3-grams | 0.6226 | best single contiguous order |
| char 4-grams | 0.5923 | |
| char 5-grams | 0.5484 |long n-grams are destroyed by noise |
| char 2–5 combined | 0.6343 | the published 0.3387 reference |
| ≥1 shared long token (len ≥ 6, ≥ 60 % char agreement) | 0.5501 | **only 11.6 % of true pairs have one** |

Two conclusions, both decisive for the design:

1. **The link is topical, not lexical.** The description says openings and contributions
   "rarely repeat the same words" and the data agree: literal word reuse has 12 % recall. A
   crude hand-built fuzzy token matcher scored 0.354 — no better than TF-IDF. There is no
   shortcut rule to find.
2. **Low-order n-grams dominate**, and the discriminative information is **high-dimensional and
   diffuse**. The per-pair cosine separates true from decoy by only 0.036 vs 0.026, with the
   top-5 shared n-grams carrying 44 % of the cosine in *both* classes — no smoking gun.

The second point was confirmed destructively: every low-dimensional bottleneck **loses** to the
raw sparse cosine — LSA at k=128/256/512 gives 0.377–0.406, supervised low-rank CCA 0.426–0.455,
a 256-d random sketch 0.441, all against 0.364 for the raw cosine on the same fold. The top
components capture subfield-level vocabulary, which is exactly the part that cannot
discriminate when all 20 candidates share the subfield.

## 6. Difficulty and distribution checks

- Train vs test: identical lengths, charset, char-class mix, token statistics; the **only**
  distribution shift is the corruption rate (20 % → 30 %), and it is large — the same TF-IDF
  scores 0.2994 at 20 % and 0.3451 at 30 % on identical folds.
- Candidate difficulty is uniform by construction: every decoy is a same-subfield abstract, and
  decoys are drawn without reference to lexical similarity (pool membership is a balanced
  design, 20 appearances each in training).
- Nothing exploitable in ids, order or frequency: the published chance scores for
  frequency-ordered (0.5073 / 0.5032) and id-ordered (0.5042) rankings confirm it, and these
  signals are banned anyway.

## 7. Implications carried into the design

1. Validation must rebuild pools from held-out abstracts of the same subfield, and must be read
   at **30 %** noise. The harness does both and reproduces the published TF-IDF reference
   (0.3490 ± 0.009 vs 0.3387 published) — a direct calibration of the proxy against the real
   evaluation set.
2. Representation must stay **high-dimensional and sparse**; no projection bottleneck.
3. Robustness to corruption is the main transferable lever, because the evaluation regime is
   noisier than the training regime by a margin worth ~0.05 of score.
