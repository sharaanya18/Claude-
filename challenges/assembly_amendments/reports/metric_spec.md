# Metric spec — assembly_amendments

## Formula (as implemented in `metric.py`)

Per board (`n` amendments, true counts `[a,r,f,w,m]` summing to `n`):

- **carried** (w=0.35): `(earned - expected) / (1 - expected)` where `earned` = share of
  the `n` items where `(pred fate == adopted) == (true fate == adopted)`, and
  `expected = (a² + (n-a)²) / n²`.
- **disposal** (w=0.35): same formula restricted to the `n-a` true-non-adopted items
  (placing one of them as "adopted" counts as a miss); `expected = (r²+f²+w²+m²) / (n·(n-a))`.
- **joint** (w=0.30): `sklearn.metrics.adjusted_rand_score(true_partition, pred_partition)`.

**Derivation of `expected`** (not assumed — derived and brute-force-verified, see
`tests_metric.py::TestCarriedDisposalFormula`): the "random settlement that respects the
counts" is a uniformly random permutation of the *full* true 5-way fate multiset over the
`n` positions. By linearity of expectation, for a fixed position whose true fate class has
size `c` (out of `a,r,f,w,m`), the marginal probability the random settlement reproduces
that fate at that position is `c/n` (standard marginal of sampling-without-replacement,
true regardless of correlation between positions). Summing over positions gives the two
closed forms above. We exhaustively enumerated all distinct permutations of the true
multiset for several small boards (`n≤5`) and the closed forms matched to float precision
in every case (`test_formula_matches_brute_force`). We also checked numerically that a
"restrict-then-permute-within-the-non-adopted-subset-only" alternative model (treating the
counts/non-adopted split itself as fixed, not part of the random settlement) does **not**
match the brute-force enumeration for `disposal` — confirming the full-multiset permutation
is the right model, not an alternative plausible-sounding one.

## Invalid outputs

- **fates**: wrong length, unparseable JSON/not-a-list, a non-integer-code entry, or any
  code used a different number of times than `counts` → **carried = disposal = -1**
  (forced, unconditional — see judgment call #1 below).
- **joint**: wrong length or unparseable → **joint = 0** (not -1 — explicit asymmetry
  verbatim in the description, preserved in `metric.py::joint_term`).

## Term table

| Term | Weight | Rewards | Excluded (not just 0) when | Dominant sub-structure |
|---|---|---|---|---|
| carried | 0.35 | correct adopted/not split, chance-corrected vs the board's own `a`/`n-a` split | `a == 0` (no adopted item) | boards with extreme `a` (near 0 or near n) have `expected`→1, so a tiny number of errors swings the term hard (small `1-expected` denominator) |
| disposal | 0.35 | correct 4-way fate among non-adopted items, chance-corrected vs `[r,f,w,m]` | `a == n` (`n-a == 0`, nothing to score) | boards where one of r/f/w/m dominates the non-adopted subset are similarly denominator-sensitive |
| joint | 0.30 | partition recovery via ARI (confirmed: all-singleton pred → 0, all-one-group pred → 0 unless truth is all-one-group, both verified numerically against `sklearn`) | true partition is all-singleton (227/622 test boards) | ARI is itself chance-corrected (no separate "expected" needed); boards with many small joint groups weight pairwise structure more than boards with one dominant group |

Board score = weighted mean (stated weights, **renormalized over only the terms defined on
that board**) of each defined term, each floored at -1 first. Final score =
`100 * mean_over_boards(board_score)`, floored at 0 once at the very end.

**Saturation / invertibility**: `carried`/`disposal` are linear rescalings of hit-rate, so
they're invertible (any hit-rate maps to a unique term value) except at the degenerate
`expected==1` edge case below. `joint`/ARI saturates at 1.0 for any partition isomorphic to
truth (any relabeling), and is unbounded below in principle but in practice rarely goes
far below 0 for realistic model outputs.

## Degenerate `expected == 1` edge case (important, common in train, absent in test)

`1 - expected == 0` happens for **carried** iff `a==0` or `a==n`, and for **disposal** iff
`a==0` **and** every non-adopted item shares one single true fate. In both cases the
counts-validity constraint on any *valid* submission leaves exactly one possible
multiset-respecting arrangement, so `earned` is **forced to 1** for every valid submission
regardless of prediction quality — not a real 0/0, just an uninformative free point. We
implemented this as: if `1-expected ≈ 0`, return `1.0` (since `earned` is provably forced to
1 there), with a defensive `-1.0` fallback branch that should be unreachable given the data's
constraints (documented, tested in `TestMalformedAndExclusions`).

**This is empirically common in train (not test):** `a==n` occurs in 915/4392 train boards,
`a==0`-with-single-non-adopted-fate in 1071/4392 — **0 of either in the 622 test boards**,
because the description guarantees test boards have "at least two different fates" (which
makes both degenerate configurations impossible there — each needs all `n` items to share
one single fate). **Implication for the strategist:** a naive CV computed on raw, unfiltered
train boards will be inflated relative to test, because ~2,000/4,392 (~45%) of train boards
hand out a free, zero-information 1.0 on `carried` or `disposal` that could never happen on
test. Our oracle test (`test_counts_respecting_random_shuffle_baseline_near_zero`) measured
this directly: a random-shuffle (counts-respecting, otherwise uninformative) baseline scores
mean board_score ≈ **0.46** on raw train vs. ≈ **0.017** on the subset of train boards with
≥2 distinct true fates (test-like). **Recommendation: when estimating CV, either filter
train boards to those with ≥2 distinct true fates to mirror test, or at minimum report CV
separately on the filtered vs. unfiltered population**, and do not compare a raw-train CV
number directly to the expected private-LB scale.

## Loss / decode design (brief — full version belongs in
`.claude/skills/eris-playbook/references/metric-and-decoding.md`)

- **carried** → train a per-item adopted-vs-not classifier (features: own item text +
  board-level pooled context), decode by taking the top-`a` items by predicted
  adopted-probability within each board (this **is** metric-aware decoding, but it is
  decoding to respect a *known, given* structural constraint — the board's `counts` — not
  decoding from leaked test statistics; the challenge does not ban using the board's own
  published `counts` column, which is a train/test input feature, not an answer).
- **disposal** → conditional on not-adopted, a 4-way classifier restricted to the predicted
  (or a second counts-respecting-decode step forcing) non-adopted subset, matched to
  `[r,f,w,m]` via a constrained assignment (e.g. Hungarian / greedy-by-margin to hit exact
  counts) rather than free per-item argmax, since free argmax will not respect counts and
  would then be scored as **invalid** (`fates` multiplicity mismatch → -1/-1). **No banned
  metric-aware decoding here** — forcing count-matching is required by the *submission
  format itself* ("each code must be used exactly as many times as counts say"), not an
  exploit of the scoring rule.
- **joint** → pairwise same-group classifier over item pairs within a board → cluster into a
  partition (e.g. connected components over a thresholded/learned pairwise affinity, or
  agglomerative clustering on learned embeddings) — training loss should be a pairwise
  BCE/contrastive loss on the "same joint discussion" label, not a fixed-K clustering loss
  (true partition's number of groups varies per board and is not predicted directly).

## Oracle decode check

`tests_metric.py::TestOracleDecodeOnRealData` confirms on a 400-board real-train sample:
- ground truth fed back as the prediction scores `score_submission ≈ 100.0` (the ceiling).
- a counts-ignoring global-majority-fate baseline (every item gets the single globally most
  common fate code, ignoring the board's own `counts`) is invalid on 77% of boards (mostly
  larger ones; small `n==1` boards slip through by coincidence) and the submission-level
  score hits the documented floor, **exactly 0.0**.
- a counts-respecting random shuffle of the true fates/joint per board (the correct "random
  settlement" construction) averages `board_score ≈ 0.017` on the test-like (≥2 distinct
  true fates) subset of train — i.e. ≈0, confirming the chance-correction is implemented
  correctly — vs. `≈0.46` on the raw, unfiltered subset (the degenerate-board inflation
  documented above).

## Judgment calls / open questions for a reviewer

1. **Weight renormalization** (explicitly asked about in the task): the description says
   "the weighted mean of the terms defined on it" without saying whether the stated weights
   (0.35/0.35/0.30) are renormalized to sum to 1 over just the defined terms, or whether
   undefined terms are dropped from both numerator *and* denominator of a fixed-weight
   average (which is mathematically the same renormalization), or some other convention
   (e.g. undefined counts as a hidden zero in the denominator only, which would be unusual
   and is not implemented). **We implemented renormalize-by-sum-of-defined-weights** (the
   two readings that seemed plausible turn out to be the same operation, so there is really
   only one reasonable reading here; flagging only because the prose never says "renormalize"
   explicitly). **Implication**: a board missing one term is evaluated purely on the
   remaining term(s)'s own scale (e.g. a board with only `joint` defined gets
   `board_score = joint_term`, not `0.30 * joint_term`) — this is what we implemented and
   tested (`test_a_zero_excludes_carried`, `test_a_equals_n_excludes_disposal`,
   `test_singleton_only_joint_excludes_joint_term`).
2. **Malformed-fates interacts with a==0/a==n exclusions**: does "this term and the next are
   -1" on malformed fates apply unconditionally, or does the a==0/a==n exclusion still take
   priority (i.e., if `a==0` and fates are also malformed, is `carried` excluded, or forced to
   -1)? We read the malformed-fates sentence as a hard, unconditional override (it is stated
   first, as its own independent clause, before the "Otherwise..." paragraph that contains the
   a==0 exclusion) and implemented **malformed ⇒ -1/-1 always, regardless of a==0/a==n**. This
   is a genuine prose ambiguity — flag to a reviewer. Practical impact is likely small (a
   model that already respects counts validity will never trigger this path), but it does
   affect how a buggy/edge-case decode gets penalized at the margins.
3. **"uses any code a different number of times than counts say"** — we implemented this as
   an exact multiset-equality check against the full `[a,r,f,w,m]` vector (every one of the 5
   codes' counts must match exactly), not merely "the codes used don't exceed availability" or
   a softer check. This reading is unambiguous from the prose and is the one implemented.
4. **Degenerate `expected==1` edge case** (above) — not mentioned in the prose at all; we
   derived the only mathematically consistent resolution (forced `earned==1` ⇒ term `1.0`)
   and verified it can only ever be hit by a *valid* submission in the direction that makes it
   harmless (never saw a counter-example in real or synthetic testing).

## Files

- `/home/user/Claude-/challenges/assembly_amendments/metric.py` — implementation.
- `/home/user/Claude-/challenges/assembly_amendments/tests_metric.py` — 17 unit tests, all passing
  (brute-force formula check, perfect/worst/reversed cases, Monte-Carlo chance baselines,
  malformed-input handling, all three exclusion rules, hand-worked n=4 example, oracle checks
  on real train data). Run: `python3 tests_metric.py`.
