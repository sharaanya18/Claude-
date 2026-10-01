# Validation recipes (the private leaderboard is decided here)

CV is the source of truth; the public LB is a weak, noisy sanity check. A split mismatch is the #1 cause of private
drops. Every item has a bias direction: state it ("this proxy is optimistic/pessimistic because ...").

## V1. Reconstruct the hidden split from the description
Read for the split rule: "project-/publisher-/speaker-/site-/camera-disjoint", "unseen characters/classes",
"later time period", "held-out workflow families", "distinct warehouses", "no source interval crosses the split".
Mirror it with the same unit and, where stated, the same size per held-out unit. If the statement is silent, assume the
strictest plausible reading (group by the largest hidden unit you can derive) and say so. Test-pool size matters:
folds of similar size to the test pool give comparable metric noise (top-k metrics on 500 rows are very noisy).

## V2. Derive groups yourself (floor, not ceiling)
Union-find over every detectable relation, on train only (`.claude/scripts/make_groups.py`): shared keys, exact and
near-duplicate text (char-5-gram MinHash/Jaccard), sliding-window overlap chains, paired views of one object,
augmented copies, same-source/same-recording, near-identical embeddings (including flips), sibling rows from one case.
Handle templated text deliberately (everything looks like a near-duplicate; raise the threshold). Cap or inspect a giant
group (>5–10% of rows). If ids are absent (route, project, institution, environment), cluster by content (background
embedding, stop-set overlap, vocabulary style) and calibrate the clustering granularity against any statistics the
description publishes (coverage rates, split sizes). Assume **sibling leakage** whenever rows look ordered or
related; "siblings" is the reviewers' shorthand for related-row leakage (S1/S2 in rejections).

## V3. Nested / cross-fit every selection step
Hyperparameters, backbone choice, blend weights, decode constants, thresholds, temperature, "keep vs drop" feature
blocks: each needs a held-out check that did not participate in the selection. Cheap pattern: split validation
components in two halves, fit the post-hoc step on half A's OOF and score on half B (swap, average); print it beside
the in-sample number. A recorded 0.328 ensemble vs 0.288 best single was exactly selection optimism on one OOF.

## V4. Repeats, folds, and the noise rule
≥ 5 folds; 2–3 split seeds when data is small or the metric noisy; report mean ± std and per-fold. Accept a change
only if the paired gain exceeds ~1 SE of the paired fold differences or is consistent across most folds and split
seeds. Compare candidates on identical splits, never across split seeds. Prefer smooth plateaus over sharp optima.
Keep one sanity holdout (≈15–20% of groups) that touches no selection until the end.

## V5. State the bias direction of every proxy
Clusters coarser than the real units → pessimistic; finer → optimistic. Row-level CV on grouped data → optimistic.
Interpolation validated while extrapolation is required → optimistic (B7). Self-built synthetic distractors → can be
off by 0.05 (B5). A proxy that matches the wrong axis is leak-free yet deployment-wrong: check the *magnitude* of the
separation (time gap, novelty, shift), not just that a group split exists.

## V6. Fold balance for rare labels
Macro-over-labels with < ~30 positive groups for some label: place whole groups greedily, rarest first, minimising
squared per-label positive load + row load; guarantee both classes of every label in every fold (B14, `make_groups.py
--label`). Fewer folds beat folds with empty classes.

## V7. Rehearse the shift
When the brief hints the private population differs in kind: cluster whole training rows into pseudo-unseen
populations (KMeans on standardised input/history features; assign whole clusters to folds, B12), or hold out whole
categories (language direction, workflow family, publisher). Calibrate the decoder in the *target regime* (transport
held-out data to the evaluation scale/sensor, B10), then ship the model that was calibrated.

## V8. Label-derived statistics without leaks
Target encodings, co-occurrence counts, answer rates, nearest-neighbour label transfer: out-of-fold for training rows
with repeated folds; no leave-one-out (a positive's count is exactly one lower than the same pair elsewhere, which
trees detect); use **same-size statistics** for train and test (matched amount of data behind both); exclude the
case's own group from its neighbours; validate with statistics fitted exactly as they will be for test. Fold held-out
labels of the same kind as the inputs back into unsupervised statistics, cross-fitted (C20).

## V9. Final fit
Refit on 100% of train with counts fixed from CV (e.g. mean best epoch × 1.1); do not ship fold-averaged ~80%
models unless the data are plentiful (recorded loss from dropping 20% of sites: +0.007 to +0.04). Average a few
seeds/snapshots (EMA, late checkpoints) as part of the fixed plan. Never choose steps by the clock.

## V10. Ceiling and slice diagnostics (before and after modelling)
Report overall and per slice (class, group size, length, rarity, domain, dark/normal, protocol): the weakest slice
often dominates a geometric-mean or macro metric. Duplicate-input/different-label rate = irreducible ambiguity.
Candidate-pool oracle recall = coverage ceiling. Shuffled-input and metadata-only controls expose shortcuts.

## V11. Compare against yardsticks
Frozen-probe score (cheap, robust) as the yardstick for any fine-tune; trivial constant under the exact metric; the
description's published reference numbers (reproduce them on train to prove the metric and split are right).

## V12. Leakage tells to hunt
Near-perfect CV on a noisy task; CV ≫ public LB; ids/row order/file size/position predictive; train rows from the same
source on both sides; features computed from "bank" entities that include the case itself (self-match exclusion, I1);
columns known only after the outcome (drop with the reasoning written down, I3); a gain that vanishes under grouped
folds.

## V13. Ambiguous instructions
When the text is ambiguous about what may be fit in the final model (e.g. "validation set is for model selection"
vs "tune on train and validation"), choose the reading compliant under every interpretation, state the two
sentences in a comment, and quantify the cost of the conservative choice.

## V14. Reporting block (put it in the plan and in the final answer)
Split design + bias direction; folds × repeats; metric mean ± std and per-fold; per-slice numbers; selection steps with
their nested check; sanity-holdout score; estimated runtime from profiling; expected private band as an estimate.
