# S-tier principles (read first, every challenge)

Distilled from the shipd_env pattern library (13 challenges, 60+ leaderboard solutions, 6 rejection
post-mortems) and from planning 25 unseen challenges. Items marked **[N×]** converged independently in N unrelated
solutions; weight them above single-sighting tricks. Everything here must stay inside CLAUDE.md §2–§3; where a
pattern in the source library conflicts with the Deterministic Execution check it is marked **DO NOT ADOPT**.

## 1. The seven moves that separate top-5 from the middle (in order of expected lift)

1. **Name the decision unit and the valid output space before any EDA.** Most Eris tasks are structured decision
   problems: learn calibrated local evidence, then choose a globally valid output object that maximises expected
   utility under the real metric: `y* = argmax_{y valid} E_{p(.|x)}[metric(y, .)]`. Write down (a) what one valid
   answer looks like, (b) what makes an answer invalid (scores below zero) rather than low, (c) what each metric term
   rewards. Often a "generation" task is really "rank a small enumerable set", and "k-way classification" is really
   "regress k metric-weighted targets". **[5×]**
2. **Put the metric into the training loss and the decode.** Back-solve invertible metrics into richer targets,
   weight losses with the metric's own (hierarchical) weights, decode by expected utility, derive closed-form
   thresholds, and verify with an oracle decode (gold scores must decode to ~100%). Every examined solution did
   this. See `metric-and-decoding.md`. **[all]**
3. **Turn every structural fact of the data-generating process into signal**: symmetry (any station can be the
   target), invariants (exactly 2 witnesses; first stop fixed; each role appears once), hidden taxonomies recovered
   from train labels, couplings between outputs (a confident output X forces output Y: the single largest recorded
   lift, +0.14), latent permutations marginalised exactly when small. Hard verified constraints go into the legal
   output space at decode time, not left for the model to learn. **[6×]**
4. **Mirror the hidden split in validation, then be paranoid about leakage.** Derive groups by union-find over every
   detectable relation (shared keys, exact/near duplicates, sliding-window overlap, paired views, siblings). Fold by
   whole groups. Validate where you will be tested (transport the held-out data to the target regime). Nested /
   cross-fit every selection step (hyperparameters, blend weights, decode constants). **[universal]**
5. **Representation first, machinery second.** Winners' first lever was almost always a stronger or larger
   pretrained representation at the resolution/length that matters (DINOv2-L@448 over base@224; large encoders over
   base), used at the cheapest adequate capacity rung: frozen probe → LP-FT of the last blocks → full fine-tune. In
   six problems the *simplest* solution won while more machinery ranked in the bottom half. **[6×]**
6. **Ensemble for diversity of modelling assumption, not seeds**, keep the blend low-dimensional and
   cross-fitted, and refit the final model on 100% of train with fixed counts. **[universal]**
7. **Engineer for survival**: exact output format, deterministic fixed plan, no whole-test statistics, strip-the-ML
   test passes, clear comments. A rejected or failed run scores zero however good the CV was.

## 1b. Platform reality: score × acceptance
Payout needs (a) a score above the printed AI baseline, (b) a top private rank or merit share, and (c) surviving the review that happens
*after* the competition ends (see `platform-facts.md`). So rank candidate approaches by compliance-adjusted value: an approach a reviewer may
reject (frozen embeddings + tabular head on CV/Fine-tuning tasks, keyword-only ranking, hand rules) is worth less than its CV suggests. Use
such components as yardsticks or minor members; make the primary a genuinely trained model the domain regime accepts.

## 2. Effort allocation (expected private gain per hour)

1. Correct contract + exact metric re-implementation + oracle/extreme tests.
2. Validation that mirrors the hidden split; information-ceiling diagnostics (see §4).
3. Output coupling / structural constraints / metric-aware target (largest single lifts on record).
4. Strongest affordable representation, correct pooling/prompt conventions, correct resolution.
5. Loss/decoder fitted to the metric; capacity ladder; augmentation that respects the physics of the task.
6. Diversity ensemble and snapshot/EMA averaging; full-data refit.
7. Small HPO inside the script; micro post-processing last. Do not start 7 before 1–5 are solid.

## 3. Three-stage diagnosis (never blame "the model")

For any generate → score → decode pipeline measure separately on the same grouped split:
- **coverage**: is the right answer reachable in the candidate pool (oracle recall)?
- **ranking**: given it is reachable, does the scorer rank it first?
- **decoding**: given correct scores, does the decode preserve validity and optimise the metric?
Different failures need different fixes (widen candidates / retrain scorer / redesign decode). The union oracle of
two weak, structurally different generators is often far above either alone.

## 4. Diagnose the information ceiling before modelling

Cheap, decisive diagnostics: rows with identical inputs but different labels (irreducible ambiguity), candidate-pool
oracle recall, performance by group/rarity/length slice, a trivial constant/prior baseline under the exact metric,
a leave-group-out estimate of what a memoriser achieves. If the ceiling is low, extra capacity only overfits; if the
metric rewards calibration, spend effort on calibration instead.

## 5. Ordering of questions on a new challenge

1. What is the decision unit and the valid output? 2. What does the metric reward, term by term? 3. What is the
hidden split and which groups must stay together? 4. Which structural facts are guaranteed? 5. What does the
description ban, and what is it silent about? 6. What is the strongest representation that fits compute? 7. What is
the cheapest honest baseline and its exact-metric OOF score? Then iterate with one change per experiment.

## 6. Anti-patterns seen to fail (private LB or review)

- Complex architecture/large ensemble ranking below the simple winner **[6×]**; many moving parts with no evidence
  they transfer.
- Selecting hyperparameters, blend weights and decode constants on the same OOF you report (double-dipped, +0.04
  optimistic); shipping fold-averaged models trained on ~80% where winners refit on 100% (+0.007 to +0.04).
- Probabilities from members on different scales averaged raw (the high-variance member silently dominates).
- Unweighted loss for a weighted/macro metric; accuracy-style training for an F1/AP/NDCG/Brier metric.
- Local proxy built from synthetic distractors or a split that does not match the real test axis (interpolation
  validated, extrapolation required): proxy 0.84 vs real 0.78.
- A "perfect" or near-perfect score on a naturally noisy task: it is a leakage/sibling tell, audit before copying.
- Reading train row order, ids, file sizes, positions as features (flagged and banned; also fragile).
- Hard-coding tuned constants, hand-written regex feature extractors that solve the task, retrieval-only or
  lexical-only primary signal on tasks that ban it: all rejected as "strip-the-ML" failures.

## 1c. Research digest (2026-10; full notes in /research/A-F, hypotheses in references/proposals.md)
1. The printed baseline is likely a single-hold-out, no-ensemble agent pipeline: win with shift-aware grouped CV + stronger representation + a diverse refit ensemble.
2. Treat the public score as a z-score (V17); accept changes only through the corrected paired test and sign consistency (V16); account for winner's curse (V18).
3. Run the train-only audit battery (V19) and the information-ceiling check BEFORE modelling (learned-patterns L001); stop tweaking when the ceiling is reached.
4. Rehearse the shift with broad, description-derived families and leave-one-family-out validation; never design constants from test statistics (CLAUDE.md 2.3A).
5. Prefer LP-FT / weight averaging / simple shrunk blends for robustness; no test-row ranks.
6. Weight provenance, licence and gating are hard compliance items (engineering-and-compliance); torchvision downloads are NOT allowed sources.
7. Profile one batch/epoch before any full GPU run; budget quota; keep long runs inside one turn.
