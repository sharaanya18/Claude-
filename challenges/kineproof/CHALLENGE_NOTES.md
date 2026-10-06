# KineProof — challenge notes

## Contract (from the description)
- Unit: one trial. Input `(256, 22, 3)` float32 markers (time, marker, xyz), metres,
  origin = frame-0 ASIS midpoint. Output: 3 JSON arrays of exactly 256 finite floats,
  body-weight units.
- Metric: `E_ia = Σ|p−y| / Σ|y|` per trial-axis; `score = mean max(0, 1−E)` over x/y/z
  then over trials. Higher is better. Zero-truth axis: exact zero earns 1, anything else 0.
- Malformed/missing/NaN/inf axis ⇒ that whole ROW scores 0 (other rows still score).
  Wrong header / row count / duplicate / missing / unknown id ⇒ global exception.
- Submission: `sample_id,force_x,force_y,force_z`, exactly 322 data rows. Row order
  does not matter (we match sample_submission.csv anyway).
- Split: person-disjoint. 32 train people / 10 test people, assigned by HMAC-SHA-256
  of a private person key under an evaluator-only key. 1,093 train / 322 test trials.
- Explicit bans: no scalar-per-trial, no foot-contact class, no force-plate index,
  no copying the placeholder submission. Not a physics-engine simulation task.
- Compute: the description states NO hardware or runtime limit. We apply the stricter
  reading (CLAUDE.md §2.4) and hardcode CPU, sized to finish well inside the budget.

## Decisions and why
1. **Loss = the metric.** Denominators measured: force_y 209.4, force_x 18.4,
   force_z 9.5 → 11.4x / 22.0x. Unweighted L1 would cap the score near 1/3.
2. **Person-disjoint validation from anthropometry.** Person ids are hidden by design;
   segment-length clustering recovers 32 clusters with a natural dendrogram gap at
   exactly 32. Random CV is 17x more leaked (val→train NN distance 0.195 vs 3.318).
3. **Physics as the residual base.** `f = a_com/g (+1 vertically)`; mass cancels.
   An untrained Newtonian estimate scores 0.616 on force_y, so the net starts there
   (zero-init head) and learns only the correction.
4. **Direct time-domain target.** Basis targets are measurably capped
   (DCT-32 → 0.915, DCT-128 → 0.982 oracle).
5. **Retrieval rejected on evidence**: kNN 0.274, oracle same-person template 0.285,
   vs 0.486 for a per-frame linear model. The signal is per-trial physics, not
   person-level — which also means cross-person transfer is less fragile than feared.
6. **One augmentation, physically exact**: left/right mirror (swap contralateral
   markers, negate lateral axis; force_z negates, x/y unchanged). Asserted at runtime.

## Open / not done (time-limited)
- No KineProof reference solution was supplied (the provided file is for a different
  challenge) → Phase 4 not executed. See reports/private_lb_reference_analysis.md.
- Single fixed configuration; no multi-split-seed finalist confirmation (V18/P-A09).
- Untested levers, in priority order: exact time-rescaling augmentation
  (`f_x → s²f_x`, `f_y → s²(f_y−1)+1`); a second model family for assumption
  diversity; capacity ladder; explicit auxiliary contact-probability head; more seeds.
- Train-vs-test distribution comparison deliberately NOT performed (compliance).
