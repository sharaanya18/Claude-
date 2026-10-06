# private_lb_reference_analysis.md — Phase 4

## Status: BLOCKED. The supplied reference is for a different challenge.

The file provided as the "#1 private leaderboard reference approach" is:

`86816fea-eris-legal-provision-ranking-approach.md`
→ **"Approach: Eris Cross-Lingual Legal-Provision Ranking Challenge"**

It is a build brief for an entirely different task: ranking candidate legal
provisions across Kazakh/Russian/English using hashed integer bucket bags
(28,800 train rows, 5,760 test rows, 6-candidate slates, utility
`0.65/r + 0.35·[r ≤ 3]`, zero-shot `rus→eng` transfer). It contains no
motion-capture, no waveform regression, no force plates, and no KineProof data.

It is also not described as a #1 private-LB solution — it is a *forward-looking
build brief* with its own reference numbers (grouped CV ≈ 0.467–0.472 for a
GBDT, best hand heuristic ≈ 0.465).

**Phase 4 as specified therefore cannot be executed, and no part of the
KineProof approach is derived from a #1 reference.** I did not substitute a
guess for it, and nothing in `final_approach.md` is attributed to one. If the
correct KineProof reference is supplied, this document is where its analysis
goes, and the analysis would be run against the already-measured baselines in
`dataset_audit.md` §8 so that any claimed gain is checked rather than assumed.

I also checked the repository itself: it contains **no KineProof solution** and
no motion/force challenge. The two digested top-solution sets it does hold
(`research/G` Shared Adam, `research/H` AnchorPerm) are unrelated task families;
what transferred from them as *principles* is recorded in
`platform_analysis.md` §3, with the non-transferable parts marked.

---

## What *was* usable from the supplied file

Treated strictly as a methodology document, not as a source of KineProof
modelling ideas. Three of its habits are genuinely good and I adopted them:

1. **Its §7 "strip-the-ML" empirical check.** It does not merely assert
   compliance — it *runs* the hand-built heuristics and tabulates them next to
   the trained model (random 0.440; best single heuristic 0.465; trained GBDT
   0.467–0.472) and then honestly notes the trained lift was "only ~0.005–0.01,
   real and consistent, but thin", and changes the model choice because of it.
   I mirrored this: `dataset_audit.md` §8 tabulates every zero-training probe
   (constant mean 0.227, constant median 0.242, untrained Newtonian 0.281–0.298,
   retrieval 0.274, oracle same-person 0.285) against the learned models, and
   `solution.py` asserts at runtime that the trained model beats the best
   constant waveform. On KineProof the trained lift is large, not thin
   (0.49 → model, vs 0.242 constant), so the strip-the-ML test passes
   comfortably — but it was measured, not assumed.

2. **Permanently logging the no-model baseline inside the pipeline** rather than
   checking it once offline. `solution.py` prints the constant-waveform
   reference beside the held-out-person score on every run.

3. **Its instinct to name a dataset-prep artefact and then refuse to use it.**
   It found that `test.csv` preserves the block-of-6 slate ordering and
   explicitly banned itself from using row order as a signal. The KineProof
   analogue is person identity: repeated trials of one walker are trivially
   identifiable from anthropometry (`dataset_audit.md` §5), which is useful for
   *validation* and must never become a test-time mechanism. I used it only to
   make the in-script validation person-disjoint, never as a model feature, and
   never computed it for test rows.

## What I explicitly did **not** carry over

- **Its §5 wall-clock training safeguard** ("force a stop around the 50–55
  minute mark / 3000–3300 s"). This is the single pattern that the platform's
  Deterministic Execution check has actually rejected, repeatedly and with high
  confidence (`platform_analysis.md` §1.4). The file follows the old Solver
  Guidebook here; the checker overrides it. `solution.py` has a fixed work plan
  and uses time for `print()` only.
- Its model family and feature design (set-overlap statistics, bucket embedding
  + MLP ranker) — irrelevant to waveform regression.
- Its reference scores, which say nothing about this task.

## Risk note

Because no KineProof-specific top solution was available, the approach in
`final_approach.md` rests entirely on (a) the physics of the task, (b) measured
cross-person baselines on the real data, and (c) general platform patterns. The
main consequence is that **I have no external calibration for what score the
leaders reach**, only the description's printed AI baseline of ≈ 0.60. The
honest expectation band in `final_approach.md` is therefore derived from my own
held-out-person measurements alone, and is stated as an estimate with its
uncertainty, not as a prediction of the private score.
