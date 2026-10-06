# Phase 0 — Platform analysis (Shipd / Project Eris)

Source of truth, in order: the Two Puffs challenge description > the workspace playbook
(`CLAUDE.md`, `.claude/skills/eris-playbook/references/*`) distilled from the official Solver
Guidebook, `shipd_docs.docx`, pre-submission-check screenshots and six rejection post-mortems >
prior challenge workspaces in `challenges/` (anchorperm, shared_adam_repair,
grounded_ip_formulation, complementary_entity_evidence, concession_forecast, es_gl_matching).

Note on scope: the user's brief mentions a GitHub repository and a #1 private-LB reference
solution. Only the dataset archive was supplied in this session. This file therefore analyses
the platform from the workspace itself (which *is* the distilled platform knowledge base);
`private_lb_reference_analysis.md` records what is and is not available for Phase 5.

## 1. Platform architecture

- A challenge ships a description plus a `public/` directory. The solver delivers **one
  self-contained `solution.py`**, invoked as `python3 solution.py <public_dir> <submission_out>`.
  Both paths come from `sys.argv`; nothing may be hardcoded.
- The script must do **everything from raw data every run**: preprocessing, feature extraction,
  training, inference, CSV writing. No cached artefacts, no weights trained offline, no attached
  files. Source must be plain readable UTF-8 under 512,000 bytes.
- Execution environment is the Kaggle Docker image. Default hardware is one A10G, but **the
  challenge text overrides it** — Two Puffs states "a single CPU in minutes", so the plan is
  sized for CPU and `device` questions do not arise.
- Internet is allowed *only* for pretrained backbone weights from Hugging Face / timm. Two Puffs
  bans pretrained weights entirely ("every trainable parameter starts random"), so the solution
  must make no network call at all.
- Credits: 6 per problem, roughly 15–25/day globally, refunded 24 h after use. **There is no free
  public-score probe**: a submission uploads both `solution.py` and `submission.csv`, runs the
  checks, and costs a credit whether or not it succeeds. All validation is therefore local.

## 2. Evaluation mechanics

- The test set is split into a **public** slice (visible score) and a **private** slice that
  decides the ranking. The public number is a small, noisy sample — treat it as a z-score, not a
  verdict (`validation-recipes.md` V17). CV is the source of truth.
- Payout = beat the printed AI baseline **and** place on the private board **and** survive the
  review that happens after the competition closes. A violation found at review forfeits the
  placement; credits are never refunded. So the objective is
  *compliance-adjusted* score, not raw score.
- Four automated pre-submission checks:
  1. **CSV Score Validation** — the uploaded CSV is graded.
  2. **Prompt Compliance** — a model reads the code against the challenge's explicit
     requirements (required/prohibited methods, training, hardware, models, data sources). Make
     compliance *legible*: a requirements map in the docstring, a comment at every fit/predict.
  3. **Held-out Answer Ingestion** — the code must never touch private/answer files or try to
     recover labels from ids, filenames or provenance.
  4. **Deterministic Execution** — "runtime conditions cannot change training or inference
     output". This is the check that has actually blocked submissions. Every recorded failure was
     wall-clock-dependent control flow: `if elapsed() > CUTOFF` truncating HPO, deadline-gated
     fallback prediction paths, clock-shrunk beam width, even a dead `deadline=None` parameter.
     The Guidebook's own suggestion of a 3000 s training cutoff is exactly what the checker
     rejects. **Fixed work plan; time may be used for logging only.**

## 3. Relevant competitor patterns (from the distilled library)

The seven moves that separate top-5 from the middle, in order of recorded lift:

1. Name the decision unit and the valid output space before any EDA; most tasks are structured
   decision problems where you learn calibrated local evidence and then choose the output that
   maximises expected utility under the real metric.
2. **Put the metric into the loss and the decode.** Every examined top solution did this.
   Back-solve invertible metrics into richer targets; derive closed-form decodes; verify with an
   oracle-decode check (gold local scores must decode to ~100%).
3. **Turn every structural fact of the generating process into signal** — invariants, couplings
   between outputs, latent variables marginalised exactly when the state space is small. The
   single largest recorded lift (+0.14) came from an output coupling.
4. Mirror the hidden split; be paranoid about leakage; cross-fit every selection step.
5. Representation first, machinery second — in six recorded problems the *simplest* solution won
   while heavier machinery ranked in the bottom half.
6. Ensemble for diversity of modelling assumption, not seeds; refit on 100% of train.
7. Engineer for survival: exact output format, deterministic fixed plan, no whole-test statistics.

Directly transferable prior lessons:
- **L006 "Estimate the latent state, then simulate"** (shared_adam_repair, confirmed 6×): when a
  hidden process state determines the metric and the generating dynamics are disclosed and
  runnable, learn the latent and then *simulate the disclosed dynamics* instead of regressing the
  observable. This is precisely the Two Puffs situation and is the backbone of the approach taken.
- **L008 "Learned uncertainty plus in-script decision-rule selection"** (3×): measure residual
  spread on held-out groups, draw states from it, and choose the decision rule inside the script
  on held-out groups with the official metric.
- **L001 "Information ceiling by Monte Carlo"**: measure the ceiling before spending effort on
  capacity. Two Puffs publishes its own ceiling (0.9116), which can be — and was — reproduced.
- **A17 (metric-and-decoding)**: for Brier-type metrics, add squared error to the loss and
  recalibrate per output slot on OOF, cross-fitted.

## 4. Reusable solver ideas applied here

- Re-implement the metric exactly and unit-test the documented extremes before modelling. The
  description publishes four reference scores and three population fractions; all were reproduced
  (§ `dataset_audit.md`), which validates metric, parsing and split simultaneously.
- Oracle-decode check: feed the *fitted* latent through the decoder and confirm it reproduces the
  published oracle. Got 0.9133 against a documented 0.9116.
- Reproduce the published reference rung (gradient boosting on recomputed indices) in the local
  harness to calibrate CV against the held-out board: got 0.5974 OOF against 0.6017 held-out.
- Repeated stratified folds, corrected paired test (Nadeau–Bengio, V16) and sign consistency
  before keeping any change; winner's-curse bookkeeping (V18) on the number of candidates compared.

## 5. Potential leakage mechanisms in this platform / this challenge

- **Whole-test statistics** of any kind: fitting scalers, encoders, PCA, clustering, quantiles or
  calibration maps on test or train+test; rank-normalising across the test batch; pseudo-labelling.
  Banned by CLAUDE.md §2.3 #5. Every transform here is fit on train (or on the shipped pool) and
  only `.transform`-ed onto test.
- **Provenance recovery.** The recordings come from a public national health survey whose
  second-session results are published. Recovering the original records would reveal the answers
  exactly and is explicitly named as cheating. No external data, no id/filename matching, no
  derivative that encodes it.
- **Sibling leakage.** Not applicable in the usual way: one row per participant, one participant
  per split, and `blows.jsonl` keys by the same pseudonymous id. There is no shared-entity
  relation to union-find over. The pool is a disjoint participant set with no targets.
- **Id / order features.** Participant keys are pseudonyms; row order carries no information and
  is never read as a feature.
- **Target-derived statistics**: the latent response targets are derived from the *published*
  train targets only, and every model fit on them is cross-fitted inside the folds.

## 6. Legitimate vs prohibited, for Two Puffs specifically

Legitimate:
- Recomputing spirometric indices from the raw traces (the description insists on it).
- The exact bootstrap over the participant's **own pre-session** acceptable blows — this is
  one-participant-at-a-time inference on that participant's own shipped inputs.
- Representations and normative reference equations learned from the shipped `pool.jsonl`.
- In-script training, in-script hyperparameter choice, OOF-fitted calibration and blend weights.

Prohibited:
- Pretrained spirometry or clinical-waveform models; any pretrained weights at all.
- External datasets; recovering the original survey records; synthetic participants or traces.
- Using the pool as extra *labelled* data (it has no targets, and it is a differently selected,
  less obstructed population — see the audit).
- Anything fitted across test rows; test-set calibration; private-leaderboard feedback as labels.
- Hardcoded constants found offline; wall-clock or hardware branches.

Grey, and therefore kept out of the primary path: nothing in this solution relies on regex,
lookup tables or hand rules. The strip-the-ML test passes by construction — remove the trained
models and the decode has no latent distribution to integrate over, collapsing to the population
prior (score ≈ 0.48 rather than ≈ 0.70).

## 7. Lessons applicable specifically to Two Puffs

1. **The labels are a disclosed, runnable generative process.** Each target is the fraction of
   4,000 bootstrap draws in which an event fired, with the draw rule, the index conventions and
   the thresholds all published. That makes this an L006 problem, not a tabular classification
   problem: learn the latent response, then run the disclosed bootstrap exactly.
2. **Half the published ceiling gap is measurement noise that is free to model.** The description
   states that an oracle given the true response magnitude, still facing blow-sampling noise,
   scores 0.9116, and that a model additionally handed the hidden session's spread scores 0.6003 —
   no better. So the latent is a *scalar per arm*, and the only missing information is its size.
3. **The bootstrap over the pre-session is exactly enumerable.** With at most 10 acceptable blows
   there are at most C(12,3)=220 distinct triples; the whole pre-side distribution is a weighted
   list. No Monte Carlo, no sampling noise, fully deterministic.
4. **Calibration is structural, not post-hoc.** Because the decode integrates a predicted
   *distribution* over the latent, the output is automatically a calibrated probability and
   automatically monotone within an arm. Isotonic patches on nine separate heads cannot recover
   this.
5. **FDS cannot be bought with RCS.** Half the score is the fragility ordering of `ats` alone; a
   constant `ats` caps the score at `sqrt(RCS * 0.5)`. The generative decode produces the
   fragility ordering for free because `4p(1-p)` falls out of the blow-sampling physics.
6. **The published reference rungs are CV calibration targets.** Reproducing 0.6017 locally before
   trying to beat it is the cheapest possible guard against a broken harness.
7. **CPU-only, minutes.** No GPU advice applies. The fixed work plan is sized for ~10 min on 4
   cores with large headroom.
