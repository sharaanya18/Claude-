# platform_analysis.md — Project Eris / Shipd, read from this repository

Phase 0 deliverable. Source: recursive inspection of `/home/user/Claude-` (CLAUDE.md, `.claude/skills/eris-playbook/references/*`,
`.claude/agents/*`, `.claude/scripts/*`, `research/A–I`, and the six prior challenge workspaces under `challenges/`).
**This repository contains no KineProof solution** — it is a solver *playbook and infrastructure* repo. What follows is what the
platform demands technically, plus the solver patterns that have repeatedly paid off on *other* Eris challenges, filtered to
those that are legitimate under the rules and plausibly transferable to a motion→force waveform-regression task.

---

## 1. Platform structure

### 1.1 How a challenge is represented
A challenge is a prose description plus a "prepared public bundle". The description is the *authoritative contract* and
overrides every default in the playbook (CLAUDE.md §2.4). It states: task, exact metric formula (usually with grader
source), dataset schema, submission grammar, row count, and a "What Not to Do" section. Workspaces in this repo follow a
fixed shape, which I have mirrored for KineProof:

```
challenges/<slug>/
  CHALLENGE.md         verbatim description (the contract)
  CHALLENGE_NOTES.md   running log / decisions
  metric.py            independent re-implementation of the official metric
  reports/             contract, data audit, split audit, red-team, final approach
  solution.py          the single submitted artefact
  dataset_public/      the bundle (git-ignored)
  working/             submission output (git-ignored)
```

### 1.2 How datasets are exposed
A single `public/` directory containing manifest CSVs (`train.csv`, `test.csv`), a targets CSV for train only, a
`sample_submission.csv`, and any binary assets referenced by relative path from the manifests. Crucially: **every asset
the solver may read is enumerated in a manifest**. Nothing else exists. The held-out answers for test rows are never in
the bundle, and the "Held-out Answer Ingestion" check exists specifically to catch code that reaches for them.

### 1.3 How submissions are evaluated — and the critical economics
This is the single most important platform fact, and it is counter-intuitive:

- The entry point is `python3 solution.py <public_dir> <submission_out>`. Both paths come from `sys.argv`.
- A submission uploads **both `solution.py` and `submission.csv`**, runs four automated checks, and **costs one credit**.
- **There is no free probe of the public score.** Every score observation is a full credit. Credits are 6 per problem
  (+1 per 4 h) with a global ~15–25/day cap.
- A malformed CSV scores **below zero** (dead last) and still burns the credit.
- Consequence, and the rule I will follow: *all* model selection happens on local cross-validation. The public score is
  used only as a sanity veto (|z| > 3 ⇒ audit the split), never as a tuning signal (research V17, P-A01/A02).

The four checks:

| Check | What it actually tests | KineProof-specific risk |
|---|---|---|
| **CSV Score Validation** | the uploaded CSV grades against the test set | 322 rows × 3 JSON arrays of exactly 256 finite floats; malformed ⇒ below 0 |
| **Prompt Compliance** | a model reads the code against the description's explicit requirements (required/prohibited methods, training, hardware, models, data sources) | must be visibly a *trained* motion→force regressor, not a template lookup |
| **Held-out Answer Ingestion** | code must not read private/answer files | never touch anything not in `train.csv`/`test.csv`; never infer person identity from test files to reach labels |
| **Deterministic Execution** | "runtime conditions cannot change training or inference output" | **this is the check that has actually blocked submissions in this repo's history** |

### 1.4 Deterministic Execution: the resolved conflict
The official Solver Guidebook recommends a wall-clock training cutoff (~3000–3300 s). The automated checker **rejects
exactly that**. Documented high-confidence rejections: `if elapsed() > TRAIN_CUTOFF:` truncating HPO; elapsed-gated
partial-data training; deadline-gated fallback prediction paths; deadline-shrunk beam width; even a *dead* `deadline`
parameter defaulting to `None` drew scrutiny. The repo's standing resolution — which I adopt — is: **fixed work plan.
Time is used for `print()` only, never in an `if`, `while`, `break`, `min()`, or a library `timeout=`/`time_limit=`.**
Also banned as env-dependent branching: `torch.cuda.is_available()` switches, `os.cpu_count()`-derived workers,
`try: import X except: use Y`, placeholder/fallback submission writes.

Early stopping on a *validation metric* is fine (data-determined, not clock-determined). The final full-data refit must
use a step count fixed from the CV runs in the same script.

### 1.5 How hidden/private evaluation differs from local evaluation
- The test set is split into a **public slice** (small, noisy, visible) and a **private slice** (the remainder), and the
  **private slice decides rankings and payout**.
- Payout = `P(clear the printed AI baseline) × P(top private rank or merit share) × P(survive post-close review)`.
  Review happens **after** the competition closes; a violation found then forfeits the placement, and credits are never
  refunded. So approaches must be ranked by **compliance-adjusted value**, not raw CV.
- The printed AI baseline (here ≈ 0.60) is the payout gate. Research E's read: it is most likely a single-hold-out,
  no-ensemble agent pipeline that improves one atomic change at a time and selects greedily on one printed hold-out
  score. The documented way to beat such a baseline with margin is *not* harder tuning of the same recipe but:
  shift-aware grouped CV + a stronger representation + a diverse, full-data-refit ensemble.

### 1.6 How train/test are constructed (and why it matters enormously here)
Every Eris challenge I inspected hides a **grouping unit** and splits on it. The descriptions name it: project-,
publisher-, speaker-, site-, family-disjoint; "unseen characters"; "held-out workflow families". KineProof names it
explicitly and unusually precisely:

> "The natural dependency unit is the person. All trials from a person stay on one side of the split. The assignment
> sorts an HMAC-SHA-256 digest of each private source-person key under an evaluator-only key and holds out the first 10
> people; the other 32 form training."

Two consequences I will act on:
1. The split is **person-disjoint and the person key is cryptographically hidden**. The HMAC under an evaluator-only key
   means person identity is *not* recoverable from `sample_id` by design — and attempting to recover it in order to
   reach labels would be a Held-out-Answer-Ingestion violation. Person structure must be *approximated from motion
   content* (anthropometry), which is legitimate because it uses only public train inputs.
2. Any validation that is not person-disjoint is **optimistic and useless**. 1,093 trials from 32 people means ~34
   trials per person; random trial-level CV would put near-replicate trials of the same walker on both sides. This is
   precisely the "sibling leakage" failure class (rejection class S) that the repo flags as the #1 cause of private
   drops.

### 1.7 How solvers are expected to work
One self-contained `solution.py`, plain UTF-8, < 512,000 bytes, that goes from raw bundle to validated `submission.csv`
in a single run with no cached artefacts from earlier runs. Preprocessing, feature construction, training, HPO,
ensembling and inference all inside the script. Hyperparameters found offline and pasted in are a compliance failure;
HPO must run in-script with a fixed trial count. Runtime target ≤ 50 min with ≥ 30 % headroom; default hardware one
A10G (24 GB) unless the description says otherwise. **The KineProof description states no compute limit**, so I apply
the default (A10G, ≤ 50 min) while keeping the architecture small enough that it would also fit CPU-only — the stricter
reading, per CLAUDE.md §2.4.

---

## 2. General solver patterns that repeatedly perform well here

Filtered from `00-s-tier-principles.md` (distilled from 13 challenges / 60+ leaderboard solutions / 6 rejection
post-mortems) and the research notes. `[N×]` = converged independently in N unrelated solutions. I mark each for
KineProof relevance.

### 2.1 Patterns that converged across unrelated tasks — and apply here

| Pattern | Evidence | KineProof application |
|---|---|---|
| **Name the decision unit and valid output space before any EDA** `[5×]` | universal | unit = one trial; output = 3 × 256 finite floats; *invalid* (NaN/wrong length) ⇒ row scores 0, structural error ⇒ exception |
| **Put the metric in the loss and the decode** `[all examined solutions]` | universal | metric is **per-trial-per-axis normalised L1**. ⇒ train with L1 weighted by `1/Σ|y_ita|` per trial-axis. This is the single highest-leverage move and is not what a naive MSE regressor does |
| **Mirror the hidden split; be paranoid about leakage** `[universal]` | #1 private-drop cause | person-disjoint GroupKFold with persons *derived from anthropometry*, repeated over seeds |
| **Representation first, machinery second** `[6×]`; simplest solution won in 6 problems | 6 problems | prefer a well-conditioned temporal CNN on good kinematic features over a large transformer on 1,093 samples |
| **Ensemble for diversity of assumption, not seeds** `[universal]` | universal | different target representations (direct waveform vs basis coefficients), different architectures, different feature views |
| **Refit on 100 % of train with fixed counts** (V9) | +0.007…+0.04 recorded | fold-averaged ~80 % models leave measurable score on the table |
| **Diagnose the information ceiling before modelling** (L001, §4) | avoided a week of tweaks on AnchorPerm | measure: axis-wise score of the mean waveform; score of an oracle that knows the true person; per-axis achievable error |
| **Three-stage diagnosis** (coverage / ranking / decoding) | universal | adapted here to: *representation* (can features carry the force?) / *regression* / *reconstruction* (does the basis preserve peaks?) |
| **Corrected paired test + sign consistency** (V16) | Nadeau–Bengio; naive SE ~1.5–2.2× too small | required before keeping any change; group splits add variance (V22) |
| **Winner's-curse bookkeeping** (V18) | E[max of N normals] | report `best_cv − E[max]·σ`; re-score finalists on fresh split seeds |

### 2.2 Methods the prompt asked me to investigate, with the repo's verdict

- **Strong validation** — load-bearing, the single most emphasised item. Adopt in the strictest form.
- **Leakage detection** — adopt the train-only audit battery (V19): label-shuffle run must score at chance; id/order/
  file-size-only model must show no lift; random-CV vs group-CV gap is the leakage gauge; validation-to-train
  nearest-neighbour distance under random vs group folds.
- **Feature engineering** — legitimate *only as inputs to a trained model*. For a kinematics task this is strongly
  favoured: derivatives and joint geometry are the physics of the problem (F8: "domain theory as architecture").
- **Target transformation / residual modelling** — directly applicable: predict a residual over a learned mean waveform,
  or predict basis coefficients. F8's "physical relation as residual base for a learned correction" is the pattern.
- **Dimensionality reduction (PCA/DCT) of the *target*** — legitimate, but must be fit on **train only**, and the repo
  warns about oversmoothing destroying exactly the peaks the metric cares about. Test empirically against direct
  prediction.
- **Ensembles / model blending** — adopt, with equal or strongly-shrunk non-negative weights fitted on OOF and
  cross-fitted (P-A12: simple blends win under noise; forward selection overfits).
- **Augmentation** — adopt only label-preserving transforms. Critical subtlety for this task: an augmentation must
  preserve the *motion→force* relationship, and forces are in body-weight units, so spatial rescaling changes the
  physics. Left/right mirroring must flip the lateral axis sign. Every augmentation needs an ablation.
- **Nearest-neighbour / retrieval methods** — legitimate as a *member or feature*, and a good yardstick. Must retrieve
  from train only, and during validation must exclude same-person training trials or the estimate is leaked.
- **Calibration / uncertainty** — fit any post-hoc map on group-held-out OOF (V21), few parameters, cross-fitted.
- **Test-time normalisation** — **BANNED.** Any statistic reduced across test rows and fed back is rejection class R.
  The acceptance test is the **half-rows test**: score the file twice, once with half the rows dropped; the kept rows'
  predictions must be byte-identical.
- **Pseudo-labeling on test** — **BANNED** (CLAUDE.md §2.3 #5).
- **Domain adaptation to test** — **BANNED** in the same clause. Shift robustness must be rehearsed from *train* only,
  via leave-one-group-out and description-derived shift families (V7).

### 2.3 Anti-patterns recorded as having failed
- Complex architecture / large ensemble ranking below the simple winner `[6×]`.
- Selecting hyperparameters, blend weights and decode constants on the same OOF you report (+0.04 optimistic).
- Shipping fold-averaged ~80 % models where winners refit on 100 %.
- Averaging members on different scales (the high-variance member silently dominates).
- A local proxy on the wrong axis: interpolation validated while extrapolation required (recorded proxy 0.84 vs real 0.78).
- A near-perfect score on a naturally noisy task — treat as a leakage tell, audit before believing.
- Reading row order, ids, file sizes or positions as features (flagged, banned, and fragile).
- Hardcoded offline-tuned constants; hand-written extractors that solve the task with a thin model on top.

---

## 3. Competitor intelligence inside this repository

There is **no KineProof solution here**. What exists are digests of top solutions for two other challenges plus this
team's own retrospectives. I read them for *principles*, and I record explicitly which parts do not transfer.

### 3.1 `research/G` — Shared Adam (hidden-optimizer-state task), 5 top solutions
- **Why it worked:** all five estimated the *latent state* of a known process from public statistics, then simulated the
  disclosed dynamics forward to score candidate actions (L006, `[6×]`). Rank 1/3/5 additionally built the correction
  basis from gradients at *several* points along a trajectory (L007). Ranks 1/2/4 used learned uncertainty plus
  **in-script selection of the decision rule** on held-out groups with the official metric (L008).
- **Transferable to KineProof:** (a) *estimate the latent physical quantity, then use known physics forward* — here the
  latent is the walker's anthropometry/gait phase and the "known physics" is that GRF ≈ body-mass-normalised
  whole-body-COM acceleration + gravity; (b) choosing post-processing **inside the script on held-out groups under the
  official metric**, never offline.
- **Does not transfer:** the exact simulator inversion (no runnable force simulator is supplied, and the description
  explicitly says the task is not a physics-engine simulation).
- **Leakage check:** none — all evidence was public per-row.

### 3.2 `research/H` + L011–L013 — AnchorPerm, 5 top solutions
- **Why it worked:** all five **pooled information across test rows** (clustered test rows into regimes, fitted an
  anchor-based map on pooled revealed anchors, self-trained on decoded pairs). This was the large lever.
- **Verdict: DO NOT ADOPT.** It is test-set use beyond one-row inference, banned by CLAUDE.md §2.3 #5 / §2.3A unless a
  *written* reviewer approval exists for that specific challenge. L013 is the standing rule: "When the leaders all break
  your rule, say so and keep the rule." L005 records the cost of drifting toward that boundary.
- **What did transfer (L012, compliant levers):** regime-invariant second-order geometry; train-only shift simulation;
  learned assignment with structural decode; seed-ensemble marginals; isotonic confidence on out-of-fold rows;
  **stratum-aware validation with a worst-stratum term**.
- **Robustness lesson (L003):** row-local models genuinely cannot learn a mapping that changed between train and test.
  For KineProof this is the key risk to size: *how much does the motion→force map differ between walkers?* If it is
  mostly a shared map plus per-person scaling, a row-local model is fine. That is an empirical question I will measure
  (Phase 3/12), not assume.

### 3.3 `research/F` + `I` — this team's own retrospective
- AnchorPerm: public 0.5934 vs baseline 0.5616 with row-local methods; the gap to the leaders was entirely the banned
  transductive lever.
- Shared Adam: public 0.589 from a first version of the right idea.
- `research/I` gap analysis: the recurring self-diagnosed misses are **not** exotic architectures. They are: not
  reconstructing the latent quantity, not putting the metric into the loss, not simulating the shift, and under-using
  in-script selection of the decode rule on held-out groups.

### 3.4 Prior workspaces as engineering templates
`challenges/*/metric.py` + `tests_metric.py` (independent metric re-implementation with hand-computed extremes),
`groups.py`/`folds.py` (derived grouping), `dev_*.py` (one experiment per file), and `.claude/scripts/` gates
(`compliance_scan.py`, `validate_submission.py`, `determinism_check.py`, `half_rows_test.py`). I will reuse this
structure. One concrete note from `concession_forecast`'s history: a solution was committed with comments/docstrings
stripped "AST-identical" — that is the *opposite* of what the Prompt Compliance check rewards, and the playbook says
reviewers read comments. I will ship a commented script.

---

## 4. What this means for KineProof — the Phase 0 decisions

Carried forward into the plan, each traceable to a source above:

1. **Validation is person-disjoint or it is worthless.** Person labels are cryptographically hidden, so I must derive
   person groups from anthropometry (segment lengths are near-constant within a walker and are a legitimate function of
   public train inputs). Bias direction must be stated: clusters coarser than true persons ⇒ pessimistic; finer ⇒
   optimistic (V5). Repeated group splits (V22), corrected paired test (V16).
2. **The loss must be the metric.** Per-trial-per-axis normalised L1, i.e. L1 weighted by `1/Σ|y|` for that trial and
   axis. Horizontal axes have small magnitudes but equal weight after their own normalisation — an unweighted loss would
   let vertical force dominate and cap the score near 1/3 + ε. The description states this outright: "A model that
   reconstructs only vertical loading can earn at most one third of the total score."
3. **No test statistics anywhere.** Every transformer (scaler, PCA/DCT basis, person-clustering) fit on train only, then
   `.transform` on test. Verified by the half-rows test.
4. **Fixed work plan, no clock, no device branches.** Seeds everywhere, fixed epochs/folds/trials/workers.
5. **Information ceiling first.** Before any deep model: mean-waveform baseline, per-axis scores, oracle-person
   baseline, and a measurement of how much of the force is linearly predictable from COM acceleration. This tells me
   whether 0.80 is reachable at all and which axis is the binding constraint.
6. **Diversity ensemble + 100 % refit, simple weights.**
7. **Compliance-adjusted choice.** This is a waveform-regression task on supplied numeric arrays with no pretrained
   backbone implied; a trained temporal neural network is the regime-appropriate primary. A retrieval/template member is
   a legitimate *minor* member and a useful yardstick, but must not be the thing doing the work (strip-the-ML, Q1).

### Open risk flagged now
The reference approach supplied for Phase 4 is for a **different challenge** (cross-lingual legal-provision ranking),
not KineProof. Phase 4 therefore cannot be executed as specified. I have recorded what *is* usable from it (it is a
build-brief template with a compliance self-audit section, and its §7 "strip-the-ML" empirical check is a good habit)
and will proceed with Phases 1–3 and 5 onward, which do not depend on it.
