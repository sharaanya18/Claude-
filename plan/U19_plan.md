# Plan U19: NLP Scientific Comment Lifecycle Span Generation (Project Eris)

Sources read: `.claude/agents/eris-strategist.md`, `CLAUDE.md`, `tasks/user/U19_lifecycle_span_generation.md`. No dataset was available, so every statement about data below is a **hypothesis (UNVERIFIED)** and is paired with the diagnostic that would confirm or refute it. Nothing was trained or run. No solution code is written here.

Notation used throughout: `L = log1p(9000) = 9.1052`, position step `delta = L/63 = 0.14453` log1p-units per position. Position `j` means `log1p(days) = j*delta`.

---

## Contract & decision unit

**One valid answer per test id** (3,622 rows, 2 held-out projects), columns exactly `["id","prediction"]` (the evaluator checks `list(submission.columns) == ["id","prediction"]`; note `sample_submission.csv` is said to use the same two columns, so verify against it at run time). `prediction` must match the evaluator's regex `^lifecycle_profile=([0-9]+(?:\.[0-9]+)?(?:;[0-9]+(?:\.[0-9]+)?){63})$`:
- exactly 64 values separated by `;`, each non-negative, finite, <= 1, positive total mass;
- **plain decimal notation only**. Scientific notation (e.g. `1e-05`, Python's default repr for small floats) does not match the regex and the whole submission would be rejected. Format with fixed decimals (`f"{v:.6f}"`).
- Decoded span must lie in [0, 9000] days (guaranteed because the mean position lies in [0, 63]).

**Invalid vs low-scoring.** Invalid: wrong columns, wrong row count, duplicate or missing ids, regex mismatch, any value > 1 or < 0, scientific notation, zero mass. Low-scoring: a valid constant (the shipped sample is a delta at position 0, i.e. predicts 0 days for every row).

**What the metric rewards, term by term.**
1. The grader reduces each 64-vector to ONE scalar: `q = sum(j*v_j)/sum(v_j)`, `days = expm1(q*L/63)`. Only the weighted mean position matters. The shape of the profile beyond its mean is irrelevant to the score.
2. Per-row utility `S_i = exp(-|log1p(p_i) - log1p(d_i)| / ln2)`. In position units this is `exp(-|q_pred - q_true| / 4.796)` (4.796 = ln2/delta). It is a Laplace kernel, so it is **mode-seeking, not mean-seeking**: it is maximised by the conditional mode of the (smoothed) posterior of `log1p(d)`, not by the mean or median. A factor-of-two error gives |delta| ~ 0.693 and `S = exp(-1) = 0.368`. **The description text says 0.5; the evaluator code gives 0.368. The code is authoritative; I follow the code.**
3. Horizon balancing: `B_h` = mean of `S_i` over rows whose TRUE horizon is `h`, score = mean(B_0,B_1,B_2). Each row therefore carries weight `1/(3*N_h(i))`, where `N_h` is the TEST count of its true horizon. Equivalent to a class-balanced weighted utility.
4. Horizon boundaries in position space: 90 days -> `q = 31.2`; 365 days -> `q = 40.8`. Short-lived occupies positions 0-31.2 (31 wide), review window 31.2-40.8 (only 9.6 wide, about 2 kernel scales), persistent 40.8-63 (22 wide). The horizon of a row only decides which `B_h` bucket it is averaged in; it does not gate the score (no "wrong horizon zeroes the score" rule), but because the three buckets are weighted equally and are far apart in log space, a prediction in one bucket earns almost nothing from rows in a far bucket (e.g. predicting 365 days for a true 10-day item: `|delta| = ln(366/11) = 3.5`, `S = 0.006`).
5. A horizon with no support counts as 0 (irrelevant here: the test surely has all three).

**True independent unit.** The row is the decision unit for the metric, but the **independent unit for generalisation is the project** (test = 2 unseen projects; train = an unknown number of projects, identity withheld). Effective sample size for "does this transfer" is the number of train projects (hypothesis: small, perhaps 3-10; UNVERIFIED), not 15,107 rows.

**Pipeline stages (diagnosed separately).**
- *Candidate coverage* is trivial here: the candidate set is the 64-position grid plus linear interpolation between neighbours, which covers every possible true target by construction (the truth is itself a two-point interpolation; to verify, see Data findings).
- *Scoring*: a model that outputs a predictive distribution `q_x(j)` over the 64 positions given the input tokens.
- *Decoding*: choose the scalar `p*` that maximises expected utility under `q_x` (and the balanced weighting), then emit the 2-point interpolated profile for `p*`.

**Core structural reduction.** The task is "generate a 64-step profile" but the score depends on one scalar per row. The smallest learnable decision that preserves every valid answer is: predict a distribution over the 64 ordered positions, pick a scalar, emit the two adjacent non-zero values `(1-f, f)` at `floor(p*)`, `ceil(p*)` so the weighted mean equals `p*` exactly. The description itself sanctions this ("convert your scalar estimate to a 64-value profile using the same log-duration interpolation convention").

---

## Compliance regime

**Domain.** NLP / "Sequence To Sequence" per the challenge header, but the inputs are two tiny categorical strings, not free text. Not labelled Fine-tuning or From-scratch by the description. Compute: A10G (no explicit time limit stated; see assumptions).

**Explicit bans extracted from the description (treated as hard constraints):**
1. Do not reconstruct or infer the held-out project identity from source filenames, opaque ids, row position, file ordering, or external provenance records.
2. Do not query any public provenance record, mirror of the source collection, or source repository to match inputs and recover test spans.
3. Do not use raw source-history files, commit logs, future project versions, or any repository-specific lookup table.
4. Do not copy/construct hidden test labels or an external lookup table.
5. `record_token`: "solvers should ignore this field" (author-keyed, no predictive meaning). Hard ban on using it as a feature, group key, or fold key. `id` likewise is opaque and not a feature.
6. Only public files may be used; the `train_targets.csv` sidecar is the ONLY lifecycle supervision. No external data.
7. Output must obey the 64-step contract and be generated "without future labels or source history".

**CLAUDE.md rules that apply on top (all binding):**
- Train-only fitting of every vocabulary / token-to-index map / class weight / calibration constant. The test file is read only to produce per-row predictions. No test-set statistics, no peeking at test token coverage, no test-based vocabulary.
- No wall-clock branching; fixed work plan; seeds; fixed device `cuda`; no `try/except` import fallbacks; no `os.cpu_count()`-derived settings.
- No LLM-generated content, no synthetic labelled data, no pseudo-labels.
- Source < 512 KB, plain readable code.

**Where the description is silent, and what I assume.**
- *Runtime limit*: silent beyond "Compute: A10G". Assume CLAUDE.md's <= 50 min ceiling and plan ~15-25 min (estimate).
- *Pretrained weights*: the requirement sentence says "Train a sequence-generation model using only the supplied public files". This can be read as "data files only" (pretrained backbones then allowed under CLAUDE.md) or as "no pretrained weights at all". **I pick the reading that is compliant under both: a model trained from scratch on the public files only, no pretrained weights, no downloads.** Measured cost expectation: approximately zero, because the input strings (`profile_defect+verification token_band_short`, `len_short style_plain`) are synthetic tokens on which a pretrained LM has no useful prior; the optional check of a small pretrained encoder is listed in the roadmap only if a reviewer confirms pretrained weights are allowed.
- *Metric-aware decoding*: the description does not ban it ("Any method is allowed if it uses the public inputs, obeys the 64-step sequence contract..."). CLAUDE.md 2.4 tells me to treat "decoding tricks are fine" style sentences as possible boilerplate. So I keep the decode **low-dimensional, derived from the metric formula given in the description (kernel scale ln2, thresholds 90/365, max 9000), and its only fitted pieces (temperature, shrinkage, decode rule) are fitted in-script on train OOF**. No constant is copied from earlier real submissions.
- *Ensembling and multi-family blending*: not banned; keep the neural generator dominant.
- *Class-balanced training weights*: a training-time loss adjustment, not a decode hack (see Metric-aware training & decode). In rules.

**Strip-the-ML test (stated up front, honestly).** If the trained model is removed, what remains is a single global scalar (the best balanced constant). That baseline probably scores a meaningful fraction of the final score because the input features are extremely coarse. The ML component is the only source of per-row variation; its measured OOF gain over the best constant must be reported in the script log and the final report. If the gain is within noise, say so rather than hide it. This is an inherent property of this dataset, not a design choice.

---

## Data findings

**NO DATASET WAS AVAILABLE. Everything below is a hypothesis or a diagnostic to run on `train.csv` + `train_targets.csv`. None of it is verified.** Use test only for schema, row count (3,622) and id format.

### Diagnostics to run on train (in a scratch notebook, not in `solution.py`)

D1. **Schema and join.** `train.csv` shape (expect 15,107 x 4: `id, record_token, comment_cues, comment_shape`); `train_targets.csv` (expect `id, target`); join one-to-one on `id`, assert 15,107 unique ids, no nulls, id length 17 / prefix `x`, record_token length 21 / prefix `r`. Check whether the `target` string carries a `lifecycle_profile=` prefix or is bare (parser must accept both).

D2. **Input vocabulary.** `nunique` and value counts of `comment_cues` and `comment_shape`; split `comment_cues` into (cue-profile part, token_band part); split the profile part on `+` into cue atoms; list the atoms. Count distinct (cues, shape) **cells** and the cell-size distribution (min should be >= 12 per the description's backoff rule; hypothesis: 50-400 cells, top 10 cells hold a large share). Share of rows in the `profile_general` / `token_band_mixed` / `shape_other` backoff buckets (hypothesis: sizeable, tens of percent, heterogeneous). Cross-tab `token_band_{short,long}` vs `len_*` (hypothesis: strongly aligned because both come from the same comment length; token_band_short <-> len_tiny/short, long <-> len_long/very_long).

D3. **Target structure.** Parse all 15,107 targets to a 15,107 x 64 matrix. Check: each row sums to 1; number of non-zero entries per row (hypothesis: exactly 1 or 2, adjacent: a linearly interpolated point mass, so `q` carries ALL the information). If any row has >2 non-zero or non-adjacent entries, the "predict a scalar then interpolate" reduction loses information and the decode must use the full distribution; revisit. Compute `q`, `days`, horizon. Report class shares (hypothesis: persistent-dominant, e.g. persistent 45-70%, review 10-25%, short 15-35%; UNVERIFIED; this is why balanced weighting matters). Histogram of `q` overall and per horizon (hypothesis: bimodal with a mass near 0-10 days and a broad persistent mode, with a possible spike at the upper cap because unremoved items are right-censored at project-history end; the censoring spike location would be project-specific).

D4. **Atoms and censoring.** Count exact duplicates of `q` (atoms). Hypothesis: spikes at 0 days (position 0) and a pile-up at large `q`. If a handful of discrete atoms/clusters appear that look like per-project history caps, record them as evidence of the number of train projects (use ONLY for understanding validation optimism; see Validation design for why I do not build folds from it by default).

D5. **Information ceiling (the key number).** On the balanced metric implemented exactly as the evaluator:
 - (a) the shipped sample's constant (0 days);
 - (b) best global constant under balanced weights (grid over positions 0..63);
 - (c) cell-wise best constant, **in-sample** (optimistic upper bound);
 - (d) cell-wise best constant, **cross-fitted** over 5 folds (honest in-distribution ceiling for a lookup);
 - (e) the same with the atoms-level (`cue atoms`, `band`, `len`, `style` main effects only) model.
 Hypothesis: (b) lands in the low 0.3s, (d)-(b) is small (+0.01 to +0.06), (c)-(d) is larger (cell lookup overfits). All numbers UNVERIFIED. A large (d)-(b) would mean the features carry real signal; a tiny one means the final result is dominated by decode and shift-robustness, not modelling.

D6. **Irreducible ambiguity.** For each cell, the horizon distribution; fraction of rows in cells where the top horizon has < 60% share (hypothesis: most cells are ambiguous, so cell -> horizon is a weak predictor). Mutual information between cell and horizon (hypothesis: small, a few hundredths of a nat).

D7. **Variance decomposition.** Fraction of `q` variance explained by cell, by cue atoms main effects, by len, by style (eta-squared; hypothesis: low single-digit to ~10%).

D8. **Order/id audit (audit only, never a feature).** Check whether train row order, id sort order, or record_token sort order correlates with `q` or horizon. Purpose: only to know whether train rows are ordered by project (a sibling-leakage tell per F2.17) and therefore how optimistic random CV is. Result is NOT used to build the model, the features, or the folds by default (ban 1/5 above and the description's "row order may differ"); if even this audit is considered uncomfortable, skip it and treat random CV as optimistic by an unknown margin.

D9. **Compositional coverage.** Per cue atom, rows and horizon mix; how many cells would be unseen if each cue profile were held out (for the stress CV below).

D10. **Prior-shift sensitivity.** Recompute (b) and (d) after re-weighting classes to a few alternative priors, to see how fragile the best constant is.

### Hypotheses (all UNVERIFIED)
- H1: features are weak; per-row signal is a small lift over a balanced constant.
- H2: the 3 horizons are well separated in log space and the review window is narrow, so the balanced-optimal constant sits near the review window / lower persistent range.
- H3: the persistent class is right-censored at project history length, so its within-class distribution is project-specific; this is the main private-LB transfer risk and cannot be measured without project labels.
- H4: test cell mixture differs from train (two new projects); some cells may be rare or unseen in train. I may not inspect test for this; the model must simply be built to degrade gracefully (UNK atoms, additive structure).
- H5: the cue profile is a composition of atoms (`defect`, `verification`, ...) so atom-level sharing helps unseen combinations.

---

## Validation design

**How the test split was made (from the description).** Project-disjoint: the test is two entire scientific-software projects not in train. Train has no project column and the project identity is deliberately withheld. So the real shift is "new project": different cell mixture, different per-class censoring/horizon distributions, possibly different base rates.

**What I cannot do.** Project-grouped CV (no project labels, and constructing them from ids / row order / external provenance is banned or at best a reviewer question). Consequently the true shift cannot be directly measured. **Be explicit about this.**

**Validation stack (all train-only):**

V1. **Primary: repeated stratified 5-fold CV on rows**, stratified by (horizon x cell-size-class). 2 split seeds for model selection, 3 for the final model. Rows with identical tokens are exchangeable given the features, so there is no row-level sibling leak through the features themselves (the only shared state across folds is the project-specific cell->target relationship). **Bias: optimistic**, because the same projects appear in train and validation folds and project-specific base rates leak through the cell statistics. Size unknown; I would guess that the *lift over the best constant* is overstated by up to ~2x and the absolute score by a few hundredths. The absolute level of the constant baseline is likely to transfer (it is what balanced metric rewards when features are uninformative) only if the pooled class-conditional shapes resemble the new projects (H3).

V2. **Stress CV A: leave-one-cue-profile-out** (group = `comment_cues` profile part). Measures how well the atom-level model handles unseen combinations. **Bias: pessimistic** (removes the whole cue value and the data behind it; also differs from the real shift axis).

V3. **Stress CV B: leave-one-length-bin-out and leave-one-style-out** (small, 5 + 4 folds). Same bias direction, checks the ordinal structure of `len_*`.

V4. **Prior-shift stress (evaluation-only reweighting of OOF).** Re-evaluate OOF predictions under alternative class mixes and under dropping the decode shrinkage; checks that the decode does not rely on the train class prior (it should not; balanced weights remove it).

V5. **Sealed holdout:** a random, stratified 15% of train, never used for selection, calibration or blend weights; scored once for the final config. Catches overfitting to the CV itself (random split, so it does NOT test shift).

**Metric implementation.** Re-implement `parse_days`, `horizon`, `grade` exactly as in the description. Unit tests: (i) perfect predictions (profile = true profile) -> 1.0; (ii) constant = true median -> a known value computed independently; (iii) reversed positions (`q' = 63 - q`) -> low; (iv) base-rate constant (the best balanced constant) -> equals D5(b); (v) the shipped sample (delta at 0) -> sanity value; (vi) regex must reject `1e-05` formatting and accept 6-decimal output; (vii) round-trip: emit a 2-point profile for `p*`, run `parse_days`, recover `days` within 1e-4 relative.

**Acceptance rules.** A change is accepted only if the paired fold-difference (same folds) beats 1 standard error AND is positive in >= 80% of (fold x repeat) pairs, AND it does not lose on stress CVs V2/V3. Ties go to the **simpler rung** (one-standard-error rule toward lower capacity), explicitly because the shift cannot be measured and low-capacity, strongly shrunk models transfer better. Every post-hoc selection step (model rung, weight decay, smoothing, temperature, shrinkage, decode rule) is evaluated **cross-fitted**: fit the constant on the OOF of folds != k and score fold k; the reported number is the cross-fitted one. The selection itself runs on split-seed A and the chosen config is re-scored on split-seeds B/C and the sealed holdout.

**Report:** per-horizon `B_0, B_1, B_2`, balanced score per fold, mean +/- std (and SE), for: shipped-sample constant, best constant, cross-fitted cell lookup, each model rung, final ensemble.

**Expected relation to the private score (estimate, low confidence).** Private will land at or below the V1 number; the gap is dominated by project shift, plausibly -0.02 to -0.08 absolute. Direction is pessimism on lift, not on the constant floor.

---

## Overfit/underfit risks

**Overfit risks**
- O1. *Few projects, many rows:* the model learns project-specific cell effects (cannot be validated). Mitigation: low-capacity rung preferred under the 1-SE rule; atom-level additive structure; weight decay; UNK-atom dropout; shrinkage of the predictive distribution toward the global balanced distribution (single lambda, cross-fitted, and chosen toward the more-shrunk end of the plateau); ensemble of seeds.
- O2. *Cell lookup memorisation* with a 64-way head: 64 outputs x ~100+ cells on 15k rows. Mitigation: low-rank/shared head (hidden width 16-32), ordinal smoothing of the target along positions (one knob sigma), weight decay, fixed epochs.
- O3. *Decode mode-flipping:* with a bimodal posterior and a mode-seeking utility, small calibration errors flip `p*` between modes across similar rows, producing high variance. Mitigation: average distributions across seeds/folds BEFORE decoding; temperature and shrinkage; fixed 0.25-position decode grid; compare against median decode and mean decode and keep the utility decode only if it wins beyond noise.
- O4. *Selection noise:* many grid configs on weak signal. Mitigation: <= 12 configs, 1-SE rule, re-score finalists on fresh split seeds + sealed holdout.
- O5. *Class prior baked into training:* if the model learned train priors, the decode would be mis-weighted for the balanced metric and for a project with a different prior. Mitigation: row weights `1/N_h` in the loss (below).
- O6. *Persistent-class censoring (H3):* the pooled persistent distribution is a mixture of train-project history caps. Unmitigable by data; only documented. Shrinkage toward the pooled balanced distribution is the default; do not try to hard-code a cap.

**Underfit risks**
- U1. Over-regularising to a constant (loses the small real lift). Mitigation: the ladder includes the additive log-linear rung and the interaction rungs; the OOF table shows (d)-(b) so I can see how much signal exists.
- U2. Wrong decode (mean instead of utility mode) leaves score on the table when the posterior is bimodal. Mitigation: three decode rules compared on OOF.
- U3. Losing the compositional structure (cue atoms, ordinal len) by treating each cell as an opaque class: unseen / rare cells then fall back to nothing. Mitigation: tokenise.
- U4. Loss mismatch: plain unweighted CE or MSE on `q` favours the persistent majority. Mitigation: balanced weights.
- U5. Ordinal neighbours ignored by a plain 64-way CE. Mitigation: ordinal smoothing and an expected-position / cumulative auxiliary loss.

---

## Recommended approach (primary + fallback)

### Primary: from-scratch tokenised neural profile generator, class-balanced loss, metric-aware decode

*Representation.* Tokenise each row into a short sequence: `[CLS]`, cue atoms (split `comment_cues` profile part on `+`; the `profile_general` backoff is its own token), band token, len token, style token (use `len_*` ordinal rank as an extra numeric input alongside its embedding), plus a learned `[UNK]`. Vocabulary built from TRAIN only; test atoms not in the vocabulary map to `[UNK]`. During training, replace each atom/shape token by `[UNK]` with fixed probability 0.05 (so `[UNK]` has a trained embedding; a structural simulation of rare/unseen cues, not synthetic data). Ignore `id` and `record_token`.

*Capacity ladder (stop at the lowest rung that wins under V1 + V2/V3, 1-SE rule):*
- R1: additive log-linear head (sum of token embeddings -> 64 logits). Few parameters.
- R2: R1 + pairwise interaction (concatenate embeddings -> small MLP, hidden 32, dropout) -> 64 logits.
- R3: tiny **encoder-decoder** transformer, d_model 32, 2 heads, 1 encoder layer over the token sequence, a decoder with 64 learned position queries cross-attending to the encoder, producing one logit per output position (non-autoregressive "generation" of the 64-step profile). This rung makes the model literally a sequence-to-sequence generator, which is the answer to the reviewer-reading question below; it is chosen only if it beats R1/R2 beyond noise.

*Output.* Softmax over the 64 positions = predicted profile distribution `q_x(j)`.

*Loss (metric-aware).* Per-row weight `w_i = 1/N_{h(i)}` (train horizon counts, normalised to mean 1) times a soft-label cross-entropy between `q_x` and the sidecar target (optionally Gaussian-smoothed along positions with fixed sigma chosen from {0, 2}), plus a small auxiliary weighted-L1 term on the decoded position `sum(j*q_x(j))` vs the true `q`. With these weights the model's `q_x` approximates the *balanced-prior* posterior `P(j | x)` under equal class weights, which is exactly the posterior the horizon-balanced metric asks to maximise utility under; this is invariant to the class prior and needs only the assumption that `P(x | horizon)` is stable across projects.

*Calibration (2 parameters, cross-fitted on OOF).* temperature `T` on logits and shrinkage `lambda` in `q <- (1-lambda)*q_x + lambda*q_bar` where `q_bar` is the balanced global distribution (train-only). Fit by weighted log loss on OOF; evaluated cross-fitted; choose `lambda` toward the shrunk end of the plateau (documented bias).

*Decode.* For each row: `p* = argmax over a fixed grid {0, 0.25, ..., 63} of sum_j q(j) * exp(-|p - j| / 4.796)`. Compare against (a) mean position and (b) weighted median of `q`, choose by cross-fitted OOF (a 3-way discrete choice). Then emit the 2-point interpolated profile at `p*` with 6-decimal formatting.

*Ensembling.* 5 folds x 3 seeds = 15 models; average the **distributions** (not logits, not positions) over models, then calibrate and decode. Test predictions come from the fold models (so the calibration constants, fitted on the OOF of those same fold models, transfer exactly). A full-data refit with the same fixed epochs is a measured alternative (use only if it improves on the sealed holdout beyond noise; expected to matter little at 15k rows with tiny models).

*Why it fits this data.* Inputs are a tiny compositional vocabulary with a weak but possibly real signal; the target is an ordered one-dimensional quantity with a Laplace-kernel, class-balanced metric. The design (a) uses a trained model as the sole source of per-row variation, (b) puts the class balancing in the loss and the kernel in the decode, and (c) keeps parameters few, which is the right bias when projects, not rows, are the effective sample.

### Fallback: scalar head with a metric-aware point loss

Same tokeniser and rung, but a scalar head trained with the **class-balanced weighted L1 loss on `log1p(days)`** (equivalently on position `q`; a convex surrogate whose minimiser is the balanced conditional median), or directly the balanced `1 - S` utility from a warm start. Predict `p`, emit the 2-point profile. Lower variance and no decode stage; use if the distributional primary shows mode-flipping or fails to beat the median/L1 variant beyond noise on V1/V2/V3.

### Optional diversity member (only after steps 1-4 are measured)
A categorical GBDT (CatBoost with `cat_features`, class-weighted multiclass over ~16 coarse position bins, expanded to 64 by uniform spreading) producing a distribution that is averaged with the neural distribution **only if** it is close in OOF quality and the blend gain exceeds noise. Compliance: tabular on categorical sketch features, neural generator stays dominant (>= 50% weight). If a reviewer says only the "sequence generation model" counts, drop it.

---

## Rejected options

- **Fine-tuning a pretrained text-to-text model (T5/BART) to emit the literal `lifecycle_profile=...` string.** Rejected: (i) the input is ~6 synthetic tokens; a pretrained LM has no relevant prior, so the extra capacity cannot help and overfits the few projects; (ii) free-form string generation risks regex-invalid outputs (a single bad row invalidates the submission); (iii) generating 64 numbers autoregressively is a high-variance route to what is one scalar for the grader; (iv) the "only supplied public files" sentence may forbid pretrained weights.
- **Per-cell lookup table of medians (or modes).** Rejected as the shipped core (it is a frequency table, the strip-the-ML failure mode, and it cannot handle unseen/rare cells). Kept ONLY as a diagnostic comparator (D5).
- **Predicting the mean position / plain MSE on `q`.** Rejected: mean-seeking estimator does not match a Laplace-kernel utility; pulls bimodal posteriors into the empty gap between short and persistent modes.
- **Unweighted CE / unweighted L1.** Rejected: optimises the train class prior, not the horizon-balanced metric.
- **Building CV folds from `record_token`, `id` order, row order or any inferred project identity.** Rejected: explicit ban / "ignore this field" (the description's bans apply to the held-out project; I extend the caution to train as a reviewer risk).
- **Pseudo-labelling, test-based vocabulary, test-based calibration, test-based prior estimation** (e.g. fitting the class prior on test predictions): banned by CLAUDE.md 2.3 #5.
- **Large ensembles / deep transformers / HPO with many trials.** Rejected: signal is weak; many knobs would select noise; effective sample = projects.
- **Hard-coding an offline-found "best constant" or per-cell medians.** Rejected: banned hard-coded tuned constants; the best constant must be recomputed in-script from train.
- **Using the persistent-class censoring cap as an engineered feature or a clipped decode.** Rejected: would require inferring project identity / history length (banned) and exploits the data-generation process.

---

## Fixed work plan & runtime budget

All counts fixed in code, no wall-clock branching, device `cuda`, seeds fixed, `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` set before importing torch, `DataLoader` replaced by direct tensor batching with a seeded `torch.Generator` (data is tiny; no workers). Time is logged only.

| Stage | Fixed plan | Est. A10G time (UNMEASURED, to profile) |
|---|---|---|
| Load, parse, join, build vocab (train only), horizon + weights | once | < 10 s |
| Metric unit tests + comparators (constants, cell lookup) | numpy | < 20 s |
| Grid selection (split-seed A, 5 folds) | 12 configs = {R1,R2,R3} x weight decay {1e-4,1e-2} x smoothing sigma {0,2}; EPOCHS = 60, batch 256, AdamW, lr 3e-3 cosine, 3 warm-up epochs, grad-clip 1.0; no validation-triggered early stopping | 60 runs x ~6 s ~ 6 min |
| Finalist re-score (top 3 on split-seeds B, C) | 3 x 5 x 2 = 30 runs | ~3 min |
| Stress CVs V2/V3 (chosen config) | ~15 runs | ~1.5 min |
| Final fold ensemble | 5 folds x 3 seeds = 15 runs, OOF + test distributions | ~2 min |
| Calibration (T, lambda), decode rule choice, cross-fitted check | numpy | < 30 s |
| Sealed holdout check, final validation, write CSV, re-read & re-validate | | < 30 s |
| Optional CatBoost member | 5 folds, fixed 300 iterations, `random_seed` fixed | ~1-2 min |

Estimated total **~15-20 min** (guess; to be confirmed by timing one epoch and one run locally, then hardcoding counts). Headroom far above 30% relative to the 50-min target. Memory: < 1 GB GPU, < 2 GB RAM. No OOM risk (batch 256, 64 logits, width <= 32).

Determinism hardening: the decode grid is 0.25-position steps so tiny float noise on GPU does not change `p*`; fold assignment, vocab ordering (sorted), and token-dropout masks use explicit seeded generators; averages are done in float64 on CPU; `PYTHONHASHSEED=0`; no `set` iteration order dependence (sort vocab before indexing). The script is run twice from a clean `working/` and the two CSVs diffed (should be byte-identical).

---

## Metric-aware training & decode

1. **Back-solve the metric.** Score depends only on the decoded scalar per row; horizon buckets average with equal weight; kernel `exp(-|dlog|/ln2)`. Targets are therefore decoded to `q` (position space) and `days` as in the description; the horizon of each train row comes from `days` with thresholds 90/365.
2. **Loss weights replicate the hierarchical averaging.** `w_i = 1/N_{h(i)}` (then normalised). Train count `N_h` stands in for the unknown test count; because the weight only needs to equalise class mass, the learned `q_x` is the balanced-prior posterior, and the metric's per-class averaging is matched.
3. **Expected-utility decode (composite/ kernel metric).** `p*(x) = argmax_p sum_j q_x(j) exp(-|p - j| * delta / ln2)`; enumerated on the fixed 0.25-grid (253 candidates x 64 bins: trivial cost). Kernel scale comes from the evaluator, not tuned.
4. **Decode alternatives compared (OOF, cross-fitted):** mean position, weighted median, utility argmax. Choose the best beyond noise; otherwise the simpler (median/mean). Also check robustness to kernel scale (ln2 vs 1.0, to cover the description/code discrepancy): the chosen decode should not change score materially.
5. **Proper-scoring calibration.** Train with soft CE (a proper scoring rule); initialise the output bias at `log q_bar`; fit `T` and `lambda` on OOF (weighted log loss), report cross-fitted. Add the ordinal auxiliary (expected-position L1) with a small fixed weight; optional cumulative "at-least-k" auxiliary heads at the two horizon boundaries (positions 31.2 and 40.8) to sharpen horizon discrimination (an auxiliary loss derived from the structure, not a decode rule).
6. **Output assembly.** `p*` -> `lo = floor(p*)`, `f = p* - lo`, vector zeros(64), `v[lo] = 1-f`, `v[min(lo+1,63)] += f`; format with `f"{x:.6f}"` joined by `;`, prefixed `lifecycle_profile=`. Verify by running the evaluator's `parse_days` on each emitted string and checking `|decoded_q - p*| < 1e-4`.
7. **Final validation function** (CLAUDE.md 5 plus the evaluator regex): columns `["id","prediction"]` equal to the sample's; row count 3,622; ids identical and in the same order as `sample_submission.csv`; no duplicate ids; every string matches the regex; all 64 values in [0,1]; positive mass; decoded days in [0,9000]; not constant across rows unless the model genuinely collapses (log how many distinct `p*` values; a collapse to a single value is a warning in the log, not a hidden fallback); re-read the written CSV with `keep_default_na=False` and re-validate.

---

## Structural signals

Each invariant is listed with how it is used and what to verify on train.
- S1. **Point-mass interpolation of the target** (hypothesis, verify D3): every target is a two-adjacent-position interpolation of a scalar. Use: reduce generation to predicting a distribution over `q` and emitting a 2-point profile; assert on every train row that decoding gives back the scalar.
- S2. **Ordinal geometry of positions and horizons:** neighbouring positions are similar; boundaries at 31.2 and 40.8. Use: Gaussian label smoothing along positions, expected-position auxiliary, cumulative at-least-k auxiliary heads at the boundaries.
- S3. **Compositional cue profile:** `profile_a+b` is a set of atoms; `profile_general` is a backoff bucket. Use: atom embeddings summed/attended so unseen combinations inherit atom effects; treat the backoff bucket as its own token (it is a mixture, so its prediction should be near the global distribution; the model will learn this; verify via per-bucket OOF).
- S4. **Ordinal length bin** (`tiny < short < medium < long < very_long`) and **token_band ~ length alignment:** feed the ordinal rank as a numeric input (hand-built feature to a trained model); allows smooth sharing across bins. Verify the monotone relationship in D2/D7 before leaning on it.
- S5. **Backoff buckets (`shape_other`, `profile_general`, `token_band_mixed`) are collision-heavy mixtures:** expect their predicted distributions to be wide; do not let the model treat them as informative cells (UNK dropout and weight decay help).
- S6. **Unseen-atom simulation:** token-level UNK dropout (fixed 0.05) is the in-training analogue of the held-out-project shift on rare cues.
- S7. **Class-prior invariance:** balanced weights make the decode independent of the train prior; only `P(x|horizon)` must transfer.
- S8. **Symmetry augmentations:** none apply (no geometric or permutation symmetry of the label). Token order is canonical (no meaning); the encoder should be permutation-invariant over cue atoms (sum/mean pooling or no positional encodings among atoms).
- S9. **No cross-row signals:** no group, id, or time structure is released that is legitimate to use; `record_token`/`id`/order are excluded by rule.

---

## Experiment roadmap

Stop criterion for every step: keep a change only if it beats 1 SE on paired folds and is positive in >= 80% of fold x repeat pairs, and does not lose on V2/V3; otherwise keep the simpler variant. Log table: id, change, CV mean +/- std, per-fold, B_0/B_1/B_2, stress-CV, runtime, public LB if submitted.

1. **Contract, metric, validation.** Implement the evaluator verbatim; run the seven unit tests; build D1-D10 in a scratch notebook; produce the comparator table (sample constant, best balanced constant, in-sample and cross-fitted cell lookup, atom-level lookup). *Stop:* metric tests pass; D3 confirms (or refutes) the 2-point target structure; the information ceiling D5 is known. If (d)-(b) is within noise, record that the model gain will be tiny and prioritise robustness and valid output over modelling effort.
2. **Strongest cheap baseline, end to end, valid.** Best balanced constant emitted as a valid profile through the final writer + validator. *Submit once* (credit 1) only after the validator passes: it establishes the floor and checks the output format on the platform.
3. **R1/R2 neural generator** with class-balanced loss and mean/median decode (no calibration). *Stop:* gain over the best constant > noise, else keep step 2's constant plus a documented reason (and reconsider whether a compliance-appropriate ML component is still load-bearing; ask reviewer).
4. **Metric-aware decode and calibration:** utility decode, `T`, `lambda`, ordinal smoothing, auxiliary losses. One change per experiment. *Stop:* gain > noise on V1 and not worse on V2/V3; otherwise revert to median decode.
5. **Capacity/diversity:** R3 enc-dec; optional CatBoost member; compare alone first, blend only members of close quality via distribution averaging. *Submit* the best single (credit 2) after sealed-holdout check.
6. **Bounded in-script selection:** the 12-config grid (fixed list, no timeout, no Optuna needed) with finalists re-scored on fresh split seeds + sealed holdout. *Stop:* choose the lowest-capacity config within 1 SE of the best.
7. **Final fixed-plan run:** clean `working/`, run the exact platform command twice, diff the two CSVs, validate, sanity-check that the distribution of decoded days looks like the OOF decode distribution (not constant, not collapsed to one horizon unless OOF did), then run the compliance audit. *Submit* (credit 3). Keep 3 credits unused for retries after transient check failures.

Do not tune on the public LB (2 projects, small, noisy); if CV and public disagree, trust CV unless a split mismatch is found.

---

## Compliance audit

CLAUDE.md 7 checklist and the strategist's B self-audits, applied to this plan:
- Test file used only for per-row inference: yes, tokenised with the train vocabulary (unknown tokens -> `[UNK]`); no vocabulary, scaler, prior, class balance, or calibration estimated from test; no cross-row normalisation over the test set; each test row's prediction depends only on its own tokens and the train-fit model.
- Wall-clock in conditions: none (time used only in `log`).
- `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks, `try/except` that change work: none planned; device fixed to `cuda`.
- Hard-coded constants: kernel scale `ln2`, thresholds 90/365 and cap 9000 come from the metric text; everything fitted (`T`, `lambda`, decode rule, architecture rung, weight decay, smoothing sigma, class weights, best constant) is derived in-script from train. Generic a-priori constants not searched in-script: UNK-dropout 0.05, lr 3e-3, EPOCHS 60, batch 256, hidden widths. They are generic defaults chosen before looking at any real submission result; documented in comments. No constant is taken from a real-leaderboard result.
- External/synthetic data, hosted weights, non-allowed libs: none; no pretrained weights; no downloads.
- Strip-the-ML: removal of the model leaves a single global balanced constant; the model is the only per-row signal. Honest disclosure of the measured lift is part of the report.
- Pure rule/regex solving: no regex beyond splitting tokens and the output-format check; no hand-written mapping from cues to durations.
- Sibling leakage: not applicable (target is not a relation among group-mates); `record_token` and `id` are never used.
- Project-identity inference: none from ids, order or files; the optional order audit D8 is for understanding only, never a feature or fold key, and can be skipped.
- Source readability: plain Python, < 512 KB, comments explaining reasoning, no blobs.
- Description-specific restrictions honoured: only public files; sidecar as sole supervision; `record_token` ignored; 64-step contract; no provenance lookups.
- Determinism: seeded everything, fixed epochs/folds/seeds, 0.25-grid decode, double run diff.

---

## Open questions & assumptions

**Reviewer questions**
1. *Is a non-autoregressive 64-position generator (R3 enc-dec with position queries; or the R1/R2 heads) acceptable as the required "sequence-generation model", given that the grader only uses the weighted mean position?* Reading A (yes): ship primary as planned. Reading B (must be a text-to-text / autoregressive generator): switch the primary to the R3 enc-dec generating the 64 positions and accept the small cost; if even that is rejected, a from-scratch tiny decoder-only transformer generating the two interpolation endpoints as tokens is the last resort (both stay from scratch and keep the same loss/decode).
2. *Is the metric-aware decode (utility argmax + two-point emission) acceptable?* The description sanctions emitting a scalar-interpolated profile and does not ban decode logic; CLAUDE.md says to distrust "decoding tricks are fine" boilerplate. Under a stricter reading, fall back to the weighted-L1 scalar head (training-time loss only), whose cost is expected to be small.
3. *Are pretrained weights allowed here ("using only the supplied public files")?* Planned compliant under both readings (from scratch). Only matters if a reviewer wants a pretrained backbone; I expect no gain.
4. *Is the train row-order / id-order audit (D8) acceptable?* Default: do not rely on it; skip if in doubt.
5. *The description's "factor-of-two = 0.5" vs the evaluator code's 0.368.* I follow the code; confirm the evaluator is the authority (the description says it is "equivalent to this complete code").

**Assumptions**
- A1. Runtime ceiling is CLAUDE.md's 50 min (none stated beyond A10G).
- A2. The `target` column format may or may not carry the `lifecycle_profile=` prefix; parser accepts both; output always carries it.
- A3. `sample_submission.csv` columns are `id, prediction` and its row order defines the output order; ids equal `test.csv` ids.
- A4. Class-conditional feature distributions `P(x | horizon)` are reasonably stable across projects (the balanced weighting relies on it); if not, nothing in the public data can fix it.
- A5. Train targets are two-point interpolations (verify D3); if not, the emitted profile must reproduce the decoded mean only (still valid, since only the mean matters).
- A6. Hidden test classes all have support; each `B_h` averages over a few hundred to thousands of rows.

**What I could not verify (all of it is unverified):** every data statistic (class shares, cell counts, target structure, signal strength, project count, censoring), the runtime numbers, the size of the CV-vs-private gap, and whether the task grader's description/code mismatch matters.

**Expected score (estimate, low confidence, no promise):** the best-constant floor probably sits around 0.28-0.38; a good, well-calibrated, shrunk model might add +0.01 to +0.06 in CV, and private will likely land 0.02-0.08 below the V1 CV figure because of project shift. Overall private range guess: ~0.27-0.42. The sample's constant (0 days) is probably ~0.10-0.15.
