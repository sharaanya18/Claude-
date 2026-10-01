# Eris plan: T3 "Mark the moments that mattered"

Status of evidence: only the paraphrased description was read. No dataset was available. Everything under "Data findings" is a diagnostic to run on TRAIN plus the hypothesis I expect, and is UNVERIFIED. Column names other than `e_00..e_23` and `episode_group` are assumptions (written `<...>`). No code is in this plan.

## Contract & decision unit

**One valid answer.** For each test row (snippet), 24 probabilities `e_00..e_23`, each in [0,1], finite, same row order/ids/columns as `sample_submission.csv`. Invalid (scores dead last): wrong columns/order/row count, NaN/inf, out-of-range probabilities. Merely low-scoring: constant base rate (skill 0, final score 0.01) or worse than base rate (skill < 0, score 0.01*exp(skill), near zero).

**Metric, term by term (as described).**
- Per-cell Brier error (p - e)^2 pooled over rows x 24 steps (assumption: pooled mean; verify the description for per-column or per-row hierarchical averaging).
- skill = 1 - BS_model / BS_base, where BS_base is the Brier of a constant base-rate predictor (assumption: train base rate; could be pooled or per-column; implement all variants and train on the pooled one).
- score = 0.01 + 0.99*skill for skill >= 0, and 0.01*exp(skill) for skill < 0. Linear in skill above 0, so maximise expected Brier skill; calibration-sensitive (proper scoring rule, not rank-only). Risk is asymmetric: negative skill is nearly worthless, so a base-rate shrinkage safety (below) is cheap insurance.
- No gating term is described. The score-change event is sparse (assumed few % per step), so most of the gain is the information about HOW MANY events happened in each 6-step window (visible as a frame difference) and WHERE in the window they fell (controls, timing).

**True independent unit.** The episode (`episode_group`), not the snippet and certainly not the step. 24 steps of one snippet are strongly dependent; snippets from one episode likely share layout, palette, and strategy. Effective sample size is the number of episodes (unknown: check).

**Pipeline stages (diagnose separately).**
1. Window-level evidence: does the frame pair (F_k, F_{k+1}) reveal how many events happened in steps 6k..6k+5? Oracle check: relation of true per-window count K_w to frame-diff statistics.
2. Within-window timing: given K_w, which of the 6 steps? Evidence is the controls (lagged: the action at t-1 changes the state seen at t) and position priors. Oracle ladder: (a) per-slot base rate, (b) true K_w with uniform timing, (c) true K_w plus control-conditioned timing, (d) perfect.
3. Forecast window (steps 18..23): only F_3 and (if available) the controls at those steps. Skill here will be far lower; slot-specific calibration matters.
4. Decode: output the marginal event probabilities (expected-utility optimal for Brier is the posterior marginal; no thresholding, no argmax).

## Compliance regime

**Domain.** Computer-vision-flavoured sequence/event prediction on tiny synthetic frames with a trained in-script model. Not NLP, not retrieval. Applicable CLAUDE.md playbook: 6.3 (never raw pixels to a tabular model; train a CNN; hand features only alongside the deep model; TTA allowed) plus the determinism rules of section 3 and the test-set rules of 2.3 #5.

**Explicit restrictions from the description.** Training must happen in-script; no external data; one GPU, about 1 hour (plan for 30 min). Pretrained weights are allowed but not required. Images are tiny synthetic frames, so a from-scratch small CNN is allowed (it is not a "from-scratch challenge" with a ban on pretrained weights, and not a Fine-tuning-labelled challenge as far as described; confirm the label).

**Allowed / grey / banned map for this plan.**
- Allowed: small CNN trained from scratch on the provided frames; embeddings of control codes; palette canonicalisation and palette augmentation done per row; flip augmentation with learned control remapping (real labelled units transformed, not synthetic data); TTA by flip; fixed-seed multi-seed ensembles; grouped CV; per-slot recalibration fitted on OOF.
- Grey: a GBDT on engineered window features (frame-diff counts, control counts) as a minor ensemble member (tabular-only on hand features, grey for CV category); a frozen pretrained backbone arm. Both are kept out of the primary.
- Banned (never planned): any statistic, normalisation, clustering, or palette grouping fitted on test rows; using other test rows from the same episode to fill the "missing" F_4 or to share context (cross-row pooling of TEST); pseudo-labels; synthetic frames; hard-coded rules such as "if control==k then event" or "events = changed cells"; constants copied from earlier real submissions or offline searches; time-dependent control flow.

**Strip-the-ML test.** Remove the trained network and what remains is a base rate. A deliberately rule-based ablation ("count changed cells / 6 per step", "control code lookup") is run only as a diagnostic and must be clearly beaten by the trained model; if the rule ablation is within noise of the full model, the model is not load-bearing and the plan must change (more learned structure, not more rules). Hand-built inputs (changed-cell mask, per-row canonicalised palette channels) are fed INTO the trained network, not used as predictors.

**Self-audits.**
- No whole-test aggregation: every prediction is a function of its own row plus train-fit parameters. Palette canonicalisation uses only that row's own four frames.
- Sibling leakage: if the target of snippet i (events) is a relation to frames, the frames of the SAME snippet are legitimate inputs (they are the contract). Frames of OTHER snippets are not. In train, consecutive snippets of one episode could provide F_4 (frame at step 24) as an auxiliary supervised target only if the diagnostic below confirms adjacency; at test time this is unavailable and must not be recreated from other test rows.
- Every constant (epochs, weight decay, dropout, K cap, calibration parameters) is either fixed a priori or derived by an in-script train-only search with its own cross-fitted check.

## Data findings

All items UNVERIFIED. For each: diagnostic on TRAIN, expected finding, decision it drives.

D1. Schema. Count control columns: 4 (steps 0,6,12,18) or 24 (every step)? Image columns: 4 x 576 or stored as arrays? Is `episode_group` present in test (it is the validation group; treat as a grouping tool only, never as a feature, never for cross-row pooling)? Expectation: the phrase "control codes at those steps" can be read as 4 controls or 24 controls; I expect 24 per row (otherwise timing is nearly unobservable). Decision: model takes a control sequence with an availability mask, so either reading works.

D2. Row counts and groups. Rows, number of distinct `episode_group` values, rows per group (min/median/max), group-size imbalance. Hypothesis: several thousand rows, perhaps dozens to a few hundred episodes. Decision: if groups < ~100, climb the capacity ladder conservatively (small CNN, strong weight decay), more CV repeats.

D3. Target statistics. Exact base rate overall (%), per slot t (24 values), per window (4 values), per snippet count of events (histogram 0,1,2,3+), fraction of snippets with zero events, fraction of all-zero windows. Hypothesis: sparse (a few % per step), a few events per snippet on average, heavy tail on counts, possibly slot-position effects at j=0 and j=5 (frame boundaries). Decision: K cap (use the 99th percentile in-script), whether a count x timing head is warranted, and the size of the Brier headroom.

D4. Information ceilings (oracle, train OOF-free closed form): skill of (a) per-slot base rate, (b) per-window rate given TRUE K_w and uniform timing within the 6 steps, (c) true K_w plus control-conditioned timing (empirical tables of e_t versus control at t, t-1, t-2 if controls are per-step), (d) separately for windows 0-2 and the forecast window 3. Hypothesis: (b) captures a large share of the attainable skill; the forecast window has far lower ceiling. Decision: effort allocation (frame evidence for windows 0-2 versus control-driven timing for window 3).

D5. Frame alignment and off-by-one. For windows 0-2, compute the number of changed cells between F_k and F_{k+1} (after per-row canonicalisation) and correlate with (i) events in steps 6k..6k+5, (ii) events in 6k+1..6k+6, (iii) events in 6k-1..6k+4. Hypothesis: one of these shifts is clearly best (a frame captured before or after its step's action). Decision: which boundary slots need neighbour-window context; whether slot 6, 12, 18 events are explained by the preceding or following diff.

D6. Control semantics. Cross-tab e_t against control code at t and t-1 (and t-2); frequency of each of the 15 codes; empirical movement vector per code by regressing the displacement of the changed region between frames on code counts (least squares on train); detect which code(s) coincide with events (e.g. a "dig" action). Hypothesis: a few codes are directional moves, one or two are actions, some are no-ops; the event rate is sharply code- and lag-dependent. Decision: embedding with lag-1 and lag-0, and whether horizontal flip is a valid symmetry with a learned code mapping (flip maps code c to code c' with reflected movement vector).

D7. Palette structure. Per row: the set of distinct grey values over its 4 frames, count of distinct values per row, number of distinct palette signatures across train, relation of signatures to `episode_group` (do episodes map to levels, and how many levels?), pixel-class frequency ranks within a row. Hypothesis: a handful of levels with distinct palettes, 4-8 grey values per palette, frequency rank (most common = background/dirt) more stable across levels than raw grey. Decision: input channels = per-row min-max intensity plus frequency-rank one-hot plus changed mask; palette augmentation by random monotone grey remap; leave-palette-out stress validation if levels are few.

D8. Spatial localisation of evidence. Per-pixel frequency of change within windows, and per-pixel correlation of change with the window event count; look specifically for a fixed-position region (a score readout or HUD) whose changes track events, and for non-uniform spatial priors (borders, fixed floor). Hypothesis: events correlate with changes in the interior (dug or collected cells); there may be a fixed score digit/bar region. Decision: add coordinate channels (CoordConv x, y) so an absolute readout is reachable while the conv trunk stays translation-equivariant; keep log-sum-exp plus mean pooling.

D9. Count evidence. Join frame-diff statistics (number of changed cells, number of cells that changed to or from each frequency-rank class) with true K_w. Hypothesis: K_w is closely related to the number of cells that change class (collected items vanish) but with noise from non-scoring digs, and from movement of the player sprite. Decision: ordinal count head versus per-slot head; auxiliary loss on count; whether the count x timing factorisation has headroom.

D10. Within-window event dependence. Within a window, the joint distribution of e patterns: P(adjacent events), P(two events | K=2), run lengths, sum-of-slots constraints; compare joint pattern frequencies to independent-slot predictions. Hypothesis: events cluster (adjacent steps) more than independence implies, and the count K_w is far more concentrated than a product of Bernoullis. Decision: whether an exact 64-configuration joint head (with an adjacency parameter) beats independent sigmoid heads.

D11. Duplicates and overlap. Hash each frame; look for identical frames across snippets (blank initial frames, static states), for pairs where F_3 of one snippet equals F_0 of another (consecutive snippets), and for (F_2,F_3) of one equal to (F_0,F_1) of another (overlapping windows). Hypothesis: snippets are cut from longer episodes, possibly overlapping or consecutive, so identical frames leak across random folds. Decision: derive own groups by union-find on non-generic frame hashes (frames occurring in <=2 snippets; blank or reset frames excluded so everything does not merge) on top of `episode_group`; whether an auxiliary F_4 target can be built from consecutive TRAIN snippets (train-only auxiliary, never used at test).

D12. Identical inputs with different labels. Group rows by (all four canonicalised frames, controls) and measure label disagreement; irreducible noise floor. Decision: realistic skill ceiling and when to stop tuning.

D13. Forecast window. Skill of the simplest controls-and-F_3 model on slots 18-23 versus slots 0-17; correlation of window-3 count with window-2 count (momentum) and with F_3 content. Hypothesis: window 3 is much harder; part of it is predictable from the same-episode momentum (K_2, F_2 to F_3 diff) which is available inside the row. Decision: give window 3 the context embedding of window 2 and its own calibration.

D14. Leakage suspects. Any column that is only known after the outcome (score columns, cumulative score, step index of last event, episode time offsets). Hypothesis: none, but a cumulative score or episode offset column would be a leak or a non-generalising feature. Decision: drop or never feed `episode_group`, row id, file order.

## Validation design

**Reconstructing the test split.** The description says validation groups are given by `episode_group`, so the test is assumed to be episode-disjoint from train. Mirror it: group-held-out folds; never random row folds.

**Folds.** Groups = `episode_group` merged by union-find with content overlap (D11: shared non-generic frame hashes, identical frame pairs). 5 folds of whole merged groups, stratified by group event-rate bins and palette signature so every fold has the same palette mix, repeated over 3 split seeds (random group permutation). Report mean and std over fold scores and over seeds; paired comparisons on identical folds only. A change is accepted only if it beats the paired-fold noise (more than 1 standard error of the paired fold differences, consistent in sign on most folds and on all seeds).

**Stress split.** If levels (palette signatures) are few (say <= 8) and differ in test (the description says different levels use different palettes), add a leave-palette-out check: hold out an entire palette signature. Test score is likely between the random-group and leave-palette-out figures; report both, optimise the group split.

**Metric.** Re-implement the exact formula (pooled and per-column base-rate variants) and unit-test: perfect predictions give skill 1 and score 1.0; constant base rate gives skill 0 and score 0.01; the reversed (1 - e) predictions give a clearly negative skill and score 0.01*exp(skill); constant 0.5 gives negative skill when the base rate is low; base-rate-plus-noise is near 0. Per-fold, also report skill separately for windows 0-2 and window 3, and per slot.

**Post-hoc steps need their own held-out check.** Per-slot recalibration, shrinkage factor toward the base rate, ensemble weights, and any hyperparameter selection are fit on the OOF predictions of the other folds and applied to the held-out fold (cross-fitted), or nested. The headline CV number is the cross-fitted one. Expect the real private score to land below the CV proxy (the proxy is optimistic since groups and palettes in test may be new).

## Overfit/underfit risks

| Risk | Where | Mitigation |
|---|---|---|
| Memorising episodes or levels (few groups, same layout and palette repeated) | Overfit | Grouped CV; small CNN (a few 10k parameters), dropout, weight decay chosen per grouped CV, palette augmentation, flip augmentation with learned control remap, fixed short schedule with EMA, no `episode_group` or id features |
| Palette shortcut: model identifies level from grey values and learns level-specific rates | Overfit | Per-row canonicalisation (frequency-rank one-hot plus min-max intensity), random monotone palette remap, leave-palette-out stress check |
| Selecting on noisy small-group CV (many knobs) | Overfit | Few learned potentials; at most ~8 grid configs, fixed; cross-fitted calibration; paired comparisons across 3 seeds |
| Label-derived leakage via overlapping snippets across folds | Overfit | Union-find regrouping (D11) |
| Brier skill < 0 in forecast window pulling down overall | Overfit/underfit | Slot-specific calibration with shrinkage toward the slot base rate, fitted cross-fitted; verify skill >= 0 on every window |
| Too little capacity or downsampled detail: 24x24 is tiny, stride early or heavy pooling erases single-cell changes | Underfit | No early striding, full-resolution conv trunk, 3-4 conv layers of 32-64 channels, changed-cell mask as an explicit channel |
| Loss mismatch: BCE sparse optimum differs from Brier | Underfit | Train on a proper score matching the metric (BCE or squared error, compared by CV), recalibrate per slot; initialise output bias at the slot base rate |
| Missing causal alignment: action at t-1 explains change at t | Underfit | Lag-0 and lag-1 control embeddings; boundary slots see neighbour windows (D5) |
| Forecast window uses a head trained on the other windows' evidence distribution | Underfit | Separate window-3 stack with a "no later frame" indicator and virtual zero channels (equivalent to a masked F_4), plus its own calibration |
| Ceiling from irreducible ambiguity (D12) | Both | Stop tuning when the OOF gap to the oracle ladder (D4) is within noise |

## Recommended approach (primary + fallback)

**Primary: lean shared CNN over frame pairs, with a count x timing head and exact marginals.**

Inputs per row (all built from that row's own data):
- Four frames canonicalised per row: min-max intensity, frequency-rank one-hot (top ~8 ranks over the row's own 4 frames; rank is a within-row statistic), and for every adjacent pair the changed mask (F_{k+1} != F_k). Coordinate channels (x, y) added as an option decided in the roadmap (D8).
- Windows 0-2: stack [F_k, F_{k+1}, changed mask, coordinate]. Window 3 (forecast): [F_3, zero channels for the missing next frame, an indicator channel set to 1] so the same trunk handles it with a distinct flag.
- Controls: embedding (dim 8) of each of the 15 codes; per step the head receives emb(c_t), emb(c_{t-1}) and the slot position j within the window (and window id). If only 4 controls exist, the mask zeroes the rest and the slot position embedding carries the timing prior.

Encoder: shared 3-4 layer conv trunk, full resolution, GroupNorm, GELU, then two pooled readouts (log-sum-exp over locations, and mean) giving window embedding z_w (dim ~64). The change in the per-window evidence is local, so scoring every location and pooling keeps the head translation-equivariant (design principle 5). Neighbour context: slots at window boundaries also receive z_{w-1} or z_{w+1} (as determined by D5), and window 3 receives z_2.

Head (explicit latent structure, few learned parameters): for each window, enumerate the 2^6 = 64 event patterns S. Energy(S) = sum_{t in S} phi_t + gamma * (number of adjacent pairs in S) + psi_{|S|}(z_w), where phi_t is a linear function of [emb(c_t), emb(c_{t-1}), slot embedding] (6 per-step potentials per window), gamma is one scalar, and psi_k(z_w) is a small linear map producing one logit per count k in 0..Kmax (Kmax in-script from the train count distribution). Normalise by a softmax over the 64 patterns. Train by exact negative log-likelihood of the observed 6-bit window pattern (the labels are observed, so the marginal likelihood is exact), plus optional BCE on the exact marginals P(e_t = 1) = sum over S containing t. Output marginals are the probabilities. Learned parameters are limited to the CNN trunk, the control/slot embeddings, phi, gamma, psi: state them in the code header for the strip-the-ML audit.

Auxiliary losses (follow from the structure): per-window any-event BCE, per-window count (ordinal at-least-k), and, only if D11 confirms consecutive train snippets, next-frame changed-cell count for window 3 as a train-only auxiliary.

Regularisation per output: weight decay, dropout, and epoch count chosen by the exact Brier metric under grouped CV from a wide fixed grid including the strong end, separately for windows 0-2 versus window 3 (different information content). Fixed schedule (cosine, EMA), no validation-triggered stopping.

Calibration: tiny per-slot logistic recalibration (a_t + b_t * logit) on cross-fitted OOF probabilities; shrinkage toward the slot base rate with a single factor if the cross-fitted skill of a window is not above 0. Report cross-fitted.

Training/shipping: 5-fold OOF x 2 split seeds produce OOF predictions and the calibration; final prediction = refit on 100% of train with fixed epochs, 5 seeds averaged in probability space, with flip TTA (if D6 validates the mapping), then apply the calibration fitted on OOF. Refit used because the CNN is small and the epoch count is fixed (verify OOF-calibration transfer to the refit models on a held-out fold once).

**Fallback (same trunk, simpler head).** Independent per-step sigmoid heads (BCE or Brier) on [z_w, z_neighbour, emb(c_t), emb(c_{t-1}), slot embedding], the same auxiliary count loss, same calibration. Used if the factorised head does not beat it by more than the paired CV noise (primary-design gate: the lean design that encodes the structural insights goes first; the 64-pattern head is kept only if measured).

**Diversity member (only if it earns a measured gain).** A LightGBM per (row, step) on engineered features: changed-cell counts per class transition, control counts per window, lag controls, slot and window ids, window-2 momentum, plus the CNN's OOF window count as a feature (cross-fitted). Grey for CV category; used only as a minor blend member, with a rule-ablation check that it is not carrying the solution.

Expected score (an estimate, not a promise): final score about 0.2 to 0.5 (central guess ~0.3), i.e. Brier skill roughly 0.2-0.5 overall; windows 0-2 plausibly skill 0.3-0.6, window 3 plausibly 0.0-0.15. Reasoning: sparse events leave timing uncertainty inside each window, the forecast window has no frame evidence, and a self-built CV is usually optimistic. Low confidence because no data was seen.

## Rejected options

1. **Fine-tuned ImageNet backbone (ResNet/ConvNeXt on upsampled 24x24 frames).** Domain mismatch (tiny synthetic palettes), costly upsampling, high capacity relative to few episodes (full fine-tunes memorise groups). A frozen pretrained arm may be tested once as a diversity check in the roadmap; not primary.
2. **Rule-based counting (changed cells / 6 per step, control lookup).** Fails the strip-the-ML test; used only as a diagnostic ablation.
3. **Transformer over 24 step tokens plus patch tokens.** Heavier than the evidence supports; enters the roadmap only if the lean model is measured and a cross-window gap is shown.
4. **Using neighbouring test snippets of the same episode** (to recover F_4, or to share context for the forecast window). Cross-row pooling of test rows: banned.
5. **Palette clustering or level identification fitted on train+test** (to canonicalise palettes). Banned; per-row canonicalisation replaces it.
6. **Pseudo-labelling or self-training on test.** Banned.
7. **GBDT-only on engineered features.** Grey/likely rejected for the CV category; kept only as a minor member.
8. **Time-boxed or validation-triggered stopping, wall-clock budgets.** Rejected by the determinism check.
9. **Sampling decoders, latent mixtures across windows.** Violate the primary-design gate; the 64-pattern per-window enumeration is the largest structure used.
10. **Large HPO sweeps with Optuna.** A tiny fixed grid with cross-fitted evaluation is enough for what CV can resolve.

## Fixed work plan & runtime budget

Assumptions (unverified): about 5,000 train rows, 24x24 frames, small CNN (~100k parameters or fewer), batch 256, A10G. Runtime figures are estimates to be replaced by in-sandbox profiling of one epoch.

| Stage | Fixed plan | Estimate (A10G) |
|---|---|---|
| Load, validate schema, canonicalise frames, build groups (union-find) | CPU, vectorised | 1 min |
| Capacity/regularisation grid | 6 fixed configs x 3-fold grouped, 12 epochs each, window-0-2 and window-3 selection separately | 5 min |
| OOF run | 5 folds x 2 split seeds, 30 epochs each, cosine schedule, EMA | 10 min |
| Cross-fitted calibration and shrinkage | CPU, per-slot logistic with 48 parameters | < 1 min |
| Optional GBDT member | 5 folds, fixed rounds | 3 min |
| Final refit on 100% | 5 seeds, same 30 epochs, flip TTA predict | 6 min |
| Validation and write | Re-read CSV, asserts | < 1 min |
| Total | | about 25-30 min, >= 30% headroom against a ~45 min target; memory under 4 GB |

All counts (folds, seeds, epochs, grid, batch, workers, threads) are hard-coded constants; seeds set for random, numpy, torch, cuda, DataLoader generators; cudnn deterministic, benchmark off; `torch.use_deterministic_algorithms(True, warn_only=True)`; device fixed to `cuda`; time used only for logging; no `try/except` fallbacks that change work; submission validated against `sample_submission.csv` before writing (columns, order, row count, finite, in [0,1]); run twice and diff.

## Metric-aware training & decode

- Back-solve the metric. Brier skill is monotone in the pooled squared error, so train on a proper scoring rule over all 24 x rows cells with equal per-cell weight (the same as the metric unless D-level reading shows per-column or per-row hierarchical averaging; in that case replicate the averaging in the loss weights).
- Loss: exact negative log-likelihood of the window pattern (primary head) plus BCE on marginals; compare with direct squared-error on marginals by CV; keep what wins within noise. Biases initialised at the slot base rate.
- Per-slot calibration: p' = sigmoid(a_t + b_t * logit(p)), cross-fitted; slot-group tying (window x position) if per-slot parameters are noisy.
- Safety shrinkage: if a window's cross-fitted skill is <= 0, replace that window's predictions by a convex combination with the slot base rate (single factor lambda in [0,1] chosen on the OOF of other folds), because the metric penalises negative skill exponentially.
- Decode: output the posterior marginals. No thresholding or argmax; expected Brier is minimised by the marginal. Clip to [0,1].
- Base-rate reference: compute skill under pooled and per-column base rates and report both (D-metric unknown definition); the model objective is identical.
- Overdispersion: not needed for pattern probabilities; the count head already has a full categorical distribution over K (no beta-binomial required).

## Structural signals

1. **Frame-pair causality.** Events in steps 6w..6w+5 shape the change between F_w and F_{w+1}; the stack uses the changed mask as an explicit input (D5 decides the exact boundary shift).
2. **Control lag.** The action at t-1 explains the event at t; emb(c_t) and emb(c_{t-1}) are both inputs; sequence ends get a zero virtual control with an availability flag.
3. **Count x timing decomposition.** Window count from frame evidence, timing from controls and slot position, joined by exact enumeration of the 64 patterns.
4. **Forecast-only window.** Separate stack with "no later frame" indicator, context from window 2, own calibration; treated as a forecast, not as an observation window.
5. **Palette invariance.** Per-row canonicalisation plus random monotone remap augmentation; frequency-rank one-hot; the changed mask is palette-free.
6. **Symmetry (measure, do not assume).** From train, estimate each control code's movement vector by least squares of frame-to-frame displacement on per-window code counts; build a code permutation for horizontal flip (and vertical or rotations only if the game has no gravity-like asymmetry; check by a train diagnostic that a flipped-and-remapped sample has similar event rates and OOF skill). Assert on every train snippet that the mapping is consistent; use matching flip TTA at inference. If the diagnostic fails, drop that flip.
7. **Local evidence, translation-equivariant head.** Score every location, pool by log-sum-exp and mean; coordinate channels only added to expose a fixed HUD region (D8).
8. **Boundary relationship across windows.** Slots at j=0 and j=5 see adjacent-window embeddings (D5).
9. **Hidden episode structure (train-only).** If D11 shows consecutive train snippets, F_4 (next frame) gives a train-only auxiliary target for window 3; never reproduced at test.
10. **Dependence structure inside the window.** The adjacency potential gamma and the count prior psi_k capture clustering of events (D10).

## Experiment roadmap

1. **Contract, metric, validation (stop when unit tests pass).** Schema checks (D1-D3), metric re-implementation with unit tests (perfect, reversed, constant, base-rate), union-find regrouping (D11), folds x 3 seeds. Stop criterion: fold balance verified, metric tests pass.
2. **Cheap valid baseline.** Per-slot base rate (skill 0, score 0.01) written end-to-end through the validator; then per-slot rate by control code and lag, and a LightGBM on engineered window features (diagnostic and possible member). Record the oracle ladder (D4). Stop when the full pipeline writes a valid CSV and the oracle ladder is known.
3. **Representation and structure.** Small CNN with independent heads (the fallback), palette canonicalisation versus raw grey, coordinate channels on/off, changed mask on/off, control lag on/off, neighbour context on/off, window-3 stack. One change per experiment, paired on the same folds and 3 seeds; keep a change only if it beats noise.
4. **Metric-aware loss and decode.** Primary 64-pattern head versus independent heads; auxiliary count loss; per-slot calibration cross-fitted; shrinkage. Keep the factorised head only if its paired gain is above noise (primary-design gate).
5. **Diversity.** Flip augmentation with learned control remap (after D6 validation); frozen pretrained arm test (one backbone, one scale) and the GBDT member; blend only when a member's solo quality is close to the CNN and the blend gain beats noise; blending in probability space with a single non-negative weight chosen on cross-fitted OOF.
6. **Bounded HPO inside script.** The fixed 6-config grid per window group (weight decay, dropout, channels) on 3-fold grouped CV; check the chosen config on the held-out fold (nested) before shipping.
7. **Final fixed-plan run.** From a clean working dir with the platform command, run twice and diff (expect near-identical probabilities), check distribution of the output versus OOF (mean per slot close to the slot base rate, no constant outputs), runtime headroom, compliance audit. Use credits only for baseline, best single model, ensemble, final.

Test a technique that independent solvers would plausibly converge on (frame diff plus control lag into a small CNN with per-slot calibration) before any proxy-driven tweak.

## Compliance audit

CLAUDE.md section 7 checklist against this plan:
- Test file read only for one-sample inference: yes (per-row canonicalisation; no test statistics, no test clustering, no test vocabularies, no test-based normalisation).
- Time in any condition: none planned (logging only).
- `cuda.is_available`, `cpu_count`, import fallbacks: none; device fixed to `cuda`.
- Hard-coded tuned constants: none; grid/calibration/K cap derived in-script from train; flip mapping measured in-script from train; everything else fixed a priori (epochs, folds).
- External data, synthetic data, self-hosted weights, banned library: none; only numpy/pandas/torch/sklearn/lightgbm; optional pretrained arm from timm only if tested and only as a diversity member.
- Model removal (strip test): the trained network is the predictor; rule ablation is diagnostic only; parameters are few and enumerated (conv trunk, control/slot embeddings, phi, gamma, psi, calibration).
- Source readable and < 512 KB; comments explain reasoning: planned.
- Challenge-specific restrictions honoured: in-script training, no external data, ~1 hour: yes.
- Section B audits: no whole-test aggregation (palette canonicalisation is per row); sibling leakage avoided (no other-row frames at test); every constant derivable from a train-only in-script search.
- Remaining grey items: hand-built frame-diff mask and palette canonicalisation features (fed into a trained CNN; standard, kept); GBDT member (grey, optional); the "exploits data generation" risk is low because nothing about the generator is hard-coded.

## Open questions & assumptions

Reviewer questions (each with a plan under both readings):
1. Are controls provided for every step or only the four frame steps? Plan handles both with a mask; if only four, timing is mostly a learned slot prior and expected skill is lower.
2. Is `episode_group` available in test? If yes, it may be used only for grouping validation, never as a feature or for pooling across test rows; confirm. If not, nothing changes.
3. Is the base rate in the skill formula pooled or per column, taken from train or test? Plan: implement both, optimise the same loss; calibration and shrinkage are base-rate-agnostic because they are fit on train OOF.
4. Is a from-scratch small CNN acceptable given that pretrained weights are "allowed but images are tiny synthetic frames"? Reading A: acceptable (plan as written). Reading B: a pretrained trunk is expected; fallback is an LP-FT of a small pretrained backbone on upsampled stacks, with the parameter-change norm and CV gain logged.
5. Are per-row hand-derived inputs (changed-cell mask, palette frequency ranks) acceptable? Plan: they are inputs to the trained network, with the strip-the-ML ablation as evidence; cost of dropping them would be measured in CV (expected to be large).
6. Does the frame at step t reflect the state before or after step t's action (alignment)? Resolved empirically by D5; both readings are covered by boundary-slot neighbour context.

Assumptions to verify: event sparsity and counts (D3), number of groups and levels (D2, D7), column names, whether consecutive snippets exist in train (D11), the validity of horizontal-flip symmetry (D6), exact metric aggregation, and every runtime number (profile one epoch on the A10G before fixing counts).
