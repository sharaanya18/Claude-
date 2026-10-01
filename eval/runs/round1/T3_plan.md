# Eris plan: T3 "Mark the moments that mattered"

Status of evidence: the ONLY inputs were CLAUDE.md, the strategist agent file and the paraphrased task text. No dataset was available. Every statement about the data below is a HYPOTHESIS (marked UNVERIFIED) plus the exact diagnostic that would confirm or refute it. Nothing here was measured. Numbers such as N, base rate and runtimes are assumptions or estimates and are labelled as such.

---

## Contract & decision unit

**One valid answer.** For each snippet row (id column + 24 columns `e_00..e_23`), 24 finite probabilities in [0,1], same columns/order/ids/row count as `sample_submission.csv`. Invalid = wrong columns/order/ids, NaN/inf, out of [0,1], crash, non-determinism flag. Merely low-scoring = anything else (including constant base rate, which scores exactly 0.01).

**Metric (paraphrased, exact variant UNVERIFIED).** score = 0.01 + 0.99*skill for skill >= 0, and 0.01*exp(skill) for skill < 0; skill = 1 - Brier(model) / Brier(base-rate predictor).
- The two branches are continuous at skill = 0 (both = 0.01). Predicting the base rate everywhere scores 0.01; perfect scores 1.0. Gains are linear in skill above 0, so every bit of Brier reduction counts equally; there is no cliff, but no credit for being "nearly" skilful either.
- Brier is a strictly proper score: the optimum per cell is the true posterior marginal P(e_t = 1 | inputs). There is no thresholding and no utility-over-output-set decode; **calibration is the entire "decode"**. Overconfidence is punished quadratically, so shrinkage toward base rate is the safe direction.
- Unknown details that change the loss weighting (see Open questions): (1) is the reference base rate one global scalar, one per column t, or one per row/episode_group; (2) is Brier averaged over all rows*24 cells, per column then across columns, or per episode_group then across groups ("Validation groups are given by episode_group" may mean hierarchical averaging). I will implement all variants in the local metric and report each; the primary training loss is plain per-cell BCE, which is optimal under all of them up to per-cell weights. If the metric is column-wise or group-wise, per-cell loss weights follow that averaging (weight_t = 1/Brier_ref_t, or 1/(n_cells in group)).

**True independent unit.** The `episode_group` (a whole game episode); snippets from one episode are NOT independent (overlapping or adjacent time ranges, same level, same palette, same layout). Rows are exchangeable only across groups. Within a row, the 24 steps are 4 windows of 6 steps; the 4 frames are the observable evidence.

**Pipeline stages (diagnosed separately).**
1. *Evidence coverage*: which of the 4 windows can have its events inferred from frames? Window k in {0,1,2} sits between frame_k (step 6k) and frame_{k+1} (step 6k+6): the pixel difference is direct evidence that something happened. Window 3 (steps 18..23) has no later frame: a pure forecast from frame_18, control code at 18 and the history of windows 0-2. Diagnose per-window skill; expect window 3 to be clearly the weakest.
2. *Window-level intensity*: P(any event in window k) and E[count in window k] (coarse, learnable from frame differences).
3. *Within-window timing*: which of the 6 steps. Information is thin (only a start-of-window control code and the two bracketing frames), so this is largely an irreducible-ambiguity stage; the target here is a well-calibrated near-flat-ish distribution over 6 steps shaped by learned step biases and by spatial cues (e.g. how far the agent travelled). Diagnose "window skill" vs "within-window skill" separately (oracle test: give the model the true window count and see how much timing skill remains).

**Information ceiling (hypothesis, UNVERIFIED).** If events are sparse (a few % of cells) and timing within a 6-step window is only weakly identifiable, then even a perfect window-level model caps skill well below 1 (roughly: skill_max ~ 1 - 1/6-ish dilution on timing for windows with exactly one event). Final skill plausibly 0.10-0.40.

---

## Compliance regime

**Domain.** Small-image (24x24 grey) multi-frame prediction = CV-flavoured, with a game-dynamics flavour. Not labelled fine-tuning or from-scratch in the paraphrase (UNVERIFIED; check the real label). Pretrained weights allowed, "training must happen in-script", no external data.

**Explicit bans/limits extracted:** no external data; training in-script; ~1 h on one GPU (CLAUDE.md target <= 50 min, I plan <= 35); different levels have different grey palettes (a robustness requirement, not a ban); window 3 must be forecast (no later frame).

**Allowed / grey / banned mapping (CLAUDE.md 2, 6.3):**
- Allowed and used: CNN trained from scratch on the provided frames; fine-tuning a pretrained timm CNN (weights from timm/HF only, pinned revision); per-snippet palette/intensity normalisation computed from the snippet's own frames; augmentation of real samples; ensembling several trained models; train-only OOF calibration fitted in-script.
- Grey: GBDT on hand-engineered summary features (diff counts, bounding boxes, ...) as a standalone model. Used only as an optional minor ensemble member, and the primary solution is a deep model so the strip-the-ML test fails for the rule parts. Hand-crafted maps (difference masks, persistence masks) are fed INTO the deep model as channels, which section 6.3 states is fine.
- Banned and explicitly avoided: any feature or statistic computed across test rows (including linking test snippets of the same episode by matching frames; see sibling leakage below), pseudo-labels, test-fitted palette maps or normalisers, rank/z-score normalisation across the test file, label-free prior estimation from the test file, hand-written rules that decide events (e.g. "diff pixels in region X => p = 0.8"), wall-clock branching, env-dependent fallbacks, tuned constants pasted from offline experiments.

**Ambiguities that change the approach (go to reviewer; plan under each reading):**
1. Are hand-engineered diff/persistence channels into a trained CNN fine? Reading A (fine, expected): use as channels. Reading B (reviewer wants raw frames only): drop the engineered channels; the CNN receives the raw (palette-normalised) frames of consecutive windows and can compute differences itself via its first conv layer. Cost: probably small (a conv can learn a difference), measured as an ablation in the roadmap; I will build the ablation in from the start so both are shippable.
2. Are control codes 0-14 provided only at steps 0/6/12/18 or at all 24 steps? Reading A (only at 4 steps): embed 4 codes. Reading B (all 24): feed the code sequence for every step (strongly informative about timing); the model design includes an optional per-step code embedding.
3. Is a per-snippet palette normalisation acceptable? It is a function of a row's own inputs only, so compliant under every reading; no train+test fit.

**Self-audits on this plan.**
- *Strip-the-ML*: remove the trained nets and nothing remains (no rule produces probabilities). PASS. The engineered channels alone are not a predictor.
- *No whole-test aggregation*: every normalisation is per-snippet; the base-rate prior is from train; calibration constants from train OOF. PASS by construction.
- *Sibling leakage*: potentially LARGE trap. If test snippets are consecutive chunks of the same episode, frame_0 of the next snippet equals the "missing" frame at step 24 for window 3 (and overlapping snippets share frames). Cross-referencing test rows to read future frames is banned (uses other test rows; transductive). The script must never join/compare test rows to each other. In TRAIN I may use such links only for group derivation and (optionally) as auxiliary training signal (see Structural signals).
- *Hard-coded constants derivable in-script*: only architecture/optimiser defaults are fixed constants (set before looking at results, kept generic); calibration temperature/biases, blend weights and any threshold-like constants are fitted in-script on cross-fitted train OOF.
- *Frozen features + small head*: not used. Pretrained arm is fully fine-tuned; the scratch model is fully trained; the training is load-bearing.

---

## Data findings

All items are UNVERIFIED hypotheses. For each: diagnostic to run on TRAIN ONLY, expected outcome. Test files: only schema, row count, id format, array sizes for runtime/memory.

**D1. Shapes, dtypes, schema.** Run: shapes of frames (expect N x 4 x 24 x 24 or N x 4 x 576, dtype uint8/float), codes (N x 4 ints in 0..14, or N x 24 if all steps given), columns e_00..e_23 in {0,1}, episode_group dtype, any level/palette id column. Hypothesis: no missing values; there may be an id-like level column (do NOT use it as a feature: fragile id).

**D2. Label prevalence.** Run: mean of e_t per t (24 values), overall base rate, fraction of rows with zero events, histogram of row event counts, per-window event counts (0,1,2,3+ for steps 0-5, 6-11, 12-17, 18-23). Expected: sparse (overall maybe 3-15%); per-step rates not flat, likely bumps at window-boundary steps (0,6,12,18) if frame/event alignment is off-by-one; count distribution overdispersed vs Poisson (events cluster). Decision impact: confirms need for per-step bias terms, aux count head, per-window calibration.

**D3. Event time structure.** Run: autocorrelation of e_t at lags 1-6 within rows; lag-6 correlation across windows; conditional P(e_t | e_{t-1}); are score changes always one-step isolated? Hypothesis: bursts (several adjacent events when a chain of digs happens); some periodicity if a game timer causes regular score changes (a strong constant-period pattern would be learnable from step bias alone, and would show up as a high-skill baseline with no image input: run a step-index-only predictor and measure its skill; this sets the floor and exposes any trivial structure that the model must at least match).

**D4. Frame-diff vs labels (the core signal).** Run: for windows 0-2, number of changed pixels between frame_k and frame_{k+1} (palette-invariant: pixel unequal) and its relation to the window's event count (AUC of "any event in window" vs changed-pixel count; mean count by changed-pixel quantile). Expected: strongly positive (digging changes cells; scoring events change cells). Also spatial extent of changes (bounding box area), number of connected changed components, whether changed pixels in some value class (e.g. cells turning from the most frequent grey to another) predict events. Hypothesis: window-level AUC of 0.8-0.95 from diff alone for windows 0-2. Also run the same with the "no change" case: P(event | zero change in window) should be low but not zero (events that leave no visible trace by step 6k+6, or reversible changes). If P(event | zero change) is large, the frames are not the whole story and the frame/step alignment must be re-examined (D5).

**D5. Alignment of frames and events.** Run: for each of steps 0, 6, 12, 18, compare event rate at that step vs neighbours; test whether the event at step 6k is visible in frame_k (already reflected) or only in frame_{k+1}, using changed-pixel counts conditioned on e_{6k}=1 versus e_{6k+5}=1 (restricted to rows where only one of those is 1). Hypothesis: off-by-one ambiguity; the step-specific logit biases (24 learned parameters) plus windows overlapping one step either side handle it. Decision: whether each window's encoder should also see the previous window's diff (for the step at the boundary).

**D6. Palettes.** Run: number of distinct grey values per frame and per snippet (expect <= ~8-12), the set of distinct palettes across train (cluster by sorted tuple of distinct greys), group counts per palette, and the number of episode_groups per palette; whether the palette is constant within an episode (expected yes), whether palette-to-role mapping is order-preserving (is the darkest always the wall? check using frequency ranks: the most frequent value per frame, the rarest value (agent) etc.). Hypothesis: palettes are arbitrary per level; raw grey values are not comparable across levels; within-frame frequency ranks and change patterns are. Decision: use palette-invariant channels (frequency-rank one-hots, difference/persistence masks) and palette augmentation; run a leave-one-palette-out stress check.

**D7. Episode and chain structure (also drives validation).** Run: rows per `episode_group` (min/median/max), number of groups, whether snippets within a group overlap or abut. Test by exact hashing: for every pair of train snippets in the same group, does frame at step 6 of one equal frame at step 0 of another (stride 6 overlap), or frame at step 24 equal frame 0 of the next (stride 24 abutment)? Also compare labels on overlapping steps (should be identical if the data is consistent; a disagreement rate quantifies label noise or the alignment error). Also check cross-group exact frame duplicates (levels start from identical initial frames: that would mean groups share content, so group CV by `episode_group` alone could leak). Build union-find groups by exact/near-exact frame matching across rows (hash of frame bytes after palette normalisation, so different palettes of the same layout also merge) and compare to `episode_group`.

**D8. Duplicates and ambiguity.** Run: count of rows with identical (frames, codes) and the label disagreement among them (irreducible noise estimate), and identical frame pairs (k, k+1) with different window event counts (information ceiling for the window-level model).

**D9. Control codes.** Run: distribution of the 15 codes at each frame step; relation of code at step 6k to the window's change pattern (e.g. which codes are no-ops: windows with zero diff), whether some codes imply movement direction (compare agent displacement, found as the rare-valued pixel, to code), whether codes are dominated by a few values. Hypothesis: codes encode direction/action combos; embedding learned; geometry augmentation requires the code remapping, which is unknown and not allowed unless recovered from train labels/diffs and verified on every train snippet (see Structural signals).

**D10. Information ceiling per window.** Run: gradient-boosted or logistic fit on the 3-4 simple palette-invariant scalars (changed-pixel count, number of components, code, step index) for the window-level target, as an oracle-ish ceiling for what a minimal feature set explains; and a "count-known" oracle (given the true window count, distribute across 6 steps with learned step biases) to bound timing skill. Hypothesis: window-level skill carries most of the achievable skill; timing adds little.

**D11. Window 3 forecastability.** Run: skill of predicting window-3 events from (frame_18, code_18, window 0-2 diff statistics and event counts-from-diff). Hypothesis: the history features (recent change rate, whether the agent is in a "digging phase") give modest skill; expect window-3 skill to be a fraction of windows 0-2's.

**D12. Target noise / sanity.** Run: a shuffled-label control and a label-time-shuffled control to make sure the pipeline returns skill ~0; a check for any leak columns (columns derived from the score).

---

## Validation design

**Reproducing the test split (from the description only).** "Validation groups are given by `episode_group`" implies train/validation (and presumably test) are separated by episode, i.e. a GROUP split with whole episodes held out. Train has "several thousand snippets" (N assumed 3-8k, UNVERIFIED), so the number of groups may be modest (hundreds?). The deployment shift to cover: unseen episodes, probably unseen layouts, possibly unseen palettes ("different levels use different palettes"; whether test levels are new is not stated and cannot be inferred from test distributions, so the validation must stress both).

**Scheme.**
1. Build `grp = union-find(episode_group, frame-hash links from D7)`; every split holds out whole `grp` components.
2. Primary CV: StratifiedGroupKFold(5) on `grp`, stratified by "row has any event" and by palette id (derived), repeated over 2 split seeds (shuffle of group order). Report mean +- std of per-fold score and skill, per-window skill (4 numbers), per-step Brier, global/columnwise/group-wise metric variants.
3. Stress check (secondary, train only): leave-palette-out (GroupKFold over derived palette clusters, 4-5 folds) for the primary model alone, to measure palette-shift fragility. Accept palette-normalisation choices only if they hold up there.
4. Holdout sanity: 1 of the 5 groups-folds (per split seed fixed) is kept as a final-untouched fold: used only for the last comparison of the final recipe vs the baseline; calibration/blend constants are cross-fitted so they never see the fold they are evaluated on.
5. Metric: re-implement exactly (all averaging variants, both branches of the formula), unit-test on (a) perfect predictions -> skill 1, score 1.0; (b) base-rate predictions -> skill 0, score 0.01; (c) constant non-base-rate predictions -> negative skill, score in (0, 0.01); (d) reversed (1-y) predictions -> strongly negative; (e) a single-cell perturbation checks monotonicity; (f) the base rate used in the reference computed from the evaluated labels vs from train (report both; the real metric's choice is an open question).
6. Accept a change only if it beats noise: mean gain > 1 s.e. of per-fold means AND consistent in sign across both split seeds and in >= 4/5 folds. Post-hoc steps (blend weights, temperature/bias, shrinkage) are each cross-fitted (fit on 4 folds' OOF, apply to the 5th).
7. Expectation management: a self-built proxy usually over-estimates the private score (private test may contain new palettes/levels); plan for private skill to land below CV.

---

## Overfit/underfit risks

**Overfitting (with mitigations).**
- *Few groups / correlated snippets*: leak via same-episode neighbours inflates CV and trains the net to memorise layouts. Mitigation: union-find groups (D7), group CV, palette augmentation, small capacity (~1M parameters), dropout, weight decay, EMA of weights, fixed epoch count (no validation-triggered stopping needed).
- *Palette/level memorisation*: raw grey values act as a level id. Mitigation: palette-invariant input channels, random bijective recolouring on the grey channel plus channel dropout of the raw grey channel, leave-palette-out stress check. Never use a level/palette-id column.
- *Many tuned knobs*: kept to near zero. Architecture/optimiser defaults fixed before any scoring; fitted-in-script scalars are low-dimensional (temperature, 4 window biases, 1 shrinkage, <= 2 blend weights).
- *Selection on reported OOF*: nested/cross-fitted calibration and blend; a final untouched fold.
- *Overconfident probabilities on a Brier metric*: BCE with weight decay + EMA + temperature/shrinkage calibration; no focal loss, no label smoothing, no mixup (all distort calibration).
- *Window 3 overfit*: forecasting from one frame invites memorising layout-specific futures. Mitigation: per-window calibration/shrinkage (window-3 bias/temperature fitted separately), history features, small head for window 3.

**Underfitting (with mitigations).**
- *Too little spatial resolution/capacity*: 24x24 inputs must not be pooled down early. Mitigation: stride-1 first block, downsample only after two conv blocks (24 -> 12 -> 6), keep final 6x6 map with attention/flatten pooling not global-mean pooling only (the position of change relative to the agent matters).
- *Information thrown away by normalisation*: rank/frequency normalisation could drop role information. Mitigation: feed both normalised grey and frequency-rank one-hot, plus difference/persistence masks; the ablation (engineered vs raw) is measured.
- *Loss mismatch*: BCE vs Brier; both tried (roadmap step 4); the metric is Brier but BCE gives sharper gradients.
- *Window 3 and timing under-modelled*: aux count head, step-bias terms, cross-window token interaction.
- *Pretrained backbone mismatch*: tiny synthetic frames; the pretrained arm is only for diversity and scored alone first.

---

## Recommended approach (primary + fallback)

**Primary (A): window-token CNN with cross-window transformer, trained from scratch, palette-invariant.**
- *Input per snippet (all derived from the snippet's own frames):* 4 frames; per frame channels: (i) grey value rescaled to [0,1] by the snippet's own min/max (random recolouring + dropout of this channel during training), (ii) one-hot of the frame's value index by within-snippet frequency rank (K=8 cap, the rarest values collapsed), (iii) for window k pairs: |f_{k+1} != f_k| mask, signed direction of change in rank space, (iv) persistence mask across all 4 frames (cells never changing), (v) "forecast" flag channel for window 3 (where only frame_3 is available, difference channels are zero and the flag is 1). Codes: embedding(15, 16) of the code at each frame step (plus per-step code embeddings if all 24 are provided).
- *Window encoder:* shared CNN stem: conv3x3(C_in -> 32) x2, downsample to 12x12 (64 ch), 2 blocks, downsample to 6x6 (96 ch), BatchNorm/GroupNorm, GELU; spatial pooling by flatten of a 1x1-reduced 6x6 map (position-sensitive) + global max/mean; output 128-d window token. Separate stem copy or learned flag for window 3.
- *Cross-window mixing:* 4 window tokens + code embedding + window position embedding -> 2-layer transformer encoder (d=128, 4 heads, dropout 0.1). Cross-window context gives window 3 its history.
- *Heads:* per-window 6 logits (+ 24 learned step biases) = 24 outputs; auxiliary heads: window event count (classes 0,1,2,3+), any-event-in-window; auxiliary losses weight 0.2 each.
- *Loss:* per-cell BCE averaged per the metric's hierarchy (per-cell weights if column-/group-wise averaging).
- *Training:* AdamW (lr 2e-3 peak, wd 1e-2), 1-cycle cosine, 30 epochs, batch 128, bf16 autocast, grad clip 1.0, EMA(0.999) weights for prediction; 5 folds x 3 seeds (15 models); test predicted per snippet by averaging logits over the 15 models (no TTA unless D9 verifies a symmetry); optional per-snippet palette-recolour TTA (a function of the single row only).
- *Post:* logit average -> cross-fitted calibration: p = sigmoid((z / T) + b_window), then optional shrink toward train per-step base rate lambda; all fitted in-script on OOF.
- *Why it fits:* the signal is a spatial change between consecutive frames (CNN with positional awareness), with timing/forecast context across windows (token mixing), palette differences handled by invariant channels.

**Secondary (B, diversity of assumption): fine-tuned pretrained timm CNN.** The same input stack mapped to 3 channels per window pair (grey_k, grey_{k+1}, diff), upsampled to 96x96 (nearest) and passed through a pinned-revision pretrained small timm backbone (resnet18 class; convnext_nano only if budget allows) fully fine-tuned (discriminative LR: backbone 3e-4, head 2e-3), same token-transformer head and losses; 5 folds x 1 seed x 12 epochs. Different inductive bias (pretrained texture/edge prior, larger receptive field). Included in the blend only if its OOF skill is within ~1 s.e. of A; blend in logit space with <= 2 weights fitted on OOF, cross-fitted.

**Optional member (C, grey): LightGBM in long format** (one row per snippet x step; features = step index, window index, step-in-window, code, palette-invariant diff summaries, A's OOF window logits as a feature if the data prove useful). Only if it beats noise as a blend member; flagged grey because tabular-on-image-summaries is discretionary. Not required for the plan.

**Fallback (F):** A with the transformer removed (per-window encoder + MLP head fed with the previous window's token and the code) and per-window calibration; or, if the architecture is unstable, A with a plain 3-conv-layer encoder and 30-epoch BCE. Both keep the palette-invariant channels. If even that crashes, the plan has no constant-output fallback shipped silently (a constant base rate scores 0.01); the script should fail loudly rather than silently degrade.

---

## Rejected options

1. **Rule-based "diff => probability" model** (e.g. changed-pixel thresholds): fails the strip-the-ML test and is hand-coding the data-generating process.
2. **GBDT on raw pixels**: raw-pixel tabular model is an immediate rejection (CLAUDE.md 6.3).
3. **Frozen pretrained features + head**: grey, and tiny synthetic frames make ImageNet features weak; not load-bearing.
4. **Large pretrained backbones (ViT-B, convnext-base) on upsampled 24x24 frames**: slow, high variance on a few thousand correlated snippets, unlikely to beat a small scratch CNN; the small pretrained arm B covers the diversity.
5. **Linking test snippets across rows to see the frame at step 24 / overlapping frames**: banned (cross-row test use, sibling leakage, transductive).
6. **Pseudo-labelling / test-time adaptation / test-fitted palette normalisation**: banned.
7. **Geometry augmentation (flip/rotate) without a verified code remap and symmetric dynamics**: digging games usually have gravity; flips would fabricate invalid dynamics. Only permitted if D9 verifies on every train snippet (see Structural signals).
8. **Focal loss, label smoothing, mixup/CutMix**: harm calibration, which is the metric. Excluded.
9. **Big ensembles / many architectures / heavy Optuna HPO**: with few groups, HPO would select noise; roadmap allows a bounded in-script check only if CV can resolve the effect.
10. **Sequence model over predicted events as a stacked second stage with a free-form decoder**: Brier needs marginals; joint decoding gives no gain.
11. **Using level/palette id columns or row order as features**: fragile ids.
12. **Time-based early stopping, time-dependent fallbacks**: banned by the determinism checker.

---

## Fixed work plan & runtime budget

Assumptions (UNVERIFIED): N_train ~ 3-8k snippets, 4x24x24 uint8; all data lives on the GPU as tensors; no DataLoader workers (`num_workers=0`, augmentation done on the GPU with a seeded `torch.Generator`); device fixed to "cuda"; one deterministic code path; no `torch.cuda.is_available()`, no `cpu_count`.

| Stage | Fixed plan | Estimated A10G time (estimate, not measured) |
|---|---|---|
| Load, validate schema, build palette-invariant tensors, union-find groups | single pass | < 1 min |
| Local metric + unit tests inside the script (asserts) | tiny | < 10 s |
| Model A (scratch CNN + tokens) | 5 folds x 3 seeds, 30 epochs, batch 128, bf16, EMA | ~8-15 min (~1-2 s/epoch on ~5k samples x 4 windows) |
| Model B (pretrained timm, 96x96 upsampled) | 5 folds x 1 seed, 12 epochs, batch 64 | ~4-8 min |
| Optional C (LightGBM long format, GPU/CPU fixed) | 5 folds, fixed 400 rounds | ~1-2 min |
| Cross-fitted calibration + blend weights (OOF only) | fixed small optimiser, fixed iterations | < 1 min |
| Test inference (fold models averaged; fixed order) | 15 + 5 models | ~1 min |
| Validation (schema, ids, finite, [0,1], re-read written CSV with keep_default_na=False) | assert | < 10 s |
| **Total** | | **~15-28 min estimated; hard worst case < 40 min; headroom >= 30% vs 50 min target** |

Memory: model + activations < 4 GB GPU at batch 128 (A) / 64 (B at 96x96); host RAM < 4 GB. Timing is logged only; no branch depends on the clock. Seeds fixed at top (`random`, `numpy`, `torch`, `cuda`, `PYTHONHASHSEED`, generators); `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch; avoid scatter/index_add nondeterministic ops (use one-hot matmul for frequency-rank encoding and fixed-shape tensors). The final model is the fold-ensemble (no retraining on 100% data) so that the number of training steps never depends on data-dependent selection; the roadmap step 7 may test a full-data refit only if the CV shows the extra data helps by more than noise.

---

## Metric-aware training & decode

(i) *Back-solve the metric:* skill = 1 - B/B_ref. Everything is linear in the per-cell squared error; B_ref depends only on labels (base rate). So minimising per-cell squared error or BCE is aligned. The two-branch score has no gating beyond continuity at skill 0.
(ii) *Hierarchical averaging:* if the metric averages per column or per episode_group, apply identical weights in the loss (per-cell weights; group-weights = 1/n_cells_in_group) and in the OOF scorer. Verify equality of ordering of models under both weightings.
(iii) *Ordinal targets / counts:* window event count (0,1,2,3+) is an ordinal auxiliary target; consider cumulative at-least-k heads for the aux count only (not for the final probability).
(iv) *Utility decode:* none; Brier optimum is the calibrated marginal. The only decode is calibration: p_t = sigmoid(z_t / T + b_{window(t)}) then p <- (1 - lambda) p + lambda * base_t with T, 4 biases, lambda (6 scalars) fitted on OOF with cross-fitting. Fit on a log-loss or Brier objective with bounded ranges (T in [0.5, 3], lambda in [0, 0.5], clip p to [1e-4, 1 - 1e-4]) so degenerate corners are unreachable.
(v) *Closed-form thresholds:* not applicable (no threshold).
(vi) *Span/run terms:* none. Adjacent-step correlation is irrelevant to the marginal Brier, but the consistency "sum of p over window ~ E[count]" is enforced softly through the aux count head.
(vii) *Constraints:* probabilities in [0,1]; per-window bounds: P(any event in window) >= max_t p_t and <= sum_t p_t (add a mild penalty or check at OOF; it holds automatically if p comes from one coherent model).
(viii) *Overdispersed counts:* if D2 shows overdispersion, the aux count head is a categorical (not Poisson) so no extra machinery.
(ix) *Taxonomies:* none known.

---

## Structural signals

Invariants the data-generating process probably guarantees (all UNVERIFIED; each has a train-only check):
1. **Static cells**: cells that never change over the 4 frames (walls, undug dirt) carry no event evidence; the persistence mask channel removes the burden of learning this. Check: changed-pixel fraction (D4).
2. **Temporal ordering of windows**: windows 0..2 are directly observed, window 3 is the forecast; position embeddings and the forecast flag encode this. Check: per-window skill (D11).
3. **Palette invariance**: relabelling grey values consistently across the 4 frames of a snippet leaves labels unchanged. Implement as augmentation (random bijective recolour, applied identically to all frames) and as normalised channels. Check: D6 plus the leave-palette-out stress check.
4. **Frame overlap / chain structure across train snippets (if D7 confirms)**: (a) derive groups, (b) validate label alignment on overlapping steps, (c) optional auxiliary targets: for window 3, a teacher model that also sees the next snippet's frame_0 (the true frame at step 24, available only in TRAIN through chain links) can generate soft targets or a representation target for a student that does not (privileged-information distillation, train-only, real labelled data, compliant). Include only if D7 shows reliable chains AND the plain model's window-3 skill is a bottleneck; otherwise skip (complexity).
5. **Code semantics**: if codes are direction-like (D9), agent displacement between frames should correlate with codes; an auxiliary head predicting the displacement is extra supervision derived from train frames. Only if D9 validates; skip otherwise.
6. **Geometry symmetry**: allowed only if (a) the game is verified mirror-symmetric (check train snippets under horizontal flip with a code swap that is consistent with agent displacement on EVERY train snippet), (b) the metric is unaffected. Otherwise no geometry augmentation.
7. **Step-in-window structure**: 24 learnable step biases and a within-window positional embedding; events at window boundaries (0,6,12,18) may behave specially (D5).
8. **Count/any-event consistency** across the 6 steps of a window (aux heads).

---

## Experiment roadmap

Each step has a stop criterion; stop when the gain is below noise (1 s.e., not consistent across seeds/folds).

0. **Diagnostics D1-D12** on train (no models yet except the tiny ones in D3/D10). Stop: produce the table of per-step base rates, per-window diff-vs-label AUCs, palette count, chain structure, code semantics. Decide: ambiguity questions to the reviewer in parallel.
1. **Contract, metric, validation**: metric (all variants) + unit tests; group construction; fold generation; validator for the submission; a step-index-only baseline and a base-rate baseline (skill ~0, score 0.01). Stop when tests pass and the baselines give their expected values.
2. **Cheap strong baseline, end-to-end valid**: per-window scratch CNN (no transformer, no engineered extras beyond difference mask), BCE, 5-fold group CV, 1 seed, 20 epochs; write a valid submission CSV through the final validator. Record per-window skill. Stop criterion: valid file + CV skill well above 0 (if not, return to D4/D5: alignment bug).
3. **Representation and structure**: (a) add palette-invariant channels and recolour augmentation (compare against raw grey; palette-stress check), (b) persistence mask, (c) code embeddings, (d) cross-window transformer + forecast flag (check window-3 skill), (e) aux count heads, (f) raw-only vs engineered ablation (for reading B of the compliance question). One change at a time, 2 split seeds each.
4. **Metric-aware loss and decode**: BCE vs Brier vs BCE+Brier; metric-hierarchy weights; cross-fitted calibration (T, window biases, shrinkage). Stop when gains stop beating noise.
5. **Diversity**: model B (pretrained timm); optional C; blend in logit space with cross-fitted weights; keep only members within ~1 s.e. of the best and adding > 1 s.e. gain in the blend. Test privileged-information distillation (Structural signals 4) only if the window-3 bottleneck is confirmed.
6. **Bounded in-script HPO** only if CV can resolve it (probably skip): at most 6-8 fixed trials over 2-3 knobs (width, dropout, lr), seeded, no timeout, with a separate held-out check (final untouched fold); otherwise keep defaults.
7. **Final fixed-plan run** from a clean `working/`, run twice and diff the output (expect byte-identical or ~1e-6 differences under deterministic mode); check the output distribution against OOF (mean per step vs train base rate, window means, std of probabilities), run the exact submission validator, confirm runtime and headroom. Use credits only for: baseline (step 2), best single model, ensemble, final.

Convergence rule: techniques that independent solvers would likely all use (group CV by derived groups, palette-invariant inputs, per-window calibration, cross-window context) get tested first, before any proxy-driven micro-tuning.

---

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits applied to the plan:
- Test used only for one-snippet-at-a-time inference: PASS (per-snippet normalisation; no cross-row joins; no episode_group use from test; calibration from train OOF).
- No time-dependent branches (`time` used only in logging): PASS by design.
- No env-dependent fallbacks (`cuda.is_available`, `cpu_count`, try/except imports): PASS by design; fixed device, workers=0.
- Hard-coded tuned constants: PASS if the scalars (T, biases, shrinkage, blend weights) are fitted in-script; architecture/optimiser defaults are set before results and kept generic. Watch: do not paste values from earlier real submissions.
- External data / synthetic data / self-hosted weights / non-allowed libs: PASS (timm/HF pinned revision only; torch, numpy, pandas, sklearn, lightgbm in the allowed stack). Recolour/crop augmentations transform real samples, not fabricated data. Privileged-information distillation uses only real train frames.
- Strip-the-ML test: PASS (no rule path produces predictions).
- Inference-only/frozen-feature solution: PASS (all models trained/fine-tuned in-script; training is load-bearing).
- Source readability: plain, commented, < 512 KB; one plain code path; no encoded blobs.
- Challenge-specific: ~1 h limit honoured (plan <= 35 min worst case); pretrained weights only from allowed hubs; no external data.
- Sibling leakage: explicitly forbidden to link test rows; in train, chain links used only for grouping/aux.
- Determinism: fixed seeds, fixed epochs/folds, deterministic kernels, no scatter ops; run twice and diff.
- Grey flagged items for the reviewer: optional GBDT member C; the engineered difference/persistence channels (reading A vs B).

---

## Open questions & assumptions

**Questions for a reviewer / challenge author**
1. Exact metric: reference base rate (global, per column, per group; computed from test labels or train?) and the averaging order (cell-wise, column-wise, per episode_group)? Plan under each: BCE with matching per-cell weights; calibrated shrinkage toward the train per-step base rate in all cases.
2. Are control codes provided only at steps 0/6/12/18 or for all 24 steps? Plan under each: 4-code embedding vs per-step code sequence (large timing gain if all 24 are given).
3. Does a frame at step 6k reflect the effect of the event at step 6k (alignment)? The plan handles either via learned step biases and the D5 diagnostic.
4. Are test episodes from new levels/palettes? Plan: palette-invariant design regardless (cannot be checked from test distributions).
5. Are hand-engineered diff/persistence channels into a trained CNN acceptable, or must raw frames be the only input? Plan: ablation prepared for both.
6. Is a small scratch CNN acceptable when the category mentions pretrained weights? (Expected yes: genuine training in-script; B covers the pretrained route.)
7. Is the task labelled fine-tuning or from-scratch? If from-scratch: drop model B entirely (no pretrained weights).

**Assumptions (UNVERIFIED)**
- N_train is a few thousand (3-8k); 24 grey levels at most per frame; events are sparse (a few percent of cells); frames are stored as arrays directly in the public files (no image decoding).
- Windows 0-2 are largely identifiable from frame differences; window 3 is mostly unpredictable beyond history and base rate.
- Train snippets from one episode share a palette; snippets of an episode may overlap or abut (to be tested by D7).
- Level ids or palette ids may be shipped as columns; they will not be used as features.

**What I could not verify:** every data claim (sizes, base rate, palette structure, chain structure, code semantics, alignment, achievable skill), the exact metric averaging, the runtime numbers (estimates from typical A10G throughput on tiny inputs, not profiled), and whether chain links exist in train.

**Expected private score (ESTIMATE ONLY, wide uncertainty):** skill ~0.10-0.40, i.e. a score of ~0.11-0.40, central guess ~0.2-0.25. Reasoning: window-level evidence for windows 0-2 is probably strong, but 6-step timing ambiguity and the unforecastable window 3 cap the per-cell skill; the proxy CV likely over-estimates the private result if unseen palettes/levels appear. A result at the base-rate floor (0.01) signals a bug (alignment or leakage control), not a hard problem.
