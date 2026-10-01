# Eris plan: U22 Align Contact Episodes Across Human and Robot Manipulations

Status of evidence. No dataset files were available to this run. Everything labelled UNVERIFIED is a hypothesis or an estimate to be checked with the listed diagnostics before it is relied on. The only numbers computed here come from a pure-Python simulation of the metric on synthetic posteriors (no data, see "Metric-aware training & decode"). Source read: the challenge text, CLAUDE.md, and the strategist agent file. Nothing else was opened.

## Contract & decision unit

**One valid answer (per test id).** A string of exactly 81 finite decimals, space separated, flattened as `27*i + 9*j + 3*k + l` (G4 fastest), each in [0,1], sum 3 (tol 0.02), every candidate marginal on every axis 1 (tol 0.03). File: `submission.csv`, columns `id,predicted_coupling`, 252 rows, rows aligned by id. Invalid row (wrong count, NaN, out of range, mass or marginal violation) scores 0 but stays in OverallMean and Bottom20Mean. No abstention.

**Validity by construction (key design choice).** Every legal hard assignment is a valid tensor (3 cells equal to 1, one per hyperedge). Any convex mixture of the 216 hard-assignment tensors is therefore automatically valid: sums to 3, marginals exactly 1, values in [0,1]. The model outputs a probability vector over the 216 assignments (fix the first group's order, permute the other three: 6^3) and the submitted tensor is the posterior-mean tensor. This makes structural invalidity impossible and removes any need for Sinkhorn or post-hoc projection. The validator still re-checks tolerance on every row before writing.

**Metric, term by term (final_score).**
- RowScore = 0.45 CouplingSkill + 0.30 PairMarginalSkill + 0.25 MassSkill. CouplingSkill is 1 - MSE(P,Y)/MSE(U,Y) over 81 cells. PairMarginalSkill is the mean over the 6 axis pairs of the same ratio on 3x3 pair marginals (gives partial credit for getting two embodiments right). MassSkill is linear in sum(P*Y): (TrueMass - 1/9)/(3 - 1/9).
- final = 0.72 OverallMean + 0.18 Bottom20Mean + 0.10 ExactSkill. Bottom20Mean is the mean of the lowest ceil(0.2*N) RowScores (N = 117 private rows gives the lowest 24). ExactSkill is the exact-hypermatch rate over chance 1/216, times min(1, OverallMean/0.05). The grader decodes the argmax over 216 assignments of the sum of the three cells of P, so exactness depends on the argmax of P-cell sums.
- Calibration-sensitive: the first two skills are squared-error, so overconfident wrong rows are punished and also drag Bottom20. MassSkill alone rewards sharpening. Net: the right output is close to the posterior mean tensor with a modest, OOF-fitted sharpness.
- Perfect scores 1; uniform scores 0 (clipped); a wrong-but-confident row can score 0 (clipping), never negative.

**True independent unit.** Not the row. Rows from one physical object are different time slices of the same object (about 9 rows per object: 1287 rows / 143 objects), each frame used once. The leakage units are object and environment: train has 115 objects in 11 environments; test has 28 objects in 5 other environments; leaderboard visibility splits test by whole environment (public 135 rows / 2 envs, private 117 rows / 3 envs). Effective sample size for the cross-embodiment mapping is on the order of 11 environments / 115 objects, not 1035.

**Pipeline stages, to be diagnosed separately.**
1. Candidate coverage: complete by construction (all 216 assignments are enumerated, nothing can be missed). The metric-contract check is the gold-score oracle (feed gold one-hot potentials, expect RowScore 1 and exact 1 on every train row).
2. Local scoring: learned clip embeddings give pair potentials for 6 embodiment pairs; diagnose by per-pair top-1 accuracy (chance 1/3) and per-pair AUC under grouped folds.
3. Decoding: exact enumeration (216 or 648 states) into posterior-mean tensor, then one-scalar sharpening fitted on OOF. Diagnose with posterior calibration (reliability of P(true config) mass) and the metric.

## Compliance regime

**Domain.** Labelled "Fine-Tuning" (CV-like: 80x80 two-frame RGB clips plus force/tactile). CLAUDE.md section 6.7 applies strictly: genuine fine-tuning of a pretrained backbone must be load-bearing in the shipped path. Frozen-embedding plus small head or GBDT only is grey and likely rejected. The LP-FT rung (linear-probe-initialised head, then unfreeze last blocks with a tiny LR) is the compliance layer and must be in the shipped predictions, not an optional extra.

**Explicit bans and hard constraints in the description.**
1. Do not use anonymous ids, `tensor_row`, CSV row order, or group aliases (G1..G4) as target signals.
2. Do not assume a fixed candidate position has a fixed interaction phase.
3. Do not assume G1 is a fixed embodiment.
4. Do not recover or join targets through source filenames, timestamps, locations, object numbers.
5. Do not submit independent per-axis rankings that violate the four-way marginal constraints.
6. No JSON, text explanations, extra columns; exact 2-column CSV.
7. Hard output constraints (81 numbers, bounds, mass/marginal tolerances).
8. Runtime limit 60 min on one A10G, 10 CPU cores, 62.5 GiB RAM.

How each is honoured:
- (1) and (3): the model never receives the alias index, candidate index, tensor_row, or id. `tensor_row` is used only as an array index to fetch the row's own tensors (the description says tensor order is sorted by anonymous id independently of CSV order; assert `tensor_row` is in range and that it matches the row, nothing more). Embodiment identity is read from `group_cards[*].view_kind`, which the description publishes as a legitimate per-row feature; groups are canonically sorted by `view_kind` (fixed order: external_human_view, wrist_human_view, robot_gripper_view, instrumented_gripper_view) and the output tensor is transposed back to the row's G1..G4 order. This is not "using G1 as a fixed embodiment"; it is the opposite (removing alias dependence). Assumption: reading `view_kind` is allowed because it is a documented feature column. Reviewer question Q3.
- (2): the architecture is permutation-equivariant over candidates by construction; a unit test shuffles candidates and groups within a row and asserts the output tensor permutes identically.
- (5): the output is a posterior over globally consistent assignments, never per-axis scores.

**Where the description is silent, and my assumptions.**
- Pretrained weights: not mentioned beyond "fine-tuning". Assume HF/timm weights only, per CLAUDE.md section 2/8 (no GitHub-hosted robot-representation weights such as R3M or VC-1 unless the same weights are hosted on the HF Hub, and then flag to a reviewer).
- External data and synthetic data: not mentioned. Assume banned per CLAUDE.md (only challenge data plus backbone weights).
- Model size or family caps: none stated. Assume none, but the shipped model must stay inside 60 min.
- Test-time use: assume strictly one-row inference. No cross-row pooling over test rows, no per-object or per-environment clustering of test rows, no test normalisation, no test-fitted PCA. Even though test rows from one object share appearance and could be tied together, that is sibling/whole-test aggregation and is excluded.
- Ensembling and TTA: not mentioned. Assume allowed (TTA is per-sample).
- Augmentation: time-reversal and flip augmentations are not mentioned; treated as real-sample augmentation (not synthetic labelled data) but listed as measured, droppable options (Q2).
- Runtime: description says 60 min; CLAUDE.md allows up to 50 min target. Plan targets about 35-40 min total (estimate) to keep at least 30% headroom against the 60-minute stated limit; no wall-clock branching anywhere (time only in logs).
- Determinism: CLAUDE.md section 3 applies in full (fixed epochs, folds, batch, seeds, threads, device `cuda`, no import fallbacks, no `os.cpu_count`, no `cuda.is_available` switches).

**Strip-the-ML test.** Remove every trained component and what remains is the 216-assignment enumeration with uniform potentials, which scores exactly 0 (uniform). Frame-difference magnitude (aux 0-2) or intensity (aux 3-5) rules alone are measured only as a floor in diagnostics and never shipped. The hand-built part (enumeration, validity, decode) contains no learned-pattern knowledge about the data; all discriminative evidence comes from trained potentials. PASS.

**No whole-test aggregation.** Each test row's prediction is a function of its own 12 clips, the train-fit backbone/head, and three scalars fitted on train OOF. PCA, scalers and the grouping used for CV are fit on train only. PASS.

**Sibling leakage.** The target is a relation between a row's own candidates, so features computed from group-mates are the point of the task. The ones used (group-centred features, within-row relative transforms) are allowed under CLAUDE.md within-row transforms. Cross-row sibling structure (other rows of the same object) is used for validation grouping and for masking false negatives only, never as a predictive feature.

## Data findings

UNVERIFIED throughout (no data available). Facts from the description are labelled DESCRIBED. Everything else is a hypothesis with the exact diagnostic to run on `train` only.

**DESCRIBED facts.**
- train 1035 rows, test 252 rows (19.58% of all rows), 143 objects, 16 environments (115 / 11 in train; 28 / 5 in test).
- rgb `(rows, 4, 3, 2, 80, 80, 3)` uint8; aux `(rows, 4, 3, 24)` float16. About 12,420 train clips = 24,840 train frames; 3,024 test clips = 6,048 test frames; the uint8 train array is about 0.48 GB.
- Target is exactly three 1s per tensor. Each row has two opening states and one closing state; one opening state and the closing state occupy similar object configurations but move in opposite directions.
- Force/torque and tactile (aux 6-23) are nonzero only in the instrumented-gripper group; aux 0-5 are frame-change and intensity summaries for all groups.
- Group order and candidate order are shuffled independently per row; ids and tensor row order are salted/sorted independently.

**Diagnostics to run on train (in this order), with the hypothesis each tests.**
1. Schema and integrity. Shapes, dtypes, NaN/inf in rgb/aux, `group_cards` JSON parses with 4 entries, exactly one `has_force_tactile_aux` True, all `candidate_count` = 3, all four `view_kind` values present once per row. Check test has identical schema (schema only; no test feature distributions).
   H1: view_kind order over G1..G4 is uniform over the 24 permutations. If not uniform, note it as an artefact and do not exploit it.
2. Target parse and contract. Parse `train_targets.csv`; assert sum 3, marginals 1, exactly three cells 1. Recover for every row the true assignment of the 216. Unit-test the metric (see Validation) and run the gold oracle through the decoder: RowScore 1 on all rows.
3. Position/alias leak audit. For each pair of axes, the empirical distribution of the relative permutation (6 options) vs uniform (chi-square); marginal of the true candidate index per axis; same for view_kind pair order. H2: all uniform (shuffled independently, as described). Any deviation is logged and NOT exploited (rule 1-3 above).
4. Row-order/id audit. Correlate `tensor_row` and id hash with any row summary (mean intensity, motion magnitude). H3: none.
5. Duplicates. Exact hash of clips and rows; near-duplicates across rows via downsampled-frame hash. H4: no exact duplicates (description rejects them); near-duplicate (same-object) rows are common.
6. Group derivation for validation (key diagnostic). On train only, compute per-clip frozen embeddings (small cheap backbone or the DINOv2 features already extracted), then cluster rows to environments and objects. Use the external_human_view clips: H5: that camera is fixed within an environment, so the mean background feature clusters tightly into about 11 groups; the object-level clusters number about 115. Checks: cluster count near 11 and 115, size distribution (about 94 rows per environment on average, 9 per object), stability across two linkage methods and two embedding backbones, and that clusters are not simply correlated with label structure. Union-find over rows whose top-k mutual similarity exceeds a threshold chosen in-script from the gap in the sorted similarity histogram (no hand constant).
7. Static-clip rate and direction evidence. Distribution of aux 0-2 (mean |RGB change|) per embodiment. H6: some fraction of clips are near-static (two frames almost identical), especially for slow executions, which caps achievable direction accuracy; this is "irreducible ambiguity" to quantify (fraction of train clips with change below the 10th percentile of their embodiment).
8. Force/tactile. Fraction zero per group (must be 100% zero outside the instrumented group by construction), per-channel variance, sign structure of the two force time samples. H7: force-difference between the two times carries the push/pull direction and so informs opening vs closing for the instrumented clip; tactile stats track contact and grip state.
9. Zero-shot floor (no training). With frozen DINOv2 features, cosine similarity between clips of different embodiments, pair top-1 accuracy under the 216-enumeration, scored with the exact metric under the same grouped folds. H8: zero-shot is weak (pair top-1 perhaps 0.35-0.5 vs chance 0.33) because common background and embodiment appearance dominate; group-centred features (subtract the group's 3-clip mean) improve it, since all three clips of a group share object, camera and hand. This is the honest yardstick the fine-tune must beat.
10. Information ceilings. (a) Known-direction oracle: if the closing clip is identified in every group, the config space collapses from 216 to 8. Simulation here (pure Python, no data) gives RowScore 0.426 and exact 1.0 only at 1/8, final about 0.40 for a uniform posterior over the 8 survivors. So reliable direction recognition alone is worth about 0.40 of the final score and is the primary lever. (b) Naive-rule floors (motion-magnitude rank match, intensity rank match) for the strip-the-ML ablation only.
11. Opening/closing confusion. After the first trained model: H9: the dominant error mode is swapping the closing clip with the opening state of similar object configuration (about a third of errors or more); check by confusion over the 3 hyperedge slots when the true state pair is known.
12. Pair difficulty. Per embodiment pair, top-1 accuracy. H10: external-human vs wrist-human is easiest (same hand, same motion, different viewpoint); instrumented-gripper vs vision pairs are hardest and depend on force/tactile routed through the shared latent code.

## Validation design

**How the split was made (from the description).** Whole physical objects and whole environments are held out: all rows of an object stay on one side, all objects of an environment stay on one side. Test uses 5 environments disjoint from train's 11. Public/private is by complete environment (2 vs 3).

**Own groups from content.** Environment and object ids are not released, so derive them in-script from train content only (diagnostic 6): cluster rows into environment-like clusters via the fixed external-view background features plus agglomerative clustering with a gap-selected threshold, with union-find merging of near-duplicate rows. Use environment-level clusters as `GroupKFold` groups.
- Primary scheme: 5-fold GroupKFold over derived environment clusters (about 2.2 environments per fold), repeated with 2 different fold assignments if the cluster count allows (11 clusters leaves few ways to split, so the second split permutes cluster-to-fold assignment, not the clusters themselves). Report mean, std over folds, and per-fold scores for the exact `final_score`.
- Fallback if environment clusters are not recoverable: object-level clusters (about 115) with 5 GroupKFold. This shares environments (backgrounds, cameras) between train and validation, so it is optimistic relative to the true env-disjoint test, by an amount I cannot size without data. State the direction (optimistic) and use it for relative comparisons only.
- A plain random row split is forbidden as a score estimate: same-object rows are near-duplicates (about 9 per object), so it would be strongly optimistic.

**Holdout sanity fold.** In dev (not shipped), hold out about 2 derived environment clusters (about 20% of train) that never touch design selection, blend/sharpening fitting, or the regularisation grid; evaluate the frozen design once. The shipped script trains on all train.

**Metric implementation and tests.** Re-implement the exact `final_score` (CouplingSkill, PairMarginalSkill, MassSkill, exact decode over 216 with lexicographic tie-break, evidence gate, bottom-20% with `ceil`). Unit tests: perfect prediction scores 1.0; uniform scores 0; reversed/constant predictions: the all-mass-on-wrong-assignment tensor scores 0 (clipped); a valid tensor that is the base-rate (uniform) scores 0; the pure-Python reference implementation used here gives RowScore 0.426 for a uniform posterior over the 8 assignments consistent with a known closing clip (cross-check the in-script implementation to 3 decimals).

**Selection discipline.** Design choices (head weight decay grid, rank D, resolution) are picked on the 5-fold OOF; because the grid is small (see work plan) selection optimism is small but is reported. The sharpening scalar and the uniform-mix scalar are cross-fitted: fit on 4 folds' OOF, score on the 5th, report that number. Compare designs paired on the same folds; accept a change only if the paired gain exceeds the standard error across folds or is consistent in at least 4 of 5 folds.

**Bias direction of each proxy.**
- Environment-cluster GroupKFold: roughly unbiased to slightly pessimistic (each fold removes about 20% of the train environments, so training sees about 9 rather than 11 environments); if cluster merging is imperfect it becomes mildly optimistic.
- Object-level grouping: optimistic (shared environments).
- Random split: strongly optimistic.
- Self-built hold-out environments are drawn from the same 11 train environments, so they cannot test the shift to the 5 new environments perfectly; expect the private score to land at or below the grouped CV, with larger variance than usual because the private side has only 117 rows from 3 environments.

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Few independent units (about 11 environments, 115 objects) and about 9 near-duplicate rows per object: a high-capacity head or backbone memorises object appearance. Mitigation: capacity ladder (frozen features, then regularised low-rank heads, then LP-FT of the last two blocks only); grouped CV by environment clusters; very wide weight-decay grid including the strong end; PCA to a few hundred dims per feature type fitted on train folds; dropout on features; embodiment-specific projections of low rank D (32-64) with diagonal pair weights (fewer than about 20k head parameters beyond the projections; state the count in the log).
2. Memorising appearance of the closing/opening confusion pairs: group-centred features remove the shared object/camera/hand component, leaving only the within-group state/direction differences.
3. Selection on OOF: small grid (about 6 configs), cross-fitted sharpening, holdout sanity fold in dev.
4. Over-sharp posteriors give catastrophic Bottom20 (a wrong row near 0 while right rows near 1). Mitigation: proper scoring rule training (exact NLL plus the metric surrogate), temperature calibration on OOF, optional uniform-mix epsilon, and a final check of the OOF RowScore tail (fraction of rows with RowScore below 0.05).
5. False negatives in contrastive batch negatives (other rows from the same object): mask using the derived object/environment clusters.
6. Fine-tune moves too far: LP-FT with tiny backbone LR (about 1e-5), 2-3 epochs, EMA or weight averaging, log the parameter-change norm and the OOF gain over the probe.

**Underfit risks and mitigations.**
1. Throwing away evidence: down-sizing is not an issue (the source is 80x80) but patch-token pooling to CLS/mean can lose spatial motion direction. Mitigation: keep CLS, mean-patch, and a 2x2 grid of patch means per frame; the motion token is the feature difference (frame 1 minus frame 0) and keep both state and motion channels.
2. Backbone too small or domain mismatch at 80x80 upsampled: start from DINOv2-base (stronger local/instance features) and measure resolution as a design axis {112, 168, 224} on frozen features; do not default to a small model.
3. Direction collisions: the opening/closing confound requires an explicit direction signal; handled by the motion channel, the force channel for the instrumented clip, and the explicit role latent (see Structural signals).
4. Loss mismatched with metric: use exact marginal likelihood plus a differentiable RowScore surrogate; calibrate sharpness on the exact metric.
5. Force/tactile only exists for one embodiment and cannot be compared to images directly; routing it through a shared episode latent is required (below).

## Recommended approach (primary + fallback)

**Primary: shared-episode-space pair potentials, exact assignment posterior, LP-FT backbone.**

*Representation.*
- Backbone: DINOv2-base from the HF Hub (`facebook/dinov2-base`; pin the revision hash obtained at dev time, UNVERIFIED which hash). Frames upsampled from 80x80 to the chosen resolution (default 168, measured among {112,168,224}), ImageNet normalisation, bf16 autocast.
- Per frame pooled features: CLS, mean-patch, 2x2 grid of patch means. Per clip: state channel (mean over the two frames) and motion channel (frame 1 minus frame 0 features), plus the 3 frame-change and 3 intensity aux values (aux 0-5), all as extra inputs. For the instrumented-gripper clip, additionally aux 6-23 (force/torque at two times, tactile stats) through a small MLP.
- Group-centring: within each group subtract the mean of the 3 candidates' features (row-own transform) and keep the centred copy alongside the raw copy.
- Per-embodiment encoder `z_e`: Linear (after train-fit PCA to a few hundred dims) to D = 32-64 dims, LayerNorm, dropout, L2 normalised. The shared dimension D is the "episode code". The instrumented-gripper encoder takes vision plus force/tactile inputs, so force relates to the other views only through the episode code.

*Potentials and posterior.*
- Pair potential for each of the 6 embodiment pairs (a,b): `phi_ab(i,j) = s * <z_a(i), z_b(j)>_diag W_ab`, with learned diagonal `W_ab` (D parameters) and a learned scalar temperature `s`. (Per-pair-diagonal bilinear form, not a full matrix, to limit parameters; the per-embodiment encoders already carry the view-specific mapping.)
- Assignment score for each of the 216 assignments `c`: `S(c) = sum over 3 hyperedges, sum over 6 pairs of phi_ab(candidate_a, candidate_b)`. Posterior `p(c) = softmax(S/ tau)`; submitted tensor `P = sum_c p(c) * T_c` where `T_c` is the 0/1 tensor of assignment c, mapped back to the row's G1..G4 order. Precompute the 216 x 3 x 4 index table once (a fixed constant derived from itertools, not from data).
- Optional but in the primary after step 4 of the roadmap: **direction-role latent.** Each hyperedge is open or close; exactly one hyperedge per row is the closing episode. Add a per-clip direction logit `d_e(x)` from the motion channel (and force for the instrumented clip), and enumerate 216 x 3 = 648 latent states (assignment x which hyperedge is closing), with `S(c, r) = S(c) + sum over groups of d(candidate in the closing hyperedge) - d(open ones)` style unary terms (3 parameters per embodiment). The training likelihood marginalises the unobserved role: `-log sum_r p(c*, r)`. This is exact enumeration of 648 states (no sampling). It adds fewer than 100 parameters and directly encodes the main structural insight (collapse 216 to 8). Keep only if it beats the pairwise-only model by more than the paired fold noise; else the fallback below is the primary.

*Fine-tuning (compliance layer, must ship).* LP-FT: train the head on frozen features to convergence (Stage A), warm-start Stage B from that head, unfreeze the last 2 transformer blocks plus final LayerNorm of DINOv2 (about 14M parameters), backbone LR about 1e-5, head LR about 1e-3, 3 epochs, AdamW, cosine with short warmup, grad clip 1.0, EMA of weights. Early blocks run under `torch.no_grad`. Log `||theta_ft - theta_0||` and the OOF gain over the probe. If the grouped-CV gain over the frozen probe is not above noise, still ship the LP-FT model (compliance), choosing the final epochs so that it is not worse than the probe (the fixed epoch count is chosen from CV, not from the clock), and say so in the report.

*Shipping.* Fold models (5, env-grouped) give OOF predictions for calibration and reporting. Shipped test predictions: average of the posterior log-scores `S` over 2 full-data refits (different seeds) with joint-flip TTA (average the potentials over original and flipped inputs; flip is applied to all 12 clips jointly), then the fitted sharpening scalar. Refit on 100% of train is preferred over averaging fold models because the number of distinct environments is so small that dropping 20% of them matters (the lesson from small-group problems); the transfer of the OOF-fitted sharpening to the refit is checked in dev on the holdout environments.

**Fallback (lean, no latent role, no fine-tune of more than the head until step 3).** Pairwise-only 216-state model on frozen DINOv2 features with group-centring and the 3 motion/aux inputs, regularised low-rank encoders, plus the LP-FT of the last block only. If Stage B does not train stably in time, the fallback is the frozen head plus last-block fine-tune with the same posterior and sharpening. It is still compliant (last-block fine-tune is genuine fine-tuning, and its parameter-change norm and CV gain are logged).

**Second backbone (only after the primary is measured).** One diverse-assumption arm: a semantic image-text backbone (SigLIP or CLIP from HF) in the same pair-potential framework; blend in potential space (average `S`) only if it is close in grouped-CV quality to DINOv2 and the paired gain exceeds noise. Expected to add little; scheduled after direction/role work.

**Expected score (estimate, UNVERIFIED, not a promise).** Simulation on synthetic evidence in the scratch pure-Python model shows the metric is steep in the pair-evidence strength: with 3-way pair evidence effect size d' (per pair-edge log-odds shift over unit noise) of 0.5, 1.0 and 1.5 the final score is about 0.15, 0.51 and 0.88; these simulations are uncalibrated and for intuition only. Real cross-embodiment evidence from generic features is probably weak, so my guess for the grouped-CV final score is 0.08-0.30 with a centre near 0.15, and the private score at or below that with high variance (3 environments, 117 rows). A reliable direction recogniser is the lever that could lift this towards 0.35-0.40 (the known-closing oracle gives about 0.40 for a uniform guess over the remaining 8 assignments).

## Rejected options

1. **Independent per-axis rankings or per-pair Hungarian matches combined afterwards.** Explicitly banned as independent rankings that violate marginals; also cannot express joint ambiguity; the exact 216-assignment posterior is both valid and cheaper.
2. **Hard 0/1 assignment tensors.** Valid but overconfident: wrong rows score about 0 and Bottom20 collapses; the posterior mean with calibrated sharpness dominates in expectation under squared-error terms.
3. **Frozen embeddings plus GBDT/pair classifier (no backbone update).** Grey/likely rejected in a Fine-Tuning challenge and weaker than a joint structured model.
4. **Full fine-tune of a ViT-B on 115 objects / 11 environments.** Memorises object and environment appearance; ladder says stop at the lowest rung that wins under grouped CV, then use LP-FT only for the compliance layer.
5. **Video backbones (VideoMAE, V-JEPA 2).** Built for 8-64-frame clips; our clips have 2 frames at 80x80; heavy and mismatched.
6. **Robot-learning representations hosted on GitHub (R3M, VC-1, etc.).** GitHub weights are banned; only HF/timm. Reconsider only if hosted on HF and cleared with a reviewer.
7. **Sinkhorn / soft-assignment relaxation of marginals.** Approximates what exact enumeration does exactly at tiny cost (216 or 648 states), and does not guarantee the tolerance.
8. **Aux-feature rules (motion magnitude rank, intensity rank) as the solver.** Fails strip-the-ML; used only as a floor diagnostic.
9. **Cross-row aggregation of test rows (tying rows of the same object/environment, per-environment normalisation, transductive clustering).** Banned (whole-test aggregation, sibling leakage); the description's leakage controls reinforce this.
10. **Time-reversal as a default augmentation.** Reversing all frames in a row preserves the matches but destroys the force-direction coupling and creates 2-close-1-open rows that never occur in test; kept only as a measured option on the visual-only branch.
11. **Joint end-to-end 4-view cross-attention transformer.** Too heavy for 11 environments; the lean pair-potential model encodes the same structure with a few thousand head parameters.
12. **Tail-weighted (CVaR) training loss.** Fragile; calibration of the sharpening scalar on OOF final score handles Bottom20 with one parameter. Revisit only if the tail remains bad after calibration.

## Fixed work plan & runtime budget

All counts below are fixed constants. Time estimates are guesses to be replaced by profiling on an A10G (UNVERIFIED); the stated challenge limit is 60 min and CLAUDE.md target is at most about 45-50 min with at least 30% headroom, so I target about 35-40 min.

Stages:
0. Load train/test tensors and csv, validate schema, derive canonical group order from `view_kind`, build the 216-row index table. About 0.5 min.
1. Frozen DINOv2-B feature extraction. 24,840 train + 6,048 test frames, original plus joint-flip = about 62k forwards per resolution; resolutions {112, 168, 224}, bf16, batch 256. Rough throughput guess 600-1000 img/s at 168 (UNVERIFIED), so about 4-6 min in total for the three resolutions (112 is about 3x cheaper than 224). Store pooled features (CLS, mean patch, 2x2 grid) as fp16: about 31k x 2 x 4608 x 2 B = about 0.6 GB per resolution. About 5 min.
2. Stage A (frozen heads, env-grouped 5-fold CV): about 6 design configs (D in {32,64}, weight decay in a wide grid {1e-3, 1e-2, 1e-1, 1, 10}, resolution via the three extractions, with at most 6-8 total combinations evaluated, fixed list), full-batch or batches of 128 rows, 150 epochs; each fold fit is seconds on cached features. Choose the design by OOF final_score paired on the same folds. About 6-8 min.
3. Stage A role-latent ablation (648 states vs 216), same folds. About 2 min.
4. Stage B LP-FT, 5 folds: each fold trains on about 828 rows (about 20k frames), 3 epochs, last 2 blocks plus LN unfrozen, batch 16 rows (192 frames), bf16, joint-flip augmentation per row. Estimated 2-3 min per fold, so about 12-15 min; produces OOF potentials for calibration.
5. Calibration on OOF: sharpening scalar alpha (grid of 15 values) and an optional uniform-mix epsilon in {0, 0.05, 0.1}, cross-fitted over folds, final_score as the objective. Seconds.
6. Full-data refit: 2 seeds of Stage B on all 1035 rows, about 3.5 min each = about 7 min. Test inference with flip TTA, about 0.5 min.
7. Write, validate (exact validator plus tolerance checks on mass and marginals, reload with `keep_default_na=False`, re-score nothing on test). About 0.5 min.

Total estimate about 35-40 min on A10G (UNVERIFIED), about 33-42% headroom against the 60-minute limit. If profiling shows more, reduce, in priority order: resolution to 112 for Stage B, fold-LP-FT to the same folds with 2 epochs, one refit seed. These reductions are decided at dev time and hardcoded, never at run time.

Fixed settings (hardcode): seeds (python, numpy, torch, cuda, `PYTHONHASHSEED`, Generator for any sampler), `torch.backends.cudnn.deterministic=True`, `benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before importing torch, device `cuda`, fixed `torch.set_num_threads`, no DataLoader workers (data held as tensors in RAM, batches built by index with a seeded Generator), `HF_HOME` under the submission output parent, `TOKENIZERS_PARALLELISM=false`. No `os.cpu_count`, no `cuda.is_available` switches, no try/except import fallbacks, no time-based branching.

Memory: train rgb 0.48 GB uint8; pooled features about 0.6 GB per resolution; LP-FT activations for 192 frames at 168 resolution with only 2 blocks under grad are small (a few GB); GPU use well under 24 GB. System RAM 62.5 GiB is ample. Precision path: bf16 autocast on A10G for the backbone, fp32 for potentials and enumeration; use the same path on dev and grading (state if dev hardware lacks native bf16, in which case dev timings are pessimistic only).

## Metric-aware training & decode

**Training loss.** For each row, compute `p(c)` over the 216 (or 648 with roles) states and minimise:
`L = NLL(c*) + lambda * (1 - RowScoreSurrogate(P, Y))`
where NLL is the exact marginal likelihood of the true assignment (marginalising the unobserved closing-hyperedge role in the 648-state variant) and the surrogate is the differentiable RowScore: `0.45*CouplingSkill + 0.30*PairMarginalSkill + 0.25*MassSkill` evaluated on the posterior-mean tensor P (clipping replaced by the unclipped ratio). lambda fixed (about 1) and decided in Stage A as one of two values {0, 1} by paired grouped-CV gain; if no gain, NLL only. Row weights uniform (the metric's own per-row weights are uniform; OverallMean averages rows, and the Bottom20 term is handled by calibration, see rejected option 12).

**Output as expected utility.** CouplingSkill and PairMarginalSkill are squared-error terms; the posterior mean tensor minimises expected MSE for both (and pair marginals of a mean are the means of pair marginals). MassSkill is linear in sum(P*Y) and is maximised by concentrating mass on the MAP assignment; ExactSkill is the argmax over 216 of the three-cell sums of P. The one-parameter family `P_alpha` (posterior at inverse temperature alpha, followed by uniform mixing with epsilon) interpolates between the posterior mean and the MAP; alpha and epsilon are fit on OOF final_score with a very coarse grid, cross-fitted (fit on 4 folds, evaluate on the 5th), and reported both in-sample and cross-fitted. A single scalar was chosen deliberately (low-dimensional, CLAUDE.md section 4A).

**Constraint enforcement.** Validity is inherited: P is a convex mixture of 216 valid tensors. Floating-point renormalisation: compute the mixture in float64, clip to [0,1], write with at least 6 significant digits, re-check mass and marginals against the 0.02/0.03 tolerances and raise before writing otherwise.

**Calibration and diagnostics on OOF.** Reliability of `p(c*)`, mean and spread of the posterior entropy, fraction of rows with RowScore below 0.05 (tail), per-pair top-1 accuracy, per-fold final score. Compare against the zero-shot floor (diagnostic 9) and against the probe (Stage A) for the fine-tune gain.

**Reference numbers (pure-Python simulation, no data).** Uniform over the 8 assignments consistent with a known closing clip gives RowScore 0.426, exact-hit probability 1/8 (final about 0.40). Under synthetic noisy pair evidence with effect size d' = 0.5 / 1.0 / 1.5 the simulated final score is about 0.15 / 0.51 / 0.88 (60 rows, uncalibrated softmax). Use these only to sanity check that the in-script metric implementation behaves sensibly, not as a target.

## Structural signals

Invariants and structure from the description, and how each becomes a model component:
1. **Three one-to-one hyperedges, 216 assignments.** Exact enumeration; valid tensors by construction; posterior mean as output.
2. **Candidate permutation symmetry within a group and group-order shuffle.** Permutation-equivariant potentials; canonical group order by `view_kind`; unit test for equivariance.
3. **Two opening + one closing episode per row (known counts).** Direction-role latent with exactly one closing hyperedge, exact enumeration over 648 states, role marginalised in training. Verified consequence on train: after training the inferred closing fraction per group should be 1/3 by construction; check that the direction logits separate clips (histogram bimodality) for each embodiment.
4. **Opening state and closing state at similar configurations, opposite direction.** The motion channel (frame 1 minus frame 0 features) is signed; the state channel is not; the pair potentials take both. The force-difference between the two instrumented time samples enters the instrumented encoder as a direction cue.
5. **Frames are chronological within a clip.** Never shuffle the two frames at training or test (except in the optional visual-only time-reversal ablation).
6. **Force/tactile only for the instrumented gripper.** Routed through that embodiment's encoder into the shared episode code; zero-filled positions for other groups are never fed (the has_force_tactile_aux flag already identifies the group; other groups simply have no force branch).
7. **Shared object, camera, hand within a group.** Group-centred features (subtract the mean of the row's 3 candidates in that group) remove the common component and leave state/direction variation. Verify by ablation (diagnostic 9, step 3 of the roadmap).
8. **Cross-embodiment viewpoint change (direction of motion differs by view).** Per-embodiment encoders and per-pair bilinear forms learn the viewpoint mapping (the lesson that conditioning geometry must be learned, not assumed).
9. **Joint flip symmetry.** Mirroring every clip of a row together preserves which clips match; use joint flip as training augmentation and TTA, with the instrumented aux unchanged on the first pass. Measure with a train diagnostic whether an aux-consistent mirror (e.g. left/right tactile swap, lateral-force sign) helps; keep the simplest version that wins. If joint flip hurts, drop it and keep only per-clip photometric augmentation (brightness/contrast jitter, small random crop/shift applied identically to both frames of a clip).
10. **Pair marginals in the metric.** The posterior's pair marginals are exact sums of p(c); partial credit for correctly matched embodiment pairs is automatic.
11. **No leakage via ids, tensor rows, aliases, candidate order.** None are inputs; tested by the equivariance and shuffle tests.
12. **Row siblings (same object, other rows).** Used only to build validation groups and to mask false negatives in any contrastive auxiliary loss, never as predictive features.
13. **Optional auxiliary: multi-positive contrastive loss across embodiments within the row plus in-batch negatives from other rows, with same-object/same-environment masks (union-find over the derived clusters).** Enabled only if it beats the exact-likelihood-only training under paired grouped CV; log train loss vs held-out score per epoch to catch memorisation.

## Experiment roadmap

Each step has a stop criterion. Log table columns: id, change, grouped-CV final_score mean±std, per-fold, est. runtime, notes. Compare paired on identical folds.

1. **Contract, metric, validation, oracle.** Implement exact metric and the 216-assignment machinery; unit tests (perfect=1, uniform=0, constant/reversed=0, 8-assignment oracle=0.426 RowScore); gold-score oracle reproduces every train target; equivariance test; derive environment and object clusters from train content, check their count and stability. Stop when all tests pass and the cluster diagnostic is documented.
2. **Zero-shot floor and cheapest baseline.** Frozen DINOv2-B features at the three resolutions, cosine potentials (no training), then a regularised linear/low-rank pair head on frozen features, pairwise 216 posterior, calibration scalar. First valid end-to-end CSV (use the free local check; no credit). Record the floor and the baseline CV. Stop when the CSV validates and CV is recorded.
3. **Representation and structure.** One change at a time, paired on the same folds: (a) group-centring, (b) motion channel, (c) aux 0-5, (d) force/tactile for the instrumented encoder, (e) resolution choice, (f) 2x2 grid vs CLS+mean. Keep each only if its paired gain exceeds the fold standard error or is consistent in 4 of 5 folds.
4. **Explicit latent role (648 states).** Add the direction-role latent; keep if the paired gain over the pairwise-only model beats noise and the exactness (hit-rate) improves; else retire it and keep the fallback as primary.
5. **Metric-aware loss.** NLL only vs NLL plus the RowScore surrogate; sharpening alpha and epsilon cross-fitted. Check tail rows (RowScore below 0.05) and Bottom20.
6. **LP-FT (compliance layer).** Warm-start from the Stage A head, last 2 blocks, 3 epochs; log parameter-change norm and the gain over the probe; if the gain is not above noise, keep the fine-tune (compliance) with the epoch count that is no worse than the probe. Choose the epoch count as a constant from CV curves (not validation-triggered stopping).
7. **Augmentation ablations.** Joint flip and aux-consistent flip, photometric jitter, optional visual-only time-reversal, optional contrastive auxiliary with false-negative masks. Keep only measured wins.
8. **Diversity (only if the primary is solid).** Second backbone arm (semantic image-text model) blended in potential space; seeds. Keep only if close in quality and the paired gain beats noise.
9. **Bounded HPO (optional, only if time remains).** At most a handful of fixed in-script Optuna-free grid points (D, weight decay, dropout, lambda) with a fold-disjoint check; no timeouts; no more than the grid in the work plan. Skip if steps 3-6 show plateaus.
10. **Final fixed-plan run from a clean working directory.** Full end-to-end, run twice and diff (near-identical submissions required; large swings mean too few seeds). Validate the CSV, compare the test predictions' distribution to OOF (mean entropy, fraction of mass on MAP assignment), no tuning.

Credits: 6 per problem. Use at most 4: (a) the baseline from step 2, (b) the best single model (step 6), (c) the diversity/ensemble result if step 8 is kept, (d) the final. Use the free local check for everything else. Do not chase the public leaderboard (135 rows from 2 environments, noisy); trust grouped CV.

## Compliance audit

CLAUDE.md section 7 and the strategist section-B self-audits against this plan:
- Test file read only for one-row inference? Yes; the test arrays are read for per-row features and predictions only (schema/size used for runtime planning). PASS.
- Time in conditions? None; elapsed time only in log prints. PASS (to be enforced by code review before submission).
- `cuda.is_available`, `os.cpu_count`, import fallback, try/except changing work? None planned; device fixed `cuda`. PASS.
- Hardcoded constants tuned offline? The design constants (D, weight decay, resolution, lambda, alpha, epsilon) are chosen by in-script grouped CV on train; fixed work-plan constants (LP-FT LR about 1e-5, epochs, batch, seeds) are a-priori defaults noted in comments and not taken from earlier real submissions. The 216-assignment table is derived by `itertools` in-script. Flag: if any constant is taken from dev experiments rather than an in-script search, say so in comments (the grid is hardcoded as the search space, results found in-script). PASS with this caveat.
- External/synthetic data, self-hosted weights, GitHub models, non-allowed libs? Only the challenge data and HF Hub backbone weights (pin revision); libs limited to torch, transformers (or timm), numpy, pandas, scikit-learn (PCA, clustering), scipy. PASS.
- Would the solution work with the ML removed? No (uniform scores 0). PASS.
- Raw pixels to a tabular model? No; CNN/ViT backbone fine-tuned; hand features (aux) only as additional inputs to the deep model. PASS.
- Genuine fine-tuning load-bearing for a Fine-Tuning challenge? LP-FT of the last 2 blocks is in the shipped path with parameter-change and CV gain logged; the model-heavy part dominates. PASS, with the lingering risk that a reviewer may consider last-two-block LP-FT thin. Mitigation: if budget allows, unfreeze more blocks (4) with the same LR and verify under grouped CV that it does not overfit; the capacity ladder decides.
- Whole-test aggregation (rank normalisation, prior estimation, PCA/vocabulary/scaler on test, per-environment clustering of test)? None. PASS.
- Sibling leakage? Cross-row structure used only for validation groups and negative masking (train). PASS.
- Source readability, size under 512,000 bytes, comments explaining reasoning, seeds, fixed threads? Planned. PASS.
- Challenge-specific restrictions honoured (ids/row order/aliases not used, no per-axis rankings, exact output grammar, 60-min limit)? Yes, see Compliance regime.
- Possible "exploits generation process" flag: the known-counts structure (2 open + 1 close) comes straight from the description's task definition rather than a data artefact; note it in the code comments once.

## Open questions & assumptions

**Reviewer questions.**
- Q1. Is fine-tuning only the last 2-4 blocks of a pretrained DINOv2 (LP-FT, with the head trained on frozen features first) accepted as "genuine fine-tuning" for this Fine-Tuning challenge, or is a deeper unfreeze expected? Plan under each reading: if reviewers want more, unfreeze 4-6 blocks with the same recipe and rely on grouped CV to decide the epochs; if last-block is enough, keep 2.
- Q2. Are time-reversal and joint-flip augmentations of real clips acceptable (they are transformations of real samples, but time reversal changes the semantic direction label)? Default plan: joint flip only; time reversal off unless it measurably helps on the visual-only branch.
- Q3. Is canonicalising groups by `view_kind` (a documented per-row feature, not the alias) acceptable given the "Do not assume G1 represents a fixed embodiment" language? I read it as encouraged; the alternative (no embodiment information at all) is permutation-symmetric over groups but discards the main structure. If a reviewer objects, use a group-symmetric model with only has_force_tactile_aux as the sole embodiment cue.
- Q4. Is deriving environment/object clusters from train content for CV acceptable? It is train-only and used only for validation and negative masking. I read it as acceptable.
- Q5. Is the exact enumeration-and-posterior-mean decode acceptable under "calibrated, globally consistent" language? I read the description as requiring exactly this (calibrated coupling tensor). No decode-time metric-gaming is involved: the sharpening scalar is fit on train OOF only.

**Assumptions (where the description is silent).**
- HF/timm pretrained weights are allowed; no GitHub-hosted weights; no external or synthetic data.
- No whole-test or cross-row test aggregation of any kind, even label-free.
- Dev hardware is an A10G (or the same precision path is used); timings are estimates until profiled.
- The grader's `answers.csv` visibility column is evaluator-only and irrelevant to the solution.
- Test schema is identical to train (as stated); no test feature distributions were inspected or will be used for modelling choices.

**Could not verify in this run (no data):** every distribution, cluster count, pair accuracy, force/tactile behaviour, the 216-assignment oracle on real targets, all runtimes, the pinned model revision hash, and any score estimate. These are the first things to check with the diagnostics list above.
