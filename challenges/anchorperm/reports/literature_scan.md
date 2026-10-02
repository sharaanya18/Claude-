# AnchorPerm literature scan: per-episode matching under unknown role-mapping shift

Date: 2026-10-02. Author: eris-pattern-researcher (public methods only; no Eris solutions or data sought).
Scope: public papers / methodology for (1) per-episode in-context matching, (2) map-invariant matching,
(3) single-source domain generalisation, (4) calibration under shift without target fitting.
Nothing here edits curated references; everything is a proposal for the main session.

Reliability legend for sources: [S] = appeared in a search result this session (title/abstract confirmed);
[R] = cited from recall, URL not fetched this session (check before relying on it).
Compliance legend: PER-ROW = uses only the row's own items/anchors (allowed, CLAUDE.md 2.2/2.3.5);
POOLED = needs statistics over several test rows (banned); TRAIN-SIDE = only changes how we train on train rows.

## 0. Context cross-check (existing project knowledge, so we do not duplicate it)

- learned-patterns L001: Monte-Carlo linear-Gaussian ceiling; familiar rows are already at ~ceiling (clean 0.56 vs 0.57-0.59).
- L002: simulated-shift CV (additive code noise ~0.7, partial rotation) tracked the platform score (0.585/0.572 vs 0.58).
- L003 (negative result, important prior): rotation / partial-rotation augmentation, QAP structure term, two-pass
  pseudo-anchors, Sinkhorn decode, richer heads and context encoders gave 0 to -0.02 accuracy on shifted rows.
  Dev scripts dev_rot.py / dev_partial.py show the augmentation used was ORTHOGONAL (expm of a skew matrix) only,
  with clip/round re-quantisation. So "general (non-orthogonal) partial-relation maps" and "meta-learned anchor
  conditioning over a map distribution" are NOT yet covered by the negative result. Independent convergence
  below: Rosenfeld et al. (IRM risks) and the in-context-learning literature both say a learner only adapts
  inside the family of tasks it was trained on, which is consistent with L003.
- Metric facts that shape priorities (CHALLENGE.md): score = 0.75*L_mean + 0.25*L_worst, L_row = 0.85*L_seq +
  0.15*(c-R)^2. Per-row Brier cost of wrong confidence is 0.15*(c-R)^2: c=0.9 on a row with expected R=0.3 costs
  ~0.054 per row (minus the unavoidable variance), i.e. comparable to a +6pp accuracy gain. Worst-stratum
  (probably transfer_sparse) has its own 0.25 weight.

## 1. Ranked top 5 (details in sections below)

1. Per-row anchor self-test as a regime / reliability signal, used for (a) confidence and (b) gating between the
   familiar scorer and a map-invariant expert. [Section 4.1 + 2.1]  (most promising: cheap, per-row, directly
   attacks the calibration term and L_worst, and it is the only way to *use* an invariant expert without a regime id)
2. Information-ceiling Monte Carlo for per-row adaptation in the PARTIAL-relation regime (extends L001) with a
   Bayes per-row scorer (shrinkage-to-familiar Procrustes / probabilistic-CCA EM). Decides whether any accuracy work
   on shifted rows is worth credits. [Section 2.4]
3. Meta-trained in-context scorer (set-transformer/FiLM conditioned on anchors) trained over a broad distribution
   of GENERAL partial linear maps (not only rotations). [Sections 1.1, 1.2, 3.1]
4. Seeded "similarity-witness" / relative-representation / fused-GW expert that depends only on within-role
   geometry plus anchors. [Sections 2.1, 2.2]
5. Calibration stack without target fitting: ensemble + perturbation disagreement, low-dimensional monotone map
   fitted on OOF of clean AND simulated-shift copies, worst-stratum-aware shrinkage. [Sections 4.2-4.4]

## 2. Findings by category

Each block: mechanism (3 lines) / why it may help here / compliance / effort / minimal experiment / tag / sources.

### 1. Few-shot, in-context, per-episode inference

#### 1.1 Transformers learn in-context estimators (ICL of linear models), Prior-Data Fitted Networks
- Mechanism: train a sequence model on many episodes, each with its own latent function (here: its own pair of role maps);
  support pairs are tokens, query is a token, the net outputs the posterior predictive directly. Trained transformers
  approximate ridge regression / Bayes-optimal predictors for linear tasks. PFNs (TabPFN) do the same with a synthetic prior.
- Why here: our task is literally "infer a linear-ish cross-role map from 2-5 support pairs plus n unlabeled items".
  The meta-learning view says generalisation to a new map regime is bounded by the support of the TRAINING map
  distribution (the "prior" must cover the test regime). Current training sees one familiar map (plus orthogonal
  perturbations), so the net has nothing to adapt. Training over a wide family of partial maps is the missing ingredient.
- Compliance: TRAIN-SIDE for the augmentation; inference PER-ROW. Caveat: generating fully synthetic latent data
  would violate 2.3.6; transforming real training rows' codes (augmentation) is the safe reading. State it in the docstring.
- Effort: medium-high (episode sampler + model + CV). Gain: uncertain; L003 says row-local tricks failed so far.
- Minimal experiment: keep the current scorer; add a set-transformer encoder over [anchor pairs + all item tokens],
  output FiLM (scale/shift) parameters for the pair-scoring head. Train with a per-epoch fresh random anchor subset
  and per-row maps M_q=(1-l)I+lG_q, M_a=(1-l)I+lG_a, G Gaussian general matrices (also: random coordinate subset
  replaced, sign flips, per-role gain), l~U(0,1), re-quantise to {-2..2}. Evaluate on held-out l ranges
  (train l<=0.6, test l in 0.8-1.0) so "unseen regime" is truly unseen.
- Tag: PUBLIC RESEARCH (mechanism) + UNTESTED HYPOTHESIS (application).
- Sources: Garg et al., "What can transformers learn in-context?" arXiv 2208.01066 [R];
  Akyurek et al. "What learning algorithm is in-context learning?" arXiv 2211.15661 [S: listed as "In-Context Learning in Transformers: Linear Models"];
  Mahankali et al. https://arxiv.org/pdf/2307.03576 [S]; Muller et al. TabPFN https://arxiv.org/pdf/2207.01848 [S].
- Reliability: high for the linear-regression theory/empirics; no evidence at all for a map-REGIME shift outside the prior (that is the whole point of PFN limits).

#### 1.2 FiLM / hypernetwork conditioning on a support set (CNAPs, Conditional Neural Processes, HyperNetworks)
- Mechanism: a set encoder pools support examples to a task embedding; a small adaptation net outputs FiLM scales/shifts
  (CNAPs) or whole layer weights (hypernetwork) of the classifier; no gradient steps at test time.
- Why here: gives the scorer an explicit per-row "how are Q and A related in this row" latent computed from the 2-5
  anchors, instead of relying on fixed bilinear weights. CNAPs report robustness in low-shot, which is our regime (2-5 shots).
- Compliance: PER-ROW (the adaptation net sees only the row's anchors). Note: do NOT pool statistics across test rows (TaskNorm/BN across tasks would).
- Effort: medium (reuse 1.1 experiment). Minimal experiment: ablate FiLM-from-anchors vs. no conditioning on
  simulated-shift CV; accept only if gain > fold SE on the shifted copies and no loss on clean.
- Tag: PUBLIC RESEARCH. Sources: Requeima et al. CNAPs https://arxiv.org/abs/1906.07697 [S]; Perez et al. FiLM arXiv 1709.07871 [R];
  Ha et al. HyperNetworks arXiv 1609.09106 [R]; Lee et al. Set Transformer arXiv 1810.00825 [R].
- Reliability: strong on image few-shot; transfer to unseen mapping classes is not demonstrated.

#### 1.3 Prototypical / matching / relation networks
- Mechanism: classify queries by distance to per-class prototypes in a learned embedding; relation nets learn the metric too.
- Why here: weak. Our "classes" are anchors with ONE example each and the query-vs-support relation is cross-role, so the
  embedding must already be role-aware, which is what the two-tower already does. Only the "episodic training" idea transfers
  (already used: fresh random anchor subset per epoch, in structured-assignment.md AnchorPerm note).
- Compliance PER-ROW. Effort low, expected gain ~0. Tag: PUBLIC RESEARCH. Sources: Snell et al. arXiv 1703.05175 [R]; Sung et al. arXiv 1711.06025 [R]; Vinyals et al. arXiv 1606.04080 [R].

#### 1.4 Per-episode test-time adaptation (strictly one episode): MEMO, MAML inner loop
- Mechanism: for ONE test input/episode, take a few gradient steps on an unsupervised/consistency loss (MEMO: marginal entropy over
  augmented copies of that single point) or on the support set (MAML), then predict.
- Why here: the anchors are a labelled support set for the row; a few inner steps of the pair-scorer on the anchors
  (with strong shrinkage to the meta-trained init) is exactly MAML-as-hierarchical-Bayes. With 2-5 anchors in 32-d it can
  only move low-dimensional parameters (a temperature, a gate, a rank-1/2 update), otherwise it overfits.
- Compliance: PER-ROW and per-sample is explicitly the allowed mode (CLAUDE.md 2.2: per-sample tricks). BN-statistic
  adaptation / Tent over the test batch is POOLED and BANNED; if any adaptation step runs in a batch, it must be per-row (batch of 1 row, deterministic steps, fixed LR and step count, no early stop on clock).
- Risk: determinism check; keep a fixed number of steps, fixed seeds. Effort medium.
- Minimal experiment: inner-loop update of only (alpha, tau): mixing weight between familiar scorer and invariant expert and softmax temperature, 5 fixed steps on leave-one-anchor-out likelihood.
- Tag: PUBLIC RESEARCH. Sources: Zhang et al. MEMO https://arxiv.org/abs/2110.09506 [S]; Finn et al. MAML arXiv 1703.03400 [R];
  Grant et al. "Recasting gradient-based meta-learning as hierarchical Bayes" arXiv 1801.08930 [R]; Wang et al. Tent arXiv 2006.10726 [R] (POOLED, do not use).
- Reliability: MEMO gains 1-10% on ImageNet-C style shift; not tested on cross-role maps.

### 2. Invariances that survive unknown linear maps

#### 2.1 Relative representations / anchor-based standardisation + Procrustes (latent-space communication)
- Mechanism: describe every item by its cosine similarity to a set of anchors in ITS OWN space (relative representation);
  this is invariant to angle-preserving maps of each space, so two spaces become comparable with no training.
  Maiorca et al.: standardise with anchor statistics then fit an orthogonal Procrustes map on the anchors to translate across spaces/modalities.
- Why here: our anchors are exactly paired anchors. Q-relative vector r_q(i)=[cos(q_i,q_anchor_k)]_k and A-relative
  r_a(j)=[cos(a_j,a_anchor_k)]_k are comparable across roles if both role maps are near-isometric on the shared latent.
  Only 2-5 coordinates, so it is a weak (but regime-agnostic) feature; use it as an additional input channel / expert, not a replacement.
  Likely already partly present in the "anchor-structure scorer"; check before building.
- Compliance: PER-ROW (anchors and items of the row only). Effort low.
- Minimal experiment: scorer S_rel(i,j) = -||r_q(i)-r_a(j)||^2 after per-row normalisation, Hungarian; report accuracy on clean vs. rotated vs.
  partial-general maps. If S_rel accuracy is regime-flat while the learned scorer collapses, gate (Section 4.1). Also train a tiny head that takes (S_learned, S_rel) per pair.
- Caveat: if the role maps are not near-isometric (general linear, weak CCA), relative similarities are noisy; expect modest accuracy (maybe 0.2-0.3 vs random 0.15 on shifted rows).
- Tag: PUBLIC RESEARCH. Sources: Moschella et al. https://research-explorer.ista.ac.at/record/14217 (ICLR 2023) [S];
  Maiorca et al. https://arxiv.org/pdf/2311.00664 [S].

#### 2.2 Gromov-Wasserstein / fused GW with seeds; seeded graph matching (SeedGNN, FAQ)
- Mechanism: GW matches two point sets using only their intra-set distance matrices (no cross-space metric): minimise
  sum |C_q(i,k)-C_a(j,l)|^2 pi_ij pi_kl. Fused GW adds a cross-feature cost; seeds enter as fixed pi entries or a linear term
  (Alvarez-Melis & Jaakkola for words; Xu et al. GW learning for graph matching; "supervised seeded graph matching" SeedGNN
  learns to use seed-derived "similarity witnesses" and generalises across graph sizes).
- Why here: per row, C_q = pairwise item distances/Gram within questions, C_a within answers; seeds = 2-5 anchors; remaining 6-10
  items matched by Hungarian over the FGW coupling. This is the cleanest regime-agnostic information in the row (Gram structure is
  invariant to rotations/sign flips/coordinate permutations of each role). It is also the QAP term already tried (dev_qap.py, ap_qap.py,
  gain 0 to -0.02 on shifted rows); the new bit would be (a) learned item-affinity C (learned metric on within-role pairs instead of raw
  Euclid), (b) seeds as hard constraints and witness features, (c) using it as a gated expert rather than an additive term.
- Compliance PER-ROW. n<=12 so exact QAP-ish search or POT solvers are trivial in cost (check POT availability; else implement
  entropic GW in 20 lines of torch; POT is not in the allowed list so implement ourselves).
- Honest caveat: under weak canonical correlation (0.5 to 0) and unequal singular spectra of the two role maps, Gram structures differ
  and GW can be at chance. The previous QAP failure is evidence in that direction. Run the ceiling MC (2.4) first.
- Tag: PUBLIC RESEARCH. Sources: Alvarez-Melis & Jaakkola https://aclanthology.org/D18-1214/ ;
  Titouan et al. FGW https://proceedings.mlr.press/v97/titouan19a.html [S]; Xu et al. https://proceedings.mlr.press/v97/xu19b/xu19b.pdf [S];
  SeedGNN https://arxiv.org/pdf/2205.13679 [S]; Vayer semi-relaxed GW https://arxiv.org/pdf/2110.02753 [S];
  Vogelstein et al. FAQ, arXiv 1112.5507 [R].

#### 2.3 Wasserstein-Procrustes / self-learning with a seed dictionary (bilingual lexicon induction)
- Mechanism: alternate (i) Procrustes fit of an orthogonal map on current pairs, (ii) re-matching by OT/Hungarian; start from a small seed dictionary. Semi-supervised
  BLI studies show seeds help most but non-isometry hurts, and unsupervised variants fail when spaces are not near-isometric.
- Why here: it is per-row EM with hard anchors as initialisation. Failure modes are known: needs near-isometry and thousands of items;
  we have 8-12 items and 32 dims, so the rotation is under-determined (5 pairs << 32 dims). Procrustes WITH A PRIOR (shrink toward the
  familiar map W0, update only along directions supported by anchors) is the only variant that makes statistical sense.
- Compliance PER-ROW (do not pool rows to fit one shared map: that is transductive). Effort medium. Reliability for our regime: low.
- Tag: PUBLIC RESEARCH. Sources: Grave et al. http://proceedings.mlr.press/v89/grave19a/grave19a.pdf [S];
  Patra et al. "BLI with semi-supervision in non-isometric spaces" https://arxiv.org/pdf/1908.06625 [S].

#### 2.4 Probabilistic CCA / per-episode EM and the information ceiling
- Mechanism: Bach and Jordan show CCA is the MLE of a two-view latent Gaussian model q=Az+e1, a=Bz+e2, so per-row inference over a
  permutation is a latent-variable marginal likelihood; with the familiar (A0,B0) as a prior, a per-row MAP update of (A,B) given
  anchors + soft-assigned remaining items is per-episode EM.
- Why here: it is the generative model our task was built from (L001 already uses it for the ceiling). Extend L001: simulate the PARTIAL
  relation regime (shared subspace of rank r fraction + independent rotated remainder, larger spread), then compute (a) the Bayes
  accuracy if the map is KNOWN, (b) accuracy of the best per-row estimator that must infer a low-rank correction from k=2..5 anchors
  with the familiar-map prior, (c) the per-row-agnostic (structure-only) floor. If (b) is within a few pp of (c), accuracy work on
  shifted rows is a dead end and effort should go to calibration (Section 4). Counting argument: 5 anchors give <=5 constraints on a 32x32 map,
  so only a rank<=5 correction is identifiable.
- Compliance: PER-ROW for inference; the simulation itself is dev-only analysis (no submission data). Effort low-medium (numpy).
- Tag: PUBLIC RESEARCH (model) + UNTESTED HYPOTHESIS (ceiling statement). Sources: Bach and Jordan, Tech Report 688
  https://statistics.berkeley.edu/tech-reports/688 [S]; Karami and Schuurmans deep probabilistic CCA https://cdn.aaai.org/ojs/16982/16982-13-20476-1-2-20210518.pdf [S].

#### 2.5 Rank / ordinal and covariance-structure (SPD) features
- Mechanism: replace raw codes by invariants: per-row rank of an item's similarity to each anchor (ordinal), or SPD covariance/Gram descriptors
  re-centred and whitened per domain (Euclidean/Riemannian alignment, Riemannian Procrustes in BCI transfer: He and Wu).
  Alignment per domain uses that domain's mean covariance as reference.
- Why here: re-centring per row (subtract row mean, whiten with row covariance) is a legitimate PER-ROW normalisation and removes gain/offset/
  anisotropy shift (larger spread). But n=8-12 items in 32-d gives a rank<=11 covariance, so whitening is ill-conditioned; use shrinkage
  (Ledoit-Wolf style) or per-row scalar standardisation only. The BCI practice of pooling a subject's trials to define the reference IS pooling:
  not allowed across test rows; within a single row it is fine.
- Compliance: PER-ROW normalisation allowed; POOLED reference (all test rows) banned. Effort low. Expected gain: small, targets "larger spread".
- Minimal experiment: per row and per role: centre, scale to unit RMS, optional shrunk whitening; compare shifted-copy CV with/without.
  (learned-patterns/structured-assignment note already recommends per-row and per-role normalisation: confirms/converges, do not duplicate.)
- Tag: PUBLIC RESEARCH. Sources: He and Wu Euclidean alignment (review) https://arxiv.org/pdf/2004.06286 [S];
  EA with deep learning https://arxiv.org/pdf/2401.10746 [S]; Riemannian domain adaptation https://arxiv.org/html/2403.15415 [S].

#### 2.6 Differentiable permutation decoding (Sinkhorn / Gumbel-Sinkhorn)
- Mechanism: train with a Sinkhorn relaxation of the Hungarian step so the pair scores are optimised for the assignment loss, not per-pair CE.
- Why here: already tried as "Sinkhorn decode" (dev_sink.py) with no gain on shifted rows; useful mainly for calibrated assignment marginals (confidence).
  Mention for completeness. Compliance PER-ROW. Tag: PUBLIC RESEARCH. Source: Mena et al. https://arxiv.org/pdf/1802.08665 [S].

### 3. Domain generalisation from a single source

#### 3.1 Feature randomisation (RandConv, Fourier/amplitude mix, domain randomisation) as map randomisation
- Mechanism: apply a random, label-preserving transform per sample/episode so the network cannot rely on one nuisance
  parameterisation: random convolutions (shape preserved, texture randomised) / amplitude mixing / randomised simulator parameters.
  Single-source DG results: broad randomisation helps only when the randomisation family COVERS the shift; mismatched families do not.
- Mapping to our setting (how to simulate partial linear-map shifts, TRAIN-SIDE, per row independently, applied to Q and A separately):
  (i) general linear mixing M=(1-l)I+lG with G ~ N(0,1/32) (not only skew/orthogonal); (ii) subspace-selective: apply G only to a random
  subset of s coordinates (or to the top-k canonical directions estimated on TRAIN) so "partially related" is literal; (iii) per-role
  gain g in [1,2.5] BEFORE re-quantisation so the clip saturates (matches "larger spread"); (iv) coordinate permutation / sign flips on a
  random subset; (v) additive noise sigma~0.7 (already L002); (vi) mix two severities in one batch (MixStyle-like, below).
  Evaluate with HELD-OUT families (train with (i)+(iii), test on (ii)+(iv)) to measure real out-of-family generalisation, because
  in-family CV overstates it.
- Compliance: augmentation of real rows = allowed (2.2); do not fabricate new latent samples. Effort low (extend dev_partial.prot).
- Tag: PUBLIC RESEARCH + UNTESTED HYPOTHESIS (for general maps). Sources: Xu et al. RandConv (ICLR 2021) arXiv 2007.13003 [R]
  (search confirmed authorship/venue); Rethinking Data Aug for single-source DG https://arxiv.org/pdf/2211.14805 [S];
  Xu et al. Fourier-based framework arXiv 2105.11120 [R]; Tobin et al. domain randomisation arXiv 1703.06907 [R];
  Mix-spectrum / cross-correlation SDG: https://arxiv.org/pdf/2307.05901 [S].
- Reliability: in-family shift gains are consistent in the literature; L003 shows orthogonal-family augmentation did not move shifted-row accuracy here, so treat as <= modest.

#### 3.2 MixStyle / DSU: statistics perturbation
- Mechanism: mix (MixStyle) or sample from a Gaussian over (DSU) per-instance feature means/stds inside the network, synthesising
  new "styles" in feature space.
- Why here: our "style" is the per-role mapping + spread. Row-level mean/std of each role's item embeddings (computed over the row's own 8-12 items)
  can be mixed with another TRAIN row's stats during training; at test the per-row statistics are just own-row normalisation (PER-ROW).
  Cheap regulariser, likely small gain; its main value is as an ablation of row-statistics dependence.
- Compliance TRAIN-SIDE + PER-ROW normalisation. Effort low.
- Tag: PUBLIC RESEARCH. Sources: Zhou et al. MixStyle https://arxiv.org/abs/2104.02008 [S]; Li et al. DSU https://arxiv.org/pdf/2202.03958 [S].

#### 3.3 Invariant risk objectives (IRM, REx, group-DRO)
- Mechanism: penalise per-environment optimal-classifier differences so the predictor uses invariant features.
- Why here: environments = augmentation families (pseudo-regimes). Rosenfeld et al. prove IRM in the linear case needs more environments
  than environment-feature dimensions and can do no better than ERM otherwise; REx likewise. With one real environment and
  hand-made pseudo-environments the guarantees do not apply. Low priority; if tried, use group-DRO/worst-environment loss over the simulated families (aligns with L_worst in the metric) rather than the IRM penalty.
- Compliance TRAIN-SIDE. Effort low-medium. Tag: PUBLIC RESEARCH. Sources: Rosenfeld et al. https://arxiv.org/pdf/2010.05761 [S]; Arjovsky et al. IRM arXiv 1907.02893 [R].

### 4. Calibration under shift without target fitting

#### 4.1 Per-row anchor self-test (leave-one-anchor-out) as a shift detector and gate  -- TOP IDEA
- Mechanism: for each of the k anchors, hide it (reveal the others), score the row with the model, record the rank/prob of the true partner among
  the n answers (or the fraction of anchors correctly retrieved). This gives k labelled mini-trials from the SAME row, measuring how well the
  familiar scorer works on THIS row's mapping. Aggregate to a per-row reliability r (e.g. mean log-prob of the anchor partners, anchor top-1 rate).
- Why here: it is regime-sensitive without a regime id and without pooling. Familiar rows should have r high; shifted rows near chance (the model
  collapses to ~0.3, near random on the anchors too). Use r to (a) set confidence c = f(r, margin, n, k) learned on TRAIN with simulated-shift
  copies (L002, structured-assignment note already advises "per-row reliability (anchor agreement)": this converges with it; the new element is
  leave-one-anchor-out so the diagnostic is unbiased), and (b) gate/mix experts: p = w(r) p_familiar + (1-w(r)) p_invariant (Bayesian model
  averaging with a per-row posterior over {familiar, shifted}); w learned on train+simulated shifts only.
- Statistical caveat: with k=2 the diagnostic is 2 Bernoulli-ish trials; use the continuous log-prob and shrink w toward a prior; sparse strata
  (anchor 2-3) are exactly where the diagnostic is weakest and where L_worst probably sits. Expect an AUC well below 1 for sparse rows.
- Compliance: PER-ROW (uses only the row's own anchors; the anchors are inputs, not labels of hidden targets). Do not average r over test rows or threshold at a test-quantile (POOLED: banned);
  fixed thresholds/learned map from train only. Determinism: fixed code path, no time use.
- Effort: low-medium. Minimal experiment: on train-OOF, build clean + simulated-shift copies (rotation, partial general, noise); compute r; report AUROC of r for
  "row accuracy > 0.4" split by anchor count; then fit isotonic/logistic c=f(r, margin, n, k) cross-fitted by fold; report the exact metric
  (0.85 L_seq + 0.15 L_cal) per stratum on clean vs shifted copies and the worst-stratum term.
- Tag: UNTESTED HYPOTHESIS (grounded in GENERAL ML PRINCIPLE: model averaging / in-sample self-validation). Convergent sources: Ovadia et al.
  https://arxiv.org/pdf/1906.02530 [S] (confidence must reflect shift); CNAPs / MEMO (per-episode conditioning) above.

#### 4.2 Ensembles and perturbation disagreement
- Mechanism: members trained with different seeds/augmentation draws disagree more on shifted inputs; Ovadia et al. found deep ensembles the most
  robust under dataset shift (highest entropy on shifted data) and post-hoc temperature scaling insufficient.
- Why here: ensemble Hungarian-assignment agreement (fraction of members giving the same permutation) and score-under-input-perturbation
  (rotate/noise a row, recompute assignment, measure stability) are PER-ROW confidence features, and the 3-5 seed ensembles recommended by CLAUDE.md 4A are needed anyway.
- Compliance PER-ROW (test-time perturbation of one row = TTA-style, allowed unless the description forbids multi-view; this one permits "perturbation sensitivity").
- Effort low. Tag: PUBLIC RESEARCH. Sources: Lakshminarayanan et al. Deep ensembles arXiv 1612.01474 [R]; Ovadia et al. above [S].

#### 4.3 Post-hoc temperature / Platt / isotonic calibration, simulated-shift calibration
- Mechanism: a monotone map from confidence features to expected accuracy, fitted on held-out predictions. Temperature scaling (Guo et al. arXiv 1706.04599 [R]) is
  known to degrade under shift (Ovadia), so fit it on a MIXTURE of clean and simulated-shift OOF copies, not clean alone.
- Why here: L002 showed simulated-shift CV tracked the platform (within 0.01), supporting this. Keep the map low-dimensional (<=4 features: r from 4.1, margin,
  ensemble agreement, anchor_count) and monotone; the target is E[R | features] under the exact metric so Brier-optimal c = predicted accuracy.
  Optionally shrink c toward the transfer-regime accuracy for rows where r is ambiguous: asymmetric loss is not present, but L_worst weights the weakest stratum, so a slight conservative shift on sparse rows costs little.
- Compliance TRAIN-SIDE fit, PER-ROW application. Fit on test would be banned (calibration with test distribution). Effort low.
- Tag: GENERAL ML PRINCIPLE.

#### 4.4 Conformal / covariate-shift conformal
- Mechanism: weighted conformal prediction (Tibshirani et al.) reweights calibration points by the likelihood ratio of test/source covariates to keep coverage under covariate shift.
- Why here: requires estimating the target density ratio = POOLED over target rows, not allowed; and our output is a permutation + scalar c, not a set.
  Only a per-row analogue is usable: split-conformal-like quantile of OOF per-row loss within bins of r (TRAIN-side).
- Compliance: weighted conformal = POOLED (banned). Per-row binning on train = ok but adds little over 4.3. Tag: PUBLIC RESEARCH.
  Source: Tibshirani et al. arXiv 1904.06019 [R].

## 3. Cross-check summary (proposals vs. existing entries)

- Independently confirms: structured-assignment AnchorPerm note (per-row/per-role normalisation, anchor-agreement reliability, calibrate on clean + simulated-shift copies); L002; L003 (adaptation only inside training family; IRM-risk and PFN-prior arguments agree).
- Not yet covered locally: general (non-orthogonal) partial-subspace map randomisation with held-out families; meta-trained anchor-conditioned scorer; leave-one-anchor-out diagnostic; per-row Bayes/prior-Procrustes estimator; Monte-Carlo ceiling for the shifted regime.
- Explicitly non-compliant here (do not use): Tent/BN adaptation over test batches, reference-covariance alignment pooled over test rows (Euclidean alignment as practised in BCI), weighted conformal with density ratios, any self-training on test rows (two-pass pseudo-anchors across rows), clustering test rows by regime.

## 4. Suggested minimal experiment order (cost-aware, local only; each needs CV gain > fold SE to be accepted)

1. (2.4) MC ceiling for partial-relation regime, per-row-estimable correction vs structure-only floor: 1-2 h, no credit spent. Decides how much accuracy work is worth doing.
2. (4.1 + 4.3) leave-one-anchor-out reliability r -> confidence map; report exact metric per stratum on clean / noise / partial-general copies. Expected sure-but-small gain from calibration term.
3. (2.1 / 2.2) relative-representation + FGW expert accuracy on shifted copies; if regime-flat and above random, add gate w(r).
4. (3.1 + 1.1/1.2) general-map randomisation + anchor-FiLM meta-training, evaluated on held-out families.
5. (4.2) ensemble/perturbation disagreement as extra confidence features.

## 5. Honest bottom line

Public literature gives no recipe that identifies an unknown 32x32 cross-role map from 2-5 anchors; the theory (ICL/PFN prior support, IRM risks, identifiability counting)
and the local negative results (L003) all point the same way: adaptation is bounded by the training-time map family, and information per row is thin.
The realistic wins are (a) better calibration and a reliability gate (certain, small), (b) a regime-agnostic structural expert used only when the anchor self-test says the familiar scorer is failing
(possible, uncertain), and (c) a meta-trained scorer over a much broader, partially-related map family (speculative, medium cost). Do not expect more than a few points of accuracy on transfer rows.
