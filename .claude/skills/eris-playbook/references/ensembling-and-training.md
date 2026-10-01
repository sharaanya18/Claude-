# Ensembling, pretrained models, training recipes

## E-A. Ensembling that survives private LB
- **Diversity of modelling assumption beats seeds**: different feature source, architecture, objective, alignment or
  decode (D1). Budget the ensemble to the compute: lean 2-model blends on CPU tabular; 2–9 members on GPU.
- Complexity does not track rank; in six problems one model won. Add a member only if it has an independent signal,
  a paired CV gain beyond noise, and the plan has the runtime for it.
- **Blend in a comparable space**: z-score or train-reference percentile per member, then simple average (D2, Q7).
  Never average raw probabilities from members with different scales/calibration (the wider-spread member dominates).
  Classification members trained with different class weights must be put on one scale first.
- Condition blend weights on a small discrete, interpretable row type (shared vs non-shared entity, language pair,
  protocol) only if grouped CV shows member strengths differ by type (D7). Otherwise equal or simple weights.
- Constrain the search so the trained neural ranker stays the majority when compliance needs it; score blends on the
  average of random CV and leave-one-group-out CV (D11).
- When a downstream learned model exists, feed a sibling model's predictions in as a *feature* (cross-fitted) instead
  of blending scores after the fact (D5).
- Complementary candidate generators: measure the UNION oracle; keep both if it is much higher (D6).
- Snapshot/EMA averaging of many checkpoints of one long run; fixed epochs; averaging fold models and full-data
  refits only after weighing the 80%-data cost (D13, V9). A greedy soup that keeps one checkpoint is a valid result
  (D4).
- Fit blend weights on OOF with non-negativity or sum-to-one, few free weights; nested check (validation V3).
- Per-member sanity guard: exclude a member automatically if its late-training loss does not beat the constant-prior
  loss (G8); assert that a "fine-tuned" backbone's parameters moved after the first step (G7).

## E-B. Pretrained representations
- Spectrum: frozen probe → LP-FT (train the head, then unfreeze the last blocks/stage at a tiny LR) → cache lower
  stages once and fine-tune only the tail (E12) → full fine-tune. Default to the cheapest rung that is strong; the
  frozen probe is the yardstick for every later rung. On 300–1500 rows with dozens of independent groups, strong
  shrinkage (C ≈ 1e-5…1e-2 on standardised features, weight decay, ridge in dual form when n ≪ d, F7) beats weak
  regularisation; search the range on the *strong* side as well.
- Use the largest model that fits compute **at the resolution/length that carries the signal** (DINOv2-L@448,
  native aspect for photographs; long-context encoders for long text). Small, low-contrast details die at 224.
- Match pooling/prefix conventions (E2): CLS vs mean vs last-token, `query:` prefixes, left padding. Concatenate
  [CLS ‖ mean-patch] for ViT features; flip-average.
- Domain-matched large encoders over generic ones when the domain is specialised (biomedical, legal, historical
  languages); continued masked-LM pretraining on all in-domain text (train inputs, unlabelled rows; never test inputs
  unless the description allows) as a warm start (E5).
- Pin every model to a commit SHA and assert config invariants after load (E8). Check the description for banned
  checkpoints (task-specific models such as species/medical detectors, API models).
- LoRA: add the new head to `modules_to_save` or use full fine-tuning; full fine-tune beats LoRA on small data (E6).
- Zero-shot NLI/templates are legitimate ensemble members or stand-ins before a trained head exists (E7); multi-token
  candidates under an MLM have three scoring conventions plus alias-token vocabulary extension (E9).
- Cross-fitted ridge probes between two embedding modalities as an alignment feature (E3); frozen aligners (E4).
- When weights are banned ("from scratch", "no pretrained"), train tokenizer and model on the provided data only;
  small convolutional/GBDT/linear models with strong augmentation and shrinkage; budget model size to distinct groups.

## E-C. Training recipes
- AdamW, warmup + cosine/linear, layer-wise LR decay for encoders, grad-clip 1.0, bf16/fp16 autocast on GPU,
  EMA, multi-sample dropout, dynamic padding. Fixed epochs and batch sizes.
- Regularise by *where you stop* (truncate a long one-cycle schedule at a fixed epoch and average EMA snapshots,
  G12) rather than by validation-triggered stopping on test-like data; best-checkpoint selection is fine on a CV
  validation fold but the final refit uses a fixed count.
- Small datasets: augmentation is load-bearing; shallower/narrower architectures; avoid label smoothing that stalls
  at ln(K); per-volume normalisation appropriate to the modality (O).
- Multi-task/auxiliary losses on correlated sub-labels (G2); multiple-instance "one of k" losses for weak mentions
  (G9); structural invariants as losses/priors (G13); soft-targets from fine-grid rasterisation (G10).
- Imbalanced verifier heads need class weights; check per-class F1 (G5).
- Hard negatives: mine from train only, refresh from the model's own current scores on a schedule (G14).
- Augment only with transformations that keep the label (and transform every geometry-tied input, G11). Copy-paste of
  real, individually-labelled components is defensible; fabricating hybrids across real entities is not (L).
- Pseudo-labelling on **test** rows is banned. Pseudo-labelling unlabelled *train* rows needs agreement between
  independent signals, not a bare confidence threshold (G4), and a statement that it uses train rows only.
- Metric-learning/contrastive setups: use swap-symmetric scores, hard negatives, and matched precision between
  development and grading code paths; if the description forbids test-time cross-row reasoning, score each case alone.

## E-D. Efficiency tricks that stay deterministic
Cache frozen activations once per run (hash-keyed within the run only; never across invocations, N1/N2); dual-form
ridge; constant incidence-matrix GEMMs instead of scatter backward (F13); integer histograms for GPU
rasterisation (F12); fixed-order gradient reduction for CPU multi-process training (F11); profile per-phase
wall-time to a side file for the runtime claim (F10).
