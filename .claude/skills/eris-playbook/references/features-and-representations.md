# Features and representations

Feature ideas recur across unrelated domains; the *principles* matter more than the domain trick. Hand-engineered
features are legitimate only as **inputs to a trained model** (see compliance-patterns.md §V). If removing the trained
model still leaves a working system, the features are doing too much of the job.

## F1. Candidate-among-siblings tasks (pick one of N, match, rank a slate)
- Add each candidate's value minus the group max/mean and the gap to the runner-up, alongside absolute values (C3).
- Relabelling-invariant membership-cell descriptors when item ids are anonymised or re-hashed per slate: how a
  candidate's elements relate to the query, to the parallel-language query, to siblings (C16).
- "Distinctive" evidence: score what two items share *relative to a background bank* (patch-level likelihood ratio),
  not absolutely (C19). Reference-normalise any similarity that common content inflates.
- Candidate-set composition models (learn which decoys co-occur with the answer) are fine if learned from train
  slates only (D12); never fit them on test slates.
- Canonicalise the order of interchangeable parts (sort by a stable key, carry attached values along) and combine
  with permutation-symmetric functions (mean, max, |a−b|, a⊙b) (D9). For sets: type embeddings, no positional
  embeddings, exactly permutation-equivariant models.

## F2. Text
- Multi-view featurisation for long text: full text, tail/lead, quoted spans, titles, per-field views (C5).
- Raw field + narrower fields extracted from it: overlap/Jaccard with each, union coverage, residual uncovered
  fraction (C13). Duplicate or near-duplicate text: group for CV first, then consider a target-encoded duplicate
  feature once safe (I5).
- Pretrained encoders: match each model's documented pooling/prefix convention (E2); never apply mean-pool to all.
- Alias, date and number normalisation (month names in several languages, roman numerals, 2-digit years) belongs in a
  *feature extractor feeding a learned scorer*, not in a rule that decides the answer.
- Whole-case-as-one-sequence encoders for sets of snippets, shuffling part order in training and averaging fixed
  orders at test (E10).

## F3. Tabular
- Explicit products/ratios for tree models when you have a specific interaction hypothesis (trees approximate
  products poorly, C1). Unit-normalise mixed encodings onto one physical basis (mass↔mole, °C↔K, local↔UTC) using
  domain-derivable constants (C2). Provide both linear and physically-motivated forms of rate-like variables (1/T, log
  concentration) (C9). Composition-weighted blend descriptors in every unit basis, plus non-ideality terms
  `x_a·x_b·f(a,b)` for mixtures (C8, C10).
- Summary statistics of distributions/windows (mean, max, margin vs best competitor, entropy, neighbour mass) instead
  of raw vectors (C4).
- Cross-field coverage and residual features; missing-value indicators; outcome-adjacent columns dropped with the
  reason written down (I3).
- Target-aware encodings inside folds only, with matched statistic sizes for train and test (validation V8).
- Many *estimators of the same quantity* (kernel-weighted averages at several bandwidths, Beta-Binomial posteriors at
  several prior strengths, ridge transfer at several λ, kNN) fed as parallel columns to GBDT (K): fine when the
  downstream model is regularised and data support it.
- Self-match exclusion when features come from a bank of "other" entities that could include the case itself (I1).

## F4. Vision
- Patch-set correspondence statistics beat global embeddings for same-scene/same-moment tests: mean best-match,
  mutual-NN rate, top-k pair similarity, flip-averaged (E11). Local-evidence and *translation-equivariant* models for
  localisation; sample/mask training to mimic the evaluation camera (G10).
- Recover latent discrete geometry (tile grid, sensor lattice) from train images and resample onto it (C18).
- Factorise event timing into where (spatial density) × when (per-location timing distribution) (C17).
- Cross-scale training: transport each crop to the evaluation object size and re-rasterise vector labels at the new
  grid; soft coverage targets from fine-grid rasterisation (J6, G10); simulate the target camera's blur/JPEG (J7).
- Augmentations that touch geometry must also transform controls/directions/orientation metadata (G11). Auxiliary
  geometric channels decoded by a classical algorithm (J1).

## F5. Audio and signals
- Frozen CTC/ASR model used as a *structural aligner* (forced alignment by Viterbi) to anchor where learned features
  are pooled (E4); distance-biased attention as a soft positional prior (C6); window-level summary statistics (C4);
  input-source dropout to stop over-reliance on one strong modality (G1).
- Causal alignment: align a clean reference to the noisy stream, then compare local evidence rather than whole clips.
- Time series: strictly backward-looking lags/rolling windows, chronological splits, sin/cos for cyclic fields,
  separate per-entity strictly-past features (validation V1).

## F6. Structured/graph/sequence outputs
- Predict the smallest learnable decision that preserves the valid-answer space (P3): a molecular-delta task reduced
  to bond-order classification; a proof-repair task to operation-family + local ranking; a JSON-generation task to
  ranking legal registry values plus a small generator only for the open field. Verify the reduction keeps every
  originally-valid answer.
- Architectural target exclusion for recurrent models that may use partial labels elsewhere in the sequence but not
  the current step's own (I4).

## F7. Aggregate-label (learning from label proportions) tasks
Name them: labels are counts/proportions over unlabelled instances. Mean-pooled linear models are the correct
baseline; any function class can serve as the per-instance model. Add DeepSets mean+max pooling with an auxiliary
instance head, bag-size augmentation, and neighbour transfer of aggregate rates from similar *training* groups (C14,
C15, D10).

## F8. Domain theory as architecture
When a named closed-form equation governs the phenomenon (cosolvency, Arrhenius, physics of exposure/light transport),
build the output computation around it with learned sub-networks for its terms (C11), or use it as a residual base for
a learned correction. Reimplement a banned toolkit's *published* algorithms exactly instead of ad-hoc approximations
(C12). Use physical metadata (voxel spacing, units) rather than array indices (O).

## F9. Ambiguous feature definitions
If two defensible formula variants exist and the data cannot decide, compute both as fixed variants and average the
resulting predictions as a nuisance axis, like a seed (D8).
