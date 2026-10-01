# Partition / clustering / per-bag grouping tasks (and per-row independence rules)

## Recognise
Each test "bag" or case must be partitioned or labelled **on its own** (e.g. snippets from 2–4 unknown batches, speakers,
sources) with ARI/NMI/purity metrics; labels in test come from classes **never seen in train**; the description bans
pooling or clustering across the whole test set, and sometimes bans length/language identity shortcuts.

## Default build
1. **Metric learning, not classification**: closed-set classifiers (assign each item to the most likely train class)
   plateau (0.21 vs a 0.40 target on a batch-origin task). Train an embedding/same-source scorer on train *pairs* with the
   hard negatives that match the shortcut features (same length, language) so the model learns source style, not length.
2. **Pair scoring**: swap-symmetric pair network (|a−b|, a⊙b, cosine of learned embeddings); train with sibling-aware bag
   simulation: build synthetic bags from train classes with the same size/class-count distribution as the test bags
   (this is resampling real data, not fabricating entities); hold out whole classes (batches) per fold.
3. **Per-bag decode**: only that bag's rows may be used. Cluster the bag's pair-score matrix with a fixed algorithm
   (average-link/spectral/correlation clustering) whose number of clusters comes from an *in-bag* model-based criterion
   or a learned estimator of K from pairwise statistics, evaluated by ARI under the bag simulation. Degenerate answers
   (all-one, all-singletons) score 0, so enforce a minimum structure.
4. **Features for noisy/OCR text**: character-level noise statistics (substitution/confusion profile, spacing,
   diacritics, ligatures, punctuation habits, digit/letter confusions), style n-grams, as inputs to a trained neural
   scorer. Do not lean on length or language identity if banned (hard-negative them).
5. **Validation**: leave-batches-out folds; evaluate ARI per simulated bag with the real size distribution; report the
   spread. Calibrate K estimation and the linkage threshold nested.

## Per-row independence
If the description says each row/bag is predicted independently: no test-wide normalisation, no threshold fitted on test
scores, no global clustering, no cross-bag priors. Verify by scoring one bag alone and inside the full file; outputs
must match. A pooled-statistics shortcut that scored 0.094 is banned; the compliant learned scorer is the target.

## Pitfalls
Number-of-clusters tuned on the test distribution; memorising train class identities; pair labels from classes
that straddle folds; ARI undefined on tiny bags; label permutation instability (compare via ARI, never by id).
