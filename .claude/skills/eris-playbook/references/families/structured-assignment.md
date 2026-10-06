# Structured prediction: assignment, matching, ordering, seriation, relations, coupling tensors

## Recognise
Output is a permutation, matching, tensor with marginal constraints, graph/relations set, distinct selections, ordered
list (stop order, execution order, midpoint), slot binding (roles ↔ cards), partition labels, or a joint answer over
several sub-outputs. Invalid outputs score below zero. Often 8!-or-fewer states: exact inference is possible.

## Default build
1. **Write the valid-answer space and the hard constraints** (marginals, distinctness, one-per-role, first fixed, sorted
   pairs). Verify that every constraint is *guaranteed* by the generator before relying on it (check on train).
2. **Learn local evidence** (pair/slot/potential scores) with a generic encoder; calibrate it.
3. **Joint model over the enumerated valid outcomes** when ≤ ~10^4–10^5: primary loss = cross-entropy over the joint space
   (sum of potentials per outcome), per-slot losses only as small auxiliaries (M); or exact NLL of the marginal slot
   likelihood with a latent permutation marginalised. Decode by posterior-mean marginals (valid by construction),
   MAP assignment (Hungarian), or expected utility under the exact metric. Sinkhorn/log-domain for soft assignments;
   enumerate order posteriors via constant incidence matrices (A18, F13).
4. **Exploit coupling**: one confident output forces another (A4); recover taxonomies from train answers (A14); two-pass
   self-conditioned decode (A15).
5. **Symmetry as signal**: any element can be the target; shuffle interchangeable parts in training and average fixed
   orders at test (E10, D9); permutation-equivariant architectures without positional embeddings; translation-
   equivariant evidence for localisation.
6. **Complete the output**: always submit a complete valid object (a complete stop permutation scores better than a partial one);
   use local search (fixed passes) to maximise the sum of calibrated pairwise precedence probabilities (Kendall-tau).
7. **Hierarchical/taxonomy outputs**: mask the child head by the decided parent; make local validation count the child
   only if the parent is right.

## Task shapes seen
- *Midpoint of a burst / next item*: exact order posterior from pairwise potentials; strongest signal was a better
  similarity (distinctive patch evidence), not a larger learner.
- *Stop-order / route reconstruction*: relation-aware set transformer with case-relative geometry + network evidence from
  *other routes' train patterns* (progress, "i before j", successor directions) + all-pairs precedence loss (the exact
  surrogate for Kendall tau); rotation augmentation with co-rotated evidence; derive route groups from stop-set overlap;
  leave-own-route-out evidence rebuilt nested per fold.
- *Quartet/pair locks, four-way contact-episode alignment*: potentials per embodiment pair over all assignments (first
  fixed), exact assignment likelihood, posterior-mean tensor as output (marginals valid by construction), sharpening
  scalar + uniform mix fitted on OOF cross-fitted; direction roles as a measured add-on.
- *Slot binding / cloze with unordered snippets*: one-sequence encoder per case, MLM-head scoring, multiple-instance loss
  for hidden mentions, exact role-order solve, two-order averaging.
- *Botanical/identification keys, couplet walks*: late-interaction scorer per couplet (description memory ↔ lead
  encodings with a null vector), per-couplet softmax, lead-order permutation in training, optional tree decode.
- *Relation graphs among items (security controls, BPMN edges, mutants)*: pair classifiers on generic pair features,
  transitivity/consistency decode, abstain option; beware sibling cross-referencing as hand features (rejections S2).
- *Selection under budgets* (portfolios, review windows, masks): enumerate minimal feasible families, expected-metric
  optimum under a noise model fitted on OOF, `[]` as a candidate.

## Validation
Group by the shared hidden unit (case, route, episode, environment, key). Check the cross-case independence assumption
before row-level splits; compare against the exact-uniform-guess score and the best structure-only baseline.

## Pitfalls
Independent per-slot argmax violating distinctness; train-time exploits of a verified constraint that is not
guaranteed on test; huge state spaces enumerated without incidence matrices; hard assignment where the metric rewards
calibrated marginals; ignoring the "none"/abstain case; unvalidated reduction that drops legal answers.

## Added from AnchorPerm (row-local few-shot matching with a regime shift)
Compute the Monte-Carlo information ceiling first (learned-patterns L001); build a loss over the hidden block only with anchors as inputs and a fresh random anchor subset each epoch; normalise per row and per role; add per-row reliability (anchor agreement) and calibrate confidence on out-of-fold predictions of clean and simulated-shift copies (L002). Do not expect row-local tricks to recover a changed mapping (L003).

## Added from Whose Second Half (6x6 prefix-to-continuation matching, five ranked solutions read)
Learn item-to-item affinity from the unlabelled full sequences with several refit-per-fold views (learned-patterns L014-L016), build pair features, double-centre them (L020), train a row-aware head on the exact 720-permutation likelihood or row + column CE (L017, L019), average many feature-bagged members (L018), choose blend / temperature / decoder in-script on OOF (L022), and score test under the same fold pools as train (L015). Do not copy supervised column selection or test-candidate sentences (L023). Per-solution digests and gap analysis: learned-patterns L025-L026 and `challenges/whose_second_half/reports/top5_digest.md`.
