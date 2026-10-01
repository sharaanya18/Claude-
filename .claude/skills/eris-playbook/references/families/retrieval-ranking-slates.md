# Retrieval, ranking, recommendation and decoy-slate tasks

## Recognise
Rank candidates for a query (letters, patents, tags, next interests, role labels, witnesses); an archive/pool far larger
than a batch; slates with engineered decoys (matched, popular, filler); NDCG/MAP/MRR/Recall@k or chance-normalised
variants; "contentless candidates" (ids only).

## Default build: generate → score → decode as three systems
1. **Candidate generation** with enough coverage (report oracle recall at K): bi-encoder or learned sparse features;
   for ids-only candidates, collaborative/co-occurrence signals learned from train interactions.
2. **Scoring**: fine-tuned cross-encoder / pair-attention ranker over generic pair features, or listwise softmax over
   the slate with in-batch + mined hard negatives refreshed from the model's own scores (G14). A LambdaRank/listwise
   GBDT over broad features is a strong default when the pool is large; train on pseudo-slates sized like the test
   pool; include out-of-fold hard negatives.
3. **Decode**: per-slate ranking by expected gain; hard constraints (one answer per role, distinct selections).
4. **Diagnose coverage vs ranking vs decode separately** (three-stage diagnosis); the union of complementary generators
   often has a much higher oracle than either.

## Slate/decoy tasks
- Features should be generic and relational: self-relative gaps (C3), relabelling-invariant membership cells (C16),
  distinctive evidence vs a background bank (C19), label-augmented co-occurrence cross-fitted (C20). Statistics of
  *other cases' labels* are allowed only train-derived, out-of-fold, with matched statistic sizes (validation V8).
- **Engineered decoys**: the generator's recipe may be published. Model-learned generic offsets (a grid of history
  lengths weighted by the network) are defensible; hand-derived generator rules, generative decoy likelihoods and EM over
  test slates are the patterns reviewers reject (Q5/Q6, R1). A near-perfect score here is an audit trigger.
- Recover hidden taxonomies (role groups, equal-size groups) from train answer sets and enforce them in decode (A14).
- Expect the metric to include a worst-group term: weight per-group losses/selection toward the weakest group.

## Reference-resolution retrieval (letters, citations, patents)
Parse dates/aliases/numerals with feature extractors feeding a learned scorer; participant-relation features
(shared sender/recipient, reply structure); FiLM/gated query-candidate towers; rolling temporal validation when the
test lies later; calibrate by slice (language, century, mention type); OCR noise via char-level features. Keep the
query's own document out of its candidate bank (self-match exclusion).

## Cold candidates
Candidates without content: embedding by interaction graph (two-tower trained on train interactions), item-item kernel
kNN, popularity-by-context priors from train, then rerank with a learned model. Rolling temporal validation for
recency shift; report cold-start slice separately.

## Compute
CPU-only variants exist: sparse + from-scratch neural scorers, hashed features inside a trained model; keep fixed
epochs/rounds. GPU: encoders at fixed K for the cross-encoder pass (profile throughput first).

## Pitfalls
Cross-batch/test-pool statistics (candidate frequencies over test slates); evaluating on random splits when users /
institutions / time are disjoint in test; leave-one-out co-occurrence leaks; ranking features that peek at the
true item's group; blending scores on different scales; ignoring the abstention/"none" option when the metric rewards it.
