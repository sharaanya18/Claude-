AnchorPerm — Few-Shot Scientific Correspondence Completion
Domain: structured sequence-to-sequence (bipartite correspondence completion). Scoring: LOWER is better. Compute: NVIDIA A10G-class GPU.

Overview
AnchorPerm is a structured sequence-to-sequence benchmark for document-local bipartite correspondence completion.
Each row contains a local question inventory (Q01, Q02, ...), a local answer inventory (A01, A02, ...), and a few revealed
question-to-answer links called anchors. IDs are row-local: A07 in one row has no meaning in another.
The task is to infer the remaining one-to-one mapping and output the hidden answer sequence in ascending hidden-question
order. Each submission also includes a confidence value estimating the expected quality of the complete predicted hidden sequence.

Public files
train.csv has 3,629 labeled rows: example_id,set_size,anchor_count,anchors,q_items,a_items,target_sequence
test.csv has 907 evaluation rows: example_id,set_size,anchor_count,anchors,q_items,a_items
sample_submission.csv has: example_id,target_sequence,confidence

Row representation
set_size is 8-12. anchor_count is 2-5, with at least five hidden relationships remaining.
Each active question/answer item is a 32-coordinate quantized code using: a=-2, b=-1, c=0, d=1, e=2.
Question and answer codes use different role-specific mappings, so direct coordinate equality is not a valid global matching rule.

Anchors and hidden target
Example: anchors = Q02=A07 Q06=A01 Q08=A05
Anchored questions and answers are excluded from the hidden output. For each row:
1. order active questions numerically;
2. remove anchored questions;
3. remove answers consumed by anchors;
4. infer a one-to-one assignment for the remaining questions;
5. output answer IDs in ascending hidden-question order.
Every remaining answer must appear exactly once.

Representation transfer
Training rows come from familiar representation regimes. Evaluation contains both familiar rows and held-out transfer regimes
absent from labeled training. No regime ID or familiar/transfer flag is public. Robustness must come from transferable matching
structure, anchors, regularization, and uncertainty signals derived from the public semantic codes.

Confidence
Every prediction requires 0.0 <= confidence <= 1.0. Confidence estimates expected accuracy of the whole hidden assignment, not
one token. Useful solver-side confidence signals include assignment margins, model disagreement, anchor-conditioned disagreement,
perturbation sensitivity, set size, anchor count, and representation-shift diagnostics.
R denotes realized sequence quality.

Submission format
Write ./working/submission.csv with exactly: example_id,target_sequence,confidence
A valid target_sequence must: have exactly set_size - anchor_count tokens; use only currently available active answer IDs;
contain no duplicates; use every remaining answer exactly once; follow ascending hidden-question order.

Evaluation metric (lower is better)
For a valid row: y = true hidden answer sequence, p = predicted, m = len(y).
L_seq = (1/m) * sum_i [p_i != y_i];  R = 1 - L_seq;  for confidence c: L_cal = (c - R)^2;  L_row = 0.85 * L_seq + 0.15 * L_cal.
Each evaluation row belongs to one hidden creator-side robustness stratum: familiar_sparse, familiar_rich, transfer_sparse, transfer_rich.
Sparse means anchor_count 2-3; rich means 4-5.
L_mean = mean L_row over all evaluated rows; L_worst = maximum mean loss across the four strata.
score = 0.75 * L_mean + 0.25 * L_worst.
Perfect prediction with confidence 1.0 scores 0.0. Valid scores lie in [0,1]. File-level structural corruption scores 2.0.

Invalid predictions
File-level errors (wrong columns, duplicate/malformed IDs, unknown test IDs) score 2.0.
Row-level errors score 1.0 for that row (missing predictions, malformed sequences, invalid confidence, repeated answers, wrong length,
unavailable answers).

Recommended validation
Use complete rows as validation units. Fit only on the training fold, predict complete hidden permutations on validation rows,
estimate confidence, and compute the exact sequence + calibration loss. Because held-out representation regimes do not occur in
labeled training, ordinary random validation is not a complete estimate of transfer robustness.

Allowed methods
Public-data model training, numerical feature engineering, row-local anchor conditioning, relational/pair-scoring models, coherent
one-to-one decoding, confidence calibration, training-only robustness augmentation, and compliant ensembling are allowed.

Prohibited methods
Do not use private grading artifacts, hidden evaluation labels, source-record lookup, hardcoded test outputs, example_id-to-target
lookup tables, or filesystem/serialization order as a target channel.

Compute: Configured for an NVIDIA A10G-class GPU.
