Overview
This challenge asks you to rank incoming software backports by the amount of human review capacity they are likely to require. You receive one non-reconstructable profile string per change request. It summarizes language and structure counts, lexical flags, destination-branch category, and an aggregate file-change footprint. Produce one continuous priority score per request: larger values should place requests that are more likely to attract substantial review closer to the top of a limited reviewer queue.

The target is a practical queue-relevance signal, not a judgment of contributor quality, code quality, or project importance. The hidden evaluation split is project-disjoint, so memorizing project-specific review norms is not a reliable strategy. A useful system should surface active requests first while still giving a meaningful ordering to light and quiet requests when capacity is tight.

The graded relevance label is derived from future human review activity in the underlying public archive: 0 means quiet (zero non-empty human review messages), 1 means light (exactly one), and 2 means active (two or more). Review records, repository identities, raw URLs, and source identifiers are not included in the public feature view. The top-20 evaluation cutoff represents a triage desk that can inspect only its first twenty recommendations during a planning cycle.

Dataset
File descriptions
train.csv: 2,418 labeled-example profile rows. It contains no target column.
test.csv: 586 profile rows from projects absent from train.csv.
train_targets.csv: Public training relevance labels, with columns id and numeric target.
sample_submission.csv: A valid-format example with columns id and prediction.
Column descriptions
id: An opaque 12-character example identifier. It is the join key and must not be used as a lookup into external data.
profile_text: A deterministic, non-reconstructable text profile containing named count bands, lexical flags, branch category, file-change aggregates, and path-kind aggregates. It does not contain the original title, body, repository URL, PR URL, PR number, author identity, source cohort, or review records. Numeric values appear as profile tokens so text and ranking models can use them without exposing raw source identifiers.
Columns in train_targets.csv
id: Matches exactly one id in train.csv.
target: Numeric graded relevance: 0 means quiet, 1 means light, and 2 means active.
Evaluation
The grader measures normalized discounted cumulative gain at 20 (NDCG@20). This is a ranking metric: it rewards placing active requests near the top, gives partial graded relevance to light requests, and gives no gain to quiet requests. The logarithmic discount models the practical loss of reviewer capacity when a high-workload request is buried deeper in the queue. The score is maximized and lies in [0, 1].

The gain mapping 2**relevance - 1 makes an active request worth more than a light request while preserving their ordinal relationship. A perfect ranking of the twenty most valuable requests scores 1.0. If the hidden set contains no relevant request, the score is defined as 0.0; the supplied holdout contains relevant requests. Stable sorting makes tied prediction scores deterministic.

import numpy as np

TOP_K = 20

def ndcg_at_k(truth, prediction, k=TOP_K):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size == 0:
        raise ValueError("The answer set cannot be empty.")
    cutoff = min(k, truth.size)
    gains = 2.0 ** truth - 1.0
    order = np.argsort(-prediction, kind="mergesort")[:cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    dcg = float(np.sum(gains[order] * discounts))
    ideal_gains = np.sort(gains)[::-1][:cutoff]
    ideal_dcg = float(np.sum(ideal_gains * discounts))
    return 0.0 if ideal_dcg <= 0.0 else dcg / ideal_dcg

truth = answers["prediction"].to_numpy(dtype=float)
prediction = submission["prediction"].to_numpy(dtype=float)
score = ndcg_at_k(truth, prediction)

The grader first validates that the submission and answer frames have exactly the same id set, sorts both by id, rejects missing or non-finite prediction values, and verifies that answer labels are in {0, 1, 2}. It then applies the code above to the aligned frames.

Submission
Submit one CSV file with exactly these columns, in this order:

id: Every identifier from test.csv, exactly once.
prediction: A finite numeric priority score. Only the relative ordering matters; larger values rank earlier.
Example:

id,prediction
000cddf25f45,0.73
00df636659ee,1.41
013160001efe,0.18
0195aa4be9a7,0.96
01995c82db9b,0.44

Requirements
Include exactly 586 prediction rows.
Preserve the test identifiers; do not sort by a different key or regenerate them.
Train only from the supplied public files. train_targets.csv is the only source of training labels.
Prediction values must be finite numeric scores; do not submit labels or strings.
Do not use external repository pages, source archives, raw review events, repository names, raw URLs, PR numbers, or source-cohort fields.
Write the final artifact to working/submission.csv when running a solution.
Allowed Methods
Use any reproducible ranking, regression, NLP, text-profile, or multimodal model that trains only on the supplied public files and permitted runtime libraries.
Use grouped validation or another leakage-resistant split design that respects the challenge's project-disjoint hidden holdout when legitimate group metadata is available during development.
Use graded-relevance losses, pairwise ranking, calibration, ensembling, and top-k error analysis when they do not access hidden review records.
Use GPU-backed encoders or fine-tuned language models if they operate on the supplied non-reconstructable profile text and remain within the selected runtime budget.
What Not To Use
Do not retrieve the original change requests or review events from a public forge or any other external service.
Do not use the raw archive, raw repository URL, PR URL, PR number, author identity, or agent/human source indicator as a feature or lookup key.
Do not infer the answer from row order, the opaque identifier, or the opaque project hash.
Do not add synthetic examples or synthetic review labels.
Do not interpret the target as a measure of code quality, contributor quality, or project importance.
Do not optimize for a hidden or external queue; evaluate only the supplied public features and the official NDCG@20 contract.
