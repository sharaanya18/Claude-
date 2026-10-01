# U2 — Sanskrit eight-card transport repair (user-supplied, verbatim)

## Overview
Four short passages have each lost two words. Their eight removed words appear as shuffled cards. Restore every word to its original gap, using each card exactly once. The gaps are coupled: a choice that fits one sentence can leave an implausible pair for another. A complete repair must satisfy both local context and the inventory shared by all four passages.

The passages use Sanskrit with preserved diacritics. This task predicts a joint eight-card transport across distinct contexts. It does not ask for grammatical role labels or a standalone masked-word completion. Each group is assembled so superficially similar grammatical positions compete for cards; token shape and an isolated verb are insufficient to determine the complete assignment. Some otherwise visible context words are masked as well.

## Files
train.csv and test.csv have identical columns: target_id, capsule_id, context_id, gap_id, capsule_byte_anchor, capsule_byte_span. There is one target row for each gap. The byte fields locate the corresponding UTF-8 line in capsules.jsonl.
capsules.jsonl contains one JSON object per eight-gap group. Each object has capsule_id, contexts, and cards. Each of four contexts has context_id, an ordered tokens array, and two gaps. A gap has gap_id and its zero-based position in tokens. The token at that position is [GAP]. Additional [MASK] tokens are context withheld from every participant and are not targets.
cards is an array of eight objects with card_id and form. Card IDs and all target, group, context, and gap IDs are opaque; their spelling and order carry no linguistic meaning.
train_targets.csv has target_id,prediction, giving the correct card ID for each training gap.
sample_submission.csv has target_id,prediction and a valid example of the submission layout. Its arbitrary card assignment is not a set of evaluation answers.
There are 1,256 training groups (10,048 targets) and 244 evaluation groups (1,952 targets). A group always has four contexts, two gaps per context, and eight cards. The word cards within a group have distinct forms. No card form remains unmasked in any of its four contexts. Candidate cards have comparable grammatical surface conditions within a group, so an individual gap should be interpreted in its passage and in the coupled assignment.

## Split and memorization control
Underlying predicate families are assigned wholly to either training or evaluation. Exact sentence texts are also kept on one side of the split, including repeated occurrences. The four contexts in a group are distinct. Consequently, memorizing complete training passages or a predicate-specific card association cannot reproduce evaluation answers. Vocabulary items and broad grammatical patterns may appear on both sides; the held-out task remains to place them in new contexts. The public files expose no family key or source identifier.

## Submission
Submit a UTF-8 CSV with exactly the columns target_id,prediction and exactly one row for each test target. The prediction is a card_id present in the target's group. Across all eight gaps of a group, each card should be used once. Row order does not matter. Match each target to its context through the shared IDs and its group's byte anchor; do not infer relationships from CSV row order.

The evaluator treats a foreign card ID, blank value, or any card assigned to multiple scored gaps in the same group as incorrect for the affected rows. A valid card assigned to the wrong gap is incorrect there. Missing, duplicate, or extra target IDs and an incorrect CSV column schema fail submission validation. When a platform evaluation slice contains only some gaps, the scorer uses the supplied scored rows and their full private group inventory. It accepts a full test submission for such a slice.

## Metric
For each group, let S be the fraction of scored gaps receiving their correct card. Let P be the fraction of represented contexts for which every scored gap in that context is correct. Let C be one if every scored gap in the group is correct, and zero otherwise. Its score is 0.50 S + 0.35 P + 0.15 C. On the complete test set, P requires both gaps of a context and C requires all eight gaps. The final score is 100 times the mean group score, clipped to [0.01, 100]. A fully correct submission scores exactly 100.

The slot term rewards progress, while pair and complete-group terms reward a coherent restoration. Reusing one attractive card across gaps cannot satisfy the one-use assignment or gain duplicate-row credit.

## Modeling and compute rules
The intended approach trains or fine-tunes a neural model on the supplied repair task and uses GPU-produced contextual scores to materially choose the eight-card assignment. The budget is one NVIDIA A10G with 24 GB VRAM and 90 minutes end to end, including training and inference.

CPU-only machine learning is ineligible. TF-IDF, BM25, bag-of-words, n-grams, fixed-embedding nearest-neighbor matching, manually written grammar, hand-built similarity scores, and other heuristic pipelines cannot create the candidate ranking or substantially determine the final assignment. A nominal GPU step followed or preceded by a decisive CPU or rule-based solver does not meet this requirement. CPU file handling, batching, and a one-use assignment algorithm applied to neural scores may support the neural solution.
