# U1 — Galley restoration (user-supplied, verbatim minus site chrome)

## Overview
A galley contains four interrupted prose passages, sixteen loose text slips, and four detached endings. Each passage has three ordered gaps. Twelve slips belong in those gaps; four are extras. Every ending belongs to one passage. Restore the three slips and route the ending for each passage while using each selected slip and each ending at most once.

The visible passages deliberately omit the words immediately beside each gap and the attachment point for the ending. A slip can therefore be grammatically plausible in several positions. The remaining, more distant words, the other slips, and the competing passages must be read together to recover the four complete routes. The target is the joint assignment of three interior pieces and one continuation per passage.

The material is noisy period prose with irregular spelling and punctuation. The released inputs are text; no images are supplied. Training or fine-tuning a neural text model to compare longer-range context with competing pieces is the intended approach.

## Public data
The prepared public directory contains only train.csv, train_targets.csv, test.csv, sample_submission.csv, and galleys.jsonl. Source documents, source identifiers, notices, and a clean evaluation transcription are not included.

train.csv and test.csv have exactly target_id,galley_id,galley_byte_anchor,galley_byte_span. Each row denotes one interrupted passage. The byte anchor and span identify one complete UTF-8 JSON line in galleys.jsonl.

Each JSON record contains galley_id, heads, slips, tails, and contract. There are four heads, sixteen slips, and four tails. contract contains head_count:4, gaps_per_head:3, slip_count:16, and tail_count:4.

A head has head_id, target_id, and template. The template contains <GAP_1>, <GAP_2>, and <GAP_3> in that order. <VEIL> replaces text near each gap. A slip has slip_id and text; a tail has tail_id and text. Tail text begins after the concealed attachment point. Gap markers give order, but neither the hidden words nor their lengths. Opaque IDs, array positions, and file order have no answer meaning.

train_targets.csv has exactly target_id,prediction. Each prediction is JSON such as {"slip_ids":["slip_...","slip_...","slip_..."],"tail_id":"tail_..."}. The three slip IDs correspond in order to the three gaps; the tail ID selects the continuation.

sample_submission.csv has the same two columns and demonstrates a legal arbitrary assignment. It contains no answer key.

A page supplies multiple galleys. The training set has 1,000 galleys, distributed over 39 pages: 25 pages supply 26 each and 14 supply 25 each, so 25 x 26 + 14 x 25 = 1,000. Ten separate pages supply 100 evaluation galleys. Each galley contributes four rows: 4,000 training and 400 evaluation targets.

Pages with substantial repeated passages are kept in one partition. A proposed evaluation passage is excluded if an eight-token sequence from its underlying window also occurs on a training page; the same screen applies to the source windows for extra slips. Evaluation source windows do not overlap one another, and repeated head templates across partitions are excluded. This prevents copying an exposed training passage to obtain an evaluation answer, while allowing ordinary short phrases to recur.

## Submission
The full evaluation set is exactly the 400 target_id values in test.csv. Submit a UTF-8 CSV with exactly target_id,prediction and one row per ID. Row order is ignored. Each prediction must provide exactly three slip_ids in gap order and one tail_id from the same galley. Across its four passages, twelve distinct slips and four distinct tails should be selected, leaving four slips unused. Missing or extra targets, duplicate target IDs, and additional columns invalidate the full submission and raise an error.

## Metric
For each scored passage, let g be the fraction of its three gaps assigned the correct slip, t be 1 if its ending is correct and 0 otherwise, and e be 1 if all three slips and its ending are correct and 0 otherwise. The passage score is 0.55g + 0.15t + 0.30e. The returned score is 100 times the unweighted mean of the scored passage scores, clipped to [0.01, 100]. An exact answer scores 100.

The platform may evaluate a board containing only some rows of a galley. Such a board scores exactly its included target rows by the same passage formula; it does not need all four passages. For the full evaluation set, averaging the 400 passage scores is also the unweighted mean of the 100 complete galley averages, because every galley has four passages. The gap term rewards partial recovery, the ending term measures continuation selection, and the complete-route term rewards a coherent assignment.

Malformed prediction JSON earns zero for its row. An unknown or foreign slip or tail earns zero for the affected choice. Reusing a candidate cannot create another correct placement: each gold slip and ending is assigned to only one position within its galley. Each row is scored independently, so a board's score does not depend on whether another passage from the same galley is present. Other valid choices retain their credit. The grader aligns rows by target_id and returns one finite score for valid submission structure.

## Modeling and compute
The intended solution trains or fine-tunes a neural model on this task and materially computes the scored text compatibility on one NVIDIA A10G GPU with 24 GB VRAM. Preparation, training, inference, and submission writing must fit within 90 minutes end to end. A trained model may score candidate combinations and a capacity-constrained decoder may resolve its neural scores into a legal assignment.

CPU-only machine learning, hand-written rules, heuristic text repair, TF-IDF, BM25, bag-of-words, n-grams, edit distance, and fixed-embedding nearest-neighbor matching are ineligible as the main solver. A token neural training step does not qualify a pipeline whose candidate choices are actually determined by CPU logic or heuristics before or after inference. CPU work may parse files, batch inputs, and assign candidates from scores genuinely produced by the trained GPU model; it may not supply a substantial independent prediction signal. External answer lookup, recovery of withheld evaluation text, and manual labeling are prohibited.
