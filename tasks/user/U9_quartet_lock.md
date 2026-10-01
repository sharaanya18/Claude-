# U9 — Quartet Lock: one record each (user-supplied, verbatim; cross-modal assignment)

## Overview
Cross-modal assignment. Data arrive in locked quartets called batches. Each batch holds four sets of photographs and four short structured records, and behind it sits one hidden perfect matching: every set owns exactly one record, every record is owned by exactly one set. Unlock every evaluation batch, reporting for each set how you rank the four records it could own. A set is three photographs of insects collected together. A record is four numbers that state when and where a collection took place (year, day_of_year, latitude, longitude). No text, name or key passes between the two sides. The only bridge is what the photographs reveal.

## The mechanism
- A closed world of four: each batch brings its own four records; no answer vocabulary to memorise, so a model must judge how well an image set fits a record it has never seen.
- One record each: the true records of a batch fill four different positions. The four rows of a batch are one puzzle; a confident match for one set constrains the other three.
- Confusable by construction: the four records of a batch fall within 30 days of the calendar and within 15 years of each other, while any two still differ by more than 25 km or by more than ten years. Coarse cues barely separate them.
- Sets, not images: the unit of prediction is a set of three photographs, and the evidence is what the three show together.
- True physical scale: every insect is cut out and placed unscaled on a fixed canvas, so pixel size is a physical measurement shared by every photograph.

## Inputs
Each entry of `candidates` holds year, day_of_year, latitude, longitude. The four sets of a batch share one batch_id and the same candidate list in the same order. The position of the true record is uniform over the four positions. Training and evaluation rows carry the same feature columns. Every evaluation set was gathered by a person who never appears in the training rows.

## Outputs
One JSON array per evaluation set, holding each of 0, 1, 2 and 3 exactly once, most likely first. sample_submission.csv holds a valid placeholder ordering per row. A submission may hold at most 1,000,000 rows. Columns exactly set_id and ranking.

## Evaluation (Set Provenance Ranking Score; higher better; max 1.0)
For a scored row i, t(i) = position of the true record in that row's candidate list, pos(i) = zero-based place of t(i) in your ranking. top_choice_accuracy A = fraction with ranking[0]==t; reciprocal_rank R = mean 1/(1+pos); S = 0.6*A + 0.4*R. Uniform random ordering scores about 0.358.
Coupled rows: two sets given the same first choice cannot both be right. Four first choices that form a permutation are right for 0, 1, 2 or 4 sets of the batch, never exactly 3.
Invalid submissions score 0.0: columns not exactly set_id and ranking, blank or duplicated identifier, any evaluated identifier missing, or any ranking not a JSON array holding each of 0,1,2,3 exactly once. Invalidity judged over the whole submission.

## Preprocessing
Each photograph is cropped by a deterministic rule to the single insect it shows and pasted unscaled on a fixed grey canvas. Identifiers are salted digests, rows and candidate orders are salted shuffles, and every set holds exactly three photographs, so none carries information.

## Rules
Train genuinely inside the submitted script. A pipeline that still works with the learned model removed is not a valid solution. No external datasets; do not manufacture training data. Use the test set for inference only, one row or one published batch at a time: no pseudo-labelling, no test-time adaptation, no statistic computed across the whole evaluation set. Using the other rows of the same published batch is allowed; nothing outside the batch may be pooled. Identifiers, row order and file metadata are not predictive signal. Hand-labelling evaluation rows is prohibited.
Edge cases: identifiers compared after whitespace strip; row order does not matter; rows whose identifiers are not evaluation identifiers must hold a valid ranking but are not scored; every evaluation batch is complete (four sets, four records); ties cannot occur.
(Compute, runtime and dataset file sizes are not stated in the pasted text.)
