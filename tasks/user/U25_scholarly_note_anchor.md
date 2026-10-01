Restore the connection between a scholarly note and the argument it comments on. You receive a complete paragraph, divided into consecutive text segments, and a detached note. Decide where the note belongs in the unfolding prose, then predict the index of the segment immediately before its original marker.

Placement determines which statement receives the note's explanation, qualification, or correction. Several segments may mention the same person or event while making different claims about them. A note that disputes an attribution belongs with the attribution it challenges; a note explaining an unfamiliar expression belongs with the passage using that expression. Recognizing the shared topic can leave this editorial decision unresolved.

Read the note in relation to the whole paragraph. Its commentary may add facts absent from the prose, disagree with a claim, or address an argument developed across several segments. The required answer is a single attachment boundary. You are not asked to label the kind of commentary or mark the full span it concerns.

The available evidence is the paragraph and the note, including any quotations or bibliographic references in their text. No previous attachment position or revision history is supplied. Predict the original marker placement, which may be more specific than the range of positions an editor could reasonably choose.

Dataset
The public directory contains three UTF-8 CSV files:

train.csv: 1,154 labeled examples from 53 chapters.
id: unique string identifying an example.
chapter_id: opaque string grouping examples from the same chapter. Keep these groups together in local validation.
note: detached note text.
segments_json: a JSON-serialized array of 4–20 nonempty strings, in reading order. Joining them with single spaces reconstructs the paragraph. A segment is a prose unit, usually a sentence, but need not be a complete grammatical sentence.
anchor_index: zero-based integer index of the segment immediately before the note marker.
test.csv: 288 examples from 18 other chapters. It has the same input columns and omits anchor_index.
sample_submission.csv: a correctly formatted submission that places each note at the paragraph's end.
Paragraphs contain 80–1,000 whitespace-separated words; notes contain 30–300. Training and test examples have disjoint chapter groups.

anchor_index is the zero-based index of the segment immediately before the note marker. A note may concern several preceding segments; predict the position where its marker appeared, not necessarily the first segment it discusses.

Evaluation
The score is exact position accuracy, with higher values better:

accuracy = number of correctly restored anchor indices / 288

Each example has equal weight, and the score ranges from 0 to 1. Predicting a nearby position gives no partial credit. The target is the original placement; another defensible editorial placement is still scored as incorrect. The exact position can sometimes be ambiguous from the supplied text.

An invalid prediction value, including a blank, nonnumeric value, nonfinite number, fractional number, or index outside that paragraph, scores zero for that example. Missing, extra, duplicate, blank, or unknown IDs and incorrect columns cause a submission error. Row order does not affect the score.

Submission
Submit a UTF-8 CSV with exactly 288 rows and exactly these two columns, in this order:


id,anchor_index

te_example_a,2

te_example_b,0

The IDs above illustrate the format; use the actual IDs in test.csv. anchor_index uses exactly the same zero-based integer format as in training. For a paragraph containing m segments, valid indices run from 0 through m-1. Index 2 inserts the note after the third segment. Each test ID must appear exactly once. Copying sample_submission.csv provides a valid starting file.

Compute
The CPU execution limit is 90 minutes end to end, using at most 10 CPU cores and 62 GB RAM.

What Not To Do
Do not access the raw upload, private answers, or another solution's predictions.
Do not use external task labels, externally retrieved passages, or source lookup. Train task-specific parameters on the supplied training data.
Do not split candidate positions from the same paragraph across fitting and validation. Keep whole chapters together when estimating generalization.
Do not report training accuracy as evidence of held-out language understandin

