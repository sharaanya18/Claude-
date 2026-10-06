# Challenge: hospital data-quality charts — consensus change marks with calibrated bracket odds (verbatim description)

Overview
Hospitals run on data that is entered by people and moved by software, and both of those break in ways nobody records. A ward stops using one form and starts using another, a lab swaps an analyser, an interface silently drops a field for three weeks, a coding rule changes on a Monday. None of it is announced. The only trace is in the shape of the data itself: a count that steps down and stays down, a measurement that suddenly scatters twice as widely, a stretch where the points simply stop. Anyone who has to trust a hospital data extract learns to read those charts, and what they are doing is not statistics - it is judgement about which wobbles are the system changing and which are ordinary noise. The charts here were each shown to a panel of roughly forty human inspectors, who drew a line wherever they believed something had changed.

You receive a chart: a plain scatter of one hospital data field over time, no title, no axis labels, no scale. For each change the panel settled on you must return two things: its horizontal pixel position, and how divided the panel was about it, as a probability distribution over three ordered brackets - weak (a minority of the panel drew that line), split (the panel divided roughly evenly), firm (most of the panel drew it). The second half is the harder half. You are not asked which bracket you would bet on; you are asked for the odds, and they are scored by a proper scoring rule, so a confident wrong call costs more than an honest hedge and there is nothing to gain by overstating certainty. An empty answer is a legitimate one, and it is the right answer for close to three charts in ten.

The evaluation is compositional. Charts belong to a cell: one of four hospital source systems crossed with one of eight kinds of field. Four whole cells are withheld -

antibiotics x categorical        lims x uniqueidentifier
micro x categorical              pas x datetime

and every chart you are scored on comes from one of those four. Each of those source systems appears in training under other kinds of field, and each of those kinds of field appears in training under other source systems, but never in that pairing, and no individual data field crosses the split either. Your score is the average of the four cells, one per withheld pairing, so transferring to three of them and collapsing on the fourth costs you a quarter of the score. What this regime rules out is worth stating plainly: a model that has learned what one extract family's charts look like, or how densely one kind of field plots, cannot carry that over by recognising it again.
A perfect score is not available and nobody should chase one. The target is a consensus of human opinion, not a fact about the series: the same panel, shown the same chart twice, does not draw the same set of lines, and its members disagree with each other far more than they disagree about anything objective. Two of the panel's marks can sit three pixels apart for one reason and twenty for another, and whether a wobble is "a change" at all is exactly the judgement the panel was asked for and did not always share. That irreducible disagreement is no longer a nuisance in the label - it is the label.

Dataset
The prepared public/ folder contains:

train.csv - one chart per row: columns id, image (the PNG filename in train_images/), n_points (how many timepoints the chart shows), aggunit (whether the series is aggregated by day, week or month), dataset_name (which of four hospital data extract families the field comes from), fieldtype (what kind of field is summarised - one of eight, such as categorical, numeric, datetime or freetext), aggfn (the summary function applied at each timepoint - one of nineteen, such as the count of values present, the percentage missing, or a mean length), and marks (the answer: a JSON list of the panel's marks). 3,675 charts carrying 7,468 marks.
test.csv - the charts to solve: the same columns as train.csv except marks, that is id, image (the PNG filename in test_images/), n_points, aggunit, dataset_name, fieldtype and aggfn. 1,367 charts, all of them in the four withheld cells.
sample_submission.csv - a correctly formatted placeholder submission (id, marks) that scores at the floor. It is built from the row id alone and contains no information about any answer.
folds.csv - columns id and fold, five folds over the training charts. A fold is a set of whole cells, never a slice of one, so holding a fold out asks for the same cross-cell transfer the score asks for. The folds are deliberately uneven in size, because the cells are.
train_images/ and test_images/ - one grayscale PNG per chart.
Where the three brackets come from. The published record stores, for each consensus mark, the share of that chart's panel who drew a line in that cluster - its support. A bracket is a reading of that share: weak below a quarter of the panel, split from a quarter up to a half, firm at a half and above. The boundary between split and firm is therefore the same half-of-the-panel line the dataset description draws when it calls a mark's agreement strong or weak; this challenge divides the lower class in two and asks for a distribution over the three rather than a single class. The training labels carry the brackets directly, so none of this has to be reconstructed.

The four descriptive columns are given for training and test charts alike, and you are free to use them. They are coarse: 27 distinct combinations of dataset_name, fieldtype and aggfn cover the test charts, the largest holding 171 of them, and they say what kind of quantity the chart plots, never which field it is or when it was recorded. They do not identify the data field that a chart comes from, and they are not a substitute for reading the chart.

Every chart is drawn by the same code in the same style, so nothing about the rendering separates train from test. Charts are not all the same width, and each one shows a different stretch of its own history, so a horizontal position means a position on that chart and nothing else - the same pixel column on two charts is not the same moment in time. Copying the marks of the most similar-looking training chart is a fair thing to try, and it scores barely above nothing.

Submission Format
Submit a CSV with exactly the columns id and marks, covering EVERY ONE of the 1,367 ids in test.csv - no missing, extra, or duplicate ids. A submission that leaves a chart out is rejected outright rather than scored, so file every row: if you have nothing to say about a chart, send [], which costs you only what an empty answer is worth on that chart and nothing more.

id,marks
hm0000000000ab-54b7,"[{""x"": 214.0, ""p"": [0.1, 0.2, 0.7]}]"
hm0000000000cd-93da,[]
hm0000000000ef-10b7,"[{""x"": 96.5, ""p"": [0.6, 0.3, 0.1]}, {""x"": 503.0, ""p"": [0.05, 0.15, 0.8]}]"

marks is a JSON list, written as a string in the CSV cell. Each element is an object with exactly two keys: x, the horizontal pixel position of the mark on that chart's image, as a number; and p, a list of exactly three numbers, the probabilities of weak, split and firm in that order. They must not be negative and must sum to 1 within a millionth. An empty list is written as [] and is a valid answer. The order of the marks within the list does not matter; the grader matches them to the panel's marks by position. There is no normalization of what you send: a p of the wrong length, one that does not sum to 1, a missing or extra key, or a non-numeric x makes that cell unreadable, and an unreadable cell is scored as no marks for that chart.

The training answers are written in the same format, with p one-hot on the bracket the panel actually landed in - so a copy of train.csv's marks column is a well-formed submission, and so is anything your model produces in that shape.

Evaluation
Each chart is scored by matching your marks to the panel's, closest pair first, each of the panel's marks used at most once, and only pairs within 5 pixels of each other allowed to match. A matched pair earns the complement of the ranked probability score over the three ordered brackets, which is the proper scoring rule for ordered classes:

credit = 1 - ( (P1 - R1)^2 + (P2 - R2)^2 ) / 2

where P1, P2 are the running totals of your three probabilities and R1, R2 those of the panel's true bracket. A mark placed correctly and called with the right bracket at full confidence earns 1; a confident wrong call earns 0; an honest hedge earns what its calibration is worth. Guessing the overall bracket mix on every mark is worth about three quarters of a perfect call, and always claiming firm is worth less than that, so calibration is the part of the task that is still open once localisation is solved.

chart = 2 * (sum of credit) / (number of your marks + number of the panel's marks)

A chart where you and the panel both mark nothing scores 1. A chart where exactly one of you marks something scores 0. Within each withheld cell the score is your skill above the best constant strategy, which is to mark nothing anywhere, and the reported score is the average over the four cells:

cell  = ( mean(chart) - mean(chart for the empty submission) ) / ( 1 - mean(chart for the empty submission) )

score = mean(cell over the four withheld cells)

floored at 0.001, which is the smallest score this challenge reports. Cells are weighted equally regardless of how many charts they hold, so the smallest withheld pairing counts as much as the largest, and the worst cell is where a score is won or lost. Each row id ends in a four-character tag naming its cell, which is how any slice of the test set can be scored; the tag tells you nothing that dataset_name and fieldtype in test.csv do not.

Because each cell's baseline is computed on the rows of that cell, submitting an empty list for every chart scores approximately 0 no matter how many charts are genuinely empty, and so does any other fixed answer: one mark in the middle of every chart, a fixed number of evenly spaced marks, the overall bracket mix on every mark, marks copied from a fixed training chart. Predicting a mark everywhere costs you, because your count enters the denominator on every chart. Scoring above 0 requires deciding, per chart, whether there is anything to mark, where, and how divided the panel was about it.

Some measured reference points, all produced by the real grader on the real test set, with no tuning on it:

the supplied sample_submission.csv: 0.001
binary segmentation on the series read back out of the chart pixels, every mark carrying the bracket mix of the training answers: 0.008
a small from-scratch convolutional network, trained on train.csv alone for nine epochs on a CPU, one decision threshold chosen on a training fold: 0.529

Direction: Maximize. Min: 0.001. Max: 1.0.

A submission is rejected outright, with an error rather than a score, if its columns are not exactly id and marks in that order, if it has duplicate ids, if it is missing any of the 1,367 test ids, or if its id set is neither the full test set nor exactly the set being scored. Individual cells are treated differently: a marks value that is blank, is not valid JSON, is not a list, or contains a mark that is not an object with a finite numeric x and a valid three-number p is read as no marks for that chart rather than as an error, so a partial submission still scores. That is not a way to earn anything - it scores exactly what an empty list scores, and an empty list is what you should send when you believe a chart has nothing to mark.

What Not To Do
Do not use external data or resources: no other chart, change-point or data-quality corpora, no annotations obtained from outside the provided files, and no attempt to identify the underlying institution or series. Public pretrained vision weights are permitted; train or fine-tune on train.csv only.
Do not use the id, the image filename, the row order, or the file position as a signal, beyond the cell tag the id carries, which only repeats what dataset_name and fieldtype already tell you. Predict each chart from its own row - its image and its four descriptive columns - together with what the training pairs taught you.
Do not attempt to recover test answers through repeated submission probing or any other side channel, and do not pseudo-label the test set.
Do not pool predictions across test charts: predict each chart on its own, with no voting, no per-batch calibration and no grouping of test rows by their descriptive columns or their cell tag. To be exact, because these two are easily confused: reading n_points, aggunit, dataset_name, fieldtype and aggfn as features of the chart you are predicting is expected and allowed, and test.csv ships them for that purpose; what is banned is letting one test row influence the answer given for another. The scoring boards are a row-level split of the test set, so pooling reads a broken group.
Do not infer a calendar for a chart and mark dates that are known to be eventful. Each chart shows a different stretch of its own series and is drawn at a different width, so there is no shared time axis to exploit, and answers keyed to absolute dates score near zero.
Your full solution - training and inference - must run on CPU only, on 10 cores with 62.5 GiB of RAM and no GPU, within 90 minutes. Do not introduce packages beyond the preinstalled environment.

---
Local dev note: files unzipped from b3772d61-public_3.zip into dataset/public/.
