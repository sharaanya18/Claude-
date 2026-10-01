Overview
Farm monitoring systems often need to summarize a whole pen before deciding whether a frame can be handled automatically or should be inspected more closely. This challenge asks you to classify an overhead frame by the number of distinct animal behavior states visible in the group: one state, exactly two states, or at least three states.

The source labels describe individual animals, but the submitted model receives only a deterministic privacy-preserving transform of the image. This makes the task different from detecting one animal: a useful system must combine multiple small, partially overlapping visual cues into a stable group-level summary. The hidden evaluation images include both ordinary and low-light views from the same within-farm monitoring setting.

Dataset
File descriptions
train.csv -- image IDs and relative paths for the deterministic public training frames.
train_targets.csv -- one public target for each training ID; join on id.
test.csv -- image IDs and relative paths for the unlabeled evaluation frames from deterministic frame-index holdout buckets; transformed duplicate groups stay on one side only.
train_images/ -- deterministic border-cropped (camera-overlay removed), 384x288 downsampled, and lightly blurred JPEG training views.
test_images/ -- the same deterministic transform applied to evaluation frames.
sample_submission.csv -- a valid-format submission with deterministic non-zero baseline predictions.
Column descriptions
id -- opaque row identifier; preserve it exactly in the submission.
image -- relative path to one RGB frame under ./dataset/public/.
target -- training-only integer class: 0 means one visible behavior state, 1 means exactly two distinct states, and 2 means three or more distinct states.
Evaluation
The score is three-class macro-F1:

from sklearn.metrics import f1_score

score = f1_score(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0,
)

y_true is the hidden target and y_pred is the submitted integer class. Each class receives equal weight even though the class frequencies are not identical. The score is maximized and lies in [0, 1]. Missing, non-finite, non-integer, or out-of-range predictions are invalid. A class with no predicted examples receives F1 zero for that class.

Macro-F1 is used because the three composition states have different frequencies, while each state is operationally important: a monitoring system should not appear strong merely by favoring the common single-state frames. Equal class weighting makes rare multi-state scenes count in proportion to their decision value. The holdout is built from complete 100-frame index buckets, with every third bucket reserved for evaluation and transformed duplicate groups kept together. This prevents near-adjacent frames from leaking across the boundary and tests generalization to temporally separated parts of the recording rather than only random-frame memorization.

Submission
Submit exactly two columns in this order:

id -- every ID from test.csv, exactly once.
prediction -- integer 0, 1, or 2.
The submission must contain exactly one row for every ID in test.csv and no other rows. Example:

id,prediction
goat_0f2c0d7d6384c1bb3a11,1
goat_1939287a7c4407dca5e8,0
goat_2ab688a64a42cbbd6e90,2

Requirements
Train an image model from the supplied public frames and use a validation split that is defined using training rows only.
Keep image paths relative to ./dataset/public/ and write the final file to ./working/submission.csv.
Report macro-F1 and per-class F1 during local validation; do not optimize accuracy alone.
Keep the model path end-to-end: preprocessing, training, inference, and submission writing must run in one script.
What Not To Use
Do not recover the source COCO annotation files or the original behavior labels through a public mirror or filename search; that bypasses the image-only group-composition task.
Do not use the source archive's original train/validation/test membership, source filenames, or annotation IDs as predictive features.
Do not create a lookup table from the released image files or manually assign the hidden test classes.
