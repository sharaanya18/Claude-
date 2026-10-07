# same_reader (GoEmotions pair: same annotator or different?)

Overview: Given an online comment and two emotion categories that each received one endorsement in a small annotation panel, estimate whether the same person selected both categories or whether different people selected them. Record-level fact about the observed panel.

Data: id, text, emotion_a, emotion_b, annotator_count (distinct annotations for the comment). Each listed emotion has exactly one endorsement. train.csv 12,427 rows, test.csv 3,659 rows, train_labels.csv id,p_same_reader (1 when one annotation selected both categories, 0 when separate annotations supplied them). Rows selected at most once per author / thread / exact normalized text / near-verbatim group. Train and eval use disjoint comment groups.

Metric: average precision of the positive class (sklearn average_precision_score). Probabilities in [0,1]; larger = more likely same-reader.

Submission: CSV id,p_same_reader, one row per test ID, finite probs in [0,1].

Rules: Train an ML model on supplied labels; predictions from the trained model. General-purpose pretrained HF/timm models may be loaded. No external data, synthetic training data, hardcoded answers, saved models/predictions from earlier runs, remote inference, installs, shell commands/subprocesses, host inspection. Train on supplied data only; test only for prediction.
Baseline to beat: AP 0.5 (user report: current script scores below it).
