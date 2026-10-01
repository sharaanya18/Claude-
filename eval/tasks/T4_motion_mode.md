# Task T4 — Shared-workspace motion-mode sequence prediction (paraphrased description)

Each row is one 4x7-bit history card sequence (4 past snapshots, 7 boolean channels, oldest first) from a shared human/robot workspace. Predict the next 10 steps as one of 7 per-step modes (S = still; A,B,C,D,E,F motion modes; A/C and B/D are an observed/commanded pair for the two symmetric arms).

Metric = 0.3 * exact macro-F1 + 0.5 * tolerant macro-F1 (greedy +-1 step class matching) + 0.2 * onset utility (onset = first non-S step; utility decays linearly with the step gap between predicted and true onset; 0 if either never activates). Every row is scored independently and must be predicted independently of other rows.
Data: a few thousand training rows with 10-step targets; rows may be fixed-length windows cut from longer recordings; the private test set may contain history/workflow types that are rare or absent in train.
Rules: CPU or GPU, ~1 hour; train in-script; no external data; no test-set-wide statistics.
Submission: id plus a 10-character string of mode letters per row.
