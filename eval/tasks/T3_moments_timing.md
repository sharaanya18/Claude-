# Task T3 — Mark the moments that mattered (paraphrased description)

Each row is a 24-step snippet of a grid-digging game observed through only 4 grayscale 24x24 frames (at steps 0, 6, 12, 18) plus the control codes (0-14) at those steps. Predict, for every one of the 24 steps, the probability that the game score changed at that step (columns e_00..e_23).

Metric: Brier skill score against the base rate: 0.01 + 0.99 * skill (0.01 * exp(skill) if skill < 0). Validation groups are given by `episode_group`. Different game levels use different grey palettes. Final window after step 18 has no later frame (must be forecast).
Data: several thousand training snippets with labels. One GPU, ~1 hour. Pretrained weights allowed but images are tiny synthetic frames. No external data. Training must happen in-script.
Submission: one probability per step per row, matching the sample file.
