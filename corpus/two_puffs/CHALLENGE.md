> NOTE: not verbatim platform text. Reconstructed from two of five ranked top solutions' docstrings
> and code (rank-1's feature/target construction, rank-2's docstring). Unconfirmed against a real
> sample_submission.csv; the exact output column grammar is inferred, not confirmed.

# Two Puffs (reconstructed)

Input per participant: one spirometry test session consisting of several "blows" (forced
exhalations), each stored as a base64-encoded raw flow-sensor byte stream (`flow_b64`) plus
calibration (`btps`) and quality flags (`acceptable`, `plateau`, `effort` grade A/B/C), from which
volume/flow-time curves and standard spirometry indices (FEV1, FVC, PEF, FEF25-75, MEF25/50/75,
FEV0.5/0.75/6, etc.) are derived (rank-1's `blow_indices`/`session_features`). Demographics
(age, sex, height, weight, BMI, ethnicity) and "number of blows / acceptable blows" are also given.
Reconstructed before/after structure: "two puffs" strongly suggests a bronchodilator-reversibility
test — a baseline spirometry session and a second session after bronchodilator administration —
though only one session's feature construction was visible in the inspected code.

Task (reconstructed from rank-1's 9-column target construction): predict, per participant, a
9-value probability vector: `P(FEV1 improves by >= each of the gain thresholds {0, 5, 10, 15, 20}%)`,
`P(FVC improves by >= each of {0, 5, 10}%)`, and `P(ATS/ERS-positive bronchodilator response)` —
the standard clinical criterion of >=12% AND >=0.2 L improvement in FEV1 or FVC. Both top solutions
explicitly model measurement uncertainty: rather than taking a single point estimate of "the" FEV1/
FVC from possibly-noisy repeated blows, they build a discretised joint distribution over plausible
true (FEV1, FVC) pairs and an independent "gain" (post/pre ratio) distribution, and report the
probability-weighted expectation of each clinical event under that joint distribution — i.e. the
label the model is scored against is itself a probability, not a single measured ratio, which is
presumably why the task is posed as probability estimation rather than point regression.

Implied metric: almost certainly a proper probability-scoring rule (log loss / Brier-style) over
the 9 (or fewer) probability columns, given both solutions fit by gradient descent on a squared-
error-in-probability-space objective (rank-1's `dP = 2 * (P - T) * w / 9.0`) with per-row weights
emphasising uncertain rows.

Compute: CPU-only in both inspected solutions (`OMP_NUM_THREADS`/`MKL_NUM_THREADS` pinned low,
no CUDA references at all) — consistent with a stated CPU-only environment. Rank-2's docstring
states "No pretrained parameters, stored predictions or external records are loaded" and "trains a
waveform representation on the unlabeled reference pool" — implying the challenge supplies an
additional unlabeled reference pool of flow waveforms for representation learning, separate from
the labelled train/test participants, and explicitly forbids pretrained assets ("from-scratch").

Unverified: the exact submission column names/order, the official metric formula, whether a true
second (post-bronchodilator) session is in fact part of the released data or the "two puffs" framing
refers to something else entirely, row counts, and any stated runtime/hardware limit.
