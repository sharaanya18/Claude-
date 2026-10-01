NLP Scientific Comment Lifecycle Span Generation
Editor View
This challenge is closed

No further submissions, edits, or withdrawals are possible. 
How payouts are processed

NLP Scientific Comment Lifecycle Span Generation
Creator

yenwee0804
Approver

duongnguyen
Flag Problem
Domain
Sequence To Sequence
Difficulty
Medium
Scoring
↑ Higher is better
Compute
A10G
Status
Accepted
Dataset source is visible after the challenge closes.
Description
Leaderboard
(0)
Your Submissions
NLP Scientific Comment Lifecycle Span Generation
Overview
This is an NLP sequence-to-sequence transfer benchmark for scientific-software maintenance. Scientific software teams leave comments that acknowledge shortcuts, fragile assumptions, missing edge cases, and unfinished maintenance. A maintenance tool should generate a 64-step lifecycle-span profile for each maintenance item so reviewers can estimate how long the item may remain relevant.

The supplied examples come from several scientific software projects. To keep the transfer test source-agnostic, each public input contains a deterministic privacy-preserving sketch of its real comment: a broad maintenance-cue profile and a collision-heavy coarse shape bin based only on broad length and punctuation/digit style. Rare cue-shape combinations are deliberately backed off into a shared general bucket, and a separate author-keyed record token keeps repeated prepared rows addressable without carrying source order or predictive meaning. Verbatim comment text, source paths, debt labels, and exact dates are intentionally omitted. The project identity is deliberately withheld, and the hidden test set contains entire projects not represented in training. The hidden lifecycle span is derived from real source-history events, but future removal information is not solver-visible.

The required output is a one-line generated sequence in the exact form lifecycle_profile=<64 nonnegative finite decimals separated by semicolons>. The 64 values are probability-like mass over log-duration positions from zero through 9,000 days; a weighted-position decoder turns the sequence into the lifecycle span used by the grader. A decoded span below 90 days is operationally short-lived, a span from 90 through 364.999 days is a review window, and a span of at least 365 days is persistent. These thresholds correspond to deprioritize, schedule-review, and retain-attention queue decisions respectively, but the submission is the generated lifecycle profile rather than a fixed queue token.

Dataset
File descriptions
train.csv -- Feature-only file with 15,107 training observations and exactly the same feature columns as test.csv; it intentionally has no target field or verbatim source text.
train_targets.csv -- The separate target file with exactly two columns, id and target, for the training observations. It has exactly the same 15,107 unique ids as train.csv, each id occurs once, and the row order may differ; join the files on id. No test id occurs in this sidecar. Each target is a semicolon-delimited 64-step lifecycle-span sequence. The values sum to one and linearly interpolate between adjacent log-duration positions; position j represents log1p(days) = j / 63 * log1p(9000). To decode one scalar training span, compute q = sum(j * target[j]) / sum(target[j]) and then days = expm1(q / 63 * log1p(9000)). This is a transformed real span target, not synthetic text or rows. The same 64-step sequence grammar is required for predictions.
test.csv -- 3,622 observations from two held-out scientific software projects. It contains the same privacy-preserving feature columns as train.csv and no target.
sample_submission.csv -- Submission template with one valid non-constant lifecycle-span profile for every test id.
Training target decoding
The public train_targets.csv sidecar is the intended and permitted training supervision. The two-file layout is intentional: target is not duplicated into train.csv. Join the sidecar to train.csv by exact equality of their id fields; every training id has exactly one sidecar target and no test id appears in the sidecar. The sidecar sequence is the same kind of supervision emitted at submission time. For a training row, parse the 64 numeric values as t0 through t63, compute q = (0t0 + 1t1 + ... + 63t63) / (t0 + t1 + ... + 63t63), and compute the scalar training duration as expm1(q * log1p(9000) / 63). You may fit a model to this duration or to its log1p transform. At inference, convert your scalar estimate to a 64-value profile using the same log-duration interpolation convention and emit the required one-line lifecycle_profile=<...> prediction.

Column descriptions
id (string) -- Opaque 17-character identifier beginning with x for one prepared observation.
record_token (string) -- Author-keyed opaque 21-character token beginning with r, included only so repeated coarse sketches remain separate submission rows. It has no order, source-derived lookup recipe, or predictive meaning; the author key is not released and solvers should ignore this field.
comment_cues (string) -- A deterministic, source-agnostic sketch derived from the real comment available at introduction time. It contains a generic cue profile such as profile_defect+verification and one of two token-count bands: token_band_short (0-30 word tokens) or token_band_long (31+). Rare cue-shape combinations use the shared profile_general token_band_mixed backoff bucket. It does not contain the comment's lexical sequence, source identifiers, labels, or dates.
comment_shape (string) -- A deliberately coarse shape sketch from the real introduction-time comment. It contains a broad length bin (len_tiny, len_short, len_medium, len_long, or len_very_long) and a broad style bin (style_plain, style_numeric, style_punctuated, or style_mixed). Rare cue-shape combinations use shape_other; the public transformation keeps every released cue-shape bucket supported by at least twelve valid source rows.
Evaluation
Submissions are scored using Horizon-Balanced Lifecycle Span Utility, a bounded maximize metric. The grader decodes each submitted 64-step lifecycle_profile into one scalar span using its weighted log-duration position. It compares durations on a log1p scale because a factor-of-two error has similar operational meaning for a 10-day estimate and a 1,000-day estimate. A factor-of-two error receives span utility 0.5. The final score gives each of the three operational horizons equal weight because short-lived, review-window, and persistent maintenance items each require useful estimates in a real triage queue. A predictor that always emits a persistent span cannot borrow credit from the larger persistent group: its short-lived and review-window components are still scored against their own durations and remain low when the estimate is far away.

For row i, let d_i be the hidden lifecycle duration and p_i be the submitted duration. Let h(d) map a duration to 0 for <90, 1 for 90 <= d < 365, and 2 for >=365. Define S_i = exp(-abs(log1p(p_i) - log1p(d_i)) / log(2)). Let B_h be the mean of S_i over rows with true horizon h, using 0 when a horizon has no support. The final score is clip(mean(B_0, B_1, B_2), 0, 1). Higher is better.

The evaluator is equivalent to this complete code:

import math
import re
import numpy as np
import pandas as pd

HORIZON_CODES = {"short_lived": 0, "review_window": 1, "persistent": 2}
TARGET_SEQUENCE_LENGTH = 64
MAX_TARGET_DAYS = 9000.0
PROFILE_RE = re.compile(r"^lifecycle_profile=([0-9]+(?:\.[0-9]+)?(?:;[0-9]+(?:\.[0-9]+)?){63})$")
LOG_TWO = math.log(2.0)

def parse_days(value):
    match = PROFILE_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError("prediction must be lifecycle_profile=<64 nonnegative decimal values>")
    values = np.asarray([float(token) for token in match.group(1).split(";")], dtype=float)
    if (len(values) != TARGET_SEQUENCE_LENGTH or not np.isfinite(values).all()
            or np.any(values < 0.0) or np.any(values > 1.0)):
        raise ValueError("lifecycle_profile values must be finite and within [0, 1]")
    mass = float(values.sum())
    if mass <= 0.0:
        raise ValueError("lifecycle_profile must contain positive sequence mass")
    expected_position = float(np.dot(np.arange(TARGET_SEQUENCE_LENGTH), values) / mass)
    days = math.expm1(expected_position / (TARGET_SEQUENCE_LENGTH - 1) * math.log1p(MAX_TARGET_DAYS))
    if not math.isfinite(days) or days < 0.0 or days > MAX_TARGET_DAYS:
        raise ValueError("decoded lifecycle span is outside the supported range")
    return days

def horizon(days):
    if days < 90.0:
        return HORIZON_CODES["short_lived"]
    if days < 365.0:
        return HORIZON_CODES["review_window"]
    return HORIZON_CODES["persistent"]

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    required = ["id", "prediction"]
    if list(submission.columns) != required:
        raise ValueError(f"Submission must have columns {required}")
    allowed_answer_columns = set(required) | {"visibility"}
    if not set(required).issubset(answers.columns) or not set(answers.columns).issubset(allowed_answer_columns):
        raise ValueError("Answers must contain id,prediction and may include visibility")
    answers = answers[required].copy()
    if len(submission) != len(answers) or submission["id"].duplicated().any() or answers["id"].duplicated().any():
        raise ValueError("Row count and ids must be one-to-one")
    if set(submission["id"]) != set(answers["id"]):
        raise ValueError("Missing or unknown ids")
    ordered = submission.set_index("id").reindex(answers["id"])
    if ordered["prediction"].isna().any():
        raise ValueError("Missing prediction")
    predicted_days = np.asarray([parse_days(value) for value in ordered["prediction"]], dtype=float)
    true_days = np.asarray([parse_days(value) for value in answers["prediction"]], dtype=float)
    true_horizons = np.asarray([horizon(value) for value in true_days], dtype=int)
    span_utility = np.exp(-np.abs(np.log1p(predicted_days) - np.log1p(true_days)) / LOG_TWO)
    balanced_components = []
    for code in range(3):
        support = true_horizons == code
        balanced_components.append(float(span_utility[support].mean()) if np.any(support) else 0.0)
    balanced_span_utility = float(np.mean(balanced_components))
    return float(np.clip(balanced_span_utility, 0.0, 1.0))

Submission
Format
Submit exactly two columns: id and prediction.
Include each test id exactly once.
Set prediction to the exact form lifecycle_profile=<64 nonnegative finite decimals separated by semicolons>.
Encode the predicted span over the 64 log-duration positions from 0 through 9,000 days. A decoded span below 90 days is a short-lived handoff, a span from 90 through 364.999 days is a review-window handoff, and a span of at least 365 days is a persistent handoff.
Include exactly 3,622 rows.
Example
id,prediction
x0016555fb9ad5a49,lifecycle_profile=1.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0
x001d73716b9dc89e,lifecycle_profile=1.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0

Requirements
Join the provided train_targets.csv supervision to train.csv by id before training; the join is one-to-one and covers every training row, while test ids are absent from the sidecar. Train a sequence-generation model using only the supplied public files and generate one 64-step lifecycle-span profile for every row in test.csv.
Treat the test set as a project-disjoint transfer set; do not infer hidden project identity from ids or file ordering.
Use the supplied maintenance-cue sketch and metadata available at introduction time. The public train_targets.csv sidecar is allowed as the only lifecycle supervision; no hidden test labels are provided or permitted.
The benchmark is intended to measure reproducible NLP sequence generation and transfer to unseen projects. Any method is allowed if it uses the public inputs, obeys the 64-step sequence contract, and generates profiles without future labels or source history.
The labeled training rows are the development data; there is no separate public validation file. Use cross-validation or a project-aware split inside the training rows when selecting a model, and use the unlabeled project-disjoint test set only for final scoring.
What Not To Use
Do not reconstruct the held-out project identity from source filenames, opaque ids, row position, or external provenance records.
Do not query any public provenance record, mirror of a released source collection, or source repository to match input records and recover hidden test lifecycle spans; source-history records are not solver inputs.
Do not use raw source-history files, commit logs, future versions of the projects, or a repository-specific lookup table to recover hidden test removal dates or lifecycle spans.
Do not copy hidden test labels or construct an external lookup table; that would bypass the held-out-project transfer setting. The provided public training sidecar is the intended supervision and is allowed.
