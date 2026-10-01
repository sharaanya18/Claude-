# U3 — Speaker-disjoint listener-vote distribution estimation (user-supplied, verbatim)

## Background
Different listeners can assign different perceived-expression labels to the same vocal performance. A useful model should represent that disagreement rather than collapse every clip to one class.

## Task overview
Given a spoken audio clip, estimate the empirical distribution of five listener responses over seven labels:
anger, disgust, fear, joy, neutrality, sadness, surprise

The target is a listener-response distribution for an acted vocal expression. It is not a claim about the speaker's internal mental state. Training and test speakers are disjoint. The opaque actor_group column is supplied in both feature tables so each speaker can remain within one validation fold.

## Distinctive structure
The benchmark retains all five listener judgments as an empirical probability vector instead of collapsing disagreement to a majority label. A successful model must learn acoustic representations from random initialization, transfer them to speakers absent from training, and calibrate probability mass across seven competing perceptions. The normalized distribution score rewards the geometry of partial agreement and gives no benefit to a uniform predictor.

## Public files
- train.csv 5,568 rows: labeled training clips
- test.csv 1,334 rows: unlabeled test clips
- response_vocabulary.json 7: required label order
- sample_submission.csv 1,334: required submission schema
- audio/ 6,902 mono 16 kHz WAV files

## Input schema
clip_id (opaque unique clip identifier); audio_path (relative WAV path); actor_group (opaque speaker-group token); response_distribution_json (training-only JSON probability object).
Every target object has exactly the seven vocabulary keys. Its values are multiples of 0.2, lie in [0,1], and sum to 1.

## Submission format
Exactly two columns in this order: clip_id,response_distribution_json
The JSON object must contain each vocabulary key exactly once. Values must be finite numbers in [0,1] whose sum differs from 1 by at most 1e-6.
abc123,"{""anger"":0.2,""disgust"":0.0,""fear"":0.2,""joy"":0.0,""neutrality"":0.4,""sadness"":0.2,""surprise"":0.0}"

## Evaluation
Let y_i and p_i be the seven-value target and submitted distributions. Uniform u=(1/7,...,1/7):
L_model = sum_i ||p_i - y_i||_2^2 ; L_uniform = sum_i ||u - y_i||_2^2 ; score = clip(1 - L_model / L_uniform, 0, 1)
Perfect = 1; uniform baseline = 0. Maximised. Squared distance evaluates the complete seven-label response profile, giving partial credit when a prediction captures listener disagreement while penalizing confident mass on unsupported labels.

Multiple clips from one speaker can share voice, channel, and performance characteristics. Keep each actor_group entirely within one validation fold. The official train and test partitions contain disjoint speakers, and the opaque group tokens do not encode response labels.

## What not to use
Train from random initialization using only the supplied public files. Do not use pretrained audio or speech models, external corpora, external retrieval, hidden answers, speaker/source lookup, or manual test labeling. The complete solution must finish within 90 minutes on one GPU.
