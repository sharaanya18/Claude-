Align Contact Episodes Across Human and Robot Manipulations
Editor View
This challenge is closed

No further submissions, edits, or withdrawals are possible. 
How payouts are processed

Align Contact Episodes Across Human and Robot Manipulations
Creator

kindasomethin
Approver

rathore_aditya
Flag Problem
Domain
Fine-Tuning
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
Align Contact Episodes Across Human and Robot Manipulations
Overview
This is a GPU fine-tuning challenge about recognizing the same physical interaction across very different bodies and cameras.

Every example comes from one articulated object, such as a drawer, cabinet, refrigerator, or hinged panel. The object was opened and closed separately by a bare human hand, a human wearing a wrist camera, a handheld robot gripper, and an instrumented gripper. The four executions differ in speed, viewpoint, hand or tool shape, occlusion, and contact force.

Each row contains four groups of three short two-frame clips. One group comes from each embodiment, but the group order changes from row to row. Within every group, the three clips are also shuffled.

Your objective is to identify the three cross-embodiment sets of clips that show equivalent mechanical episodes. A correct set contains one clip from each group. The three sets together must use every clip exactly once.

You do not submit three independent labels. You submit a calibrated 3 x 3 x 3 x 3 coupling tensor. A cell is large when its four candidate clips are likely to represent the same episode.

The task is intentionally harder than matching static object configurations. Every row contains two opening states and one closing state. One opening state and the closing state occupy similar object configurations while moving in opposite directions. Correct recovery therefore requires temporal direction, cross-view appearance, and, where available, force and tactile evidence.

Domain: Fine-Tuning.

Hardware: one NVIDIA A10G GPU, 10 CPU cores, and 62.5 GiB RAM.

Runtime limit: 60 minutes.

Real-world collection process
The source is an open laboratory and in-home collection of 3,048 articulated-object manipulation sequences covering 381 physical objects in 38 environments. The benchmark retains the 143 objects in 16 environments that have complete confirmed interactions, sufficient camera coverage, and synchronized contact sensing for every required embodiment.

The same physical objects were operated under four embodiments:

a human hand observed from an external camera;
a human hand observed from a wrist-mounted camera;
a handheld robot gripper with a manipulation-centered camera;
an instrumented gripper with a manipulation-centered camera, six-axis force/torque sensing, and two fingertip tactile sensors.
Recordings were synchronized with nanosecond timestamps inside each execution. Human and robot executions were recorded separately, so matching cannot be solved by shared timestamps. Opening and closing intervals were manually confirmed in the source records. The released benchmark removes the original timestamps, paths, object numbers, and environment names.

The benchmark extracts non-overlapping two-frame microclips from those confirmed intervals. The two frames are consecutive samples near a selected interaction phase, so their difference contains short-horizon motion direction. Each original RGB frame is used in exactly one prepared row. Force and tactile values are synchronized only to the instrumented-gripper clips.

Exact prediction task
For one row, let G1, G2, G3, and G4 be the four row-local groups. Each group contains candidates numbered 0, 1, and 2 in tensor order.

Predict a tensor P with shape 3 x 3 x 3 x 3.

P[i,j,k,l] is the confidence that:

candidate i from G1;
candidate j from G2;
candidate k from G3; and
candidate l from G4
belong to the same latent mechanical episode.

A valid tensor represents three one-to-one hyperedges. Its total mass is 3, and every candidate has marginal mass 1 along its own axis.

Dataset files
The prepared public data contains six files.

train.csv contains 1,035 training rows and public row metadata.
test.csv contains 252 hidden-target rows with exactly the same feature columns as train.csv.
train_targets.csv contains the training coupling tensors.
sample_submission.csv contains the valid uniform coupling tensor for every test ID. It scores exactly 0.
train_tensors.npz contains the training RGB and auxiliary arrays.
test_tensors.npz contains the test RGB and auxiliary arrays.
The private directory contains answers.csv. Its scoring columns are the same id and predicted_coupling columns used by sample_submission.csv. It also contains an evaluator-only visibility column that never appears in public test data or participant submissions.

train.csv and test.csv columns
id: string. A deterministic anonymous row identifier.
tensor_row: integer. The row to read from the corresponding tensor asset.
group_cards: JSON list describing the four row-local tensor groups.
Training targets are deliberately stored in train_targets.csv rather than as an extra train.csv column so that train and test feature schemas are identical.

group_cards schema
group_cards is a JSON list with four objects. Each object contains:

group: string. One of G1, G2, G3, or G4.
view_kind: string. One of external_human_view, wrist_human_view, robot_gripper_view, or instrumented_gripper_view.
has_force_tactile_aux: Boolean. True only for the instrumented-gripper group.
candidate_count: integer. Always 3.
Group aliases are row-local. For example, G1 is not always the human view.

Tensor asset schema
Load an asset with:

data = np.load("train_tensors.npz")
rgb = data["rgb"]
aux = data["aux"]

The rgb array has shape:

(rows, 4 groups, 3 candidates, 2 frames, 80 height, 80 width, 3 RGB channels)

Its dtype is uint8. Frames are chronological within each two-frame clip.

The aux array has shape:

(rows, 4 groups, 3 candidates, 24 features)

Its dtype is float16.

Auxiliary positions have the following meanings:

positions 0 through 2: mean absolute RGB change from frame 0 to frame 1, separately for R, G, and B;
positions 3 through 5: mean RGB intensity in frame 1, separately for R, G, and B;
positions 6 through 17: robustly standardized six-axis force/torque at the two frame times;
positions 18 through 20: left tactile-frame mean, standard deviation, and mean spatial-gradient magnitude;
positions 21 through 23: right tactile-frame mean, standard deviation, and mean spatial-gradient magnitude.
Positions 6 through 23 are zero for groups without contact sensors. group_cards explicitly identifies whether those values are available, so zero is not a hidden missing-value code.

Training target and submission grammar
train_targets.csv, sample_submission.csv, and submitted files contain exactly two columns in this order:

id: string;
predicted_coupling: string containing exactly 81 finite decimal numbers separated by ordinary spaces.
Flattening is lexicographic with the G4 index changing fastest:

flat_index = 27 * i + 9 * j + 3 * k + l

Every predicted value must be between 0 and 1 inclusive.

The 81 values must satisfy:

sum(P) = 3

For every axis and every candidate on that axis:

candidate_marginal = 1

The grader accepts a total-mass error of at most 0.02 and a maximum marginal error of at most 0.03.

The training target tensor contains exactly three values equal to 1. All other values are 0.

Valid prediction example
The uniform tensor is structurally valid:

0.037037037037037035 0.037037037037037035 ... repeated until 81 values

It expresses no information and scores 0.

Invalid prediction examples
These predictions are invalid and receive zero for the affected row:

an empty string;
fewer or more than 81 numbers;
NaN, inf, or a non-numeric token;
a value below 0 or above 1;
total mass outside the allowed tolerance;
any candidate marginal outside the allowed tolerance.
Invalid rows remain in the overall mean and the bottom-20% mean. Abstaining can never remove a difficult row or improve the score.

Evaluation
Let Y be the true 3 x 3 x 3 x 3 tensor. It has three cells equal to 1.

Let U be the uniform valid tensor:

U[a,b,c,d] = 1 / 27

For a valid prediction P, define:

BaselineMSE = mean((U - Y)^2 over all 81 cells)

PredictionMSE = mean((P - Y)^2 over all 81 cells)

CouplingSkill = clip(1 - PredictionMSE / BaselineMSE, 0, 1)

For every unordered pair of axes (x,y), sum P and Y over the other two axes to obtain PredictedPair[x,y] and TruePair[x,y], each with shape 3 x 3. Let PairUniform be the 3 x 3 matrix whose entries are all 1/3.

PairBaselineMSE[x,y] = mean((PairUniform - TruePair[x,y])^2)

PairPredictionMSE[x,y] = mean((PredictedPair[x,y] - TruePair[x,y])^2)

PairSkill[x,y] = clip(
    1 - PairPredictionMSE[x,y] / PairBaselineMSE[x,y],
    0,
    1
)

PairMarginalSkill = mean(PairSkill[x,y] over all 6 unordered axis pairs)

This component gives visible partial credit when a model correctly associates two embodiments but has not yet resolved the complete four-way hyperedge. The uniform sample receives zero pair skill.

Define the probability mass placed on the four true hyperedges:

TrueMass = sum(P * Y over all 81 cells)

UniformTrueMass = 3 / 27

MassSkill = clip(
    (TrueMass - UniformTrueMass) / (3 - UniformTrueMass),
    0,
    1
)

The continuous row score is:

RowScore =
    0.45 * CouplingSkill
  + 0.30 * PairMarginalSkill
  + 0.25 * MassSkill

The grader also decodes a globally consistent four-way matching. It enumerates the 3!^3 = 216 assignments obtained by fixing the G1 order and permuting the other three groups. The assignment with the largest sum of its three tensor cells is selected. Deterministic lexicographic tie-breaking is used.

ExactHypermatch equals 1 only when all three decoded hyperedges equal the target hyperedges. Otherwise it equals 0.

Across the scored rows:

OverallMean = mean(RowScore)

Bottom20Mean = mean(the lowest ceil(0.20 * number_of_rows) RowScore values)

ExactRate = mean(ExactHypermatch)

ChanceExact = 1 / 216

ExactSkillRaw = clip(
    (ExactRate - ChanceExact) / (1 - ChanceExact),
    0,
    1
)

ExactEvidenceGate = min(1, OverallMean / 0.05)

ExactSkill = ExactSkillRaw * ExactEvidenceGate

The final score is:

final_score = clip(
    0.72 * OverallMean
  + 0.18 * Bottom20Mean
  + 0.10 * ExactSkill,
    0,
    1
)

All constants are public. The metric does not use a hidden reference model or hidden normalization constant. A perfect oracle scores exactly 1. The uniform sample submission scores exactly 0.

Submission format
Submit a CSV named submission.csv with exactly these columns in this order:

id,predicted_coupling

Every test ID must appear exactly once. Rows are aligned by id, never by CSV order.

Structural submission errors raise a clear error. These include missing columns, extra columns, duplicate IDs, malformed IDs, missing IDs, unknown IDs during full-set evaluation, and an invalid row count.

Malformed row-level coupling tensors receive zero rather than crashing the grader.

Generation and leakage controls
The true leakage units are the physical articulated object and its recording environment.

All rows from one physical object remain in one split.
All objects from one environment remain in one split.
Training uses 115 objects in 11 environments.
Test uses 28 objects in 5 different environments.
Training contains 1,035 rows.
Test contains 252 rows, or 19.58% of all rows.
Public/private leaderboard visibility is assigned by complete recording environment after the test set is fixed.
The public side contains 135 rows from two complete environments; the private side contains 117 rows from three different complete environments.
No recording environment or physical object crosses leaderboard visibility.
No RGB source frame is reused in another row.
Every retained instrumented clip has six-axis force and both fingertip tactile streams within 250 ms; objects failing this synchronization check are removed as complete groups.
No source interval crosses train and test.
Public IDs are salted hashes unrelated to source filenames, locations, timestamps, or object indices.
Tensor row order is deterministic but independently sorted by anonymous ID.
Candidate positions and embodiment-group positions are shuffled independently for every row.
No raw timestamp, source filename, environment name, object number, or original record ID is released.
Exact public feature-row duplicates are rejected during validation.
Near-duplicate protection follows the physical source hierarchy rather than a fragile pixel threshold: every view of one object and every object from one environment stays on the same side of the split.
What not to use
Do not interpret anonymous IDs, tensor row numbers, CSV row order, or group aliases as target signals.
Do not assume a fixed candidate position has a fixed interaction phase.
Do not assume G1 represents a fixed embodiment.
Do not recover or join hidden targets through source filenames, timestamps, locations, or object numbers.
Do not submit independent per-axis rankings that violate the four-way marginal constraints.
Do not submit JSON, natural-language explanations, filenames, videos, or extra columns.
Benchmark boundary
Existing articulated-object benchmarks commonly estimate joint type or articulation axis, regress force from one video stream, imitate a demonstration, or learn a shared human-robot policy representation.

This benchmark evaluates a different capability: calibrated, globally consistent four-way data association among separate human and robot executions, with opening/closing configuration collisions and asymmetric contact sensing. It is not force regression, action classification, ordinary pair matching, video retrieval, or policy learning. A solver must combine temporal direction, object state, viewpoint invariance, and contact evidence while satisfying a multi-marginal coupling contract.
