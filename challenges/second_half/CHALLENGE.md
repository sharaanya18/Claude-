# Whose Second Half Is It  (verbatim description as supplied by the owner)

## Overview
Every row of this challenge is a small group of six people, each seen on one day of music listening. For each of them you get the first half of that day's listening, as an ordered sequence of plays. Separately, you are handed six short "continuations": for each of the six people, the first two artists they went on to play in the second half of the same day that they had not played in the first half. The continuations are shuffled. Your job is to work out which continuation belongs to which person.

The six people in a row were grouped because their first halves resemble one another, so broad taste alone rarely tells them apart: they tend to like the same kind of music, at similar levels of popularity. What separates them is finer: which specific artists each person was drifting toward, learned from how thousands of other listeners move from one artist to the next. Every artist, release, recording and person is an opaque token, so all knowledge has to be learned from the listening data supplied here.

## Objective
The prediction unit is one row: six first halves (prefixes) and six candidate continuations. For each row return six integers, match_1 to match_6. match_i is the position (1 to 6) in the row's candidates list of the continuation you believe belongs to the i-th prefix of the row's prefixes list.

Two capabilities are coupled. The first is recommendation: a continuation never contains an artist that appears in its own prefix, so linking the two needs learned artist-to-artist affinity from the listening days. The second is assignment: each continuation belongs to exactly one of the six prefixes, so the six answers of a row constrain each other. A continuation you give confidently to one prefix cannot also be right for another, and deciding the six entries jointly is part of the task.

## Inputs (public directory)
listens.csv: one row per play: session, relative_position, seconds, half, artist, release, recording. session = opaque token for one person's day (single UTC calendar date). relative_position = 0-based play order within the day / number of plays that day, rounded to 6 decimals, in [0,1), increasing and distinct within a session (sort by it for exact order; several plays can share the same second). seconds = whole seconds since the session's first play. half = 1 for the first floor(n/2) plays, 2 for the rest. artist/release/recording are opaque tokens; release empty when not reported. 1,150,550 plays from 23,990 sessions. Sessions in test.csv are published with their first half only; every other session is published in full.
continuations.csv: continuation, rank, artist, release, recording. rank is 1 or 2 in order of first play; the other columns describe that artist's first play in the second half. Every continuation holds exactly two distinct artists.
train.csv: 1,081 rows: id, prefixes (JSON list of six session tokens), candidates (JSON list of six continuation tokens), match_1..match_6 (true answer). Full days of these sessions are in listens.csv.
test.csv: 597 rows: id, prefixes, candidates.
sample_submission.csv: exact layout, one row per test id; seeded random permutations, no information.

## Outputs
For each test row, six integers match_1..match_6 in 1..6. match_i = k: the continuation listed k-th in the row's candidates is claimed for the i-th session listed. True answer is a permutation of 1..6. Repeats allowed but cost score.

## Submission format
id (string, every test id exactly once); match_1..match_6 integer 1..6 (3.0 accepted). Ids compared as strings, whitespace stripped. 597 rows.

## Evaluation
chance_corrected_matching_accuracy. A = mean over rows of (1/6) sum_i 1[p_ri == t_ri]; score = (A - 1/6)/(1 - 1/6). Constant answer / published order / prefix-ignoring answer = 0; perfect = 1; lowest -0.2.

## Preprocessing
One daily export of a public-domain listening log. People spanning > 3 days removed. Only the busiest UTC date kept per person, so each session is exactly one day, ordered by play time. People, artists, releases, recordings replaced by salted opaque tokens; artist identity = artist credit name after trimming and case folding. A play with unreported release keeps artist and recording tokens, empty release cell. Rows, prefixes and candidates are listed in token order, which carries no information.
A continuation: walking through the second half in play order, the first two distinct artists that did not occur in the first half, and that occur in the first half of at least three sessions across the data. Only sessions with at least 16 plays appear in rows. A row groups six sessions whose first halves are similar in taste, and no artist of any continuation in a row occurs anywhere in the published plays of another person of that row. Training rows are built exactly like test rows.

## Constraints
CPU-only: 10 cores, 62 GB, no GPU. One script: python solution.py <public_dir> <submission_out>; must finish within 90 minutes (read public files, train, write submission in a single run). Cap thread pools to the 10 cores; avoid nested parallelism.

## Rules
- Genuine training or fitting inside the script. Structure must be learned from training material, not hardcoded; if the pipeline still works with the learned model removed it is not valid.
- Training material = train.csv, continuations.csv and the plays in listens.csv of every session that does not appear in test.csv. Representations and models must be fitted on the training material only.
- No external dataset; no solver-generated synthetic training data.
- The test set is used per row at inference time. Deciding the six entries of one row jointly is allowed and intended; using other test rows, fitting anything on the first halves of test sessions, pseudo-labelling or any statistic computed across the whole test set is not.
- Answers must come from the model's reading of the plays: ids, token spellings, file order and row order carry no information and may not be used as signals.
- Hyperparameter search runs inside the same script within the time limit.

## Edge cases
Repeated values within a row are valid; each true position credited at most once. 4.0 read as 4; 4.5, 0, 7 invalid. Prefixes vary in length 8..1,123 plays; long prefix may contain artists that never appear elsewhere; rare artist tokens carry little co-listening evidence. A continuation artist can be absent from every training day even though it opens >= 3 sessions in the data (some belong to test rows). About 4% of plays have empty release. A single row graded alone uses the same formula; score lies in [-0.2, 1].
