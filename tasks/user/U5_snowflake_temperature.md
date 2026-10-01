# U5 — How cold was it (user-supplied, verbatim; Domain: From Scratch, Medium, CPU only)

## Overview
Three photographs of one falling snowflake, taken at the same instant from three directions; predict the air temperature beside the cameras in degrees Celsius. A snowflake's shape is a record of the air it passed through (needles/columns in some ranges, plates/stars in others; near freezing flakes clump, rime or melt; in hard cold they stay small and sharp), but the air beside the cameras is not the air high above where the crystal grew, so the record is loose and no single view shows the whole flake.
Training rows 24,939 with temperature; test rows 4,958 unlabelled.

## The target
Air temperature measured beside the cameras (thermometer), degrees Celsius, one decimal. Training range -15.5 to 3.7, median -4.8, quartiles -8.5 and -1.9. The training median -4.8 is the metric's reference value.

## The rows
One row = one snowflake: three grey-level views, 80x80 each, three cameras from different directions (same three cameras in all rows). Every view keeps the flake's true size (one fixed pixel scale; a large flake can run off the edge). Weeks are split, not flakes: test rows come from 13 weeks that give no training row; training from 23 other weeks. Days are thinned: a training day gives at most 400 flakes, a test day at most 80 (random among that day's flakes). Only falling snow (blown-up ground snow excluded). Not given: date, time of day, other weather, flake type labels; rows are in an uninformative order.

## Evidence
train_images.npy uint8 (24939,3,80,80); test_images.npy uint8 (4958,3,80,80). Entry [r,v,y,x]. Flake bright on dark background (0 = empty). The three views are not aligned with one another. View order is the same in every row. The CSV evidence column names file and row such as test_images.npy:17.

## Evaluation
e = mean absolute error of your temperatures; b = MAE of answering every row with -4.8. score = max(0, 1 - e/b). Constant -4.8 scores 0 (so does anything worse). Public and private leaderboards each apply this to their own share of test rows. A malformed row is scored as the reference value.

## Validating
Validate by week: validation_folds.json assigns every training row to one of 5 folds so each calendar week lies in a single fold. Fold scores spread widely; test weeks can sit warmer or colder than the training ones. Judge by pooled out-of-fold error.

## From-scratch requirement
Every fitted or learned state (network weights, tree structures, split points, regression coefficients, scalers, calibration maps, any threshold or hyperparameter chosen by looking at data) must be created from the released training rows inside solution.py during the graded run. Networks start at random; models with nothing to initialise (gradient-boosted trees, linear fit, kernel machine, neighbour index) satisfy this by being fitted there. Loading weights or any fitted state produced beforehand does not.
Allowed: CNN/attention/other networks over the views (one encoder per view or shared, any combination); features you compute yourself (size, outline, texture, brightness, branching, symmetry) with linear/tree/kernel/neighbour models; any loss that fits the metric such as absolute error, and any output calibration fitted on your own week folds; augmentations built from the released training rows (flips, right-angle rotations of a view, small shifts, brightness changes, reordering the views) and averaging predictions over such transforms at inference.
Not allowed: any pretrained or externally trained weight/embedding/feature extractor; any external data; any attempt to date a row, identify a snowflake, or match against outside snowflake collections (automatic disqualification); a rule engine as the predictor (hand-written image processing may shape features, but the temperature must come from components fitted in solution.py); hosted inference or network access; any use of test rows beyond ordinary per-row inference (no training on test, pseudo-labelling, test-time adaptation, statistic fitted over the test data as a whole, or grouping of test rows that look like the same snowfall); hard-coded predictions or fingerprinting identifiers/row order.

## Dataset files
train.csv (id, evidence, temperature_c), test.csv (id, evidence), train_images.npy, test_images.npy, sample_submission.csv (id, temperature_c), validation_folds.json (n_folds, leakage_unit, guarantee, rows_per_fold, fold_of), task_manifest.json.
Submission: CSV id,temperature_c, one row per test id, any order, no extra/missing/duplicate/unknown ids.
Notes: part of every error cannot be recovered from the flake; test has only 13 weeks; weeks differ more than flakes; expect a gap between fold scores and test score.
Compute: CPU only - 10 cores, 62 GB memory, no GPU, no network - 1.5 hours wall clock for training and inference together. Output: submission.csv in the working directory.
