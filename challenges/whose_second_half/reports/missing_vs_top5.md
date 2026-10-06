# Whose Second Half: things the top-5 used that our approach did not

Source: five ranked private-LB solutions supplied by the user (ranks only, no scores). Read from code, nothing re-run, no gains measured.
"Our approach" = the submitted pipeline (public 0.54, user-reported). Details per solution: `top5_digest.md`.

## Evidence and features
- **New-in-second-half target:** "first-half artist -> artist new in the second half" matrices, kNN votes against those artists, and a first-half -> second-half autoencoder (ranks 1, 2, 3). Ours uses all plays.
- **More embedding views:** several skip-gram variants (wide window, window 2 to 5, sittings cut by time gaps, whole-day random pairs); release- and recording-level versions in ranks 1 and 3. Ours has one window-20 model.
- **Input and output vectors:** rank 1 scores in.out and out.in as well as cosine; rank 3 also scores in.out. Ours uses cosine of input vectors only.
- **Other embeddings:** PPMI-SVD, LSA, all-graph SVD (ranks 2, 3, 4, 5).
- **Windowed session kNN:** corpus days cut into 25- or 50-play windows, matched at artist, release and recording level, with query expansion (rank 2).
- **Time features:** recency in seconds, played-in-last-sitting flag, gap statistics (ranks 3, 4).
- **Listener-habit features:** repetition, run length, album share, track popularity (rank 4); empty-release rate, rare-recording share, binomial likelihood of the continuation's missing releases (ranks 2, 3).
- **Row context:** how many other prefixes in the row share an artist; similarity to the other prefixes' centroids (rank 3).

## Models and training
- **Row-aware head:** layers mixing a pair with its row, column and whole-row means (ranks 1, 2, 4). Ours scores each pair independently.
- **Token-level interaction head:** attention over the prefix artists instead of a pooled score (rank 1 only).
- **Gradient-boosted models:** LightGBM / CatBoost (ranks 2, 3, 5); rank 3 used a custom 720-permutation objective, rank 5 added linear heads on the same likelihood.
- **Bigger, randomised ensembles:** rank 1 averages 20 members, each with a random 70 % of channels and 80 % of group features; rank 2 averages 20 networks. Ours averages 3 seeds.
- **Tuning inside the script:** Optuna with 24 seeded trials (rank 4); blend weight, temperature and decoder chosen on out-of-fold predictions (ranks 2, 3, 4). Ours tunes temperature only.

## Pipeline
- **Test scored under each fold's corpus and averaged** (ranks 2, 4). Ours builds training features from 20 fold pools and test features from the full pool.
- **Feature copies:** row-max and column-max normalised copies, z-scores and ranks within prefix and within candidate (ranks 2, 5).
- **Diagnostics:** unseen-fold scores, single-seed spread, strata, train-vs-validation gap, style ablation (rank 4).

## Where the gap most likely sits (hypothesis)
The evidence (target, views, in.out) and the row-aware head, not our validation setup.
