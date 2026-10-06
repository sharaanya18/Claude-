## 2. Final models

Seven families are fitted per depth cell and averaged with equal weights. They
were not picked off a shelf: each one encodes something about this problem's
shape, and they fail differently, which is what makes averaging them pay.

| family | what it is | why it is in the ensemble |
|---|---|---|
| **ordgb** | cumulative-link ordinal model whose three binary learners per cell (`P(band > k)`) are gradient-boosted trees | the strongest single family. Uses the band ORDER, so each of the three fits sees all 1,179 records instead of splitting them four ways - materially more sample-efficient on 239 independent stations - and costs what one four-class booster costs (3 binary trees per round against 4) |
| **hgb** | four-class gradient-boosted trees | second strongest, and the **best family on the 0-5 m cell**, where ordgb is weaker |
| **ord** | the same cumulative-link construction with logistic regression | the **best family on the 5-10 m cell**, a smooth linear surface, and it fits in one second |
| **et** | extremely randomised trees | heavily decorrelated from boosting; different error structure |
| **ldak** | shrunk LDA to a 3-D discriminant space, then a neighbour vote in it | the brief's 5-NN tier shows similarity in the right space is strong; this learns that space instead of assuming it |
| **lr** | strongly regularised multinomial logistic regression | the smooth linear reference |
| **pls** | partial least squares on all four numeric band targets at once | fits the problem's shape - many correlated inputs, few samples, four correlated ordered outputs driven largely by one stiffness factor - and is the best linear family on the 10-20 m cell |

Equal weights are used deliberately. Fitted blend weights have more freedom
than 239 independent stations can resolve, and greedy weight selection was
measured against the plain average (section 4) rather than assumed better.

Every model is constructed untrained and fitted inside `solution.py`. The five
fold models of each family also predict the evaluation records, and those
predictions are averaged over folds - a free bagging ensemble, since those fits
already exist for the out-of-fold stage.

## 5. Generalisation argument

Five independent reasons to expect this to transfer to unseen stations:

1. **The validation condition is the evaluation condition.** `folds.csv` holds
   out whole stations and no station spans two folds, so every reported number
   is a prediction for stations the model has never seen. Decisions were taken
   on the mean over several such splits, not one.
2. **There is no measurable distribution shift to transfer across.** A
   train-vs-test discriminator reaches AUC 0.618, *below* the 0.634 reached by
   a discriminator separating two disjoint sets of *training* stations. The
   evaluation stations look like a random draw, so station-held-out CV is the
   right estimator and no test-adaptive step is needed (or permitted).
3. **The representation is physical, not incidental.** H/V ratios, coda H/V and
   trend-removed spectral shape are the quantities site-response practice uses
   precisely because they cancel source and path. Horizontal polarisation
   enters through eigenvalues only, so an arbitrary E/N orientation cannot be
   learned. Nothing keys on an id, a row position or a file order.
4. **Capacity is matched to 239 units, and the selection was noise-aware.**
   Strong regularisation throughout, feature subsampling in the boosters, equal
   blend weights, and a small coarse hyperparameter search. Two families were
   built and **dropped on measurement** (Konno-Ohmachi smoothing; a
   high-frequency family that failed to lift the cell it targeted), and the
   headline ablation gains were only accepted because they held across splits.
5. **Variance is reduced by averaging, which is the one thing that reliably
   survives a new sample**: seven families x five fold models, all averaged.

Honest counterweights:
* The cells' per-band F1 shows errors are overwhelmingly **adjacent-band
  confusions**, which is the non-uniqueness the brief warns about rather than a
  fixable modelling defect.
* The two middle bands are intrinsically harder than the extremes (per-band F1
  roughly 0.28-0.42 against 0.44-0.66), and macro-F1 weights them equally, so
  a large part of the remaining gap is information the record does not contain.
* The brief states the private board has a standard deviation of about 0.019 for
  a solver near 0.20, and a public-private gap averaging 0.043. A CV figure
  should therefore be read with roughly +-0.02 of sampling noise on the private
  board, on top of any bias.
