# Phase 5b - Reference reproduction plan

No reference solution was supplied (see `private_lb_reference_analysis.md`), so
this plan reproduces the published reference *tiers* to prove the harness is
sound, then improves on them along four named axes.

## Step 1 - reproduce the tiers (done, `exp01.py`)
Independent implementations, scored on site-held-out OOF with the exact metric:

| Tier | brief (eval set) | mine (OOF CV) |
|---|---|---|
| 2 scalars + LR | 0.0678 | 0.0287 |
| 5-NN in log-spectral space | 0.1721 | 0.1352 |
| LR / GB on hand-built features | 0.2165 | 0.1818 / **0.2295** |
| CNN from scratch | 0.1993 | not attempted (see below) |

The ordering is reproduced and the top tier is matched (0.2295 vs 0.2165). My
numbers run slightly low on the weaker tiers, as expected: these are 5-fold
station-held-out OOF estimates on 239 stations, not scores on their evaluation
shard, and my 2-scalar pair is not their 2-scalar pair. The harness is
directionally consistent, which is what the brief asks for.

The from-scratch CNN is deliberately **not** reproduced: the brief reports it
*below* the hand-built features while consuming 53 of 90 available minutes, and
239 independent labelled units is too few to train one. Spending the budget
there would trade a measured loss for an unmeasured hope.

## Step 2 - what to improve, and why each should transfer
1. **Better source/path cancellation.** One record is source x path x site;
   only the site term is wanted. Add families that cancel the shared term:
   coda-window H/V (late coda approximates a diffuse field, where H/V is tied
   most directly to the site response), early-to-late H/V drift, and spectra
   taken relative to their own smooth trend. *Measured: group K worth +0.011.*
2. **Variance reduction, not capacity.** Multi-seed averaging, several views of
   the same record (sub-window crops, averaged at inference), and a small
   ensemble of genuinely different families. On 239 units this is the most
   reliable source of private-board gain.
3. **A macro-F1-aware decode.** Macro-F1 scores a never-predicted band as 0.
   With four ordered bands and a dominant stiffness latent, the middle bands
   are under-predicted by a plain argmax. A per-band weight vector fitted on
   training-fold material to push the predicted marginal onto the training
   prior costs nothing at inference and stays row-local.
4. **Use the cross-cell structure, not a monotonic constraint.** Cells
   correlate 0.71-0.88 but only 52.8% of records are non-decreasing, so
   ordering must not be enforced. The legitimate route is a structured decode
   against the empirical joint distribution over profiles (68 of 256 occur),
   with the prior taken from the training folds only.

## Step 3 - rejected ideas
* Forcing `cell4 >= cell3 >= cell2 >= cell1` - wrong on ~47% of training records.
* Any clustering of evaluation records into stations, or group-level voting -
  explicitly forbidden, and the boards are split by record.
* Pseudo-labelling the evaluation set, test-fitted scalers/PCA, test-set
  quantile decoding - forbidden.
* A large neural network - measured to lose on this corpus.
