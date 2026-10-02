# Tabular, scientific (chemistry/biology/physics) and forecasting tasks

## Recognise
Rows of numeric/categorical features, molecular/protein strings, sensor aggregates, or entity-history features; target
numeric, class, ranking within a candidate set, or a per-group aggregate. Also "calibration/reference-bank" tasks
(small observed sample + bank of reference entities → unobserved quantity) and ranking inside a context (4 candidates,
pairwise-weighted metric).

## Default build
1. **Contract first**: the decision unit is often a *context* (all candidates in one solute/solvent context), not a
   row. Re-implement the metric; derive per-row/per-group weights (A2/A7); check whether the metric is invertible (A1).
2. **Baseline**: LightGBM (+ XGBoost/CatBoost for diversity) on grouped 5-fold CV, exact metric. Add a regularised MLP
   or two-tower/set network only where it brings a different assumption.
3. **Ceiling diagnostics**: identical-input/different-label rate, group-size slices, leave-group-out score of a memoriser.
4. **Domain features**: unit normalisation, composition-weighted blend descriptors, both linear and physically-shaped
   forms of rate variables, explicit products for trees, non-ideality terms, siblings-relative features, summary
   statistics; reimplement a banned toolkit's published algorithms exactly (features-and-representations F3, F8).
5. **Metric-aware target**: back-solve latent scales, RankNet or asymmetric hinge, hierarchical weights.
6. **Two structurally different models blended per context type** (GBDT + neural/pairwise), z-scored within context.
7. **Final**: refit on 100%, 5–10 seeds, fixed counts; optional in-script Optuna with `TPESampler(seed)`, fixed
   `n_trials`, no timeout.

## Task shapes and what worked
- *Pairwise/ranking inside a context* (solvent quartet: six independent solvers converged on): metric-aware regression
  target reconstructed from pair weights; mass↔mole unit normalisation; composition-weighted blend descriptors in both
  bases; two model families blended per context; solute-grouped CV on the exact metric. Our own honest-but-plain
  LightGBM scored below all six.
- *Reference-bank calibration* (observed subset + bank → hidden item difficulty): many estimators of the same
  quantity as parallel features (K), self-match exclusion (I1), distributional heads, case-level cross-item attention;
  held-out-reference-as-synthetic-case is defensible only as resampling which real slice is observed (L).
- *Aggregate labels / LLP* (genre census, composition vectors): mean-pooled linear baseline, DeepSets mean+max with
  aux head, dual-form ridge (n ≪ d), Beta-Binomial MC decode for Bray-Curtis (validation, metric files).
- *Occupancy/sensor lifecycle regression* where inputs are tiny categorical strings: from-scratch additive log-linear
  head → interaction MLP → tiny encoder-decoder (capacity ladder, 1-SE rule); decode the metric's kernel (a
  mode-seeking Laplace utility ≠ mean); beware that row-level CV is optimistic when hidden units are projects.
- *Regression with censoring/clipped labels*: asymmetric loss for bounds, quantile or distributional heads.

## Forecasting / temporal
Chronological splits, strictly backward lags, sin/cos cycles, per-entity past-only features, rolling-origin CV with
the deployment gap's magnitude (B7). Direct vs recursive horizon by CV. GBDT + a sequence model for diversity.

## Small N discipline
Distinct entities, not rows, set capacity. Strong regularisation, shallow models, domain normalisation, physical
units from metadata, best-checkpoint on a *validation fold*, repeated CV; no label smoothing that stalls at ln K.

## Pitfalls
Target-encoding leaks (leave-one-out, mismatched statistic sizes); outcome-adjacent columns; random KFold on grouped
rows; reading ids/order; regex "solving" a templated synthetic dataset (strip-the-ML) — feed the structure to a model
instead; blindly averaging members with different scales; selecting on the OOF you report.

## Research additions (2026-10, /research/B)
- Three GBDT libraries at meta-tuned defaults + a capped feature factory (arithmetic, group-conditional, pseudo-categorical features) with target/group statistics cross-fitted inside the outer-training fold; a regularised MLP family (TabM-style BatchEnsemble MLP / RealMLP-TD settings, implemented from scratch) as the structurally different member; blend per ensembling-and-training.
- Metric-aware: F1 threshold = F*/2 on calibrated scores, log1p for RMSLE, regress-then-few-OOF-thresholds for ordinal/QWK tasks (public-board threshold chasing caused shake-ups).
- Forecasting: horizon-length rolling-origin holdout with a gap, strictly backward lags, global GBDT + causal sequence NN for diversity; no online learning on test.
- Early-stopped fold scores are optimistic: pick rounds, then re-score cleanly.
