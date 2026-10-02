# Research B: tabular, scientific-data and forecasting winning recipes (2023-2026), classified

Date: 2026-10-02. Author: eris-pattern-researcher (public sources only). Nothing here edits curated references and nothing was run
locally: every item is a hypothesis for the main session. Condensed, dated one-paragraph versions are appended to
`/home/user/Claude-/.claude/skills/eris-playbook/references/proposals.md` as P-B01..P-B12.

## 0. Reading guide, legends, limits

- Tags (exactly one per item): VERIFIED PLATFORM REQUIREMENT, GENERAL ML PRINCIPLE, PUBLIC RESEARCH, COMPETITION PATTERN,
  LOCAL EXPERIMENT RESULT (none this round), UNTESTED HYPOTHESIS.
- Source marks (same convention as `challenges/anchorperm/reports/literature_scan.md` and research A):
  [S] = page, abstract or PDF text read through a fetch this session; [S-snip] = only a search-result summary of the page was
  available (Kaggle write-up pages are JavaScript-rendered and the fetch tool returns only their titles, so every Kaggle
  write-up claim below is [S-snip]); [R] = recalled, not fetched.
- Item ids RBnn. Cross-checks name existing playbook entries (V1..V15 validation-recipes, E-A..E-D ensembling-and-training,
  F3 features, metric-and-decoding sections, engineering-and-compliance sections/Q-classes, 00-s-tier moves) and research-A
  proposals (P-A01..P-A17, in proposals.md).
- Numbers read off figures (RB01, RB15) are visual reads and approximate. The fetch tool summarises pages with a small model:
  anything marked [S] was confirmed against text it returned, but not re-checked by a second reader. One paper
  (arXiv 2602.12469, "Regularized Meta-Learning", Playground S6E1 stacking) is marked withdrawn and is NOT relied on.
- Out-of-bounds check: no Shipd/Eris solution, private code or held-out material was sought or encountered. Several winning
  solutions used things Eris bans (original/external datasets, train+test fits, test-time online learning, LLM-generated game
  data); they are quoted for METHOD only and listed in RB36 as DO NOT ADOPT.
- Cost classes are my estimates, to be profiled locally: C0 seconds, C1 minutes, C2 about 10 min, C3 tens of minutes.
  Where a paper gives a ratio it is quoted.
- Not retrievable / unverified: Kaggle write-up bodies (snippets only), the Jane Street 2025 winning approach, the full M5 paper
  (page blocked), exact licence terms of TabPFN-2.5 weights, whether `tabpfn`/`tabm`/`pytabkit`/`autogluon`/RDKit are present in
  the platform image (none is on the CLAUDE.md section 8 list), and the actual size/shape of Eris tabular tasks.

## 1. Strong-GBDT-era recipes: settings, features, encodings, categoricals/missing, GPU

### RB01 · PUBLIC RESEARCH · Literature meta-tuned GBDT defaults ("TD") as the fixed starting config
- Mechanism: Holzmueller, Grinsztajn, Steinwart meta-tuned default hyperparameters for LightGBM, XGBoost and CatBoost on 118
  datasets (1K-500K rows) and tested them on 90 disjoint ones; defaults are used with n_estimators=1000 and early stopping 300 on a
  validation split. Values read from Tables C.1-C.3 (PDF pages 40-41):
  LightGBM classif/reg: num_leaves 50/100, learning_rate 0.04/0.05, subsample 0.75/0.7 (bagging_freq 1), colsample 1.0,
  min_data_in_leaf 40/3, min_sum_hessian_in_leaf 1e-7. XGBoost (hist): max_depth 6/9, lr 0.08/0.05, subsample 0.65/0.7,
  colsample_bylevel 0.9/1.0, min_child_weight 5e-6/2.0, lambda 0, max_bin 256. CatBoost: boosting_type Plain, Bernoulli bootstrap,
  depth 7/9, lr 0.08/0.09, subsample 0.9, l2_leaf_reg 1e-5, random_strength 0.8/0.0, one_hot_max_size 15/20,
  leaf_estimation_iterations 1/20, border 254. Library defaults for contrast: LightGBM 31 leaves lr 0.1 100 trees; XGBoost
  depth 6 lr 0.3 100 trees; CatBoost Bayesian bootstrap depth 6 automatic lr.
- Expected gain: read off Fig. B.6 (meta-test): LightGBM classification error about 0.133 (library default) to 0.125 (TD) to 0.117
  (50-step random-search HPO); XGBoost regression nRMSE about 0.455 to 0.435 to 0.405. So TD captures roughly half of the HPO gain
  at about 1/30-1/50 of the training time; HPO still wins on these medium/large benchmarks. Appendix B.9: TPE vs random search at
  50 steps differ little ("often slightly better, differences relatively small").
- Cost: C0-C1 per fit (1000 trees with early stopping).
- Compliance: constants are published defaults, not found by tuning on this task (CLAUDE.md 2.1 forbids pasting offline-found
  params; a cited literature default is the "principled default" the engineering notes allow, Q3). Put the citation and the
  statement "defaults from Holzmueller et al. 2024, not tuned on this data" in a code comment. Tuned for 1K-500K rows with the
  authors' preprocessing; on far smaller N prefer shallower/more regularised variants (RB19).
- Cross-check: extends tabular-and-scientific "Default build" steps 2 and 7 (it supplies concrete starting values and a
  baseline any in-script Optuna must beat by more than noise, V4); converges with CLAUDE.md 4A "plateaus not sharp optima".
- Sources: https://arxiv.org/abs/2407.04491 [S]; PDF pages 28-31 and 40-41 read directly (https://arxiv.org/pdf/2407.04491) [S].

### RB02 · PUBLIC RESEARCH · Across-family ensemble of defaults beats any single-family HPO; winners mix GBDT libraries
- Mechanism: in the same paper the ensemble of tuned-default LightGBM + XGBoost + CatBoost + RealMLP ranks ahead of every
  single-family 50-step-HPO model on the meta-test benchmarks (Fig. B.8 average ranks, visual read; only the all-HPO ensemble and
  the per-split best of HPO models do better). Independent evidence from competitions: in the 2025 tabular winners XGBoost and
  LightGBM were each used 14 times and CatBoost 8, and "winners typically combined multiple GBDT libraries"; in 2024 LightGBM 16,
  CatBoost 13, XGBoost 8 with the same combine-libraries pattern; library choice was driven by categorical handling (CatBoost),
  speed (LightGBM) and memory.
- Expected gain: library diversity is a free first ensemble step; magnitude is data-dependent (no number in the sources).
- Cost: sum of members (C1 each).
- Compliance: all three libraries are allowed; nothing fitted on test.
- Cross-check: converges with E-A "diversity of modelling assumption beats seeds" and 00-s-tier move 6; adds that three GBDT
  libraries alone already count as measurable diversity.
- Sources: https://arxiv.org/pdf/2407.04491 [S] (Fig. B.7/B.8, pages 30-31); https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S];
  https://mlcontests.com/state-of-machine-learning-competitions-2024/ [S].

### RB03 · PUBLIC RESEARCH · Three feature families that transfer (TabPrep, 2026), all leakage-safe by construction
- Mechanism: TabPrep targets three structural patterns: (1) arithmetic structure via ordered expansion of +,-,*,/ between
  numeric features; (2) group-conditional effects via out-of-fold target encoding, target-encoded categorical combinations, and
  "relative group-by" features (numeric relative to its category-conditional statistic); (3) pseudo-categorical numerics via
  discretisation plus random-subset feature compression. All target information is nested inside inner folds on training data only.
- Expected gain: reported to improve LightGBM, TabM and foundation models on average across datasets, with LightGBM and TabM
  "catching up close to" foundation models; no per-model numbers extracted.
- Cost: C1-C2; LightGBM training time is about 6x with TabPrep (autofeat 180x, OpenFE 19x in the same paper).
- Compliance: features are inputs to a trained model (CLAUDE.md 2.2); statistics from train only; the nested encoding is the
  required form (RB05). Not a rule engine.
- Cross-check: converges with features-and-representations F3 (products/ratios, target-aware encodings inside folds, summary
  stats); adds the pseudo-categorical family and a named, bounded generator set.
- Sources: https://arxiv.org/abs/2606.02384 and https://arxiv.org/html/2606.02384 [S].

### RB04 · COMPETITION PATTERN · Feature factory at scale: aggregates, categorical pairs, digits, NaN pattern, then select
- Mechanism: the Backpack-competition winner generated >10,000 candidate features and kept the best 500 for the final XGBoost:
  groupby(COL1)[COL2].agg(mean/std/count/min/max/nunique/skew) over thousands of pairs; all pairwise combinations of 8
  categoricals (28 new); histogram bins and quantiles (5/10/40/45/55/60/90/95) of a numeric within groups; rounding a numeric to
  7-10 decimals and extracting its digits as features; the NaN pattern packed into one base-2 column; divisions of engineered
  columns. When the aggregated column is the target, nested CV is used. Other winners: "treat every number as numeric and
  categorical at once", about 270 target-encoding features in a 3rd-place Podcast model (custom TE, no smoothing, no fillna;
  stacking helped because the most important feature was heavily missing) [S-snip].
- Expected gain: the dominant lever in 2025 Playground wins (author-reported, no ablation numbers).
- Cost: C2-C3 for a large candidate pool; a capped pool (a few hundred candidates, selection by in-fold importance) fits a
  45-minute plan.
- Compliance: group statistics computed on train rows only, test rows mapped through train-fit tables (unseen category -> global
  prior). The Playground versions that merged the "original" dataset (the Backpack winner merged original MSRP values) or computed
  counts on train+test are banned (RB36).
- Cross-check: converges with F3 and V8; the cap-and-select step guards the "many moving parts" anti-pattern (00-s-tier section 6).
- Sources: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/ [S];
  https://www.kaggle.com/c/playground-series-s5e4/discussion/575862 [S-snip]; https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5 [S-snip; page returned 403 to the fetch].

### RB05 · GENERAL ML PRINCIPLE · Target/group-statistic encodings: cross-fit inside every outer training fold
- Mechanism: scikit-learn documents that `fit(X, y).transform(X)` differs from `fit_transform(X, y)` because the latter cross-fits
  (default cv=5) so each training row is encoded from the other folds; `fit` alone "can introduce data leakage"; unseen
  categories get the training target mean; `smooth="auto"` is an empirical-Bayes shrinkage. Inside an outer CV the encoder must be
  built from the outer-training rows only (cross-fit for those rows, plain transform for the validation rows), otherwise the
  OOF score is inflated.
- Expected gain: protects against a common private-LB killer rather than adding score; singleton or tiny categories are where
  the shared-prior leak shows (one GitHub issue, weak [S-snip]).
- Cost: C0-C1.
- Compliance: fit on train only (CLAUDE.md 2.3 #5); count/frequency tables also train-only.
- Cross-check: independently confirms V8 and P-A11 (leakage battery); the sklearn doc is the verifiable reference.
- Sources: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html [S];
  https://scikit-learn.org/stable/modules/preprocessing.html [S]; https://arxiv.org/abs/1706.09516 [S] (CatBoost ordered target statistics: the same leakage problem solved inside the booster).

### RB06 · PUBLIC RESEARCH · Native categorical and missing handling: settings that matter
- Mechanism: LightGBM docs list cat_smooth 10, cat_l2 10, min_data_per_group 100, max_cat_threshold 32, max_cat_to_onehot 4
  (cat features capped at int32 range); XGBoost enable_categorical uses max_cat_to_onehot and max_cat_threshold; CatBoost uses
  ordered target statistics, ordered boosting by default on small data, and one_hot_max_size. TabArena: CatBoost ranks first
  in the conventional (default/tuned, un-ensembled) regime. NaN is treated as its own category by sklearn TargetEncoder and handled
  natively by GBDTs and TabPFN; neural models need an explicit indicator plus a fill (TabM: quantile normalisation, one-hot).
- Expected gain: raising min_data_per_group / cat_smooth on rare levels is a cheap anti-overfit move on high-cardinality
  categoricals (UNTESTED here); CatBoost as a default family is robust.
- Cost: C0.
- Compliance: nothing special; keep category vocabularies train-fit.
- Cross-check: converges with F3 "missing-value indicators"; new: the concrete parameter names.
- Sources: https://lightgbm.readthedocs.io/en/stable/Parameters.html [S]; https://xgboost.readthedocs.io/en/stable/parameter.html [S];
  https://catboost.ai/docs/en/concepts/speed-up-training [S]; https://arxiv.org/html/2506.16791 [S].

### RB07 · PUBLIC RESEARCH · Booster determinism: GPU CatBoost/LightGBM are documented non-deterministic; Optuna needs a deterministic objective
- Mechanism: CatBoost docs: "Training on GPU is non-deterministic, because the order of floating point summations is
  non-deterministic". LightGBM FAQ calls GPU run-to-run variation "normal and expected" (suggests gpu_use_dp or CPU); its
  `deterministic=true` promises stable results on the same data/params (and different num_threads) but may differ across
  versions/compilers and is documented for the CPU path; bagging was once thread-dependent. The XGBoost GPU documentation page
  I read contains no determinism statement; single-GPU `hist` aims at bit-exact reproduction but issues (#7295, #8820) report
  first-run differences. Optuna FAQ: a fixed sampler seed reproduces suggestions only when run sequentially AND the objective is
  itself deterministic ("if your objective function behaves in a non-deterministic way ... you cannot reproduce an optimization").
- Expected gain: none in score; prevents unstable HPO selection and unequal double-run outputs.
- Cost: choosing CPU boosters costs time (CatBoost CPU is the slowest of the three).
- Compliance: the platform's Deterministic Execution check as documented targets control flow (clock, env, workers, seeds,
  backend logic); bitwise float equality is not a listed pattern, so the pass/fail effect of GPU float noise is UNVERIFIED, but a
  `torch.cuda.is_available()`-style backend switch is exactly what is flagged. Plan: fixed device, fixed thread_count,
  CPU for CatBoost and for any booster inside an Optuna objective, and verify with `determinism_check.py` (double-run diff).
- Cross-check: extends engineering-and-compliance section 1 (which names LightGBM deterministic/force_row_wise and the Optuna seed
  but not the GPU-booster caveat).
- Sources: https://catboost.ai/docs/en/features/training-on-gpu [S]; https://lightgbm.readthedocs.io/en/latest/FAQ.html [S];
  https://lightgbm.readthedocs.io/en/stable/Parameters.html [S]; https://github.com/dmlc/xgboost/issues/8820 [S];
  https://github.com/dmlc/xgboost/issues/7295 [S-snip]; https://optuna.readthedocs.io/en/stable/faq.html [S].

### RB08 · COMPETITION PATTERN · Final fit: multi-seed, full data, fixed rounds = 1.25 x mean CV best iteration
- Mechanism: the S6E2 (Playground) 1st place retrained GBDT/RGF members on the full data after OOF generation, averaged 20 seeds, and set
  n_estimators to 1.25 x the average best iteration seen in CV [S-snip]. The Fertilizer winner averaged 100 XGBoost seeds, and
  NVIDIA's playbook advises refitting on all training data after HPO [S]. The factor above 1 compensates for the 20-25% more
  data.
- Expected gain: seed averaging on a noisy top-k metric (MAP@3) moved 5-fold XGBoost from 0.376 to about 0.379-0.380 over 100 seeds;
  refit gains are data-size dependent.
- Cost: seeds multiply training (C1 x seeds); choose the seed count from the profile.
- Compliance: the iteration count is computed in-script from CV (data-determined, not clock-determined); allowed (CLAUDE.md 3 rule 5).
- Cross-check: independently confirms V9 (refit at fixed count, mean best epoch x 1.1) and the CLAUDE.md 4A "mean best iteration
  x 1.1" rule; suggests 1.1-1.25 is the range winners use.
- Sources: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t [S-snip];
  https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ [S].

## 2. Stacking and blending that held up

### RB09 · COMPETITION PATTERN · How many models: winners use 70-150 members from 500-850 experiments; the transferable core is a few diverse families
- Mechanism: Podcast (RMSE, 2025) 1st: 75 level-1 models selected from 500 experiments, level-2 models (at least one GBDT and one
  NN) on OOF predictions, level-3 weighted average; CV 11.54, private 11.44. Level-1 diversity came partly from four
  formulations of the target (direct, ratio to episode length, residual from a linear relationship, and imputing the missing
  feature), plus level-2 features "confidence" (std across members) and "consensus" (mean). Churn (AUC, S6E3, 2026): four-level stack of 150 from
  850 with hill climbing. S6E2: about 150 OOFs, Optuna subset search, Ridge combiner [S-snip]. Contrast: 4th place S6E2 reached
  0.95534 private with depth-2 stumps (one-hot), periodic-embedding MLPs, RealMLP and rank averaging ("less is more") [S-snip]; and a 2025
  Open-Polymer winner reported AutoGluon beating an Optuna-tuned XGB+LGBM+TabM ensemble at a fraction of the budget [S].
- Expected gain: stack over best single typically hundredths of a metric unit (3rd-place Podcast stack 11.66 -> 11.62 CV [S-snip];
  Fertilizer 0.380 -> 0.386 MAP@3 with an NN level-2 trained with cross-entropy [S-snip]).
- Cost: scales with members; the winners relied on GPU farms and hundreds of experiments. A 45-minute A10G plan realistically
  holds 6-20 members (3 GBDT libraries x 1-2 feature views x a few seeds + 1-2 MLP/other).
- Compliance: OOF predictions are train-only; level-1 test predictions come from models trained on train. The Podcast winner merged
  train and test to impute a missing feature: banned here (RB36).
- Cross-check: converges with E-A (diversity, OOF-fitted low-dimensional blend); the "target formulations as diversity" idea
  extends A1 (reconstruct signal) and the "confidence/consensus" L2 features are new to the playbook.
- Sources: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ [S];
  https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/ [S];
  https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution [S-snip]; https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S].

### RB10 · COMPETITION PATTERN · A regularised ridge/logit combiner beat unconstrained hill climbing on private LBs; CV can rise while LB falls
- Mechanism: S5E12 (Diabetes) 1st: hill climbing stalled; adding models raised CV but lowered public LB; they then fit a Ridge
  (alpha 10, rank-transformed predictions, top 36 from hill climbing): CV 0.70860 vs 0.70886 for hill climbing but better
  public and private [S-snip]. S4E9 3rd place ("Gather, Ridge, Repeat"): ridge beat hill climbing, with and without negative
  weights, on CV and private [S-snip]. S6E4 1st: blend in logit space with LogisticRegressionCV on 5 folds [S-snip]. S6E2 1st:
  Optuna-chosen OOF subsets combined by Ridge [S-snip].
- Expected gain: not larger CV but more stable private; a gap of the size shown (CV -0.0003, private better) is within noise, so
  the point is variance, not bias.
- Cost: C0.
- Compliance: fit combiner on OOF only; any "rank transform" must use per-member train/OOF-reference percentiles, never ranks
  computed over the test batch (CLAUDE.md 2.3 #5; engineering-and-compliance class R; P-A14).
- Cross-check: independently confirms E-A "few free weights / non-negativity", V3 (nested check) and P-A12; the S5E12 pattern
  is also an instance of the 00-s-tier section-6 anti-pattern "select on the OOF you report".
- Sources: https://www.kaggle.com/c/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl [S-snip];
  https://www.kaggle.com/competitions/playground-series-s4e9/writeups/optimistix-3rd-place-solution-an-open-secret-gathe [S-snip];
  https://www.kaggle.com/competitions/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach [S-snip].

### RB11 · PUBLIC RESEARCH · Ensemble-weight search overfits small validation sets; the known fixes
- Mechanism: Caruana et al. 2004: forward selection overfits when ensembles are small and when the library is large; remedies are
  sorted initialisation (start from the best N) and bagged selection (select from random library subsets). Purucker & Beel
  (71 AutoML-benchmark datasets): CMA-ES weight optimisation "overfits drastically" for ROC AUC (significantly for multiclass),
  succeeds for balanced accuracy, and is repaired by normalising the weights the way greedy selection implicitly does.
  TabArena: highest-ranked models were not the highest-weighted in cross-model ensembles; some DL models (ModernNCA, RealMLP)
  looked overfit to validation data.
- Expected gain: avoids a CV-optimistic blend; no absolute number.
- Cost: C0.
- Compliance: fit on OOF only; weights non-negative and normalised; cap free weights to roughly the number of model families.
- Cross-check: converges with E-A, V3, P-A12; adds the metric-dependence (AUC-driven weight search is the dangerous case).
- Sources: https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf [S-snip]; https://arxiv.org/abs/2307.00286 [S];
  https://arxiv.org/html/2506.16791 [S].

### RB12 · COMPETITION PATTERN · Many seeds for threshold- or top-k-sensitive metrics; then a probability-calibrated level-2
- Mechanism: MAP@3 on the Fertilizer task was "very sensitive to the randomness of training and predicted probabilities": averaging
  100 five-fold XGBoost runs lifted a per-run 0.376 to about 0.380 (NVIDIA blog quotes 0.379), and a level-2 NN trained with
  categorical cross-entropy reached CV 0.386 [S-snip / S]. Mechanism: top-k decode is only as good as probability ranking;
  averaging reduces rank noise near ties; cross-entropy trains calibrated probabilities.
- Expected gain: +0.003-0.004 (seeds), +0.006 more from stacking, on that metric.
- Cost: linear in seeds (the write-up credits fast GPU experimentation for making 100 x 5 fits practical); on the A10G budget choose
  5-20 seeds from the profile.
- Compliance: seeds are fixed constants; averaging is within-train; no test statistics.
- Cross-check: complements 00-s-tier move 6 / E-A ("diversity beats seeds"): both help, seeds mainly for noisy decodes (top-k,
  thresholds); consistent with research-A 4.4 and P-A17.
- Sources: https://www.kaggle.com/competitions/playground-series-s5e6/discussion/587393 [S-snip];
  https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ [S].

### RB13 · PUBLIC RESEARCH · Reimplement AutoGluon's bagging + multi-layer stacking recipe in-script (do not import it)
- Mechanism: AutoGluon-Tabular stacks models in layers trained layer-wise, with skip connections from raw features and
  repeated k-fold bagging over OOF predictions to curb overfitting; its stated principle is that multi-layer combination of many
  models uses a time budget better than searching for the best single model [S-snip]. In 2025 competitions AutoGluon appeared in two
  winning solutions [S].
- Expected gain: strongest off-the-shelf baseline in TabArena-style comparisons (AutoGluon 1.4 extreme, a 4-hour tuned ensemble,
  is the reference TabPFN-2.5 is compared against [S]); no other number extracted.
- Cost: C2-C3 for a full stack; a two-layer variant with 5-fold bagging fits.
- Compliance: `autogluon` is not on the library list in CLAUDE.md section 8 (ask a reviewer first); the recipe itself
  (bagged OOF layers, raw-feature skip connection, constrained level-2) is plain training and reimplementable with
  sklearn/lightgbm/xgboost/catboost.
- Cross-check: new as an explicit "skip-connection" level-2 (raw features + OOF predictions); otherwise converges with RB09/RB10.
- Sources: https://arxiv.org/abs/2003.06505 [S-snip]; https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S];
  https://arxiv.org/abs/2511.08667 [S].

## 3. Neural tabular and foundation models: what is competitive, size limits, compliance

### RB14 · PUBLIC RESEARCH · TabM: a BatchEnsemble-style MLP ensemble in one network (implement from scratch)
- Mechanism: k=32 implicit MLPs share weights with per-member rank-1 adapters, are trained jointly on shared batches with the
  mean of member losses (not the loss of the mean), and the k outputs are averaged at inference (probabilities, not logits). Early
  stopping watches the ensemble. Numerical features get piecewise-linear or periodic embeddings; numerics are quantile-normalised;
  categoricals one-hot. TabM-mini keeps only the first adapter with similar accuracy. Defaults from the official repo: AdamW lr 0.002,
  weight decay 3e-4, n_blocks 1-5 (fewer with embeddings; avoid n_blocks=1 on small data), d_block 64-1024; paper: patience 16,
  batch 128-1024, Optuna TPE 50-100 trials (25-50 acceptable), 46 datasets with 1.8K-723K training rows; TabM is reported more
  reliable than attention-based models, particularly on domain-aware (shifted) splits. Individual members are weak and overfit; the
  average is strong.
- Expected gain: top-ranked model in TabArena after tuning + ensembling; competitive with GBDT and a different inductive bias.
  A transplant-survival winner and the MCTS-strength 1st place used TabM as an ensemble member.
- Cost: C1-C2 on an A10G for about 100K rows (estimate); TabArena notes DL inference is 15-100x slower than CatBoost/EBM.
- Compliance: the `tabm` and `rtdl_num_embeddings` packages are not on the library list: write original torch code from the paper
  (about 150 lines: adapter linear layers, embeddings, joint loss), fixed seeds and epochs; pure training, no pretrained weights.
- Cross-check: new to the playbook; supplies the "regularised MLP second family" named in tabular-and-scientific step 2.
- Sources: https://arxiv.org/abs/2410.24210 and https://arxiv.org/html/2410.24210 [S]; https://github.com/yandex-research/tabm [S];
  https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S]; https://github.com/jday96314/MCTS [S] (method only).

### RB15 · PUBLIC RESEARCH · RealMLP-TD recipe (strong pre-tuned MLP without per-task HPO)
- Mechanism: robust scaling + smooth clipping of numerics, PBLD numeric embeddings (dim 4), categorical embeddings (dim 8), one-hot
  up to 8 categories, hidden [256,256,256], AdamW (beta 0.9/0.95), batch 256, 256 epochs, classification lr 0.04 / regression
  lr 0.2, dropout 0.15, weight decay 0.02, SELU + cross-entropy with label smoothing 0.1 (classification), Mish + MSE (regression),
  scheduling and parametric activations; tuned only on the meta-train benchmark.
- Expected gain: Table B.5 (meta-test): RealMLP-TD cuts error 15.2% (classification) and 14.9% (regression) relative to a plain
  tuned-default MLP; ensembles with GBDT defaults are the strongest cheap combination (RB02).
- Cost: C1-C2 (256 epochs; roughly 5 s per 1K rows on CPU in their timing plot, visual read; far less on a GPU).
- Compliance: `pytabkit` not allowed; reimplement. Label smoothing helps classification but see tabular-and-scientific "Small N
  discipline" (no smoothing that stalls at ln K); the S6E2 4th place also distilled RealMLP students [S-snip].
- Cross-check: new; shares the "regularise by where you stop and shrink" philosophy of E-C.
- Sources: https://arxiv.org/html/2407.04491 (Table A.1) [S]; https://arxiv.org/pdf/2407.04491 (Table B.5, page 28) [S].

### RB16 · PUBLIC RESEARCH · What the benchmarks say about model choice, tuning protocol and what to skip
- Mechanism: TabArena (51 datasets, 500-250K rows, 200 random HPO configs per model, 8-fold CV, repeated CV): after tuning +
  ensembling the best DL models (TabM, RealMLP) equal or beat GBDTs; CatBoost leads in the conventional un-ensembled regime;
  ensembling changes the ranking sharply (paper: "top three models would all be worse than the actual fourth-best model without
  post-hoc ensembling"); deep models need ensembles of 25+ configurations to dominate GBDTs; holdout validation instead of CV
  "greatly underestimates" all models; foundation models dominate datasets of at most 10K rows and 500 features. TabM paper:
  MLP-like models beat attention- and retrieval-based architectures on efficiency and robustness. TALENT (300+ datasets):
  ensembling helps trees and nets alike; the best family depends on the categorical/numerical mix. Retrieval models (ModernNCA,
  TabR) and FT-Transformer are therefore low priority for a 45-minute plan.
- Expected gain: guidance, not a number.
- Cost: n/a.
- Compliance: n/a.
- Cross-check: converges with E-A and tabular-and-scientific step 2; confirms "complexity does not track rank".
- Sources: https://arxiv.org/html/2506.16791 [S]; https://arxiv.org/abs/2506.16791 [S]; https://arxiv.org/abs/2407.00956 [S].

### RB17 · PUBLIC RESEARCH · Tabular foundation models are the strongest small-N predictors but are inference-only: do not ship
- Mechanism: TabPFN v2 (Nature 2025, weights on Hugging Face, Prior Labs licence = Apache 2.0 with attribution, `pip install
  tabpfn`) handles about 10K rows, 500 features, 10 classes; TabPFN-2.5 (arXiv 2511.08667) 50K x 2K and a distillation engine
  into MLP/trees; TabICLv2 (ICML 2026) reports surpassing RealTabPFN-2.5 with open training code and million-row scale;
  TabPFN-3 report (arXiv 2605.13986) claims 1M rows and 200 features on one H100; Mitra, TabFlex, TabPFN-Wide exist; a secondary
  source says RealTabPFN-2.5 needs a commercial licence. In 2025 competitions TabPFN first appeared as a feature generator feeding
  GBDTs (PREPARE winners) and as a level-1 member of a 72-75-model stack; a polymer challenge winner fine-tuned
  CodeBERT/ModernBERT with external data.
- Expected gain: biggest on N <= 10K (TabArena); on larger N GBDT/DL ensembles are competitive.
- Cost: C1 inference but needs the package and a GPU for speed.
- Compliance (decisive): fit() does not train, so the model "does not do the learning" (CLAUDE.md 2.1 inference-only); the
  package is not on the allowed-library list (section 8: ask a reviewer first); weight licence terms beyond v2 were only reported
  second-hand. Treat as NOT shippable; see RB18 for the legitimate use.
- Cross-check: matches platform-facts section 5 (Tabular regime accepts GBDT/NN/ensembles; frozen-embedding + tabular is grey).
- Sources: https://huggingface.co/Prior-Labs/TabPFN-v2-clf [S]; https://arxiv.org/abs/2511.08667 [S]; https://arxiv.org/abs/2602.11139 [S];
  https://arxiv.org/abs/2605.13986 [S]; https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models [S];
  https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S].

### RB18 · UNTESTED HYPOTHESIS · Offline TFM yardstick as an information-ceiling probe for small-N tasks (never shipped)
- Mechanism: on a small-N task run a foundation model locally on the same grouped folds once. If it beats the best in-rules model
  by far, the in-rules model underfits or lacks structure (add shrinkage, features, a different family); if it ties, the data
  ceiling is near. This is a diagnostic of capacity vs information, extending the "diagnose the information ceiling" step.
- Expected gain: saves effort on ceiling-limited tasks; no direct score.
- Cost: C1 locally; zero in solution.py.
- Compliance: local diagnostics only; fit on train folds, score on held-out train folds, no test row touched; do not import
  the package in solution.py and do not ship its predictions. Package availability in the local sandbox is unverified.
- Cross-check: extends 00-s-tier section 4 and V11 (compare against yardsticks).
- Sources: https://arxiv.org/html/2506.16791 [S] (TFM dominance at most 10K rows); own derivation.

### RB19 · COMPETITION PATTERN · Capacity ladder on noisy/synthetic tabular data: stumps and shallow models can win
- Mechanism: S6E2 4th place: gradient-boosted depth-2 trees on one-hot features (local CV about 0.95574), periodic-embedding MLPs
  (dim 8), 8-member simultaneous RealMLP, rank averaging; their stated reason: deeper trees fit the synthetic generator's noise
  [S-snip]. Same direction: small-N discipline in tabular-and-scientific.md.
- Expected gain: top-5 Playground result with minimal machinery; dataset-specific (synthetic data).
- Cost: C0-C1.
- Compliance: fine; a model, not a rule. Rank averaging must be train-reference (RB10).
- Cross-check: independently confirms tabular-and-scientific "Small N discipline" and the 00-s-tier "simplest solution won in
  six problems" evidence.
- Sources: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution [S-snip].

## 4. In-script HPO and why a small/noisy CV misleads it

### RB20 · PUBLIC RESEARCH · The HPO-overfitting evidence set and a bounded in-script protocol
- Mechanism: (a) Bates-Hastie-Tibshirani (JASA 2023): no unbiased estimator of the CV standard error from one CV run; naive fold SE is
  too small because fold errors are correlated [S-snip, abstract]. (b) Reshuffling the resampling splits for every configuration
  "drastically improves" single-holdout HPO, makes holdout competitive with CV at lower cost, with the largest benefit on small data
  [S]. (c) A solubility-prediction study found HPO "did not always result in better models, possibly due to overfitting", preset
  hyperparameters matching optimised ones at about 10,000x less compute [S]. (d) Holzmueller B.9: TPE vs random search differ
  little at 50 steps (RB01). (e) TabArena: holdout-based selection underestimates every model; use repeated CV.
  (f) TabM: Optuna TPE 50-100 trials, 25-50 if top accuracy is not critical.
  Bounded protocol: start from RB01 defaults; if the plan includes Optuna use a fixed 20-40 trials, TPESampler(seed), n_jobs=1,
  fold seeds re-drawn per trial (reshuffling) on the repeated CV, a small low-dimensional space (lr, depth/leaves,
  min_data, subsample/colsample, 1-2 regularisers), and accept a tuned config only if it beats the defaults on fresh
  splits by more than the corrected paired threshold (P-A07/P-A08).
- Expected gain: HPO helps medium/large data (RB01 numbers) and often harms small noisy sets.
- Cost: trials x one CV; the dominant cost of a plan, so budget it explicitly.
- Compliance: HPO must be in-script with fixed trial count (CLAUDE.md 2.1, 3 rule 4); objective must be deterministic (RB07).
- Cross-check: converges with V3/V4, P-A07 (corrected t), P-A08 (winner's curse); adds reshuffling and the TD baseline.
- Sources: https://arxiv.org/abs/2104.00673 [S-snip]; https://arxiv.org/abs/2405.15393 [S]; https://arxiv.org/abs/2407.20786 [S];
  https://arxiv.org/pdf/2407.04491 [S]; https://arxiv.org/html/2506.16791 [S]; https://github.com/yandex-research/tabm [S].

### RB21 · UNTESTED HYPOTHESIS · Average the top-K Optuna configs instead of shipping the argmax
- Mechanism: TabArena finds post-hoc ensembling of different HPO configurations is where most of the tuned gain comes from
  (ensembles of 25+ configurations). With a noisy CV the single best trial is partly luck (winner's curse); averaging the K best
  trials' OOF/test predictions converts selection noise into variance reduction at no extra search cost. Test K in {3, 5, 8}
  against argmax and against RB01 defaults on fresh split seeds before adopting.
- Expected gain: unknown; plausible of the order of the seed-averaging gains (RB12).
- Cost: C0 extra if trial predictions are kept (store predictions, not models).
- Compliance: in-script, train only, fixed K.
- Cross-check: new combination of E-A "average, do not select" with the in-script HPO rules.
- Sources: https://arxiv.org/html/2506.16791 [S]; own derivation.

## 5. Metric-aware tricks, calibration, post-processing that survived

### RB22 · GENERAL ML PRINCIPLE · F1/threshold metrics: one calibrated-probability threshold, derived not searched
- Mechanism: Lipton, Elkan, Narayanaswamy: for a classifier with well-calibrated probabilities the F1-optimal threshold equals
  half of the optimal F1 value. Fit it as a fixed point on OOF (one parameter) rather than a free grid; when probabilities are
  poorly calibrated first apply one cross-fitted temperature or Platt map (isotonic needs about 1000+ samples or overfits, per
  scikit-learn; temperature scaling has a single parameter for multiclass).
- Expected gain: removes a noisy multi-parameter search; the gain over a 0.5 threshold can be large on imbalanced data.
- Cost: C0.
- Compliance: constants are found in-script on OOF (B9 style); no test statistics; if the description bans decode-time
  tuning, move to logit-adjusted loss (metric-and-decoding section 9).
- Cross-check: converges with metric-and-decoding section 4 "closed-form per-label threshold" and section 6.
- Sources: https://arxiv.org/abs/1402.1892 [S]; https://scikit-learn.org/stable/modules/calibration.html [S].

### RB23 · COMPETITION PATTERN · Ordinal/QWK and balanced-accuracy tasks: regress or decompose first, then few OOF thresholds, then distrust the public board
- Mechanism: CMI-PIU (QWK, 2024) 1st place: ignored the provided 4-class `sii` label, regressed the underlying PCIAT-Total score with a
  voting ensemble (LightGBM, two XGBoost, CatBoost, ExtraTrees) and converted to classes with thresholds; mostly ignored
  leaderboard feedback [S-snip]. AES 2.0 and CMI saw QWK-threshold tuning on the public board cause large shake-ups (P-A04/P-A17).
  S6E4 (balanced accuracy, 3 ordered classes) 1st: two binary one-vs-rest models arranged as P(Low)=P1, P(Medium)=(1-P1)(1-P2),
  P(High)=(1-P1)P2, logit-space blend with LogisticRegressionCV, two greedy threshold searches on OOF balanced accuracy
  [S-snip].
- Expected gain: S6E4 ensemble CV 0.98155 vs 0.9805 single models; QWK: the larger lever is the regression target choice.
- Cost: C0 after models.
- Compliance: thresholds are OOF-fitted in-script; keep them few (K-1) and check cross-fitted gain (metric-and-decoding section 6); not
  hardcoded after seeing the public score (L005, Q3).
- Cross-check: independently confirms A1 (reconstruct the continuous latent behind a bounded label), A8 (ordinal cumulative
  heads), P-A17; the winner's choice of PCIAT-Total is the clearest recent instance of A1.
- Sources: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/552638 [S-snip];
  https://www.kaggle.com/competitions/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach [S-snip].

### RB24 · GENERAL ML PRINCIPLE · RMSLE = RMSE on log1p; match the target transform to the metric, mind the back-transform for raw-scale metrics
- Mechanism: RMSLE(y, p) equals RMSE(log1p y, log1p p), so train MSE on log1p(y) and predict expm1 with no bias correction. If the
  metric is RMSE on the raw scale but the target is skewed, a log-target model predicts a conditional median-like value:
  evaluate on the raw scale on OOF and compare with a raw-target and a Tweedie/gamma objective; a smearing factor is a one-parameter
  OOF fix. The Calories (RMSLE) winner used GPU hill-climbing over XGBoost/CatBoost/NN/linear members [S].
- Expected gain: large when the metric is log-scale and training is raw-scale; small otherwise.
- Cost: C0.
- Compliance: fine.
- Cross-check: converges with CLAUDE.md 6.1 (log1p for RMSLE) and metric-and-decoding section 2.
- Sources: identity and smearing are [R]; https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ [S];
  https://x.com/NVIDIAAIDev/status/1929614499817893966 [S-snip].

### RB25 · UNTESTED HYPOTHESIS · AUC: rank-pairwise objective vs plain logloss for the boosters
- Mechanism: for AUC metrics one can train XGBoost `rank:pairwise` or LightGBM `lambdarank`/`rank_xendcg` instead of `binary` logloss.
  Evidence found is weak (educational pages claiming alignment; the 2026 AUC-metric churn winner stacked models with no reported
  pairwise gain). Test locally against logloss on repeated CV, per fold.
- Expected gain: unknown, likely small versus stacking and seeds; matters mostly on imbalanced tasks.
- Cost: C1.
- Compliance: fine; a trained model either way.
- Cross-check: extends the metric-and-decoding table row "NDCG/MAP/MRR" and A5 to AUC; new, unconfirmed.
- Sources: https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-9-gradient-boosting-specialized-tasks/ranking-objective-functions [S-snip, weak];
  https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/ [S].

### RB26 · GENERAL ML PRINCIPLE · Calibration after ensembling: averaging already helps; fit at most one cross-fitted scalar
- Mechanism: for log-loss/Brier metrics, ensemble averaging itself usually lowers calibration error; in a deep-ensemble study,
  post-hoc temperature scaling on top gave inconsistent extra gains [S-snip]; temperature scaling is the better method in the
  data-limited regime and isotonic in the data-rich regime (sklearn: isotonic overfits below about 1000 samples). sklearn
  `CalibratedClassifierCV(ensemble=True)` averages k (classifier, calibrator) pairs and warns that every class must be present in
  every split [S].
- Expected gain: small, positive only when the ensemble is miscalibrated (check reliability on OOF first).
- Cost: C0.
- Compliance: one scalar fit on OOF, applied per row; no test-set calibration (CLAUDE.md 2.3 #5).
- Cross-check: converges with metric-and-decoding section 2 (Brier row) and P-A15.
- Sources: https://scikit-learn.org/stable/modules/calibration.html [S]; https://openreview.net/pdf?id=wTWLfuDkvKp [S-snip];
  https://arxiv.org/pdf/2511.04160 [S-snip].

## 6. Time series and forecasting

### RB27 · PUBLIC RESEARCH · What Kaggle forecasting winners did (Bojer & Meldgaard, six competitions)
- Mechanism (PDF pages 12-19 read): global models trained across all series; GBDT with rolling statistics (means/medians) and
  event/promo counters at several hierarchy levels; ensembles over direct/iterated strategies and feature subsets; global ensembles
  beat local single models; neural nets win when data are huge and exogenous variables weak (Wikipedia, 145K series: RNN ensemble
  of 3 seeds with checkpoint averaging and SWA, SMAC3 HPO "relatively insensitive"); the Favorita winner trained only on the most
  recent 1-5 months chosen on validation; horizon-specific models (one model per horizon: 16 in Favorita, 42 for a
  Recruit 5th place) gave improvements that "might not be substantial compared to the growth in the number of models"; the main GBDT
  weakness is trend extrapolation, handled by ensembling with a linear trend model; promotions, holidays and events were the useful
  exogenous inputs, while covariates that must themselves be forecast (weather, macro) were not. Validation: a hold-out
  whose length equals the forecast horizon, taken from the end of the data, used by the top three in Favorita; exceptions placed
  well too (a 4th place used grouped K-fold plus time-series CV; the 7th and 8th in Recruit used plain K-fold).
- Expected gain: top solutions beat seasonal-naive benchmarks by 20-74% across those competitions (author figures).
- Cost: C1-C2 for one global GBDT with rolling features; a sequence NN adds C2.
- Compliance: features strictly backward from the forecast origin; no test statistics.
- Cross-check: converges with tabular-and-scientific "Forecasting / temporal" and F5 (strictly backward lags); new: recent-window
  training, trend-blend, horizon-specific models are low value per cost.
- Sources: https://arxiv.org/abs/2009.07701 and https://arxiv.org/pdf/2009.07701 [S]; M5 conclusions (recursive LightGBM models more accurate
  but less stable than non-recursive; winners mixed both) [S-snip] https://www.sciencedirect.com/science/article/pii/S0169207021001874 (page blocked, snippet only).

### RB28 · GENERAL ML PRINCIPLE · Rolling-origin validation with a gap/embargo and leakage-safe lags
- Mechanism: scikit-learn `TimeSeriesSplit(n_splits, test_size, gap, max_train_size)` returns superset training windows and a
  `gap` of excluded samples before each test block "to prevent leakage from recent training data to test data". Match `test_size`
  to the forecast horizon and `gap` to the deployment lag or label overlap (embargo/purging: López de Prado [R]). Lags and rolling
  windows must end strictly before the origin (shift before roll); target-derived features recomputed inside each fold. An Optiver
  entrant purged by `date_id` and tested that changing data after time t never changes a feature at t [S-snip].
- Expected gain: protects from time-leak optimism, the largest forecasting private-LB gap; no score gain.
- Cost: C0 (but more folds than random CV).
- Compliance: lag features of the test period may use only information the description makes available at forecast time; fit
  nothing on test.
- Cross-check: converges with V1/V5/V7 and B7 (deployment-gap magnitude); the unit test (feature at t invariant to later data)
  is a cheap new gate analogous to the half-rows test.
- Sources: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html [S];
  https://github.com/vivek-varma/Optiver-Trading-at-the-close-Sub [S-snip].

### RB29 · COMPETITION PATTERN · 2023-2025 forecasting winners: GBDT + GRU/Transformer blends; two tricks are NOT allowed here
- Mechanism: Optiver "Trading at the Close" 1st: CatBoost 0.5 + GRU 0.3 + Transformer 0.2 on 300 shared features chosen by
  CatBoost importance, CV 5.8117 / private 5.4030, with online learning during inference and a post-processing step [S-snip];
  another top entry blended LightGBM/GRU/Transformer (5.4777 with 0.3/0.1/0.6), the 6th used little feature engineering plus
  rolling training and zero-sum post-processing. Enefit (energy): a 1st-place description combined XGBoost, GRU and XGBoost+GRU on
  about 600 features [S-snip, page unclear]. Jane Street 2025 drew 3,700+ teams; its winning approach was not retrievable [S].
- Expected gain: blend weights of that kind are competition-specific; the transferable point is GBDT + a causal sequence NN for diversity.
- Cost: C2-C3 for the sequence NN on an A10G.
- Compliance: DO NOT ADOPT online learning on test rows (test-time adaptation, CLAUDE.md 2.3 #5) or zero-mean/zero-sum
  cross-sectional post-processing over test rows (whole-test statistic) unless the description makes each timestep batch a legitimate
  single inference unit; check the per-row independence text first.
- Cross-check: extends tabular-and-scientific "GBDT + a sequence model for diversity"; compliance caveat new.
- Sources: https://zhuanlan.zhihu.com/p/689864294 [S-snip; fetch 403]; https://blog.csdn.net/qq_51175703/article/details/137109118 [S-snip];
  https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S].

### RB30 · PUBLIC RESEARCH · Time-series foundation models lead zero-shot benchmarks, but zero-shot is inference-only
- Mechanism: Chronos-2 (Oct 2025) handles univariate, multivariate and covariate-informed forecasting in-context and leads fev-bench
  (100 tasks, 46 with covariates), GIFT-Eval and Chronos Benchmark II; TimesFM-2.5 and TiRex also dominate classical and
  task-specific deep models; a snippet reports covariates raising Chronos-2's score from 40.9% to 47.0% on the 42 dynamic-covariate
  tasks (metric as reported there) and also helping TabPFN-TS (TabPFN on time-feature regression) [S-snip]; LightGBM scored below
  these models in the extended results [S-snip].
- Expected gain: a strong yardstick, especially at short history.
- Cost: C1 inference.
- Compliance: zero-shot inference does not satisfy "model must do the learning"; fine-tuning from Hugging Face weights is allowed but
  must be genuine; `chronos-forecasting` / `tabpfn-time-series` are not on the library list.
- Cross-check: new; same compliance logic as RB17 (use as local yardstick, RB18).
- Sources: https://arxiv.org/abs/2510.15821 [S]; https://arxiv.org/abs/2509.26468 [S]; https://arxiv.org/html/2509.26468v4 [S-snip].

## 7. Private-LB killers in regression/classification (what actually happened)

### RB31 · COMPETITION PATTERN · CMI-PIU: missingness shift between train and test, imputer choice, and public-board chasing
- Mechanism: public-vs-private rankings in CMI-PIU reshuffled massively; reported facts: top-10 public teams averaged 212
  submissions (median 199) vs 64 (median 25) for top-10 private teams; the proportion of missing values differed sharply (reported
  about 40-50% in test vs about 3% in train for some features), and a widely shared KNNImputer notebook was believed to have
  dropped everyone who submitted it; the winner mostly ignored the public score [S-snip].
- Expected gain: avoidance.
- Cost: C0.
- Compliance: Eris forbids looking at test distributions inside the pipeline; the usable part is "rehearse the shift on train":
  read the description for stated missingness or population shifts, add missing indicators, and train with random feature
  masking at higher rates than observed (UNTESTED as a method here); imputers are fit on train only.
- Cross-check: converges with V7 (rehearse the shift), V12 (leakage tells), P-A01/P-A04 (public board is noisy).
- Sources: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/552638 [S-snip];
  https://dongsunseng.com/entry/Child-Mind-Institute-%E2%80%94-Problematic-Internet-Use-The-Greatest-Shake-Up [S-snip; host unreachable on direct fetch].

### RB32 · COMPETITION PATTERN · Group, time and stability structure decide the CV (MCTS, BELKA, Home Credit)
- Mechanism: MCTS-strength 1st place used a shuffled GroupKFold keyed on the game ruleset because the same game appears in many
  rows [S], with agent-swap augmentation (swap Agent1/Agent2 and the dependent features) and feature views from tree-search
  statistics (one supplement reported at about +0.045 CV and +0.012 LB [S-snip]). BELKA: the test contains building blocks absent
  from train [S-snip]. Home Credit stability: weekly Gini with a penalty on downward slope and residual spread (metric form [R]),
  rewarding features stable over time; a third-party reproduction found that dropping 13 skew features raised weakest-fold
  stability but won only three of five folds [S-snip].
- Expected gain: avoidance of a 0.02-0.05-sized CV optimism.
- Cost: C0-C1.
- Compliance: groups derived from train/description; augmentations that swap symmetric roles are train-side only.
- Cross-check: independently confirms V1/V2 (group by the largest hidden unit) and V5; the symmetric-role augmentation echoes F1
  "canonicalise interchangeable parts".
- Sources: https://github.com/jday96314/MCTS [S] (method only; its external-data and LLM-generated-game parts are banned here);
  https://nootlabs.substack.com/p/kaggle-competition-review-leash-belka [S]; https://github.com/alvaromendizabal/home-credit-model-stability [S-snip].

### RB33 · GENERAL ML PRINCIPLE · Early-stopped fold scores are optimistic; pick rounds, then re-score cleanly
- Mechanism: when the same validation fold both selects the best iteration and reports the OOF score, the OOF is biased upward
  by the selection (small for 1000-tree GBDTs, larger for noisy small folds). Mitigations: take the median best iteration
  across folds and re-evaluate with that fixed count; or early-stop on an inner split of the training fold; never report the
  early-stopped OOF as the final selection number when comparing close candidates.
- Expected gain: protects selection among close candidates.
- Cost: C1 (one extra pass for the fixed-count re-score).
- Compliance: data-determined plan, allowed (CLAUDE.md 3 rule 5).
- Cross-check: sharpens V3/V9; recalled principle, no dedicated source found [R].
- Sources: [R].

### RB34 · PUBLIC RESEARCH · Benchmark-repository misuse shows the same pitfalls: model selection, baselines, preprocessing
- Mechanism: Tschalzev et al. (2025) list three recurring flaws when tabular datasets are used without reflection: model-selection
  strategies that ignore data structure (time, groups, duplicates), overlooked strong baselines, and inappropriate preprocessing.
- Expected gain: audit checklist only.
- Cost: C0.
- Compliance: n/a.
- Cross-check: converges with V1/V2/V8 and the leakage battery of P-A11.
- Sources: https://arxiv.org/abs/2503.09159 [S].

## 8. Scientific-data notes and the DO NOT ADOPT list

### RB35 · COMPETITION PATTERN · Molecular/polymer property recipes: descriptors first, curated multitask training, TFM-on-embeddings is grey
- Mechanism: Polaris/ASAP ADMET: 2nd place used multitask Chemprop D-MPNN over 55+ public tasks incl. calculated properties
  (second only to a team with proprietary data); 3rd place took a descriptor-first approach (mechanistic, physicochemical,
  fragment and metabolic feature groups); TabPFN on frozen CheMeleon embeddings won 50 of 58 tasks (86.2%) on a benchmark suite
  [S-snip]; a related paper reports up to 100% win rates on 30 MoleculeACE tasks [S]. The Open Polymer Prediction winner fine-tuned
  CodeBERT/ModernBERT and used external data [S].
- Expected gain: descriptors and multi-task auxiliary targets are the portable levers.
- Cost: C1-C2.
- Compliance: external datasets and proprietary data are banned; TFM-on-frozen-embeddings is grey/likely rejected
  (CLAUDE.md 2.5, 6.7); only multitask/auxiliary targets built from the challenge's own columns, and RDKit availability
  must be confirmed before any descriptor plan.
- Cross-check: converges with tabular-and-scientific "Domain features" and 6.7 (Fine-tuning labelled challenges).
- Sources: https://chemrxiv.org/doi/10.26434/chemrxiv-2025-q12vh [S-snip]; https://pubmed.ncbi.nlm.nih.gov/41381398 [S-snip];
  https://arxiv.org/abs/2604.16123 [S]; https://mlcontests.com/state-of-machine-learning-competitions-2025/ [S].

### RB36 · VERIFIED PLATFORM REQUIREMENT · Winning-Playground-style tricks that Eris bans (do not copy their code or recipe)
- Mechanism and ban (each checked against CLAUDE.md 2.3, section 7 and engineering-and-compliance 3, all repo text):
  (1) merging the "original" / external dataset, proprietary data, or LLM-generated synthetic games for training: external data /
  synthetic data ban; (2) imputing or count/target-encoding with statistics from train+test (the Podcast winner merged train
  and test): test-set fitting ban; (3) pseudo-labelling test rows ("k sets of pseudo-labels" in the NVIDIA playbook): banned;
  pseudo-labelling unlabelled train rows only with independent-signal agreement (E-C); (4) rank averaging across test rows and zero-sum
  cross-sectional post-processing: whole-test aggregation (use train-reference percentiles, P-A14); (5) online learning at
  inference: test-time adaptation; (6) frozen TabPFN/TabICL/Chronos zero-shot predictions: inference-only; (7) pip-installed
  `tabm`, `pytabkit`, `tabpfn`, `autogluon`, `rtdl_num_embeddings`: not on the allowed list, reimplement or ask.
- Expected gain: avoids rejection; no score.
- Cost: n/a.
- Compliance: this is the compliance entry.
- Cross-check: converges with engineering-and-compliance class R/Q2 and CLAUDE.md 2.3; extends the list with tabular-specific cases.
- Sources: CLAUDE.md and platform-facts (repo); https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ [S];
  https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ [S].

## 9. Top 10 highest-leverage items for Eris tabular / scientific / forecasting (ranked by expected private gain x compliance safety / cost)

1. RB01+RB02+RB08: three GBDT libraries at literature meta-tuned defaults (cite the paper in a comment), early-stopped on folds, final
   refit at 1.1-1.25 x mean best iteration, a few seeds.
2. RB03+RB04+RB05: capped feature factory (arithmetic, group-conditional, pseudo-categorical) with nested cross-fitted target/group
   statistics built from the outer-training fold only.
3. RB10+RB11: OOF blend with a low-dimensional regularised combiner (non-negative ridge / LR on logits, or hill climbing with a
   few steps), never more free weights than model families; train-reference percentiles instead of test ranks.
4. RB14+RB15+RB16: one regularised MLP family implemented from scratch (TabM-style k-member BatchEnsemble MLP, RealMLP-TD settings) as the
   structurally different second member.
5. RB20+RB21: HPO discipline: defaults first, <= 20-40 seeded TPE trials with per-trial reshuffled repeated CV, accept only beyond the
   corrected paired threshold; test top-K trial averaging versus argmax (hypothesis).
6. RB22+RB23+RB24: metric-aware target and decode with few OOF-fitted constants: continuous latent instead of a bounded class
   label (CMI), one-vs-rest for ordered classes (S6E4), F1 threshold = F*/2 for calibrated scores, log1p for RMSLE.
7. RB27+RB28+RB29: forecasting recipe: horizon-length rolling-origin holdout with gap, strictly backward lag/rolling features, global
   GBDT + causal sequence NN for diversity, recent-window and trend-blend checks; no test-time online learning or cross-row
   post-processing.
8. RB07: booster determinism: CPU for CatBoost and for any booster inside an Optuna objective, fixed thread counts, double-run diff;
   GPU CatBoost/LightGBM documented non-deterministic.
9. RB31+RB32: shift robustness: group/time folds that mirror the hidden unit, missingness rehearsal on train, no public-board
   threshold chasing (CMI shake-up).
10. RB17+RB18+RB36: do not ship TabPFN/TabICL/Chronos zero-shot or any pip-only package; use a foundation model offline as a ceiling
    yardstick for small-N tasks and reimplement the recipes in plain torch/sklearn.
