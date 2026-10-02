# Research A: making the PRIVATE score match CV (hidden-leaderboard generalisation)

Date: 2026-10-02. Role: pattern researcher (public sources only; no Shipd/Eris solver material read; nothing here is a decision).

## 0. Reading guide and limits

- Source marks: [S] = confirmed by a search or fetch this session. [S-full] = I read the primary text (Roelofs, Varoquaux, Wainer-Cawley, SIIM winner paper, sklearn docs, Park post-mortem, Ladder, several abstracts). [S-snip] = only a search-result snippet seen (Kaggle write-up pages and some Medium pages are not fetchable: JS-rendered or 403). [R] = recalled, not re-verified this session.
- Tags (exactly one per item): VERIFIED PLATFORM REQUIREMENT, GENERAL ML PRINCIPLE, PUBLIC RESEARCH, COMPETITION PATTERN, LOCAL EXPERIMENT RESULT, UNTESTED HYPOTHESIS. No local experiments were run: this session had no code-execution tool, so every "Check" below (and the appendix code) is an UNTESTED recipe sized for a 4-core CPU, not a measured result.
- Cross-checks name existing playbook entries (validation-recipes V1-V15, ensembling-and-training E-A, engineering-and-compliance R/S/Q, platform-facts, CLAUDE.md sections). Where an item converges with an entry it says so; only genuinely new material is proposed.
- Companion file with one dated paragraph per finding: `/home/user/Claude-/.claude/skills/eris-playbook/references/proposals.md`.

---

## 1. Shake-ups and adaptive overfitting: how much to trust a small public slice

### 1.1 The public slice is mostly noise; compute its standard error before reading it  [PUBLIC RESEARCH]
- Practice: before interpreting any public score, compute `se_pub` from the OOF per-row loss (or an OOF bootstrap of size n_pub for AUC/F1-type metrics) and express the CV-vs-public gap as a z-score. Gaps with |z| < 2 are noise: do not change course. Do not compute CV-LB "correlation" from 3-5 submissions.
- Why: binomial error bars are about +-7% at n=100 and +-2% at n=1000; a public/private pair of n=30/28 differed by about +-15% [S-full, Varoquaux Fig. 1]. In the SIIM-ISIC winner's 18 single models the score std was 0.0012 (cv_all), 0.0043 (cv_2020), 0.0060 (private), 0.0093 (public); the public score's correlation with CV was "essentially 0" for single models [S-full, arXiv 2010.05351]. One LANL competitor's 4 submissions had corr(CV, public)=+0.9986 but corr(public, private)=-0.7884 [S-snip].
- Check (seconds): see appendix `public_z`. For AUC, repeat 200x: sample n_pub OOF rows (stratified like the public slice), compute the metric, take the std. Print the z beside every submission.
- Source: https://arxiv.org/pdf/1706.07581 [S-full]; https://arxiv.org/abs/2010.05351 [S-full]; https://medium.com/kaggle-lanl-earthquake-prediction/kaggle-lanl-earthquake-prediction-b028ae69871a [S-snip].
- Cross-check: extends CLAUDE.md 4A ("public LB is a soft signal") and platform-facts section 6.3 with a number instead of a posture. New: the z-score and the "no correlation from <=5 points" rule.

### 1.2 Adaptive overfitting to a leaderboard is empirically small; split mismatch, tiny test sets and selection among many local candidates are what hurt  [PUBLIC RESEARCH]
- Practice: spend effort on split fidelity and on limiting local selection, not on fear of "probing" the public board with the 3-6 credit-limited submissions Eris allows. Keep the final pick CV-based anyway.
- Why: Roelofs et al. (NeurIPS 2019) analysed 120 Kaggle competitions (34 accuracy competitions with >=1,000 submissions, up to ~35,000 each): public and private scores lie close to y=x, effect sizes are typically <1% accuracy, and the outliers were non-i.i.d. public/private splits and tiny test sets (n=100, 209; one MEG set derived from 7 subjects). They suggest >=10,000 test examples as a reasonable minimum. Among the top 10% of submissions the public score overstates private slightly (about 1% in one competition, reversed in another). Large rank changes among near-tied scores are "a natural consequence of random noise" [S-full]. The Ladder paper shows the theoretical risk exists: a boosting attack on a plain leaderboard biases it by about sqrt(k/n), while the Ladder (release a score only on a statistically significant improvement) gives (log k / n)^(1/3) [S].
- Check: from the description's public/private sizes, simulate N_solvers models whose true scores lie within 1 SE and look at the expected rank spread. If the spread is wider than the prize gaps, ranks near the top are partly luck: maximise expected score and compliance, skip micro-gains.
- Source: https://proceedings.neurips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf [S-full]; https://arxiv.org/pdf/1502.04585 [S].
- Cross-check: converges with CLAUDE.md 4A "Generalise, don't overfit". New: the evidence that the dominant risks are shift/split/size, and the ">=10,000 rows" yardstick for how much to trust any held-out slice (including our own CV folds).

### 1.3 Absolute scores fall under shift, orderings mostly survive; use CV for ordering, not as a promised level  [PUBLIC RESEARCH]
- Practice: quote an expected private band (CV mean minus a shift allowance), not the CV mean. Judge candidate ordering under a shift-faithful split; if a candidate's lead over another shrinks or flips from random-CV to group/time-CV, trust the group/time number and suspect a shortcut.
- Why: freshly collected test sets for CIFAR-10 / ImageNet dropped accuracy 3-15% / 11-14% while model ranking was largely preserved, and the authors attribute the drop to distribution shift, not to adaptivity [S-snip, Recht et al.]. In-distribution and out-of-distribution accuracy are strongly correlated (linear after a probit transform) across many models and shifts [S, Miller et al.], but inverse correlations appear on some real-world shifts [S, Teney et al.], so ordering carry-over is a default, not a guarantee.
- Check: score the same 4-8 candidates under random-CV and group/time-CV; print Spearman rho and the biggest rank flips. A large flip is a flag to audit that candidate.
- Source: https://arxiv.org/pdf/1902.10811 ; https://arxiv.org/pdf/2107.04649 ; https://arxiv.org/pdf/2209.00613 [S-snip/abstract].
- Cross-check: extends V5 (bias direction of proxies) with the "ordering vs level" distinction. Our private-slice level is unknowable; ordering is what CV can buy.

### 1.4 What separated winners in public post-mortems  [COMPETITION PATTERN]
- Practice: (a) design validation first, before modelling; (b) ignore or heavily down-weight the public board; (c) average many diverse members; (d) keep post-processing low-dimensional and cross-fitted; (e) limit the number of experiments scored against any single feedback source.
- Evidence: SIIM-ISIC winner: "did not use public LB score feedback in any way", validated by 5-fold on a combined set with a stabler positive rate (the external data is not allowed on Eris), sanity-tracked `cv_2020`, 18 diverse models, 5 folds split differently per model, final = simple average of probability ranks; their `cv_2020` 0.9600 -> private 0.9490, i.e. even the winner's CV overshot private by ~0.011 [S-full]. Park: 2nd on public -> 52nd on private with 42 submissions (median 6) and feature selection outside the CV folds [S-full]. LANL: winners used leave-one-earthquake-out CV because the train file held only ~15 independent cycles [S-snip]. AES 2.0: heavy shake-up attributed partly to QWK threshold overfitting to the public board; the eventual winner moved from 619th public to 1st [S-snip]. Playground S5E12 1st: a hill-climbing blend had better post-cutoff CV (0.70886) than a ridge blend of the top 36 members (alpha=10, 0.70860) but the ridge blend scored better on public and private [S-snip].
- Check: none new; use 1.1, 2.2 and 4.1 as gates.
- Source: https://arxiv.org/abs/2010.05351 [S-full]; https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem [S-full]; https://hippocampus-garden.com/kaggle_aes2/ [S-snip]; https://www.kaggle.com/c/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl [S-snip].
- Cross-check: converges with 00-s-tier moves 4 and 6 and with V3 ("A recorded 0.328 ensemble vs 0.288 best single was exactly selection optimism") and V9. The SIIM "CV already overshoots private by ~0.011 even for the winner" calibrates how large the allowance in 1.3 can be.

### 1.5 Final selection under noise  [UNTESTED HYPOTHESIS]
- Practice: treat CV and the public score as two noisy measurements and combine by inverse variance: `w_pub = (1/se_pub^2) / (1/se_pub^2 + 1/se_cv^2)`. With a small public slice `w_pub` is tiny, so pick by CV; use the public score only as a veto (|z| > 3 after 1.1: investigate a split mismatch or leak, do not tune to it). Prefer the more regularised, more diverse of two near-tied (within 1 NB-corrected SE) candidates.
- Why: this is the standard precision-weighted estimate; Kaggle practitioners state the same instinct qualitatively ("discard experiments that widen the CV-LB gap", "average the best-CV and best-LB models") [S-snip, Medium handbook]. Whether Eris scores the best, latest or a chosen submission is not stated in the files I read: open question, ask a reviewer/docs.
- Check: compute `w_pub` per submission; if it is < 0.1, a credit spent for a tiny expected delta buys no information (resolvability: paired SE of the OOF per-row loss difference / sqrt(n_pub) must be well below the expected delta).
- Source: own derivation; https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e [S-snip].
- Cross-check: new. Interacts with platform-facts section 2 (credits are costly).

---

## 2. Validation recipes

### 2.1 Derive the split from the data generator, and size folds like the test pool  [GENERAL ML PRINCIPLE]
- Practice: group by the largest unit that the description or data implies (patient, subject, site, source, episode, cycle); time-order if the future is predicted, with a `gap`/embargo; stratify within group-aware splitters for rare labels.
- Why: ignoring structure "results in serious underestimation of predictive error" and block CV should be used wherever dependence exists even when residuals look uncorrelated [S, Roberts et al.]. sklearn documents GroupKFold / StratifiedGroupKFold (greedy, can be imbalanced) / TimeSeriesSplit(gap) and says group-wise CV is "safer" when the generator has group structure [S-full]. Purging and an embargo remove overlapping-label leakage in temporal data [S, Lopez de Prado]. LANL winners built leave-one-cycle-out CV [S-snip]. Patient-level leakage inflated scores by up to 41% even under 10x5 repeated CV [S, Bussola et al.].
- Check (4-core, minutes): run the same model under random K-fold vs group K-fold (and time-split if relevant); the gap is the leakage/shift magnitude. Report both. If the gap exceeds ~1 NB-corrected SE, the split matters and the grouped number is the one to trust.
- Source: https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf ; https://scikit-learn.org/stable/modules/cross_validation.html ; https://en.wikipedia.org/wiki/Purged_cross-validation ; https://link.springer.com/chapter/10.1007/978-3-030-68763-2_13 .
- Cross-check: converges with V1, V2, V5, V6 and CLAUDE.md 4A ("split mismatch is the #1 cause of private-LB drops"). New: the explicit random-vs-group gap as the severity gauge, and the purge/embargo vocabulary for overlapping labels.

### 2.2 Accept a change only with a corrected paired test; repeated CV reduces split noise but not the data-sampling floor  [PUBLIC RESEARCH]
- Practice: for candidates A, B on identical splits, collect the paired per-fold metric differences d over r repeats x k folds and compute `t = mean(d) / sqrt((1/(r*k) + n_test/n_train) * var(d))` (Nadeau-Bengio corrected resampled t, df=r*k-1). Accept only when |t| clears roughly 2.1-2.6, or the sign is consistent across all folds and repeats and the effect is practically meaningful.
- Why: naive fold-to-fold SE ignores training-set overlap and is too small: at k=5 the correct variance is (1/5+1/4)/(1/5)=2.25x larger (SE x1.5); at k=10 about x1.45 in SE; for 3 repeats of 5 folds about x2.2 (my arithmetic from the published formula). Varoquaux found fold-SE underestimates the true error bars by a factor ~0.7 even in the best case [S-full]; Bates-Hastie-Tibshirani show naive CV intervals can have coverage "far below the desired level" because fold accuracies are correlated, and that CV estimates the average over training sets rather than the one fitted model [S]. Bouckaert-Frank recommend 10x10 CV with this correction (or 100 resamples) for replicability [S]. Kohavi: ten-fold stratified is the best default for model selection, even with spare compute [S].
- Check: appendix `nb_corrected_t`. Add to the CV driver: print `t`, mean difference, and the per-fold sign pattern for every comparison. Cheap (no extra training beyond the repeats you already run).
- Source: http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf ; https://link.springer.com/chapter/10.1007/978-3-540-24775-3_3 ; https://arxiv.org/abs/2104.00673 ; https://mlanthology.org/ijcai/1995/kohavi1995ijcai-study/ .
- Cross-check: REFINES V4 ("1 SE of the paired fold differences" understates the true SE by ~1.5x at 5-fold). Not a contradiction of the spirit; change the number.

### 2.3 Proportionate selection discipline: cap candidates, account for the winner's curse, cross-fit post-hoc steps  [PUBLIC RESEARCH]
- Practice: (a) keep flat CV for choosing among a handful of model families; (b) cap the number of configs/HPO trials to what the CV noise can resolve and state N; (c) subtract an expected-maximum allowance when reporting a best-of-N; (d) cross-fit blend weights, thresholds, decode constants, temperature and feature-block decisions (fit on group-half A, score on half B, swap).
- Why: flat CV is optimistically biased as an estimate but, for choosing among classifiers with few hyperparameters, nested CV rarely changes the winner (12 algorithms x 115 datasets) [S-full, Wainer-Cawley]; the bias matters for the reported number and grows with the search size (Cawley-Talbot, Varma-Simon) [S]. "Overtuning": in ~10% of HPO benchmark cases the selected config generalised worse than the default; worst in small data and with holdout validation [S, Schneider et al.]. Selection-induced bias scales with the candidate pool and can be estimated from order statistics [S, McLatchie-Vehtari]. Krstajic et al. stress repeating CV when selecting and repeating nested CV when assessing [S].
- Check: expected maximum of N standard normals: N=5: 1.16, 10: 1.54, 20: 1.87, 50: 2.25, 100: 2.51 [R, standard order-statistic values; appendix `best_of_n_allowance` recomputes them by Monte Carlo]. If N configs are truly equal and the selection noise per config is sigma, the best-of-N CV looks ~E[max]*sigma better than truth. Example: sigma=0.005, N=20 -> ~0.009 of pure luck. Report `best_cv - E[max(N)]*sigma_sel` as the honest number. Cross-fit check: appendix `cross_fit_blend`.
- Source: https://arxiv.org/pdf/1809.09446 ; https://www.researchgate.net/publication/220320908_On_Over-fitting_in_Model_Selection_and_Subsequent_Selection_Bias_in_Performance_Evaluation ; https://arxiv.org/abs/2506.19540 ; https://arxiv.org/abs/2309.03742 ; https://link.springer.com/article/10.1186/1758-2946-6-10 .
- Cross-check: converges with V3 and CLAUDE.md 4A (limit HPO trials). New: the E[max] arithmetic and the "flat CV is enough for choosing, nested for reporting" cost split, which saves hours on a 4-core CPU.

### 2.4 Fresh-seed confirmation and a sanity holdout, because your own CV is also a reused test set  [GENERAL ML PRINCIPLE]
- Practice: pick finalists with the working folds; then re-score the top 2 on 3 fresh split seeds (re-drawing group assignment, same split rule) and on the never-touched sanity holdout. Ship the one that wins on fresh seeds; if neither separates, ship the simpler/more regularised.
- Why: repeated experimentation against fixed folds is the same adaptive-reuse mechanism that the Ladder and Roelofs studies analyse; the Ladder's remedy is to credit only improvements beyond noise [S]. Models selected by repeated looks can still be near-identical in predictions, which limits (but does not remove) the damage [S, Mania et al.].
- Check: a `--fresh-seeds` run for the top 2 only (cost: 2 x 3 CV runs).
- Source: https://arxiv.org/pdf/1502.04585 ; https://arxiv.org/abs/1905.12580 .
- Cross-check: converges with V4 (sanity holdout) and CLAUDE.md 4A ("keep a holdout sanity fold"). New: the explicit fresh-seed re-draw for finalists.

### 2.5 For rare positives and threshold-sensitive metrics, stabilise the yardstick before comparing models  [COMPETITION PATTERN]
- Practice: with few positives the metric is noisy even under 5-fold; average over repeats, compare on a smooth secondary metric (log-loss/Brier/rank loss) as well, put thresholds on a plateau and fit them cross-folded, and average several seeds before declaring a gain.
- Why: SIIM-ISIC had 1.76% positives, AUC unstable "even with 5-fold cross validation", worse on the 1/10-size public board; ensembling reduced the noise ("the bigger the ensemble, the more stable the LB score") [S-full]. AES 2.0 QWK was sensitive to chosen thresholds and tuned-to-public thresholds shook up [S-snip].
- Check: compute seed-to-seed std of the metric on one fixed split; if it exceeds half the gain you are chasing, average >=3-5 seeds first. Print the metric at thresholds +-delta around the chosen one; pick the middle of the plateau.
- Source: https://arxiv.org/abs/2010.05351 ; https://hippocampus-garden.com/kaggle_aes2/ .
- Cross-check: converges with V15 (saturated metrics) and metric-and-decoding section 6.

### 2.6 Adversarial validation: use as a design diagnostic only; most "uses" cross the test-fitting ban  [GENERAL ML PRINCIPLE]
- Practice: a train-vs-test classifier (AUC ~0.5 means same distribution) diagnoses shift [S, FastML; Lopez-Paz & Oquab C2ST]. On Eris the compliant version works from the description and train only: (i) fold-vs-rest AUC under the chosen CV (how separable are validation folds from their training folds); (ii) AUC between train subsets that the description says differ (period, site); (iii) severity ladder: random-CV AUC vs group-CV AUC.
- Why: Rabanser et al. find domain classifiers are poor in the low-sample regime (<=100) and fine with more data; domain-discriminating approaches mainly help characterise shift qualitatively and judge harm [S]. CLAUDE.md 2.3 #5 bans test-based reweighting, calibration on the test distribution and fitting anything on test or train+test; section 4 step 3 allows "adversarial thinking only; do not adapt to test" [verified in this repo's CLAUDE.md text].
- Which uses break the ban (my reading, not a reviewer ruling): inside `solution.py`: importance weights p(test|x)/p(train|x); selecting validation rows most similar to test (FastML's original recipe); dropping features by test-vs-train AUC; any train+test model. Offline and not encoded: reading test features in scratch is grey (it is still a test read; learned-patterns L002/L005 already record the user's caution). Safest: do (i)-(iii) only.
- Check (4 cores, ~1 min): LightGBM 100 trees, n_jobs=4, 5-fold inner CV, label = in-validation-fold vs in-training-fold, repeated for each outer fold.
- Source: https://fastml.com/adversarial-validation-part-one/ ; https://arxiv.org/abs/1610.06545 ; https://arxiv.org/pdf/1810.11953 .
- Cross-check: new to the playbook (grep found no adversarial-validation entry). Related: learned-patterns L002, L005.

---

## 3. Leakage tells and audits

### 3.1 Fit every transformer, selector, encoder inside the fold; prove the pipeline with a label-shuffle test  [GENERAL ML PRINCIPLE]
- Practice: wrap preprocessing + selection + model in one pipeline fitted per fold. Run the full CV on randomly permuted labels (group-level permutation if the label is group-level): the score must sit at chance +- noise.
- Why: Park's "increasingly optimistic CV" came from feature selection and preprocessing outside the folds [S-full]. sklearn's guide shows scaler/selection leakage and recommends Pipelines [S-full]; unsupervised preprocessing fitted before CV also biases it [S, arXiv 1901.08974]. Kapoor-Narayanan survey 294 papers in 17 fields where leakage inflated results, some of which vanished once fixed [S-snip]. Permutation testing gives good statistical control of CV accuracy [S-full, Varoquaux].
- Check (minutes): appendix `shuffle_label_check`; expect AUC ~0.5 / R2 ~0. Anything above noise is a leak in the pipeline.
- Source: https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem ; https://scikit-learn.org/stable/modules/cross_validation.html ; https://arxiv.org/pdf/1901.08974 ; https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9 .
- Cross-check: converges with CLAUDE.md 2.3 #5 (compliance and validity coincide here) and V12. New: the shuffle-label gate as a cheap whole-pipeline detector.

### 3.2 Duplicates and sibling rows  [PUBLIC RESEARCH]
- Practice: find exact and near duplicates and shared-source rows on train only; fold by whole groups; measure how much CV changes when the duplicates are split vs grouped.
- Why: 3.3% (CIFAR-10) and 10% (CIFAR-100) of test images have near-duplicates in train, and accuracy drops noticeably on a duplicate-free test set [S, Barz-Denzler]. Patient-level leakage inflated scores up to 41% under 10x5 repeated CV [S, Bussola et al.]. Kaufman et al. frame leakage as information "not legitimately available" and propose a learn-predict separation [S].
- Check: cosine/L2 nearest neighbour (sklearn NearestNeighbors, n_jobs=4) on standardised features or train-only embeddings: compare the distribution of nearest-train distance for validation rows under random-CV vs group-CV. A far smaller random-CV distance means siblings.
- Source: https://arxiv.org/abs/1902.00423 ; https://link.springer.com/chapter/10.1007/978-3-030-68763-2_13 ; https://dl.acm.org/doi/epdf/10.1145/2382577.2382579 .
- Cross-check: converges with V2 and rejection class S (siblings) in engineering-and-compliance.

### 3.3 Group aggregates and target statistics: cross-fit, with matched statistics at test time  [GENERAL ML PRINCIPLE]
- Practice: any statistic computed from labels or from a group that contains the row itself must be out-of-fold for training rows; the encoder fitted on all train is used only for held-out rows.
- Why: sklearn's TargetEncoder uses internal cross-fitting in `fit_transform` and discourages `fit().transform()` on train because it leaks the target; the docs say train data should always go through `fit_transform` [S-full].
- Check: score the same model with in-fold (leaky) vs cross-fitted statistics; a large gap shows how much leak would have been believed. Also compare the distribution of a statistic on train rows vs on validation rows (same-size statistics).
- Source: https://scikit-learn.org/stable/modules/preprocessing.html .
- Cross-check: converges with V8 (adds the sklearn reference).

### 3.4 Id/order/size shortcuts and the "too good" CV tell  [GENERAL ML PRINCIPLE]
- Practice: train an "id-only" model (row index, numeric id, file position, text length, file size) under the grouped CV; any lift above the constant baseline is a shortcut that private data will not share. Treat near-perfect CV on a noisy task as a leak until proven otherwise.
- Why: leakage via ids/ordering is a classic data-mining failure [S, Kaufman et al.; details of specific contests are [R]]. The playbook already lists row order, ids and file sizes as banned/fragile features.
- Check: LightGBM, 100 trees, n_jobs=4, features only from the id/order set; grouped CV; report lift. Minutes.
- Source: https://dl.acm.org/doi/epdf/10.1145/2382577.2382579 .
- Cross-check: converges with 00-s-tier anti-patterns (ids/row order/file sizes as features), V12 and Q5. Adds the concrete "id-only model" probe.

### 3.5 Telling a leak from noise when CV exceeds the public score  [UNTESTED HYPOTHESIS]
- Practice: use the z-score of 1.1. If the CV-public gap is within 2 SE_total, call it noise. If it exceeds ~3 SE_total, audit 2.1 and 3.1-3.4 before touching the model; do not "fix" it by tuning to the public value.
- Why: pure noise is common (1.1); a real mismatch usually shows up as a persistent gap in the same direction across several candidates (Roelofs found systematic gaps only in non-i.i.d./small-n cases) [S-full].
- Check: one line per submission. Needs >= 2 submissions to look for a consistent sign; with <= 4 points do not run a correlation.
- Source: own derivation from https://proceedings.neurips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf .
- Cross-check: refines V12 ("CV >> public LB" tell) by separating noise from mismatch.

---

## 4. Ensembling for robustness

### 4.1 Why simple blends win under shift, and how to fit weights if you must  [PUBLIC RESEARCH]
- Practice: default to equal weights (on a comparable scale), or a non-negative, sum-to-one, strongly shrunk (ridge) combination with weights capped to the number of families, cross-fitted by groups. Treat many-free-weight hill climbing as a selection step that needs a nested check.
- Why: estimated combining weights add variance; when the true optimum is close to equal weights the simple average beats the weighted one (forecast combination puzzle) [S, Smith-Wallis]. Breiman found non-negativity necessary for stacking to help [S]. Caruana-style forward selection "sometimes overfits to the hillclimbing set", worse as the library grows, mitigated by bagged selection [S]. Playground S5E12 1st: ridge (alpha=10) over the top 36 beat hill climbing on public and private despite slightly lower CV; S6E2 1st: "aggressive weight search tended to overfit easily" [S-snip].
- Check: appendix `cross_fit_blend` (fit on group half A, evaluate on B, swap; compare to equal weights using 2.2's corrected t). Keep fitted weights only if the held-out gain is clear.
- Source: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0084.2008.00541.x ; https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf ; https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf ; https://www.kaggle.com/c/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl .
- Cross-check: converges with E-A ("few free weights", "nested check") and V3. New: the theoretical reason (weight-estimation variance) and the ridge-over-hill-climb anecdote.

### 4.2 Diversity: model family, feature view, target, resolution, split, init; do not expect special OOD magic  [PUBLIC RESEARCH]
- Practice: build members that differ in representation or assumption (backbone/size/resolution, target definition, a metadata branch, a different fold split) and add a member only if the blend gains beyond the 2.2 threshold. For neural nets, independent random inits are a real diversity source; for GBDTs, seeds alone add little, so vary features/objective instead.
- Why: the SIIM winner's diversity axes were backbone, input size, target (4 vs 9 classes), a metadata branch and a different fold split per model [S-full]. Deep ensembles beat other approximate-Bayesian methods under larger shift [S, Ovadia]; random-init decorrelation is hard for subspace methods to match [S, Fort et al.]. Caveat: an ensemble's OOD performance is largely determined by its in-distribution performance, and a single larger model can replicate the gain, so ensembles are variance reduction, not a robustness spell [S, Abe et al.]. Predictions agreeing between models also limit test-set overuse [S, Mania et al.].
- Check: print the OOF prediction correlation matrix of candidate members and the blend gain from adding each; keep only members whose marginal gain beats the corrected threshold.
- Source: https://arxiv.org/abs/2010.05351 ; https://arxiv.org/abs/1912.02757 ; https://arxiv.org/abs/2202.06985 ; https://mlwave.com/kaggle-ensembling-guide/ .
- Cross-check: converges with E-A ("diversity of modelling assumption beats seeds"). Adds the nuance that for deep nets init-diversity is real and that gains transfer through ID accuracy.

### 4.3 Rank averaging must be per-member train-reference percentiles, never test-batch ranks  [VERIFIED PLATFORM REQUIREMENT]
- Practice: map each member's score through a monotone function fitted on its OOF/train scores (percentile or z-score), then average. Never rank or z-score across the test rows.
- Why: Kaggle winners (including SIIM) average within-test ranks, which is a test-wide statistic: CLAUDE.md 2.3 #5 and section 7 list "rank-normalising over the test set" as a red item, and engineering-and-compliance class R lists "test-batch rank/z-score normalisation" as a rejection reason [verified in repo text]. MLWave's rank-averaging rationale (members not equally calibrated) is real and is met by the train-reference version [S].
- Check: the existing half-rows test: drop half the test rows; the surviving rows' predictions must be identical.
- Source: https://mlwave.com/kaggle-ensembling-guide/ ; repo: `/home/user/Claude-/.claude/skills/eris-playbook/references/engineering-and-compliance.md` (class R).
- Cross-check: CONVERGES with E-A ("train-reference percentile per member") and R/Q7. Included to stop public write-up code being copied literally.

### 4.4 Seeds and folds: cheap variance reduction, and the stability check  [GENERAL ML PRINCIPLE]
- Practice: average several seeds/fold-models within the fixed plan; verify stability by running the script on two seeds and correlating test predictions.
- Why: ensembling lowers single-model public-board noise (SIIM) [S-full]; CLAUDE.md 4A already asks for the two-seed stability run.
- Check: two seeds, Pearson/Spearman of test predictions and the OOF metric difference; large swings mean too few seeds/folds.
- Source: https://arxiv.org/abs/2010.05351 .
- Cross-check: converges with CLAUDE.md 4A "Robustness checks".

---

## 5. Uncertainty and calibration under shift without target-set fitting

### 5.1 Calibrate on group-held-out OOF, not random OOF (multi-domain calibration)  [PUBLIC RESEARCH]
- Practice: when probabilities, thresholds or decode temperatures matter, fit the post-hoc map (temperature/Platt/isotonic, few parameters) on OOF predictions from the grouped/time-shifted CV, and score it on groups not used to fit it. Prefer a single global temperature over per-slice maps unless slices differ on grouped CV.
- Why: temperature scaling is calibrated in-distribution but degrades under dataset shift; ensembles hold calibration best as shift grows [S, Ovadia et al.]. Recalibrating across several domains in the validation data improves performance on unseen domains (WILDS) [S, Wald et al.]. Guo et al.'s temperature scaling is a one-parameter fit on a validation set, so fitting it on shift-faithful OOF is the natural carry-over [S].
- Check: log-loss on held-out groups with T fitted on random-OOF vs on group-OOF; the difference is the shift-sensitivity of your calibration. Also ECE per group vs pooled.
- Source: https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf ; https://arxiv.org/abs/2102.10395 ; https://arxiv.org/abs/1706.04599 .
- Cross-check: converges with V3, V7 and metric-and-decoding sections 5-6 ("temperature fitted on OOF", "Calibrate decode parameters honestly"). New: group-OOF as the fitting set and the ECE-by-group diagnostic.

### 5.2 Ensembles as the train-only uncertainty lever  [PUBLIC RESEARCH]
- Practice: multiple independently initialised/fitted members, averaged in probability or logit space after calibration, are the strongest train-only way to improve calibration under shift; do not add test-time adaptation.
- Why: Ovadia et al. [S]. The Abe et al. caveat from 4.2 applies: the OOD benefit tracks ID accuracy [S].
- Check: log-loss/ECE of single vs averaged members on group-held-out OOF.
- Source: https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf ; https://arxiv.org/abs/2202.06985 .
- Cross-check: converges with E-A.

### 5.3 Per-row novelty shrinkage fitted on grouped OOF  [UNTESTED HYPOTHESIS]
- Practice: compute a per-row novelty score as the distance to a frozen train reference set (per row, no test-wide statistic); shrink predicted probabilities toward the train prior with a slope fitted on grouped OOF. Skip if grouped CV shows no gain.
- Why: under shift the model is over-confident (Ovadia); a per-row, train-only novelty function satisfies "each prediction depends only on its own row and a frozen train-fit model" (engineering-and-compliance class R, half-rows test). No external study found with this exact recipe: hypothesis. Methods needing test covariates (importance-weighted calibration, weighted conformal prediction) would violate CLAUDE.md 2.3 #5 [R].
- Check: grouped-CV log-loss/Brier with vs without shrinkage; two extra parameters only; then the half-rows test.
- Source: own hypothesis; background https://arxiv.org/abs/2102.10395 .
- Cross-check: new; related to metric-and-decoding "shrinkage toward a train-only prior".

---

## 6. Cross-check summary against the existing playbook

| Finding | Status vs playbook |
|---|---|
| 1.1 public-slice z-score; no correlation from <=5 points | NEW (extends CLAUDE.md 4A, platform-facts 6.3) |
| 1.2 adaptivity small; shift/split/size dominate; >=10,000 rows yardstick | NEW evidence for CLAUDE.md 4A |
| 1.3 level drops, ordering holds (with exceptions); report a band | NEW (extends V5) |
| 2.1 split from generator; random-vs-group gap; purge/embargo | CONVERGES V1/V2/V5; adds the gap gauge |
| 2.2 Nadeau-Bengio corrected t | REFINES V4 (naive SE too small by ~1.5x at 5-fold) |
| 2.3 winner's-curse arithmetic; flat CV for choice, nested for report | CONVERGES V3; adds E[max] numbers and cost split |
| 2.4 fresh-seed confirmation of finalists | NEW (extends V4 sanity holdout) |
| 2.5 stable yardstick for rare positives/threshold metrics | CONVERGES V15 / metric-and-decoding 6 |
| 2.6 adversarial validation (design-only, train-only variants) | NEW; compliance split spelled out |
| 3.1 shuffle-label pipeline test | NEW cheap gate |
| 3.2-3.4 siblings, cross-fit stats, id-only model | CONVERGES V2/V8/V12/Q5; adds probes |
| 4.1 simple blends / ridge over hill-climb | CONVERGES E-A; adds theory |
| 4.2 diversity axes; ensembles are variance reduction | CONVERGES E-A; adds Abe nuance |
| 4.3 rank averaging = train-reference percentiles | CONVERGES E-A / R / Q7 (warn against copying public code) |
| 5.1-5.3 calibration under shift | CONVERGES metric-and-decoding 5-6 / V7; adds group-OOF fitting |

No contradictions with the curated files found. One tension to note: public winners' "rank average" code (SIIM) is non-compliant here as written (4.3).

---

## 7. The 10 most useful practices, ranked by expected private-LB gain per hour

1. Reconstruct the hidden split and run the random-vs-group (and time) CV gap; fold by derived groups. (2.1, 3.2) Largest historical cause of shake-ups; ~1 h; protects against large losses (LANL, Bussola 41%).
2. Whole-pipeline hygiene: everything fitted inside folds, cross-fitted target/group statistics, label-shuffle test. (3.1, 3.3) ~30 min; removes the most common silent optimism (Park).
3. Read the public score only through its z-score and never tune to it; no correlation from <=5 submissions. (1.1, 3.5, 1.5) ~5 min; avoids chasing noise (SIIM public std 0.0093 vs cv_all 0.0012).
4. Corrected paired test (Nadeau-Bengio) with 5-10 folds x 2-3 repeats before accepting any change; sign consistency across folds. (2.2) ~15 min to implement; replaces most false "improvements".
5. Simple/regularised blend of diverse members; cross-fit weights; per-member train-reference percentiles. (4.1, 4.2, 4.3) Variance reduction is the most reliable gain; moderate compute.
6. Cap candidates and quote a winner's-curse-adjusted CV; flat CV for choice, nested/half-split only for reporting and post-hoc constants. (2.3) ~30 min; avoids selecting noise (HPO overtuning ~10% of cases).
7. Fresh-seed re-draw for the top 2 finalists plus the untouched sanity holdout. (2.4) 2 x 3 CV runs; catches CV-level adaptivity.
8. Id/order-only probe and duplicate/sibling nearest-neighbour audit. (3.2, 3.4) ~30 min; cheap insurance on 'too-good CV' tasks.
9. Stabilise the yardstick for rare/threshold metrics: repeated CV, seed averaging, plateau thresholds, secondary smooth metric. (2.5) Matters mostly for AUC/QWK/F1-type metrics with few positives.
10. Calibrate post-hoc maps on group-held-out OOF; ensemble for calibration; test an optional train-only novelty shrinkage; keep adversarial validation to train-only variants. (5.1-5.3, 2.6) Gain depends on the metric (log-loss/Brier/threshold tasks).

---

## 8. Compliance notes and out-of-bounds items

- Anything above that touches test rows must be a per-row function of a frozen train-fit model (half-rows test). Rank averaging across test rows, importance weighting, test-based feature dropping and train+test adversarial models are not.
- Not recommended and not used: leaderboard probing to infer labels, including the published "Climbing the Kaggle Leaderboard by Exploiting the Log-Loss Oracle" line of work (it appeared in search results for the Ladder; I did not read it and do not endorse using it).
- SIIM winner's use of 2018-2019 external data for a stabler validation is banned on Eris (CLAUDE.md 2.3 #3); only the idea (stabilise the yardstick) transfers, via repeats and seed-averaging.
- No Shipd/Eris solver code or private material was fetched. Official Eris docs were not found in public search beyond marketing pages, so the platform rule for which submission counts as "final" remains unverified.

## 9. Open questions

1. Eris: does the best, the latest or a selected submission determine the private rank? (Affects 1.5.)
2. Eris: exact public/private split sizes per challenge (the description usually prints test size); needed for 1.1.
3. Kaggle write-up pages could not be fetched; items marked [S-snip] rest on search-result summaries and should be re-read from the originals before being promoted to curated references.
4. Nothing here was run locally; promote a recipe only after a local CV experiment (e.g., confirm the Nadeau-Bengio correction ratio and the E[max] bias by simulation on a real challenge's OOF).

---

## Appendix: minimal check code (UNTESTED: written without an execution tool; stdlib + numpy/scipy only; fine on 4 CPU cores)

```python
import numpy as np
from scipy import stats

def nb_corrected_t(d, n_test, n_train):
    """d = paired per-fold metric differences (A - B) over r repeats x k folds, SAME splits for A and B."""
    d = np.asarray(d, float); m = len(d)
    se = np.sqrt((1.0 / m + n_test / n_train) * d.var(ddof=1))   # Nadeau-Bengio variance correction
    t = d.mean() / se
    return t, 2 * stats.t.sf(abs(t), df=m - 1)

def public_z(pub, cv_mean, se_cv, oof_row_loss, n_pub):
    """z-score of the public score against the CV estimate (row-additive loss metrics)."""
    se_pub = np.asarray(oof_row_loss, float).std(ddof=1) / np.sqrt(n_pub)
    return (pub - cv_mean) / np.hypot(se_cv, se_pub)

def best_of_n_allowance(n_cfg, sigma_sel, n_mc=20000, seed=0):
    """Expected optimism of the best of n_cfg equally good configs (selection noise sigma_sel per config)."""
    z = np.random.default_rng(seed).standard_normal((n_mc, n_cfg))
    return z.max(axis=1).mean() * sigma_sel

def shuffle_label_check(run_cv, X, y, groups, n_rep=3, seed=0):
    """run_cv(X, y, groups) must execute the FULL fold pipeline and return the OOF metric.
    On permuted labels the score should sit at chance level (AUC ~0.5, R2 ~0)."""
    rng = np.random.default_rng(seed)
    scores = [run_cv(X, rng.permutation(y), groups) for _ in range(n_rep)]
    return float(np.mean(scores)), float(np.std(scores))

def cross_fit_blend(P, y, groups, fit_weights, metric, seed=0):
    """P: (n, m) member OOF predictions. fit_weights(P, y) -> w. Returns held-out gain of fitted vs equal
    weights on each group-half (sign convention: metric higher is better)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); half = set(rng.permutation(ug)[: len(ug) // 2].tolist())
    a = np.array([g in half for g in groups]); gains = []
    for tr, te in ((a, ~a), (~a, a)):
        w = fit_weights(P[tr], y[tr])
        gains.append(metric(y[te], P[te] @ w) - metric(y[te], P[te].mean(axis=1)))
    return gains
```
