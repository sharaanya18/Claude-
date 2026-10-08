# Split audit — assembly_amendments (CV harness diagnostics)

Author: eris-validation-architect. Computed by `validate.py` (`python3 /home/user/Claude-/challenges/assembly_amendments/validate.py`) from `dataset/public/train.csv` + `train_items.csv` + `train_targets.csv` only — never reads test.* (see module docstring).

## Hidden-split reading

The challenge states bills, not boards, are split ("Bills are split by legislative dossier"), and that committee + plenary + every reading of a bill sit on the same side (contract/data_audit: 0 `bill_id` overlap between train (209 bills) and test (111 bills), confirmed again here). We read this as the strictest plausible held-out unit — one bill, not one board, not one (bill, examining_body) pair — because the description's own leakage rationale (55% of bills re-table identical text across committee/plenary/readings) implies that anything coarser than whole-bill disjointness would leak. Held-out unit size is not stated beyond "the test set is divided into two evaluation slices by bill" (unknown boundary, unknown sizes); we therefore do not assume a specific private/public split ratio, only that CV must be bill-grouped and should report subgroup slices (division/body/kind) in case either hidden slice correlates with one of them.

## Group derivation

- 209 train bills -> 203 merged groups after union-merging bills that share >= 3 identical dispositif texts of >= 40 normalized characters (10 bills touched by a merge, 6 triggering pairs, out of 248 distinctive texts that appear in >=2 bills at all).
- Largest merged group: 3 bills.
- **Critical finding, load-bearing for this design**: without the length floor, the SAME rule (exact text match, >=3 shared, union-merge) chains 72/209 bills into ONE group covering 84.9% of all boards and 90.8% of all items — the transitive closure of generic one-line amendments (dominant offender: normalized `"supprimer cet article"`, 1553 occurrences across train, present in nearly every bill regardless of topic) collapses almost the whole dataset into a single unsplittable component, which would make bill-grouped K=5 CV impossible. We verified the length floor is not a knife-edge choice: any floor in [40, 100] normalized characters gives the identical 203-group result (6 triggering pairs, largest merged group 3 bills), and the 6 surviving pairs share texts with a median length over 300 characters, each citing a specific code article/provision — exactly the budget-cycle re-tabling the challenge describes, not boilerplate. This is reported, not hidden: a reviewer who prefers the literal unfiltered rule should know it is unusable as stated.


## Fold composition (test-like subset is the primary balance target)

| fold | all boards | all items | bills | test-like boards | test-like items | test-like bills |
|---|---|---|---|---|---|---|
| 0 | 1024 | 5701 | 22 | 331 | 4253 | 22 |
| 1 | 736 | 8482 | 25 | 296 | 4250 | 25 |
| 2 | 860 | 6036 | 32 | 334 | 4253 | 32 |
| 3 | 854 | 5674 | 64 | 337 | 4252 | 32 |
| 4 | 918 | 5673 | 66 | 354 | 4249 | 33 |

Totals: 4392 boards / 31566 items / 209 bills; test-like subset 1652 boards / 21257 items / 144 bills (matches the plan's own profiling of ~1652/144/~21257, confirmed independently here).

Test-like items are balanced to within 0.1% across folds by construction (the primary sort/assign key); test-like boards vary more (the secondary tie-break), reflecting that merged bill-groups differ in average items-per-board.

## Rare-category coverage per fold (test-like subset)

Per `data_audit.md`'s warning: CMP (`texte de commission mixte paritaire`) and `deuxième lecture` boards are bill-clustered and overwhelmingly fail the test-like filter on their own (CMP boards are ~97% adopted -> usually <2 nonzero `counts` entries; `deuxième lecture` is ~93% rejected with little other variety) -- confirmed here: neither reading category appears in the test-like subset's per-fold table below except `nouvelle lecture`, which appears only in one fold. **This is an expected, direct consequence of the mandated bill-grouped split intersected with the test-like filter, not a bug** -- a rare reading clustered in a handful of bills will, by definition of grouping by bill, land in whichever fold(s) those bills land in.

```
reading  nouvelle lecture  première lecture
fold                                       
0                       0               331
1                       0               296
2                       4               330
3                       0               337
4                       0               354
```

## Categorical coverage per fold (test-like subset)

`bill_kind`:
```
bill_kind  (non précisé)  Projet de loi de financement de la sécurité sociale  Projet de loi de finances de l'année  Projet de loi de finances rectificative  Projet de loi ordinaire  Projet de loi relative aux résultats de la gestion et portant approbation des comptes  Projet ou proposition de loi constitutionnelle  Projet ou proposition de loi organique  Proposition de loi ordinaire  Résolution
fold                                                                                                                                                                                                                                                                                                                                                                                                          
0                      1                                                  230                                     0                                        1                       68                                                                                      0                                               0                                       1                            27           3
1                      0                                                    0                                   237                                        0                        0                                                                                      0                                               0                                       0                            51           8
2                      0                                                    0                                     0                                        0                      164                                                                                      0                                               0                                       0                           168           2
3                      0                                                    0                                     0                                        0                      200                                                                                      0                                               0                                       2                           134           1
4                      0                                                    0                                     0                                        3                       86                                                                                      6                                               1                                       0                           257           1
```

`examining_body` (Séance publique vs committee):
```
_is_seance  False  True 
fold                    
0             185    146
1             165    131
2             186    148
3             180    157
4             245    109
```

`division` type:
```
_division_type  additionnel_apres  additionnel_avant  numbered_article  other
fold                                                                         
0                              82                  0               228     21
1                              76                  2               184     34
2                              71                  1               217     45
3                              73                  0               222     42
4                              77                  0               237     40
```

`n_amendments` bucket:
```
_n_bucket  10-29  30-80  4-9
fold                        
0            133     27  171
1            110     36  150
2            101     30  203
3            124     27  186
4            124     25  205
```

## Oracle / sanity checks

- Gold fates/joint fed back through `metric.score_submission`: final score = 100.0000 (expected 100.0) -- PASS.

- Counts-respecting random-shuffle baseline on the test-like subset: 0.2622 (metric_spec.md's own measurement was ~1.7 on a 100-scale with its own sample/seed; this run confirms the fold/filter code integrates correctly with `metric.py`, within a generous band) -- PASS.

- Fold assignment reproduced twice with the same seed: identical -- PASS.

- No `bill_id` spans more than one fold -- PASS.

- All 6 merge-triggering bill pairs stay within a single fold (checked over every merged group, not just the pairs) -- PASS.

- Bill-level bootstrap demo (`bootstrap_ci`, 1000 resamples) on the counts-respecting random-shuffle baseline's test-like detail (the noise floor, NOT a model number): mean=0.4645, 90% CI=(0.0000, 1.5297) over 144 bills -- demonstrates the helper end-to-end; a trained model's OOF will be run through the same function in `solution.py`'s dev logs for the real interval.

## Bias direction of the proxy

- **Optimistic sources**: (1) in-script decode-constant selection (tau, Sinkhorn on/off, pair weighting) is chosen on the same OOF it is then reported on unless `nested_select` is used for the final number -- expect ~1-2 points of optimism from this alone, per the plan's own estimate, and `nested_select` exists precisely to quantify and report it. (2) Stacking: stage-2 models are trained on stage-1 OOF, so stage-2's own OOF retains a sliver of stage-1's fold-boundary smoothing. (3) Test-time stage-1 features are 5-fold-averaged (smoother than any single-fold OOF column), a slight train/test calibration mismatch whose sign is not obvious but is unlikely to be large.
- **Pessimistic / neutral sources**: bill-grouped folds mirror the actual held-out unit exactly (no optimism from an under-grouped split); the length-floored text-merge rule, if anything, UNDER-merges relative to a hypothetical reviewer who wanted the literal rule applied with no floor (which we showed is unusable) -- so if the true leakage surface is broader than our 6 pairs, our CV is mildly optimistic in that one respect (a few extra cross-bill text echoes not accounted for), but we judge this a small residual next to the demonstrated alternative (85% of the data in one group) being strictly worse.
- **Sampling noise**: the private slice is an unknown subset of 111 test bills (~55 at a guess if it is a roughly even two-way split); per-board score SD on the test-like train subset propagates to a private-score SD of roughly `per_board_sd / sqrt(n_private_boards)`, measured concretely by `bootstrap_ci` (see the noise-floor demo in the oracle-checks section above; a real model's OOF interval will typically be narrower than the pure-noise demo since a working model's per-board scores have lower variance than random). Grouped splits are inherently noisier than random splits (V22) -- expect visible rank churn between near-tied experiments; use the paired-fold/noise rule (CLAUDE.md §4A, V4/V16) before keeping any change.
- **Net**: plan for private-LB ≈ test-like OOF minus ~1-2 points of selection optimism, ± a bootstrap CI half-width of the same kind reported above (computed on the real model's OOF, not the noise-floor demo), consistent with the strategist's plan's own estimate (35-50 expected band).

## Reproduce

```
python3 /home/user/Claude-/challenges/assembly_amendments/validate.py
```
Rebuilds the fold assignment, runs the oracle/sanity battery, and rewrites this report, from `dataset/public/{train,train_items,train_targets}.csv` only (SEED=42, N_FOLDS=5, MIN_SHARED_TEXTS=3, MIN_DISTINCTIVE_TEXT_LEN=40).
