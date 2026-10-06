# Challenge notes: chart_change_marks

## Status
- [ ] contract  - [ ] data audit  - [ ] strategist plan  - [ ] validation  - [ ] baseline  - [ ] experiments
- [ ] review (compliance, runtime, red-team)  - [ ] presubmit  - [ ] submitted  - [ ] closed (lessons written)

## Contract (decision unit, valid answer, metric terms, constraints, bans, compute, runtime)

## Data audit (shapes, groups, duplicates, label distribution, structure, ceiling diagnostics)

## Validation design (mirror of the hidden split, groups, bias direction of the proxy)

## Plan (primary, fallback, rejected options, fixed work plan)

## Experiment log (id, hypothesis, change, CV mean +- std, per-fold, runtime, kept?, notes)
Proxy = mean skill over look-alike cells (held out in CV). All-5-fold proxy (8 cells) / 2-fold proxy (folds 0,3: pas|categorical, lims|datetime, pas|uniqueidentifier). Best threshold per row.
| id | change | proxy | notes |
|---|---|---|---|
| base | 1-D U-Net+BiGRU, 14 ep, hflip/vflip aug, hflip TTA | 8-cell 0.436 @thr .6; 2-fold 0.409 | recall .48 / precision .81; mean bracket credit .91 (prior .83) |
| e24 | 24 epochs | 2-fold 0.426 | |
| e24s | + horizontal stretch aug (0.8-1.25, p .5) | 2-fold 0.435 | kept |
| ens | mean of e24, e24s heat | 2-fold 0.453 | ensembling +0.02 -> 2 members/fold in solution |
| e24n | drop metadata vector (zeros) | 2-fold 0.395 (vs 0.435 with) | metadata helps; kept |
Error analysis: recall by true bracket firm .82 / split .32 / weak .17; 84% of true marks are within 5 px of some peak >= .15. Remaining gap = human disagreement on weak/split marks.
Kaggle: kernels chart-change-marks-run (v1, 1 member/fold, 24 ep) and chart-change-marks-v2 (2 members, 20 ep) pushed as CPU backups.

## Error analysis

## Submission history (sub, based on exp, public LB, credits left, gap, notes)

## Key insights (what was unique, biggest gain, biggest surprise, what to do differently)
