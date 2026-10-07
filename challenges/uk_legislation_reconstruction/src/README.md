# Development modules

`solution.py` at the challenge root is the single self-contained deliverable. It is **generated**
from these modules by `python3 src/mksol.py`, so edit the modules and regenerate rather than editing
`solution.py` by hand.

| file | role |
|---|---|
| `lib.py` | text normalisation, label parsing, Act/section reference extraction |
| `metric.py` | local re-implementation of the challenge metric (both readings of the insertion edit) |
| `apply.py` | amendment-instruction parser and the text applier |
| `index.py` | corpus parsing, Act resolution (abbreviations, "the YYYY Act", schedule propagation) and the `(act, section)` inverted index |
| `feats.py` | reranker features, including the parser-derived ones |
| `gate.py` | edit-gate features and the metric-gain labelling walk |
| `pipe.py` | pipeline helpers shared by the CV drivers and `solution.py` |
| `main_block.py` | the `main()` of `solution.py` plus the submission validator |
| `mksol.py` | assembles `solution.py` from the above |
| `cvgate.py` | end-to-end Act-grouped CV (the headline 61.02 number) |
| `nested.py` | fold-disjoint (nested) evaluation of the two selection thresholds |
| `diagnostics/` | the one-off audit and error-analysis scripts behind the numbers quoted in `reports/` |
