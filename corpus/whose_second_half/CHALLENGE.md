> NOTE: not verbatim platform text. The user supplied only five ranked top solutions (no challenge
> statement). This description is reconstructed from the solutions' own docstrings/comments
> (primarily rank-2's header docstring and rank-3/rank-4's module comments). Treat every claim below
> as a solver's own gloss on the task, not an authoritative contract. Column/field names are as used
> in the solutions' code, not confirmed against a real sample_submission.csv.

# Whose Second Half Is It (reconstructed)

Each listening day ("session") is split into a first half and a second half. `listens.csv` gives,
per session, every play in order (`relative_position`), with `artist`, `release`, `recording`,
`half` (1 or 2), and `seconds` (a timestamp within the day). `continuations.csv` gives, for each
"continuation" token, up to two (artist, release, recording) entries ranked 1-2 — a short
description of how a second half started.

A row of `train.csv` / `test.csv` lists six `prefixes` (six session ids, i.e. six first halves) and
six `candidates` (six continuation tokens, shuffled). The task: for each of the six prefixes, pick
which of the six candidate continuations is that session's own real second-half opening — a
one-to-one assignment (bijection) over 6 items, so the hidden answer is always a permutation.
`train.csv` gives the true assignment as `match_1..match_6` (1-indexed candidate rank per prefix,
each a permutation of 1..6); `test.csv` withholds it and the submission must supply it in the same
shape.

Implied metric (from every solution's own validation code): mean per-prefix accuracy of the
predicted assignment against the true permutation, i.e. fraction of the six slots correctly matched,
averaged over rows (several solutions report a "chance-corrected" version, (acc - 1/6) / (5/6), as a
diagnostic only — not necessarily the platform's own metric).

Compute: CPU-only in at least two of the five solutions (explicit `device_type="cpu"`, 10 threads);
others assume a GPU is available but still run on CPU without hard requiring it. No stated runtime
limit was captured in the reconstructed text; solutions self-impose ~10-45 min fixed-plan budgets.

Unverified: exact column names of `sample_submission.csv`, the official metric formula, the row
count of train/test, any stated runtime/hardware limit, whether "from scratch" or pretrained
embeddings were restricted (all five solutions train every embedding from scratch on the released
listens; none uses an external pretrained model, which may reflect either a real restriction or
just convergent practice).
