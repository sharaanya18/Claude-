# Phase 2–3 — Dataset forensic audit and split forensics

## Corpus (`corpus.csv`)
| property | value |
|---|---|
| rows / unique `unit_id` | 145,091 / 145,091 (no duplicates) |
| documents (by citation) | 1,067 — 509 Acts, 558 statutory instruments |
| `document_date` range | 1989-11-16 → 2026-07-28 |
| `text` length | median 497, p90 2,291, p99 7,061, max 120,017 chars |
| label kinds | `Sch. …` 87,775 · `s. …` 43,332 · `reg. …` 7,881 · `art. …` 4,820 · bare `schedule` 1,229 · other 54 |

Most provisions amend something else or nothing at all; the long tail of very large provisions is
repeal / revocation **tables** (one unit listing dozens of Acts and their repealed words), which
need segmenting before they can be parsed.

### Commencement information is essentially absent
Only 6 of 1,067 documents are commencement or appointed-day instruments. A targeted search for
commencement provisions naming a specific amending Act (e.g. `2007 c. 27`, whose Schedule 8 supplies
many training targets) returns nothing usable. **The in-force date of an amendment is therefore not
recoverable from the corpus** and has to be inferred statistically.

## Queries
| | train | test |
|---|---|---|
| rows | 512 | 1,177 |
| Acts | 14 | 26 |
| Act overlap | — | **none** (verified: empty intersection of `act_citation`) |
| `section_label` kinds | 100% `s. …` | 100% `s. …` |
| date range | 2002-12-31 → 2031-02-28 | 2002-08-28 → 2029-03-30 |

Train is heavily concentrated: `2002 c. 29` (Proceeds of Crime Act 2002) alone supplies 179 of 512
queries, `1991 c. 56` 76, `1996 c. 18` 53. Test is flatter (largest Act 167 of 1,177).

## Targets (`train_targets.csv`)
- `amending_ids` per query: median 2, mean 2.93, max 32; distribution
  1 → 216, 2 → 136, 3 → 63, 4 → 31, 5 → 18, ≥6 → 48 queries.
- All 1,500 target ids exist in the corpus (1,180 distinct provisions).
- `text_at_date != enacted_text` for **100%** of queries, as the date construction implies.
- Length ratio `at_date / enacted`: min 0.35, median 1.09, mean 1.24, max 5.22.
- Edit profile across the training set: 14,146 enacted tokens deleted, 49,958 tokens inserted,
  mean 7.1 change blocks per query (median 4) — **insertion-dominated**.

## How a target provision points at its section
Measured over all 1,500 (query, target) pairs:
| evidence | share |
|---|---|
| target section number appears in the provision's own text | **95.3%** |
| `document_date <= query date` | **99.9%** (1,499/1,500) |
| Act identified by full title in text or context | 44% |
| Act identified by citation `(c. N)` | 28% |
| Act **not named at all** by title or citation | 26% |

The 26% are the decisive case: provisions reading "Section 29 (interpretation) is amended as
follows" whose Act comes from the opening paragraph of the enclosing schedule, or from an
abbreviation ("SSCBA 1992"), or from "the 2002 Act". Resolving those three mechanisms is what took
candidate recall from 92.9% to **97.6%** at 18.5 candidates per query.

## Leakage risks and how they are handled
| risk | handling |
|---|---|
| Random row-level CV would mix Acts and flatter wildly | **All CV is grouped by Act** (5 folds, balanced by query count), mirroring the real split |
| An instrument holds provisions for both train and test Acts | Permitted by the description and kept — but no target label ever crosses: labels come only from `train_targets.csv`, and corpus indexing uses no labels at all |
| Fitting anything on test rows | Both boosters, the IDF table and the two thresholds are fitted on the 512 training queries only; each test row is scored from its own candidates |
| Act-specific memorisation (section numbers, document titles) | Features are mechanism-level (reference frames, quoted-operand matching, date gaps, parser verdicts); no Act identity, document title or `unit_id` is ever a feature |
| `unit_id` ordering | Confirmed random and carries no information; never used as a feature |

## Recommended validation (adopted)
5 folds grouped by `act_citation`, Acts assigned largest-first to the lightest fold → query counts
[179, 79, 87, 85, 82]. The largest Act (179 queries) dominates one fold, so fold-to-fold spread is
wide: per-fold id F1 [0.847, 0.794, 0.791, 0.850, 0.749], std 0.038 ⇒ **standard error ≈ 0.017**.
Changes smaller than ~2 points of a term are not distinguishable from noise and were only kept when
they were also mechanically motivated.
