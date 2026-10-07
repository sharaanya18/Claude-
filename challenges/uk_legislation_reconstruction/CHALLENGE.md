# Historical UK Legislation Reconstruction (Project Eris / Shipd)

## Task
For each query — a UK Act (title + citation), one of its sections **as enacted**, and a date —
predict:

| column | meaning |
|---|---|
| `amending_ids` | JSON list of corpus `unit_id`s whose **textual** amendments to that section were **in force on or before** the date |
| `text_at_date` | the section's text as it stood on that date, rendered like the corpus (bracketed sub-provision numbers in reading order, no section number/heading/notes) |

## Data (`public/`)
| file | rows | notes |
|---|---|---|
| `corpus.csv` | 145,091 provisions / 1,067 documents | `unit_id, document_title, citation, document_date, label, context, text` — **as enacted or made**, never consolidated |
| `train.csv` | 512 queries / 14 Acts | `item_id, act_title, act_citation, section_label, date, enacted_text` |
| `test.csv` | 1,177 queries / 26 Acts | disjoint Acts; test split further into two slices by Act |
| `train_targets.csv` | 512 | the submission layout |
| `sample_submission.csv` | 1,177 | placeholder claiming nothing (scores 0 on both terms) |

## Metric
`score = 100 × mean_q [ 0.40 × setF1(amending_ids) + 0.60 × editF1(text_at_date) ]`

Edit F1: the submitted and the true text are each diffed against the **enacted** text; an edit is an
enacted token deleted or a new token inserted at a position of the enacted text; the term is the F1
between the two edit sets. An empty id list scores 0; copying the enacted text scores 0; a text
keeping fewer than half as many enacted tokens as the true text scores 0 on the text term.

## Explicit restrictions (from the description)
- No external copy of the statute book, of consolidated or point-in-time versions, or of the
  editorial record of effects, **and no model trained on them**.
- Work from the supplied corpus and training queries only.

## Compute / runtime
The description states no hardware or runtime limit, so the CLAUDE.md defaults apply: target
≤ 50 min, plan for ≤ 1 h. The solution is pure CPU (numpy / pandas / LightGBM); it needs no GPU.

## Key structural facts established from the data
- An amending provision identifies its target Act by full title, by citation, by an abbreviation
  defined elsewhere in its document ("SSCBA 1992"), by "the 2002 Act", or **not at all** — in which
  case the Act comes from the opening paragraph of the enclosing schedule
  ("The Police Reform Act 2002 is amended as follows.").
- 95% of true amending provisions mention the target section number in their own text.
- 1,499 / 1,500 true amending provisions have `document_date <= query date`; but only 21% of
  candidates satisfying that are true, so the date is a weak filter, not a rule.
- **Commencement orders are almost absent from the corpus** (6 of 1,067 documents), so the in-force
  decision has to be learned from the date gap and provision characteristics, not read off a record.
