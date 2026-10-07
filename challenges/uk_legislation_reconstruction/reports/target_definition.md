# Phase 4 — Exact target semantics

## `amending_ids`
The corpus `unit_id`s of provisions that the publisher's editorial record lists against the queried
section as **textual** effects whose in-force date is on or before the query date.

Consequences established from the training data:
1. **Textual only.** A provision that merely applies, modifies or saves the section (non-textual
   effect) is not a target. This is why the parser's verdict ("does this provision yield a concrete
   textual edit to this section?") is a retrieval feature.
2. **In force, not merely made.** 1,499 of 1,500 training targets have
   `document_date <= query date`, but only 21% of candidates meeting that condition are targets —
   so the date is a weak signal, not a rule. Commencement orders are nearly absent from the corpus
   (6 of 1,067 documents), so timing must be learned, not looked up.
3. **Granularity is the corpus unit**, which may be a single paragraph
   (`Sch. 14 para. 8`) or a whole repeal schedule covering dozens of Acts (`Sch. 17`).
4. A target provision need not change the section visibly in isolation: 121 of 1,500 target
   provisions contain no clause naming the queried section at all. They contribute to the id term
   without contributing an edit.
5. Counts are skewed: median 2, mean 2.93, max 32 target ids per query.

## `text_at_date`
The section's official text on the date, in the corpus rendering: bracketed sub-provision markers
in reading order, no section number, no heading, no editorial notes. Every training query's answer
differs from the enacted text (by construction the date is the midpoint between two consecutive
in-force dates), and the length ratio `text_at_date / enacted` has median 1.09 and mean 1.24 —
**insertions dominate**: across the training set 49,958 tokens are inserted against 14,146 deleted.

## What the two terms reward
- Getting the retrieval right but the editing wrong caps the score at 40.
- Getting the editing right but the retrieval wrong caps it at 60.
- The `sample_submission` baseline (no ids, enacted text unchanged) scores **exactly 0** on both —
  verified against the local metric.
