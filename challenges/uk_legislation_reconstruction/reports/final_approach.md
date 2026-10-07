# Final approach — Historical UK Legislation Reconstruction

## 1. Problem interpretation
Two coupled problems over one corpus of **as-enacted** provisions:

- **Retrieval + temporal filter.** Which corpus provisions made a *textual* change to this section,
  and were in force by this date?
- **Program synthesis + execution.** What did each of them instruct, and what does the section read
  like once those instructions are applied in date order?

They are coupled in both directions. The chosen provisions determine the edits; and whether a
candidate yields a *concrete, applicable* edit to this very section is one of the strongest
retrieval signals there is. The pipeline exploits that: the amendment parser runs during feature
extraction, and its verdict is fed to the reranker.

## 2. Candidate retrieval — why it works
Amending provisions are written in a stereotyped referential style, so the recall problem is
reference *resolution*, not similarity search. For each of the 145,091 provisions the script
extracts every Act reference with its character position and every section reference with the 26
characters of grammatical frame before it, then builds an inverted index on
`(act citation, section number)`.

Act references are resolved five ways, because 26% of true amending provisions never name their Act:
1. full title (`Police Reform Act 2002`) against a title→citation map built from the corpus;
2. citation (`2002 c. 30`, or a trailing `(c. 30)` after a title);
3. abbreviations mined from the corpus's own interpretation sections
   (`"SSCBA 1992" means the Social Security Contributions and Benefits Act 1992`);
4. `the 2002 Act`, resolved per document to the most-referenced Act of that year;
5. **propagation** — a schedule's opening paragraph ("The Police Reform Act 2002 is amended as
   follows.") governs the paragraphs that follow it, so provisions are ordered by parsed label
   within their schedule and the declaration is carried forward while the context path still shares
   its trunk.

Section references are paired with the nearest Act references on either side (which is what makes
huge repeal tables usable: a section only matches the Act whose table row it sits in) plus the
unit's dominant Acts, the propagated Act and the context Acts.

**Result: 97.6% recall of the true amending provisions at 18.5 candidates per query**, up from
92.9% before mechanisms 3–5.

## 3. Reranker (MODEL 1)
LightGBM binary classifier, 61 features, two seeds averaged, 400 rounds, trained in-script on the
512 training queries. Feature groups:

- **Provenance of the act match** — title in text / citation in text / in context / propagated /
  dominant / self — plus the log distance between the Act mention and the section mention and the
  number of *other* Act mentions between them (the dilution signal for tables).
- **Frame of the section mention** — "In section N…", "Section N … is amended", "omit section N",
  "After section N insert", "section N of the <Act>", and whether an amendment verb occurs within
  160 characters.
- **Operand grounding** — every quoted string in the provision is checked against the enacted
  section: how many are found, the longest found, and specifically whether the `old` side of a
  `for "X" substitute "Y"` occurs. A substitution whose target words are absent from the section is
  usually about a different provision.
- **Anchor existence** — whether the sub-provision markers the provision addresses exist.
- **Temporal** — `query date − document date`, sign, document year, instrument type.
- **Parser verdict** — number of edits parsed for *this* section, number that actually apply, their
  total size, and their operation mix.
- **Within-query normalisations** — rank of the date gap, of lexical overlap and of the parser
  verdict among that query's own candidates (computed from that query's candidates only).

Selection: every candidate with probability ≥ 0.25, and if none clears it the single best candidate
(an empty list scores 0, so claiming nothing is never right). Act-grouped OOF **id F1 = 0.810**.

## 4. Temporal reasoning — and its honest ceiling
The description says the corpus holds the instruments that brought the amendments into force. In
practice only **6 of 1,067 documents** are commencement or appointed-day instruments, and searching
for commencement provisions naming a given amending Act yields nothing usable. I built the
alternative — mining each amending document's *own* commencement section for explicit dates, "end of
the period of two months beginning with", and "such day as … may appoint", and matching those dates
to the candidate's schedule or section — and it produced **no gain** (0.8140 → 0.8121, inside noise).
It was removed.

So commencement is learned, not looked up: the date gap is the single strongest feature in the model,
and the model learns the empirical delay distribution for Act schedules versus statutory instruments.
This is the limiting factor on the id term. Among candidates that are made before the date *and* for
which the parser produces an applicable edit, only 57% are true targets; the date gaps of true and
false members of that group overlap heavily (median 2,033 vs 2,597 days). Without commencement
records that residual is not separable, and I am not willing to invent a source for it.

## 5. Amendment parser — supported operations
Each selected provision is split into instruction clauses, and each clause into a scoped edit.

*Clause splitting* is the subtle part, because inserted content looks exactly like instruction
structure. A marker starts a clause only if it (a) can open a sub-provision — preceded by
`. ; : , -`, a quote, a closing marker with a space, or a `; and` style continuation — (b) continues
a strict sequence `(1)(2)(3)…` or `(a)(b)(c)…`, and (c) is followed by instruction language
(`in`, `for`, `after`, `omit`, `insert`, `section`, `the words`, …). The same three-part test,
applied to the section text, separates real sub-provision markers from cross-references such as
"subsection (7)" and "section 130(1)(a)" — the sequence rule does the work there, since a reference
points backwards.

| operation | source form |
|---|---|
| `sub` | `for "X" substitute "Y"` / `there is substituted` (with every-place detection) |
| `after` / `before` | `after "X" insert "Y"` |
| `omit` | `omit "X"` / `the words "X" are repealed` |
| `omitspan` / `subspan` | `the words from "A" to "B"` (+ `to the end`) |
| `subprov` / `insprov` / `omitprov` | `for subsection (5) substitute — …` / `after paragraph (f) insert — (fa) …` / `omit subsection (2)` |
| `subdefn` / `insdefn` / `omitdefn` | definition-scoped forms (`after the definition of "X" insert …`) |
| `approp` | `at the appropriate place insert …` — placed in the list's own numeric order |
| `endins` | `at the end insert …` |
| table forms | repeal / revocation tables: `In section 6(3)(a), the words "…"`, `In section 14, subsections (7) and (7A)`, `The words "…" in — section 1AA(9); section 8(8)(b); …` |

Scope is a chain of markers (`['1','ca']` for section 27(1)(ca)) resolved against the *running*
text, backing off to the deepest resolvable prefix. Anchors are matched with word boundaries, so
`for "it" substitute "the transfer of value"` cannot fire inside "condition".

Three fixes mattered more than any new pattern: punctuation-aware joining (no space before a leading
comma), keeping the inserted text's own terminator while dropping the amending sentence's duplicate
full stop, and refusing any substitution whose replacement parsed as empty — that last one was
silently deleting whole subsections.

## 6. Edit gate (MODEL 2) and text reconstruction
Walking each training query's plan once, every parsed edit gets a label: the change in the exact
challenge metric when that edit was applied at its turn. Plans are built from **out-of-fold**
retrieval scores, so the gate is trained on the same imperfect provision sets it meets at inference.
A LightGBM classifier on "did this edit improve the metric" uses the operation type, whether the
anchor resolved in scope or only globally, the number of occurrences, chain depth and resolution,
inserted/deleted fraction, whether the inserted content starts with a marker the section does not
already have, the reranker's confidence in the provision, and whether the clause addresses a
different section. Edits scoring ≥ 0.20 are applied in date order.

35% of parsed edits improve the metric; the gate lifts the text term from 46.5 to 47.5 and the total
from 60.4 to 61.0. An oracle gate would reach text 56.1, so roughly 5 further text points sit in
better *ranking of edits* — the clearest remaining lever after parser coverage.

## 7. Validation
Five folds grouped by `act_citation`, Acts assigned largest-first to the lightest fold; mirrors the
hidden split, where no test query concerns a training Act.

| | id F1 | text edit F1 | total |
|---|---|---|---|
| sample_submission | 0.000 | 0.000 | **0.00** |
| oracle ids + this reconstruction | 1.000 | 0.497 | 69.80 |
| **this solution (Act-grouped OOF)** | **0.813** | **0.475** | **61.02** |

Per-fold id F1: 0.847 / 0.794 / 0.791 / 0.850 / 0.749 — std 0.038, standard error ≈ 0.017. The
largest Act holds 179 of 512 training queries, so fold spread is wide and only changes above roughly
2 points of a term are distinguishable from noise. Both selection thresholds sit on flat plateaus
(0.20–0.30) rather than on spikes, which is what I want for a split I cannot see.

Reported under the stricter reading of the insertion edit (position *and* offset-in-block must
match), the text term is 46.3 instead of 47.5 and the total 60.3 instead of 61.0; every design
decision above holds under both readings.

## 8. Error analysis — what still fails
Of 512 training queries scored with oracle ids, 136 score below 0.08 on the text term:

| failure | count | comment |
|---|---|---|
| edits produced at the wrong position or with wrong content | 64 | scope chain resolves to the wrong sub-provision, or the inserted content's rendering differs from the publisher's |
| instructions parsed but none applied | 31 | the anchor only exists after an amendment we never retrieved — a genuine dependency gap, not an ordering bug (a retry pass recovers nothing) |
| no instruction parsed | 29 | unmatched drafting forms, mostly inside large consequential-amendment tables |
| deleted more than half the section | 12 | over-firing omissions; largely addressed by the empty-replacement guard |

Across all queries, 337 of 1,500 true amending provisions yield no instruction at all; 121 of them
contain no clause naming the queried section, so they can only ever contribute to the id term.
Edit-level precision/recall is 0.40 / 0.37 — recall is the binding constraint.

## 9. Runtime and resources
Single full run on the development container (CPU only): **202 s** end to end —
corpus index 74 s, training features 26 s, Act-grouped OOF + both models 28 s, test scoring 62 s,
prediction and validation 10 s. Peak memory is dominated by the normalised corpus text
(~1.5 GB). No GPU, no network, no cached artefacts. Against a 50-minute target that is ~93%
headroom, which is why both boosters average two seeds.

## 10. Rule compliance
- **No external statute book.** The only inputs are the five files in `<public_dir>`. No HTTP, no
  file reads outside `public_dir`, no `pip install`, nothing cached between runs.
- **No consolidated or point-in-time text, and no model trained on it.** Nothing is downloaded:
  both models are gradient-boosted trees fitted in this process on the 512 training queries. There
  is no pretrained checkpoint, tokenizer or embedding anywhere in the pipeline.
- **No editorial record of effects, no external commencement data.** The commencement evidence I
  tried came from the supplied corpus itself, and was dropped for lack of signal.
- **Test rows are used one at a time.** Both boosters, the IDF table and both thresholds are fitted
  on training queries and the supplied corpus only. Within-query rank features use that query's own
  candidates and nothing else; no statistic is pooled across test rows.
- **The models do the learning.** Remove either booster and the solution collapses: without the
  reranker all 18.5 candidates per query are claimed (id F1 falls to ~0.35 at that operating point)
  and the text is corrupted by every spurious provision's edits; without the gate the text term
  falls by a point and the pipeline has no way to choose between competing parses. The parser is
  feature extraction and an execution engine, not the decision-maker.
- **Determinism.** Fixed folds, rounds, seeds, thresholds and thread count; `deterministic=True`
  and `force_row_wise=True` on every booster; fold assignment is a deterministic function of the
  data. Wall-clock time appears only inside `log()`. No hardware, library or environment branch
  exists anywhere in the file.
- **Source** is one plain UTF-8 `solution.py` of ~70 KB (limit 512 KB), no encoded blobs, no
  generated code, no `exec`/`eval`.

## Note on Phase 7 of the brief
The brief asks for a forensic analysis of a "supplied private-LB #1 reference solution". **No such
reference was supplied** with this challenge — the uploads were the task brief and the Eris solver
guidebook, and the dataset archive. I have not guessed at one, and
`reports/private_lb_reference_analysis.md` records that rather than inventing content.
