Cross-Publisher Kazakh News Headline Generation
Overview
You're given the body text of real, published news articles written in Arabic-script Kazakh — the written form used in Xinjiang, China, distinct from the Cyrillic Kazakh used in Kazakhstan. Your task is to generate the article's headline. The twist: every test article comes from a news publisher whose articles never appear in training at all. You have to generalize your notion of "what a headline looks like" to a publisher's house style you've never seen, not just get locally fluent on the training distribution.

This is a cross-publisher generalization task — not plain in-distribution headline generation, translation, extractive summarization, or script transliteration. Input and output are both already Arabic-script Kazakh; a model that only learns one training publisher's house style will measurably underperform on the two held-out test publishers (see Evaluation).

Task
For every row in test.csv, generate a single headline string for the given text. Output must be Arabic-script Kazakh (matching the input script) — do not transliterate, translate, or romanize. A solver should never need to guess the input/output format from this description; the files below are the complete, literal contract.

Dataset
dataset/public/
├── train.csv             (50,660 rows)
├── test.csv               (6,420 rows)
└── sample_submission.csv  (6,420 rows)

| File | Rows | Columns | Purpose |
|---|---|---|---|
| train.csv | 50,660 | id, outlet, text, title | Labeled training articles, from 5 publishers. |
| test.csv | 6,420 | id, outlet, text | Evaluation articles, from 2 publishers absent from train.csv. |
| sample_submission.csv | 6,420 | id, title | Deterministic valid-format example, not a baseline. |

| Column | Type | Meaning |
|---|---|---|
| id | string | Opaque identifier, freshly assigned per split; carries no source information. |
| outlet | string | Publisher code. 5 distinct codes in train.csv, 2 distinct codes in test.csv, zero overlap between the two sets. |
| text | string | Full article body, Arabic-script Kazakh. |
| title | string | Real published headline. Present only in train.csv (your training target); 

Split and Integrity
Training and test articles come from disjoint real news publishers — 5 publishers in train.csv, 2 different publishers in test.csv, chosen for being the most stylistically distinct from the training pool so the generalization gap is real rather than nominal. The outlet column is an opaque code (outlet_01, outlet_02, ...), never the real publisher domain, and no code appears on both sides of the split.

Because several publishers run the same national wire stories, the same underlying story can appear more than once, lightly reworded, across different publishers. Visibility (train vs. test) is assigned per story family, not per row: every article is first clustered with every other article sharing an exact or near-duplicate title (shingle-based Jaccard similarity >= 0.6), transitively — so a chain of re-wordings of the same story is caught as one family even if the two most different-looking copies wouldn't individually cross the similarity threshold. Any family with at least one held-out-publisher article has every one of its non-held-out-publisher copies removed from training (693 rows dropped in the final split). Verified directly against the prepared files, not assumed: 0 exact-duplicate title groups and 0 near-duplicate title pairs cross the train/test boundary. Original per-article identifiers and source URLs are not present anywhere in the public or private files; every id is a freshly assigned sequential value with no relationship to the source.

Evaluation
Score = mean over test rows of a per-row composite:

ROUGE-L F1 (word-level longest-common-subsequence overlap between your generated title and the real title) — always computed.
Number precision: for rows where the real title contains at least one digit sequence (a date, statistic, anniversary count, etc.), the fraction of the digit sequences you output that actually appear in the real title. Rows whose real title has no numbers skip this component.
Composite = 0.8 * ROUGE-L F1 + 0.2 * number precision when numbers apply, else ROUGE-L F1 alone.
Direction: maximize. Range: [0, 1]. This metric is used because plain lexical overlap alone lets a model "sound right" while inventing a different number/date than the source supports — a common, hard-to-notice failure in headline generation specifically, so it's scored separately.

Submission Format
CSV with exactly these columns:

| id | title |
|---|---|
| test_00000 | \<your generated headline for this row, in Arabic-script Kazakh\> |
| test_00001 | \<your generated headline for this row, in Arabic-script Kazakh\> |

(Illustrative only — test_00000 and test_00001 are real ids from test.csv, shown to demonstrate the id format; the title values are placeholders, not real content.)

id: string, must exactly match every id in test.csv (6,420 rows, every id required — missing ids fail the submission outright).
title: string, your generated headline. Non-empty text expected; a blank or missing value scores 0 for that row rather than failing the whole submission.
Duplicate id rows: only the first occurrence is used.
Extra rows with ids not in test.csv are ignored, not penalized or rewarded.
Row order does not matter — scoring joins on id.
What Not To Use
Live internet lookup to retrieve the original article or headline. These are real, previously-published articles; searching for the source defeats the point of the task — the scored capability is generation from the given text, not retrieval of the real headline from the web.
An off-the-shelf multilingual tokenizer (e.g. mBERT, XLM-R) as an unmodified drop-in, expecting strong results out of the box. Measured directly against real sample text from this dataset: mBERT produces a 16.5% unknown-token rate on Arabic-script Kazakh vs. 0% on Cyrillic Kazakh and Modern Standard Arabic controls; XLM-R fragments this text into ~3.1 tokens/word vs. ~1.3-1.7 tokens/word on those same controls. Either can still be used as a starting point, but budget for the extra difficulty rather than assuming out-of-the-box quality.
An Arabic-script-specific model such as AraT5 on the assumption that a shared script implies transfer. AraT5 is trained for the Arabic language specifically, not the Arabic script in general, so it has no meaningful exposure to Kazakh vocabulary or grammar despite the shared alphabet.
Everything needed to solve this well is in train.csv's article bodies and headlines: a model that learns the general shape of Kazakh headline writing — not one training publisher's specific house style, and not a retrieved copy of the real answer — is what this task measures. Each restriction above bypasses that by fetching the real answer directly or by leaning on a checkpoint's out-of-the-box assumptions instead of engaging with the actual difficulty of the task.


