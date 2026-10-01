# Task T5 — Archival letter reference retrieval (paraphrased description)

Each query is a mention, inside the OCR'd text of one 17th-19th-century letter, of a DIFFERENT letter being referred to ("your letter of the 5th", "beantwortet d. 27ten", ...). Fields: document_id, date, sender, recipient, text, mention_start, mention_end. The task: rank the top 10 candidates from the full shared archive (`letters.csv`, thousands of letters) most likely to be the referenced letter. `train.csv`/`validation.csv` give gold letter_ids.

Metric: MRR@10. The domain is hard: dates in German/French/Latin/English/Italian month names, roman numerals, "7bre"-style abbreviations, relative idioms ("huj.", "vorigen Monats", "vor 8 Tagen", weekday names), 2-digit years needing century inference; names with nobility particles, initials and orthographic variants; OCR noise.
Rules: CPU-only, standard libraries, ~1.5 h; no pretrained weights are required; no external data; the model must be trained in-script from the four supplied CSVs.
Submission: query id plus ranked list of 10 letter ids.
