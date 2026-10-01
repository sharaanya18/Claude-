# Task T1 — Field-capture burst midpoint recommendation (paraphrased description)

Each row is one 8-frame field-photo burst. You get the FIRST photo (L) and LAST photo (R) of the burst, plus the 6 interior frames in shuffled order (candidates c0..c5). Photos are JPEGs shipped in `train_images.zip` / `test_images.zip`; the release applies random crops, flips and photometric changes per image independently.

Goal: submit the 2 interior candidates most likely to be closest in time to the temporal midpoint of L and R, as `ranked_candidates_json` (best first).

Score per row: 1.0 if the gold midpoint candidate is ranked first, 0.2 if second, 0 otherwise; averaged over rows. Random ranking scores 0.2.

Data: ~1,250 training rows with gold midpoint labels; test set a few hundred rows. Bursts continue across rows (one row's R photo can be the next row's L photo). No group column is provided.
Rules: general-purpose pretrained weights (HF/timm) are allowed; no external data; runs on one A10G; ~1 hour budget; solution must train a model in-script.
Submission: `id, ranked_candidates_json`; exactly one row per test id.
