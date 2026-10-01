# Plan U25 - Scholarly note anchor restoration

Status: plan only. No dataset files were available, so every statement under "Data findings" is an UNVERIFIED hypothesis or a diagnostic to run. Numbers quoted from the description are marked (desc). Score ranges are estimates, not promises.

What the description file is missing (read in full, twice):
- No title and no domain label (NLP / fine-tuning / from-scratch is not stated). Assumption: ordinary NLP ranking task, pretrained general-purpose weights allowed per CLAUDE.md section 2.2 (the description is silent on them; see Open questions).
- The file is truncated mid-word in the last "What Not To Do" bullet ("...held-out language understandin"). More bullets may have followed. Assumption: any further bans are the standard CLAUDE.md bans (no external data, no test fitting, no LLM-generated content, no hard-coded tuned constants), which I already apply.
- No allowed-library list, no public/private split description, no per-step limits. Assumption: Kaggle image stack from CLAUDE.md section 8.
- Compute header IS present and overrides CLAUDE.md's A10G assumption: **CPU only, 90 min end to end, at most 10 CPU cores, 62 GB RAM**. There is no GPU.

---

## Contract & decision unit

- Input per example: `id`, `chapter_id` (opaque group), `note` (30-300 words), `segments_json` (JSON array of m segments, 4 <= m <= 20, reading order; joining with single spaces gives the paragraph, 80-1,000 words). Train 1,154 rows from 53 chapters (about 21.8 rows/chapter); test 288 rows from 18 other chapters (16.0 rows/chapter) (desc).
- One valid answer: one integer `anchor_index` in [0, m-1] for each test `id` (index of the segment immediately BEFORE the note marker). Submission: UTF-8 CSV, columns exactly `id,anchor_index`, 288 rows, each test id exactly once, row order irrelevant (desc).
- Invalid vs low-scoring: blank, non-numeric, non-finite, fractional, or out-of-range index scores 0 for that row; wrong columns / missing, extra, duplicate or unknown ids is a submission error (whole submission dead). A wrong but valid index is just a miss.
- Metric: accuracy = (# exact anchor matches) / 288. Equal weight per example, no partial credit for adjacent positions, no hierarchical averaging. Because it is 0-1 loss on a single categorical decision, the Bayes-optimal decode is argmax of the posterior over the m candidates; there is no composite-metric expected-utility trick to exploit.
- Independent unit: the chapter (53 train, 18 test). Rows in the same chapter share author voice, topic, citation style and probably note conventions. Effective sample size is closer to 53 clusters than to 1,154 rows.
- Pipeline stages, diagnosed separately:
  1. Candidate coverage: trivially 100 % (all m indices are the candidate set, m <= 20). No retrieval stage is needed.
  2. Scoring: a learned score s_k for each candidate k given (note, paragraph).
  3. Decode: argmax_k of a softmax over the row's own m candidates. Output is automatically valid (0..m-1).
- Baselines to compute on train: sample_submission rule (anchor = m-1), uniform 1/m, majority relative position, oracle-of-two-simple-rules (see Data findings).

## Compliance regime

Domain: NLP / structured ranking over an enumerable legal set (placement), CPU only. Playbook CLAUDE.md 6.2 plus the ranking discipline of 6.4 (trained scorer is mandatory; hand features alone are not enough).

Explicit bans in the description (hard constraints) and how the plan honours each:
1. "Do not access the raw upload, private answers, or another solution's predictions." Never touched; plan uses only `public/`.
2. "Do not use external task labels, externally retrieved passages, or source lookup." No retrieval of the original works, no web search for the quoted text, no gazetteers of known scholarly works. Consequence: no generative LLM scoring either (a large LM that memorised the source is functionally a source lookup, and it is infeasible on CPU anyway). Frozen general-purpose sentence encoders are not task labels or retrieved passages; this is my reading (see Open questions).
3. "Train task-specific parameters on the supplied training data." Satisfied by trained listwise scorers (conditional-logit head, LightGBM ranker, low-rank interaction head) whose task-specific parameters are fit only on train.
4. "Do not split candidate positions from the same paragraph across fitting and validation. Keep whole chapters together." All CV is GroupKFold on `chapter_id`; a paragraph's m candidates are always one unit (the softmax is over the whole row).
5. "Do not report training accuracy as evidence." Only grouped OOF accuracy is reported.

Where the description is silent, and what I assume:
- Pretrained weights/internet: silent. CLAUDE.md allows HF/timm backbone downloads only; I use HF `from_pretrained` for small encoders with pinned revisions, nothing else from the network. Silence is a risk (the grading container may or may not have HF access). Fallback design below needs no weights at all.
- Time: 90 min CPU stated. CLAUDE.md asks for headroom >= 30 %, so target <= 55 min wall (about 61 % of 90).
- Hardware switches: CLAUDE.md forbids `torch.cuda.is_available()` branches; here the device is the constant `"cpu"`. Thread counts are fixed constants (8 for torch/BLAS/LightGBM, below the 10-core cap).

CLAUDE.md bans applied to this plan (all hard): no wall-clock branching; no test-set statistics of any kind (IDF, vocab, scaler, PCA, clustering fit on train only); no pseudo-labels; no synthetic data (so no "move the marker in a real paragraph to fabricate new examples", no re-attaching a note to a different paragraph, no generated notes); no id / row-order / chapter_id features; no cross-row use of test rows (see Structural signals: excluded signals).

Strip-the-ML test: remove every trained component. What is left is a lexical argmax or the "end of paragraph" rule. Both are expected (unverified) to be well below the trained model; this gap must be measured and reported (zero-shot cosine argmax, BM25 argmax, position-only prior vs trained model under the same folds). If the trained model does not beat the best untrained rule by more than CV noise, the solution is too rule-based and must be reworked before submission.

## Data findings

All items below are UNVERIFIED hypotheses plus the exact diagnostics to run on train. Nothing was measured.

Diagnostics to run on train (scratch notebook, not in solution.py):
1. Shapes, dtypes, missing values, empty strings, duplicate ids; JSON parse of every `segments_json`; assert 4 <= m <= 20, all segments non-empty, " ".join(segments) word count within 80-1,000.
2. m distribution (mean, quantiles, histogram); words per segment (quantiles, max; any segment > 256 tokens?); note length in words and tokens; paragraph tokens (how many exceed 512 tokens).
3. Target: histogram of `anchor_index`; histogram of relative position r = anchor/(m-1); share with anchor == m-1 (this is exactly the accuracy of the supplied sample_submission baseline); share anchor == 0; accuracy of best fixed relative-position rule per m (a position-only learned prior, evaluated under grouped CV); uniform-chance accuracy mean(1/m).
4. Chapter structure: rows per chapter (min/median/max; expect 5-60); per-chapter anchor-position distribution and per-chapter last-position share (does note convention vary by chapter? If chapter-level variance of the last-position share is much larger than binomial noise, position priors are chapter-specific and a position-only model will transfer poorly).
5. Duplicates: identical `segments_json` across rows (several notes on one paragraph?); identical or near-identical notes across rows; same paragraph appearing in two chapters. Build groups by union-find over exact/near-duplicate paragraphs and notes in addition to `chapter_id`, and compare to `chapter_id` groups.
6. Possible book-level clusters: cluster the 53 train chapters by chapter-level TF-IDF (train only). Test chapters may come from the same books as some train chapters (desc says only the chapter groups are disjoint), so this only informs a stress validation, not the primary one.
7. Anchor-segment forensics: last character class of the anchor segment (period, closing quote, closing bracket, semicolon, other) vs a random segment; does segment anchor+1 start with a lowercase letter (continuation) or a connective ("However", "Thus", "But")? Fraction of anchors that are a "complete sentence" ending vs mid-sentence breaks (description says segments are usually sentences but need not be).
8. Note-type forensics: fraction of notes containing quoted text (straight/curly quotes); fraction whose quote is found verbatim (normalised) in some segment, and where that segment sits relative to the anchor (equal, anchor-1, anchor-2, later); bibliographic markers (years, "p.", "pp.", "cf.", "ibid.", "see", "n."); capitalised names in the note and their first/last occurrence in the paragraph relative to the anchor; leading-headword pattern (note begins with a quoted/short term then gloss) and where the term occurs.
9. Lexical oracle ladder (diagnoses ranking quality given reachability): accuracy of argmax over segments of (a) BM25/TF-IDF overlap, (b) rare-token overlap, (c) quotation containment where present; plus "anchor within +-1 of argmax" and "oracle choice between argmax and last position" as a crude ceiling of simple rules.
10. Information ceiling: examples where two or more neighbouring candidates are equally plausible. Proxy: for rows with a verbatim quote match, what share still has the anchor not equal to the matched segment (the marker follows the discussed span, "more specific than the editorial range"). Expect a nonzero irreducible band; the description itself says the exact position "can sometimes be ambiguous".
11. Language/script check (English? Latin/Greek quotations? transliteration? OCR noise?): decides the encoder family (English vs multilingual) and the lexical tokenisation.
12. Leakage audit: any column known only post-outcome? Only `id`, `chapter_id`, `note`, `segments_json` exist. `id` ordering inside chapter might encode footnote order; verify but do NOT use (cross-row/id feature banned by plan).

Hypotheses (unverified):
- H1: anchors skew late in the paragraph (footnote marker after the concluding claim) but with large spread; last-position baseline is meaningfully above 1/m, possibly 15-30 %.
- H2: the note's most informative content is a verbatim quotation, a name, a date/number, or a headword, shared with exactly one or two segments; the anchor is that segment or the end of the contiguous run it opens.
- H3: segment k vs k+1 contrast matters: the marker follows segment k when k carries the content and k+1 changes topic or starts a new claim.
- H4: attachment prior depends on the segment alone (claim-bearing sentences with names/figures attract notes) independent of the note.
- H5: within a chapter, conventions are stable; across chapters they differ, so position priors need shrinkage and grouped CV matters.

## Validation design

- Reproduce the split: test is 18 whole other chapters (desc). Mirror it with chapter-disjoint folds.
- Primary: repeated grouped K-fold on the 53 chapters, 5 folds x 3 repeats (different seeded assignments of chapters to folds, balanced on total rows per fold and, if a dispersion diagnostic shows it helps, on last-position share). Each fold is about 10-11 chapters and about 230 rows, mimicking the 18-chapter test. Report per-fold accuracy, mean +- std over folds, and the repeat-to-repeat spread. Never put two rows of one chapter on both sides.
- Frozen encoder embeddings and label-free lexical statistics: embeddings depend on no labels, so computing them once is leak-free. IDF / vocabulary / any fitted lexical statistic is recomputed per training fold from that fold's training paragraphs only, and in the final run from all train paragraphs only.
- Noise size: with n rows and accuracy p near 0.45, the binomial SE over 1,154 rows is about 1.5 pts; with chapter clustering the real SE is perhaps 2-3 pts. A single fold (about 230 rows) has SE of about 3.3 pts. The test has 288 rows, so the private LB SE alone is about 2.9 pts. Rule: accept a change only if it wins paired on the same folds and same repeats, by more than roughly 1 SE of the paired fold-difference (and does not lose in most folds). Gains below about 1.5 pts are treated as noise and rejected in favour of the simpler model.
- Nested selection: every post-hoc choice (L2 strength, LightGBM rounds, ensemble membership) has its own held-out check: select on an inner grouped split inside each outer-training part, evaluate on the outer fold. Cheap here because heads train in seconds once embeddings are cached.
- Sanity holdout: 8 chapters (about 170 rows) are set aside before any HPO or blend selection, scored once at the end of experimentation. Because 53 chapters is small, this is used only as a final overfit-to-CV check, then folded back into training for the shipped refit.
- Stress validation (secondary, pessimistic by design): book-cluster-disjoint folds, using train-only chapter clusters from diagnostic 6. Used only to see how fragile the model is if test chapters come from unseen books.
- Metric implementation and unit tests: accuracy from exact index equality; tests on perfect predictions (1.0), all-last (equals diagnostic 3 share), constant 0, uniform-random expectation (mean 1/m), and reversed positions (m-1-anchor). The submission validator (CLAUDE.md section 5) plus asserts: ids equal sample ids, `anchor_index` integer dtype, 0 <= index < m per row, 288 rows, no NaN.
- Direction of bias of each proxy:
  - Chapter-grouped CV: slightly PESSIMISTIC from training-set size (80 % of 53 chapters is about 42 chapters vs 53 in the final refit; gain from refit estimated +0 to +1.5 pts), slightly OPTIMISTIC from selection on the same OOF (mitigated by nesting and the 8-chapter sanity set; net about 0 to -1 pt).
  - Book-cluster-grouped CV: pessimistic if test shares books with train (likely, since only chapters are disjoint), optimistic never. Treat as a lower bound.
  - Zero-shot floor (no training) under the same folds is the honest yardstick the trained model must beat.
- Expected private band (estimate, unverified): last-position rule maybe 15-30 %; position prior plus lexical overlap about 30-45 %; trained listwise model on frozen encoders plus lexical and boundary features about 42-58 %. Central guess about 48-50 %, with +-3 pts private-LB sampling noise on 288 rows. These come from analogy with listwise placement tasks, not measurements.

## Overfit/underfit risks

Overfit:
- 53 chapters / 1,154 rows with up to 20 candidates per row; high-dimensional frozen embeddings (384-768 dims) vs a few hundred effective independent units. Mitigation: capacity ladder (position prior -> linear conditional logit on scalar features -> LightGBM on scalars -> low-rank interaction head); L2/weight decay and rank r chosen per head by nested grouped CV with a wide grid including the strong end; multi-seed averaging; no PCA/vocab fits on test.
- Chapter-specific position conventions. Mitigation: shrink position effects (few parameters, smooth bins), grouped CV, never use chapter_id or id-derived features.
- Selection noise from many knobs. Mitigation: at most about 10 tuned scalars total, all selected in-script; no Optuna beyond a small fixed grid; compare paired.
- Sibling leakage: several notes on the same paragraph or near-duplicate paragraphs across chapters. Mitigation: union-find groups (diagnostic 5) merged with chapter_id groups for fold assignment.
- Label-derived statistics (e.g., "anchor-prior by segment type"): computed only inside training folds, same amount of data at train and test time (final refit uses all train for the test-time statistic; fold-wise OOF for training-row features if any are label-derived; plan avoids label-derived features at first, using learned weights instead).

Underfit:
- Small or domain-mismatched encoder. Mitigation: two or three encoders of different assumption (a small fast model, a base-size retrieval model, and a lexical stack), each scored alone first; the base-size encoder gets the budget since it is the main lever.
- Truncation: notes up to 300 words and some segments may exceed 128 tokens. Mitigation: note max_len 512 tokens, segment max_len 256 (check diagnostic 2), chunk-and-mean for the rare longer text; log how many texts are truncated.
- Information thrown away by sentence-level embeddings (a quotation fragment is diluted inside a long segment). Mitigation: explicit lexical/quote/name/number overlap features alongside embeddings, plus neighbour-window features.
- Loss mismatch: use listwise softmax NLL (equivalent to maximising the accuracy surrogate on a single categorical), not a per-candidate binary loss ignoring competition; LightGBM LambdaRank/softmax objective with per-row groups.
- Position-only underfit: leaving out m and relative position loses a strong prior. Features include k, k/(m-1), m-1-k, is_last, is_first, m.

## Recommended approach (primary + fallback)

**Primary: lean listwise scorer over a broad within-row feature family built on frozen encoders and lexical overlap, plus a tiny trained interaction head.**

Representation (all label-free, computed once per text, CPU):
- Frozen encoders via `transformers` AutoModel with mean pooling: (E1) small sentence encoder (about 22-33M params, e.g. MiniLM/bge-small class) and (E2) base-size retrieval encoder (about 110M, e.g. bge-base/e5-base/gte-base class; use a multilingual sibling only if diagnostic 11 says so). Revisions pinned; fixed batch size, length-sorted batching, `torch.set_num_threads(8)`, fp32 on CPU. Encode every note and every segment; also encode "segment k plus its previous segment" windows only if they earn a measured gain (roadmap step 5).
- Per-candidate k features (all relative to the row's own candidates):
  - Position: k, k/(m-1), m-1-k, is_first, is_last, m, paragraph words, note words.
  - Dense: cosine(note, seg_k) for each encoder; within-row z-score, rank, gap to row max, gap to runner-up; sim at k-2..k+2 (lag/lead); cummax of sim up to k and its argmax flag; sim(k) minus sim(k+1) and minus sim(k-1) (boundary evidence: content at k, change at k+1); similarity to paragraph centroid.
  - Lexical: BM25 and TF-IDF cosine (IDF from train fold paragraphs only), rare-token overlap, proper-name overlap (capitalised non-initial tokens), number/year overlap, character 4-gram overlap, longest-common-substring ratio between any quoted span in the note and the segment (regex used only to extract spans as a feature input), headword match (leading short term in the note vs segment), first-mention and last-mention flags for the note's entities across the paragraph.
  - Boundary/form: last-character class of seg_k, closing-quote/bracket flags, seg_{k+1} first token class (lowercase continuation, connective from a vocabulary fit on train), segment lengths, word-count of k, k+1.
- Models (each trained with exact listwise NLL over the m candidates of a row; all train-only, fixed seeds):
  - M1 conditional logit: linear (few parameters, standardised features, standardisation fit on train fold). Strong-end L2 grid. Floor for trained models.
  - M2 LightGBM ranker (`lambdarank` or `rank_xendcg` with group = row, k=1 style; or binary with per-row softmax post-normalisation), `deterministic=True`, `force_row_wise=True`, `num_threads=8`, fixed seed; rounds chosen by nested CV in-script, final refit at mean best iteration x 1.1.
  - M3 low-rank interaction head on frozen embeddings (the genuinely trained neural component): s_k = u . e_k (segment-only attachability) + e_note^T W e_k with W = A B^T (rank r about 16-32) + a neighbour-contrast term using e_{k+1} and e_{k-1} + a linear head over the scalar feature vector. Parameters on the order of 10^4-10^5; dropout, weight decay, fixed epochs, 5 fixed seeds. Latent-structure framing: P(anchor=k) = softmax_k(s_k) with potentials {relevance of k, boundary contrast to k+1, position prior}; trained by exact NLL; decode argmax. Few learned parameters, enumerable configurations, no sampling.
- Ensemble: average per-row log-probabilities (all members already live on the same softmax scale, no cross-test normalisation). Members enter only if close to the best single (within about 2 pts under paired grouped CV); otherwise ship the single best. Final prediction refits all members on 100 % of train with fixed rounds/epochs; fold models are used only for the OOF table.
- Genuine-training argument (compliance): the trained heads (M1-M3) are load-bearing; the strip-the-ML ablation quantifies it. If a reviewer treats frozen-encoder-plus-head as insufficient, the LP-FT rung is the compliance layer (roadmap step 7): fine-tune only the top 1-2 layers of E1 plus the head with tiny LR for 1-2 epochs on (note, segment) pairs, log parameter-change norm and CV gain over the probe. CPU cost makes this optional and measured.

**Fallback (no pretrained weights, no network): lexical + structure + trained scorer only.** Same feature family minus dense similarities, M1 and M2 only. Used if the reviewer rejects pretrained weights, or if the grader has no HF access (this is a separate file/plan switch decided offline, NOT a runtime try/except). Measured cost of this reading to be reported from CV (expected a few points below the primary; unverified). Both designs must exist as compiled, tested code paths of the plan; only one ships.

## Rejected options

- Full fine-tune of a cross-encoder (note, segment[+context]) on CPU: 1,154 rows x about m candidates (order 10-15k pairs/epoch at 256-512 tokens) is hours per fold on 10 cores, so OOF estimation is infeasible; also memorises 53 chapters.
- Zero-shot or few-shot LLM scoring/generation: inference-only (banned), CPU-infeasible, and a source-memorising LM is functionally source lookup.
- Rule-only solver (quote-containment regex, "last segment", TF-IDF argmax): fails strip-the-ML; description-style "decision rules" are banned as a complete solution.
- chapter_id / id-order / footnote-number features, or enforcing "one note per paragraph / one anchor per position" across test rows: cross-row or id exploitation, absent from the train-only line, rejected.
- Pseudo-labelling, test-time adaptation, test-fitted IDF/PCA/vocab, test-set rank normalisation: banned (CLAUDE.md 2.3 #5).
- Synthetic data (shifting markers inside real paragraphs to manufacture new examples, generated notes): banned (2.3 #6).
- Big many-seed ensembles of the same family and Optuna with dozens of trials: selects noise at 53 chapters; fixed small grids instead.
- Token-level span marking / sequence tagging: not asked; anchor is one boundary, reduction to ranking m candidates loses no valid answer.
- Joint decoding of paragraph-wide assignments or sampling-based decoders: unnecessary for single-label argmax; primary-design gate says no.

## Fixed work plan & runtime budget

Hardware: CPU, constants `DEVICE="cpu"`, `NUM_THREADS=8` (below the 10-core cap; set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` before importing torch; `torch.set_num_threads(8)`, `torch.set_num_interop_threads(1)`; LightGBM `num_threads=8`). RAM: embeddings for about 1,442 notes + about 15-29k segments x 768 floats < 0.1 GB; plenty of headroom under 62 GB.

Estimated stage times (estimates, to be confirmed by profiling on a 10-core CPU, not verified):

| Stage | Work | Est. time |
|---|---|---|
| Load, parse, validate | 1,442 rows, JSON | < 1 min |
| E1 encode | segments (about 15-29k, about 40 tokens) + notes (1,442, up to 512 tokens) | 3-5 min |
| E2 encode (base-size) | same texts | 10-16 min |
| Lexical + boundary features | BM25/TF-IDF per fold, overlaps, LCS on quotes | 3-6 min |
| M1 + M2 + M3 repeated grouped CV (5 folds x 3 repeats) incl. nested grids | cached embeddings, small heads | 8-12 min |
| Final refit (100 % train) + test predict, 5 seeds | | 2-4 min |
| Validation, write CSV | | < 1 min |
| Total | | about 30-45 min (budget <= 55 min; ceiling 90 min leaves >= 30 % headroom) |

Fixed counts (to be frozen after profiling, never branching on time): folds 5 x 3 repeats for the in-script OOF table; seeds 5 for M3; LightGBM rounds from nested CV (mean best iteration x 1.1) with fixed learning rate and num_leaves grid of at most 3 values; L2 grid of about 8 log-spaced values; fixed `max_len` per text type; fixed batch size for encoders; `random`, `numpy`, `torch`, `PYTHONHASHSEED` seeded; DataLoader generators seeded (or plain deterministic loops). Time used only for `print` logging. No try/except import fallbacks; no `os.cpu_count()`. Early stopping on validation score inside CV is fine; final refit uses fixed counts. If the profiled E2 time exceeds the table by more than about 40 %, drop E2 to a smaller encoder at plan time (a code constant edit before submitting), never at run time.

Memory/OOM: fp32 CPU, batch of tokens bounded; no GPU memory issue. Output validation: columns `id,anchor_index`, ints in [0,m-1], ids equal to sample_submission ids, reload with `keep_default_na=False` and re-verify.

## Metric-aware training & decode

- Metric is exact-match accuracy over a single categorical: loss = listwise softmax NLL over each row's m candidates, one weight per row (matches equal per-example weight; do not reweight by chapter size or by 1/m).
- Decode = argmax. Posterior calibration affects only blending across members (log-prob averaging on a common softmax scale); each member gets at most a single cross-fitted temperature if needed. No decode constants, no thresholds, no post-processing; none to tune on OOF.
- Position prior: learned inside the model through smooth position features and m (not a hard-coded table).
- Per-slot calibration (F2.7) does not apply (no probabilistic output is scored).
- Tie handling: argmax ties broken by lower index deterministically, though ties are essentially impossible with float scores.
- Invalid-output protection: argmax over m candidates cannot go out of range; the validator still asserts it.
- "Predict the original placement which may be more specific than the editorial range": the contrast features (k vs k+1) and position terms let the model choose the end of the discussed run rather than its first segment; verify on OOF that errors are mostly off-by-one in the early direction and examine whether adding the "end-of-run" lead features reduces them.

## Structural signals

Turn each invariant into a feature, prior, or auxiliary supervision, never a rule that replaces the model:
- Marker placement follows the content it comments on; the anchor is the LAST segment of the discussed run, so lead/lag similarities and the "k relevant, k+1 not" contrast are explicit inputs (H3).
- Quotation and bibliographic text in the note must be reflected in the paragraph (quote overlap, name/number/year overlap) and in the cited claim; these are within-row features, standard and legitimate, fed to a trained scorer (CLAUDE.md F2.9).
- Notes of different kinds map to different position patterns: attribution disputes attach to the segment containing the attribution; term glosses attach to the segment using the term (headword match); correction/qualification notes attach to the claim they qualify. The model learns these from the note text through the interaction head and lexical flags; no explicit note-type classifier is hard-coded (an auxiliary note-type head is not possible without labels).
- Segment-only attachability (H4) via u . e_k.
- Boundary form: sentence-final punctuation, closing quotes, continuation starts (lowercase or connective) of seg_{k+1}.
- Candidate-set invariance: scores are relative to the row's own candidates; relative transforms (z-score, rank, gap to best) are within-row, so no whole-test aggregation.
- Augmentation: none that creates synthetic labelled data. Allowed: encoder dropout in training and multi-seed averaging. No marker shifting or paragraph re-splitting (that changes labels, equals synthetic data).
- Excluded signals (compliance triage, each flagged for a reviewer question where relevant): id ordering within chapter, rows of the same paragraph in test (cross-row), chapter-level statistics from test rows, test-based assignment constraints. Chapter-level statistics from TRAIN labels are not used as features (test chapters are unseen).

## Experiment roadmap

1. Contract, metric and validation: parse and assert data; implement accuracy and the unit tests; build chapter groups (plus union-find merge); build the 5x3 grouped folds and the 8-chapter sanity holdout; validator. Stop when tests pass and baselines (last-position, uniform, position-only prior) are tabulated.
2. Strongest cheap baseline, end to end and valid: M1 (position + BM25/TF-IDF + E1 cosine features) -> CSV through the validator; record grouped CV mean +- std and per-fold. This is submission 1 (valid baseline).
3. Representation and structural insight: add E2, lead/lag/cummax/contrast features, quote/name/number/headword features, boundary-form features. One change per experiment, paired on the same folds. Stop when a step stops beating about 1 SE of the paired difference.
4. Metric-aware model: M2 (LightGBM ranker with listwise grouping) and M3 (low-rank interaction head); compare each alone first. Zero-shot (untrained cosine argmax) and position-only rows in the table are the floors.
5. Diversity and ensemble: blend only members within about 2 pts of the best; average log-probs; optional window-encoded segments (segment plus previous segment) if it beats noise. Submission 2/3 = best single and ensemble.
6. In-script bounded selection: L2/rank/rounds grids with nested grouped CV; the outer-fold evaluation must be disjoint from selection. Score the 8-chapter sanity holdout once; if its accuracy falls well below the OOF mean (beyond about 2 SE), simplify.
7. Optional LP-FT rung (only if the reviewer wants more fine-tuning or CV shows headroom): top 1-2 layers of E1, tiny LR, 1-2 epochs, fixed seeds; log parameter norm change; keep only if CV gain over the probe exceeds noise and CPU time fits.
8. Final: refit on 100 % train with fixed counts, run `python3 solution.py <public_dir> <out>` from a clean `working/` twice, diff predictions (should be identical or near), check that predicted position histogram resembles OOF predictions and train labels; submission 4 = final. Credits: baseline, best single, ensemble, final (6 per problem); do not tune against the public LB.

## Compliance audit

Run against CLAUDE.md section 7 and the strategist section B self-audits (expected answers for the plan as designed; verify on the code):
- Test file read only for one-row-at-a-time prediction and for schema/row-count/size checks: yes. No stats, vocab, IDF, scaler, PCA, clustering, dedup, rank-normalisation over test rows: none planned.
- Time use: logging only; no `elapsed()` in a condition; no timeouts in LightGBM/Optuna: planned.
- No `cuda.is_available`, `os.cpu_count()`, import fallbacks, env-based recipe selection: planned; constants `DEVICE="cpu"`, `NUM_THREADS=8`.
- Hard-coded tuned constants: none from offline experiments. Every grid selection (L2, rank, rounds, ensemble membership) runs inside the script on train-only nested CV. Fixed architecture constants (encoder names, max lengths, grid ranges) are design choices, listed in comments.
- External data / synthetic / self-hosted weights / non-allowed library: only HF encoder weights (pinned revisions) via `from_pretrained`; `transformers`, `torch`, `numpy`, `pandas`, `scikit-learn`, `lightgbm` (all in the allowed stack); no GitHub, no `pip install`.
- Strip-the-ML: measured ablation required (zero-shot cosine/BM25 argmax and position-only vs trained); trained component must clearly dominate.
- Cross-row (sibling) leakage: candidate features use only the row's own note and paragraph; features derived from other rows' labels are avoided or made out-of-fold with matched train/test construction.
- Quote/regex use: regex only extracts spans (quotes, numbers, capitalised names) as inputs to the trained scorer; no regex that decides the anchor.
- Source readable and < 512,000 bytes, plain UTF-8, comments explaining each step: yes by design. No embedded weights, base64, or runtime-generated code.
- Challenge-specific bans (no source lookup, no external task labels, grouped validation, no train-accuracy claims): honoured as listed in Compliance regime.
- "Do not access the raw upload / private answers / another solution": honoured.
- Reviewer risk items: frozen-encoder-plus-head acceptability; lexical overlap features (grey but kept as inputs, never the core); regexed quote matching.

## Open questions & assumptions

Questions for a reviewer (plan under each reading):
1. Are pretrained general-purpose HF encoders (frozen embeddings plus trained task heads) allowed here, given the description is silent and only bans external task labels / retrieved passages / source lookup? Reading A (allowed): primary plan. Reading B (not allowed): fallback lexical + structure plan with M1/M2 only (cost to be measured). Compliant under both readings: the fallback.
2. Is a pretrained encoder with no fine-tuning (frozen) plus a trained interaction head sufficient as "genuine training", or must an encoder layer be fine-tuned? Plan: M3 plus the optional LP-FT rung with logged parameter-norm change.
3. Is feeding regex-extracted quoted spans, capitalised names, years and numbers as features into a trained scorer acceptable (not "exploiting the data-generation process")? Plan keeps them as inputs only and runs the strip-the-ML ablation.
4. The description is truncated at the end of the "What Not To Do" list; are there further bans (for example on internet access for model weights)? If HF downloads are disallowed in the grader, only the fallback ships.

Assumptions (all unverified until data arrive):
- Text is predominantly English (diagnostic 11 may change the encoder family).
- Average m is about 11-15 and anchors are not trivially concentrated at one position (diagnostic 3).
- Test chapters resemble train chapters in style and book families (only the chapter ids are disjoint).
- Profiled CPU speed is around 10 cores at modern server speed; stage times above are estimates and must be re-profiled before freezing counts.
- The platform's credit/determinism checks follow CLAUDE.md section 3; CPU thread counts are fixed constants so numerics should be reproducible within the same container.
- Not verified: dataset schema contents, class balance, the actual position prior, whether other scholarly-note duplicates exist, and any score level. No files, internet or other plan/eval folders were read.
