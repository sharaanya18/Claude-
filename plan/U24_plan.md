# U24 Plan: Cross-Publisher Kazakh (Arabic-script) Headline Generation

Status legend used throughout: **[DESC]** = stated in the challenge description; **[UNVERIFIED]** = hypothesis or estimate, no dataset was available when this plan was written; **[ASSUME]** = the description is silent and this plan assumes something. No dataset files were opened, no test file was studied, no internet was used, no solution code is written here.

---

## Contract & decision unit

**One valid answer.** For each of the 6,420 ids in `test.csv`, one non-empty string `title`, in Arabic-script Kazakh, in the same script as the input. CSV columns exactly `id,title`, ids exactly as in `test.csv`/`sample_submission.csv` (`test_00000`, ...). [DESC]

**Invalid vs merely low-scoring.**
- Invalid (whole submission fails): missing ids. [DESC]
- Zero for that row: blank or missing title. [DESC]
- Ignored: duplicate ids (first used), extra ids. [DESC]
- Low-scoring but valid: wrong script, translated or romanised output. The description forbids it ("do not transliterate, translate, or romanize"), so treat Latin/Cyrillic output as a compliance defect, not just a low score.
- CLAUDE.md section 5 is stricter than the grader (no NaN, no empty strings, exact column order), so we follow section 5.

**Metric, term by term.** [DESC] with [ASSUME] on tokenisation:
- `Composite_row = 0.8*ROUGE-L_F1 + 0.2*NumPrec` when the REAL title contains at least one digit sequence. Otherwise `Composite_row = ROUGE-L_F1`.
- ROUGE-L F1 is the word-level LCS between generated and real title. The word tokenisation is not specified. [ASSUME] whitespace tokens after stripping punctuation. The local metric must implement two variants (whitespace split; Unicode `\w+`) and check that rank order of models agrees. Arabic-script Kazakh may contain combining marks (category Mn) that `\w+` splits on, so this matters.
- NumPrec = fraction of digit sequences in OUR output that appear in the real title. Unspecified: (a) whether it is a set or a multiset; (b) the digit regex (`[0-9]+` vs Unicode `\d+`, which includes Arabic-Indic digits); (c) the value when we output zero digit sequences on a row whose real title has digits. [ASSUME] undefined or zero. The plan must be robust to both "zero" and "vacuously 1".
- Number precision is a precision-only term. Emitting an unsupported number costs up to 0.2 on rows where the real title has numbers, and costs only a little ROUGE precision on rows where it does not (the component is skipped there). A model that never emits digits is "safe" under convention B (vacuous 1) but forfeits the term under convention A (0).
- Aggregation is a flat mean over test rows (no per-publisher macro). The two test publishers' sizes are unknown (6,420 rows total, so about 3.2k each on average). A larger publisher counts more.
- Range is [0, 1], maximise.

**True independent unit.** The publisher (outlet), then the story family within it. Effective sample size for "house-style generalisation" is 5 training publishers, even though there are 50,660 rows. Rows are exchangeable only within a publisher. Wire stories recur reworded across publishers, so a story family (title-shingle Jaccard >= 0.6, transitive) is the unit for leakage control. [DESC]

**Pipeline stages, to be diagnosed separately.**
1. Input coverage: does the first K tokens of the body contain what the headline needs (words, numbers)? Oracle: ROUGE-L of the real title against the best extractive sentence/window.
2. Generation quality: the seq2seq model's greedy/beam output.
3. Decode: n-best reranking, MBR and number-grounding. The oracle for this stage is the best candidate within the n-best list (an upper bound on what reranking can gain).

---

## Compliance regime

**Domain.** NLP, text-to-text generation (summarisation-style, abstractive, low-resource agglutinative language, non-standard script). Whether the challenge is labelled "Fine-tuning" or "From-scratch" is not stated. [ASSUME] pretrained backbones from HF are allowed, because CLAUDE.md allows them by default and the description itself says mBERT/XLM-R "can still be used as a starting point". Open question 1 covers the other reading.

**Explicit bans in the description (hard constraints).**
| Ban [DESC] | Plan's handling |
|---|---|
| Live internet lookup to retrieve the original article or headline | No network access in `solution.py` except HF `from_pretrained` for backbone weights (CLAUDE.md section 1). No search, no URL fetch, no GitHub. |
| Off-the-shelf multilingual tokenizer (mBERT, XLM-R) as an unmodified drop-in expecting out-of-the-box quality | mBERT/XLM-R tokenizers and encoders are not used. The primary model uses the mT5 SentencePiece (a different tokenizer from the same multilingual-web family). We do NOT rely on it out of the box: we measure its unk rate and round-trip exactness on train, patch failures by train-derived vocabulary extension, and fine-tune. Whether a reviewer reads mT5's tokenizer as "unmodified drop-in" is open question 1. |
| Arabic-script-specific models such as AraT5 on the assumption that shared script implies transfer | Treated as banning the whole class: no AraT5, AraBART, AraGPT2, ARBERT/MARBERT, CAMeLBERT or similar. |
| Retrieving the real answer | No retrieval of train headlines as predictions. Test families have no copies in train anyway. [DESC] |
| Rewording: "do not transliterate" output | Output stays Arabic-script. An internal Arabic-script to Cyrillic mapping is NOT used in the primary (see Rejected options, grey). |

**Silent items and assumptions.**
- Compute: description gives no runtime or GPU. [ASSUME] CLAUDE.md: single A10G 24 GB, target <= ~40-45 min, hard ceiling 1 h.
- Model size cap: none stated. [ASSUME] none.
- Pretrained seq2seq (mT5, ByT5, mBART): not named in the ban list. [ASSUME] allowed as general-purpose multilingual backbones.
- `outlet` column: present in train and test. Use in train only (sampling weights, held-out publisher selection). It is never an input feature (test codes are disjoint from train codes, so it carries no usable information) and is never used to group or normalise test rows.
- Metric-aware decoding (MBR, n-best reranking, number grounding): not banned by this description. Allowed under CLAUDE.md if tuned on train-held-out data only and it is a per-row function of its own n-best list.
- ROUGE tokenisation, digit regex and empty-number convention: unspecified (see Contract).

**CLAUDE.md regime applied.** Section 2 and 3 in full: train-only fits of every tokenizer extension/vocabulary/statistic; no test-set statistics; fixed work plan; no time-based branching; no `torch.cuda.is_available()` or `os.cpu_count()` switches; no try/except import fallbacks; fixed seeds; `device="cuda"`; HF cache redirected to `WORK_DIR`. Strip-the-ML test: removing the fine-tuned model leaves nothing that works (Lead-k is only a logged reference, never shipped).

---

## Data findings

**Nothing below has been verified against the data.** Facts from the description [DESC]: train 50,660 rows, 5 outlets, columns `id,outlet,text,title`; test 6,420 rows, 2 disjoint outlets, columns `id,outlet,text`; 693 train rows were dropped because their story family had a held-out-publisher member; there are 0 exact and 0 near-duplicate (Jaccard >= 0.6) title pairs across train/test; ids carry no information; the description reports mBERT 16.5% unk rate and XLM-R about 3.1 tokens/word on this text (as measured by the authors).

**Diagnostics to run on TRAIN only (scratch notebook, not in the final script), each with the hypothesis and the decision it drives.** Test files: schema, row count and id format only.

| # | Diagnostic (train) | Hypothesis [UNVERIFIED] | Decision it drives |
|---|---|---|---|
| D1 | Rows per outlet; per-outlet mean/P50/P90 of title words, text words, title and text chars | Unequal outlet sizes (one or two dominate). Title length varies visibly by outlet (e.g., 6-9 vs 10-15 words). | Sampling temperature across outlets; choice of in-script held-out outlet H (needs enough rows, >= ~4k, so the val SE is small); `MAX_TGT`. |
| D2 | Tokenizer audit: mT5 (and for reference ByT5, byte count) tokens per word, `<unk>` rate, and exact round-trip rate `decode(encode(s)) == s` on titles and on texts, per outlet | Tokens/word about 2.5-3.5; unk rate small but non-zero for Kazakh-specific letters; round-trip imperfect because SentencePiece applies NFKC (Arabic presentation forms, high-hamza variants, ZWNJ). | If title round-trip < 99.5% or unk > ~0.5%: apply the train-derived vocabulary patch (Fixed work plan, stage 1b) and disable lossy normalisation. Also sets `MAX_SRC`. This is the most important diagnostic: a lossy round trip is a hard ROUGE cap. |
| D3 | Character inventory per outlet for titles and texts: Unicode code points, Arabic letter variants (yeh ي/ی, kaf ك/ک, heh ه/ھ/ە, hamza marks), Mn marks, ZWNJ/ZWJ, Latin/Cyrillic/CJK leakage, digit forms (`0-9` vs `٠-٩`), punctuation sets (`، ؛ ؟ « » " ' ! ?`) and whether titles end with punctuation | Orthographic/Unicode variants differ by outlet, so a test publisher may use letter variants not seen in train. Titles in one outlet end with punctuation or quotes; others do not. | Whether to normalise at all. Default is NO normalisation of targets (output must match the real title's characters). If variants differ by outlet, the model must learn to copy the source's orthography, which supports raw input to raw output. Also feeds ROUGE tokenisation variants. |
| D4 | Digit usage: share of titles with digits per outlet (exact %), digit forms, share of title digit sequences that appear verbatim in the text, share in the first 384 tokens, typical roles (year, date, count, percent) | 20-35% of titles contain digits; over 85% of title numbers appear verbatim in the article body; most in the lead. | Number policy (copy-grounded), source truncation, how much weight the 0.2 term deserves (about 0.2 x 25-35% = 0.05-0.07 of the score ceiling). |
| D5 | Extractive ceilings per outlet: ROUGE-L of title vs (a) first sentence, (b) first 8/12/20 words, (c) best single sentence (oracle), (d) best window of the first 256/384/512 tokens | Lead-1 about 0.10-0.20; oracle sentence about 0.30-0.45; strong lead bias. | Truncation length; a floor the model must clearly beat (strip-the-ML evidence); an upper bound for pure extraction. |
| D6 | Title-in-text leakage: share of rows where the title (or a >= 0.8 word-Jaccard variant) occurs in the body, especially as the first line/sentence, per outlet | One or two training outlets prefix the body with the headline or an unmarked duplicate. If so, the model learns a copy-first-line shortcut that will not transfer to the held-out outlets, or will transfer perfectly if they do the same (unknowable, and we must not look). | Lead-dropout augmentation (randomly start the body at sentence 2 or later with small probability), outlet-balanced sampling, and honest LOPO checks (train on outlets with the pattern, test on one without). |
| D7 | Boilerplate: most frequent first-60-char and last-60-char strings and token n-grams per outlet (datelines, bylines, agency tags, footers, "read also") | Datelines at the start ("city, date, agency") and source footers differ by outlet. | Whether to strip them. Generic structure removal must be learnable or minimal; no hand-written per-outlet regex (compliance). Prefer letting the model skip them; if needed, a single generic rule (e.g., drop a leading parenthetical/dateline of <= N tokens) validated on LOPO. |
| D8 | Word-level overlap structure: share of title words that appear exactly in the text, and share whose first 4-5 characters (stem proxy) appear | Exact-word copy 55-75%; stem-level copy 80-90% (agglutinative suffixes change the surface form). | Confirms need for a generative model rather than pure copy; sets realistic ROUGE-L ceilings (I expect a trained model to land well below the oracle in D5). |
| D9 | Story families inside train: union-find over title shingles (word 2-grams, Jaccard >= 0.6, transitive, via an inverted index on rare shingles, not O(n^2)) including cross-outlet families; family size distribution; share of rows in cross-outlet families | A noticeable share of train rows (perhaps 10-30%) are in multi-outlet families (national wire stories). | Mandatory for validation: when outlet H is held out, drop every training row in any family that contains an H row, replicating the task's own split rule. Also a near-duplicate memorisation risk inside training. |
| D10 | Exact duplicates `(text,title)`, duplicate titles with different texts, empty or very short texts/titles, texts with fewer than ~20 words, extremely long titles | A small share of exact duplicates and stubs. | Drop exact duplicates and rows with a degenerate title (e.g., < 2 words), applied identically at every run; fixed rule. |
| D11 | Title token-length and text token-length percentiles (P50/P90/P99) under the chosen tokenizer, per outlet | Titles P99 about 40-60 tokens; texts P50 about 600-1200 tokens (far above 384). | `MAX_TGT`, `MAX_SRC`, `max_new_tokens`; runtime estimate. |
| D12 | Style-distance between outlets: standardised fingerprint (mean title words, share with digits, share ending in punctuation, quote-mark rate, Latin/Cyrillic char rate, mean first-token position of title words in text), pairwise distances | Outlets form a spread; the test publishers are chosen as the most distinct ([DESC]), so the largest within-train distance is the best proxy. | Selection rule for in-script held-out outlet H (outlet farthest from the rest among those with enough rows). |
| D13 | Information ceiling via n-best oracle after a first fine-tune (later, roadmap step 3): best-of-8 ROUGE-L | Best-of-8 exceeds the 1-best by 0.05-0.10. | Upper bound for MBR/reranking gains; if the gap is < 0.02, skip MBR. |
| D14 | Numbers in generated output (after first fine-tune): share of generated digit sequences that are absent from the article ("hallucinated numbers") | Several percent; more on held-out outlets. | Value of number-grounding rerank. |

**Expected-structure hypotheses [UNVERIFIED]:** headlines are short, lead-driven, copy-heavy with suffix changes; wire-story families give several differently worded headlines for the same body (useful style variety); orthographic variants and title conventions (length, punctuation, quote style, numerals) are the main cross-publisher shift.

**Not verifiable here:** all numbers in the table, which outlet is largest, whether some outlet's body repeats its title, the actual tokenizer behaviour of mT5 on this script.

---

## Validation design

**Test-split reproduction.** Test = 2 publishers entirely absent from train, "most stylistically distinct" from the training pool, with all train copies of any story family touching a test article removed. [DESC] So the correct proxy is **leave-one-publisher-out (LOPO)** with the same family removal rule, and with the held-out publisher chosen as the most distinct one.

**Offline (development) validation.**
- Folds: hold out each of the 5 outlets in turn (or at least the 3 most distinct), train on the other 4 minus every story family that contains any held-out row (D9 union-find on titles, Jaccard >= 0.6 on shingles, transitive). Same removal on the validation side is unnecessary (all rows of H are kept).
- A cheap proxy model (mT5-small, shortened schedule) is used for the comparisons, because 5 folds of the real model do not fit in a development session of reasonable length. Report per-outlet composite, ROUGE-L, NumPrec (rows with numbers only), plus Lead-1 and Lead-k references, and the mean and standard deviation across held-out outlets.
- Accept a design change only if the paired per-row difference (bootstrap over rows, resampling story families) is positive in at least 3 of 5 held-out outlets AND the mean paired gain exceeds 1 standard error. With only 5 publisher groups, small differences (< ~0.005) are unresolvable; do not chase them.

**In-script (shipped) validation.** One LOPO fold inside `solution.py`: hold out outlet H, selected deterministically by the D12 fingerprint rule (largest distance to the pooled rest among outlets with >= ~4k rows). Then:
- Train on the other outlets (minus H-overlapping families).
- Decode a fixed seeded subset of about 3,000 H rows (n-best), compute the exact composite, ROUGE-L and NumPrec.
- Calibrate decode scalars on this subset (see Metric-aware training & decode) with 2-fold cross-fitting by story family (fit on one half, score on the other, swap) so the reported number is not optimistic.
- Log the Lead-k reference on the same H rows (no zero-shot floor exists for this script, so Lead-1/Lead-k is the honest floor the model must beat).

**Which way each proxy is biased [UNVERIFIED reasoning].**
- LOPO with the most distinct training outlet: pessimistic for training diversity (the shipped model sees 4 outlets, the final-quality model would see 5) but optimistic for the real shift, because the test publishers are chosen to be more distinct than any training outlet is from the others. Net expectation: the real private score lands at or below the LOPO number on the most distinct outlet. I would plan for a drop of 0.01-0.04 composite below it.
- Random k-fold within outlets: strongly optimistic (same publisher style, wire-story families spanning folds). Never used for model selection.
- My own ROUGE tokenisation may differ from the grader's; rank order across models should be stable, absolute values may shift by a few points.

**Metric unit tests (before any modelling).** Perfect output = 1.0; word-order reversal gives the LCS-based value; constant string; empty string = 0; digits-only mismatch; a row with digits in the real title and none in the output under both conventions (0 and vacuous); a row without digits in the real title (component skipped); duplicates in the digit list under set vs multiset.

**Anti-overfit hygiene.** The shipped script makes few free choices: decode scalars from a small grid on H (2-fold cross-fitted), nothing else tuned in-script. Offline choices (sampling temperature, augmentation on/off) are decided on LOPO with the acceptance rule above and then frozen as fixed constants.

---

## Overfit/underfit risks

| Risk | Evidence/hypothesis | Mitigation |
|---|---|---|
| **Style memorisation (overfit).** Only 5 publishers; the model can key on outlet-specific conventions (length, punctuation, dateline, orthography) | The task is built to punish exactly this ([DESC]) | Do not feed `outlet`; temperature-sampled outlets (alpha=0.5 as in mT5 pretraining: weight proportional to n_o^0.5 / sum) so a dominant outlet does not set the style; no more than about 2 epochs; low learning rate, dropout 0.1, EMA of weights; LOPO model selection. |
| **Copy-first-line shortcut (overfit)** if some outlets' bodies start with the headline (D6) | Hypothesis | Lead-dropout augmentation with small probability (start the body at a later sentence boundary); LOPO test on an outlet without the pattern. |
| **Near-duplicate memorisation inside train** (wire families republished) | D9 | Dedupe exact duplicates; the in-script validation removes whole families from training; do not count repeated families as independent evidence. |
| **Single held-out publisher is noisy** | One fold, one publisher | Use only large effects for decisions; offline LOPO over several outlets before freezing the design; the in-script fold is used for decode scalars and for a sanity number, not for model selection. |
| **Decode over-tuning** | Grid + MBR + number policy on about 3k rows | Few scalars, cross-fitted by family, accept only gains consistent across both halves. |
| **Underfit: backbone too small or too little training** | New script for the pretrained model; mT5-small (300M with embeddings, tiny body) | Primary is mT5-base; about 2 epochs; compare at equal compute offline (small ~5 epochs vs base ~2 epochs) before shipping. |
| **Underfit: truncated source** loses the number or key entity | Texts P50 likely >> 384 tokens (D11) | Lead-biased truncation justified by D5 oracle; check the share of title digits that sit beyond `MAX_SRC` (D4); raise `MAX_SRC` to 512 only if it pays for its runtime. |
| **Underfit: tokeniser** (high fragmentation, unk, lossy round trip) | Description reports bad behaviour of mBERT/XLM-R tokenisers | D2 audit; train-derived vocabulary patch; raw (non-normalised) targets; fall back to ByT5 only if the patch cannot reach exact round trip. |
| **Loss mismatch**: token CE vs ROUGE-L/number precision | Metric-sensitive | Metric-aware decode (length normalisation, MBR, number grounding); optionally 1.5-2x loss weight on digit tokens (offline-validated only). |
| **fp16 instability of mT5** | Known T5/mT5 behaviour | bf16 autocast on A10G, fp32 master weights. |
| **Cross-publisher number hallucination** | D14 | Number-grounding rerank; model learns to copy from the source. |

---

## Recommended approach (primary + fallback)

**Primary: fine-tune `google/mt5-base` (HF, revision pinned at dev time, bf16 autocast) as a seq2seq headline generator, trained on all training outlets except one held-out outlet H, decoded with n-best beam search and a small, cross-fitted metric-aware rerank.**

- **Representation.** Raw Arabic-script text in, raw headline out. No transliteration, no outlet feature. Input is the article body truncated at the head to `MAX_SRC` tokens (initially 384, pending D5/D11); target truncated to `MAX_TGT` (initially 48, pending D11). Unicode normalisation of targets is avoided (D2/D3); if the round-trip audit shows loss, patch the tokenizer rather than normalise the text (see below).
- **Tokenizer patch (conditional, derived from train only).** If D2 shows unk or inexact round-trip, add the missing characters/pieces seen in train titles+texts as new tokens (mean-initialised from their constituent pieces' embeddings), and disable lossy normalisation. All determined in-script from train text; test text is never used to build vocabulary.
- **Loss/optimiser.** Token cross-entropy; Adafactor with constant lr 1e-3, `scale_parameter=False`, `relative_step=False` (the standard published T5 fine-tuning recipe, not tuned here), bf16 autocast, grad-clip 1.0, weight EMA (fixed decay, e.g. 0.999) evaluated at the end. Effective batch 32 (micro-batch 16 x accum 2). [ASSUME the sizes fit in 24 GB: fp32 weights 2.3 GB, grads 2.3 GB, Adafactor states small, activations at 16x384 well below 12 GB; confirm by profiling.]
- **Outlet handling.** Temperature-weighted sampling across outlets (alpha = 0.5, formula, not tuned); no outlet token.
- **Augmentation.** Lead-dropout (small probability, only if D6 shows title-in-text outlets); random source-length jitter (e.g., sample `MAX_SRC` in a small range) as light regularisation. Both validated offline on LOPO before inclusion; both are input-side only on real articles (not synthetic labelled data).
- **Fixed schedule.** About 2 epochs over the training pool (about 80k examples, about 2,500 steps of batch 32), linear warmup then constant/linear decay. No validation-triggered stopping; the number of steps is a fixed constant from profiling, written as a formula of the training-set size.
- **Decode.** Beam search `num_beams=8`, `num_return_sequences=8`, `max_new_tokens=MAX_TGT`, `no_repeat_ngram_size=3`; then rerank the n-best list per row with the in-script calibrated scalars (below). Deterministic batching: sort by source length, fixed batch size.
- **Fallback chain (decided before shipping, never at runtime).**
  1. If profiling shows mT5-base training + decode exceeds the budget: reduce `MAX_SRC` to 256, then reduce to about 1.5 epochs, then switch to **mT5-small** with the same recipe and about 5 epochs (3x cheaper per step). The switch is made offline and the constant hard-coded; the script never switches by clock.
  2. Speed lever (apply only if profiling says needed): restrict the **output** layer (`lm_head`) to the pieces seen in train titles and texts plus all single-codepoint pieces of the Arabic blocks, ASCII and digits. The lm_head is the dominant per-token decode cost for a 250k vocabulary. Input embeddings stay full. Build the piece list from train only.
  3. If the tokenizer patch cannot reach an exact title round trip: ByT5-small with a longer-than-usual byte budget (cost: about 2 bytes per Arabic character, so about 768 bytes covers only about 40-50 words of lead; slower and weaker on coverage). Reserve as a last resort.
- **Why this fits this data.** Headline writing is lead-biased and copy-heavy; a pretrained encoder-decoder already knows how to copy and compress, and fine-tuning teaches it the script and house-style-independent headline shape. The largest backbone that fits the budget is chosen first (agent-file heuristic: under-sizing is the most common loss), and extra machinery (MBR, number grounding, vocabulary patching) enters only where a diagnostic shows a measurable gain.

**Expected private score [ESTIMATE, low confidence].** Composite roughly 0.17-0.28 (ROUGE-L 0.15-0.27, number precision 0.4-0.7 on rows with numbers). Reasoning: abstractive ROUGE-L for headline generation in low-resource agglutinative languages with a backbone that has not seen this script is typically in the low 0.2s in-domain; cross-publisher shift on the most distinct outlets costs a few points; Lead-1 sits near 0.10-0.20 in my hypothesis. mT5-small fallback: 0.01-0.03 lower. These numbers are not measured.

---

## Rejected options

| Option | Why rejected |
|---|---|
| AraT5 or any Arabic-specific model (AraBART, AraGPT2, ARBERT, CAMeLBERT) | Explicitly warned against in the description; treated as banned. No Kazakh exposure. |
| mBERT/XLM-R tokenizer or encoder as drop-in (including `EncoderDecoderModel` warm-start from XLM-R) | Description reports 16.5% unk (mBERT) and about 3.1 tokens/word (XLM-R). A warm-started cross-attention would be learned from scratch in a short budget. |
| Extractive Lead-k / TextRank / best-sentence selection as the shipped solution | Rule-based; fails the strip-the-ML test. Kept only as an offline and logged reference baseline. |
| Retrieval of nearest train headline (TF-IDF kNN or embedding) | Description guarantees no family overlap across the split, so little to retrieve; and it is a lookup, not generation (strip-the-ML). |
| Internal Arabic-script to Cyrillic Kazakh transliteration to exploit Cyrillic pretraining, then map back | Potentially large transfer gain [UNVERIFIED], but the description lists "script transliteration" among things this task is not and says "do not transliterate"; a hand-written mapping table is a rule component; the reverse map may not be exactly invertible (loanword spellings, hamza/harmony marks), and ROUGE needs exact surface forms. Grey; not in the primary under the "compliant under every plausible reading" principle. If a reviewer approves, it becomes an experiment gated on exact round-trip of train titles (>= 99%) and a measured LOPO gain. |
| From-scratch transformer with a custom SentencePiece | Clean under every compliance reading, but 50k pairs and about 25-30 min of training will give a weak model; keep as the plan under reading 2 of open question 1. |
| Decoder-only LLM with LoRA (Qwen/Llama class) | Unverified tokenisation of this script (byte-level BPE likely 4-6 tokens/word, so about 600+ tokens per 128-word lead), roughly 5x the training cost of mT5-base per example, and slower decode; does not fit the 1 h budget at useful scale. |
| mBART-50 | It knows Cyrillic `kk_KZ` and Arabic, not Arabic-script Kazakh; heavier than mT5-base; no clear advantage. |
| Full-ensemble of several seq2seq models | Cannot afford in about 40 min on one A10G; seed diversity adds little; n-best MBR gives within-model consensus at far lower cost. |
| Pseudo-labelling, transductive normalisation, test-fitted vocabulary, outlet grouping of test rows | Banned (CLAUDE.md 2.3 #5). |
| Time-based epoch/decode cut-offs | Banned by the determinism checker (CLAUDE.md section 3). |

---

## Fixed work plan & runtime budget

**All counts are fixed constants, chosen offline by profiling; no wall-clock branching, no environment-dependent fallbacks. Estimates below are [UNVERIFIED] and must be profiled on an A10G-class GPU.**

| Stage | Fixed plan | Est. A10G time |
|---|---|---|
| 0. Setup, load, validate schema, drop exact duplicates/degenerate rows, seed everything, `device="cuda"`, `torch.use_deterministic_algorithms(True, warn_only=True)`, cudnn deterministic | 1 pass | 0.5 min |
| 0b. Outlet fingerprint (D12 rule) -> H; story-family union-find (inverted index on rare title shingles, Jaccard >= 0.6, transitive); drop training rows in any family with an H row | train-only | 1 min |
| 1. Tokenise train (fast tokenizer), measure unk/round-trip, apply train-derived vocabulary patch if the fixed criterion triggers | 50k x (<= 384 + <= 48 tokens) | 1.5 min |
| 1b. Load `mt5-base`, resize embeddings for patched tokens | | 1 min |
| 2. Fine-tune | about 2 epochs over about 40k examples (4 outlets minus families), effective batch 32, about 2,500 steps; bf16; Adafactor; EMA | about 20-25 min (about 14 PFLOP per epoch estimated at about 20 TFLOPs effective) |
| 3. H decode, n-best (8 beams, 8 returned) on about 3,000 seeded rows; calibrate rerank scalars with 2-fold family cross-fit; log composite/ROUGE-L/NumPrec and Lead-k reference | 1 pass + CPU reranking grid | 4-5 min |
| 4. Test decode, n-best on 6,420 rows, apply calibrated rerank, build submission, validate, write, re-read | 1 pass | 8-10 min |
| **Total** | | **about 38-45 min** (limit 1 h; headroom about 25-35%) |

**Levers if profiling disagrees.** Reduce `MAX_SRC` to 256 (about 33% less encoder cost); reduce `num_beams`/`num_return_sequences` to 4; use the lm_head output restriction; reduce epochs to 1.5; fall back to mT5-small. Applied offline, then frozen.

**Memory estimate.** mT5-base fp32 weights about 2.3 GB; grads about 2.3 GB; Adafactor and EMA about 2.5 GB; activations at micro-batch 16 x 384 about 6-9 GB (gradient checkpointing available if needed); peak about 14-17 GB < 24 GB. Decode micro-batch 32 x 8 beams within headroom. Confirm on hardware.

**Determinism checklist.** `PYTHONHASHSEED`, `random`, `numpy`, `torch` + `cuda` seeds; fixed `torch.Generator` for the sampler and shuffle; `num_workers=0` (tensors pre-tokenised in memory); fixed micro-batch and accumulation; deterministic length-sorted decode batches; no `timeout`, no `time_limit`, no `elapsed()` in conditions (time only in log lines); fixed `n_trials` is not applicable (no Optuna).

**Input/output validation in the script.** Assert `sample_submission.csv` columns and ids; assert ids equal `test.csv` ids; after decode, ensure every title is non-empty and single-line (replace newlines, collapse whitespace); data-determined fallback for an empty candidate is the next non-empty n-best entry, and if all are empty a short lead snippet of the row's own text, with the count logged and asserted below 1% (a silent degraded path is worse than a loud failure); reload the written CSV with `keep_default_na=False` and re-validate with the section 5 validator.

---

## Metric-aware training & decode

1. **Back-solve the metric.** Composite = 0.8*RL + 0.2*NumPrec when the real title has digits. The 0.2 term is a precision-only gate on digits, so the training signal is "copy digits that are in the source and likely in the title; do not invent any". Since digits in titles are usually copied from the source (D4, to be verified), token CE already pushes this; no special training loss is needed in the first iteration.
2. **Optional training-time lever (offline-validated only).** Up-weight digit-token loss by 1.5-2x (a low-dimensional change that tracks the 0.2 term). Accept only if the paired LOPO gain beats noise (see Validation design).
3. **Length.** ROUGE-L F1 is symmetric between precision and recall, so the optimal length is near the headline length distribution of the unseen outlet; headline length varies by outlet (D1) and the model must infer the length convention from the body alone. Decode scalars: length normalisation exponent alpha in a small grid (e.g., 3 values) applied to n-best log-probs; `max_new_tokens` fixed at the D11 P99.
4. **MBR over the n-best list (expected-utility decode).** For each row, pick the candidate maximising the sum over other candidates c' of softmax(score(c')) x U(c, c'), with U the exact composite metric (ROUGE-L plus digit overlap with the candidate acting as pseudo-reference). Per-row, uses only the row's own n-best; no cross-row information. On/off is chosen on H with the cross-fit; gain bounded by D13 (the oracle gap).
5. **Number grounding rerank.** Add a penalty lambda (small grid, including 0) per digit sequence in the candidate that does not occur in the row's own article. This is a per-row function of the row's own input and a trained model's n-best. Chosen on H with cross-fit under BOTH numeric conventions (empty = 0 and empty = vacuous 1); pick the setting that is non-negative under both. If the two conventions disagree on sign, choose the setting that is at least neutral under both and raise open question 3.
6. **Calibration protocol.** The grid (alpha x MBR on/off x lambda) is about 12 combinations; selection on half A of H (split by story family), evaluation on half B, swap, report both and the average. Drop any element whose gain is not consistent across both halves. No tuning on the test set or on the public leaderboard.
7. **Hard constraints at decode.** Output is Arabic-script by construction (the model emits only pieces from the Arabic-script train vocabulary; the optional lm_head restriction also enforces it). A guard rejects n-best candidates with no Arabic-block characters (data-determined, not time-determined).
8. **Hierarchical averaging.** The metric is a flat row mean, so selection reports micro-average on H and, offline, per-outlet macro as a robustness view.

---

## Structural signals

- **Lead bias.** Headline content is concentrated in the opening sentences (D5). Source truncation to the head is a structural prior, not a loss of information; the check is the fraction of title words and digits in the first `MAX_SRC` tokens.
- **Copy-with-inflection.** Kazakh is agglutinative; title words are often suffix variants of body words (D8). A subword model with a pretrained copy mechanism handles this; a word-level extractor cannot.
- **Numbers are almost always copied** from the body (D4): the model should learn copy-grounded numerals and the decode check (number grounding) enforces it per row.
- **Wire families.** The same story recurs reworded across publishers (D9). Within train, families supply several differently styled headlines for one body, a free signal for "headline shape" versus "house style". Use as a split constraint (never straddle H/train or the two calibration halves). Do not synthesise blended examples (that would be synthetic data). Recombining real articles/titles is not needed.
- **Orthographic variants** (D3) are copy-consistent within a publisher: if the body uses a given letter variant, the headline usually does too. The model should copy the surface form from the input; hence no normalisation and raw-to-raw training.
- **Dateline/boilerplate** (D7) is style that differs across outlets; teach the model to skip it via diverse outlets and optional lead dropout rather than per-outlet regexes.
- **Invariants not used:** no geometric symmetries, no group constraints on the output beyond Arabic-script.

---

## Experiment roadmap

| Step | Action | Stop / accept criterion |
|---|---|---|
| 1 | Contract, metric and validation correct. Implement the composite metric (both tokenisation variants, both number conventions), unit-test on perfect/reversed/constant/empty/base-rate predictions. Build the LOPO splits with D9 family removal. | All unit tests pass; family removal verified (no title with Jaccard >= 0.6 to any H title remains in train). |
| 2 | Run diagnostics D1-D12 on train. Write the findings back into this plan and update `MAX_SRC`, `MAX_TGT`, normalisation policy, lead-dropout need, H selection. | Plan updated; any D2 failure routes to the tokenizer patch. |
| 3 | Baselines on LOPO (offline): Lead-1, Lead-8 words, Lead-12 words (references only); then a short mT5-small fine-tune, greedy decode. Record per-outlet composite and the D13 n-best oracle. | End-to-end valid CSV; model beats the best Lead-k reference by a clear margin on a majority of held-out outlets (else rethink tokenisation/backbone before anything else). |
| 4 | Compute-matched comparison offline: mT5-small (about 5 epochs) vs mT5-base (about 2 epochs) vs base with `MAX_SRC` 256. Profile time/memory on A10G. | Choose the model/length; freeze constants. |
| 5 | Cross-publisher levers, one change at a time: outlet temperature 0 vs 0.5 vs 1; lead dropout on/off; digit-weighted loss on/off. | Accept only if gain beats the rule (3 of 5 outlets and > 1 SE). Otherwise keep the default. |
| 6 | Decode levers: beam 4 vs 8, length alpha, MBR, number grounding (both conventions). | Accept only if the cross-fitted gain is consistent in both halves; otherwise drop. |
| 7 | Assemble `solution.py` with the fixed plan; run from a clean `working/` using the exact command `python3 solution.py ./dataset/public ./working/submission.csv`; run twice; compare outputs and in-script H scores; check runtime headroom >= 30%. | Run-to-run agreement (near-identical titles or equal H score within noise), validator passes, source < 512 KB. |
| 8 | Credits: baseline (mT5-small or short schedule), best single (mT5-base, greedy-ish), final (with decode levers). The free CSV check is used for format only. | Do not spend credits on tiny tweaks. |

Do NOT start step 6 before steps 1-5 are solid. When two independent solvers would plausibly converge on mT5-class fine-tuning, test it first (step 3) before any proxy-driven micro-tweak.

---

## Compliance audit

CLAUDE.md section 7 and agent-file section B self-audit, against the plan:

- Test file read only for one-sample-at-a-time inference (and schema/row-count for validation and runtime). No stats, vocab, scaler, clustering, dedup, rank-normalisation or pseudo-labelling on test. PASS.
- Wall-clock used only in log lines; no `elapsed()` in conditions, no `timeout`/`time_limit` arguments. PASS (to be enforced by code review before submit).
- No `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks or try/except that changes the model or the work. Fallback model choice (mT5-small) is made offline and hard-coded. PASS.
- Hard-coded constants tuned offline: model size, `MAX_SRC`, epochs, outlet temperature, optional digit-loss weight. These are fixed-plan design choices chosen from LOPO and profiling (not leaderboard-derived), but a reviewer could read them as "tuned offline". Mitigation: use literature defaults where possible (Adafactor 1e-3, alpha 0.5), record the evidence in comments, and keep every decode scalar found in-script on H with cross-fitting. Flagged as open question 4.
- External or synthetic data, self-hosted weights, non-allowed libraries: none; only HF `from_pretrained` for a general-purpose backbone. PASS (verify `transformers`/`sentencepiece` availability).
- Strip-the-ML: removing the fine-tuned seq2seq removes the ability to produce any headline; Lead-k is a logged reference only. The reranking, MBR and number-grounding steps operate on the model's n-best list. PASS.
- Model dominates the pipeline (no thin wrapper over a rule engine); regex use limited to digit extraction for number grounding and whitespace cleanup. PASS (digit extraction is the allowed "deterministic number extraction").
- Explicit bans: no internet lookup; no unmodified mBERT/XLM-R drop-in; no AraT5/Arabic-specific models; no transliteration. PASS under the stated reading; tokenizer reading is flagged.
- Source readable, < 512 KB, no blobs. Comments explain each stage and every fixed constant. Seeds fixed; fixed workers. PASS (to verify at build time).
- Output validity: columns `id,title`, 6,420 rows, ids match, non-empty single-line titles, `to_csv(index=False)`, reload with `keep_default_na=False`. PASS (to verify at build time).
- Sibling leakage: story families are used to build splits and calibration halves; the model never sees test-family copies (removed by the organisers). No cross-row test features. PASS.
- Reporting required by CLAUDE.md before declaring done: per-fold/per-outlet CV, mean +/- std, fixed-work plan and A10G runtime estimate, this audit, validator output.

---

## Open questions & assumptions

**Reviewer questions (with a plan under each reading).**
1. *Pretrained backbones and the tokenizer ban.* Is fine-tuning a general-purpose multilingual seq2seq (mT5-base, with its own SentencePiece plus a train-derived vocabulary patch) acceptable given "off-the-shelf multilingual tokenizer as an unmodified drop-in" and an unlabelled fine-tuning vs from-scratch status?
   - Reading 1 (pretrained OK, patched tokenizer is fine): primary plan as written.
   - Reading 2 (tokenizer must be trained on the provided data, or from-scratch): train a SentencePiece (unigram, identity normalisation, byte fallback) on train text only, then train a small from-scratch encoder-decoder (about 40-60M params) on train pairs; expect a clearly lower score (estimate 0.03-0.08 lower) because 50k pairs and about 30 min of training is little. Cost stated for the "compliant under every reading" choice.
2. *Internal transliteration.* Is an internal Arabic-script to Cyrillic Kazakh mapping (output mapped back to Arabic script) acceptable, given the description says the task is not script transliteration? Default: avoid.
3. *Number precision conventions.* Is NumPrec computed on sets or lists, with `[0-9]+` or Unicode `\d+`, and what is the value when we output no digit sequences but the real title has some? The plan calibrates under both extremes and picks the policy non-negative under both.
4. *Offline-chosen design constants.* Are fixed design choices selected on offline LOPO (model size, source length, outlet temperature) acceptable alongside in-script decode calibration? Default: keep them minimal and literature-standard.
5. *Metric-aware decode.* Is n-best MBR with number grounding acceptable? The description does not ban it; CLAUDE.md allows OOF-tuned post-processing.
6. *Use of the `outlet` column for training-side sampling and held-out selection.* [ASSUME] fine (train only).

**Key assumptions (all [ASSUME]).**
- Hardware and runtime as in CLAUDE.md (A10G 24 GB, <= 1 h).
- ROUGE-L word tokenisation by whitespace after punctuation stripping; digit regex Unicode `\d+`.
- `transformers`, `sentencepiece`, `torch` available in the Kaggle image; HF download of `google/mt5-base` works from the grading host.
- The held-out publisher H (about 4k+ rows) exists among the 5 training outlets.

**What I could not verify.** Every dataset statistic (outlet sizes, title lengths, digit rates, boilerplate, title-in-text leakage, orthographic variants), mT5's behaviour on this script (unk rate, round trip, tokens/word), real A10G throughput and memory, the expected score band, whether the held-out publishers share the training outlets' conventions, and the grader's exact tokenisation and number conventions. The first thing to do once data exists is D2 (tokenizer audit), D6 (title-in-text), D9 (families) and D12 (outlet style distance), because those four can change the architecture, the validation or the compliance story.
