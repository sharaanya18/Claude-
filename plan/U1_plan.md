# U1 Galley restoration: build plan (eris-strategist)

Status of evidence. Read only: the agent file, CLAUDE.md, and the challenge text. No dataset files were available, so nothing under "Data findings" is measured. Every statement about the data is either (V-desc) a fact taken from the description or (UNVERIFIED) a hypothesis plus the diagnostic that would test it. Every number in the runtime table is an estimate to be re-profiled locally before constants are hard-coded.

## Contract & decision unit

**One valid answer.** One CSV row per `target_id` in test.csv (400 rows, 100 galleys x 4). Columns exactly `target_id,prediction`. `prediction` is a JSON string `{"slip_ids":[s1,s2,s3],"tail_id":t}`. The three slips fill GAP_1, GAP_2, GAP_3 in order. All ids come from the same galley record that the row's byte anchor/span points to.

**Valid vs low-scoring.**
- Invalid, whole submission (V-desc): missing or extra target ids, duplicate ids, extra columns.
- Zero for that row: malformed JSON.
- Zero for that choice: an unknown or foreign slip or tail.
- Legal but suboptimal: reusing a slip or tail. The description says twelve distinct slips and four distinct tails "should" be selected. I treat distinctness as a hard decoder constraint. It is not an enforced validity rule, so I would not rely on it for scoring, but it is also how gold is built.

**Metric, term by term (V-desc).** Per passage, score = 0.55 g + 0.15 t + 0.30 e.
- g = (number of gaps whose slip equals the gold slip at that gap) / 3. Each gap is worth 0.1833.
- t = 1 iff the tail is correct (0.15).
- e = 1 iff all 3 slips and the tail are right (0.30, a gated bonus).
- Final score = 100 x unweighted mean of passage scores, clipped to [0.01, 100].
- Rows are scored independently; row order is ignored. A galley mean equals the row mean because each galley has 4 rows. There is no hierarchical weighting beyond that.
- Closed-form reference points: random legal assignment scores about 100 x (0.55/16 + 0.15/4) = 7.19. Perfect slips with a random tail score about 66.3. A perfect tail with random slips scores about 18.4 (the 3.4 from slips is 55 x 1/16).

**True independent unit.**
- Slate/coupling unit: the galley (4 heads, 16 slips, 4 tails). Choices are coupled inside it: 12 of 16 slips are used once, and the tails form a 4x4 bijection with the heads.
- Generalisation unit: the page. The test is 10 unseen pages. Training is 39 pages (25x26 + 14x25 = 1000, checked). Page identity is NOT provided, so I must reconstruct it (see Validation design).

**Pipeline stages, to diagnose separately.**
1. Candidate coverage: trivially 100%. The 16 slips and 4 tails are all enumerated, and no pruning is allowed by the bans anyway.
2. Scoring/ranking: a trained pair scorer for (head, gap k, slip s) and for (head, tail). Diagnostics are per-gap top-1 accuracy, MRR over 16, tail top-1 over 4, and AUC of used-vs-extra.
3. Decoding: a capacity-constrained assignment (12 positions to 16 slips injectively; tails bijective) that maximises expected metric under calibrated marginals.
- Oracle checks: gold one-hot scores through the decoder must give exactly 100. Gold scores with a random decode order must give 100 too. Report per-stage numbers separately.

## Compliance regime

**Domain.** NLP: cross-encoder ranking with structured (assignment) decoding. The challenge is a "fine-tune a neural text model" task on one A10G with a 90-minute end-to-end ceiling.

**Explicit bans taken from the description (hard constraints).**
1. CPU-only ML is ineligible as the main solver.
2. Hand-written rules and heuristic text repair are banned.
3. TF-IDF, BM25, bag-of-words, n-grams, edit distance and fixed-embedding nearest-neighbour matching are ineligible as the main solver.
4. "A token neural training step does not qualify a pipeline whose candidate choices are actually determined by CPU logic or heuristics before or after inference." CPU work may only parse, batch, and assign from scores genuinely produced by the trained GPU model. It may not supply a substantial independent prediction signal.
5. External answer lookup, recovery of withheld evaluation text, and manual labelling are prohibited.
6. Compute: one A10G (24 GB), 90 minutes end to end (prep, training, inference, writing).

**Chosen reading (compliant under every plausible interpretation).** No CPU-computed predictive feature of any kind enters the model or the decode: no lexical overlap, length priors, popularity counts, n-gram or character statistics. All evidence comes from a fine-tuned transformer scoring GPU-side. The only CPU logic is (a) parsing and batching, (b) the capacity-constrained assignment on the neural score matrix (explicitly allowed), and (c) a handful of decode scalars fitted on out-of-fold neural scores. I would compute (c) in torch on GPU where practical.

**Where the description conflicts with CLAUDE.md / the agent guidebook (the description wins, per CLAUDE.md 2.4 and agent file section 0).**
- Agent file F2.9, F2.12 and CLAUDE.md 2.5 treat hand-built lexical or popularity features fed to a learned scorer as grey/acceptable. Here they are dropped completely. Cost: probably a few points of gap accuracy that a GBDT-over-features stacker could have added. I accept that cost.
- Agent file F2.12 recommends listwise GBDT (LambdaRank) as the strong default. That is CPU-only ML and is excluded here as the solver. GBDT stacking over neural scores is also excluded because it would add an independent CPU signal.
- Runtime: the description states 90 minutes. CLAUDE.md Section 1 targets at most 50 minutes and plans for at most 60. I adopt a nominal 45 minutes with a 60-minute design cap, with 90 as the true ceiling.
- CLAUDE.md Section 5 says IDs "in the same order" as the sample; the description says row order is ignored. I follow the stricter one: write in sample order.
- CLAUDE.md's guidance that a trained model plus rules or TF-IDF is "grey" does not apply; the description makes it a ban.

**Other CLAUDE.md rules that apply unchanged.**
- HF/timm weights only, with a pinned revision.
- Fixed work plan; no wall-clock branching.
- No `torch.cuda.is_available()` switches.
- Seeds fixed.
- Fit every tokenizer/encoder/vocabulary on train only (here: use the pretrained tokenizer unchanged and add no test-fitted state).
- The source file must be under 512 KB, plain UTF-8.

**Ambiguities that change the compliant approach (reviewer questions listed in the last section).**
- Whether in-galley joint decoding counts as a forbidden "whole-test aggregation". Under the description's own text it does not: it is a capacity-constrained decoder per galley, using only that galley's record. No statistic crosses galleys.
- Whether page-proxy grouping by 8-gram union-find (CPU, train-only, used only to build validation folds) is acceptable. I believe so, since it supplies no prediction signal.

## Data findings

All items below are UNVERIFIED unless marked V-desc. Diagnostics are to be run on TRAIN only. For test I would take only schema, row count (400), id format and size statistics (JSON line lengths, template/slip/tail token counts) for runtime planning.

**Facts from the description (V-desc).**
- Train: 1000 galleys on 39 pages, 4000 rows. Test: 100 galleys on 10 other pages, 400 rows.
- Per galley: 4 heads, 16 slips, 4 tails; 12 slips used, 4 extras; 4 tails bijective with heads.
- Text only, noisy period prose, irregular spelling and punctuation.
- The words adjacent to each gap are removed (`<VEIL>`), as is the attachment point for the ending.
- Test pages are screened against training pages by 8-token overlap (passages and extra-slip windows). Repeated head templates are excluded across partitions. Evaluation source windows do not overlap each other.
- Pages with heavy repetition are kept in a single partition, so near-duplicate passages exist inside train pages and inside test pages.

**Diagnostics to run, each with the hypothesis and what it decides.**
1. Parse and contract: every `galleys.jsonl` line parses; each row's `(byte_anchor, byte_span)` returns exactly one line; the contract fields match; `heads[i].target_id` maps one-to-one onto train.csv and train_targets.csv. Check that every `train_targets` slip id and tail id belongs to the same galley, that the 12 gold slips are distinct, and that the 4 gold tails are distinct. Hypothesis: all hold. If any fail, the decoder's capacity constraints need revising.
2. Target structure: for each train galley, which 4 slips are extras (derive from gold). Hypothesis: extras are drawn from windows near the same source documents, or are far enough that a trained model separates them well. The AUC of a "used vs extra" probe from the trained scorer will be informative. Record per-galley extras count (expect exactly 4).
3. Size statistics. Token counts (with the candidate backbone's tokenizer) for template, template segments before GAP_1, between gaps, after GAP_3; slips; tails; the number of `<VEIL>` tokens per gap and per template. Hypothesis: templates are about a few hundred tokens, slips tens of tokens. This sets MAX_LEN and the sampled-negative group size G. If the template exceeds 512 tokens, windows around each gap are needed, and the left/right context budget becomes the main information-loss risk.
4. Tokeniser noise: unk/unknown-byte rate and subword fragmentation on train text for deberta-v3-base versus roberta-base (byte-level BPE never produces unk). Hypothesis: irregular spellings fragment but are tolerable; if deberta unk is above about 0.5% of tokens, roberta-family is the primary.
5. Artefact audit for non-answers: (a) does array position of a slip/tail/head in the record predict its role? Expect chance (gap accuracy near 1/16 for any position-based rule). The description says no. If not chance, record it and make sure the model never sees positions or ids, and that arrays are shuffled during training. (b) Does slip length alone separate used from extra, or gap 1 from gap 3? Compute the AUC of length alone. This is a diagnostic of what the neural model could exploit; it is not a feature. (c) Does gold-slip length distribution differ by gap index?
6. Group structure (critical for validation): union-find over train galleys that share any 8-token window across heads/slips/tails (the organiser's own screen), plus exact-duplicate head templates and exact-duplicate slip texts. Hypothesis: this recovers roughly the 39 pages, or components that are unions of pages if different pages share boilerplate. Cross-check against contiguous file-order blocks of 25/26 galleys: compare the rate of shared 8-grams within a block versus across blocks. If within-block is much larger, the file order carries page structure, and blocks of 25/26 are a usable secondary grouping. Expected failure mode: if overlaps are sparse, the union-find gives near-singleton components and under-merges, which makes random-ish folds optimistic. Then the block grouping is the conservative choice.
7. Repetition inside pages: count galleys whose head template (or whose long visible segment) repeats within a page with a different gold assignment; count slip texts that appear as gold in one galley and as an extra in another. This quantifies the memorisation risk and the leak of a random split.
8. Ambiguity ceiling: oracle with a naive best rule is not allowed to ship. As a diagnostic only, run an inference-only masked-LM pseudo-likelihood scorer (frozen, GPU, not shipped) to obtain a zero-shot floor under the same held-out folds. Expected, labelled unverified: well above 1/16 for gap accuracy because slips are drawn from the same source windows, but far from solved because adjacent words are veiled. Use only as the yardstick the fine-tune must beat.
9. Information-ceiling bounds from the metric alone (no data needed): slips-perfect and tail-random is about 66; tail-perfect and slips-random is about 18; random legal assignment about 7.2. Verify the 7.2 by Monte Carlo over train galleys as a metric unit test.
10. Tail geometry: how many tokens of the end of the head and the start of the tail are likely informative (the tail "begins after the concealed attachment point"). Hypothesis: the end segment after GAP_3 plus the first 64-128 tokens of the tail carry the match; much of the middle of the head is irrelevant for tails.

**Hypotheses about where the signal lives (UNVERIFIED).**
- Slip to head: topical, named-entity and stylistic compatibility with distant context, since the adjacent words are veiled.
- Gap order: slips for gaps 1, 2, 3 come from a contiguous window in order, so they have positional and slip-to-slip (adjacency) cues.
- Within a galley the four passages may come from the same page, which makes the other-passage slips the hard negatives.
- Extras come from other windows; they are separable by "belongs to no passage here" evidence.
- Tail to head: continuation of the sentence or topic at the end of the passage; 24 permutations per galley.

## Validation design

**How the test split was made (V-desc, from the description).** Whole pages are held out (10 pages in test versus 39 in train), with a screen against 8-token overlap and repeated templates. So the generalisation shift is "unseen pages, no copy-able overlap, same kind of distractors".

**Primary scheme.** In-script 3-fold GroupKFold over page-proxy groups. Fold assignment uses the union-find component (diagnostic 6), optionally merged with file-order blocks if diagnostic 6 shows blocks coincide with pages. About 13 pages per fold. Each fold model trains on about 667 galleys. Out-of-fold scores cover all 1000 train galleys and drive (a) the reported in-script CV and (b) the decode scalars. Fold models' test scores are averaged in logit space. Training is on whole galleys only, so the 4 passages and 16 slips of a galley are never split across folds.

**Development-time scheme (offline, not in the script).** 5-fold GroupKFold repeated over 2 group-to-fold assignment seeds, used for paired comparisons of design choices (backbone, MAX_LEN, G, Sinkhorn on/off). Accept a change only if it wins on the same folds and seeds by more than one standard error of fold means, or consistently across folds and both seeds. Report mean +/- std of the exact metric, plus component metrics: per-gap top-1, tail top-1, e rate, and per-gap-index accuracy for gaps 1/2/3.

**Metric unit tests (exact re-implementation, 0.55 g + 0.15 t + 0.30 e, mean x 100, clip [0.01, 100]).**
- Perfect gold gives 100.
- Reversed slip order within each passage gives g close to 0 for distinct slips.
- A constant answer (same slip three times, same tail for all) gives at most one correct gap and a quarter of tails.
- A random legal assignment (Monte Carlo over train galleys) gives about 7.2 with std about a point.
- Malformed JSON gives 0 for the row; a foreign id gives 0 for that choice.
- Test the row-independence property: scoring a board of a subset of rows equals scoring the same rows in the full set.
- Verify position-wise credit and the "reuse cannot create another correct placement" rule on a hand-built case.

**Post-hoc selection needs its own held-out check.** The decode scalars (temperature tau, three per-gap-index offsets, one tail temperature, and any route-bonus weight) are fitted on OOF scores. I would report their effect cross-fitted: fit on two folds' OOF, evaluate on the third, rotating. Training hyperparameters are fixed constants chosen by the development scheme, not by any in-script search (see risks and open questions).

**Direction and size of bias of each proxy (estimates, UNVERIFIED).**
- If page-proxy recovery works (diagnostic 6 near 39 components), grouped 3-fold is mildly pessimistic by a few points. Fold models see 67% of the data versus the 100% a full refit would see, and the shipped test prediction averages 3 models.
- If the union-find under-merges (components much more numerous than pages), the folds leak same-page vocabulary and repeated passages. The proxy is then optimistic, potentially by 10 or more points. This is the main way validation could mislead, and why the block-order cross-check is mandatory.
- The test screen removes any training overlap of 8 or more tokens, so a split that leaves 8-gram overlap across folds is easier than the real test. Check cross-fold 8-gram overlap is near zero after grouping.
- My self-built proxy has the same negative distribution as test (the same galley construction), so there is no extra distractor-difficulty shift to correct for.

**Expected private band (an estimate, not a promise; low confidence, no data seen).** Reasoning: random is 7.2; an illustrative model with about 60% per-gap accuracy and 70% tail accuracy lands near 50 (g 0.6 gives 33, t 0.7 gives 10.5, a correlated e of about 0.2 to 0.25 gives 6 to 7.5). The veil explicitly defeats local grammar, so I would not assume high accuracy. Stated band: roughly 30 to 70, with a central guess in the mid-40s to 50s. Real value depends heavily on the unmeasured separability of slips within a galley.

## Overfit/underfit risks

**Overfit risks and mitigations.**
1. Memorising pages: only 39 independent pages. Near-duplicate passages sit inside pages, so a random split would show inflated scores. Mitigation: page-proxy grouping; each epoch log train loss versus held-out score in development to catch memorisation (loss falls, held-out plateaus); fixed short schedule (2 epochs), low LR with decay, dropout on.
2. Capacity ladder (agent F2.1). Frozen strong features, then a linear head, then partial fine-tune, then full fine-tune. Here there are about 12k gap positives and 4k tail positives across about 667 galleys per fold. That is more than the "few hundred rows" regime, but the independent-unit count is only dozens of pages. The compliance regime demands genuine fine-tuning, so the shipped model is a full fine-tune of a base-size encoder with conservative LR (about 2e-5, layer-wise decay) and 2 epochs. I would measure the frozen-features-plus-head and zero-shot rungs under the same folds as the floor and log the parameter-change norm and the held-out gain over the probe. If full fine-tune does not beat the linear probe by more than noise, fall back to the last-layers-only rung (LP-FT) and keep compliance by having genuinely trained layers.
3. Selection on OOF: few decode scalars only (about 5), cross-fitted. No blend weights in the primary.
4. Hard-coded training constants (LR, epochs, MAX_LEN, G) picked in development: noted as the audit risk. Mitigation: choose conventional values from a coarse, pre-declared grid evaluated by the paired development scheme; record all.
5. Position or ID artefacts: array shuffle each epoch; never feed ids; confirm via diagnostic 5.

**Underfit risks and mitigations.**
1. Context truncation: the evidence lives in distant words. A MAX_LEN that cuts the template loses it. Mitigation: window the template around each gap with a left/right budget set from diagnostic 3 and keep the other gap markers visible; consider larger windows before bigger models.
2. Under-sized backbone: base-size is chosen for budget. Roadmap step tests a large backbone with reduced G; keep only if the gain beats noise.
3. Loss and metric mismatch: row-wise softmax ignores exclusivity across passages. Mitigation: Sinkhorn marginals plus a capacity-constrained assignment at decode; the tail is trained with an exact 24-permutation marginal likelihood.
4. Sampled negatives (G of 15) vs. all 15 at test. Mitigation: re-sample negatives each epoch uniformly from the same-galley slips so the training negative distribution equals the test one; verify G=6 versus G=15 on a small development run.
5. Noisy spelling and fragmentation: tokenizer diagnostic 4 decides the backbone family.
6. Gap-position blindness: if the model cannot tell gap 1 from gap 3 it fails to separate same-passage slips. Keep the position marker `<GAP_k>` in the text, with the other gaps shown as blanks.

## Recommended approach (primary + fallback)

**Primary: two trained cross-encoders (slip scorer, tail scorer) plus a Sinkhorn/assignment decoder.**

1. Slip scorer (neural, GPU).
   - Input for (head h, gap k, candidate slip s): `[CLS] left_context(gap k) <GAP_k> right_context(gap k) [SEP] slip text [SEP]`. The veil tokens are kept as a single marker (e.g. the model's mask token or one added token) and are never filled. Other gaps appear as visible `<GAP_j>` blanks so position is learnable. No ids, no array positions.
   - Backbone: deberta-v3-base, or roberta-base if the tokenizer diagnostic says so; revision pinned. bf16 autocast, fixed device `cuda`.
   - Head: one linear layer on pooled output (CLS plus mean of tokens) giving a scalar score.
   - Loss: per (head, gap) softmax cross-entropy over a sampled group of G candidates: the gold slip plus G-1 negatives re-sampled each epoch uniformly from the 15 other slips in the same galley. This matches the test negatives (same-passage other-gap slips, other-passage slips, extras).
   - Training: AdamW, LR about 2e-5, layer-wise decay about 0.9, linear warmup/decay, grad clip 1.0, 2 epochs, fixed seeds, fixed batch. Fixed schedule; no validation-triggered stopping.
   - Context-window jitter augmentation (random shrink of left/right context) and per-epoch shuffle of slip order.
2. Tail scorer (neural, GPU).
   - Input for (head h, tail j): the head's end segment (from near GAP_3 through the veiled attachment point) plus the first T tokens of the tail. Cross-encoder with the same backbone family.
   - Loss: exact marginal likelihood over the 4! = 24 head-to-tail permutations per galley. Score all 16 (head, tail) pairs of a galley, take the softmax over permutations of the summed scores (exact, 24 terms). This builds the bijection into the model rather than leaving it as a post-hoc decode.
3. Decode (per galley, from neural scores only).
   - Slips: the score matrix S (12 gap-positions x 16 slips). Divide by a calibrated temperature (plus a per-gap-index offset), append 4 "unused" dummy rows (capacity: 12 positions use exactly 12 slips, 4 slips unused), run Sinkhorn normalisation in torch with a fixed iteration count, then take the marginal matrix M. Assign slips by a linear assignment (maximise the sum of M, which is the expected g).
   - Tails: exact enumeration over 24 permutations, picking the one maximising the sum of marginals (the expected t).
   - Hard checks: 12 distinct slips from this galley, 4 distinct tails from this galley.
4. Training/inference organisation. 3-fold cross-fit over page-proxy groups. Fold models produce OOF scores for the decode scalars and the in-script CV; test scores are the average of the 3 fold models in logit space. A full-data refit is not planned because of the budget (see the fixed work plan); it enters the roadmap if profiling shows headroom, with decode scalars reused only after checking logit-scale transfer.

Why this fits: it uses the description's own recommendation (neural long-range comparison, capacity-constrained decoder). It encodes all three structural facts (injective slips with 4 unused, tail bijection, gap order) with few learned parameters and zero CPU-side signal. The strip-the-ML test is clean: remove the cross-encoders and the decoder receives constant scores and degrades to about 7 points.

**Fallback (if the profile overshoots the budget or the cross-encoder under-trains).** A fine-tuned dual-tower scorer with the same decode and tail enumeration. Context-with-gap and slip are encoded once each per galley (12 + 16 encodings instead of 192 pairs), trained with the galley-level softmax over all 16 slips and the exact column softmax, which allows a larger backbone at the same cost. It is cheaper and exposes exact two-way competition, but it is weaker at fine-grained matching. It is a trained fine-tuned encoder, not a fixed-embedding lookup, but a reviewer might read it as "embedding nearest-neighbour". The mitigation is that both towers are fine-tuned and the score is learned end to end. Second fallback (budget only): shrink MAX_LEN and G, never the number of folds.

## Rejected options

- TF-IDF, BM25, bag-of-words, n-gram, edit-distance or lexical-overlap scoring or priors, and any CPU feature stacked into a GBDT or logistic model. Banned as main solver or as an independent CPU signal.
- Fixed-embedding nearest-neighbour matching (a frozen sentence encoder plus cosine). Banned as the main solver.
- Zero-shot MLM or LLM pseudo-likelihood used as the shipped scorer: inference-only, no training, against the challenge's intent and CLAUDE.md 2.1. Kept only as an offline diagnostic floor.
- LoRA fine-tune of a multi-billion-parameter causal LM scoring all 192 pairs per galley: too slow for 90 minutes with a safe margin, with 12k positives times sampled negatives. A small decoder LM could be a later diversity arm only if the profile shows slack.
- Joint scoring of all 3360 ordered slip triples per head (16 x 15 x 14): wasteful. The independence-based decode through marginals plus an optional route-bonus refinement is leaner.
- Hand-written rules on punctuation, capitalisation or slip length to prune candidates or to repair text: banned.
- Using array positions, file order, ids or galley sequence: the description says no answer meaning. Used only as a diagnostic.
- Pseudo-labels, test-time clustering, or fitting anything on test galleys: banned.
- Wall-clock-based schedule tricks, environment fallbacks: banned by CLAUDE.md Section 3.
- Large ensembles of seeds as the primary: first the lean design must be measured; extra arms are only roadmap items.

## Fixed work plan & runtime budget

**Fixed constants (to be set after local profiling, then hard-coded; the values below are the starting plan).**
- Backbone: deberta-v3-base (or roberta-base per diagnostic 4), pinned revision, bf16 autocast, `device = "cuda"`.
- MAX_LEN about 320 tokens (slip up to about 40, context about 280), to be reset from diagnostic 3.
- G = 6 sampled candidates per gap instance; 2 epochs for the slip scorer; tail scorer 2 epochs, 16 pairs per galley per step.
- 3 folds, 1 seed for training, global seed fixed, `PYTHONHASHSEED`, torch/cuda/numpy/random seeds, DataLoader Generator, `num_workers` fixed, `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`.
- Sinkhorn: 50 fixed iterations. Decode scalar fit: fixed small grid or a few-step torch optimisation on OOF, with a fixed number of steps.

**Runtime estimate on one A10G (ESTIMATES; every throughput is an assumption to profile).**

| Stage | Work | Est. time |
|---|---|---|
| Setup | parse 1100 galleys, derive page-proxy groups (train only), download backbone, tokenise on the fly | 3 min |
| Slip scorer training | 3 folds x 8,000 gap instances x G=6 x 2 epochs = 288k sequences at about 180 seq/s | about 27 min |
| Tail scorer training | 3 folds x 667 galleys x 16 pairs x 2 epochs = about 64k sequences at about 200 seq/s | about 5 min |
| OOF scoring | slips 192k pairs + tails 16k pairs at about 550 seq/s | about 6 min |
| Test scoring | 3 models x (19,200 slip pairs + 1,600 tail pairs) at about 550 seq/s | about 2 min |
| Decode/calibrate/validate/write | Sinkhorn, assignment, cross-fitted scalars | about 1 min |
| Total | | about 44 min nominal |

Headroom: about 27% against my 60-minute design cap, about 51% against the 90-minute ceiling. Worst case at 1.3x slower is about 57 minutes. If profiling shows more than 45 minutes, reduce G first, then MAX_LEN, then slip-scorer epochs; never the fold count, and never by elapsed time.

**Memory.** A base-size encoder at batch 32 x 320 tokens in bf16 needs on the order of 10 to 14 GB, which fits in 24 GB. Free the GPU between folds. Dev note: bf16 on both dev and grading hardware; if the dev GPU lacks bf16, the dev run uses a documented constant that is reset before submitting.

**Validation inside the script (CLAUDE.md Section 5).**
- Check schema and ids against `sample_submission.csv` and the galley records, row count 400, no duplicates, every prediction parses as JSON with exactly 3 slip ids and one tail id, all from the right galley, 12 distinct slips and 4 distinct tails per galley, finite values.
- Re-read the written CSV with `keep_default_na=False`.
- No constant fallback prediction is written early. If a stage fails, fail loudly.
- Log every stage, with elapsed time used for logging only.

## Metric-aware training & decode

1. **Back-solved structure.** Expected passage score = (0.55/3) x sum of the three slip marginals + 0.15 x tail marginal + 0.30 x P(route entirely correct). The slip term is linear in correct gaps, so maximising it under the exclusivity constraint is a linear assignment on calibrated marginals, not on raw argmax or log-probabilities. The tail term is the same for the 4x4 problem.
2. **Training losses match the structure.** Slip: listwise softmax over same-galley candidates, with negatives drawn from the actual test negative distribution. Tail: exact marginal likelihood over the 24 permutations, which is the exact structure of the bijection. No label smoothing (small data, and calibration matters).
3. **Calibration (per slot).** Fit by out-of-fold log-loss of the Sinkhorn marginals: a shared temperature tau, three per-gap-index offsets (gap 1/2/3), one tail temperature, and the "unused" dummy bias. About 5 scalars, found by a fixed-step torch optimisation on OOF scores and reported cross-fitted. Initialise biases at the base rate (12/16 used).
4. **Capacity-constrained decode.** Sinkhorn with 4 dummy rows gives approximate marginals consistent with the constraints. The assignment is on marginals M. Exact enumeration for tails (24 permutations).
5. **Exact-route bonus (second-order, roadmap step, not primary).** The 0.30 bonus is only reached when all four choices of a passage are right, and it gates on the weakest marginal. A first-order refinement is coordinate ascent over passages maximising the sum of (0.55/3) x sum of slip marginals + 0.15 x tail marginal + 0.30 x (product of the four marginals), starting from the assignment above. Keep it only if it improves the cross-fitted OOF score by more than noise.
6. **Hierarchical averaging.** Every passage has equal weight; no per-galley weighting is needed. Loss weighting: uniform per gap instance.
7. **Hard constraints inside the valid space.** Distinct slips, distinct tails, ids from the right galley. Combined with model evidence; a pure constraint search is not a predictor, and the stripped decoder is uninformative.

## Structural signals

Each invariant, and how it is used (always through a learned model or an exact constraint, never a hand-written rule):
1. **Gap order.** Slips for GAP_1 to GAP_3 are in textual order within one window (UNVERIFIED). Use: keep the `<GAP_k>` marker, so the scorer learns position; add a position-aware calibration offset (3 scalars). Optional roadmap: a slip-to-slip adjacency potential (a small learned scorer over pairs of slips for neighbouring gaps) after the lean primary is measured.
2. **Injectivity.** Each slip is used at most once (12 of 16 used, 4 unused). Use: dummy "unused" rows in Sinkhorn and the linear assignment.
3. **Tail bijection.** Four heads, four tails, one each. Use: exact 24-permutation likelihood in training and exact enumeration at decode.
4. **Extras from other windows.** The "belongs nowhere in this galley" evidence is learned through the column competition in Sinkhorn and the unused-slot bias.
5. **Interchangeability of ids and array order.** Use: shuffle slip, tail and head order each epoch; never give ids or positions to the model.
6. **Competing passages as hard negatives.** The 3 same-passage slips for other gaps and the 8 slips of other passages are natural hard negatives; the sampling keeps exactly the test distribution.
7. **Tail depends on the end segment only.** The tail scorer reads the end of the head and the start of the tail, not the full head. This is a bias that puts the model's capacity where the evidence is (agent F2.5).
8. **Optional stage-2 conditioning (roadmap).** Fill the other gaps with stage-1 predicted slips (from OOF predictions, to avoid train/test mismatch) so the model sees coherent context. Add only if it beats the lean design by more than noise.
9. **Augmentation.** Context-window jitter and candidate re-sampling per epoch are augmentations of real units only. No synthetic text is created.
10. **Pages as groups.** Repeated passages inside pages: group-aware folds and group-aware negative sampling (do not train on a "negative" that is a near-duplicate of the gold slip from a repeated passage). Check diagnostic 7 first.

## Experiment roadmap

1. **Contract, metric, validation (stop when all unit tests pass).** Parse galleys by byte anchor; exact metric with the unit tests listed; page-proxy groups and the cross-fold 8-gram check; oracle (gold scores through the Sinkhorn/assignment decoder must give exactly 100 and its runtime recorded). Report random-assignment Monte Carlo of about 7.2.
2. **Cheap end-to-end baseline (stop when valid CSV and a CV number exist).** A small cross-encoder (e.g. a base-size backbone with short MAX_LEN, 1 epoch, G=4), row softmax plus linear assignment, tails by pairwise argmax then assignment. Validate the CSV with the CLAUDE.md validator. This is the first credit candidate.
3. **Representation and structural insight.** Paired comparisons on the same folds and 2 split seeds: MAX_LEN (256, 320, 448); gap-position markers on/off; backbone family (deberta-v3-base vs roberta-base); frozen-feature probe and zero-shot MLM floor as yardsticks; large backbone with reduced G. Stop when a step no longer beats noise.
4. **Metric-aware training and decode.** Row-softmax versus Sinkhorn marginals plus capacity assignment; 24-permutation tail likelihood versus pairwise tail; cross-fitted calibration scalars. Then test the exact-route bonus refinement.
5. **Diversity.** Only after the lean design is measured: stage-2 conditioning with OOF-filled gaps; a second backbone family if within about 2 points of the first, averaged in z-scored logits; the fine-tuned dual-tower as a diversity arm only if it adds. Each member enters only if it earns a gain over noise.
6. **In-script bounded HPO.** Limited to the decode scalars (fixed-step optimisation on OOF, cross-fitted). Training hyperparameters stay fixed constants chosen in development; see the open question about whether that is acceptable.
7. **Final fixed-plan run.** Run the whole script twice on a clean working directory and diff predictions (expect near-identical; large swings mean too few folds or seeds). Check the CSV prediction distribution against OOF (12 distinct slips per galley, tail bijection). Spend credits only on: baseline, best single, ensemble, final.

Process rule: when two or three independent solvers would plausibly converge on a technique (here: windowed cross-encoder plus assignment decode), test it before any proxy-driven tweak.

## Compliance audit

CLAUDE.md Section 7 and the agent file's self-audits, against this plan (the plan, not the implementation):
- Test file read only for one-sample (one-galley) inference: yes. Test galleys are tokenised and scored independently; nothing is fitted on them. The decode uses only the same galley's record. Page-proxy grouping uses train galleys only.
- Elapsed-time conditions: none; time is for logging only.
- `cuda.is_available()`, `cpu_count()`, import fallbacks: none; `device="cuda"` and fixed counts.
- Hard-coded constants tuned offline: RISK. Training constants (LR, epochs, MAX_LEN, G) are chosen in development, not found by an in-script search. The decode scalars are found in-script from OOF. Mitigation: use conventional values from a coarse pre-declared grid; reviewer question below.
- External data, synthetic data, pre-trained weights from a non-HF source: none. HF backbone only, pinned revision.
- Strip-the-ML test: remove the neural scorers and the decoder sees a flat score matrix, so it outputs arbitrary assignments (about 7 points). No rule, lookup or CPU feature carries a signal. Passes.
- No whole-test aggregation: no statistic crosses galleys (no normalisation across the test file, no prior estimation from test lists, no vocabulary or encoder fit on test).
- Sibling leakage: the scorer inputs are the head and the candidate (pairwise). Competition among slips and tails is intrinsic to the task and explicitly intended by the description ("the other slips, and the competing passages must be read together"). The risk is noted for the reviewer, not a rule violation.
- CPU logic before or after inference supplying a substantial independent signal: none. The assignment acts on neural marginals (explicitly allowed). Calibration scalars are fitted on neural scores.
- Source readable and under 512 KB, no blobs: planned.
- Challenge-specific restrictions honoured: no TF-IDF/BM25/n-gram/edit-distance/rules/fixed-embedding NN; no external lookup; one A10G; at most 90 minutes with the nominal 44-minute plan.
- Determinism risk: bf16 and atomic-add kernels are not bit-exact. Mitigations: calibrated scalars are robust to small noise; run twice and diff.
- Page-proxy grouping uses 8-gram overlap, a CPU n-gram computation. It is used only to build validation folds, never as a feature or a decode input. Stated explicitly in a code comment and in the open questions.

## Open questions & assumptions

**Questions for a reviewer (each with plans under both readings).**
1. Is building validation folds with an 8-token-overlap union-find (train only) acceptable, given that n-grams are banned as the solver? Plan if yes: as written. Plan if no: use contiguous file-order blocks of 25/26 galleys as page proxies and accept the risk that block structure is imperfect (and say so).
2. Are conventional, development-chosen training constants (LR, epochs, MAX_LEN, G, MAX folds) acceptable, versus a strict in-script search? Plan if yes: fixed. Plan if no: shrink to a three-point in-script search over LR on fold 0 with a fixed trial count, at the cost of about 5 to 8 minutes of budget.
3. Does the within-galley Sinkhorn/assignment decode over neural marginals count as a "capacity-constrained decoder" and not as a banned CPU signal? Description wording says yes. Fallback reading: take the argmax per gap independently and fix only duplicate conflicts; I expect this to cost several points.
4. May a lexical signal (lengths, overlap counts) be given to the neural model as an auxiliary input? The plan assumes no, and drops it, losing perhaps a few points.
5. Pre-trained backbones may have seen public-domain period texts. This is inherent to any allowed pretrained weights and is not "external lookup", but I flag it.
6. The description says slips should be distinct and tails distinct, but this is not an enforced validity rule. Assumption: obey it.

**Assumptions (UNVERIFIED).**
- galleys.jsonl contains both the train and the evaluation galleys, and the byte anchor/span of each row addresses one complete JSON line of its galley.
- Token lengths allow a window of about 320 tokens to retain the informative distant context; if not, windows grow and G or epochs shrink.
- A base-size backbone is enough to beat the zero-shot floor under grouped folds; if not, test the large backbone with reduced G.
- Throughput numbers (180 seq/s training, 550 seq/s scoring) are guesses to be profiled.
- The expected private band of 30 to 70 is an estimate with no data behind it.

**What I could not verify.** Everything under "Data findings" and all runtime numbers; the number and shape of page-proxy components; whether slips separate by topic, position or length; tokenizer behaviour on noisy spellings; whether the 4 passages of a galley share a page; how large the gap between OOF and test will be.
