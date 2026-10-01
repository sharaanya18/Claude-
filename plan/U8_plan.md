# U8 - Code-switched speech linking: build plan

Status of evidence: no dataset was available to the strategist. Everything under "Data findings" is a diagnostic to RUN and a hypothesis, all marked UNVERIFIED. Numbers quoted from the description are marked (desc). Numbers I derive are estimates. Only three files were read: the agent file, CLAUDE.md and the challenge text.

## Contract & decision unit

**One valid answer.** One CSV row per test query (1,791 rows + header): `query_id, ranked_paragraph_ids`, where the second cell is the FULL list of that query's own candidate ids (27 to 71 ids, median 50 (desc)), space-separated, most likely first. Row order is free.

**Invalid vs low-scoring.**
- Invalid, scores zero or is rejected: duplicated `query_id`, missing column, no shared `query_id`.
- Zero for that row: an empty cell, or a list that omits the correct paragraph.
- Silently ignored: ids that are not candidates of that query. A repeated id counts at its first position only.
- Low-scoring but valid: any permutation of the candidates.
- Therefore always emit every candidate exactly once. The script asserts set-equality with the query's candidate list and re-reads the file.

**Metric.** Mean reciprocal rank over queries, each query weighted equally (no per-gallery averaging). Exactly one correct candidate per query, so RR = 1/rank of that candidate.
- Reference values (desc): random 0.0913, best frozen public encoder 0.2701, char 3-5-gram TF-IDF 0.2067, word TF-IDF 0.1305.
- Reference values (desc): length-closeness 0.0963 and candidate frequency 0.0945 are text-ignoring and close to random. Candidate frequency is banned as a signal anyway.
- The metric is rank-only and MRR-optimal ordering for a single relevant item is "sort by posterior probability of being the answer, descending". No threshold and no cross-query calibration is needed. Only the WITHIN-query order matters.
- Ties: the evaluator's tie rule is not stated. Use a pessimistic tie rule in the local metric (rank = 1 + number of candidates with score >= the correct one, excluding itself) and make exact ties practically impossible (see decode).

**True independent unit.**
- The unit is the gallery (a two-week window of sittings), not the query. Queries of one gallery share all candidates and the same political week.
- Speeches can recur across overlapping galleries. The description gives a hint: a "candidate frequency across galleries" baseline exists, so candidates repeat across galleries.
- Effective sample size is the number of galleries or time blocks, not 4,287 queries. I estimate roughly 100 train galleries: 4,287 queries / (about 0.8 x 50 speeches per gallery). UNVERIFIED, to be measured.

**Pipeline stages and how each is diagnosed separately.**
1. Candidate coverage: trivially 100%, because the gallery is given. Assert on every train query that the label is in its candidate list.
2. Scoring and ranking: the entire problem. Report MRR, recall@1/5/10 and MRR by gallery size.
3. Decode: only a sort. It is validated by the gold-score oracle (feed the label as score 1 and the rest as 0 through the sort and the writer: MRR must be exactly 1.0, with the row format and set equality verified).

## Compliance regime

**Domain.** Header says RAG, Medium, A10G, tag text. CLAUDE.md 6.4 applies: a frozen off-the-shelf embedder for retrieval is fine, but the ranking layer must be genuinely trained. A fully frozen pipeline is not allowed. Hand-features plus LambdaMART alone is not enough. Fine-tuning the encoder is recommended. The challenge text explicitly permits fine-tuning a public multilingual encoder "provided the fitting genuinely uses the supplied training queries and answers".

**Explicit bans from the description (hard constraints; every one is honoured).**

| # | Ban (desc) | Consequence for the plan |
|---|---|---|
| B1 | Hard-coding rankings for specific evaluation queries | No per-query rules. Ranking comes only from the trained scorer. |
| B2 | Using `query_id` / `paragraph_id` / `gallery_id` strings as signal | Ids are only join keys. Never fed to a model, never sorted on, never hashed for tie-breaks. Gallery membership (which candidates belong to which query) is dataset structure, not an id-string signal. |
| B3 | Row order, or the order of ids inside `candidate_paragraph_ids` | The candidate list is converted to an unordered set. Training batches are drawn with a seeded shuffle independent of list position. Ties are broken by content, not position. Do not use "supplied order" in any way, including as a stable-sort tie-break. |
| B4 | "How often a candidate appears across evaluation galleries" (gallery or frequency signals) | No count, set-membership or dedup statistic computed over `test.csv` galleries. Embeddings are computed per paragraph text, and any memoisation is pure caching, not a feature. No candidate-popularity feature from eval. Per-candidate popularity from TRAIN is also avoided: the description itself shows such a baseline is a text-ignoring trick, and train popularity would not transfer across terms. |
| B5 | Reasoning across evaluation queries: using "each candidate answers at most one query", reallocating candidates between queries of a gallery (so no Hungarian, no mutual-nearest-neighbour, no dual softmax over queries, no column normalisation over queries, no clustering of test Spanish queries) | Each ranking is a function of its own query text, its own candidate texts, and a model fitted on train. Nothing computed from other test queries. The same candidate-tensor trick is also forbidden in the form "subtract how well this candidate matches the other queries". |
| B6 | Outside copies of the proceedings, search services, external translation service or API, private / role-gated / API-key models, external inference APIs, non-reproducible weights | Only ungated public Hugging Face weights, loaded inside the script. No gated models (e.g. Llama / Gemma class). No MT. |

**Compliance triage notes (CLAUDE.md F2 #9).**
- Within-row relative features are allowed: z-score or margin among the query's OWN candidates. "Its own candidates" is explicitly allowed by the description.
- Statistics fitted on TRAIN text are allowed. Statistics from test queries or test candidates are not. CLAUDE.md 2.3 #5 also forbids fitting TF-IDF / vocab / PCA on test or train+test.
- A hubness reference bank built from TRAIN Spanish query paragraphs (CSLS-style, "reference-normalised evidence") is allowed under CLAUDE.md and does not touch the eval galleries. It is kept as an optional feature behind one reviewer question (Q1).
- Candidate-vs-other-candidates features inside the same gallery are "sibling" features. The target is a relation between query and candidate, so these are suspect. They are excluded from the primary design (Q2).

**Local-only ambiguity resolved conservatively.** Running an open-weights MT model locally is not literally an "external translation service or API", but it is grey, and the task says translation is not the signal. It is excluded.

**Silent areas and assumptions.**
- Runtime is silent beyond "Compute: A10G". I assume CLAUDE.md: target <= 45 min, >= 30% headroom, 1 h worst case.
- Model-size cap is silent. I assume none but stay <= ~600M parameters.
- Silent on: ensembling, TTA, unlabelled train paragraphs, per-gallery softmax over own candidates (assumed fine), tie rule.
- All assumptions are listed again under "Open questions & assumptions".

## Data findings

**None of the following is verified. Everything is an unrun diagnostic or a hypothesis.** Run all on `train.csv`, `train_labels.csv` and the train-side rows of `paragraphs.csv` only. From test files use only schema, row count, id format and size statistics for runtime planning (1,791 rows; paragraph counts per gallery for memory).

**Diagnostics to run (train only).**
- D1 Structure.
  - Number of train galleries and queries per gallery.
  - Candidates per gallery (expect 27-71, UNVERIFIED for train) and unique queries versus unique paragraphs.
  - Assert the label is always inside the candidate list.
  - Fraction of candidates that are the answer for no query (expect about 20% (desc, eval)).
  - Position of the correct candidate in the shuffled list (expect about 2.0% first (desc)). Diagnostic only; never used.
- D2 Overlap graph.
  - Paragraph-id sharing between galleries, as an equality relation only.
  - Connected-component sizes: if galleries come from sliding two-week windows, a single giant component is likely.
  - Seriation of galleries (spectral order / Fiedler vector of the gallery-overlap graph) to recover a time order, since no dates are given.
  - Check the recovered order against the shared-paragraph chain.
- D3 Speech clusters.
  - Union-find over (query paragraph, answer paragraph) pairs and over paragraph ids shared across galleries.
  - Cluster-size distribution: how many speeches have more than one Spanish and more than one Basque paragraph.
  - Hypothesis: most clusters hold one ES and one EU paragraph per gallery. If many speeches span several galleries, there are extra multi-positive pairs.
- D4 Text and tokenisation.
  - Characters per paragraph by language (desc: 120-1,500, median 274).
  - Token counts under each candidate backbone's tokenizer, especially Basque tokens per character.
  - Fraction truncated at max_len 192, 256 and 384.
  - Fraction of Basque paragraphs containing Spanish tokens and vice versa (code-switch residue).
  - Confirm language labels are as stated (queries `es`, candidates `eu`) and assert it.
  - Remaining proper-noun residue: all-caps acronyms (party names), quote marks.
- D5 Baselines under the same splits as the validation scheme.
  - Random, length-closeness, word TF-IDF, char 3-5-gram TF-IDF (each fit on train folds only), and frozen encoders.
  - Expect the eval-reported ordering to hold on train (word < char < frozen). Large departures indicate the train galleries are easier or harder than the eval galleries.
  - Report MRR and MRR normalised by random for the same gallery sizes (the random value depends on gallery size).
- D6 Information ceiling and failure anatomy.
  - Rank histogram of the frozen best encoder.
  - Recall@k.
  - MRR by gallery size.
  - Top-1 errors inspected by embedding-neighbour cluster as a proxy for "same-topic different-speaker" versus "same-speaker different-speech". No speaker labels exist, so the same-speaker decoy rate (75.5% in eval (desc)) cannot be measured on train directly.
- D7 Cheap language-agnostic style cues.
  - For true pairs versus gallery decoys: ES/EU length ratio, sentence count, question-mark and exclamation rates, comma rate, mean sentence length, formulaic-address presence, paragraph-initial markers.
  - Score each as a single-feature MRR within gallery.
  - Hypothesis H4: each contributes about 0 to +0.01 MRR; the desc shows length-closeness alone is 0.0963 vs 0.0913.
- D8 Hubness.
  - k-occurrence skew of candidates in frozen-encoder space against a train Spanish query bank (the bank would be train-only).
  - Hypothesis H5: some Basque paragraphs (short, generic, ceremonial) are near everyone, and CSLS-style correction helps by +0.01 to +0.03. UNVERIFIED.
- D9 Temporal shift proxy.
  - Fit on early recovered-order blocks, test on late blocks.
  - Measure the MRR drop versus within-block random CV for the lexical, frozen and fine-tuned arms.
  - This is the only available proxy for the cross-term shift.
- D10 Near-duplicates and formulae.
  - Frequency of ceremonial boilerplate paragraphs.
  - Near-duplicate Basque pairs inside a gallery (possible irreducible ambiguity).
  - Identical-text candidates with different labels across train queries.

**Hypotheses (UNVERIFIED).**
- H1: frozen large multilingual encoder, zero-shot: 0.27-0.31 (desc has 0.2701 for the best public one).
- H2: gallery-listwise fine-tune of a large encoder, temporal-block CV: 0.35-0.45.
- H3: char-n-gram cosine adds +0.005 to +0.02 inside the learned fusion.
- H5: train-bank hubness correction adds +0.01 to +0.03.
- H6: epochs beyond about 3 lower temporal-block MRR through memorising first-term topics and ministers.
- H7: a cross-encoder reranker adds +0.02 to +0.05 but costs a lot of runtime and needs OOF rerank data. Deferred.
- H8: two backbones in the fusion add +0.01 to +0.03.
- H9: gallery size explains much MRR variance (a gallery of 71 has about half the random MRR of a gallery of 27).

**Irreducible ambiguity.** In 75.5% of eval cases another speech by the same speaker is a decoy (desc). Speaker cues therefore cannot resolve those cases, and the speech-specific content (a particular bill, opponent, argument line) must. A ceiling below 1 is expected. The estimate of that ceiling is itself unverified. A plausible oracle proxy is the within-cluster ranking of a very strong model.

## Validation design

**Reproduce how the split was made.** The test split is time- and sitting-disjoint across a legislative-term boundary. Galleries are built separately on each side, so no candidate is shared. Therefore:
- Hold out WHOLE galleries and whole speeches, never queries.
- Groups come from content-relation union-find (D2 and D3): union galleries that share any paragraph; union paragraphs of one speech through the label pairs.
- If the union-find produces one giant component (plausible if windows overlap), cut it into contiguous blocks along the seriated order, and drop (embargo) the galleries sitting on each block boundary that share paragraphs with the other side.
- Using paragraph ids as equality keys for grouping is a validation device, not a model input (B2 concerns signal; Q4 asks the reviewer).

**Folds.** K = 4 contiguous temporal blocks (about 25 galleries, roughly 1,070 queries each, UNVERIFIED), with a second scheme whose block boundaries are offset by half a block. Compare paired on the same folds. If seriation fails, fall back to K clustered group folds from the overlap graph (state that this is optimistic).

**Metric re-implementation and unit tests.** MRR per query with pessimistic ties.
- Perfect scores: 1.0.
- Correct item last: 1/n per query.
- Constant scores: 1/n (pessimistic).
- Random permutations: about H_n/n for gallery size n; mean over the train gallery-size distribution should land near 0.09. If train sizes differ from eval 27-71, report MRR relative to the same-size random value.
- Gold-score oracle through the writer must give 1.0 (stage 3 check).

**Noise.** Per-query RR has std about 0.3. The OOF set has 4,287 queries but correlated within galleries. Use paired cluster bootstrap over galleries (1,000 resamples). Accept a change when the paired difference's lower 95% bound exceeds 0 and it wins in at least 3 of 4 folds in both fold schemes; as a rule of thumb that needs a gain of roughly 0.01 or more. These are estimates, to be recalibrated on the real fold results.

**Post-hoc selection needs its own held-out check.** Epoch choice, fusion weights and any optional feature switch are chosen on OOF. Report them nested: fit the fusion on the OOF rows of 3 folds and score on the 4th (cross-fitted).

**Proxy bias direction and rough size (CLAUDE F2 #14).**
- Temporal-block CV inside the first term is OPTIMISTIC relative to the real term-boundary split. The real shift is larger (new governments, ministers, agendas).
- Speaker overlap in my blocks is probably higher than eval's 36/84 (about 43%). This again inflates, though only for the part of the signal that is speaker-driven.
- A mitigating factor makes the proxy slightly pessimistic: each fold model trains on 75% of the train queries.
- Net: expect the private score to land BELOW the proxy. I would guess by 0.02 to 0.06 MRR; this guess is an estimate, not measured.
- D9 gives the only available empirical handle on the size of the shift.

## Overfit/underfit risks

**Overfitting.**
- OF1: topical memorisation. Train-term vocabulary (ministers, bills) does not carry over. Mitigation: 2-3 epochs, small learning rate, partial layer unfreezing (top layers plus pooling and projection) compared against full fine-tuning under temporal CV, weight averaging over the last epoch, passage augmentation (random sentence-span crops and sentence drops within a paragraph, which are real-text augmentations). Log the parameter-change norm. The first-term topical shift is also the reason lexical TF-IDF is fit on train only and kept as a minor fusion member.
- OF2: few independent groups (about 100 galleries, UNVERIFIED). Mitigation: temporal blocks, paired comparisons, K = 4 with two boundary offsets, a fusion head of at most about 8 parameters.
- OF3: selection on the reported OOF (epoch, fusion weights, feature switches). Mitigation: nested or cross-fitted reporting.
- OF4: same-gallery speech decoys by the same speaker: a model that learns "speaker" rather than "speech" will fail on the 75.5% decoy cases (desc). Mitigation: listwise loss over the whole gallery, which contains such decoys naturally; check per-error analysis in D6.
- OF5: leakage in CV. Speech and paragraph recurrence across overlapping train galleries would leak across folds if grouped by gallery only. Mitigation: union-find (D2, D3) and embargo.

**Underfitting.**
- UF1: backbone too small. Mitigation: start from a large multilingual encoder (approximately 560M parameters), not a MiniLM/base class.
- UF2: truncation. Median 274 characters but up to 1,500 (desc). Mitigation: choose max_len from D4, probably 320-384 tokens for Basque, with length bucketing. Plan for the truncated fraction to be reported.
- UF3: loss not matched to the metric. Mitigation: per-query softmax CE over the gallery's own candidates, which is the exact decision problem (see Metric-aware section).
- UF4: information thrown away by only using sentence embeddings. Mitigation: an optional lexical member (char n-grams) and, later in the roadmap, a bounded cross-encoder rerank of the top 10-15.
- UF5: not enough training signal from 4,287 queries. Mitigation: use all labelled pairs in both directions (ES to EU and EU to ES, training only), plus multi-positive pairs from speech clusters if D3 shows they exist.

## Recommended approach (primary + fallback)

**Primary (ranked first by expected private score).** A gallery-batched, listwise-fine-tuned cross-lingual dual encoder, fused with a few within-query-normalised cheap members.

1. Backbone. A public, ungated, large multilingual embedding model: BGE-M3 first choice (XLM-R-large lineage, strong cross-lingual, covers Basque via XLM-R). Candidate alternates: multilingual-e5-large. Pin the revision hash in the script. Whether each exists and is ungated is UNVERIFIED (no internet); check before implementation. Use the pooling that the model's own recipe specifies (CLS for BGE-M3, mean for e5; e5 also wants the `query:` prefix, which I would apply on both sides because the task is symmetric).
2. Gallery-batched training (the key efficiency and distribution-matching trick). One step takes one train gallery (or two small ones). Encode all of its paragraphs ONCE: the candidates (about 50) plus the gallery's queries (about 40). Build the query x candidate similarity matrix. Loss is per-query cross-entropy over THAT gallery's own candidates, with a learned temperature initialised near 0.05 (or the backbone's own). Because the negatives are exactly the eval-style same-week, often same-speaker decoys, the model learns what separates speeches within a week. Cost: about 9k sequence passes per epoch instead of about 73k with 16 hard negatives per query.
3. Auxiliary symmetric term (training only): the reverse softmax over the gallery's Spanish queries for each answered candidate. This is cross-query use during TRAINING only; it is not used at inference (B5), and each test query is scored independently.
4. Capacity ladder (F2 #1), stop at the lowest rung that wins under temporal CV: (a) frozen encoder plus a low-rank learned linear map on both sides; (b) LP-FT style warm start with a tiny learning rate and 1-2 epochs; (c) fine-tune top N layers; (d) full fine-tune with learning rate about 1e-5. Under the platform rules, the shipped rung must be one where the trained component is load-bearing: the CV gain over the frozen probe on the same folds is logged and must be clearly positive (the "strip-the-ML" check is "frozen encoder plus char TF-IDF scores about 0.27-0.30, so the fine-tuned model must add materially to that").
5. Fusion: a tiny listwise linear head (at most about 8 parameters, trained by the same per-query softmax CE on OOF scores). Inputs, each z-scored among the query's OWN candidates before weighting (a z-score changes rank in a multi-feature sum even though it does not for a single score):
   - fine-tuned dense cosine (averaged over fold models),
   - frozen second-backbone dense cosine (zero-shot, adds diversity of assumption),
   - char 3-5-gram TF-IDF cosine fit on train folds only,
   - optional train-bank hubness correction of the candidate (Q1),
   - optional language-agnostic style deltas (ES/EU length ratio, punctuation rates), only if D7 shows a measured gain.
6. Shipping. Average the K = 4 fold models' per-query score matrices (each query scored independently by each model, then fused with weights fit on OOF), rather than refit on 100% and risk a calibration transfer mismatch. This is a deviation from the "refit on 100%" default, with a stated reason (fusion fit on fold-model OOF scores). The roadmap includes one 100%-refit test; adopt it only if the gain exceeds noise and transfer is checked.
7. Decode: sort by fused score; tie-break by a secondary score, then by candidate text (content), never by list position or id.

**Fallback.** Rung (a) of the ladder: frozen strong encoders (BGE-M3 and a second backbone), a trained low-rank bilinear/linear map plus the same fusion, all trained by the same listwise gallery loss. Cheap (about 8-10 min), valid, easy to debug, uses the supplied training answers genuinely. Expected 0.29-0.34 (estimate). Grey under the "frozen features plus head" caution, so it is a fallback, not the primary.

**Expected score (estimate, not a promise).** Temporal-block proxy for the primary: 0.36-0.48. Private: 0.32-0.44, centre about 0.38, wide because the term shift is unmeasurable here. Fallback private: 0.28-0.34.

## Rejected options

- Translation-matching or bitext-mining encoders as the main signal (e.g. LaBSE): the description says the paragraphs say different things and a translation-matching model "finds the paragraph most similar in content, usually a different speaker". LaBSE-class models may still serve as a frozen fusion member.
- Authorship attribution or a speaker classifier: no speaker labels, and 75.5% of eval galleries contain another speech by the same speaker (desc).
- Word or char TF-IDF / BM25 as the solution: 0.13 and 0.21 (desc) and grey as the core per CLAUDE.md. Kept only as a minor fusion input.
- Any cross-query step (Hungarian assignment, mutual NN, dual softmax or column normalisation over a gallery's queries, "each candidate answers at most one query"): banned (B5).
- Candidate-frequency, popularity, or gallery-membership counts: banned (B4). Row order, supplied order, id strings: banned (B2, B3).
- Candidate-vs-candidate features inside a gallery (e.g. speaker clustering of candidates): sibling-leakage risk and a grey reading of "its own candidates". Excluded from the primary; revisit only if the reviewer says yes (Q2).
- Local machine translation (NLLB / Marian), back-translation, LLM paraphrase: grey or banned (external translation, synthetic data, LLM-generated content).
- Zero-shot LLM reranking: inference-only, gated models are banned, too heavy.
- Gradient-boosted LambdaRank over hand features as primary: CLAUDE.md 6.4 says not enough, and it does not carry cross-lingual semantics.
- Domain-adaptive MLM on the eval paragraphs, pseudo-labels, self-training on test: banned (CLAUDE.md 2.3 #5). MLM on train paragraphs only: not needed, and grey (Q5).
- Large cross-encoder as the primary: runtime and OOF-hard-negative cost, and it breaks the lean-primary gate. Deferred to the roadmap as an optional top-K reranker.
- Full fine-tune with a high learning rate or many epochs: will memorise first-term topics.
- Training negatives drawn from other fortnights: easy negatives; the eval negatives are same-week.

## Fixed work plan & runtime budget

All counts below are fixed constants in the script. Time is used for logging only (CLAUDE.md section 3). No `torch.cuda.is_available()` switches, no `os.cpu_count()`, fixed seeds, `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)`, pinned model revisions, fixed bf16 or fp16 autocast path, fixed device `cuda`, fixed worker count, batches composed with a seeded generator.

Assumptions for the estimates (UNVERIFIED, to be profiled once on an A10G): about 100 train galleries, about 9k sequence passes per epoch (cands + queries), mean length about 90 tokens with length bucketing, large-model training throughput about 100 seq/s and inference about 300-400 seq/s.

| Stage | Fixed plan | Est. A10G time |
|---|---|---|
| S0 load, validate schema, build groups, seriation, folds | K = 4 temporal blocks | < 1 min |
| S1 char n-gram TF-IDF per fold (fit on train fold only) and cosine over each query's own candidates | fixed ngram range 3-5, fixed vocabulary cap | 2 min |
| S2 frozen second-backbone embeddings of all needed paragraphs | one pass, fixed max_len | 2-3 min |
| S3 fine-tune primary encoder, 4 folds x 3 epochs, one gallery per step | LR a standard default (about 1e-5, linear warmup 10%), fixed epochs | about 4 x 3.5 = 14 min |
| S3b score held-out fold and all test queries at the end of each epoch (for the epoch pick) | about 2.2k + 4k seqs per fold-epoch | about 4 min |
| S4 fusion fit on OOF, cross-fitted check, final fused test scores | at most about 8 parameters, fixed optimiser | < 1 min |
| S5 write, re-read and validate CSV (1,791 rows, set equality) | | < 1 min |

- Total about 25-30 min against a 45 min target (so at least 30% headroom, ideally 40%). If profiling shows more, reduce first to 3 folds or 2 epochs by editing the constants BEFORE submission, never by a runtime branch.
- Memory: about 560M params with bf16 autocast and AdamW is about 9-10 GB static, plus activations for roughly 90-130 sequences at length <= 384 (use gradient checkpointing and a fixed token cap per micro-batch). Under 24 GB with margin. The fixed micro-batch size must be chosen with the longest gallery (71 candidates plus about 60 queries) in mind.
- In-script search budget: epoch count in {2, 3, 4} chosen by mean OOF MRR across folds (a data-determined, deterministic choice that needs per-epoch test snapshots, which is what S3b is for), fusion weights by OOF. The learning rate is a fixed standard recipe default. If an LR choice is made at all, it is a two-point grid {1e-5, 2e-5} run in-script only when the profiled budget allows; otherwise fixed. No offline-tuned magic constants.
- Input validation: all candidate ids exist in `paragraphs.csv`, languages are `es` for queries and `eu` for candidates (assert), no NaN text, nothing empty after cleaning. Output validation: row count, id set equals `test.csv` query ids, each ranked list is a permutation of its own candidates, no duplicates, finite scores, re-read with `keep_default_na=False`.
- No constant-output fallback is written early because the plan has no time-dependent path; any failure is a loud failure, and any logged fallback is explicit.

## Metric-aware training & decode

(i) Loss. Per-query listwise softmax cross-entropy over the query's own gallery candidates. For a single correct item it is the log-likelihood of the answer; the MRR-optimal decode (sort by posterior) matches this exactly. Each query has equal weight, matching the metric's per-query averaging. Because gallery sizes differ (27-71), the loss on a larger gallery is a harder problem; no reweighting is needed since the metric also counts each query once.

(ii) Hard negatives. They come for free and match the eval distribution: the gallery's other speeches, including same-speaker ones. No synthetic negatives, no negatives from other fortnights. If a second stage is added later, its hard negatives come from OUT-OF-FOLD scores of the first stage.

(iii) Symmetric auxiliary loss (training only). For each answered candidate, softmax over the gallery's Spanish queries; this strengthens the shared embedding space. Both directions use only train data. It is never used at inference (B5).

(iv) Temperature and bias. Learned scalar temperature; no calibration across queries is needed because the metric is rank-only per query.

(v) Regularisation. Fixed epochs (2-4 by OOF), warmup, weight decay, dropout as in the backbone, optional EMA over the last epoch, no label smoothing, no validation-triggered stopping beyond the per-epoch snapshot pick above.

(vi) Decode. `argsort` of the fused within-query score, descending. Ties are broken by the secondary members in order, then lexicographically by candidate text; never by the list order, row order or id strings (B2, B3). The full candidate list is emitted.

(vii) Optional listwise features. Z-score, margin to the runner-up and gap to the top within the query's OWN candidates only. Not computed across queries.

(viii) No MRR-aware decode constants exist, so no decode search is needed. The only fitted decode-time objects are the fusion weights, nested-checked.

## Structural signals

Each invariant, and what it turns into:
- S1 Exactly one correct candidate per query: the per-query softmax over the gallery (training loss).
- S2 The gallery and the query share the same political week, often the same people: the negatives are matched, so train with the whole gallery rather than random negatives. The model has to learn speech-specific content over week-level topic and speaker.
- S3 A pair is symmetric in what it asserts (same speech): train in both directions (ES to EU, EU to ES) and share the encoder weights across languages.
- S4 Speeches recur across overlapping galleries (D2, D3): derived speech clusters give (a) leak-free CV groups, and (b) optional multi-positive supervised-contrastive pairs, including same-language pairs (ES-ES, EU-EU), with class-aware masks so same-speech paragraphs are not used as negatives (F2 #16). Recover clusters from TRAIN labels only; verify on every train case that a query's answer lies in its cluster.
- S5 Candidate position is uninformative (2.0% vs 2.05% (desc)): nothing may depend on it (B3).
- S6 Each candidate answers at most one query: a KNOWN structure that is BANNED at inference (B5). Not used even as a train-time mask: other queries' answers are legitimate hard negatives and the model must not learn a mask it cannot use.
- S7 Code-switch residue: Spanish words inside Basque paragraphs and vice versa, plus shared sub-word forms (cognates, loan words). Character n-grams capture it (0.2067 (desc) without training); the fine-tuned encoder and a learned member carry it too.
- S8 Rhetorical habits that cross languages (question vs exclamation use, sentence length, address formulae, rhetorical question frequency): language-agnostic style features, kept only if D7 shows a measured gain, fed into the learned scorer. Hand-coded rules that replace the model are not allowed.
- S9 Text length is nearly uninformative (0.0963 vs 0.0913 (desc)): at most a tiny fusion feature (ES/EU length ratio), only if D7 shows its use.

## Experiment roadmap

1. Contract, metric and validation correct and unit-tested (perfect, reversed, constant, random, gold-oracle through the writer). Build groups (D2, D3), seriation, folds. Stop criterion: all tests pass and the random MRR on train sizes is about 0.09 (a different value indicates a size difference to account for).
2. Cheapest valid end-to-end baseline: frozen backbone cosine plus char TF-IDF, fused by the tiny listwise head, full CSV written and validated. Log MRR per fold, two fold schemes. This is also the honest yardstick the fine-tune must beat. Stop: valid file and reproducible numbers close to the desc baselines.
3. Representation and structure: gallery-batched listwise fine-tune, capacity ladder rungs (a) to (d), paired on the same folds. Stop at the lowest rung whose paired gain over the previous rung exceeds noise (bootstrap lower bound > 0 and at least 3 of 4 folds in both fold schemes). Log train loss vs held-out MRR per epoch (memorisation check), the parameter-change norm and CV gain over the probe (compliance evidence).
4. Metric-aware refinements: symmetric auxiliary loss, passage augmentation, multi-positive cluster loss if D3 supports it, epoch choice by OOF. Each is accepted only when it beats noise.
5. Diversity of assumption: add the second frozen backbone and the char-n-gram member (done at step 2; re-test marginal gains), then hubness correction (Q1), then style deltas (D7). Only add what is individually measured. Optionally a bounded top-10-15 cross-encoder reranker, only after steps 1-4 are measured and only if runtime allows (needs OOF rerank training data; H7).
6. In-script bounded search with fixed counts (epoch in {2, 3, 4}; at most a two-point LR grid), with its own cross-fitted check. Stop: nothing exceeds noise.
7. Final fixed-plan run from a clean `working/`, then run twice and diff (expect near-identical rankings; large swings mean too few folds or seeds). Check the submission's rank distribution against OOF behaviour as a bug-catching sanity check, not for tuning. Include one 100%-refit test only if time permits and only adopt it if transfer is checked.

Credits: spend on baseline (step 2), best single (step 3), ensemble/fusion (step 5), and the final run only. If two independent solvers would converge on a technique (listwise fine-tuning of a strong multilingual encoder), test it before any proxy-driven tweak.

## Compliance audit

Applied against CLAUDE.md section 7 and the agent file section B self-audits.

- Test-set use: the script reads `test.csv` only to score each query from its own text and candidate texts. No TF-IDF/vocab/PCA/stats fit on test or train+test, no pseudo-labels, no dedup or counting over test galleries. Embedding caching is memoisation, not a feature. PASS (by design; re-verify in the code review).
- Wall clock: time is logged only. No `elapsed()` in a condition. PASS.
- Environment fallbacks: fixed device, fixed workers, no import fallbacks. PASS.
- Hard-coded tuned constants: epoch and fusion weights found in-script on OOF. LR is a standard recipe default or a two-point in-script grid. The ES/EU length-ratio feature (if kept) is learned, not a hand-set threshold. PASS with the stated LR caveat.
- External data / synthetic / LLM content / gated or non-public weights: none. Weights only from ungated HF repos, pinned revision. PASS.
- Strip-the-ML test: remove the fine-tuned encoder and the remainder (frozen encoder, char TF-IDF, fusion head) is the Step 2 baseline of about 0.27-0.31 (estimate). The fine-tuned component must add a clearly measured gain; otherwise the primary is not compliant as a "trained" solution. Gain over the probe on the same folds is logged.
- No whole-test aggregation: each ranking depends only on its own query, its own candidates, and train-fit state. Cross-query reasoning (B5) is absent, including the symmetric reverse softmax (training only) and any assignment step. Any normalisation is within the query's own candidate set. PASS.
- Sibling leakage: the model scores query-candidate relations. Candidate-vs-candidate features are excluded from the primary. The optional hubness feature uses a train-only bank. Flagged for the reviewer (Q1, Q2).
- Hard bans B1-B6: all honoured. Id strings and list order are never used, and tie-breaks are content-based. Candidate frequency across galleries is never computed.
- Constants derivable in-script: yes, per the work plan.
- Source readable, under 512,000 bytes, no embedded weights or blobs: planned.
- Validator and the output re-read per CLAUDE.md section 5, plus per-row set equality with the candidates: planned.

## Open questions & assumptions

**Reviewer questions (with a plan under each reading).**
- Q1. Is a CSLS-style hubness correction of candidates against a TRAIN-only bank of Spanish query paragraphs acceptable next to the ban on candidate frequency across eval galleries? Reading A (yes): include it if D8 shows a measured gain (+0.01 to +0.03, H5). Reading B (no, conservative): drop it, at an estimated cost of at most 0.03. The plan works under both readings.
- Q2. Does "its own candidates" permit features that compare a candidate to the OTHER candidates of the same query (e.g. mean similarity to the rest of the gallery)? The primary assumes NO. If yes, the benefit is untested and carries sibling-leakage risk.
- Q3. Gallery-batched training uses other queries of the same TRAIN gallery as in-batch signal. This is training only. At inference each query is scored independently. Confirm that training-time cross-query structure is fine.
- Q4. Union-find over paragraph ids to build leak-free validation folds: ids are equality keys in the validation harness, never inputs to the predictor. Confirm this is outside the id-signal ban.
- Q5. Is a train-paragraph-only unsupervised step (domain MLM, hubness bank from unanswered train paragraphs) acceptable? The primary uses the bank only if Q1 is yes and uses no MLM.
- Q6. Is a fusion of a fine-tuned encoder, a frozen second encoder and char TF-IDF acceptable under the RAG rule that the ranking layer must be genuinely trained? Reading: yes if the fine-tune is load-bearing. Cost under the strictest reading: a single fine-tuned dual encoder with no frozen members.

**Assumptions (the description is silent).**
- A1 Runtime: no limit beyond A10G; I assume CLAUDE.md's <= 45 min target and a 1 h worst case.
- A2 No model-size cap; I stay at <= ~600M parameters.
- A3 Ensembling over fold models and fusion of several scores is allowed.
- A4 Ties are broken pessimistically by the unknown evaluator; I make exact ties negligible and break them by content.
- A5 Train galleries resemble eval galleries in size and structure (27-71 candidates, about 4/5 of speeches queried). To be checked in D1; adjust the MRR comparisons for size if not.
- A6 Public encoders named above exist, are ungated, and are downloadable from the Hugging Face Hub in the grader environment. UNVERIFIED; check before implementation, and swap for another public multilingual encoder if needed.
- A7 Gallery membership (which candidate ids belong to which query) is dataset structure, not an "id string signal".

**What could not be verified.**
- Every number in Data findings and every runtime estimate (no data, no GPU).
- Train gallery count and sizes, speech-cluster structure, recoverability of a time order, the share of same-speaker decoys in train.
- Existence and licensing of the named backbones.
- The shift between the two terms, which is the biggest unknown between proxy and private.
