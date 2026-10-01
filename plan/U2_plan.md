# U2 Sanskrit eight-card transport repair: Eris build plan

Status of evidence: no dataset files were available when this plan was written. Everything under "verified from the description" comes from the challenge text. Everything marked **[UNVERIFIED]** is a hypothesis or a diagnostic to run on `train` before trusting it. Any number labelled "estimate" is an estimate, not a promise.

## Contract & decision unit

**One valid answer.** For each of the 244 test groups (1,952 rows), a bijection from the 8 gaps (4 contexts x 2 gaps) to the 8 cards of that group, written as one row per `target_id` with `prediction` = a `card_id` from that group's inventory. Columns exactly `target_id,prediction`, UTF-8, `index=False`, one row per test target, no blanks.

**Invalid vs low-scoring.**
- Invalid (fails validation): wrong columns, missing/duplicate/extra `target_id`.
- Scored as wrong but accepted: a foreign card ID, a blank, or a card used on more than one gap of a group. The evaluator marks the affected rows incorrect. Our decoder must therefore always emit a full permutation (never row-wise argmax).
- A valid card on the wrong gap is simply wrong.

**Metric, term by term (verified from the description).** Per group: `score = 0.50 S + 0.35 P + 0.15 C`.
- `S` = fraction of the 8 gaps correct.
- `P` = fraction of the 4 contexts with both gaps correct.
- `C` = 1 iff all 8 gaps are correct.
- Final = 100 x mean over groups, clipped to [0.01, 100]. Every group has equal weight (hierarchical: gaps -> group -> mean of groups).

Consequences I derived (unit-test them):
- For a bijection, `S` can only be 0, 1/8, ..., 6/8 or 1. Exactly one wrong gap is impossible.
- Uniform random permutation: E[S]=1/8, E[P]=1/56, E[C]=1/40320, so the expected score is about **6.88**. This is the floor, and the sample submission scores near it.
- One transposition inside a context: S=6/8, P=3/4, C=0 -> group score 0.6375.
- One transposition across two contexts: S=6/8, P=2/4, C=0 -> group score 0.5500.
- So when an error is unavoidable, it is cheaper inside a context than across contexts (0.6375 vs 0.55). A metric-aware decode can use this (see Metric-aware training & decode).
- The score is mildly gating: `P` and `C` need coherent restoration, but there is no zeroing gate.

**True independent unit.**
- Decision unit = the group (8 gaps, one shared inventory).
- Generalisation unit = the hidden *predicate family*. The split assigns whole families to train or test and keeps exact sentence texts on one side. The public files expose no family key. The effective sample size is the number of families, not 1,256 groups or 10,048 rows. **[UNVERIFIED]** The number of families is unknown; it is likely far below 1,256 because families recur across groups.

**Pipeline stages and how to diagnose each separately.**
1. *Candidate coverage.* Trivial by construction (8 cards, the gold is always among them, no pruning allowed). The check is only that every card tokenises sensibly and that the gold is always present in train.
2. *Scoring.* A neural 8x8 score matrix per group. Diagnose with per-gap top-1 accuracy and MRR from row argmax, ignoring the one-use rule.
3. *Decoding.* One-use assignment on the neural scores. Diagnose with (a) the oracle test (gold scores through the decoder must give exactly 100 on every train group) and (b) the gain of the decoded score over duplicate-allowing row argmax, given fixed scores.

## Compliance regime

**Domain.** NLP: a multiple-choice / cloze-style fine-tuning task with a structured (permutation) output. CLAUDE.md sections 6.2 (fine-tune a transformer) and 6.7 (genuinely fine-tune) apply. The challenge text is stricter than the generic playbook.

**Explicit bans and requirements extracted from the description (all hard constraints).**
1. *CPU-only ML is ineligible.* All scoring must be produced on the GPU by a trained neural model. Use `device="cuda"`, fixed config, no availability switches (CLAUDE.md section 3.3 agrees).
2. *Banned as the source of the candidate ranking or as a substantial determinant of the assignment:* TF-IDF, BM25, bag-of-words, n-grams, fixed-embedding nearest neighbour, manually written grammar, hand-built similarity scores and any other heuristic pipeline. This also covers hand-coded Sanskrit morphology, case/ending agreement rules, sandhi parsers, lexicons, card-frequency priors and character n-gram scorers. None of these appear in the shipped pipeline. They may exist only as throwaway dev-time diagnostics (e.g. "how far does token shape alone go").
3. *A nominal GPU step with a decisive CPU/rule solver is ineligible.* So: no rule-based candidate pruning before the neural step, no GBDT/linear stacker on top of neural scores, no rule-based post-fix. Allowed by the description: CPU file handling, batching, and "a one-use assignment algorithm applied to neural scores".
4. *Training or fine-tuning on the supplied repair task is the intended approach.* Frozen features plus a head, kNN on frozen embeddings, or zero-shot masked-LM scoring alone is therefore not acceptable as the shipped solution. Genuine encoder fine-tuning (many encoder weights updated) must be load-bearing.
5. *Budget:* one A10G (24 GB), **90 minutes end to end**, including download of weights, training and inference.
6. *Memorisation control:* families and sentences are held out. Memorised passages or predicate-specific card associations will not transfer. Do not add card-ID, target-ID, group-ID or row-order features. IDs are opaque.
7. *Join rule:* map each target to its context through shared IDs and the group's `capsule_byte_anchor` / `capsule_byte_span`. Never rely on CSV row order.
8. *Full submission always:* the platform may score a slice but accepts the full test file.

**Conflicts between the description and CLAUDE.md / the general strategist guidance (challenge wins).**
- *Runtime.* The description gives 90 min. CLAUDE.md says target <= 50 min and plan for <= 1 h worst case, with the 1.5 h figure as an official ceiling. Resolution: treat 90 min as the hard cap but plan <= ~55 min estimated (>= 40% headroom) because throughput is unverified.
- *Hand-built features.* The strategist guide (F2.9, F2.12) encourages hand-coded domain parsers and GBDT/LambdaRank members. This challenge bans heuristic pipelines and CPU ML from deciding the ranking. Here we drop that family entirely. This is a cost we accept, not a oversight.
- *Capacity ladder (frozen features -> linear head -> fine-tune).* F2.1 would stop at frozen features on small data. Here that rung is an explicit non-starter for shipping (ban 4). The frozen probe is kept only as a dev-time yardstick. The compliance layer is real partial fine-tuning with logged parameter movement.
- *Generic "TF-IDF as auxiliary input is grey-OK" (CLAUDE.md 2.5)* is overridden by the explicit ban here. Not used even as an auxiliary input.
- *Metric-aware decoding.* Many Eris tasks ban it. This description does not. Only "one-use assignment algorithm applied to neural scores" is explicitly allowed, so the plan uses Hungarian/MAP as the clean decode and treats expected-utility (MBR) decoding on the neural posterior as an optional step with a reviewer question (Open questions, Q1).
- No conflict on determinism: CLAUDE.md section 3 applies unchanged (fixed work plan, no wall-clock branching, seeded, `device="cuda"`, fixed workers).

**Hard-constraint translation for the script:**
- Pretrained weights only via Hugging Face (`from_pretrained`, pinned revision hash looked up at dev time).
- No `peft`/`bitsandbytes` (not in the allowed list); if LoRA were ever wanted it would be hand-rolled in plain torch.
- No pseudo-labels, no test statistics. Scoring of a test group depends only on that group's own capsule and the train-fit weights.
- Process each group independently of batch composition (LayerNorm backbone, no BatchNorm, fixed eval batch size), so predictions never depend on what else is in the test file.

## Data findings

**Verified from the description.**
- Train: 1,256 groups / 10,048 targets. Test: 244 groups / 1,952 targets. Test is 244/1500 = 16.3% of all groups.
- Every group: 4 contexts, 2 gaps per context, 8 cards with distinct forms. The `[GAP]` token sits at the stated zero-based position. Extra `[MASK]` tokens are withheld context, not targets.
- No card form appears unmasked in any of the four contexts of its group, so "is this word visible in the sentence" carries no signal.
- Cards in a group have "comparable grammatical surface conditions". The group is built so that similar positions compete for cards. Token shape and an isolated verb are said to be insufficient.
- Train and test families are disjoint. Exact sentences are disjoint. Vocabulary and broad patterns overlap.
- Test size/ID statistics may be used for runtime planning only: 244 groups x 64 (gap, card) pairs = 15,616 scored sequences for a single-gap cross-encoder.

**Diagnostics to run on `train` only (none have been run). Expected outcomes are hypotheses [UNVERIFIED].**

| # | Diagnostic (exact) | Why | Expected [UNVERIFIED] |
|---|---|---|---|
| D1 | Open `capsules.jsonl` in binary, `seek(anchor)`, `read(span)`, `json.loads`. For all 1,256 train groups assert: `capsule_id` matches the rows, 4 contexts x 2 gaps, token at each gap position is `[GAP]`, 8 cards with distinct forms. Check the ids of train rows resolve only to train groups. Check whether the file also holds test groups (count of lines vs 1,500). | Contract and join rule. If the file mixes train and test, never scan it whole or build any vocabulary from it. | 1,500 lines if the file holds all groups |
| D2 | Gold validity: `train_targets.csv` gives each of the 8 cards in a group exactly once. | Confirms the bijection and decode space (8! = 40,320). | All groups valid |
| D3 | Script and normalisation: share of tokens containing combining marks or non-ASCII Latin (ā ī ū ṛ ṃ ḥ ś ṣ ṭ ḍ ṇ ñ ṅ) vs Devanagari code points; share of strings not in NFC. | Decides tokenizer behaviour and whether to NFC-normalise (deterministic, loss-free). | Latin IAST-like with diacritics; some NFD/NFC mixing possible |
| D4 | Lengths: tokens per context (percentiles), `[MASK]` share per context, distance from each gap to nearest `[MASK]` / other gap, gap relative positions, two gaps adjacent or not; card char length and subword length under each candidate tokenizer. Test: only max and percentile sequence lengths for memory planning. | Sets `max_len`, input format, runtime. | Short sentences (tens of words); IAST fragments to ~2-3 subwords per word under XLM-R |
| D5 | Duplicate structure: number of distinct context texts among 5,024 train contexts; counts of identical visible-token sequences; label consistency (same sentence + same gap -> same gold form?). | Quantifies sentence recurrence and noise ceiling; shows how easily train memorisation happens. | Heavy recurrence (families repeat). Consistent labels for repeated sentences |
| D6 | Family reconstruction: union-find over train groups with edges (a) near-identical visible context (shingle Jaccard on non-`[MASK]`/`[GAP]` tokens) and (b) shared card forms whose group-document-frequency <= tau. Sweep tau and the Jaccard threshold. Report component-size histogram, largest component share, number of components. Pick the rule where the largest component is below ~10% of groups and component count plateaus. | Needed for held-out-family validation. | A giant component appears if all shared card forms link groups; rare-form linking yields tens to a few hundred families |
| D7 | Card pair structure: within a group, do cards share long prefixes/suffixes (same inflection class)? Is the two-gap pair in one context drawn from related cards? | Understand the adversarial design ("comparable surface conditions"). | Cards share endings; endings alone do not identify the gap |
| D8 | Positional leakage audit: chi-square of gold gap index vs card position in `cards`, vs context order, vs gap order within context, vs card-id string. | Confirms opaque ids carry nothing. Not exploited; just ensures no feature accidentally depends on order. | No dependence |
| D9 | Shape-only baseline (scratch, never shipped): predict by ending/character-shape agreement between each card and the neighbouring words, then Hungarian. | Quantifies how much token shape alone gives, to confirm the description's claim. | Low (maybe 10-20% per-gap) vs 12.5% chance |
| D10 | Zero-shot GPU masked-LM probe: fill the gap with each card, score pseudo-log-likelihood of the card tokens under XLM-R, then Hungarian, evaluated on the same family folds. | Honest floor that fine-tuning must beat. | Weak (single-digit to modest above the 6.9 floor) |
| D11 | Partner-conditioning check: with a trained single-gap scorer on the dev folds, compare top-1 accuracy when the other gap in the same context is shown as `[GAP]` vs filled with its gold card. | Decides whether within-context pair potentials are worth building. | Small to moderate lift; unknown |
| D12 | Coupling check: per-gap row-argmax accuracy vs accuracy after the one-use assignment. | Measures how much the transport constraint helps. | Meaningful lift when scores are moderately informative |

**Information ceiling / irreducible ambiguity (what to measure).** Any (sentence, gap) pair repeated in train with different gold forms counts as label noise (D5). If masks remove the decisive context, some gaps are truly ambiguous; the cleanest ceiling estimate is the held-out score of the best fine-tuned model, not a hand rule. **[UNVERIFIED]** No claim about the ceiling can be made without data.

**Hypotheses I expect to hold [UNVERIFIED].**
- H1: text is IAST-like Latin, so a multilingual subword tokenizer fragments words heavily.
- H2: random group CV is strongly optimistic because families recur across groups (likely a gap of tens of points); family-disjoint CV is the only trustworthy yardstick.
- H3: semantic selectional fit (which verb suits which arguments) is the main signal, and it must come from pretrained lexical knowledge plus fine-tuning; 1,256 groups is too little to learn Sanskrit semantics from scratch.
- H4: the model overfits within 2-3 epochs because repeated sentences make training loss collapse fast.

## Validation design

**How the test split was made (from the description only).** Hidden predicate families go wholly to train or test; exact sentence texts are kept on one side including repeats. So the right proxy is **family-disjoint, sentence-disjoint hold-out**.

**Scheme (dev time).**
- Build groups-of-groups ("families") by union-find per diagnostic D6, computed on train only. Do not rely on any provided id.
- `GroupKFold`-style 5 folds over components, balanced by component size, repeated with **2-3 different seeds** of the component-to-fold assignment. Hold-out fold ~ 20% of groups, close to the test's 16.3%.
- Assert zero exact-sentence overlap between a fold's training part and its hold-out. If overlap exists, extend the linking rule.
- Report mean and std of fold means for each seed, and **paired** differences between designs on identical folds. Accept a change only if the paired gain exceeds the fold-to-fold standard error and is positive on most folds and on both split seeds.
- Re-implement the metric exactly. Unit tests: perfect -> 100; any within-context transposition -> 63.75 for that group; across-context transposition -> 55.0; reversed/derangement -> low; duplicate-card submission handled as the description says (affected rows incorrect); a Monte-Carlo mean over random permutations -> about 6.88; clipping at 0.01.
- Report separately: ranking quality (per-gap top-1, MRR), decode gain (decoded score minus row-argmax-with-conflicts-resolved), oracle decode (gold scores -> exactly 100 on all train groups, plus decode runtime).

**In-script holdout (shipped script).** Compute the same union-find on train inside the script (CPU, train-only, deterministic) and hold out ~12% of families (about 150 groups) for model M1. This gives a printed family-disjoint number and a train-fit estimate of the decode temperature. It is a loud-failure guard only (see Fixed work plan).

**Direction and size of bias for each proxy (estimates, with reasons).**
- *Random group CV:* strongly optimistic (families and sentences recur). Never used for decisions; run once only to measure the leakage gap.
- *Family-disjoint 5-fold (my union-find):* roughly unbiased if my linking captures the true families; **optimistic** if it misses links (leaks through unseen links); **pessimistic** if it over-merges or removes too much training data per fold. Each fold trains on ~80% of the data, so the shipped 100%-data model should be slightly better than the CV model (small positive transfer).
- *12% in-script holdout (~150 groups):* noisy. A group score has a standard deviation of perhaps 0.2-0.3, so the standard error is about 2 points of the 0-100 scale. Not for model selection; only a sanity number.
- *A self-built proxy usually overstates the real score;* plan for the private number to land at or below the family-disjoint CV, because my families are only a reconstruction of the hidden ones.

**Post-hoc selection steps need their own held-out check.** The backbone tier, number of frozen layers, epochs, loss weights and the decode temperature are each chosen on the fold CV, so the final comparison reports a **cross-fitted** number (temperature fitted on folds not scored). With so few knobs the main protection is keeping the search grid tiny.

## Overfit/underfit risks

| Risk | Evidence / reason | Mitigation |
|---|---|---|
| Few independent units (families) vs ~560M-parameter encoder | Effective n = number of families; repeated sentences let the model memorise passages | Partial fine-tune (freeze embeddings + bottom layers), low LR, fixed 3 epochs, token-mask augmentation; log train loss vs held-out score per epoch; log parameter-change norm and the CV gain over the frozen probe |
| Memorising predicate-specific card associations | Explicit in the description | Family-disjoint CV; no card-form or id features; no per-card embeddings outside the encoder |
| Optimistic split | Families recur | Union-find families; assert zero sentence overlap; never use random CV for decisions |
| Overfitting to the in-script/dev holdout | ~150 groups gives ~2 point standard error | Fixed constants, tiny grid, paired multi-seed comparisons; holdout sanity fold untouched until the final config |
| Decode-temperature noise (if MBR is used) | One scalar fitted on ~150 groups | Cross-fit it in dev; ship MBR only if it beats Hungarian by more than noise; otherwise ship Hungarian |
| Underfit from a small backbone | Sanskrit semantics are low-resource; large multilingual models know more | Prefer the large tier if the budget allows (Tier L below); base tier is the fallback |
| Underfit from tokenisation | IAST diacritics fragment into many subwords [UNVERIFIED] | Consistent NFC normalisation; measure subword counts (D4); test a byte-level model and a deterministic IAST->Devanagari mapping only as roadmap arms |
| Information lost by input design | Truncation or dropping the other gap | `max_len` set from D4 so no gap is truncated (assert); keep both gaps visible (the other as a marker); measure partner-conditioning (D11) |
| Loss mismatch with the metric | Row-wise cross-entropy ignores the one-use constraint and P/C terms | Exact permutation likelihood plus row/column cross-entropy; evaluate S/P/C directly |
| Determinism noise (bf16, atomics) | Near-ties in scores | Fixed seeds, deterministic algorithms (warn-only), same precision path in dev and grading; Hungarian tie-break deterministic; run twice and diff |
| Runtime overrun | Throughput numbers are unverified | Dev profiling of one epoch, fixed work plan, >= 40% headroom, no wall-clock branching |

## Recommended approach (primary + fallback)

**Primary: multiple-choice cross-encoder over (gap, card) pairs, trained with an exact permutation likelihood, decoded by one-use assignment.**
- *Representation.* A pretrained multilingual masked-LM encoder, preferably `xlm-roberta-large` (it includes Sanskrit in its pretraining; [UNVERIFIED] how well it handles Latin-script diacritics). Revision pinned. All strings NFC-normalised. Withheld `[MASK]` tokens map to the tokenizer's `<mask>` token. The other gap in the same context is shown as a distinct literal marker.
- *Input per (gap g, card i).* The context of g with `[GAP]` replaced by the card form, the card span delimited so it can be pooled, and the other gap marked. The head reads `concat(<s> state, mean of the card-span states)` and outputs one scalar `S[g, i]`. For a group this gives an 8x8 matrix of 64 sequences.
- *Why this form.* It reuses the pretrained plausibility knowledge directly ("does this word fit here"), which is where the signal must come from for unseen families. It has few new parameters (one small head), so the strip-the-ML test is trivially passed: remove the encoder and nothing meaningful remains.
- *Loss.* Gibbs distribution over the 8! bijections, `p(pi) proportional to exp(sum_g S[g, pi(g)])`. Loss = `-log p(pi*)` computed exactly by a subset-DP log-permanent (2^8 = 256 states, run in fp32 on GPU), plus an auxiliary mean of row-wise and column-wise cross-entropy (dual softmax). Equal fixed weights; ablate the auxiliary in roadmap step 4.
- *Regularisation.* Freeze the embedding matrix (about 250M of XLM-R-large's 560M parameters) and the bottom 8 of 24 layers; AdamW, low LR, weight decay, dropout 0.1, warmup + linear decay, grad clip 1.0, fixed 3 epochs. Augmentation: seeded random replacement of ~10% of visible context tokens by `<mask>` (the data already withholds context, so this matches the noise model); never touch the candidate span or markers.
- *Training schedule and refit.* Model M1 trains on ~88% of train families (the holdout gives the in-script family-disjoint score and the temperature). Model M2 trains on 100% of train with the same fixed recipe. Shipped score matrix = mean of M1 and M2 logits per group. Rationale: ship the full-data model, keep M1 as a free second member and sanity check.
- *Decode.* Hungarian / exhaustive MAP assignment on the averaged matrix (clean, explicitly allowed). Optional upgrade: expected-utility decode (see Metric-aware training & decode), shipped only after it passes its own cross-fitted check and the reviewer question.
- *Compliance status.* Clean: genuine fine-tuning, GPU scoring, only an assignment algorithm after the neural scores.

**Fallback: the same code at `xlm-roberta-base` (Tier B).** Roughly 3-4x cheaper. Used if (a) profiling shows Tier L cannot meet the budget with >= 40% headroom, (b) Tier L fails to beat Tier B by more than noise on the family CV, or (c) Tier L shows unstable training. The code path is identical; only the constants differ, and the shipped file fixes one tier.

**Second-round candidates (enter only after the primary is measured; each must beat noise).**
- *Candidate C: context-level pair potentials.* Score the ordered card pair (i, j) in the two gaps of one context, so the permutation score is `sum over contexts of Phi_c(pi(g1), pi(g2))`. Train with sampled pair negatives (about 32 sequences per group); partition function by enumerating the 40,320 permutations (cheap). Build only if D11 shows that conditioning on the partner gap lifts accuracy materially. Inference is 224 pair sequences per group (about 55k for the test), affordable.
- *Candidate D: a byte-level encoder arm (ByT5 encoder)* for diversity of assumption (no subword fragmentation of diacritics). Blend (z-scored logits or train-reference ranks) only if it is within a few points of the primary alone. Heavy, so likely not affordable within 90 minutes alongside Tier L.
- *Candidate E: a deterministic IAST->Devanagari character mapping* as input normalisation so the backbone's Devanagari Sanskrit knowledge becomes reachable. Only if D3 shows IAST and the paired CV gain is large; it needs the reviewer question in Open questions Q2.

## Rejected options

- **Zero-shot / frozen masked-LM pseudo-likelihood as the shipped solution.** Inference-only; fails the prime directive and the "trains or fine-tunes" requirement. Kept as the dev-time floor (D10).
- **Frozen embeddings + linear/GBDT/kNN head.** The description bans fixed-embedding nearest-neighbour matching; a CPU GBDT on neural features risks the "decisive CPU solver" clause. Kept only as the frozen-probe yardstick in dev.
- **TF-IDF / BM25 / n-gram / character-shape / ending-agreement scorers, hand grammar, lexicons, card-frequency priors.** Explicitly banned from creating or substantially deciding the ranking.
- **A stacked GBDT or logistic recalibrator over neural scores.** CPU ML deciding the output; adds noise; skipped.
- **Row-wise argmax decode.** Produces duplicate cards (scored incorrect) and ignores the coupling the task is about.
- **Random group K-fold.** Optimistic by construction; used only to measure the leakage gap.
- **Full fine-tune of all weights including embeddings.** Memorises sentences/families with so few independent units. Embeddings and bottom layers stay frozen.
- **Recombining contexts with cards from other train groups (extra negatives).** Real units, but it breaks the "comparable surface" hard-negative design and risks the synthetic-data rule. Not in the primary.
- **Decoder-LLM (7B-class) LoRA scorer.** 64 sequences per group and multi-epoch training do not fit 90 minutes on one A10G; smaller decoders likely know less Sanskrit than XLM-R-large [UNVERIFIED]. `peft`/`bitsandbytes` are not on the allowed list.
- **Packed whole-group encoder with a custom 3D attention mask (one pass per group, bilinear gap-card scores).** Much cheaper, but it gives up the pretrained "fill-in plausibility" signal and needs custom masking. Kept as a contingency only if the primary proves too slow at both tiers.
- **Test-set transduction of any kind** (cross-group normalisation, priors from test, pseudo-labels, test-fit vocabularies). Banned.
- **Wall-clock branching, environment fallbacks.** Banned by CLAUDE.md section 3.

## Fixed work plan & runtime budget

Defaults below are starting values to be confirmed by the roadmap (steps 3-5). The shipped script hard-codes one set; none is read from the clock or environment.

**Constants (Tier L).**
- Seed 42 for `random`, `numpy`, `torch`, CUDA, DataLoader `Generator`; `PYTHONHASHSEED=0`; `cudnn.deterministic=True`, `benchmark=False`; `use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG` set before importing torch.
- `device="cuda"`, bf16 autocast (A10G supports bf16), no GradScaler; same path used in dev.
- Backbone `xlm-roberta-large` at a pinned revision; `max_len` from D4 (about 128), dynamic padding, truncation asserted never to cut a gap or the candidate span.
- Frozen: embeddings + bottom 8 layers. Trainable: top 16 layers + small head.
- AdamW, encoder LR ~1.5e-5, head LR ~1e-3, weight decay 0.01, warmup 10%, linear decay, grad clip 1.0, dropout 0.1.
- Micro-batch = 1 group (64 sequences), 4 groups per optimizer step, 3 epochs, group order shuffled by a seeded generator. `num_workers=2`. Eval batch 128 sequences.
- M1 holdout fraction 12% of families; M2 uses 100%.

**Stages and estimated A10G time (all estimates, [UNVERIFIED]; profile one epoch in dev and re-derive).**

Assumptions: about 80 subword tokens per sequence after dynamic padding, about 20 TFLOP/s effective in bf16.

| Stage | Tier L (large) | Tier B (base) |
|---|---|---|
| Imports, data load, byte-anchor index, NFC | ~1 min | ~1 min |
| Weight download (fixed pinned revision; fails loudly on error) | ~2 min | ~1 min |
| Train-only union-find, holdout split | < 1 min | < 1 min |
| M1 training (about 1,105 groups x 64 seq x 3 epochs) | ~20 min | ~6 min |
| M1 holdout eval + temperature fit | ~1 min | ~0.5 min |
| M2 training (1,256 groups x 64 seq x 3 epochs) | ~23 min | ~7 min |
| Test inference, both models (2 x 15,616 sequences) | ~2 min | ~1 min |
| Decode, validation, write, re-read | < 1 min | < 1 min |
| **Total (estimate)** | **~50 min** | **~17 min** |

- Headroom against 90 min: about 44% for Tier L. If real throughput is 40% worse, Tier L takes about 70 min, still inside 90 but over the CLAUDE.md 1 h worst-case. **Rule:** ship Tier L only if the dev-profiled run is <= ~55 min on an A10G-class card; otherwise drop to Tier B, or freeze 12 layers, or keep one model (M2 only).
- Memory estimate: Tier L at 64 sequences x ~100 tokens with the bottom 8 layers frozen, about 10-14 GB peak. Inference at batch 128 is lower.
- Pair-potential arm (Candidate C), if built, adds about 3 min of inference and a training-time cost similar to the primary.

**In-script validation (inside `solution.py`).**
- Load via `capsule_byte_anchor` / `capsule_byte_span`; assert schema, 4x2 structure, `[GAP]` positions, distinct card forms, and for train a valid gold bijection.
- Divergence guard: after M1, if the holdout score is below 10 (the analytic random floor is 6.88, not a tuned constant), raise. A silent degraded run is worse than a loud failure. No constant-output fallback is written.
- Output validation: use CLAUDE.md section 5 validator against `sample_submission.csv`, plus a per-group check that every group's 8 predictions are the group's 8 distinct card IDs; re-read the written CSV with `keep_default_na=False`.
- Time only in `print` logs.

## Metric-aware training & decode

**(i) Back-solve the metric.** The target is a bijection, so the loss works on bijections: exact permutation likelihood (log-permanent via subset DP) is the proper scoring rule for "all 8 correct" (the `C` term), and its marginals give per-gap and per-pair probabilities for `S` and `P`.

**(ii) Weighting and hierarchical averaging.** Every group has equal weight in the metric and has the same size, so the loss is the plain mean over groups. Report the metric exactly as defined (per group, then mean).

**(iii) Loss.** `L = -log p_Gibbs(pi*) + 0.5 (rowCE + colCE)` with equal fixed weights; the auxiliary term supplies dense early gradients and per-slot calibration. Ablate it (roadmap step 4).

**(iv) Decode.**
- *Primary decode (safe):* MAP bijection = exhaustive max or Hungarian on the averaged score matrix. Deterministic tie-breaking. This is the explicitly allowed "one-use assignment algorithm applied to neural scores".
- *Optional expected-utility decode (roadmap step 6).* The posterior over the 40,320 bijections is exact: `p(pi) proportional to exp(sum S / T)`. From it compute the cell marginals `m[g, a]` and, for each context c, the pair marginals `M_c[a, b]` (probability that its two gaps receive cards a and b). For any candidate bijection `sigma`: `U(sigma) = 0.5 * (1/8) * sum_g m[g, sigma(g)] + 0.35 * (1/4) * sum_c M_c[sigma(g1_c), sigma(g2_c)] + 0.15 * p(sigma)`. Evaluate all 40,320 candidates per group on the GPU (a few lookups each) and pick the argmax. `T` is a single scalar fitted by NLL on the M1 holdout, cross-fitted in dev. Because a within-context error costs less than an across-context error, this decode can trade a little `S` for `P`. Expected gain is small (likely under a couple of points); ship only if it beats MAP by more than the paired noise on family CV and the reviewer agrees (Open questions Q1).
- When the posterior is sharp, MAP and expected-utility coincide.

**(v)-(ix) Not applicable** (no thresholds, spans, counts or taxonomy in this task).

**Calibration.** Probabilities are not submitted, so no per-slot recalibration is needed; the one scalar `T` exists only for the optional decode.

## Structural signals

Turn each guaranteed invariant into a loss term, decode constraint or augmentation:
- **Bijection** between the 8 gaps and 8 cards: exact permutation likelihood in training, one-use decode at inference.
- **Distinct card forms** within a group: no tie handling needed between identical forms.
- **No card form is visible unmasked in its group's contexts:** no copy-detection feature is needed or useful; do not add one.
- **Card order, context order, gap order and all IDs are meaningless:** the scorer takes no ordinal or ID input; shuffle card/context order in training only as a no-op check that outputs are invariant.
- **Withheld `[MASK]` context is noise by design:** the seeded token-masking augmentation matches it and teaches robustness to missing context.
- **Four contexts / two gaps:** within-context pair coupling is a candidate extension (Candidate C). Across-context coupling is carried by the bijection.
- **Metric asymmetry** (within-context errors cheaper than across-context errors): used only by the optional expected-utility decode.
- **Family structure (hidden):** recovered from train by union-find to build leak-free folds and the in-script holdout. It is not a feature and is never applied to test.
- **Hard negatives are built in:** the 7 other cards of a group have comparable surface conditions, so every training sequence pair is already a hard negative. No extra mining is needed.

## Experiment roadmap

Ordered, each with a stop criterion. Credits (6 per problem): use them for baseline, best single, ensemble/final only.

1. **Contract, metric, validation (no model).** Build the capsule reader (D1, D2), metric with the unit tests listed above, family union-find and folds (D6), oracle decode (gold scores -> 100 on every train group; log decode runtime), leakage audits (D8). *Stop when* all tests pass, fold sentence overlap is zero, and the largest component is acceptable.
2. **Floors.** Zero-shot masked-LM probe with Hungarian (D10) and a frozen-backbone linear probe on the same family folds; also random-permutation Monte Carlo (about 6.88) and the shape-only diagnostic (D9). *Stop when* the floors are recorded. Not shipped.
3. **Strongest cheap end-to-end baseline (Tier B).** The primary architecture on `xlm-roberta-base`, row-CE loss only, MAP decode. Produces a valid `submission.csv` early. *Stop when* the 5-fold family CV is recorded and the script runs clean in one command (credit 1).
4. **Representation and structure (paired, same folds, 2 split seeds).** (a) add permutation likelihood and the dual-softmax auxiliary; (b) frozen-layer depth {0, 8, 12} and epochs {2, 3, 4} with constants held fixed per run (no validation-triggered stopping); (c) large vs base with measured throughput; (d) augmentation on/off; (e) log train loss vs held-out score per epoch and parameter-change norm. *Stop when* every retained change beats noise or the grid is exhausted. Decide Tier L vs B by the runtime rule.
5. **Diversity and input arms (only if step 4 is stable).** D3/D11 diagnostics decide: IAST->Devanagari mapping arm (Q2), pair potentials (Candidate C), byte-level arm (Candidate D). Blend only members within a few points of the primary, and cross-fit blend weights. *Stop* if no arm beats noise.
6. **Decode.** MAP vs expected-utility with cross-fitted `T`. Report S, P and C separately. *Stop* once the better option is decided and the reviewer position is known.
7. **In-script fixed plan.** Hard-code the winning constants, the two-model M1/M2 schedule and the divergence guard. Profile a full run on an A10G-class card; confirm <= ~55 min. Run twice from a clean `working/`, diff the CSVs (expect near-identical; large swings mean the plan is too noisy). Run the CLAUDE.md section 5 validator and section 7 audit (credit 2-3).
8. Keep a log table (id, change, family-CV mean +/- std per seed, per fold, runtime, public LB if submitted). Ignore public-LB chasing; only check CV and public-LB agree in rank.

When two or three independent solvers would plausibly converge on the same technique (cross-encoder with permutation loss, family-disjoint CV), test it before any proxy-driven tweak. Here that means steps 1, 3 and 4a come before steps 5-6.

## Compliance audit

CLAUDE.md section 7 and the strategist self-audits, applied to this plan.

| Check | Result |
|---|---|
| Test file read only for one-group-at-a-time inference | Yes. Test groups are fetched by byte anchor; no whole-file stats; the vocabulary is the pretrained tokenizer's; nothing fit on test. If `capsules.jsonl` mixes train and test lines, only train anchors are read for training and only test anchors for inference. |
| Whole-test aggregation | None. A group's prediction is a function of its own capsule and the train-fit weights. The optional decode temperature comes from the train holdout, not from test. No cross-group normalisation. |
| Wall-clock or environment-dependent branches | None planned. Time is used only in log prints. `device="cuda"`, fixed workers, fixed batch sizes. |
| Hard-coded constants tuned offline | LR, epochs, frozen depth, loss weights are design constants confirmed on dev family CV. They are flagged in Q3. The only constant fitted by code in-script is `T` (if used), from the M1 holdout. Mitigation: start from conventional defaults and keep the dev grid tiny. |
| External/synthetic data, self-hosted weights, banned libraries | None. HF weights only; no `peft`, no GitHub downloads. |
| Strip-the-ML test | Remove the encoder and the head: only a random-ish bijection remains (floor ~6.9). No rules, no regex, no lexicon, no priors. Passes. |
| Hand-built features / heuristics deciding the ranking | None shipped. The shape-only baseline and the union-find family builder are not predictors: the first is a scratch diagnostic; the second only partitions train rows for validation and never touches test. |
| Sibling leakage | The target is a relation between a gap and the group's cards. Sibling information (other gaps, other cards) enters only through the bijection in the loss and decode, which the task defines. Features are computed from a gap's own context plus one candidate; no cross-referencing of group-mates beyond the allowed inventory. |
| Inference-only / frozen-features-plus-head | Not the design. About two-thirds of encoder layers are updated. Log parameter-change norm and the CV gain over the frozen probe to show fine-tuning is load-bearing. |
| Source readability and size | One plain UTF-8 `solution.py`, well under 512 KB, no blobs. |
| Challenge-specific restrictions honoured | GPU model, genuine training, one-use assignment only, 90-minute cap, join by ids/anchor, no row-order use. |
| Determinism | Seeds, deterministic kernels (warn-only), same bf16 path, fixed plan. Run twice and diff. Residual risk: atomics may cause tiny numeric differences, so keep the decode robust and the margin check in mind. |

## Open questions & assumptions

**Reviewer questions (with a plan under each reading).**
- **Q1. Is expected-utility (MBR) decoding on the neural posterior acceptable as a "one-use assignment algorithm applied to neural scores"?** Reading A (yes): ship MBR if it beats MAP by more than noise. Reading B (no, only plain assignment): ship Hungarian/MAP. The compliant-under-both choice is MAP. Measured cost of choosing MAP: expected to be small (likely 0-2 points, estimate).
- **Q2. Does a deterministic IAST->Devanagari character mapping (a hand-written table used only to normalise input before the neural model) count as "manually written grammar / heuristic pipeline"?** Reading A (no, orthographic normalisation): use it if D3 shows IAST and the paired CV gain is large. Reading B (yes): don't use it; the plan loses only that arm.
- **Q3. Do dev-time-chosen architecture constants (backbone size, frozen depth, epochs, LR) count as "hardcoded constants tuned offline"?** Reading A (no, fixed design constants like batch size/epochs): fine. Reading B (yes): ship conventional defaults without tuning, and use only the divergence guard plus the in-script M1 holdout. The measured cost is whatever the dev grid showed beyond defaults.
- **Q4. Is partial fine-tuning (embeddings plus bottom 8 layers frozen) sufficiently "fine-tuning"?** The top 16 layers train, with logged parameter movement. If the reviewer wants more, unfreeze more layers (memorisation risk, runtime cost).
- **Q5. Is mixing the family-reconstruction union-find (CPU, train-only, used for the in-script holdout split) acceptable?** It only splits training data and does not influence predictions of test rows.

**Assumptions.**
- Dev access to an A10G-class GPU for profiling; if there is none, the Tier L runtime rule cannot be verified and Tier B is the safe ship.
- Hugging Face downloads work at grading time, and the pinned revision hash (to be looked up at dev time) is available. If the download fails the script fails loudly. There is no fallback model.
- `xlm-roberta-large` and `xlm-roberta-base` are in HF and handle the corpus script reasonably [UNVERIFIED]. If D3 shows Devanagari, other Indic-aware backbones (e.g. an IndicBERT-type or MuRIL-type model) become bake-off candidates; their availability and quality are unverified.
- The platform's deterministic checker tolerates small bf16 numeric differences that do not change the decoded bijection.

**What could not be verified (no data was available).** Every number tied to the data: the script/normalisation, sequence and card lengths, recurrence and number of families, the size of the random-vs-family CV gap, the zero-shot floor, throughput and runtime, memory, and the expected score.

**Expected private-leaderboard band (estimate, low confidence, not a promise).**
- Floor (random bijection): about 6.9.
- If a fine-tuned model reaches about 50% per-gap accuracy on held-out families: S about 0.5, P about 0.2, C about 0.02 -> score about 32.
- If it reaches about 70% per-gap accuracy: S about 0.7, P about 0.45, C about 0.1 -> score about 52.
- So a plausible range is **about 25-55, centred near 35**, driven mainly by how much Sanskrit lexical-semantic knowledge the backbone brings to unseen predicate families. A self-built proxy will probably overstate the real score; expect the private number at or below my family-disjoint CV.
