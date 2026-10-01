# U12 Botanical keys: build plan (Eris Strategist)

Status note: no dataset files were available for this run. Everything under "Data findings" is either arithmetic derived from the challenge text (marked DERIVED) or a hypothesis plus the exact diagnostic that would test it (marked UNVERIFIED). Nothing here was measured. All runtime and score figures are estimates. No solution code is written here.

## Contract & decision unit

**One valid answer.** For each test row `id` (one taxon in one key), a cell `decisions` that lists exactly one lead label (for example `3b`) for every couplet of that row's key, space-separated (commas also accepted). Columns exactly `id,decisions`; every test id exactly once; same row count (482). Labels must be real leads of that row's key (`key_leads.lead`), and no couplet may be answered twice.

**Invalid versus low-scoring.**
- Invalid, so the whole cell scores as no decisions: a token that is not a lead label, the same couplet answered twice, or a missing couplet (counts as wrong).
- Invalid, so the whole file is rejected: wrong columns, a missing, duplicated or unknown id, or a different row count.
- Merely low-scoring: a wrong lead at a couplet.
- Decisions for off-walk couplets are free (ignored), so always answer every couplet.

**Metric, term by term.**
- Only couplets on the correct walks of test rows are scored.
- For couplet c of one key, R_c is the set of test rows whose true walk passes c, and m_c is the number of distinct leads those walks take at c.
- If m_c = 1, c is not scored.
- Otherwise recall_b is, for each lead b taken by some row in R_c, the share of rows truly taking b whose decision is b.
- J_c = (mean over taken leads of recall_b − 1/m_c) / (1 − 1/m_c).
  - This is balanced-recall Youden J.
  - Always answering the same lead scores 0. Perfect scores 1. Anti-informative decisions go negative.
- Final = Σ_c |R_c| J_c / Σ_c |R_c|, clipped to [0,1]. The clip is global, so negative couplets cancel positive ones; they are not floored per couplet.
- Consequences:
  - No lead prior helps. Any decision rule that ignores the description scores about 0, including "lead above more taxa".
  - The metric is flat across couplets and keys, weighted by |R_c|. Couplets reached by many test rows (roots and upper levels of big keys) dominate the score.
  - Each couplet is an independent binary (rarely ternary) decision. Wrong ancestors do not void descendants.

**Decision unit and independence.**
- Decision unit: (row, couplet), a choice among 2-3 leads, conditioned on the couplet being on the row's true walk.
- Independent unit for validation and variance: the key (970 train keys), refined by content-overlap groups (see Validation design). Rows in a key share lead texts and often vocabulary, so 4,989 rows are far fewer independent samples.
- Effective sample for statistical power: roughly 970 keys and a few thousand distinct couplet texts, not about 20k decisions.

**Pipeline stages, diagnosed separately.**
1. Candidate coverage. This is trivially complete, because the candidates are the couplet's own leads, so coverage is 100% by construction. The only coverage question is evidence reachability: does the description contain text that decides the couplet? Diagnose this as the "silent-couplet rate" (see Data findings, item 7).
2. Scoring (a learned local scorer, p(lead | description, couplet)). This is the model.
3. Decoding. The primary decode is an independent per-couplet argmax over the symmetric scores. An optional within-row tree-structured decode is a roadmap item.

**Why the local conditional is the right target.** Because only on-walk couplets are scored, the quantity to maximise per couplet is P(b | row passes c, description). An independent argmax of a model trained on on-walk examples already optimises the expected metric. A full-tree path decode only helps if descendants carry extra evidence about which child subtree holds the taxon (roadmap step 5).

## Compliance regime

**Domain.** NLP (matching natural-language lead statements to natural-language descriptions), a medium-difficulty task on one A10G. It is not labelled from-scratch and not labelled fine-tuning, but the description explicitly requires training or fine-tuning a model and allows general-purpose BERT-style encoders or small LMs. So CLAUDE.md §6.2 plus the challenge text applies.

**Explicit bans from the description (hard constraints).**

| # | Ban (quoted/paraphrased) | How the plan honours it |
|---|---|---|
| 1 | Real ML required; purely rule-based or algorithmic solutions not accepted however well they score | Every decision comes from a trained neural scorer. Numeric or unit parsers, if used, are input features to the scorer, never decision rules. Strip-the-ML test is in the audit. |
| 2 | No weights fine-tuned elsewhere (e.g. on this data); no hosted model APIs | Only the original public backbone checkpoint is downloaded from Hugging Face (pinned revision) and trained inside the script. No API calls. |
| 3 | No zero-shot/few-shot prompting, and no choosing test decisions with a pretrained model not trained on the provided data | No decision may come from a frozen similarity or LM score. A frozen encoder is only the starting point of a model whose head and top layers are trained on train rows. Any frozen-similarity baseline is a dev-time diagnostic on train folds and is never shipped. |
| 4 | No external data (other floras, keys, trait databases, herbarium records, copies of this data, web look-ups) | Only the provided csvs. No botanical dictionaries or trait lists are hand-typed in. |
| 5 | No synthetic data, including generating extra training data with another model or otherwise | Augmentation only transforms real labelled units (lead-order permutation, mild token dropout in the description memory). No fabricated descriptions, keys or couplets. |
| 6 | Test data is for one-row (or one-batch) inference only. Each test row predicted on its own. | A row's decisions are a function of its own description fields, its own key's leads, and a train-fit model. Batching is allowed only if it cannot mix rows (no batch statistics, no cross-row attention). |
| 7 | No pseudo-labelling, test-time adaptation, reweighting training samples using the test set, or calibrating/balancing outputs using the distribution of test predictions | None used. All calibration (if any) is cross-fitted on train OOF. |
| 8 | No coordinating decisions across test rows that share a couplet or key | This rules out any assignment or matching across test rows of a key (for example Hungarian, "each leaf is used once", balancing leads per couplet across rows, or consistency smoothing between rows). Each row is decoded alone. |
| 9 | No manual work on test rows; no hard-coded answers or test-id dictionaries | None. |
| 10 | TF-IDF / n-gram / Markov / frequency methods at reviewer discretion | Not the core. If used at all (idf weights for the numeric/lexical helper features), they are fit on train-key text only and only feed the trained scorer. |

**Where the description is silent, and the assumption made.**
- Runtime limit: silent beyond "Compute: A10G". Assume CLAUDE.md §1 (target at most 50 min, plan for at most 1 h).
- Internet: silent. Assume the CLAUDE.md rule holds (Hugging Face weights download allowed, nothing else), consistent with "openly downloadable" backbones being permitted.
- Model size cap: silent. "BERT-style encoder or small language model". DeBERTa-v3-large (435M) is a BERT-style encoder; a decoder LM is kept to at most about 1.5B and is not the primary.
- Whether using the key's own tree structure (parent/child couplets, subtree sizes) at inference counts as using other data: the key is the row's own input. Assume allowed; fit nothing on it beyond train keys (see audit).
- Whether using the description's stated facts (test walks have at least 4 couplets; keys are bigger) to choose a validation subset is allowed: assume yes. It is description text, not test data. It is used only to select held-out train rows for validation, never to reweight training.
- Pre-computing frozen backbone states for all rows of all keys before training: assume fine (they are forward passes; nothing is fit on test). The shipped script computes no test-dependent statistic.

**Reading chosen.** The one compliant under every plausible interpretation: independent per-couplet decisions from one trained scorer, no cross-row coordination, no frozen-similarity decisions, no test-fit statistics.

## Data findings

**DERIVED from the challenge text (arithmetic, not measured):**
- Leads total 12,865 with exactly three ternary couplets, so couplets C = (12,865 − 3) / 2 = 6,431 across all 1,026 keys (970 train + 56 test).
- In a key tree each non-root couplet is entered by one lead and every other lead ends at a taxon, so taxa (terminal leads) = 12,865 − (6,431 − 1) = 6,435 across all keys, about 6.3 per key.
- Train: 4,989 rows over 970 keys is 5.14 rows per key. Hypothesis (UNVERIFIED): train keys are small (about 4-5 couplets, about 5-6 taxa) and nearly every taxon of a train key has a row (coverage about 90%+). Test: 482 rows over 56 keys is 8.6 rows per key, with keys of 4-144 couplets, half the rows in keys of at least 22 couplets, so test coverage of each key's taxa is probably low (roughly 20-30% on average, UNVERIFIED).
- Test walks all pass at least 4 couplets (median 6, max 30) versus train median 3 (1-27). The test distribution is shifted to bigger, deeper keys. That shift is the main generalisation risk.
- Implication for the metric: in a big test key with few rows, deep couplets are reached by 1-2 rows, so m_c = 1 and they are unscored. Score mass concentrates on upper couplets of big keys and on all couplets of small test keys.

**Diagnostics to run on train (all UNVERIFIED; expected hypotheses given):**

| # | Diagnostic | Expected / why it matters |
|---|---|---|
| 1 | Shapes, dtypes, empties per field (`habitat`, `distribution`, `phenology`); `path` parse check (labels exist in `key_leads`, one lead per couplet, starts at 1x, each step follows `goes_to`) | Habitat/distribution/phenology empty rates likely 20-60%. Path parse should be 100% valid; any failure is a data bug to quarantine. |
| 2 | Walk-length histogram; share of rows with walk at least 4 (the validation subset); mean walk length; total on-walk decisions = Σ path length | Median 3, so perhaps 30-40% of rows have walk at least 4 (about 1.5-2k rows); about 17-22k labelled decisions. Sizes the proxy and the power of the validation. |
| 3 | Keys: couplets/key, leaves/key, rows/key, coverage = rows/leaves per key; ternary couplets (3 in all data) and where they are | Train coverage near 90%+ means nearly all train couplets have m_c = 2. Check whether any ternary couplet is in train. |
| 4 | Lead label balance on on-walk decisions: P(a), P(terminal lead), P(lead has more leaves beneath) | Likely skewed (small or specific alternative first). Irrelevant for the metric (balanced recall) but tells whether lead-position features are a shortcut to exclude. |
| 5 | Lead text stats: length in words and tokens, share with digits/units/ranges, with negation ("not", "without", "never"), with "[taxon]", with catch-alls ("not combining", "other", "otherwise"), with habitat/range words, with "or", ";" clause counts | Expect many numeric/range leads (moss/plant measurements) and a few percent catch-all leads decided by elimination. Sets the need for numeric features and null-evidence handling. |
| 6 | Description stats: tokens (p50/p90/p99/max) with the real backbone tokenizer, sentence counts, per-field lengths | Likely 150-600 tokens for the description with long tails. Decides chunking (see work plan). |
| 7 | Silent-couplet rate and information ceiling: for each on-walk couplet take content stems that distinguish the two leads (set difference, stopwords removed, light stemming, fit on train text only) and check whether any appears in the description. Also report a naive lexical-overlap decision rule's J (diagnostic only, never shipped) | If the lexical rule gets J around 0.1-0.25 and the silent rate is 20-35%, the practical ceiling is roughly 0.65-0.8 and a good model should land well above the lexical floor. This is the strip-the-ML yardstick. |
| 8 | Duplicates: exact and near-duplicate descriptions (normalised text, 5-gram shingle Jaccard at least 0.8), taxa appearing in more than one train key, keys with identical or near-identical lead sets, couplets (lead text pairs) repeated across keys | Hypothesis: a few percent of rows or keys are duplicates (sub-keys, multi-level floras). If large, merge into groups (see Validation design). |
| 9 | Lead-only control: a model that sees the couplet but no description, trained with the balanced weights, evaluated on held-out keys | Should be near 0 under balanced weights. Any residual is a lead-text shortcut and sets the shuffle-control baseline. |
| 10 | Shuffle control: evaluate the final model with descriptions permuted among rows of other keys | Must collapse to about 0. If not, the model is reading lead text or couplet position, not the description. |
| 11 | Numeric prevalence: share of on-walk couplets where both leads contain a numeric range for the same noun phrase and the description has a number with the same unit | Expect 15-35%, concentrated where BERT-style models are weakest. Sets whether the numeric feature channel (roadmap step 3) can pay off. |
| 12 | ID/order leak audit: `id` pattern vs key vs path length; row order inside train.csv; correlation of terminal label `Txx` index with depth-first order (the text says it carries no meaning) | Expect no usable leak; do not use any of these as features even if a correlation appears. |
| 13 | Gold-score oracle: feed the true path as scores through the decoder and the metric; must reproduce the true decisions and give 1.0 exactly | Proves formatting, label parsing and metric implementation before modelling. |
| 14 | Group structure: connected components under the union-find rule in Validation design (largest component share of rows, number of groups) | Largest component must stay at most 5% of rows or the merge rule is too loose. |

**Irreducible ambiguity (hypotheses).**
- The text itself says descriptions sometimes do not mention the asked character, some leads can only be decided by elimination, habitat or range, a few rows are subspecies or hybrids described by comparison, and keys and descriptions occasionally disagree.
- So per-couplet recall has a ceiling below 1. Do not chase individual disagreeing rows; do not overfit to them.

**Data-generation caveats.**
- The `path` is the walk the key itself uses, even if the description disagrees. Treat disagreeing rows as label noise, not as something to fix.
- Taxon names are masked as `[taxon]` in both leads and descriptions, so a "name appears in lead" shortcut is not available. Verify with diagnostic 5 that the mask is applied consistently, and do not special-case it in code.

## Validation design

**Reconstructing the split.** The test keys and their taxa never appear in training, so the test split is by whole key (and by taxon). Mirror it: hold out whole keys.

**Groups (built unconditionally, not only if found).**
- Start with `key_id` as the base group.
- Union-find merge keys that share a near-duplicate description (5-gram shingle Jaccard at least 0.8, or identical normalised text) or an identical long couplet (both lead texts identical and combined length at least 120 characters, so generic couplets like "Leaves opposite" do not chain everything).
- Fold-assign groups with `GroupKFold`.
- Sanity: report the largest group share of rows, which must stay at most 5%.
- Rationale: the same taxon can appear in more than one key, and sub-keys of one genus can be near-copies.

**Evaluation subset and metric.**
- Report the exact official metric implemented from the description, computed per held-out fold on that fold's rows only (so m_c and R_c come from held-out rows only, as in the real test).
- Primary proxy "V4": held-out rows with walk length at least 4 (matches the test's stated minimum), counting all couplets on their walks (including couplets 1-3), with held-out keys whole.
- Secondary "V-all": all held-out rows.
- Tertiary "V4-sparse": V4 averaged over 20 random subsamples that keep about 25% of each held-out key's V4 rows, to mimic the low coverage of the test (hypothesised; see Data findings). Report it because it reweights toward upper couplets, as the real metric probably does.
- Also report per-depth-bucket J (couplets 1-2, 3-5, 6+), per key-size bucket, by lead type (numeric, habitat/range, qualitative, catch-all) and by number of leads.

**Repeats and noise.**
- 5 grouped folds, repeated with 3 different group-split seeds during development. In the shipped script, 5 folds with one fixed split seed.
- Report mean ± std across folds, and the paired difference across the same folds when comparing two designs.
- Accept a change only if the paired gain on V4 exceeds the fold-to-fold standard error and is consistent across the split seeds.
- Expected V4 standard error is large (a few hundred independent keys among the at-least-4-deep subset, UNVERIFIED), so resist tiny tweaks.

**Metric unit tests (must pass before any modelling).** Perfect = 1.0; constant "always a" = 0 (equal to the sample submission); every decision the wrong lead = 0 after clipping, with the unclipped value negative; random decisions at the base rate close to 0; a prior-only rule (always the lead above more taxa) = 0; the gold path through the decoder reproduces the path exactly.

**Bias direction of each proxy, with reasons.**
- V4 proxy versus the real test: probably optimistic. Held-out train keys are smaller and have easier distinctions than the test's 22-144-couplet keys, which separate close relatives with fine characters.
- Training on 80% of the keys in each fold (versus 100% for the shipped model): slightly pessimistic.
- V4-sparse versus real: roughly right in structure if test coverage really is low, but this is a guess.
- Net expectation: the real private score will be below the V4 number by a few to several tenths of a point of J (rough guess 0.03-0.12 absolute; this is an estimate, not measured). Use the proxies for relative comparisons only.

**Selection steps need their own held-out check.**
- Choosing the rung (probe versus partially fine-tuned) in-script uses OOF J, which is optimistic for the chosen rung; the report must show it cross-fitted: select on 4 folds' OOF and score on the 5th.
- A stacker, if adopted, is trained on cross-fitted OOF margins and reported through an outer cross-fit.
- A final sanity fold: hold out one extra group-fold that is never used for rung choice or stacker fitting, scored once for the final config at dev time.

## Overfit/underfit risks

**Overfit risks and mitigations.**

| Risk | Evidence | Mitigation |
|---|---|---|
| Few independent groups (about 970 keys, a few thousand couplet texts) and a large backbone | 20k decisions are highly correlated within keys | Capacity ladder (see approach): frozen features, then head, then top-k layers at tiny LR. Do not fully fine-tune the large model. Fixed epochs. EMA. Log train loss vs OOF each epoch. |
| Memorising lead texts (the same couplet text with many rows) | Lead-only control above 0 | Balanced per-couplet weights; shuffle control; group folds by key; token dropout on the memory. |
| Lead-position/structure shortcuts (terminal vs subtree, depth) | Diagnostic 4 | Score leads with a permutation-symmetric scorer; random lead-order permutation in training. Structural scalars (depth, subtree size) enter only as optional inputs gated by CV, never as priors. |
| Selection on OOF (rung, stacker) | Optimistic reported score | Cross-fit those selections (see Validation design); keep candidates to 2 rungs and a 4-coefficient stacker at most. |
| Train-test depth/size shift | Train median walk 3 vs test 6; keys up to 144 couplets | Report V4 and per-depth J. Use size-free features for any stacker. Do not let shallow-key statistics decide anything for deep keys. |
| Disagreeing key/description rows (label noise) | Stated in the challenge text | Plain cross-entropy, no label smoothing and no loss reweighting by confidence. Accept the ceiling. |
| Many hard-coded constants | CLAUDE.md §7 red item | Few defaults only (LRs, epochs, layers unfrozen), set a priori and sanity-checked by repeated CV during development, not fine-tuned. The only fitted choices are in-script and cross-fitted. |

**Underfit risks and mitigations.**

| Risk | Mitigation |
|---|---|
| Numeric, range and unit reasoning ("1.5-2.3 mm" vs "c. 2 mm") is a known weakness of BERT-style encoders | Numeric pair features (value in range, distance, unit match, trait-word overlap) fed to the scorer as a measured add-on (roadmap step 3). Regex is for number extraction only. |
| Comparative leads need the sibling lead for meaning | Encode the couplet's leads jointly (text of all leads in one sequence with lead markers), and average over lead-order permutations at test time. |
| Silent description: "choose by elimination" | A learnable null token in the description memory, so a lead can attend to "no evidence". Catch-all leads become learnable. |
| Truncation of long descriptions | Chunk at sentence boundaries into at most 384-token pieces; concatenate chunk states into one memory; the memory cap comes from the train token-length quantile (diagnostic 6). Do not truncate the tail silently. |
| Backbone too small or lacking botanical vocabulary | Start from the largest backbone that fits (DeBERTa-v3-large); the dev-time bake-off may swap it (see roadmap). |
| Too little training of the top layers | Two rungs (probe, then top-k layers tuned), fixed epochs, log parameter-change norm and CV gain over the probe. |
| Loss mismatched with the metric | Per-couplet balanced weights (see Metric-aware training). |

## Recommended approach (primary + fallback)

### Primary: late-interaction couplet scorer on a strong pretrained encoder (LP-FT ladder)

**Representation.**
- Backbone: a large BERT-style encoder, default `microsoft/deberta-v3-large`, pinned revision, bf16 autocast. A dev-time bake-off against `roberta-large` and one retrieval-pretrained BERT-large (decided by grouped CV on the frozen probe, not by the public LB) fixes the shipped backbone.
- Description stream: text assembled as `Habitat: ... Distribution: ... Phenology: ... Description: ...` (empty fields written as an explicit "none"), split at sentence boundaries into chunks of at most 384 tokens, each chunk encoded once per row, chunk states concatenated into a memory M (plus one learnable null vector).
- Couplet stream: the couplet's lead texts encoded jointly in one sequence, `[A] lead_a [B] lead_b ([C] lead_c)`, giving per-lead token spans. Encoded once per (key, couplet), reused for all rows of that key.
- Both streams share the backbone (shared weights).

**Scorer head (few parameters).**
- 2 blocks of cross-attention from each lead's tokens (queries) to the description memory (keys/values), then attention-pool over the lead span and a small MLP, giving one scalar score s_l per lead.
- Couplet probability = softmax over that couplet's leads. Cross-entropy on the true on-walk lead.
- Optional additional input to the MLP (gated by CV): the numeric-agreement feature vector for (lead, description) and two structural scalars (depth, log subtree size) with random dropout so they cannot dominate.
- Permutation symmetry: the scorer sees leads in a random order in training. At inference, average the margin over all permutations of the leads (2 for binary couplets, 6 for the rare ternary ones). No lead-letter feature is ever fed to the model.

**Capacity ladder (stop at the lowest rung that wins under grouped CV; per F2.1).**
- Rung 0, probe: backbone frozen, head trained on cached states (cheap).
- Rung 1, LP-FT: warm-start from the probe, unfreeze only the top k layers (default k = 4 of 24) at a tiny LR, 2-3 epochs. Cache the frozen lower stack's outputs once per run so only the top k layers recompute (this is the main speed lever).
- Rung 2, full fine-tune: not shipped. Tested only at dev time on one fold to confirm it memorises groups (expected).
- The in-script rung choice uses cross-fitted OOF J (2 candidates only).
- Log the parameter-change norm of the unfrozen layers and the CV gain over the probe, as evidence the training is real (compliance and underfit check).

**Training data.**
- Examples: every (train row, on-walk couplet) with its true lead. About 17-22k decisions (UNVERIFIED), each with the row's description memory and the couplet encoding.
- Row-major batches: 16 rows per step, each row contributing all its on-walk couplets.
- No pseudo-labels, no test rows, no synthetic units.

**Regularisation and averaging.**
- Weight decay, attention and dropout, mild token-span dropout in the memory (about 10%; this is the only description augmentation, kept mild because dropping the deciding sentence creates unlearnable examples), gradient clip 1.0, warmup plus cosine, EMA of weights (decay fixed at 0.99), no label smoothing, fixed epochs.
- Final shipped predictions: refit on 100% of train with 3 fixed seeds, average the lead-order-averaged margins. OOF folds only produce numbers, the rung choice and calibration.

**Decode (primary).** For each couplet of the row's key, pick argmax of the permutation-averaged lead scores. Emit all couplets, in couplet order, space-separated.

**Why it fits this data.**
- It reads each couplet against the description in one trained model with explicit evidence alignment (cross-attention), cheap enough to fine-tune only the top layers of a large encoder in minutes.
- The couplet-level softmax with permutation symmetry matches the metric (a comparison among leads, not a prior).
- It avoids retrieval cutoffs and avoids a joint cross-encoder pass for every (row, couplet) pair, which is about 4x more tokens.

**Compliance status: clean to lightly grey.** It is a genuine trained scorer. The only grey elements are the optional numeric regex features and the optional within-row tree stacker (roadmap, both gated).

### Fallback (a dev-time design alternative, not a runtime branch)

If the primary fails the budget or the memory check, or is not stable, switch the constants at development time to the same architecture on `microsoft/deberta-v3-base` (about one third of the compute, same code path, k = 4 of 12 layers unfrozen). There is no runtime fallback of any kind (CLAUDE.md §3 rule 3).

### Second family for diversity (roadmap step 6, only if each alone is close in quality)

- Evidence-conditioned joint cross-encoder: DeBERTa-v3-base on `[lead_a][SEP][lead_b][SEP]` plus the description chunks, run as a multi-instance classifier over chunks (max or log-sum-exp pool of chunk logits per lead), 2 epochs, train/val on one split first.
- Different assumption (joint attention inside one encoder, no separate memory), so the errors may decorrelate.
- Blend: z-score or train-reference rank of the margin before averaging (not raw logit averaging); fit at most one weight on OOF.
- Rough cost of the extra member: 15-20 min of the 50, so adopt only with a measured gain over noise.

## Rejected options

| Option | Why rejected |
|---|---|
| Zero-shot or few-shot prompting of an LLM, or frozen similarity (embedding or lexical) used to choose decisions | Explicitly banned (not trained on provided data). Frozen similarity is kept only as a dev-time diagnostic floor, never shipped. |
| Rule-based key walker (lexical or numeric matching with hand-written thresholds) | Banned ("purely rule-based or algorithmic"). Fails the strip-the-ML test. |
| Hungarian or one-to-one assignment of test rows to leaves, per-couplet rebalancing of leads across the rows of a key, or smoothing across rows | Banned ("coordinating decisions across test rows that share a couplet or key"). Tempting because each key has about 8.6 test rows. Not used. |
| Estimating lead priors per couplet from test predictions | Banned ("balancing outputs using the distribution of test predictions"). Not needed anyway: balanced training already implements the optimal prior-free rule for balanced recall. |
| Fitting IDF, vocabulary, scalers or normalisers on `key_leads.csv` as a whole | `key_leads.csv` contains the test keys' text; fitting on it is fitting on test. Any fitted statistic uses train-key rows and train-key leads only. |
| kNN or lookup to similar train taxa's paths | Test taxa and keys never appear in train; a lookup is not ML and not transferable. |
| Full fine-tune of DeBERTa-v3-large on about 970 keys | Memorises groups, unstable and over budget; the capacity ladder says stop earlier. |
| Joint cross-encoder as the primary (every row-couplet pair through a large encoder) | About 4x the tokens of the late-interaction design; only viable at base size, which gives up the biggest cheap lever (backbone size). Kept as the diversity member. |
| Small decoder LM with LoRA as the primary | Per-row cost with long descriptions is the largest of all options and the 1-hour limit is tight. Allowed ("small language model") but kept as a dev-time bake-off option only if it clearly beats the encoder on the frozen probe. |
| Retrieval of evidence sentences with a frozen embedder | A hard cutoff can drop the deciding sentence; the learned memory-attention does the same job without that failure mode. Grey on the "frozen pretrained model picks things" reading. |
| GBDT over handcrafted similarity features as the core | Grey ("not enough" for NLP) and unlikely to read comparative couplets. Only acceptable as a minor stacker input. |
| Label smoothing, focal loss, hard-negative mining | No evidence they help here; small noisy data; keep the plain weighted CE. |
| Leaf-level (CRF-style) global softmax over all leaves of the key as the training loss | Off-walk leads are unlabelled, not negative, so it injects false negatives. Roadmap only as a measured experiment, not the primary. |
| Pseudo-labelling or self-training on test rows | Banned. |

## Fixed work plan & runtime budget

All numbers are design estimates for one A10G (24 GB). They have not been profiled. The implementer must time one epoch of each stage locally, then hard-code the counts. No branch may depend on elapsed time (CLAUDE.md §3).

**Fixed configuration (initial defaults, to be sanity-checked by repeated CV, not finely tuned).**
- Seeds: split seed 7; model seeds 13, 29, 47 for the final refits. `PYTHONHASHSEED`, `random`, `numpy`, `torch`+cuda, `DataLoader` generators fixed. `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `benchmark=False`. If `CUBLAS_WORKSPACE_CONFIG` is set, set it before importing torch.
- Device `cuda`, bf16 autocast, fixed `num_workers` (0, since the cached states are tensors), fixed thread counts.
- Folds: 5 grouped folds, 1 split seed in the script.
- Probe epochs (rung 0): 6 on cached states. LP-FT epochs (rung 1): 3 (two rungs, fixed).
- Top layers unfrozen: 4 of 24. LR head about 3e-4, top layers about 1.5e-5 with layer decay 0.9. Weight decay 0.01. Warmup 10%, cosine. EMA 0.99. Memory token dropout 0.1. Batch 16 rows. Gradient clip 1.0.
- Lead-order permutations at inference: all (2 for binary, 6 for ternary).

**Stages and estimated A10G time (UNVERIFIED).**

| Stage | Work | Est. time |
|---|---|---|
| 0 | Read csvs; assert schema; build tree (couplets, parents, children, subtree sizes); parse paths; build groups (shingles, union-find) | 1 min |
| 1 | Tokenise; frozen forward through lower layers once for all descriptions (about 5.5k rows, about 2.5M tokens) and all couplets of all keys (about 6.4k couplets, about 0.6M tokens); cache layer-(L-4) states on GPU/CPU (about 5-7 GB in fp16) | 3 min |
| 2 | 5-fold grouped CV: rung 0 (6 probe epochs, head only) then rung 1 (3 epochs, top 4 layers + head), per fold about 3 min | 15-16 min |
| 3 | In-script selection: rung choice by cross-fitted OOF J; optional 4-coefficient stacker (only if the roadmap gate passed at dev time) | 1 min |
| 4 | Final refit on 100% of train with the chosen rung, 3 seeds, about 3.5 min each | 10-11 min |
| 5 | Inference on 482 test rows (all couplets of their keys, all lead permutations, 3 seeds): head cost only, the backbone states for couplets are cached | 1-2 min |
| 6 | Validate (see below), write csv, re-read | under 1 min |
| Total | | about 32-35 min, about 30%+ headroom on a 50-min target |

**Memory estimate.**
- Cached states about 5-7 GB (fp16).
- Training top 4 layers with batch 16 rows (each row plus its on-walk couplets, memory at most about 768 tokens) is under 12 GB.
- Peak under 20 GB, no OOM at the largest batch if the memory cap and batch are fixed (the cap is derived from the train token-length quantile).

**Input and output validation inside the script.**
- Assert the required files and columns exist.
- Assert every `goes_to` and lead label is parseable, and the key is a tree rooted at couplet 1.
- Output validator (adapted from CLAUDE.md §5): columns exactly `id,decisions`; row count and id order equal to `sample_submission.csv`; for each row, the set of couplets answered equals the key's couplet set (the sample picks lead a everywhere, so its couplet set is the reference); every token is a real lead of that key; one token per couplet; no empty cell; no NaN. Raise before writing a broken file. Reload with `keep_default_na=False` and recheck.
- Print only logs for time; no time in any condition.

## Metric-aware training & decode

**(i) Back-solving the metric.**
- J_c is balanced recall across the leads taken at c, so per couplet every lead that is taken matters equally, independent of how many rows take it.
- The Bayes-optimal rule for balanced recall is argmax over leads of P(b | x) / π_b, which equals argmax of the likelihood P(x | b). It does not depend on the unknown test prior π_b.
- Hence train with weights that make every lead count the same inside its couplet. The model then outputs prior-neutral scores and no prior correction at inference is needed. No test-based prior estimate is used or needed.

**(ii) Per-example weights (the metric's own).** For a training example (row r, couplet c, true lead b), with n_{c,b} = number of train rows taking b at c, R_c = Σ_b n_{c,b}, and M_c = the number of leads of c (2 or 3):
- w = R_c / (M_c × n_{c,b}), then normalised to mean 1 across the training set.
- This gives each couplet a total weight proportional to |R_c| (as in the metric) and splits it evenly across leads.
- For couplets where only one lead is observed in train, the same formula gives w = 1/M_c; these examples still teach the true lead but cannot be scored in the real test (m_c = 1).
- A safety cap on w (for example at 8) is a numeric safeguard, not a tuned constant.
- Flat across keys, like the metric (no per-key normalisation).

**(iii) Output type.** Per-couplet softmax over the leads (a categorical decision among 2-3 mutually exclusive options), trained by cross-entropy. This matches the choice among leads; no ordinal or cumulative structure exists.

**(iv) Expected utility.** The decision at one couplet does not interact with the decision at another (each is scored on its own and recall at c depends only on c's decisions). The expected metric per couplet is maximised by the balanced-recall rule above, so the primary decode is a plain argmax of the permutation-averaged balanced model. No decode constants are fitted for the primary.

**(v) Diagnostic only on OOF.**
- Report recall for each lead position per couplet and the mean per-lead recall gap.
- If a systematic skew exists (for example always favouring terminal leads), allow a single scalar offset on the margin, cross-fitted, and adopt it only if it helps across folds and split seeds.
- Otherwise drop it.

**(vi) Within-row tree decode (optional, roadmap step 5, gated).**
- For each (couplet, lead) take the row's own model scores at the descendant couplets and compute size-free subtree-evidence features: the best path score (mean, not sum, of per-couplet log-likelihood ratios), the mean margin at the child couplet, and the local margin.
- Combine them in a tiny logistic stacker (at most 4-6 coefficients) trained on cross-fitted OOF, validated through an outer cross-fit.
- Adopt only if the gain on V4 and V4-sparse exceeds noise and the gain does not shrink on the deepest train keys.
- It uses only the row's own key, so it is not coordination across rows. It is still optional; the primary does not depend on it.

**(vii) Hard constraints.**
- Every couplet of the row's key gets exactly one lead of that couplet (enforced at the formatter, not by a search).
- No other constraint exists; in particular do not enforce path consistency across ancestors, because off-walk answers are free and ignored.

**(viii) Calibration.**
- Only needed if two families or two rungs are combined: use z-scored (or train-reference-ranked) margins, one blend weight fitted on OOF, cross-fitted.

## Structural signals

Each invariant is turned into a model input, an augmentation, a loss or a validator, with the check noted.

1. **Permutation symmetry of the leads inside a couplet.** The label is the lead's identity, not its letter, so the scorer is permutation-symmetric. Training randomly permutes lead order; inference averages all permutations. Diagnostic: check swap invariance on OOF (the margin averaged over permutations should be antisymmetric; the gap between permutations measures the residual position bias). The lead letter is never a feature.
2. **Tree structure of the key.** Leaves (taxa), couplet depth and subtree size are input facts of the row's own key. Uses: (a) the walk validator; (b) optional scalar inputs, gated and dropped-out; (c) the optional within-row tree decode. Never fit on test keys.
3. **Catch-all and elimination leads** ("plants not combining the above features", "other"). The learnable null memory vector and the balanced weights let the model learn "choose this lead when no evidence supports the sibling". Check on the OOF per-lead-type recall.
4. **Description fields as different evidence sources.** Habitat, distribution and phenology get explicit field markers and are encoded in the same memory. Leads about habitat or range then attend to those spans. Diagnostic: J on habitat/range-type leads, with and without the fields.
5. **Numeric and range agreement.** Number/unit parsing (regex for extraction only) gives pair features: whether a description value falls in the lead's range, the distance to the range, the unit match, trait-word overlap, counts of numbers. Fed to the scorer, never a rule. Gate on V4 numeric-type couplets.
6. **Reference-normalised evidence.** Raw similarity is dominated by common botanical vocabulary ("leaves", "margin", "cells"). The cross-attention head learns what is specific to a lead relative to its sibling because the leads are encoded jointly and scores compete in a softmax. An explicit train-bank likelihood-ratio feature is not planned (bank would need train descriptions only, would add machinery before it is measured).
7. **Depth-first numbering and sub-key duplicates.** Use the numbering only to parse the tree. Do not use couplet number as a feature (shortcut to depth and key size that does not transfer to larger test keys), apart from the gated depth scalar.
8. **Symmetry that does not exist.** No flip-like symmetry of the description exists. Do not invent one. Sentence order within a description is not label-bearing; sentence shuffling is a possible mild augmentation to test, not assumed.
9. **Opaque `Txx` labels** are meaningless by statement. They are only used to detect leaves when parsing the tree.

## Experiment roadmap

Ordered, each with a stop criterion. One change at a time, paired on the same folds, repeating the split seed. Keep a table (id, change, V4 mean ± std, V-all, V4-sparse, per-fold, per-depth, runtime).

1. **Contract, metric and validation correct and unit-tested.**
   - Parse the keys; implement the exact metric; run the unit tests (perfect, constant, reversed, base-rate, prior rule, gold-score oracle through the formatter); build groups and report the largest group share.
   - Stop when the gold oracle gives 1.0, "always a" gives about 0, and the group sanity passes.
2. **Strongest cheap baseline, end-to-end and valid.**
   - Frozen backbone plus trained head (rung 0) on cached states with the balanced weights; write a valid csv through the validator.
   - Also report the diagnostic floors: lexical-overlap rule, lead-only control, shuffle control.
   - Stop when the end-to-end csv validates and the rung 0 V4 clearly beats the lexical floor and the controls (about 0). If rung 0 does not beat the lexical floor, stop and debug the data flow before adding anything.
3. **Representation and structural insight.**
   - (a) Backbone bake-off on the frozen probe (DeBERTa-v3-large versus roberta-large versus one retrieval-pretrained large encoder; optionally a small decoder LM's middle-layer states).
   - (b) Joint encoding of the couplet leads versus encoding each lead alone.
   - (c) Null memory vector on versus off.
   - (d) Numeric-agreement features on versus off, judged on numeric-type couplets and overall.
   - Keep only changes whose paired gain exceeds noise. Stop at the first design that wins; do not stack unmeasured additions.
4. **Capacity ladder.**
   - Rung 1: unfreeze top k = 2, 4, 8 layers (LP-FT) with the log of the parameter-change norm and train-loss-versus-OOF per epoch.
   - Full fine-tune once, on a single fold, to confirm that it memorises.
   - Stop at the lowest rung that wins. If rung 1 does not beat rung 0 by more than the noise, ship rung 0 (still genuine training of the head, but flag the compliance question; see Open questions).
5. **Metric-aware decode checks.**
   - Per-lead recall skew diagnostic and the optional scalar offset.
   - The within-row tree-decode stacker (gated by the conditions in Metric-aware training (vi)).
   - Test the leaf-level global-softmax loss only here, as a measured experiment.
6. **Diversity.**
   - Score the joint cross-encoder (DeBERTa-v3-base, chunk-MIL) alone on the same folds.
   - Blend only if it is close in quality to the primary; z-scored margins, one weight, cross-fitted.
   - Seeds: confirm that 3 seeds reduce variance.
7. **In-script bounded selection.**
   - Only the two-candidate rung choice, and the stacker if adopted. No Optuna search is planned (too few independent groups to resolve it). If one is added, fixed `n_trials`, seeded TPE, no timeout, nested held-out check.
8. **Final fixed-plan run.**
   - Clean `working/`, the exact command `python3 solution.py ./dataset/public ./working/submission.csv`.
   - Run twice and diff (the margins should be near-identical; large swings mean too few seeds).
   - Check the validator, runtime headroom of at least 30%, memory, and the prediction sanity (balanced use of leads at on-walk couplets, no constant outputs, decision label sets valid).
   - Compare the OOF and test margin distributions as a bug check only, not tuning.

**Credits (6 per problem).**
- Submission 1: the rung 0 baseline (valid pipeline).
- Submission 2: the best single model (rung 1).
- Submission 3: the ensemble (3 seeds, plus the diversity member if adopted).
- Submission 4: the final.
- Keep the remaining credits in reserve for platform glitches. The free CSV check can compare candidates.
- Do not select on the public LB.

**Techniques that two or three independent solvers would plausibly converge on, tested before any proxy tweak:** a larger backbone with a probe, balanced per-couplet weights, joint couplet encoding and a lead-order-averaged decode. These are in the primary by design and verified at steps 2-4.

## Compliance audit

CLAUDE.md §7 red and orange items, then the §B self-audits from the agent file.

- Test file read for anything but one-row inference? No. Test descriptions and the test keys' own lead texts are only forward-passed to produce that row's decisions. Statistics (IDF, group-merge shingles, token-length quantiles, normalisers) are computed from train rows and train-key leads only. Frozen-state caching for test rows is a pure forward pass with no batch statistic.
- Time in an `if`, `while`, `break`, `min()` or a timeout argument? No. Time is only logged.
- `torch.cuda.is_available()`, `os.cpu_count()`, import fallbacks or try/except that change the work? None planned. Fixed device `cuda`; the fallback is a development-time constant change (base size), not a runtime branch.
- Hard-coded constants tuned offline? Only generic design defaults (layers unfrozen, LRs, epochs, dropout) fixed a priori and sanity-checked by repeated CV; no per-lead or per-couplet constants, no copied "best params". The rung choice and any stacker are fitted in-script on train OOF. Reviewer question 5 flags the residual risk.
- External data, synthetic training data, self-hosted weights, GitHub-hosted models, non-allowed libraries? None. The pinned Hugging Face backbone is the only external artefact.
- Strip-the-ML test: remove the trained scorer and what remains is a tree parser, a formatter and optional number extractors, none of which can choose a lead. Score drops to about 0 (the constant-lead floor). Pass.
- Raw-pixel tabular or inference-only? Not applicable; the head and top layers are trained.
- Whole-test aggregation? None. No rank or z-score normalisation across test rows, no label-free prior from test lists, no vocabulary fit on train plus test. The primary decode is a pure function of (row, its key's leads, trained model).
- Cross-row coordination? None (assignment, balancing and cross-row consistency are all excluded; see Rejected options).
- Sibling leakage in features? The only cross-row structure is shared lead texts within a key at training time, handled by key-grouped CV and by training on row-level targets; no feature of a row is computed from its group-mates' labels.
- Hard-coded constants derivable by an in-script search? Rung choice and stacker coefficients: yes (in-script, cross-fitted). The generic defaults are not tuned values; see the open question.
- Source size and readability: plain Python, under 512 KB, no blobs.
- Challenge-specific restrictions honoured? Yes (see the ban table in Compliance regime).
- Comments, seeds, fixed workers/threads: planned.

## Open questions & assumptions

**Reviewer questions (with a plan under each reading).**
1. *Within-row tree-structured decode.* The ban is on coordinating decisions across test rows that share a couplet or key. Is using a row's own model scores at the descendant couplets of its own key to decide an ancestor couplet allowed?
   - Plan if allowed: roadmap step 5, gated by measured gain.
   - Plan if not: the primary already ships without it (independent per-couplet argmax). Cost: the foregone gain, measured at step 5 (UNVERIFIED, probably small).
2. *Numeric/unit regex features fed to the trained scorer.* Is deterministic number extraction plus agreement features acceptable (CLAUDE.md §2.5 says regex for deterministic number extraction is fine, but over-reliance on exploiting the data process is not)?
   - Plan if allowed: include only if the gain exceeds noise.
   - Plan if not: drop them; cost measured on numeric-type couplets at step 3d.
3. *Does a frozen lower stack with the top layers and head trained count as "train or fine-tune a model"?* Plan: yes, since the head and the top layers update and the parameter-change norm is logged. If a reviewer wants deeper tuning, move the unfrozen depth up (k = 8) or run full fine-tuning of the base-size fallback, at roughly 2-3x the cost of the cached design.
4. *Use of the description's stated test facts (walks at least 4 couplets, large keys) to define the validation subset and the sparse-coverage subsampling.* Plan: validation use only; training weights stay train-only. If disallowed even for validation, use V-all and per-depth J instead.
5. *Hard-coded generic defaults* (layers unfrozen, learning rates, epochs): does the guidance about constants "found by code in this script" require an in-script search for these too?
   - Plan: if yes, add a three-candidate in-script selection over the number of unfrozen layers (2, 4, 8) using the already-computed cross-fitted OOF, at about +8-10 minutes.
6. *Backbone allowance.* Is a 435M encoder (and, if the bake-off wins, a small decoder LM of at most about 1.5B) within "BERT-style encoder or small language model"? Plan if not: DeBERTa-v3-base.

**Assumptions to confirm when data is available (all UNVERIFIED).**
- Runtime limit is the CLAUDE.md default (about 50 min target, at most 1 h).
- Internet is allowed for the Hugging Face weights download only.
- Train keys are small and nearly fully covered by rows; test keys are large, with low coverage (derived arithmetic above; check diagnostic 3).
- Every train `path` parses cleanly and follows `goes_to`.
- Description token lengths allow a 384-token chunk with a memory of at most about 768 tokens for most rows (diagnostic 6).
- A few percent of rows or keys are duplicates (diagnostic 8); the merge rule keeps the largest group at most 5%.

**Expected score, with reasoning (estimate, low confidence, no data seen).**
- Floors: constant answers and lead-only scores about 0; a lexical-overlap rule perhaps 0.10-0.25.
- Rung 0 probe: perhaps 0.25-0.40.
- LP-FT primary: perhaps 0.35-0.50 on the V4 proxy.
- A realistic private-LB band after the expected optimism of the proxy (shift to larger, deeper keys): roughly 0.30-0.50. The ceiling is capped by the silent-couplet rate and by key/description disagreement.
- These are guesses from the structure of the task, not from any measurement. Treat them as unverified until diagnostics 7 and the step-2 baseline are run.

**What this plan could not verify.** Every number about the data distribution (rows per key, walk lengths, description lengths, silent rate, numeric prevalence, duplicates) and every runtime and score figure. The implementer must run diagnostics 1-14 first and update this file when a hypothesis is refuted.
