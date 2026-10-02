# Text and NLP tasks (classification, tagging/spans, QA/cloze, seq2seq/generation)

## Recognise
Text in, label/span/sequence out; free-text fields plus metadata; multilingual or domain-specific corpora; OCR'd
historical text; headline/summary generation; slot filling; repair/restoration of damaged text.

## Default build
1. **Contract**: valid output grammar, exact-match vs soft terms, empty cases, per-row vs set outputs; ban list
   (TF-IDF/BM25 as complete solution, regex extractors, test-wide statistics, pretrained model restrictions).
2. **Representation first**: strongest affordable pretrained encoder at the right length (large > base; domain-
   matched; long-context where documents are long). Check tokenizer coverage on the script (unknown-token rate, NFKC
   round trip) *before* choosing a model (Arabic-script Kazakh broke mBERT/XLM-R tokenisation).
3. **Capacity ladder** by distinct-group count: frozen embeddings + trained head → LP-FT → full fine-tune; 3–5 folds
   or multi-seed averaged at the logit level; layer-wise LR decay; warmup; EMA; mean pooling often > CLS.
4. **Task-shaped head**: GlobalPointer/biaffine span scoring when exact boundaries dominate (A13); per-token BIO+CRF
   as a fallback; cumulative-binary heads for ordinal; whole-case-as-one-sequence for unordered snippets (E10);
   cloze with the pretrained MLM head, multi-mask scoring, alias tokens (E9).
5. **Metric-aware decode**: expected utility, closed-form thresholds, same-label persistence for run terms,
   hierarchy masking; a gate for empty outputs.
6. **Augmentation that removes shortcuts**: entity swapping, keyword masking (check literal answer leakage first, G3);
   continued MLM on all in-domain text as warm start (E5).
7. **Diversity**: a second encoder family with a different tokeniser; per-fold models averaged; full refit.

## Generation / seq2seq
- Reduce generation to the smallest enumerable decision when possible (P3). Otherwise fine-tune a seq2seq/decoder
  with fixed beams, n-best + MBR rerank under the exact metric (composite ROUGE/exact/number terms), fixed length
  penalty, no clock-based beam shrinking.
- Validate leave-one-source-out (publisher/project/language); drop story families that straddle the held-out source.
  Never feed the source identifier as an input; sample sources with temperature to avoid one house style winning.
- Metrics with digits/entities: add a penalty for numbers not present in the source; calibrate under both readings of an
  ambiguous rule and ship what is good under both.
- Non-autoregressive profile/vector outputs (e.g. a 64-position distribution) are legitimate when the grammar is
  fixed; write values in fixed decimal notation to satisfy regex graders.

## Retrieval-flavoured text tasks
See `retrieval-ranking-slates.md`: bi-encoder retrieve → cross-encoder rerank → learned features, never TF-IDF alone.

## Low-resource / banned-weights variants
When pretrained weights are banned: train a tokenizer/BPE on the provided text only, small transformers or CNN/GRU
encoders, hashed char n-gram features *as inputs to a trained neural model*, strong regularisation; budget the model to
the group count. OCR/noisy text: char-level features, noise-robust augmentation, batch/engine style signals learned
from train.

## Pitfalls
Truncation silently dropping the answer span (check 99th percentile length; use windows with fractional-coverage
bookkeeping, F9); LoRA with a random head; label smoothing stalls on ~350 rows; mean-pooling every model;
test-text vocabularies; hand-written regex features that already solve the task; plain argmax where the metric is
exact-boundary F; single noisy fold deciding the encoder.

## Research additions (2026-10, /research/C)
- **Outcome-verified self-training (expert iteration) checklist:** one verified target per row (hard-EM style), refit from the base weights each round, cross-input consistency filter against false positives, label-shuffle negative control, train-side only. Measure pass@K minus selected-hit rate (coverage vs ranking gap) before adding rounds. K=4 gives a weak cluster vote.
- **Selection among the model's own samples:** execution-consensus / structure-signature voting with log-prob tie-break; a small learned verifier trained on round-1 samples of non-seed rows is a later lever. Grammar-constrained decoding is grey/high-risk where hand-written production of outputs is banned: off without written approval (CLAUDE.md 2.3A).
- **Base-model arithmetic:** KV bytes/token differ ~9x (Qwen2.5-0.5B ~12 KB vs Qwen3-0.6B ~112 KB) and ~16x (SmolLM2-1.7B ~192 KB): batch sizes collapse with the wrong base. Check gating/licence first.
- **Encoder recipes:** DeBERTa-v3-large default, LLRD, mean pooling, multi-sample dropout, EMA, warmup, seeds averaged at logits, soups; decoder LLM + LoRA + classification head only when compute allows (roughly ties domain encoders at ~1.5k examples).
- **Retrieval/rerank:** positive-aware hard-negative mining from out-of-fold retrievers, same-scenario items excluded, listwise or MarginMSE cross-encoder loss.
