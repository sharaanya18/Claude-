# Generation, program synthesis, structured text outputs

## Recognise
Output is a command pipeline, program, JSON object, restored text, repaired record, headline, or step sequence; metric is
exact match, token F1, execution equivalence, or a composite of similarity and exact terms. Unseen *combinations* of
known parts appear in test (unseen adjacent pairs, unseen schemas).

## Default build
1. **Reduce to the smallest learnable decision** (P3): rank legal skeletons/heads; fill slots from the input by
   pointer/copy; generate only the genuinely open field. Verify the reduction preserves all valid answers.
2. **Compositional split first**: build validation that withholds pseudo-unseen combinations (adjacent head pairs, schema
   combos), including as sub-paths; drop 5-word-span neighbours and canonical-output duplicates from training for the
   held-out fold (this mimics the real split; random CV is badly optimistic).
3. **Models**: (a) skeleton ranker with a low-rank adjacency term so unseen pairs are not driven to −∞; (b) stage-factored
   pointer-generator (each stage realised from description + head + slot, no inter-stage history) so unseen adjacency
   never has to be generated; (c) a small seq2seq/LM fine-tune for open text. Pool candidates from several fold models,
   rescore, pick by MBR under the exact metric (e.g. 0.5·similarity + 0.5·exact). From-scratch variants: copy-attention
   GRU/transformer trained on the provided pairs.
4. **kNN retrieval is a dev-only control**, never the shipped primary (strip-the-ML).
5. **Executors as decode**: a deterministic interpreter (shell parser, BPMN token game, schema validator) may *check or
   order* what the model produced; it must not substitute for learning. Keep compliance wording explicit.
6. **Restoration/repair tasks** (galleys, Sanskrit cards, OCR repair): treat as ranking among edit candidates with a
   cross-encoder or MLM scorer; exact assignment likelihood when pieces permute (8! cards); Sinkhorn/assignment decode when
   several pieces compete for slots; calibrate an abstain/keep option.
7. **Open-text generation**: see `text-nlp.md`; fixed beams, n-best + MBR; leave-one-source-out validation.

## Pitfalls
Recombination augmentation of stages/literals may count as synthetic data (ask a reviewer; off by default); favouring
adjacencies absent from train is exploiting the split; grammar-restricting outputs without evidence; tokenization regex
assumptions in the metric (replicate the grader's tokenizer exactly); validators asserting row counts that differ from
the sample file.
