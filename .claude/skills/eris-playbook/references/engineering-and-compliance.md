# Engineering, determinism and compliance patterns

A rejected or failed submission scores below a bad one. These are the checks that actually blocked solutions, with
the source-library advice that **conflicts** with the Deterministic Execution checker flagged **DO NOT ADOPT**.

## 1. Deterministic Execution (fixed work plan)
- Fixed epochs, folds, trials, boosting rounds, beam width, batch size, workers, threads. Time is for `print` only.
  Never in `if`, `while`, `break`, `min()`, or library arguments (`timeout=`, `time_limit=`). Early stopping on a
  validation *metric* is fine; keep patience and max epochs fixed.
- **DO NOT ADOPT** from the source library: placeholder/fallback submissions (H1), time guards that skip folds/steps (B4, "3100 s guard"), tiered run modes
  chosen by flags or runtime pressure (H3), `try/except` around optional model loads that silently drop members (H4),
  `os.environ`-selected members/modes (F8 says never), `cuda if available else cpu` switches. They look protective
  and are exactly what the checker rejects. Pick one plan, assume one A10G (or the stated CPU), hardcode it.
- GPU determinism checklist: `cudnn.deterministic=True`, `benchmark=False`, TF32 off, `use_deterministic_algorithms(
  True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` set before importing torch, math/exact attention if the
  fused kernel is non-deterministic, fixed `Generator` for DataLoader, `num_workers` constant, `torch.set_num_threads`.
  Selective thread limiting (OpenMP for GBDT, 1 BLAS thread) instead of blanket single-threading (F1); LightGBM
  `deterministic=True, force_row_wise=True`; Optuna `TPESampler(seed)`, fixed `n_trials`, `n_jobs=1`.
- Verify by **running twice and diffing** the output (`.claude/scripts/determinism_check.py`), not by reasoning (F6).
  Replace gather/scatter backward with constant incidence GEMMs (F13); integer histograms for scatter (F12).
- Profile per-phase wall-clock to a side file so the runtime claim is evidence (F10); aim ≤ 50 min with ≥ 30%
  headroom against the stated limit (a description limit shorter than 1 h overrides the default).

## 2. Contract and I/O
- `python3 solution.py <public_dir> <submission_out>`: read both from `sys.argv`; no argparse requirements; never
  hardcode `./dataset/public`; `mkdir(parents=True)` the output parent; write only under that parent.
- Reload the written CSV with `keep_default_na=False` in the validator: empty strings vanish into NaN on reload.
- Inline `validate_submission` inside the script; atomic write (temp + `os.replace`) (F4, H2). Validate against the
  description's grammar (JSON, `lifecycle_profile=` prefix, fixed-decimal floats without scientific notation, sorted
  letter pairs, distinct selections), not only against the sample.
- **No placeholder or fallback submissions** (the library's "placeholder-first" H1 is DO NOT ADOPT here): a file
  written from `sample_submission.csv` before training is a constant-output fallback path, which the platform's
  Deterministic Execution/Prompt Compliance checks and human reviewers read as a red flag. Write the submission once, at
  the end, validated; any failure must **raise**. Never write output inside an `except` block.
- Source < 512,000 bytes, plain readable, no base64/zlib blobs, no `exec/eval`, no runtime-generated code, no
  environment probing. Comments explain reasoning; reviewers read the script as training data.
- Cache only inside one run, hash-keyed by every input that determines the result (N1); never reuse model
  checkpoints across invocations (N2). Hash source/output for the audit trail (F5). Assert fractional coverage sums
  to 1 when chunking long inputs (F9). Parity assertions between fast and naive paths (F3).

## 3. What reviewers reject (rejections log R–V, plus rules)
- **R whole-test calibration**: any statistic reduced across test rows and fed back into those rows' predictions
  (EM prior estimation over test candidate lists, test-batch rank/z-score normalisation, test-wide thresholds). Each
  prediction must be a function of its own row and a frozen train-fit model. Use train-reference percentiles (Q7).
  Test: score the file twice, once with half the rows dropped; kept rows' predictions must be identical.
- **S "siblings"**: related-row leakage and cross-referencing group-mates to reconstruct an answer-adjacent
  quantity (canonical token sequences from other mutants of the same case; near-duplicate events across a split);
  a flawless CV on a noisy task is the tell. Group before folding; do not engineer features that compare a row with
  its hidden siblings unless the description explicitly makes them visible and legitimate.
- **T banned primary signal**: wrapping a banned lexical mechanism (TF-IDF/BM25/Jaccard) in extra ML does not
  exempt it when the task bans keyword-only approaches as the complete solution. If a description bans a method, treat
  features derived from it as the same method.
- **U plagiarism/similarity**: distinctive reused documents/techniques; keep original structure and comments.
- **V hand-built extractors**: regex pipelines that pull the dataset's own structure and feed a thin model when the
  task is about learning that structure. The model must learn it from raw sequences.
- **Q1 strip-the-ML test**: remove every trained component; if the remainder still produces a useful answer, too much is
  hardcoded. **Q2** label-free fitting on train+test (vocabulary, TF-IDF, scalers, PCA, clustering, EM) is test
  adaptation. **Q3** constants tuned across real submission rounds and then hardcoded leak through the human: every
  hardcoded number must be traceable to an in-script train-only search or a principled default; keep a log of which
  constants came from submission feedback and re-derive them. **Q4** if decode-time metric tuning is banned but
  training-time rebalancing is not, use a named published technique (logit-adjusted cross-entropy, Menon et al.
  2021) with a CV-chosen offset and ship plain argmax. **Q5** a perfect score on a decoy-slate task is a generator
  tell: run the strip-the-ML test and check whether the model discovered structure or hand rules exploited it. **Q6**
  on engineered-decoy tasks decide lever by lever, ask a reviewer about ambiguous ones (test-history statistics,
  generative decoy likelihood, synthetic slates) before the deadline; the defensible set is train-only statistics,
  repeated-OOF statistics with matched test statistics, label-augmented co-occurrence cross-fitted, learned (generic)
  offset grids.
- **Hard bans** (CLAUDE.md §2.3): external data, self-hosted fine-tuned weights, pseudo-labels/TTA-adaptation/
  calibration on test, fitting anything on test or train+test, synthetic labelled data, private code sharing,
  API/LLM calls, pip install, GitHub downloads, opaque source. Re-read every challenge's "What not to use" and treat it
  as a hard constraint; treat *silence* as an assumption you write down.
- **Synthetic-data boundary (L)**: resampling which slice of a real entity is "observed" and copy-pasting real labelled
  components are defensible; blending values across different real entities into a fabricated hybrid is not.

## 4. Self-audit before every upload (yes to any red item → fix first)
Red: test rows read for anything but one-row inference; clock/hardware/environment in a condition; hardcoded tuned
constants; external data/weights/libraries; strip-the-ML passes; output path writes inside `except`; unpinned or
non-hub model loads. Orange: unreadable source; thin ML wrapper over a rule engine; challenge restriction missed
(model family, size, runtime, from-scratch, TTA ban, per-row independence). Green: seeds fixed, workers fixed, comments
present, double-run diff clean, validator green, runtime profiled.
Run: `compliance_scan.py` → `validate_submission.py` → `determinism_check.py`, then the three reviewer agents.

## 5. Trivially separable or templated data (strip-the-ML tension)
If the data are separable by simple keywords or fixed templates, a rule table would also score perfectly, which is exactly what the strip-the-ML test punishes. Keep a trained model as the
primary and make the discovery *visible*: log learned weights/importances, show that the model recovers the structure from raw inputs (no hand-written keyword list, regex or lookup in the code),
state in a comment that the structure was found by the model, and stress-test it (V15). Never encode the discovered pattern as a rule outside the model.

## Research additions (2026-10, /research/B,C,D)
- **Weight provenance (VERIFIED against torchvision source).** torchvision pretrained weights download from download.pytorch.org, which is NOT Hugging Face/timm and conflicts with CLAUDE.md section 1. Safe path: timm backbones (weights on the HF hub) + torchvision detection/segmentation classes built with random heads, or `transformers` models with HF checkpoints. Audio checkpoints common in write-ups (BEATs, PANNs, EfficientAT, Perch) are hosted elsewhere [R]: assume unusable; AST/wav2vec2/Whisper are on HF.
- **Licences and gating.** ConvNeXt-V2 weights are CC-BY-NC-4.0, layoutlmv3 CC-BY-NC-SA-4.0, DINOv3 carries a custom Meta licence; PaliGemma/Gemma/Llama are gated (an unattended run cannot supply a token; some challenges ban gated models). Safe choices: ConvNeXt v1, EfficientNetV2, EVA02, DINOv2, SigLIP2, AST, Qwen2/2.5. Check the model card before pinning a revision.
- **Library/version hazards.** The Kaggle image's timm may predate new model tags and a missing name is a hard failure (no try/except fallback allowed). Not confirmed present: peft, bitsandbytes, albumentations, segmentation_models_pytorch, ultralytics, ensemble-boxes, POT: verify in a Kaggle run or write plain-torch equivalents.
- **Do not ship inference-only foundation models** (TabPFN/TabICL/Chronos zero-shot, pip-only packages). Use them offline as an information-ceiling yardstick on small-N tasks if at all.
- **Determinism.** adaptive_avg_pool2d backward has no deterministic CUDA kernel; with warn_only=True it will not crash, so double-run diff the output and swap the op if drift matters, without device branches.
- **Generation speed (measured today on a T4).** fp32 master weights with autocast re-cast every decode step were ~2-3x slower than a low-precision copy; static batches run to the longest member (waste ~3-4x on skewed lengths); a 2.4 h sampling stage went unnoticed because nothing was profiled. Profile ONE batch before any full run; budget GPU hours against the weekly quota (30 h/week on Kaggle).
- **Training memory.** Compute the LM head only on target positions (large-vocab logits caused a T4 OOM); keep fp32 master weights or use LoRA rather than pure-bf16 params (silent no-update cancellation) and assert the parameter-change norm > 0.
