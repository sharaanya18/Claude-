# CLAUDE.md — Project Eris (Shipd) Solver Playbook

You are solving **Project Eris** ML benchmark challenges on Shipd. Each challenge gives a problem description and a dataset (`public/`). You deliver **one self-contained `solution.py`** that trains a real model end-to-end and writes `submission.csv`. Goal every time: a **compliant** solution (rejections and failed checks score nothing) with the **best generalizing private-LB score** the 1-hour budget allows.

Priority order when anything conflicts:
1. **Challenge-specific description** (real restrictions override this file, see §2.4)
2. **Compliance** (rules in §2–§3) — a rejected solution is worth zero, however high its score
3. **Validity** (script runs, `submission.csv` exactly right, deterministic)
4. **Score on the private LB** (generalization via trustworthy CV, diverse ensembles, no public-LB chasing — see §4A)

> Sources: `Project Eris - Solver Guidebook.pdf`, `shipd_docs.docx`, `solution_sub_instructions.txt`, `libs_allowed.txt`, `error_types.png`, and 7 screenshots of real pre-submission check failures. Where sources disagree, this file says which to follow.

---

## 1. Non-negotiable contract

| Item | Requirement |
|---|---|
| Entry command | `python3 solution.py <public_dir> <submission_out>` — read **both** from `sys.argv` (`Path(sys.argv[1])`, `Path(sys.argv[2])`). Do not hardcode paths. For local dev only, fall back to `./dataset/public` and `./working/submission.csv` when argv is absent. |
| Output | Write the CSV to `submission_out` (`mkdir(parents=True, exist_ok=True)` on its parent). Only write under that parent directory (`./working/`). |
| Runtime | Target **≤ 50 min total** on an **NVIDIA A10G (24 GB)**. Official ceiling is 1.5 h (+≤30 min grace, rarely granted); one section says 1 h and some challenges set shorter limits. Plan for ≤ 1 h worst case. Follow any shorter challenge-stated limit. |
| Self-contained | Preprocessing, features, training, inference **all inside the script, from raw data, every run**. No cached embeddings/weights/artifacts from earlier runs, no attached files. |
| Source file | Plain UTF-8, human-readable, **< 512,000 bytes**. No base64/zlib blobs, no embedded weights, no code generated at runtime, no `exec`/`eval` tricks. Opaque code fails the compliance checker ("source cannot be inspected"). |
| Libraries | Only what is in the Kaggle Docker image (core stack: numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, torch, torchvision, transformers, timm, tensorflow/keras, optuna, etc.; see §8). Anything else → ask a reviewer first. |
| Internet | Allowed **only** to download pretrained backbone weights from Hugging Face / timm (`from_pretrained`, `timm.create_model(pretrained=True)`). **Never** GitHub or other sources, never `pip install`. |
| Credits | 6 per problem (+1 per 4 h) and a global ~15–25/day (each refunds 24 h after use). A failed script run still burns the credit, so **validate locally first**. The local "upload CSV" check is free but does not count for the leaderboard. |

---

## 1A. How the platform pays and checks (full detail: `.claude/skills/eris-playbook/references/platform-facts.md`)

- **Payout = clear the AI baseline × top private rank (leaderboard $500: 250/150/100; plus a $150 merit pool among baseline-beaters) × survive the review that happens after the competition ends.** Credits are never refunded; a rejected solution forfeits its placement. Choose the best score *among approaches a reviewer will accept* (compliance-adjusted value).
- The private LB (the remaining part of the test set) decides rankings; public LB is a soft signal. Closing: 24 h countdown after 10 distinct graded solvers, then possibly a short lockdown grace period; have the final solution in early.
- Four automated pre-submission checks: **CSV Score Validation**, **Prompt Compliance** (code read against the challenge's explicit requirements: required/prohibited methods, training, hardware, models, data sources), **Held-out Answer Ingestion** (never touch private/answer files), **Deterministic Execution** (no wall-clock, fallback, worker, seed or backend logic that changes training or inference). A requirements map in the `solution.py` docstring makes the Prompt Compliance check pass legibly.
- No placeholder/fallback submissions, no time guards, no hardware or environment branches: they are exactly what the checks flag.

---

## 2. Rules — what gets a solution rejected

### 2.1 The prime directive: the model must do the learning
The solution is later used to train AI agents, so it must **demonstrate real ML training or fine-tuning**. Inference-only solutions do not count.
- **Strip test:** remove the ML model from your pipeline. If it still basically works, it is non-compliant.
- If you find a pattern/trick in the data, **do not hardcode it** (regex template, lookup table, magic constant). Make the model discover it (features fed to a trained model, learned embeddings, etc.).
- If you tune hyperparameters, **run the HPO inside `solution.py`** (fixed trial count, see §3). Never paste "best params found offline".

### 2.2 Clearly allowed
- Loading **general-purpose pretrained backbone weights** (HF/timm) inside the script, then fine-tuning them.
- Real training/fine-tuning/HPO/ensembling of several models inside the one script.
- Per-sample (or per-batch, BN-style) inference tricks that work in production on one sample at a time: **TTA**, batch-norm at batch level.
- Hand-engineered features **fed into a real model** (for CV: alongside the image into a deep model).

### 2.3 Clearly NOT allowed (reject, no discussion)
1. Pure algorithmic / rule-based solving instead of ML.
2. Training/fine-tuning **outside** the script then loading your own hosted weights.
3. **External datasets** of any kind. Only the challenge data + pretrained backbone weights.
4. Sharing code/approaches privately with other solvers (public `#general` Discord posts are fine).
5. **Using the test set for anything beyond one-sample inference:** pseudo-labelling, test-based reweighting, test-time adaptation, calibrating with test-set distribution, **fitting scalers / imputers / encoders / vocabularies / TF-IDF / PCA / clustering on test or train+test**, test-set quantile stats, dedup/leak tricks using test rows, transductive anything. **Fit every transformer on train only, then `.transform` test.**
6. **Generating synthetic data** (augmentation of real samples within training, e.g. flips/crops, is normal; creating new synthetic labelled data to train on is not).
7. Multiple accounts / team solving.
8. Any LLM-generated content in the *submission data*; the solution must be your own pipeline.

### 2.4 Challenge-specific text
Restrictions in the challenge description (allowed model families, size caps, shorter runtime, "from scratch") **override** this file. But ignore leftover boilerplate that contradicts the guidebook ("no internet", "decoding tricks are fine", "fully rule-based is OK"). Unsure whether something is real or boilerplate → ask a reviewer, don't gamble.

**Defaults that the description routinely overrides** (read the description's Compute / Runtime / "What not to use" lines *before* applying any default in this file; the stricter reading wins):
- **Hardware:** many tasks are CPU-only (e.g. 10 cores, 62 GB, 1.5 h). Then the A10G/AMP/bf16 advice, "assume one A10G" and `device="cuda"` do not apply; hardcode `device="cpu"` and size the fixed plan for CPU.
- **Pretrained weights:** "no pretrained weights" / "from scratch" / "no external checkpoints" variants exist even outside From-scratch labelling. Then no HF/timm weights and no pretrained tokenizers; train everything on the supplied data.
- **Runtime:** a challenge-stated limit shorter than 1 h replaces the 50-min target; keep ≥ 30 % headroom against *that* limit.
- **TTA:** allowed by default (§2.2) but banned whenever the description forbids test-time augmentation, multi-view or cross-row inference; check before using any flip/multi-crop averaging.
- **Per-row independence:** if the description says each row/bag/case must be predicted on its own, treat any pooling, clustering, normalisation or calibration across test rows as banned, even when the data would permit it (§2.3 #5 applies to the whole test set, not just labels).
- **Provenance:** never match test rows back to a source archive, filename, or ID pattern, even when the description only hints at it.

### 2.5 Grey areas (accept risk, or avoid)
- **Regex:** fine for cleaning and deterministic number extraction; over-reliance or exploiting the data-generation process → reject.
- **TF-IDF / n-grams / Markov chains / frequency stats:** reviewer's discretion, may differ per solver. Prefer them only as **extra inputs** to a trained model, never as the core. Default for text: fine-tune a transformer.
- **Tabular model on frozen embeddings** (CV, fine-tuning challenges): grey, not banned. Prefer true fine-tuning.
- Ensembling a fine-tuned model with a non-DL model on a "Fine-tuning" challenge: grey; only accepted if genuinely strong.

---

## 3. Determinism: the check that has actually blocked submissions

Pre-submission runs four checks: **CSV Score Validation**, **Prompt Compliance**, **Held-out Answer Ingestion**, **Deterministic Execution**. Deterministic Execution failed repeatedly in the screenshots. Cause each time: **wall-clock-dependent control flow.**

**CONFLICT RESOLVED:** the guidebook recommends a time safeguard that stops training at ~3000–3300 s. The automated checker rejects exactly that ("changes training or inference plan based on runtime conditions"). **Follow the checker. Do NOT branch on elapsed time.** Real failures flagged:

| Flagged pattern | Why rejected |
|---|---|
| `if elapsed() > TRAIN_CUTOFF_SECONDS * 0.6: break` in HPO loop | which configs get evaluated depends on speed |
| `if elapsed() > CUTOFF and best_fold_model: use fold model instead of retrain` | different final model depending on time |
| `if elapsed() > ... and i > len(train)*0.3:` partial-data training | training data amount depends on time |
| `if elapsed() > ABSOLUTE_DEADLINE:` fallback prediction path | inference quality depends on time |
| `if elapsed() > HARD_CUTOFF:` shrink beam width | decoding breadth depends on time |
| `budget_ok = lambda: elapsed() < TIME_BUDGET` breaking an HPO loop | time-dependent hyperparameter selection |
| Even a deadline parameter that is `None` by default | flagged as present-but-inactive; the mechanism itself draws scrutiny |

**Rules:**
1. **Fixed work plan.** Fixed number of epochs, folds, HPO trials, boosting rounds, beam width, batch size, `num_workers`, thread counts. Choose them so the whole run is ≈ 35–45 min on A10G with ≥ 30 % headroom.
2. **Time may only be used for logging** (`print(f"[{time.time()-T0:.0f}s] ...")`). Never in an `if`, `while`, `break`, `min()`, or a library `timeout=`/`timelimit=` argument (no Optuna `timeout=`, no `time_limit` in any tuner/boosting API, no `early_stopping` driven by wall time).
3. **No environment-dependent fallbacks.** No `try: import X except: use Y`, no `if torch.cuda.is_available()` model/size switches, no `os.cpu_count()`-derived worker counts or batch sizes. Assume one A10G; write `device = "cuda"` and a fixed config. (If you must support a CPU dev run, do it with a hardcoded constant at the top of the file that you flip locally and **set back before submitting**.)
4. **Seed everything** and fix the sampler: `random`, `numpy`, `torch` (+`cuda`), `PYTHONHASHSEED`, `Generator` objects for `DataLoader`, library `random_state`/`seed`. Optuna: `TPESampler(seed=…)`, fixed `n_trials`, `n_jobs=1`. Boosters: `seed`, `deterministic`/`n_jobs` fixed.
5. **Early stopping on a validation metric is fine** (the plan is data-determined, not clock-determined). Keep `patience` and max-epochs fixed.
6. Deterministic torch without crashing: `torch.backends.cudnn.deterministic=True; benchmark=False`. Use `torch.use_deterministic_algorithms(True, warn_only=True)` (hard-mode raises `RuntimeError` on some CUDA ops). If you set `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, set it before importing torch.
7. Keep the fixed-work estimate honest by **profiling locally**: time one epoch / one trial, multiply, compare to the A10G budget, then hardcode the counts.

---

## 4. Standard workflow for every new challenge

0. **Use the `.claude/` system** (map in §12). For a new challenge run `/eris-solve` (full pipeline) or at least: `/eris-contract` → `eris-data-auditor` + `eris-metric-engineer` → `eris-strategist` (writes `challenges/<slug>/reports/eris_plan.md`) → `eris-validation-architect` → `/eris-baseline`. Implement from the plan and keep it updated as experiments confirm or refute it.
1. **Read the description twice.** Extract: task type, metric (and direction), submission columns/order/dtypes, row count, ID column, runtime/model-size limits, allowed/forbidden methods, "from scratch" or "fine-tuning" labelling.
2. **Classify the domain** → pick the playbook in §6 and the compliance regime in §2/§7.
3. **EDA (quick, in a scratch notebook, not in the final script):** shapes, dtypes, target distribution/imbalance, missingness, duplicates, ID/ordering leakage, train-vs-test distribution shift (adversarial thinking only; do not adapt to test), group/time structure, text lengths/image sizes/sequence lengths.
4. **Decide the validation scheme to mirror the test split.** Random stratified K-fold; `GroupKFold` if rows share entities; time-ordered split for temporal data. A trustworthy CV beats a lucky public LB (Private LB decides prizes, §1.2 of guidebook).
5. **Baseline first** (simple, valid, end-to-end, fast) → check CSV format → record CV score. First submission = solid baseline.
6. **Iterate with a log**: one change at a time (features → model → HPO → ensemble). Keep CV table. Submit 3–5 progressive versions if you have credits (the platform wants to see progression).
7. **Freeze the final plan** (fixed counts, §3), full end-to-end run from clean `working/`, validate output (§5), then upload.
8. **Comments in code explain reasoning** at each step (checklist item). Concise but real.

---

## 4A. Maximising the PRIVATE leaderboard (the one that pays)

The public LB is a small, noisy slice of test; the private LB decides prizes and is hidden until the end. Treat **CV as the source of truth** and the public LB as a weak sanity check (a soft signal only; a gap between the two is normal and not a reason to change course).

**Validation that predicts private**
- Reproduce the *way the test split was made* using only the description and train data: random vs stratified vs group vs time. Mirror it exactly in CV. A split mismatch is the #1 cause of private-LB drops. Do not inspect or adapt to test distributions inside the pipeline (§2.3 #5).
- Use enough folds/repeats that the CV standard error is small relative to the gains you chase: 5 folds minimum, **repeated CV (2–3 seeds of the split)** when data is small or the metric is noisy. Report mean ± std.
- Only accept a change if the **CV gain exceeds the CV noise** (rule of thumb: > 1 std-error of fold means, or consistent across all/most folds and across split seeds). Reject "improvements" that only show up on the public LB.
- Keep a **holdout sanity fold** that never touches HPO or blend-weight fitting once before the final run, to catch overfitting to CV itself. With HPO, prefer nested or at least fold-disjoint evaluation of the final config.
- Check CV–public-LB agreement across your submissions: if rank order disagrees, trust CV unless you find a split mismatch.

**Generalise, don't overfit**
- Favour **more regularised, more diverse ensembles** over one heavily tuned model: multi-seed × multi-fold × 2–4 diverse model families, averaged. Variance reduction is the most reliable private-LB gain.
- Limit HPO search space and trial count to what CV can resolve; many trials on a small/noisy CV *selects noise*. Prefer robust plateaus over a sharp optimum (smooth, mid-range params; no extreme values).
- Blend weights: fit on OOF with non-negativity / sum-to-one, or just use simple/rank averages. Avoid many free weights on small OOF sets.
- Post-processing (thresholds, clipping, calibration of *OOF-derived* mappings) must be tuned on OOF only, be low-dimensional, and be shown to help across folds; otherwise drop it.
- Avoid features/encodings that are fragile to distribution shift (high-cardinality IDs, row order, file order, target-leaky aggregates). If a feature is great in CV only because of a train-specific artefact, it will not survive private.
- Early stopping on validation folds is fine, but when the final model is retrained on all data, use a **fixed** step count derived from the CV runs in the same script (e.g. mean best iteration × 1.1), never a clock.
- Choose losses/targets that match the metric (log1p for RMSLE, ranking loss for AUC-style objectives, focal/weighted losses only if the metric rewards it); verify on OOF with the **exact official metric** implemented from the description.

**Experiment discipline**
- One change per experiment; log each in a table (id, change, CV mean±std, per-fold, est. runtime, public LB if submitted, notes). Never compare scores from different CV splits.
- Order of work by expected private gain per hour: (1) correct validation + metric → (2) strong baseline GBDT/pretrained model → (3) key features/representation → (4) loss/target/augmentation fit to the metric → (5) model diversity → (6) seeds/folds ensemble → (7) HPO → (8) micro post-processing. Do not start (7)–(8) before (1)–(4) are solid.
- Use submissions strategically: baseline (valid pipeline) → best single model → ensemble → final. Don't burn credits on tiny tweaks; the free CSV check can compare candidates against the public LB with no credit.
- Final run must reproduce the best CV config **from scratch in one command** with the fixed plan; verify the produced CSV's distribution looks like the OOF predictions (mean/std/class balance), as a bug-catching sanity check, not tuning.

**Robustness checks before the final submit**
- Prediction sanity: no constant outputs, class balance close to train priors where expected, ranges valid.
- Stability: run the full script twice (or two seeds) and confirm the submission is near-identical or equal in quality (CV same, correlation of test preds ~1); large swings mean too few folds/seeds.
- Runtime headroom ≥ 30 % on a clean run; no OOM at the largest batch. A crash or a late timeout loses everything the CV promised.

---

## 5. Submission-validity checklist (run before every credit is spent)

A malformed CSV scores **dead last, below a score of 0**, and still costs a credit.

- [ ] Run the *exact* command: `python3 solution.py ./dataset/public ./working/submission.csv` on a **clean** `working/` dir; exits 0.
- [ ] Columns, **order**, and **names** identical to `sample_submission.csv`; same row count; IDs match `test` ids exactly (same order unless told otherwise); no duplicate IDs.
- [ ] No NaN/inf/empty values. For string predictions (text, labels, JSON), empty strings turn into NaN on CSV reload, so ensure non-empty and verify by reloading with `pd.read_csv(..., keep_default_na=False)`.
- [ ] dtypes/range: probabilities in [0,1] when asked, integer labels as ints, class labels from the allowed set.
- [ ] `to_csv(index=False)`.
- [ ] Final-validation function in the script asserts all of the above against `sample_submission.csv` and **raises before writing a broken file**.
- [ ] No time-dependent branches, no env-dependent fallbacks (§3); seeds set.
- [ ] Source < 512 KB, plain readable code, no hardcoded tuned params / external data / test fitting.
- [ ] Only imports from the allowed stack; weights only from HF/timm.
- [ ] Wasn't the CSV/solution from a *different* challenge (a "wrong CSV columns" error came from this).

Reusable validator (paste into every solution):

```python
def validate_submission(sub: pd.DataFrame, sample_path: Path, test_ids=None) -> None:
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    id_col = sample.columns[0]
    assert sub[id_col].astype(str).tolist() == sample[id_col].astype(str).tolist(), "id mismatch/order"
    for c in sample.columns[1:]:
        s = sub[c]
        assert not s.isna().any(), f"NaN in {c}"
        if pd.api.types.is_numeric_dtype(s):
            assert np.isfinite(s.to_numpy(dtype=float)).all(), f"non-finite in {c}"
        else:
            assert (s.astype(str).str.len() > 0).all(), f"empty string in {c}"
```

---

## 6. Domain playbooks (aim for top-tier, stay compliant)

General principles for **every** domain: strong CV; ensemble several diverse models/seeds/folds (all inside the script, fixed count); out-of-fold (OOF) predictions to pick blend weights (fit weights on OOF only, never on test); optimize the *actual metric* (threshold/post-processing tuned on OOF, not test); average fold models or retrain on full data with a fixed number of steps; use mixed precision (`torch.autocast` bf16/fp16) and gradient checkpointing for A10G speed/memory.

### 6.1 Tabular (classification / regression / ranking)
- Compliant: GBDTs and NNs are fine here.
- Strong recipe: LightGBM + XGBoost + CatBoost (GPU where useful), + optionally a well-regularized MLP/FT-Transformer for diversity; **5–10-fold CV, multi-seed**, blend on OOF with a constrained optimizer (non-negative weights) or rank-average.
- Features: target-aware encodings **inside folds only** (nested/OOF target encoding to avoid leakage); frequency/count encodings computed on **train only**; datetime parts; ratios/differences of related numerics; group aggregates computed on train; missing-value indicators. Do not compute any statistic with test rows.
- HPO inside script: Optuna, seeded TPE, **fixed `n_trials`** (e.g. 30–60 for GBDT on moderate data), no timeout, `n_jobs=1`; then retrain with the chosen params.
- Imbalance: right metric first (AUC/PR-AUC/F1), `scale_pos_weight`/class weights, threshold tuned on OOF; don't resample test.
- Regression: consider target transforms (log1p) matched to the metric (RMSLE ⇒ train on log1p); clip predictions to training range if sensible.

### 6.2 NLP / seq-to-seq / text
- Fine-tune a pretrained transformer (DeBERTa-v3, RoBERTa, ELECTRA for classification/regression/NER; T5/BART/Flan-T5/small seq2seq or small decoder LMs with LoRA for generation). Download from HF only.
- Tricks: layer-wise LR decay, warmup + cosine/linear schedule, fp16/bf16, dynamic padding + length-sorted batching for inference, 3–5-fold or multi-seed ensembles averaged at logits, max-length chosen from token length stats, gradient clipping, EMA/SWA, pooling choices (mean-pooling often > CLS), multi-sample dropout. Metric-specific post-processing tuned on OOF.
- **Synthetic-data structure:** if text looks templated, the *model* must learn it; don't write regex/templates that solve it. TF-IDF/n-gram models are grey (§2.5); use only as auxiliary inputs if at all.
- Generation: fixed decoding config (fixed beams/length penalty; **no time-based beam shrinking**).
- Beware the OOM/`RuntimeError` risk: choose batch size and max length with headroom for 24 GB, fixed.

### 6.3 Computer vision / detection / segmentation
- **Never feed raw pixels to a tabular model** (immediate rejection). Train/fine-tune a **CNN/ViT** (timm: convnext, efficientnet, swin, regnet, eva; torchvision detection/segmentation heads; SMP-style models only if the library is available, otherwise implement/ use torchvision).
- Hand-crafted image features (color hist, edges) are OK **only fed into the deep model** alongside the image; tabular-only use is grey.
- Tabular model on frozen backbone embeddings is grey; prefer fine-tuning (possibly adding an embedding-GBDT as a minor ensemble member only if it adds real value and the challenge isn't labelled Fine-tuning).
- Recipe: strong augmentations (RandAugment/Mixup/CutMix/Albumentations-style implemented with torchvision if albumentations is unavailable), AdamW + cosine, EMA, label smoothing, progressive resizing, channels_last + AMP, **TTA (flip/multi-scale) is allowed**, K-fold ensemble. Fixed `num_workers`.
- Detection: fixed NMS thresholds, score threshold tuned on OOF/val, not test.

### 6.4 RAG / retrieval / ranking
- Frozen off-the-shelf embedding model for the **retrieval** step is OK. The **ranking/generation** layer must be genuinely **trained** (cross-encoder re-ranker or fine-tuned bi-encoder, learned scoring head). Hand-engineered features + off-the-shelf LambdaMART alone is **not enough**. Fully frozen end-to-end pipeline is not allowed. Fine-tuning the retriever is recommended.
- Recipe: fine-tune bi-encoder with in-batch + mined hard negatives (mined from train only) → retrieve top-K (fixed K) → train cross-encoder re-ranker on train positives/hard negatives → fixed blend. BM25-style lexical signals as extra input are grey: okay as a feature to a trained model, not the core.

### 6.5 Time series / forecasting
- Respect time order in validation (rolling-origin / blocked CV). No future leakage in lag/rolling features; compute lag features per entity strictly from the past; for test, only use information legitimately available at forecast time **per the challenge description** (and never fit on test).
- Models: GBDT with lag/rolling/calendar features, plus a neural sequence model (N-BEATS/TFT-style/LSTM/transformer) for diversity; direct vs recursive multi-horizon strategy chosen on CV.

### 6.6 Audio / signal
- Spectrogram (mel) + pretrained audio CNN/AST from HF/timm, or train from scratch if "From-scratch". Same CV/ensemble discipline.

### 6.7 Fine-tuning-labelled challenges
- **Genuinely fine-tune** a domain-appropriate pretrained model (e.g. ChemBERTa/MolFormer for chemistry, ESM/ProtBERT for protein, BioBERT/PubMedBERT for biomedical text). GBDT alone is not tolerated; hand-engineered features may be extra inputs to the model head. Frozen-embedding + GBDT is grey/likely rejected.

### 6.8 From-scratch challenges
- **No pretrained weights at all** (no embeddings, no distillation, no pretrained retrievers). Train on provided data only, same time limits (no extra time). Tokenizers/BPE are tolerated unless the challenge forbids or the reviewer objects. For safety: train a tokenizer on the provided training text yourself, or skip pretrained tokenizers when the challenge is strict.

### 6.9 Biology / chemistry / other without their own category
- Bucketed under NLP or Fine-tuning by the challenge. If NLP-like: regex / TF-IDF / tabular on features (fingerprints, k-mers, descriptors) are relatively acceptable since the domain has real structure, still the reviewer decides. If labelled Fine-tuning: §6.7 applies strictly. RDKit/biopython may not be installed: check against the allowed list or ask.

---

## 7. Compliance self-audit (do this before every upload)

Answer each honestly; any "yes" to a red item → fix first.

- 🔴 Does any code path read the test file for something other than producing predictions of one sample at a time? (stats, vocab, scaler fits, clustering, dedup, rank-normalising over the test set, pseudo-labels)
- 🔴 Is there any `time.time()`/`elapsed()`/`perf_counter()` inside an `if`/loop condition/argument? (§3)
- 🔴 Any `torch.cuda.is_available()` / `os.cpu_count()` / import-fallback / try-except that changes models or work?
- 🔴 Any hardcoded constants that were tuned offline (params, thresholds, class weights, magic offsets) rather than found by code in this script?
- 🔴 Any external data, synthetic generated training data, self-hosted fine-tuned weights, GitHub-hosted model, or non-allowed library?
- 🔴 Would the solution still "work" with the ML model removed (rules/regex/lookup)? Raw-pixel tabular model? Inference-only?
- 🟠 Is the source readable, < 512 KB, free of encoded blobs?
- 🟠 Does the model-heavy part genuinely dominate (not a thin ML wrapper over a rule engine)?
- 🟠 Did I honour every challenge-specific restriction (model family/size/runtime/from-scratch)?
- 🟢 Are comments explaining reasoning present? Fixed seeds? Fixed workers/threads?

---

## 8. Allowed libraries (from `libs_allowed.txt` + core Kaggle stack)

`libs_allowed.txt` lists *additional* notable packages present in the image. The core Kaggle packages (numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, nltk, opencv, pillow, etc.) are also present; the creator docs explicitly name pandas, numpy, scikit-learn, xgboost, lightgbm, TensorFlow, PyTorch. If a library is neither listed nor core Kaggle, **ask a reviewer first**.

Listed highlights:
- **DL:** torch, torchvision, torchaudio, torchdata, torchmetrics, torchinfo, torchsummary, torchtune, torchao, pytorch-lightning, pytorch-ignite, kornia, transformers, datasets, tensorflow, keras, keras-hub, keras-nlp, keras-cv, keras-tuner, tf_keras, tensorflow-datasets/-hub/-text/-io/-probability/-decision-forests, jax/jaxlib (+cuda12), onnx.
- **Classic ML / tabular:** scikit-learn (+intelex), catboost, TPOT, optuna, bayesian-optimization, scikit-optimize, scikit-multilearn, scikit-plot, scikit-surprise, category-encoders, featuretools, Boruta, rgf-python, h2o, hep-ml, deap, cesium, lime, ray.
- **NLP:** gensim, fasttext, langid, emoji, Janome, PyArabic, fuzzywuzzy, pytesseract, easyocr.
- **CV / medical / geo:** SimpleITK, pydicom, openslide, Wand, ImageHash, mne, nilearn, dipy, vtk, shapely, fiona, Cartopy, geojson, libpysal, haversine, gpxpy.
- **Other:** pandas-profiling / ydata-profiling, pandasql, igraph, Rtree, boto3, docker, kaggle, kaggle-environments, google-genai / google-cloud-* (not for use as an external model API: this violates the no-LLM / no-inference-only intent).
- Not in the explicit list (verify before use): `timm` (guidebook says downloading from timm is fine, so it is treated as available), `segment-anything` via git is listed but GitHub model installs are otherwise banned, so avoid.
- Pretrained weights: only via HF Hub / timm. HF cache location may be redirected (`os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")` with `WORK_DIR = submission_out.parent`) — set before importing transformers; also `TOKENIZERS_PARALLELISM=false`.

---

## 9. Frequently seen failures and fixes (from `error_types.png` + screenshots)

| Symptom | Likely cause | Fix |
|---|---|---|
| Exit code 2 | `argparse` rejected platform args | Use `sys.argv[1]/[2]` (or `parse_known_args`); don't require extra flags |
| `FileNotFoundError` | Hardcoded path / missing asset | Resolve everything relative to `public_dir`; assert required files exist up-front |
| Wrong CSV columns | Submitted CSV/solution from another challenge | Read `sample_submission.csv` and enforce columns/order/IDs/rows (§5) |
| Missing prediction values | Empty strings became NaN on reload | Ensure non-empty; reload with `keep_default_na=False` |
| Generic `ValueError` | Schema, malformed JSON, tensor shape, invalid output value | Add stage/row context, keep full traceback, reproduce with exact source and command |
| Generic `RuntimeError` | CUDA deterministic-op conflicts, OOM, unavailable device, thread init | Test on A10G-like setup; `use_deterministic_algorithms(True, warn_only=True)`; fixed batch sizes with headroom |
| "Script grading failed. Solution evaluation failed unexpectedly. Try again shortly." | Transient platform/server fault or silent crash | Resubmit rather than wait; check runtime and memory; rare AWS crash → report |
| Deterministic Execution rejected | Time-based stopping, env fallbacks, dynamic worker counts, unseeded ops | §3 — fixed seeds, epochs, batches, workers, threads, models; never change work by elapsed time |
| "Deterministic execution could not be checked" | Checker timeout/glitch on complex source | Simplify, retry after ~5 min; keep one plain code path |
| Source cannot be inspected | Oversized, encoded, compressed, generated, over-complex source | Plain UTF-8 `solution.py`, < 512,000 bytes |
| Prompt Compliance could not confirm | External/pretrained data, test fitting, forbidden methods/metadata, code the checker cannot reason about | Only permitted data/methods; remove test adaptation and opaque paths; make the code's compliance obvious with clear comments; retry in 5 min (credit is returned) |
| Works locally, fails remotely | Mac vs Kaggle/A10G differences in paths, libs, CUDA kernels, launch args | Clean Linux/A10G replay; Mac success is only a smoke test |
| Score on private LB far below public | Overfit public LB | Trust CV; keep ensembles broad; don't tune on the public LB |

Prompt-compliance and determinism checks may return *check errors* (checker unavailable) rather than real violations; the platform returns the credit and says retry in ~5 minutes. Real violations come with evidence lines (file:line + explanation).

---

## 10. Solution skeleton (copy, then specialise)

```python
"""Solver for <challenge>. Reads public_dir, trains real models inside this script, writes submission.csv.
Compliance notes: all fitting on train only; fixed work plan (no wall-clock branching); seeded; HF/timm weights only."""
import os, sys, random, time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")

import numpy as np, pandas as pd
import torch

SEED = 42
N_FOLDS = 5          # fixed plan: tuned so full run ≈ 35-45 min on A10G
N_TRIALS = 40        # fixed Optuna trials, no timeout
EPOCHS = 4           # fixed
BATCH = 32
NUM_WORKERS = 4      # fixed
T0 = time.time()     # LOGGING ONLY — never used in a condition


def log(msg):        # elapsed time is telemetry only
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def main():
    seed_everything()
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    # 1) features/preprocessing: FIT ON TRAIN ONLY, transform test
    # 2) K-fold CV: train models (GBDT / fine-tuned transformer / CNN), collect OOF + test preds per fold
    # 3) (optional) Optuna with fixed n_trials and seeded TPE sampler — HPO happens HERE, not offline
    # 4) blend weights fit on OOF only; post-processing (thresholds) tuned on OOF only
    # 5) build submission from sample columns; validate; write
    sub = sample.copy()
    # sub[<target col>] = final_test_predictions
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv")
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
```

Per-domain training-loop essentials (fixed plan, no time branching): AdamW, linear/cosine warmup schedule, `torch.autocast`, grad-clip 1.0, fold-wise best-checkpoint by validation metric, save OOF, average test logits across folds/seeds, free GPU memory between folds (`del model; torch.cuda.empty_cache()`).

---

## 11. Working style for this repo

- Develop in the repository working directory; keep scratch notebooks/EDA out of the final `solution.py`.
- Local layout to mimic the platform: `dataset/public/{train.csv,test.csv,sample_submission.csv,...}` and `working/submission.csv`.
- `device="cuda"` / A10G are the defaults only; a CPU-only or other-hardware statement in the description overrides them (§2.4).
- If there is no GPU in the dev sandbox, smoke-test on a tiny subset by editing top-of-file constants, then **restore the full fixed plan** and record the expected A10G runtime estimate in a comment. Never ship a smoke-mode branch.
- Before declaring a solution done, report: CV metric (per fold + mean±std), the fixed-work plan and estimated A10G runtime, compliance self-audit result (§7), and validator output (§5).
- Do not copy Kaggle competitions or use LLM outputs as submission data. Never paste competitor or template code: the pattern library under `.claude/skills/eris-playbook/` is distilled hypothesis material (from the shipd_env pattern files and rejection log); write original, commented code.
- Prefer iterating 3–5 submissions: baseline → features → model/HPO → ensemble → final.

---

## 12. The `.claude/` system (map)

| Piece | Where | Use |
|---|---|---|
| Pipeline skills | `.claude/skills/eris-*/SKILL.md` | `/eris-solve` (everything), `/eris-start`, `/eris-contract`, `/eris-data-audit`, `/eris-validation`, `/eris-baseline`, `/eris-experiment`, `/eris-error-analysis`, `/eris-plateau`, `/eris-implement`, `/eris-review`, `/eris-presubmit`, `/eris-postsubmit`, `/eris-close`, `/eris-pattern`, `/eris-status`, `/eris-resume`, **`/eris-learn`** (give it a challenge + its top solutions: blind plan, independent digests, gap grading, library update) |
| Pattern library | `.claude/skills/eris-playbook/` | `/eris-playbook` routes to `references/00-s-tier-principles.md`, family playbooks (`families/*.md`), metric/decoding, validation, features, ensembling/training, engineering/compliance. Hypothesis material: validate on grouped CV; conflicts with §3 are marked DO NOT ADOPT |
| Subagents | `.claude/agents/` | `eris-strategist` (plan), `eris-data-auditor`, `eris-metric-engineer`, `eris-validation-architect`, `eris-error-analyst`, `eris-compliance-reviewer`, `eris-runtime-reviewer`, `eris-red-team` (private-LB auditor), `eris-solution-analyst` + `eris-gap-grader` (used by `/eris-learn`), `eris-pattern-researcher` |
| Scripts (stdlib only) | `.claude/scripts/` | `eris_check.sh` (all gates), `compliance_scan.py`, `validate_submission.py`, `make_groups.py`, `determinism_check.py`, `half_rows_test.py`, `test_schema_audit.py`, `local_run.py`, `new_challenge.sh`, `cv_driver_template.py`, `kaggle_gpu_run.py` (Kaggle GPU runs, needs KAGGLE_API_TOKEN), `corpus.py` (corpus scaffold/index/protocol check), `solution_template.py`; report skeletons in `.claude/templates/`; tests in `scripts/tests/` |
| Hooks | `.claude/settings.json` | SessionStart orientation; PostToolUse runs `compliance_scan.py` on every edit of a `solution*.py` and feeds ERRORs back |

Defaults that never change: argv contract, real in-script training, no test statistics, fixed work plan (no clock/hardware/env branches), grouped CV that mirrors the hidden split, exact-metric training and decode,
validated output grammar, reviewers before every submission credit. The challenge description overrides any default (§2.4).
