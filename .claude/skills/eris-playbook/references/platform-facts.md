# Shipd / Project Eris: how the platform actually scores, checks and pays (solver side)

Sources, in order of authority: the challenge's own description > the official Eris documentation and Solver Guidebook > screenshots
of real pre-submission checks (CSV Score Validation, Prompt Compliance, Held-out Answer Ingestion, Deterministic Execution) and platform
error messages > pasted challenge pages. Public web pages on shipd.ai are marketing-level only (bounty ranges, "determinism check,
score validation, overfitting check, code quality, memory usage", community review); nothing there contradicts this file.
`shipd_env` was reference material for *techniques*; its scaffolding (time guards, fallbacks, placeholder writes, tiered modes) fails
current platform checks and is not used.

## 1. What decides your payout
`P(clear the AI baseline) × P(top-3 private rank, or a merit share) × P(survive post-close review)`.
- Pool per challenge ≈ $1,150–1,250: creator $400–500, **leaderboard $500** (documented split 1st/2nd/3rd = $250/$150/$100), **solution
  pool $150 split by merit among solvers who beat the AI baseline**, rubric $100. **You must beat the AI baseline to earn anything.** The
  baseline is a score produced by an AI agent attempting the task at creation time (descriptions often print it, with the board high).
- The **private leaderboard** (remaining part of the test set) decides rankings; the public board is a small noisy slice, a soft signal.
- **Review happens after the competition ends.** A violation found then costs the placement/payout; credits are never refunded, whether a
  rejection stands or is appealed (Discord private channel). A suspiciously high public score may simply be an unreviewed violation: ignore it.
- Therefore: choose the highest-score approach **among those a reviewer will accept**. A grey-area primary that adds +0.01 CV is a bad bet.

## 2. Timeline mechanics
- Once 10 distinct solvers have graded solutions a 24 h closing countdown starts and is visible to everyone; at the end the board is
  finalised. A "lockdown grace period" can follow in which everyone may still edit/submit for a short window and rankings at the cutoff
  decide who continues. Have the final, reviewed solution submitted well before the countdown ends; credits refill 1 per 4 h.
- Credits: 6 per problem (+1 per 4 h, full refill 24 h) and a global ≈ 15–25/day, each returning exactly 24 h after use. Only a real script
  submission spends one; the CSV upload check, drafts, edits and failed pre-checks that say "credit returned" cost nothing. A script that fails
  at run time still burns the credit (an AWS-side crash can be reported/resubmitted). A malformed or incomplete CSV ranks **below a score of 0**.
- Wanted progression: 3–5 submissions (baseline → improvements → best), with comments that explain reasoning at each step.

## 3. The four pre-submission checks (what each really tests)
1. **CSV Score Validation**: your uploaded CSV is graded against the test set (free; only for you).
2. **Prompt Compliance**: a model reads the submitted code against the challenge's *explicit implementation requirements*: required or
   prohibited methods, training, **hardware**, models, data sources. Make compliance legible: a requirements map in the docstring, comments
   at each fit/predict, no opaque code. "Could not be completed" = checker glitch: credit returned, retry in ~5 min.
3. **Held-out Answer Ingestion**: the code must not read held-out answers or private files (do not reference `private/`, `answers.csv`,
   source archives; never try to recover labels from filenames/ids/provenance).
4. **Deterministic Execution**: "runtime conditions cannot change training or inference output". Findings seen with confidence "High":
   `if elapsed() > TRAIN_CUTOFF:` truncating HPO or choosing between fold-model and retrain; `elapsed()` gating partial-data training;
   deadline-gated fallback prediction path; deadline-shrunk beam width; `budget_ok = lambda: elapsed() < BUDGET`; even a dead `deadline`
   parameter draws scrutiny. The check also covers fallback, worker, seed and backend logic. Time used for logging only is accepted.
   Errors "Deterministic execution could not be checked" = retry; "Script grading failed. Solution evaluation failed unexpectedly" = transient, retry.
→ The Guidebook still suggests a 3000–3300 s training cutoff; the automated checker rejects exactly that. **Follow the checker**: fixed work plan.

## 4. Environment and format (unless the challenge says otherwise)
Kaggle Docker image only; HF/timm weights only (never GitHub, never self-hosted fine-tuned weights, never pip); A10G; documented expected max
runtime 1.5 h (+ up to 30 min grace, in practice 5–10 min) while another section caps at 1 h: **plan ≤ 50 min**; a challenge stating a shorter
limit or CPU-only wins. `python3 solution.py <public_dir> <submission_out>`; self-contained and fresh from raw data every run (no cached
embeddings/weights from earlier runs); comments explain reasoning.

## 5. Acceptance regimes by domain (guidebook §4–5) and what to ship
| Regime | Accepted | Grey (risk of rejection) | Banned |
|---|---|---|---|
| CV / detection / segmentation | fine-tuned CNN/ViT/U-Net; hand features fed *into* the deep model | tabular/linear head on frozen embeddings; tabular model on image-derived features only | tabular model on raw pixels |
| NLP / seq2seq | fine-tuned transformer/seq2seq | TF-IDF, n-grams, Markov chains, heavy regex (reviewer discretion, even per solver) | rule/regex/template solving of synthetic structure |
| RAG / retrieval / ranking | frozen off-the-shelf retriever + **trained** ranker (cross-encoder, fine-tuned bi-encoder, learned scorer) | hand features + off-the-shelf LambdaMART alone | fully frozen end-to-end pipeline; inference-only |
| Fine-tuning-labelled | genuine fine-tuning of a domain-appropriate backbone; hand features into the head | frozen embeddings + tabular head; ensemble of fine-tuned model with a GBDT | GBDT/tabular alone |
| From-scratch | training on provided data only; tokenizer/BPE unless the text bans (or train your own) | pretrained tokenizer (reviewer may object) | any pretrained weights, embeddings, distillation |
| Tabular | GBDT/NN/ensembles, in-script HPO | — | — |
| Bio/chem/other | fitted by the bucket they are filed under (NLP vs Fine-tuning); regex/TF-IDF/tabular relatively accepted | still reviewer's call | fine-tuning bucket requires deep fine-tuning |
Always banned: inference-only solutions, external data, synthetic training data, test-set use beyond one-sample inference (pseudo-labels,
reweighting, TTA-adaptation, calibration on test distribution, fitting on test or train+test), hardcoded tuned constants and offline-found
parameters (HPO must run in the script), self-hosted weights, sharing solutions privately, multiple accounts.

## 6. Design consequences for a top private score
1. **Rank by compliance-adjusted value**: among approaches within ~1 SE of each other on grouped CV, ship the one with genuine in-script
   training the regime accepts (LP-FT/real fine-tune, trained ranker); keep frozen-probe/lexical/GBDT members as yardsticks or as
   *minor* ensemble members only when they add measured value. Show the fine-tune is real (assert parameters moved, log the change norm).
2. **Everything tuned is tuned in the script** on train-only evidence (fixed `n_trials`, seeded sampler). Constants found from submission feedback
   are re-derived in-script.
3. **Mirror the private split** (the hidden test is a different slice of the same held-out set); use CV, not the public number.
4. **Beat the printed AI baseline with margin** (it is the payout gate); the reference numbers in the description are your first CV calibration target.
5. **No whole-test statistics, no test-fit anything**, per-row predictions from a frozen train-fit model; the half-the-rows test must pass.
6. **Legibility is score**: a clear requirements map, comments on every fit/predict, small readable functions; reviewers and the compliance model
   read the code. Do not exploit generator structure by hand; make the model learn it.
7. **Credit economy**: validate locally to exhaustion (compliance scan, run twice, validator; the platform has NO free CSV probe: each check is a full credit-costing submission), then spend credits on
   baseline → improved → best; never submit tweaks that CV cannot resolve.
8. **Rubric view of "good"** (task rubrics cover data handling, modelling, feature engineering, training/validation without leakage, code
   quality, communication): handle the stated imbalance/missingness, validate on a holdout/CV that avoids leakage, justify choices in comments.

## Research additions (2026-10, /research/E)
- **What the printed AI baseline probably is.** Agent pipelines in the AIDE family draft a simple first solution without ensembling or HPO, improve by one atomic change at a time and select greedily on one printed hold-out score; MLE-bench-style agents medal in roughly a sixth to a third of tasks, lower on live competitions [S, KompeteAI table]. No public source states how Eris builds its baseline (UNVERIFIED). Strategy: clear it with margin through shift-aware grouped CV, a stronger representation and a diverse refit ensemble rather than harder tuning of the same recipe.
- **Agent failure modes to exploit (and to avoid).** Validation leakage through preprocessing (MLE-STAR: validation 0.819 -> 0.868 while test fell 0.803 -> 0.734 [S]), time handled as categories, shortcut-taking on planted entity overlap (BaitBench: 57% of runs [S]), winner's-curse selection (AIRA-dojo: oracle selection would add 9-17 pp medal rate; top-3 averaging recovers about 10 pp [S]).
- **Hidden-shift designs to expect.** Time, entity, scaffold/cluster, domain, subpopulation, label prevalence; WILDS/Wild-Time show ~20% ID-to-OOD drops; a tuned ERM with an honest selection rule is within ~1 point of most "robust" algorithms (DomainBed [S]).
- **Public sources say nothing about Eris leaderboards or closing; public repos of other solvers' Eris solutions exist and must not be read or copied.**
