# U3 plan: speaker-disjoint listener-vote distribution estimation (from scratch, audio)

Status of evidence: no dataset files were available when this plan was written. Everything under "Data findings" is a diagnostic to RUN plus an UNVERIFIED hypothesis. Numbers marked (est.) are reasoning, not measurements. Sources read: the strategist agent file, CLAUDE.md, and the challenge description U3_listener_vote_audio.md. Nothing else.

## Contract & decision unit

**One valid answer.** A CSV with exactly two columns, `clip_id,response_distribution_json`, 1,334 rows, ids identical to `sample_submission.csv` in the same order. Each cell is a JSON object with exactly the seven vocabulary keys (order from `response_vocabulary.json`: anger, disgust, fear, joy, neutrality, sadness, surprise), values finite, each in [0,1], sum within 1e-6 of 1.

**Invalid vs low-scoring.** Invalid (scores below zero, burns a credit): wrong columns or order, missing or extra keys, NaN, a value outside [0,1], |sum-1| > 1e-6, duplicate or missing ids, empty cells. Low-scoring but valid: anything else, including uniform (score 0 after the clip).

**Metric, term by term.** score = clip(1 - L_model / L_uniform, 0, 1), where L is a sum over ALL test clips of the squared L2 distance, so it is a pooled multiclass Brier score. L_uniform is a fixed constant of the test labels. Consequences:
- No per-clip weights in the numerator: every clip counts equally. The goal is to minimise total squared error to the 5-vote empirical vector y (multiples of 0.2).
- It is a proper scoring rule. The optimal prediction for a clip is E[y | audio], the posterior mean of the vote distribution. Do not output argmax, do not round to multiples of 0.2, do not sharpen. Hedged mass on plausible neighbours (for example fear/surprise) is rewarded.
- Calibration-sensitive. A confident wrong label is penalised quadratically.
- Pooled, not hierarchical: speakers with more clips weigh more. Report both pooled OOF score and per-speaker-mean score as a diagnostic.
- Gating: the clip at 0 means a model worse than uniform scores 0, so there is no negative-score information. Compare designs on the unclipped value 1 - L_model/L_uniform.

**True independent unit.** The speaker (`actor_group`), not the clip. Clips from one speaker share voice, channel and performance style. Effective sample size is the number of train speakers (unknown; diagnostic D4), not 5,568. Each clip's label is also a noisy 5-sample draw from a latent perceived distribution q(x): y ~ Multinomial(5, q)/5. Irreducible per-clip squared error is (1 - ||q||^2)/5, which bounds the achievable score (diagnostic D3).

**Pipeline stages (single-stage regression, diagnose separately anyway).**
1. Representation: waveform to log-mel features to learned acoustic embedding (from random init).
2. Scoring: 7-way softmax head giving the posterior-mean estimate.
3. Decode: none beyond a low-dimensional calibration (temperature and bias, about 8 scalars) fitted on out-of-fold predictions, then simplex-safe JSON formatting.

Candidate coverage is not an issue (the output space is the 6-simplex, every valid answer is reachable). Diagnose (a) representation quality via a held-out-speaker probe floor, (b) ranking/argmax agreement as a secondary read-out only, (c) calibration gain from the final step, cross-fitted.

## Compliance regime

**Classification.** Audio, FROM-SCRATCH regime. CLAUDE.md sections 6.6 and 6.8 apply: no pretrained weights of any kind, only supplied public files, same time limits as any other challenge. Training is genuine (random-init convolutional/recurrent nets trained end to end), so the "model must do the learning" test is met.

**Explicit bans extracted from the description (treated as hard constraints).**
1. Train from random initialization using only the supplied public files.
2. No pretrained audio or speech models (wav2vec, HuBERT, AST, BEATs, Whisper, CLAP, etc.). Reading chosen: also no ImageNet-pretrained CNN on spectrograms (`pretrained=True`, `from_pretrained`, `torch.hub`, `torchaudio.pipelines`), because sentence 1 says "random initialization" without qualification. This is the reading that is compliant under every interpretation.
3. No external corpora (including noise, impulse-response or music corpora for augmentation), no external retrieval.
4. No hidden answers.
5. No speaker/source lookup. Consequence: never map `actor_group`, `clip_id` or `audio_path` strings to identities or to anything outside the data; never use path/file-name/id patterns as features.
6. No manual test labeling.
7. Runtime: whole solution within 90 minutes on one GPU.

**Conflicts between the description and CLAUDE.md / guidebook, and the resolution.**
- Pretrained weights: CLAUDE.md sections 1 and 2.2 allow downloading HF/timm backbones and section 6.6 suggests pretrained audio CNN/AST. The challenge bans them. The challenge wins (CLAUDE.md 2.4). In practice: no internet access at all during the run (nothing to download), no `from_pretrained`, no `timm.create_model(..., pretrained=True)`. Prefer a plain-torch hand-written ResNet/CRNN over timm to remove any doubt that weights could be loaded (timm with `pretrained=False` would be legal but adds nothing).
- Runtime: the description states 90 min; CLAUDE.md section 1 targets <= 50 min with a 1 h worst case and says to follow a shorter challenge limit only. The description's limit is longer, so the stricter CLAUDE.md/agent target applies: plan about 40 min on an A10G with >= 30% headroom (about 63 min budget at 70%). This also leaves more than 50% slack against the 90 min limit.
- Hardware: the description says "one GPU" without naming it. Assumption: A10G 24 GB per CLAUDE.md. Fixed `device="cuda"`, no `is_available` switches.
- CLAUDE.md allows frozen-embedding/GBDT hybrids in some domains; here a GBDT or ridge on hand-built acoustic functionals does not "learn acoustic representations from random initialization" and fails the spirit of the description. It may appear only as a dev-time floor/diagnostic, never as the shipped core (see Rejected options).
- Mixup/CutMix (listed as normal in CLAUDE.md 6.3) is not banned, but label-mixing distorts the very calibration the Brier metric rewards; left out of the primary on technical, not compliance, grounds.

**Whole-test-aggregation rules (CLAUDE.md 2.3 #5).** `actor_group` is present in test.csv, which invites per-speaker normalisation of test clips (speaker-level mean/variance normalisation of features, speaker-level label prior). That is cross-row pooling of test rows, plus effectively speaker lookup, so it is NOT used. `actor_group` is used only (a) to build grouped folds on train, and (b) optionally as a train-only auxiliary label (adversarial head, gated, see Structural signals). The script drops `actor_group` from the test frame immediately after reading it (assert the remaining columns are `clip_id, audio_path`).

**Strip-the-ML test.** Remove the trained networks and nothing remains but a log-mel front end (deterministic DSP, no learned content) and a prior constant. Passes.

**Grey items and how each is handled.** Log-mel front end (standard preprocessing, input to the network): keep. Hand-built F0/energy/voicing channels: classical DSP, not a model; potentially read as "hand-coding what the network should learn". The primary excludes them; they enter only as a gated upgrade after reviewer answer and a measured gain (cost of the conservative reading is measured by that ablation). Speaker-adversarial head using train `actor_group`: not lookup, but optional and gated.

## Data findings

Not verified: no data was read. For each item: the exact diagnostic to run on TRAIN, then the expected result as an UNVERIFIED hypothesis. Test files may contribute only row count (1,334), id format, WAV count, durations/total seconds for memory and runtime planning.

**D1. Integrity and audio profile.** Count rows and unique `clip_id`; assert every `audio_path` exists (6,902 WAVs total = 5,568 + 1,334); read every WAV (scipy.io.wavfile): sample rate, channels, dtype, length in samples; fraction of non-finite, all-zero or clipped (|x| >= 0.999) clips; peak and RMS (dBFS) distribution; leading/trailing silence fraction (frame RMS < peak - 40 dB); duration histogram per clip. Hypotheses (unverified): all 16 kHz mono, durations roughly 1 to 6 s with a modal band around 2 to 4 s, loudness varies by speaker/channel by tens of dB.

**D2. Target structure.** Parse JSON; assert exactly seven keys in vocabulary order, values multiples of 0.2 (|5y - round(5y)| < 1e-6), sum 1. Report: mean y per label (class prior; sums to 1), argmax frequencies (ties split), distribution of max vote count (5/5, 4/5, 3/5, 2/5 plurality), number of distinct 7-count patterns observed (at most 462), mean ||y||^2, mean entropy, 7x7 correlation of label columns across clips, share of clips with at least one zero-vote label. Hypotheses: neutrality and one or two high-arousal classes dominate; surprise is rare; unanimous clips are a minority; strong confusion pairs (fear/surprise, anger/disgust, neutrality/sadness).

**D3. Ceiling and floors (all from train labels).** Q = (5 * mean||y||^2 - 1) / 4 estimates E||q||^2 under the multinomial-5 model. Noise floor per clip = (1 - Q)/5. Uniform loss per clip = mean||y||^2 - 1/7. Ceiling score = 1 - floor / uniform_loss. Also: score of the constant train-mean predictor under grouped CV (prior floor). Illustration only (not a finding): if mean||y||^2 were 0.45, Q = 0.31, floor 0.138, uniform loss 0.307, ceiling about 0.55. The real ceiling comes from the real mean||y||^2.

**D4. Group structure.** Number of `actor_group` tokens in train; clips per group (min/median/max); label-mean per group; intraclass correlation (ICC) of each y column across groups; row-order check (is train sorted by group or label; run-lengths of `actor_group`, lag-1 autocorrelation of argmax label by row index). Test: only the number of distinct groups and clips per group, as a size statistic for planning (expect train/test group counts proportional to 5,568 / 1,334). Hypotheses: tens of train speakers (rough guess 60-90), tens to ~100 clips each; between-speaker ICC of label priors is small but non-zero; rows shuffled. Few groups means a noisy private score (SE of order 0.02, est.).

**D5. Duplicates and near-duplicates.** SHA-1 of PCM bytes for exact duplicates (also across groups); near-duplicates via cosine similarity of time-averaged log-mel descriptors plus duration equality; union-find to build content components. Hypothesis: no duplicates; if the corpus has a small fixed set of spoken sentences repeated across speakers and emotions, that is a nuisance factor (same text, different delivery), not a leak, because speakers are disjoint.

**D6. Speaker-grouping audit (train only).** Using long-term-average-spectrum descriptors, check whether groups are internally coherent (within-group nearest-neighbour rate) and whether any two tokens are mutually nearest (possible split speaker). Merge suspected fragments into derived groups by union-find and compare to provided tokens. Hypothesis: tokens are clean speakers; if a group is a bundle of several speakers, CV is still safe (coarser), only less balanced.

**D7. Shortcut audit.** Correlate mean agreement ||y||^2 and each label mass with duration, RMS level, silence fraction, peak, row index; inspect `clip_id` and `audio_path` character patterns for anything that looks like encoded identity or class. Anything found is logged and NOT used (ban on speaker/source lookup). Hypotheses: some correlation of anger/joy mass with loudness and of sadness with duration/low energy; these are legitimate acoustic cues the network may learn from the signal, but gain-invariant input handling must be tested (see Validation).

**D8. Learnability probes (train, grouped 5-fold, all unverified outcomes).** (a) constant prior score (expect small positive); (b) duration+loudness-only linear model (expect small positive; this is the shortcut yardstick); (c) ridge/softmax regression on mean/std functionals of log-mel (expect clearly above (b)); (d) a small CNN, 15 epochs, trained on 30% / 60% / 100% of train speakers to get the learning-curve slope per extra speaker (decides how much the 100% refit is worth); (e) train-vs-held-out loss per epoch to see memorisation onset.

**D9. Expected score band (est., not a promise).** From-scratch audio models on a few thousand clips with unseen speakers typically recover roughly 40-65% of the achievable (noise-floor-limited) explained variance. With the illustrative ceiling in D3 that is a private score of roughly 0.20 to 0.35, central about 0.27. This is a guess whose location depends entirely on the measured agreement level (D2/D3); recompute after D3.

## Validation design

**How the test split was made (from the description).** Speaker-disjoint partition: roughly 19% of clips (1,334 of 6,902) from speakers absent from train. Mirror it with grouped folds over `actor_group`.

**Folds.** Grouped 5-fold on train tokens (validation fold about 20% of speakers, matching the test proportion). Folds built by a seeded greedy balanced assignment of whole groups (balance clip counts, shuffle by seed), after merging any derived groups from D5/D6. No clip is ever in a training fold with a group-mate in the validation fold. Stratification on labels is not possible per group; instead check each fold's mean-y against the global prior and report the spread.

**Dev protocol.** 5 folds x 3 split seeds (different group assignments), paired comparisons on identical folds, fixed training constants (epochs, EMA decay, schedule) with no validation-triggered stopping. Report pooled OOF score (computed from pooled sums, never mean of per-fold ratios) and per-fold mean +/- std plus std across split seeds. Accept a change only if the paired gain exceeds the larger of 1 SE of fold differences and the seed-to-seed std, and is positive in most folds. The final shipped script runs one 5-fold CV pass (split seed fixed) for monitoring and calibration only.

**Metric implementation and unit tests.** Re-implement exactly: pooled L_model and L_uniform, clip at the end, plus the unclipped value for development. Tests: perfect predictions give 1.0; uniform gives 0.0; predicting the one-hot of each clip's least-voted label gives clipped 0 (unclipped negative); constant train-mean gives a small positive number equal to the D8(a) value; pooled score differs from mean-of-per-clip ratios on a crafted example (guards against the wrong aggregation); hand-computed value on a 3-clip example. Also unit-test the JSON writer: re-read the written CSV with `keep_default_na=False`, parse each cell, assert 7 keys in vocabulary order, finite, in [0,1], |sum-1| <= 1e-6, ids equal to the sample file.

**Nested/cross-fitted post-hoc steps.** Calibration scalars (temperature, per-class bias or a single prior-shrink weight), the blend weight between model families (equal by default, fit only if one scalar helps), and any numeric knob that depends on OOF: each is evaluated cross-fitted (fit on four folds' OOF, score on the fifth). The selection of architecture variants uses the dev protocol above; the reported final CV is optimistic by the amount of selection, expected small (<= 0.005 est.) because few discrete choices are made.

**Proxy biases (direction and reason).**
- Grouped 5-fold trains on 80% of train speakers; the shipped refit uses 100%. Pessimistic by the learning-curve slope, measured by D8(d); expect +0.005 to +0.02 on the real test (est.).
- Selection on OOF (variants, epoch constant, calibration): mildly optimistic, small.
- Train-speaker pool vs test-speaker pool differences (sex/age/channel mix, never inspected from test): unknown sign, large variance; with few speakers the private score SE is large, so a CV difference smaller than the speaker-level noise is not evidence.
- OOF predictions come from a single model per family while test predictions are an average of several; the ensemble is less noisy, so test Brier is likely slightly better than OOF for the same models (optimistic direction for the real score), and a calibration fitted on single-model OOF may over-sharpen the averaged ensemble. Mitigation: calibrate only with a temperature plus a shrink weight (2 or 3 scalars) and verify transfer in dev using members counts 1 vs 3 (repeat-split OOF averages).

**Per-slice reporting.** Per-fold, per-speaker-mean score, per-label squared error contribution, and score by agreement tercile (unanimous vs ambiguous clips), to see where the loss sits.

## Overfit/underfit risks

**Overfit.**
1. Few independent units (speakers, probably tens): the net can memorise speaker identity, channel and the recurring sentences. Mitigation: grouped CV only; small capacity (about 1.5-3M parameters); SpecAugment (time and frequency masks), random gain, time shift/crop, small speed perturbation, additive Gaussian noise; dropout before the head; AdamW with weight decay; EMA of weights; fixed short schedule; log train loss vs held-out score each epoch and fix the epoch count from the point where held-out plateaus (fixed constant, no validation-triggered stopping); optional speaker-adversarial head (gradient reversal over train `actor_group`), gated.
2. Label noise: each y is a 5-sample draw. Training to the empirical vector with MSE or soft cross-entropy is consistent for q, but a high-capacity net fits the noise. Mitigation: low capacity, EMA, snapshot averaging, seed ensembling; no label smoothing (it would bias a Brier-scored output).
3. Selection noise: many architecture ablations on a small number of speakers select noise. Mitigation: paired comparisons over 3 split seeds, only accept gains above noise, cap the number of variants, equal-weight blend instead of tuned weights.
4. Calibration overfit: only 2-3 scalars, cross-fitted.
5. Hard-coded constants: standard values (lr, wd, batch, epochs), each logged; no pasted offline "best params". Any numeric constant that dev CV influenced is flagged as a reviewer question (see Open questions).

**Underfit.**
1. Too coarse a representation: only 40-64 mels, or 10 ms hop with no harmonic resolution, loses pitch and voice-quality cues. Mitigation: 128 mel bins, 25 ms window with n_fft 1024, 10 ms hop; test 64/96/128.
2. Truncated context: cropping to 2 s removes the prosodic arc. Mitigation: use full clips (pad to a fixed cap taken as a documented constant from the train duration distribution, with masks); random crops only as light augmentation with a minimum length.
3. Over-normalisation: per-clip mean/variance normalisation erases absolute loudness, an emotion cue; global normalisation keeps channel/gain confounds. Mitigation: test global (train-fit constants) vs per-clip vs both-channels in dev.
4. Loss mismatch: plain classification loss on argmax discards disagreement. Mitigation: train on the full vector with Brier loss (matches metric), compare with soft cross-entropy and a mixed loss.
5. Under-training from scratch: few epochs. Mitigation: 40 epochs with cosine schedule and warm-up (estimate, to be confirmed against the D8(e) train/held-out curves).

## Recommended approach (primary + fallback)

**Capacity ladder (from-scratch version; stop at the lowest rung that wins under grouped CV).** (0) constant train prior; (1) ridge/softmax regression on mean/std/percentile functionals of log-mel, used only as a floor and diagnostic; (2) small ResNet on log-mel with statistics pooling; (3) add a second family with different assumptions; (4) larger/deeper variants only if (2)/(3) show clear underfitting. Frozen-feature/LP-FT rungs from the pretrained ladder do not exist here.

**Primary: a two-family seed ensemble of small from-scratch networks, probability-averaged.**
- Front end (deterministic, in-script, GPU, per clip independent): int16 WAV to float; log-mel via `torch.stft` (n_fft 1024, win 400, hop 160) and a mel filterbank built in plain torch (128 bins, 20-8000 Hz), log with floor; normalisation constants fit on TRAIN clips only (global mean/std per mel band), variants per dev. All features precomputed once and held on GPU in fp16 (about 0.5-1.1 GB, est.); augmentation runs on GPU with seeded generators, no DataLoader workers.
- Family A (spectro-temporal CNN): 4-stage ResNet (channels 32/64/128/256, 2 blocks per stage, about 2M parameters, own implementation), frequency-axis pooling to a sequence, masked mean+std (statistics) pooling over time, dropout 0.3, linear to 7 logits, softmax. Output bias initialised at log of the train mean distribution (train-only statistic).
- Family B (different assumption, temporal modelling): conv front end (2 conv layers, frequency down-sampled 4x) feeding a 2-layer bidirectional GRU (hidden 128) with masked attention pooling, linear head. Assumption diversity: A pools local patterns, B models the utterance-long prosodic contour.
- Loss: Brier (MSE between softmax output and y), plus a small auxiliary loss weight on argmax cross-entropy and on predicting the clip's agreement ||y||^2, each justified only if it wins in dev (default: Brier alone plus the two aux heads at weight 0.1, to be confirmed by paired CV).
- Training: AdamW, peak lr about 2e-3 with warm-up and cosine decay, wd 0.05 (decoupled), batch 64, 40 epochs (confirmed from D8(e)), grad clip 1.0, bf16 autocast for training, EMA of weights used for evaluation, fixed seeds, deterministic kernels. Inference in fp32, eval-mode BatchNorm, masked pooling: a clip's output depends on that clip only (unit-test: same clip alone vs inside a batch, tolerance 1e-4).
- Ship: for each family, refit on 100% of train with fixed epochs and several seeds (A: 3, B: 2); average probabilities within the ensemble with equal weight across families; then apply the calibration (temperature + one shrink weight toward the train prior) fitted on the cross-fit OOF of the same recipe; renormalise; write.
- Probability averaging (not z-score blending) because all members emit softmax outputs on the same scale and the metric is Brier on the simplex; averaging probabilities keeps outputs on the simplex and cannot increase squared error versus the mean member (convexity).

**Fallback.** Family A alone (3 seeds, 100% refit), with the 2-scalar calibration and the same validator. It is the leanest design that encodes the structural insights (Brier loss, duration-aware masked pooling, speaker-disjoint validation) and is shipped if family B does not beat noise in paired dev comparison or its runtime is too high.

**Gated upgrades (only after the lean design is measured, each needs a paired gain over noise).** (i) F0/energy/voicing contour channels computed per clip by classical DSP and concatenated as extra input channels (depends on reviewer answer); (ii) speaker-adversarial auxiliary head; (iii) an in-script second CV round (2 split seeds) to fit calibration on a 2-member OOF if the member-count transfer check shows a gap; (iv) train-only masked-spectrogram pretraining on train clips (never test audio), only if a learning-curve probe shows clear data starvation.

**Why this fits this data.** The target is a soft vector, so a Brier-trained softmax regression directly estimates the metric-optimal posterior mean. Few speakers and 5.5k short clips argue for small convolutional/recurrent nets with heavy spectrogram augmentation rather than transformers from scratch. Speaker-disjoint evaluation argues for gain-robust input handling and speaker-invariance pressure rather than speaker-specific tricks.

## Rejected options

1. **Any pretrained audio/speech model** (wav2vec 2.0, HuBERT, WavLM, AST, BEATs, Whisper encoders, CLAP): explicitly banned.
2. **ImageNet-pretrained CNN/ViT on spectrograms**: not named but conflicts with "random initialization"; rejected under the conservative reading.
3. **Per-speaker normalisation / speaker-level priors using `actor_group` on test clips**: whole-test aggregation and speaker lookup. Estimated cost of forgoing it: this is a commonly strong trick in speech-emotion work (est. +0.02 to +0.05, unverified). Listed as a reviewer question; not planned.
4. **Pseudo-labelling, test-time adaptation, self-supervised pretraining or BN statistics on test audio**: banned (test used beyond one-sample inference).
5. **External noise/speech/impulse-response corpora for augmentation; any extra data**: banned. Synthetic noise generated in-script from Gaussian draws is plain augmentation and kept.
6. **GBDT or ridge on hand-built functionals as the shipped solution**: does not learn acoustic representations; fails the strip-the-ML spirit. Allowed only as a dev floor.
7. **Majority-label classifier and decode by argmax**: collapses the disagreement the metric rewards.
8. **Rounding predictions to multiples of 0.2 or sharpening**: worse under squared error; the optimal output is the posterior mean.
9. **Large from-scratch transformers/Conformers, deep ensembles of many architectures**: too little data (tens of speakers); capacity ladder says stop at the lowest rung that wins.
10. **Learned-filterbank raw-waveform front end as a primary arm**: harder to train on 5.5k clips from random init; kept as a possible diversity arm only if time remains.
11. **Mixup/CutMix with label mixing**: allowed by CLAUDE.md but distorts calibration for a Brier-scored output; not in the primary.
12. **Early constant-prior fallback CSV**: the challenge does not ban it, but a silently degraded output is worse than a loud failure; the script validates and crashes rather than ships a degraded path.
13. **Wall-clock-based stopping/branching, `cuda.is_available` switches, `cpu_count` workers**: banned by CLAUDE.md section 3.

## Fixed work plan & runtime budget

All counts are constants at the top of `solution.py`; time is logged only, never used in a condition. Runtime estimates are (est.), A10G, to be profiled (time one epoch of each family, multiply, then hard-code counts per CLAUDE.md 3.7).

| Stage | Fixed plan | Est. time |
|---|---|---|
| Load train/test CSV, vocabulary, validate schema, read 6,902 WAVs, log-mel to GPU fp16 | one pass | 1.5 min |
| Grouped 5-fold CV, family A | 5 folds x 40 epochs, ~3.5 s/epoch | ~12 min |
| Grouped 5-fold CV, family B | 5 folds x 40 epochs, ~3 s/epoch | ~10 min |
| Calibration fit (2-3 scalars, cross-fit report) | closed-form/small LBFGS | < 1 min |
| Full-data refit: A x 3 seeds, B x 2 seeds | 5 runs x 40 epochs | ~14 min |
| Test inference (fp32, per-clip masked), blend, calibrate, write, validate, re-read | one pass | ~1 min |
| Total | | about 40 min |

Headroom: about 36% against a 63 min internal cap (70% of 90) and over 55% against the stated 90 min; it also fits the CLAUDE.md target of <= 50 min. If profiling shows more than 45 min, drop family B CV to 3 folds-equivalent by reducing epochs (a fixed constant change before submitting), not by any runtime branching.

Memory (est.): features 0.5-1.1 GB fp16; activations at batch 64 with 128 mels x up to ~600 frames and 32-channel first stage, a few GB; peak under 8 GB with headroom on 24 GB. Fixed batch sizes with headroom, no OOM-dependent fallbacks.

Determinism: seeds for random/numpy/torch/cuda, `PYTHONHASHSEED`, `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set before importing torch, augmentation via seeded torch generators on GPU, `num_workers` not used. Run the final script twice and diff the submissions; GPU kernels may still create tiny differences, so calibration and blending are low-dimensional and robust to them.

Input/output validation inside the script: assert required files, sample rate 16 kHz and mono for every WAV, finite samples, label schema in train, 1,334 test ids equal to the sample file ids, `actor_group` removed from test, outputs finite, simplex-valid, then write via the formatting rule below and re-read the CSV with `keep_default_na=False`, parsing JSON and re-checking everything. Float formatting rule: round each value to 9 decimals, add the residual to the largest entry, verify |sum-1| <= 1e-7 after re-parsing (rounding to 6 decimals would risk a 3.5e-6 sum error and an invalid file).

Source hygiene: plain readable UTF-8, well below 512 KB, no encoded blobs, no `exec`/`eval`, comments stating each compliance point (train-only fitting, random init, fixed plan). A string scan before submission must find none of: `pretrained=True`, `from_pretrained`, `torch.hub`, `torchaudio.pipelines`, `transformers`, `huggingface`, `timm.create_model` with weights, `time.time` inside a condition.

## Metric-aware training & decode

(i) Back-solve: the metric is plain squared error on the probability vector; the only transform is the pooled normalisation. No invertible transform to exploit; the target is the 5-vote frequency vector, which is a noisy sample of q(x).
(ii) Weights and averaging: no per-example weights in the numerator; pooled over clips. Training loss is the unweighted mean over clips of ||p - y||^2, which is exactly the metric numerator; a clip with a larger denominator share does not get a different weight. Report pooled and speaker-mean scores.
(iii) Ordinal heads: not applicable (nominal labels).
(iv) Expected-utility decode: for a squared-error score the Bayes-optimal output is the posterior mean. The softmax regression trained with Brier estimates it directly; no argmax, no thresholds, no rounding. Competing losses compared under paired CV: Brier alone, soft cross-entropy, Brier + small CE; choose the lowest cross-fitted Brier. Cross-entropy is also consistent but weights tail errors differently; the empirical comparison decides.
(v) Closed-form thresholds: none needed.
(vi) Span/ordering/proportion structure: the label is an aggregate of five votes (a proportion). Pool-then-loss is already satisfied because the net outputs a distribution compared to the pooled vote vector.
(vii) Hard constraints: simplex (softmax), values in [0,1], sum 1. At the file stage: renormalise, round to 9 decimals, push the residual to the largest entry.
(viii) Overdispersion: not needed; the score is a point-estimate squared error, not a count likelihood.
(ix) Taxonomy: none.

Calibration: initialise the output bias at the log train mean distribution, train with a proper scoring rule, then fit a temperature tau and one prior-shrink weight w on OOF predictions by minimising Brier: p_cal = (1-w) * softmax(logit/tau) + w * prior. At most 3 scalars (compare against a per-label bias variant with 7+1 scalars; keep the smaller if the cross-fitted gain is within noise). Report the cross-fitted Brier gain and the member-count transfer check. Auxiliary losses that follow from the structure: argmax-class CE and agreement ||y||^2 regression; each kept only with a measured paired gain.

## Structural signals

1. **Simplex plus 5-vote quantisation.** Counts are multiples of 0.2 and y ~ Multinomial(5, q)/5. Gives a noise floor (D3) used as the honest ceiling, and the instruction to predict E[y|x], not a quantised mode. Verify on train: values exactly multiples of 0.2 and each sums to 1.
2. **Speaker disjointness.** Train/test speakers differ: grouped CV and speaker-invariance pressure (augmentation that perturbs channel/gain/formant-neutral aspects; optional adversarial speaker head over train tokens). Not used at test time.
3. **Shared text across emotions (hypothesis, verify with D5).** If the same sentences are spoken in different emotions, the lexical content is a nuisance; time masking and statistics pooling push the model toward prosody and voice quality rather than word identity.
4. **Gain/channel invariance.** Random gain, mild equalisation-like frequency tilt (a few dB), noise; check with a dev diagnostic that held-out predictions barely change under +/- 6 dB gain applied to validation clips. Symmetries that are NOT valid and must not be used: time reversal, large pitch shifts (both change perceived emotion); small speed changes (+/- 5%) are assumed safe and are checked by paired dev CV rather than assumed.
5. **Label geometry.** The seven classes are confusable in structured ways (D2 correlation matrix). A shared softmax lets the net learn them; no hand-written confusion rules. A low-rank or label-correlation prior is optional and only through learned parameters.
6. **Coupling between outputs.** argmax and agreement are implied by y; use as auxiliary heads, and test both directions: does an agreement head improve the Brier of the main head (dev), and does predicting the argmax class from the soft output match a separately trained classifier (diagnostic only)?
7. **Duration/loudness shortcuts.** If D7 shows large label correlation with duration or level, keep them as legitimate acoustic cues only if they are derived from the clip itself and survive grouped CV (speaker-disjoint). They must not become speaker identity proxies; the group-level ICC check and the adversarial head diagnose this.
8. **Per-clip independence.** Inference depends only on a clip and the train-fit model: BN eval statistics from training, masked pooling, constants from train. This is both a compliance and a symmetry property (batch-composition invariance), unit-tested.

## Experiment roadmap

1. **Contract, metric, validation (stop when all unit tests pass and D1-D8 are logged).** Implement the pooled metric with the tests listed above, the JSON writer/validator, grouped fold builder (with derived groups), the diagnostics D1-D8, and the ceiling formula D3. Decide durations cap and training length from D1/D8(e).
2. **Cheap baseline end-to-end and valid.** Constant prior and ridge/softmax on functionals (dev floor only), then the same pipeline in a minimal Family-A form (10-15 epochs, 1 seed) producing a valid submission. Stop when the CSV validates and CV is above the prior floor by more than noise. Credit 1 (baseline).
3. **Representation and structural insight.** Family A full recipe; paired ablations on mel resolution (64/96/128), input normalisation (global/per-clip/both), crop length vs full clip with masks, pooling (mean, mean+std, attention). Each over 3 split seeds. Stop climbing a rung when the gain is not above noise. Credit 2 (best single).
4. **Metric-aware loss and calibration.** Brier vs CE vs mixed; aux heads; temperature+shrink calibration cross-fitted; member-count transfer check. Stop when cross-fitted gains stop exceeding noise.
5. **Diversity.** Family B; blend only if each family alone is within about 0.01 of the other and the blend gain exceeds noise; equal weights. Credit 3 (ensemble).
6. **Gated upgrades, bounded.** Reviewer-dependent F0/prosody channels; speaker-adversarial head; stronger augmentation; second CV round for calibration if the transfer check demands it. Each is a single paired experiment with a fixed trial count; no open-ended search. Any in-script HPO must have fixed trials (Optuna with a seeded sampler, no timeout) and its own held-out check; default is none, to keep runtime and selection noise down.
7. **Final fixed-plan run.** Freeze constants, run from a clean `working/` with the exact platform command, run twice and diff, check runtime headroom >= 30%, validator output and output-distribution sanity (mean per label close to the OOF mean and train prior, no constant outputs). Credit 4 (final).

Stop criteria are expressed as: gain over noise (paired, 3 split seeds) and runtime within budget. Log every experiment (id, change, pooled OOF mean +/- std across seeds, per-fold, runtime).

## Compliance audit

CLAUDE.md section 7 checklist against this plan:
- Test file read only to produce per-clip predictions: yes; per-clip feature extraction, train-fit constants, no test statistics, no vocabulary or scaler fit on test, no dedup, no rank normalisation across the test set. Pass.
- Time in conditions: none; fixed work plan. Pass.
- `cuda.is_available`, `cpu_count`, import fallbacks: none; fixed `device="cuda"`. Pass.
- Hard-coded constants tuned offline: architecture/optimiser hyperparameters are standard fixed values; the epoch count and aux-loss weights come from train-only grouped CV during design and are documented. Risk flagged below as a reviewer question.
- External data, synthetic training data, self-hosted weights, non-allowed libraries: none. Only torch, torchaudio-free or plain torch DSP, numpy, pandas, scipy.io.wavfile (core Kaggle stack; avoid librosa/soundfile, which are not in the listed stack). Gaussian noise generated in-script is augmentation of real clips. Pass.
- Strip-the-ML test: only log-mel DSP remains; fails to solve the task. Pass.
- Inference-only/frozen-feature solution: no, all weights trained from random init. Pass.
- Source readable and under 512 KB, comments present: planned.
- Challenge restrictions honoured: random init, no pretrained models, no external data or retrieval, no speaker/source lookup, no manual labeling, runtime within the 90 min limit (planned about 40 min).

Plan-specific self-audits:
- Whole-test aggregation: none (see compliance regime). Per-speaker normalisation is excluded.
- Sibling leakage: grouped folds by speaker plus content components; clips of the same speaker never straddle train/validation. Rows from the same speaker are not cross-referenced in features.
- Label-derived statistics: only the train mean distribution as bias/prior, computed on training folds inside CV (and on 100% for the refit); trivial dimensionality, no per-group stats.
- Constants derivable from in-script train-only search: the calibration scalars are fit in-script on OOF; epochs, lr, wd, aux weights are fixed design constants chosen by train-only dev CV (flagged, not derived from public-LB feedback).
- Output formatting: simplex-safe JSON writer with residual fix and re-read validation.
- Unverified: all dataset-dependent claims above.

## Open questions & assumptions

Reviewer questions, each with the plan under each reading:
1. **Per-speaker normalisation using `actor_group` on test.** Reading A (not allowed; plan default): no use of test group tokens at all. Reading B (allowed because the column is provided in both tables): add speaker-level mean/variance normalisation of log-mel computed from each speaker's own clips, applied identically in train and test; est. +0.02 to +0.05 (unverified), grouped CV unaffected. Compliant under every reading is A; B is only pursued after an explicit yes.
2. **Hand-built prosody channels (F0/energy/voicing from classical DSP) as extra network inputs.** Reading A (not allowed): log-mel only (plan default). Reading B (allowed as hand-engineered features fed to a trained model, per CLAUDE.md 2.2): add the channels; the measured cost of A is the paired CV delta in roadmap step 6.
3. **Speaker-adversarial auxiliary head using train `actor_group` as a training label.** Probably allowed (not lookup, train-only); optional.
4. **Do dev-time ablations on train CV count as "tuning offline"?** The epoch count, aux weights and normalisation choice come from train-only grouped CV experiments; the final script reproduces CV and calibration in-script. If a reviewer requires in-script HPO for every constant, the fallback is standard values with no tuned numerics.
5. **Libraries.** Plain-torch DSP plus `scipy.io.wavfile` assumed allowed; `torchaudio` is on the allowed list but its pretrained pipelines are avoided; `librosa`/`soundfile` avoided because they are not named in the allowed list.
6. **GPU type.** Assumed A10G 24 GB; the description only says "one GPU". If it is smaller, the fixed batch size and epoch counts would be reduced before submission, never at runtime.

Assumptions (unverified): audio is 16 kHz mono as stated; durations are short (a few seconds); there are tens of train speakers; train rows are not ordered by label or speaker; `actor_group` tokens are clean speakers; the platform's GPU is comparable to an A10G.

What could not be verified: every dataset-dependent claim (distributions, speaker counts, durations, agreement level, ceiling, shortcuts, runtime per epoch, learning-curve slope), because no data or GPU was available. The expected private band (about 0.20-0.35, central about 0.27) is an estimate, not a promise; recompute it from D3 once the real agreement level is known.
