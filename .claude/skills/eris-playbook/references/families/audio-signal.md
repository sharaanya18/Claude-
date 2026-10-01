# Audio, speech, signals and sensor streams

## Recognise
Waveforms/spectrograms, speech with reviewer votes or phoneme-level targets, code-switched speech linking, seismic or
wavefront arrays, accelerometer/tactile/force streams, event timing in sensors.

## Default build
1. **Contract and decision unit**: the unit is usually the episode/recording/site, not a frame. Derive groups by exact
   waveform hashing + union-find; speaker-/actor-disjoint folds when the test is speaker-disjoint.
2. **Representation**: frozen speech/audio encoders (wav2vec2/WavLM/HuBERT/AST) as feature or alignment sources, or
   a from-scratch CNN on mel features when weights are banned. Use a frozen CTC recogniser as a *structural aligner*
   (forced alignment by Viterbi, CTC blank rules) to anchor where evidence is pooled (E4); causal alignment of clean
   reference vs noisy stream before comparing local evidence.
3. **Evidence → features**: window summary statistics (mean, max, margin vs best competitor, entropy, neighbour mass)
   (C4); distance-biased attention around the expected position (C6); input-source dropout (G1).
4. **Heads**: structurally different heads/encoders in one ensemble (posterior-only, local-MLP, transformer-context,
   alternate encoder, alternate alignment) — the 9-head winner; or a focused 2-encoder end-to-end fine-tune with a learned
   alignment prior. IRT-style reviewer-link latent models as a zero-init neural residual baseline.
5. **Metric**: joint mask/vote structures → enumerate valid joint outputs and decode by expected utility; Brier →
   squared-error loss + per-slot recalibration; per-slot, per-reviewer-count slices.
6. **Signal arrays** (wavefront/seismic): per-frequency spatial covariance, any-station-as-target augmentation,
   `n−2`-style verified constraints for exact subset MAP search, nearest-observed-site substitution for unseen
   coordinates, assign-then-refine pipelines.
7. **Sensor/tactile/force**: model each stream with its own encoder into a shared latent, deterministic synchrony checks
   from the description, group by environment/object hierarchy (every view of an object and every object in an
   environment stay together).

## Speech linking / retrieval across languages
Gallery-batched listwise training for dual encoders (each query scored against its own gallery only); fine-tune with
hard negatives; no cross-query reasoning if the description bans it; code-switch robust tokenisation or audio-text
encoders; calibrate on held-out speakers.

## Pitfalls
Sliding windows cut from continuous recordings (chain them before folding, B13); speaker/session leakage; TF32 or fused
attention nondeterminism; per-test-batch normalisation; ignoring that metric slices (reviewer counts) differ; fixed
sample-rate/length assumptions; using the clock for decoding budgets.
