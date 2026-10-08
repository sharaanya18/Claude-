> NOTE: not verbatim platform text. Reconstructed from five ranked top solutions' own docstrings
> and code (primarily rank-1's detailed header docstring and rank-2's constants). Unconfirmed
> against a real sample_submission.csv.

# KineProof: Echoes of an Invisible Floor (reconstructed)

Input: motion-capture marker trajectories per trial — fixed-length sequences of `(256, 22, 3)`
(256 frames x 22 markers x 3D position), read from `train.csv` / `test.csv` plus per-trial marker
arrays named by a `motion_file` column, and (for training only) `train_targets.csv` giving the
target force sequence. `TRAIN_PEOPLE = 32` distinct people in the released training data (rank-2).

Task: predict the ground-reaction-force waveform (three axes: `force_x`, `force_y`, `force_z`,
each a length-256 sequence, per rank-2's `AXES`/`FRAMES` constants) produced by a person's body
during the trial, from the marker motion alone — a frame-wise sequence-to-sequence regression,
submitted per rank-1 as length-256 JSON arrays per axis per trial.

Implied metric (rank-1's docstring, used directly as the training loss too): per trial and per
axis, `sum(|pred - true|) / sum(|true|)`, capped at 1 with a shallow slope beyond the cap — i.e. a
normalised L1 / "relative force error", averaged over axes and trials.

Physical structure both top solutions model explicitly: force plates only cover part of the
walkway, so the true whole-body force is the sum of per-foot plate contributions while a foot is on
an instrumented plate, and a foot's share drops out while that foot is on the floor between plates
(rank-1's "plate head": output a correction to a Newtonian center-of-mass force estimate `W0`, a
between-feet difference `D`, and two gates `g_R, g_L`). Augmentation used by rank-1: per-marker
constant positional offset per batch (does not change velocity/acceleration or the force label,
since it's a rigid per-marker bias, not a motion change).

Compute: explicitly CPU-only in both inspected solutions (`THREADS = 10`/`2`, deterministic
torch, no CUDA calls at all), consistent with a stated CPU-only environment for this challenge.

Unverified: exact submission grammar/column names (JSON-array-per-axis is rank-1's own choice, not
confirmed from a real sample file), the official metric (very likely but not certain to equal the
training loss both solutions use), row counts, stated runtime limit, and whether the marker set,
frame count and `TRAIN_PEOPLE` figure exactly as coded or were rank-specific assumptions.
