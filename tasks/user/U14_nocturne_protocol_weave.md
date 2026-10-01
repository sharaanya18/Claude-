# U14 — Nocturne Protocol Weave: latent pairing decisions under unseen cameras (user-supplied, verbatim)

## Overview
Every row is a complete nocturnal camera-triggered event represented by three 128x128 grayscale frames (first, middle, last event frames, not RGB) and one protocol code 0..6. The required prediction is one of four protocol-specific slots. A slot does not have a permanent visual meaning: the latent identities assigned to it change with the protocol. The seven protocols partition eight unobserved visual identities into four pairs, and every unordered pair of latent identities occurs together in exactly one protocol (all 28 pairs occur exactly once across the seven protocols; each protocol has four two-identity slots). Slot order is also permuted, so no latent identity has a protocol-invariant output; the four slot labels are positionally balanced and carry no stable semantic meaning across protocols. Training exposes only the protocol and the correct slot, never the eight latent identity labels or the pairing table. A solver must infer a shared visual state from overlapping supervision and apply the row's protocol. Evaluation cameras are absent from training. No camera identifiers, timestamps, filenames, locations or semantic identity names are released.

## Objective
For every test event predict the correct slot, an integer 0-3. Two visually similar events can have different targets under different protocols, while two different latent identities deliberately share a slot in one protocol and separate again in the others.

## Why difficult
A single frame can be empty, blurred, or partly obstructed even when the event contains an animal, so the three time-ordered frames should be interpreted jointly. All panels are nocturnal, camera overlays and metadata removed, held-out camera sites prevent direct background memorization. The supervision is indirect: a training target says only which slot is correct under that row's protocol, not which of the paired latent identities is present. Evidence linking identities is distributed across other protocols and other cameras. A model that ignores images, ignores protocols, memorizes row order, or learns seven unrelated constant rules remains near the chance-corrected floor.

## Dataset
train.csv (5,834 events: id, protocol, target), train_folds.csv (id, fold: five predefined camera-disjoint folds), test.csv (2,434 events: id, protocol), train_images.npz, test_images.npz (images array (N,3,128,128) uint8; ids array aligned to the CSV order), sample_submission.csv (id, prediction; target-independent constant template). Protocols and slot targets are balanced in training. IDs are arbitrary. Test events come from 131 camera sites that do not occur among the 298 training cameras. Use complete supplied folds for validation; a random row split leaks camera-specific appearance.

## Modelling guidance (from the description)
Practical approach: compact pretrained image encoder on each frame, aggregate temporal evidence, condition a four-way decision head on the protocol. Alternatives: seven coupled heads with a shared bottleneck, a latent eight-state model whose protocol maps are learned jointly, or a consistency loss that exploits the complete pair-cover structure. Compute tier: one NVIDIA A10G; long pretraining not required. Generic vision weights permitted only when otherwise allowed by the platform. External training data, reverse lookup of released samples, task-specific pretrained models not permitted.

## Evaluation
Balanced accuracy inside each protocol p: BA_p = mean recall over the four slots; c_p = clip((BA_p - 0.25)/0.75, 0, 1). g = (exp(mean(log(0.05 + 0.95*c_p))) - 0.05)/0.95 (stabilized geometric mean across the seven protocol components). score = g^2. Higher is better, range 0..1. Oracle = 1.0. Constant or chance predictions score 0 after correction on the complete evaluation set. Geometric aggregation rewards broad protocol competence.

## Submission
submission.csv with exactly id,prediction; every test ID exactly once; prediction finite non-Boolean integer 0..3. Extra columns, missing/duplicate/unknown IDs, fractional values, NaN, inf, Boolean rejected.
Restrictions: use released challenge data and allowed model resources; no external matching, reverse search, inferring targets from row order or serialization details, or leaderboard probing; predictions from released event panels, protocols and training supervision.
