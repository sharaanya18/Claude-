# Vision tasks (classification, multi-label, detection, segmentation, geometry, change/timing, multi-view)

## Recognise
Images/frames/boards as input (photos, microscopy, satellite/rover, game frames, rendered panels); labels per image,
per box, per pixel/polygon, or per set of views; small datasets with strong site/object grouping.

## Hard rules
Never feed raw pixels to a tabular model. Deep model or classical algorithm decoding a deep model's channels. TTA is
allowed by default but check bans (some descriptions forbid it or any cross-row use); when TTA applies un-transform
directional output channels. Banned task-specific checkpoints (species/medical/detector models) when stated.

## Default build by sub-type
- **Classification / multi-label (few hundred–few thousand photos, grouped by site)**: the winner on a flood-photo task
  was the *simplest*: one large frozen backbone (DINOv2-L at 448, CLS ‖ mean-patch, flip-averaged) + per-label
  logistic regression with the regularisation C chosen per label on grouped folds, refit on 100%; fine-tuned multi-member
  ensembles ranked below. **Acceptance caution**: the guidebook treats a linear/tabular head on frozen embeddings as grey area (reviewer may reject), so ship LP-FT (real parameter updates, asserted) or full fine-tuning as the primary and keep the pure probe as yardstick/minor member. Mirror: frozen probe yardstick → LP-FT of the last stage with cached lower activations (E12) →
  full fine-tune only if the probe is weak. Site-weighted loss; balanced folds on rare positives (V6); per-label model
  selection nested. Native aspect/resolution for photographs; EXIF transpose.
- **Segmentation / instances**: U-Net with GroupNorm for small batches; auxiliary geometric channels + watershed for
  touching objects (J1); expected-IoU mask per instance (A16); cross-scale training by transporting crops to the target
  size and re-rasterising vector labels (J6); simulate target blur/JPEG (J7); sample the training distribution toward the
  evaluation sensor (G10); copy-paste of real labelled components (J5); EMA; transform-aware TTA with channel un-transform.
- **Detection + selective human review / budgeted regions**: detector (fixed NMS, class-specific thresholds from OOF,
  per-warehouse macro weighting), then a region-selection step that maximises the *expected metric gain* of the review
  window under the linearised formula; calibrate scores; control spurious classes (a false class in a warehouse that has
  none costs a full class-F1).
- **Geometry / metric reconstruction** (scale, sun direction, footprints from two views): physics-informed heads,
  co-rotated augmentation (rotate/flip images *and* angle labels), expected-utility decode over angular errors, no TTA if
  banned; two-view fusion with a shared encoder and view-aware heads; polygonisation by watershed.
- **Same-scene / same-moment / burst ordering**: patch-set correspondence and distinctive-evidence statistics from frozen
  ViT tokens (E11, C19) + exact latent-order marginalisation over ≤ 8 items (A18); extremity and midpoint structure as losses
  (G13).
- **Frame-event timing**: lag-aligned frame pairs (the action at t−1 causes the change at t), translation-equivariant
  head, where×when factorisation (C17), recover the discrete grid (C18), transform actions with the geometry (G11).
- **Rendered boards / panels (queries + candidates)**: crop panels by layout, shared encoder with swap-symmetric pair
  scores, fine-tune with hard negatives mined from train characters, hold out whole character classes and artifacts,
  exact assignment decode for distinct selections; ignore JPEG/size/position priors.
- **Detection of diagram structure** (flowcharts, BPMN): detector/keypoint model for nodes and links + a learned link
  reader, then a deterministic executor *as decode* (token-game semantics) — the model must read the diagram, the
  executor only orders what the model found.

## Capacity, data, augmentation
Count distinct sites/objects, not images. Strong augmentation is load-bearing on tens of samples; physical
augmentations only (check handedness, directionality, text orientation). Mixed precision + channels_last; fixed workers.
Hand-crafted image features may enter the deep model alongside the image.

## Validation
Group by site/object/camera/scene/crop-parent/near-duplicate image; balanced rare labels; target-regime transport of
held-out images before tuning decode; report per-site/per-domain slices (dark vs normal, scale buckets).

## Pitfalls
Interpolation validated while extrapolation (new camera/scale) is tested; downsizing small details; BN with tiny batches;
train/test crop-size mismatch; letterbox padding as a shortcut; label rasterisation after geometric transforms done at the
wrong grid; ranking solutions by CV on random splits when sites repeat.
