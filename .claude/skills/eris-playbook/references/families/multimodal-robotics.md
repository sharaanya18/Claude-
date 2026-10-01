# Multimodal, video, robotics and cross-embodiment tasks

## Recognise
Several views/streams per case (external camera, wrist camera, force/tactile, audio, metadata), anonymous embodiment groups
with shuffled candidate positions, episodes with interaction phases, tensors that must satisfy multi-marginal constraints.

## Default build
1. **Per-stream encoders into a shared episode code**; canonicalise groups by documented kind columns (not by anonymous
   ids or positions); never treat row numbers, group aliases or candidate order as signal.
2. **Frozen strong visual backbone (DINOv2-B/L)** at an input size chosen from {112, 168, 224} by grouped CV, state channel +
   motion channel (frame difference) + group-centred copies; LP-FT of the last blocks as the compliance layer if fine-tuning is
   expected.
3. **Joint assignment head**: potentials per embodiment pair; enumerate assignments (first group fixed) and train by exact NLL
   plus a differentiable surrogate of the row metric; posterior-mean output; fitted sharpening and uniform-mix scalars (OOF,
   cross-fitted).
4. **Hierarchy-aware grouping**: every view of one object and every object in one environment stay together; derive groups in
   script from background clustering + union-find; random-row CV is forbidden (near-duplicate rows).
5. **Control-aware augmentation**: transform controls together with images; learn per-view embeddings for action codes.
6. **Metric slices**: bottom-quantile terms punish over-sharp wrong rows; guard with temperature/uniform mixing, report
   Bottom-k separately.

## Pitfalls
Collapsing ~100 independent environments into thousands of images (capacity overshoot); leaking via anonymised ids; hand
rules for opening/closing phases; evaluating on the same environments used to pick the sharpening constant.
