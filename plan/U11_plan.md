# U11 plan: execution order of activities in hand-drawn BPMN diagrams

Status of evidence: no dataset files were available to the strategist. Everything under "Data findings" is either (V) arithmetic from the challenge description or (U) an UNVERIFIED hypothesis plus the exact train-only diagnostic that would test it. Nothing here has been run. No solution code is included.

## Contract & decision unit

One valid answer: for each of the 156 test ids, one row `id,relations` where `relations` holds exactly n(n-1)/2 space-separated tokens `X-Y:rel`, X<Y alphabetically, rel in {before, after, either_order, exclusive}, covering every unordered pair of the keys in that row's `activities` JSON. The file must have exactly the columns `id,relations`, one row per test id, no duplicate or unknown ids. Invalid (whole submission rejected): wrong columns, row-count mismatch, missing/duplicate/unknown id. Low-scoring but valid: missing pair (counts as FN for the true class, never a FP, so omission never helps: always emit a token), wrong relation, unknown tokens (ignored).

Metric: all 3,608 test pairs pooled across drawings, macro-F1 over the four relation classes. Consequences:
- Pooling means a drawing with n activities weighs n(n-1)/2. Test mean is 7.0 activities (train 5.5), so large drawings dominate and per-drawing errors compound (V: train 6,290 pairs / 462 = 13.6 pairs per drawing; test 3,608 / 156 = 23.1 per drawing, about 1.7x).
- Macro over classes means exclusive (train 16.9%) and either_order (20.9%) count as much as after (33.1%) and before (29.0%). Class shares are V from the stated counts. Rare-class recall matters; argmax of a posterior is not F1-optimal.
- before/after are mirror images: relation(X,Y)=before iff relation(Y,X)=after. Equivalent view: two booleans per unordered pair, a = "X can be followed by Y", b = "Y can be followed by X"; (1,0) before, (0,1) after, (1,1) either_order, (0,0) exclusive.

The true independent unit is the drawing (and above it the modelling scenario: test scenarios never appear in train). All 462 train relations are exactly reproduced by running the stated token-game on the gold diagram (description claim, to be re-verified as oracle check #1), so the whole task is "read the diagram correctly". The answer is a deterministic function of: (i) the set of activity/event/gateway nodes with the few executor-relevant types, (ii) sequence and message flows between them, (iii) pool membership, (iv) the key-to-box assignment.

Pipeline stages, to be diagnosed separately:
1. Node coverage: detect activity boxes, events, gateways, pools (candidate pool recall).
2. Node typing: executor-relevant class only (see Structural signals).
3. Link reading: for candidate node pairs, is there a sequence or message flow u->v (ranking quality given nodes are right).
4. Key assignment: which detected activity box is which key/name (given names, handwritten text in the image).
5. Decode: structurally valid graph, then the execution rules, then macro-F1-aware choice of relation.
Four oracle variants (gold nodes/edges/keys swapped in one at a time) give the loss budget per stage.

Important asymmetry: arrows have NO stroke/polyline annotation. `sequence_flows[]` is only (source node id, target node id). Link reading is therefore a weakly supervised pair-classification problem (given image evidence between two boxes, is there a flow), not an arrow-tracing problem.

## Compliance regime

Domain: Computer Vision (structured decision: image -> graph -> relations), "Medium", A10G. Not labelled fine-tuning or from-scratch; the description allows ImageNet backbones, general OCR/handwriting recognition models and general vision-language backbones. Real training is required inside the script.

Explicit bans in the description, treated as hard constraints:
1. Training must happen in the submitted code. Inference-only is not allowed.
2. No weights fine-tuned on hand-drawn diagrams or flowcharts elsewhere. So no diagram/flowchart/sketch/BPMN/chart-structure checkpoints (e.g. flowchart or chart-parsing derivatives, sketch-to-graph models). Treat document-layout/chart/screenshot-trained models (Pix2Struct/DePlot/MatCha-family, layout detectors) as the same risk and avoid.
3. Only the provided data. No other copies of these drawings, no other diagram datasets, no external datasets. Consequence: no synthetic BPMN diagrams rendered from graphs or fonts, no collage-style generated drawings (also "synthetic data" under CLAUDE.md 2.3 #6); only augmentations of real images.
4. No test-set adaptation: each test drawing predicted independently; no pseudo-labels, no test-time training, no tuning on the test set as a whole. So no test-wide normalisation, vocabulary, threshold or prior estimation; decode constants come from a train-only holdout.
5. No lookups or manual work: no reverse image search, no source-archive matching, no hand labelling, no hard-coded ids/answers, no hosted AI APIs (no GPT/Claude/Gemini vision API, no HF Inference API).
6. Compute: A10G.

Explicitly allowed: implementing the execution rules (code turning a recognised diagram into relations). General OCR/handwriting recognisers are allowed. The description does NOT say the rule engine may carry the solution: the trained recognisers must be load-bearing, which they are (without them there is no diagram to execute).

Where the description is silent, assumptions (each also listed under open questions):
- Runtime limit: silent. Assume CLAUDE.md: target <= ~45 min on A10G, hard worst case 1 h, >= 30% headroom.
- Internet: silent. Assume CLAUDE.md: HF Hub / timm downloads only. This matters concretely: torchvision `weights=`/`weights_backbone=` download from download.pytorch.org (not HF/timm) so they are avoided; backbones come from `timm` (HF Hub). EasyOCR fetches weights from GitHub releases: avoid.
- Whether COCO-pretrained detectors (DETR/RetinaNet) count as "general-purpose": silent; the primary uses ImageNet backbones only (compliant under every reading).
- Whether TrOCR handwritten checkpoints count as a "general handwriting recognition model": the description explicitly allows general handwriting recognition models; assume yes (IAM line handwriting, not diagrams).
- Whether using test row's own `activities` (names, count n) as decode constraints is fine: assumed yes; it is that row's own input, no cross-row pooling.
- Whether rule-based decode constraints (degree repair, top-n activity boxes, pool-crossing means message) are acceptable: assumed yes as constraints on a trained scorer (strip-the-ML test below).
- Whether train_diagrams.jsonl geometry (boxes, words) can be used as supervision: yes, it is provided data.

## Data findings

Arithmetic from the description (V):
- Train: 462 drawings; 2,438 tasks of ~2,541 activities (5.5 mean x 462), so ~100 activities (~4%) are subProcess/callActivity; 796 exclusive + 576 parallel + 133 event-based gateways = 1,505 (3.26 per drawing); 6,207 sequence flows (13.4 per drawing), 794 message flows (1.7 per drawing); 17,415 word boxes (37.7 per drawing). Relation shares: after 33.1%, before 29.0%, either_order 20.9%, exclusive 16.9% of 6,290 pairs.
- Test: 156 drawings, 3,608 pairs; 7.0 mean activities vs 5.5 train: larger graphs. Expect about 1.3x nodes and edges per drawing and about 1.7x pairs. (V from description)
- Metric sanity numbers for unit tests (V by arithmetic on train counts): constant "before" gives macro-F1 about 0.113 (description says about 0.12 on test); constant either_order about 0.087; class-prior random about 0.25; perfect 1.0; before<->after swapped everywhere with the rest perfect: 0.5.

Train-only diagnostics to run first (all UNVERIFIED; hypothesis given with each):
1. Executor oracle: implement the rules, run on all 462 gold diagrams, require 462/462 exact reproduction of `relations`. Log states explored and runtime per diagram (max, p99), loop and 2-token-cap behaviour, collapsed-pool and terminate cases. Hypothesis: passes; worst-case state counts stay modest; a fixed deterministic state cap is rarely or never hit.
2. Node/edge inventory per drawing: counts by type, in/out degree by node type, fraction of tasks with in-degree 0, fraction of non-startEvent nodes triggered by rule 5, nodes receiving only a message flow. Hypothesis: tasks are in=1/out=1 in the large majority (~85-90%); gateways split or join (1-in-many-out or many-in-1-out); startEvents in=0; end events out=0; an in-degree-0 non-start activity is almost always the message-receiver case.
3. Executor-relevant class merge: counts of (activity | event non-terminate | terminate end | choice gateway = exclusive+eventBased | parallel gateway). Hypothesis: terminate end events are rare (a few %) but occur in pool drawings; blank vs X diamonds never matter.
4. Pools: count of drawings with pools, expanded vs collapsed pools, nodes per pool, whether every flow whose endpoints lie in different pools is a message flow and every same-pool flow is a sequence flow, whether message flows ever occur in drawings without pools, whether message flows ever target/originate at collapsed pools. Hypothesis: pool crossing implies message flow in ~all cases (the description says plain arrows between pools are recorded as message flows), so edge type can be derived from membership with a learned head only used to check agreement. Edges to/from collapsed pools do not affect relations (rule 6: they never block), so they need not be read.
5. Edge geometry: box-gap and center distance distribution of gold edges (normalised by image size and box size), fraction right-to-left, vertical, long-range or loop-back edges; recall of "target within the K nearest nodes of source" for K = 4, 6, 8, 10, 12. Hypothesis: >= 97% of edges within K = 8 by box gap; long loop-back edges are the tail.
6. Error-sensitivity curve (the key planning number): on the gold graphs, randomly delete 5/10/20% of edges, add 3/5/10% spurious edges, flip 1/2 gateway types, swap two keys, and report pooled macro-F1 against the gold relations through the executor. Hypothesis: macro-F1 drops steeply; ~10% edge errors plausibly gives macro-F1 in the 0.5-0.7 region and one swapped gateway type costs many pairs. This sets stop criteria for each stage and tells which stage deserves capacity.
7. Key vs order leakage tell: correlation of key letter with node x-position and with flow order per drawing; sign test of before vs after per drawing. The pooled counts (after 2,085 vs before 1,827, a gap of about 4 sigma if pairs were independent, pairs are clustered so less) suggest checking this. Hypothesis: keys are random and the gap is clustering noise. Either way the keys/key order are never used as a feature, and training uses random key relabelling (mirror swap before<->after).
8. Names and OCR difficulty: activity names per drawing, character length, words, language, fraction with misspellings, similarity of names within a drawing (edit distance; names are distinct ignoring case, but near-duplicates such as "Check order"/"Check orders" may exist), words per activity box and lines per box from `words[]` bboxes inside task boxes, null-key activity nodes. Hypothesis: names 2-5 words, English mostly with some other languages, 1-3 text lines per box; few null keys.
9. Orientation and size: `words[].angle` distribution per image, image_width/image_height versus actual JPEG size and EXIF orientation tag, bbox alignment with the EXIF-corrected vs raw image. Hypothesis: a minority of images are rotated (90/180) or have non-trivial EXIF; one consistent exif_transpose policy must be fixed so jsonl boxes align with pixels.
10. Resolution: node box and stroke size in pixels at original resolution (<= 2000 px) and after resizing to 1024/1280; fraction of events and gateway markers below 24 px after resize. Hypothesis: events and the X/+ markers fall under 20 px at 1024, so type classification needs native-resolution crops.
11. Nesting: nodes whose boxes contain other nodes (expanded subprocesses), overlapping or touching boxes, lane vs pool nesting. Hypothesis: rare; lanes nested in pools.
12. Scenario groups: derive scenario clusters by union-find on shared rare name tokens, near-identical activity-name sets, pool/lane names; report cluster count and sizes. Hypothesis: several drawings per scenario (students re-drawing the same scenario), tens of scenarios in total; grouped CV needs these groups because a random split leaks scenario text and layout.
13. Size strata: relation distribution, executor complexity and graph size versus n (n = 3..14), because test is shifted to n ~ 7. Hypothesis: exclusive and either_order shares rise with n; larger n has more gateways per activity.
14. Relation provenance: share of exclusive pairs from XOR branches vs from never-run activities (deadlocks, unreachable); share of either_order from parallel gateways vs separate pools vs loops. Hypothesis: parallel and pool-separated either_order dominate, loop-induced either_order is a tail.
15. OCR/assignment oracle (gold boxes): crop each gold activity box, run the zero-shot recogniser over four rotations, score all names, Hungarian assignment; report per-activity and per-drawing exact assignment, character error rate, effect of near-duplicate names. Hypothesis: CER high (hand-drawn, misspellings) but assignment accuracy still >= 90% per activity because candidates are only n <= 14 distinct names; near-duplicate names and short names are the failure mode.
16. Stage-oracle matrix (the decomposition): {gold nodes, detected nodes} x {gold edges, predicted edges} x {gold keys, OCR keys} to get macro-F1 for each cell. Hypothesis: key assignment and edge recall are the two largest losses; node detection is the smallest.
17. Duplicates: identical or near-identical images (hash) across train, and near-duplicate activity sets across drawings. Hypothesis: same scenario drawn by several students gives near-duplicate name sets but distinct images.
18. Test-side allowed stats (schema/size only): image count, size distribution, n per row, id format; used for runtime planning only.

## Validation design

How the test split was made (from the description): whole scenarios held out ("modelling scenarios that never appear in training"). Therefore:
- Dev CV: GroupKFold, 5 folds, groups = scenario clusters built from diagnostic 12 (union-find over shared rare name tokens, near-duplicate name sets, pool/lane names). Repeat with 2-3 seeds of group-to-fold assignment when group count is small. Report mean +/- std over folds and seeds. Accept a change only if paired same-fold gain exceeds the fold-to-fold noise.
- Report pooled macro-F1 (the exact official metric, re-implemented and unit tested on perfect, reversed (before<->after swapped), constant, class-prior random predictions; expected values listed in Data findings). Also report: stratum n >= 7 only (to mimic test size), per-class F1, per-drawing exact-relation rate, and the stage metrics: node AP/recall by executor-class, edge precision/recall/F1 by type, drawing-level exact graph rate, key-assignment accuracy, executor failure count.
- Stage decomposition via the oracle matrix (diagnostic 16), repeated on every candidate.
- In-script calibration split: one deterministic group-disjoint holdout (~15% of train drawings, ~70 drawings, ~900 pairs) built in the script from train only; fits the handful of decode constants (edge threshold, repair strictness, class multipliers, temperature). It is a separate set from the dev CV used for design decisions. Because it is small and noisy, constants are chosen on plateaus (smoothed grid, mid-range).
- Hyperparameters of the networks are fixed from dev CV (not re-searched in script) unless a bounded in-script search is added in the roadmap step 9 with its own held-out check.

Proxy biases (direction and reason):
- Scenario-grouped CV is the right axis, so scenario novelty is matched. Writers (students) probably overlap between train and test while scenarios do not; if instead writers are disjoint in test, the proxy is optimistic by an unknown small amount (handwriting domain shift). Cannot be derived from the files; flagged.
- Size shift (test n mean 7.0 vs 5.5) makes the global CV optimistic: larger graphs mean more edges to get right and ~1.7x more pairs weight. Estimated: mean-n-matched strata score several F1 points below the all-sizes number (U, size of gap measured by stratum analysis).
- The in-script 15% holdout trains on less data than the full train, so its numbers are slightly pessimistic relative to a refit on 100%.
- A self-built "gold box + OCR" oracle is optimistic; the full pipeline cross-fitted number is the one to quote.
Plan to quote the expected private band from the n >= 7 stratum of the grouped CV minus a margin for handwriting/scenario shift, not from the all-sizes mean.

## Overfit/underfit risks

Effective sample size is roughly scenarios x students, not 462 drawings or 6,207 edges.

Overfit risks and mitigations:
- Scenario/layout memorisation by the detector and pair classifier (few scenarios). Mitigation: strong geometric/photometric augmentation of real images (scale, small rotation and 90-degree rotation, flips with labels transformed consistently, perspective, brightness/contrast, blur, JPEG), fixed short schedules, group-disjoint validation, ImageNet-initialised backbones with moderate LR and weight decay (capacity-ladder rung: pretrained backbone fine-tuned end-to-end only for detection where there are thousands of boxes; pair model with strong aug).
- Pair classifier shortcutting on absolute layout (left-to-right bias) or on node type alone. Mitigation: relative geometry only; flip/rotate augmentation; hard negatives (nearest non-linked neighbours, crossing-arrow cases); compare with the geometry-only baseline.
- Text/semantic shortcut (text strokes inside crops hint at scenario). Mitigation: ink in node masks is retained but scenario-grouped CV catches it; no text features are given to the edge model.
- Decode constants (5-6 scalars) fitted on a ~70-drawing holdout: plateau selection, report cross-fitted effect in dev CV.
- Executor-derived loss weights (below) can overfit; keep as one measured step.

Underfit risks and mitigations:
- Down-sizing destroys markers and arrowheads (events, X/+ gateway marks, terminate fill, thin arrowheads). Mitigation: detector at ~1024 px only localises; type classification and link evidence use native-resolution crops.
- Too weak link evidence: arrows cross, curve, touch wrong shapes. Mitigation: pair crops include source/target/other-node mask channels so the model sees what else the ink could belong to; hard-negative training; degree-aware decode; then, only if measured, a set-level (all-pairs-in-drawing) context model.
- Loss mismatch with metric: edges that change many pair relations matter more; weight drawings by pair count (metric's own weights); measure effect-weighted edge loss.
- Key assignment weaker than the rest: constrained recognition (score the given names) instead of free OCR; rotation search.
- Too little training for the detector given 5-6 box classes and ~2,500 activities: use a pretrained ImageNet backbone with FPN, at least ~20 epochs, report AP per class.

## Recommended approach (primary + fallback)

Primary: "detect -> classify -> link -> match -> execute", lean, four small trained components plus the allowed rule engine.

1. Preprocessing (fixed): apply the single EXIF policy identified in diagnostic 9; cache each image at native size (<= 2000 px) in RAM and a 1024-px long-side copy for the detector.
2. Node/pool detector (trained in script): torchvision Faster R-CNN (or FCOS) heads on a timm ImageNet backbone with FPN (convnext_tiny or resnet50; pin revision), 1024 px, AMP. Classes: activity (task/subProcess/callActivity merged), event, gateway, pool, lane (lane only to disambiguate nested rectangles, unused downstream). Merging the event subtype and gateway subtype away from the detector pushes the small-detail decisions to stage 3. Loss weights per drawing by pair count.
3. Node-type crop classifier (trained in script): native-resolution crops (e.g. 128-160 px with padding) of each detected node box, small ImageNet-pretrained CNN, classes: activity, event (any non-terminate), terminate end, choice gateway (X, blank, event-based), parallel gateway (+). Trained on gold boxes with jitter and on detector boxes. Executor-relevant classes only (see Structural signals); a 5-way head with strong flip/rotate augmentation.
4. Link reader (trained in script): for each ordered candidate pair (u,v) with v among the K nearest nodes of u by box gap (K from diagnostic 5, about 8-10), build a union-box native-resolution crop (padded, warped to ~224 px) with 6 channels: RGB plus masks for source u, target v, all other nodes; ImageNet backbone with extended stem (copy weights for RGB, zero-init mask channels), small head, plus embeddings of the two node classes and normalised relative geometry. Output 3-way {none, sequence, message}. Ordered pairs are classified separately so direction (arrowhead end) is learned; each undirected candidate is scored twice. Trained on gold boxes with jitter plus detector-proposed boxes (to remove exposure bias), with hard-negative mining (nearest non-linked, pairs through crossing arrows). Edge type is cross-checked against pool membership (diagnostic 4).
5. Key assignment (trained/fine-tuned in script plus an allowed pretrained recogniser): for each detected activity box (top-m by score, m = n + a small fixed margin), run a general handwriting recogniser (TrOCR small/base handwritten or a native-transformers general OCR/VLM; HF-hosted, no remote code, no diagram-tuned checkpoint) over four rotations of the crop, score each given name by teacher-forced length-normalised log-likelihood (plus character n-gram similarity of the free decode as a second score), and solve a Hungarian assignment of the n names to the m boxes with unmatched boxes dropped. A small learned combiner (logistic over the two scores, rotation, box-size features) is fitted on train gold boxes. If the oracle assignment accuracy (diagnostic 15) is below ~95%, add a fine-tune of the small recogniser's decoder on train word crops (17,415 word boxes) as a measured roadmap step.
6. Decode to a valid graph: exactly n activity nodes (top-n by matching, within-row only); node classes from stage 3; edges from stage 4 above a threshold; message edges also where pool-crossing; structural repair using degree priors learned from train gold (diagnostic 2): events/start no incoming, end no outgoing, every other node needs >= 1 incoming unless it is the target of a message edge and >= 1 outgoing unless it is an end/terminal, gateway split/join patterns; repair adds the best-scoring missing edge and drops the worst violating edge. Terminate scope by pool containment.
7. Execute: implement the stated token game (rules 1-8) as a memoised exploration of marking states with a backward fixpoint "which activities can still fire after this marking", then X>Y iff some transition fires X into a marking from which Y can still fire. Fixed deterministic state cap with a logged counter (data-determined, never clock-based). Relations by the stated before/after/either_order/exclusive rule. Write all n(n-1)/2 tokens.

Why this fits THIS data: the answers are an exact function of the diagram graph, the train set has full graph supervision (nodes, flows, pools) so each stage has direct labels; the executor removes the need to learn relation logic from only 6,290 pair labels; activity names are given, so recognition reduces to constrained assignment among <= 14 candidates.

Capacity ladder: ImageNet backbones, fine-tuned end-to-end for detection (thousands of boxes) and for the crop heads (tens of thousands of crops); recogniser frozen/zero-shot first, then decoder-only fine-tune only if the oracle shows it pays.

Fallback (also the step-2 baseline to get a valid end-to-end submission quickly): same detector, crop classifier, key assignment and executor, but the link reader replaced by a trained geometry-and-type pair scorer (small GBDT or MLP on normalised box geometry, node classes, gap, alignment, plus one or two learned image scalars from a tiny CNN on the straight segment between facing box sides), with the same degree-repair decode. Expected to score clearly below the primary; its value is validity and a stage-decomposition baseline. A second-level fallback if the detector underperforms: raise detector resolution or swap backbone along the capacity ladder, not change the pipeline.

Compliance of the primary: real training of detector, classifier, link reader (and optionally recogniser decoder) inside the script from ImageNet/general weights; executor explicitly allowed; strip-the-ML test: removing the learned components leaves only the executor and constraints with no diagram to read.

## Rejected options

- Generative VLM (LoRA fine-tune of a general VLM to emit the graph as text) : general VLMs are allowed in principle, but 462 drawings, 2000 px inputs, long structured outputs and a ~45 min budget on A10G make it slow and unstable, spatial arrow reading is a known VLM weakness, deterministic decoding of long JSON is fragile, and VLM pretraining mixes may contain flowchart data (grey against the diagram-weights ban). Kept only as a possible OCR component, not as the graph reader.
- End-to-end set-prediction graph models (DETR/Relationformer style): too data- and schedule-hungry for 462 images in <= 45 min from ImageNet init.
- Direct pairwise relation classifier from image + two box positions (skip the graph): relation is a global function of the graph; only 6,290 pair labels versus 6,207+794 edge labels and per-node labels; discards the executor (which the rules explicitly provide) and cannot enforce validity. Expected to land near the geometry-order floor (rough guess 0.35-0.45).
- Classical vision arrow tracing (skeletonise, Hough, hand-written arrowhead rules): rule-based, violates the "model must do the learning" spirit and the strip-the-ML test, and has no stroke labels to calibrate against.
- COCO-pretrained detector weights (torchvision download host not allowed; DETR on HF is grey on "general-purpose"): rejected for the primary; revisit only after a reviewer answer, gain unlikely to justify.
- DINOv2/CLIP/SAM-style backbones: "general-purpose" is plausible but not named in the description; keep behind a reviewer question. ImageNet timm backbones are compliant under every reading.
- Synthetic diagrams rendered from gold graphs or collage of real crops to enlarge data: banned (synthetic data / external generation).
- Pseudo-labelling on test, test-time training, test-wide thresholds: banned.
- EasyOCR/Tesseract-based text: EasyOCR weights come from GitHub (not allowed); Tesseract is weak on handwriting and not HF/timm-sourced.

## Fixed work plan & runtime budget

All counts are planned values to be profiled once on A10G then hardcoded; no wall-clock branching, no `os.cpu_count`, no `cuda.is_available` switches, fixed `num_workers`, seeds, pinned timm/HF revisions, `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)`. Known determinism hazard: RoIAlign/grid_sample backward are non-deterministic on CUDA; prefer decode constants robust to that noise and run the final script twice and diff.

Training uses a deterministic scenario-group split of the train set: ~85% fit, ~15% calibration holdout. Default ship: models trained on the 85% fit set, decode constants fitted on the 15% holdout, no refit. Optional refit on 100% with the identical fixed recipe only if the dev CV shows the extra 15% data beats the cost of reusing constants (verify transfer first).

Estimated A10G time (to be profiled; estimates, not measurements):
| Stage | Plan | Est. |
|---|---|---|
| Read, EXIF policy, cache 462+156 images | fixed | 1-2 min |
| Detector train (85%, ~390 drawings) | ~20 epochs, batch 4, 1024 px, AMP, convnext_tiny-FPN | 11-13 min |
| Node-type crop classifier | ~6 epochs on ~12k crops (gold-jitter + detector boxes), small CNN | 2 min |
| Link reader | ~8 epochs over ~40-60k ordered candidate pairs per epoch, 224 px, 6-channel resnet34-class backbone, AMP, 4-8 fixed workers | 6-8 min |
| Detector/classifier/link inference on holdout + test | 70 + 156 drawings | 2-3 min |
| Recogniser scoring | ~226 drawings x ~8 boxes x 4 rotations x ~7 names teacher-forced | 3-5 min |
| Decode-constant fitting on holdout (executor runs over small grids) | fixed grids | 1-2 min |
| Test decode, execution, write, validate | fixed | 1-2 min |
Total about 28-37 min, target <= 45 min with >= 30% headroom; shrink epochs or resolution (fixed constants) if profiling exceeds. Memory: batch 4 at 1024 px AMP for the detector and 128-256 crops at 224 px fit in 24 GB; images cached in RAM (~618 x ~12 MB uncompressed worst case, store JPEG bytes or resized copies to keep it below a few GB).

In-script checks: input schema and file existence, sample_submission columns, id order, every pair of every test row present exactly once, tokens in the allowed set, re-read written CSV with `keep_default_na=False`. Executor state-cap hits, repair counts and any drawing falling to a deterministic fallback relation set are logged loudly (data-determined, not clock-determined); assert the fallback fraction is tiny rather than silently degrading. No per-drawing try/except that hides failures; no import fallbacks.

## Metric-aware training & decode

(i) Back-solving the metric: the metric is a deterministic function of the graph, so the supervised targets are the graph objects, not the relations. The executor is the differentiable-free "loss layer"; its sensitivity is used for weighting.
(ii) Metric-matching weights: weight each drawing's losses by its pair count n(n-1)/2 (the pooled metric's weight), tested against sqrt-weighting for variance. Detector and link reader share the weighting.
(iii) Effect-weighted edge loss (roadmap step, measured): from the gold executor, compute for each gold edge the number of pair relations that change if it is deleted, and for hard negatives the number that change if added; scale the BCE weight modestly by this (capped), because "a few misread arrows change many pair relations".
(iv) Probabilistic outputs need calibration: train edge head with BCE, initialise biases at base rates (per node-pair class), fit one temperature plus per-type threshold on the 15% holdout (cross-fitted in dev CV).
(v) Decode, first order (primary): threshold edges, repair degree/type violations, execute once. Constants: edge threshold, repair strictness, class multipliers for relation choice (see vi).
(vi) F1-aware relation choice: macro-F1 over four classes is not argmax-optimal; choose argmax_k w_k * P_k where w_k are four multipliers fitted on the holdout (positive, bounded by the class prior, plateau-selected). With a single executed graph P is one-hot, so w_k matters only in step (vii).
(vii) Expected-utility decode (roadmap step 6, only after the lean design is measured and shows a gain above noise): draw S fixed-seed graph samples from the calibrated edge/node-type posteriors under the repair constraints (and the second-best key assignment), run the executor on each, average the two booleans a, b per pair into P(before/after/either/exclusive), then apply the class multipliers. Executor speed from diagnostic 1 decides feasibility; S is fixed (e.g. 16-32). This is the principled handling of misread-arrow risk but is not part of the primary (primary-design gate).
(viii) Hard constraints at decode: exactly n activities, one-to-one name assignment, degree and type constraints, messages only across pools. Constraints combine with model evidence; they are not a predictor by themselves.
(ix) Mirror structure: relation(Y,X) is the mirror of relation(X,Y); no key-letter information is used anywhere.

## Structural signals

Invariants of the data-generating process, each turned into an augmentation, constraint, prior, canonical form, or auxiliary target:
- Executor-relevant node classes are only five: activity; event (start/intermediate/plain end, all behave as "other node"); terminate end event; choice gateway (exclusive and event-based behave identically under rules 4 and 6); parallel gateway. Subprocess/callActivity markers and loop/multi-instance markers are irrelevant (rule 1), event message/timer subtype is irrelevant (message behaviour comes from explicit message flows). Lanes are irrelevant. This shrinks the vision problem.
- Start events and any node without an incoming flow are all triggered once at the start (rule 5), so missing an incoming edge turns a node into an extra start (strong effect); the decode and loss respect that asymmetry (recall of incoming edges is protected by the repair step).
- Collapsed-pool message flows never block and the pool is not a node: they have no effect on relations; do not read them.
- Pool membership by box containment gives (a) message vs sequence classification for flows crossing pools, (b) terminate scope; verify on train (diagnostic 4).
- Gateway degree patterns (split: 1 in, >= 2 out; join: >= 2 in, 1 out), start events in=0, end events out=0, tasks mostly in=1/out=1: learned degree priors from train gold, used in repair and as features.
- Exactly n activity boxes per drawing, names distinct (one-to-one assignment); n is given per row.
- Augmentation symmetries: horizontal/vertical flips and 90-degree rotations of the image with node boxes and edge labels transformed consistently; the edge direction label is unchanged (it is semantic), and masks flip with the image. Do not assume reading direction is invariant: keep a flip probability and measure under grouped CV (left-to-right flow is a real prior). Small rotation, scale, perspective, photometric and JPEG augmentation (photos and scans). Text recognition is run on unflipped crops over four rotations.
- Relation mirror symmetry: random relabelling of keys during any training that uses relations.
- Sibling/leakage control: groups derived by union-find (not just an id) for splits; no key or id features; no cross-row features at test.

## Experiment roadmap

1. Contract, metric, grouping (stop when all pass): metric unit tests (perfect 1.0; reversed before/after 0.5; constant-before about 0.113 on train; prior-random about 0.25); scenario groups built and inspected; EXIF/box alignment verified; submission writer and validator tested on sample_submission shape.
2. Executor oracle: 462/462 gold reproduction, profile state counts and runtime, fix the state cap. Stop only on exact reproduction.
3. Error-sensitivity study (diagnostic 6) and stage-oracle matrix (diagnostic 16): fixes realistic per-stage targets (for example edge F1, key accuracy) before any modelling.
4. Cheap end-to-end valid baseline (the fallback): detector, crop classifier, geometry-based linker, constrained-OCR assignment, executor, validator. Record grouped-CV stage metrics and macro-F1. Submit once (credit) to confirm format and pipeline validity.
5. Representation and structural insight: primary link reader (6-channel pair crops) vs geometry baseline, paired on identical folds, 2 split seeds; edge-recall gain must exceed noise. Then resolution/backbone ladder (convnext_tiny vs resnet50; 1024 vs 1280 for the detector; native-crop sizes for the link reader), stop at the lowest rung that wins.
6. Key assignment: oracle accuracy; add rotation search, combiner, and (only if < ~95%) decoder fine-tune on train word crops. Stop criterion: pairs lost to assignment errors < one third of total loss in the oracle matrix.
7. Metric-aware loss and decode: pair-count drawing weights, effect-weighted edge loss, degree repair on/off, threshold/temperature, class multipliers (cross-fitted). Then, only if a gain > noise remains, the Monte-Carlo expected-utility decode (primary-design gate).
8. Diversity (later, optional): a second link reader with a different backbone, or a set-level context model (transformer over all node tokens with pair heads) fed by the first model's out-of-fold scores; blend by z-score/rank. Each must beat the single model by more than noise.
9. In-script holdout calibration implemented exactly as shipped (fit 85%, calibrate 15%), compare against dev CV; decide refit on 100% with a verified-transfer test; bounded fixed-trial HPO only if CV can resolve it, with its own held-out check.
10. Final fixed-plan run from a clean `working/`, run twice and diff predictions; check runtime headroom, executor state-cap hits, fallback count, output distribution of relations versus train shares, validator output.
Credits: baseline, best single, ensemble/final. Do not tune on the public LB.

## Compliance audit

Run against CLAUDE.md section 7 and the strategist self-audits:
- Test file use: only per-drawing inference (detection, classification, link scoring, OCR scoring against that row's own names, execution). No test-wide statistic, normalisation, vocabulary or threshold. Result: pass.
- Time/environment branching: none planned; time used for logging only; fixed counts; one device. Result: pass (to be re-checked in code).
- Hard-coded constants: all decode constants, thresholds, degree priors and class multipliers are fitted in-script on the train-derived holdout/train labels, not pasted from offline runs or from public-LB probing. Network hyperparameters fixed from dev CV are architecture/schedule choices, state them in comments. Result: pass with note.
- External data/weights: only provided data; timm/HF general ImageNet and general OCR/handwriting weights; no diagram-fine-tuned weights, no hosted API, no synthetic data. Result: pass; COCO/DINOv2-style choices deferred to reviewer.
- Strip-the-ML test: remove detector, classifier, link reader and recogniser: nothing remains to execute. Residual rules are the explicitly allowed executor plus constraints on trained scores. Result: pass; the weak point is that the decode repair rules and "pool crossing means message" are hand-written, keep them thin and measure the ablation.
- Whole-test aggregation: top-n activity selection and Hungarian use only the row's own n and names; no rank normalisation across test; MC samples (if used) per drawing. Result: pass.
- Related-row leakage: grouped splits by scenario; no key/id/order features; names matched only within the row; no train-name vocabulary prior. Result: pass.
- Label-derived statistics: degree priors and class shares from train labels, fit on the fit split only for the shipped holdout calibration. Result: pass.
- Prime directive: real training of 3+ models in script. Result: pass.
- Source hygiene: readable, < 512 KB, no encoded blobs. To verify at implementation.

## Open questions & assumptions

Reviewer questions (with the plan under each reading):
1. Is a handwriting recogniser such as TrOCR (IAM-fine-tuned) acceptable as a "general handwriting recognition model"? If no: fall back to a general OCR component only, or train a small recogniser from ImageNet features on the 17,415 train word boxes (more training effort, lower accuracy, more reliance on the constrained assignment).
2. Are COCO-pretrained detectors (DETR from HF) and general vision backbones such as DINOv2/CLIP counted as "general-purpose"? Primary assumes no (ImageNet timm only); if yes, test as a swap in step 5 with the same gating.
3. Is scoring the row's supplied names against box crops (constrained recognition) allowed? Assumed yes (row's own input).
4. Are thin rule layers around trained scores acceptable (pool-crossing message classification, degree repair, exactly-n activities, executor)? Assumed yes per the description's allowance for the execution rules; plan to report the ablation to show the models carry the result.
5. Is holding out ~15% of training for decode-constant fitting (no refit by default) acceptable? Assumed yes.
6. Is a 1-h ceiling on A10G assumed correctly? Description is silent on runtime.

Assumptions: single A10G; HF/timm downloads only; time limit as above; `train_diagrams.jsonl` usable as supervision; the jsonl boxes align with the EXIF-handled images (diagnostic 9); the executor reproduces all train labels as claimed (diagnostic 1).

Expected score (estimate, not a promise): pooled macro-F1 on the test pairs roughly 0.45-0.65, central about 0.55, reasoning: strong node detection and gateway typing are plausible, but one-in-ten edge errors and a handful of key swaps already move many pair relations (to be quantified by the sensitivity study), and test drawings are about 1.3x larger. Fallback geometry-linker baseline estimated 0.35-0.45. Random about 0.25. The proxy (scenario-grouped CV) is expected to be a few points optimistic on the n >= 7 stratum and more optimistic overall.

Could not verify (no data): every item under Data findings marked U, the executor oracle, detector/link-reader/OCR accuracy, runtime numbers (planned values only), and whether EXIF/box alignment, pool/message regularities and scenario cluster structure hold.
