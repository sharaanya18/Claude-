# U7 — Dual-angle ground-footprint reconstruction (user-supplied, verbatim; domain not stated; compute not stated)

## Objective
Reconstruct every visible building's ground footprint from two co-registered overhead views of the same chip. The views have materially different viewing angles and share one normalized ground frame. This is classless instance segmentation. It is not scene classification, candidate ranking, footprint-count regression, or future-state forecasting. A submission must recover a variable number of polygon instances and their geometry; predicting only a count or one enclosing region is insufficient.

## Case construction
Each case is a 225 m ground cell derived from one 450 m parent tile. The two images are real multispectral observations of that cell from a near-vertical view and a severe oblique view. They were resampled onto the same published ground grid before deterministic RGB rendering. The view-angle separation is large enough that roof displacement, occlusion, shadow, and facade visibility differ substantially between the images.
Gold instances come from published ground-footprint vectors in the shared map reference system. Footprints are clipped to the child-cell boundary, converted to normalized image coordinates, and simplified with a fixed sub-pixel ground tolerance. They are not threshold-derived masks or invented rectangles.

## Split and leakage controls
1,600 training cases and 400 evaluation cases. Complete 450 m parent tiles are assigned before the four 225 m child cases are derived. Both angle views, all four children, and every clipped copy of a boundary-crossing footprint remain with their parent. Training parents occupy a western spatial region and evaluation parents occupy an eastern region. One full 450 m source-tile column between them is unused. Zero decoded-image hash overlap and zero cross-split perceptual-hash pairs within Hamming distance two. Source coordinates, collection names, tile indices, and original filenames are absent; release IDs are opaque.

## Public files
train.csv and test.csv: case_id, view_a_path, view_b_path. Images are 160x160 RGB JPEGs. train_labels.csv: case_id, footprints_json. sample_submission.csv supplies the exact test IDs and output columns. Test targets exist only in the private answer file.
Submit CSV with exactly: case_id,footprints_json. footprints_json is a JSON list of polygons; each polygon is a list of at least three [x,y] vertices with finite coordinates in [0,1]. Origin upper-left; x right, y down. Do not repeat the first vertex at the end. Use [] when no building is present. At most 128 polygons and 128 vertices per polygon.

## Score
Mean classless instance-segmentation F1 across IoU thresholds 0.50, 0.55, ..., 0.95. Polygon IoU by deterministic scanline rasterization in the shared ground frame at 192x192 cells. An exactly identical valid polygon receives IoU 1 even when smaller than one raster cell; two different polygons never match merely because both rasterize to empty masks. At each threshold, deterministic maximum bipartite matching gives each prediction and gold polygon at most one match. Unmatched predictions and missed buildings lower F1, so overprediction is penalized. Invalid JSON, invalid polygons, missing/extra cases, duplicate IDs, or wrong schema score zero.
The threshold average rewards footprints accurate under strict IoU, not approximate centroids. Empty predictions score zero on nonempty cases, and speculative polygons reduce precision. A full-chip polygon, centroid-rectangle prior, training mean shape, single-view thresholding, single-view edge components, and severe-view thresholding all score below 0.003; the complete known answer scores 1.0.

## Allowed methods
Train segmentation, detection, polygonization, registration, or multi-view fusion models using only the supplied public images and training polygons. Use deterministic image processing, geometric augmentation, cross-validation, and pretrained visual encoders that do not reveal source identities. Use either view independently, although the task is designed to benefit from their complementary occlusion and parallax evidence.

## Not allowed
Reverse opaque identifiers or recover source coordinates, tile names, or collection metadata. Query external imagery, maps, building databases, source archives, or hidden answer files for evaluation cells. Manually annotate evaluation images or exploit row, filename, or polygon ordering as a target proxy.

## Scope and limitations
One metropolitan acquisition campaign and a fixed sensor family. Severe off-nadir views contain orthorectification artifacts and genuine occlusion. Edge buildings are clipped by the case boundary; published labels can contain omissions or ambiguous outlines. Scores measure agreement with this frozen annotation release.
