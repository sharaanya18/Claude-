> NOTE: not verbatim platform text. Reconstructed from five ranked top solutions' own docstrings and
> code (primarily rank-2's header docstring and constants visible in rank-1 and rank-2, e.g.
> `META_COLS`, image sizes, threshold names). Unconfirmed against a real sample_submission.csv.

# Change-mark localisation on data-quality scatter charts (reconstructed)

Input: rendered chart images (scatter/line charts of some aggregated metric over time, per
rank-1's `C0`/`R0`/`R1` geometry constants and rank-2's `H_IN=64`, `L_IN=512` chart dimensions),
each with descriptive metadata columns `aggfn`, `aggunit`, `fieldtype`, `dataset_name` (rank-1's
`META_COLS`). Some charts carry one or more small visual "change marks" (annotations placed where a
data-quality issue was introduced); some charts carry none.

Task (per rank-2's docstring): for each chart, localise every change mark along the chart's
horizontal (time) axis — effectively a 1-D peak/keypoint detection problem per chart, with an
explicit "no marks on this chart" case the model must also decide (rank-2's `THR_EMPTY` /
"chart-level head for no marks"). Likely submission shape: one row per chart with either mark
positions (a list/JSON of horizontal bin or pixel coordinates) or a per-chart indicator plus
positions; exact column names were not visible in the slice of code inspected.

Implied scoring: a position-matching metric with a tolerance window (NMS/dedup constants like
rank-1's `NMS_PX=6.0` and rank-2's peak/empty thresholds suggest the metric rewards marks within
some pixel/bin tolerance and penalises both missed marks and false positives, but the exact formula
was not seen in the inspected lines).

Compute: rank-1 forces `DEV = torch.device('cpu')` explicitly; rank-2 allows CUDA if available
(`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) but sets deterministic cudnn flags,
suggesting a CPU-first or CPU-only environment is plausible. rank-1 uses `timm` pretrained backbones
(`hgnetv2_b0.ssld_stage2_ft_in1k`, `convnext_atto.d2_in1k`) fine-tuned per-chart; rank-2 trains a
from-scratch small conv/dilated-1D-conv model per chart pixel column, with decoding thresholds fit
on its own out-of-fold predictions.

Unverified: exact submission grammar, the real metric, runtime/hardware limit, row counts, whether
pretrained backbones (as rank-1 uses) are actually allowed by the challenge text — this is a live
compliance question for this slug and is flagged, not resolved, by this reconstruction.
