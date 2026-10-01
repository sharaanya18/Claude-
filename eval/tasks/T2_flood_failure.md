# Task T2 — Visible flood-failure mechanism recognition (paraphrased description)

One photograph of a flood-damaged bridge per row (~841 train, ~215 test; 960x640 JPEGs). Multi-label: predict a probability for each of 4 visible failure mechanisms: `support_scour`, `debris_obstruction_or_impact`, `approach_or_embankment_washout`, `structural_displacement_or_collapse`. Base rates ~13.6% / 12.6% / 27.7% / 11.3%.

Photos come from bridge SITES; the site is the id prefix before the first `-`. 54 train sites, 13 test sites, disjoint. Using the site prefix as a predictive feature or to look up site metadata is banned (it may be used only for grouping).

Metric: macro average over the 4 labels of site-weighted average precision (each photo weighted by 1 / photos-at-that-site, so every site counts equally).
Submission: `id, prediction` where prediction is a JSON object with the 4 float keys.
Rules: runtime limit 30 minutes on an A10G (overrides the general default); general-purpose pretrained weights from HF/timm allowed; real training/fine-tuning inside the script is required by the platform; no external data.
Positive-site counts in train (of 54): washout 44, debris 28, scour 21, collapse 14.
