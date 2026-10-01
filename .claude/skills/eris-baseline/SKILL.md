---
name: eris-baseline
description: Write the first end-to-end solution.py for an Eris challenge from the compliant scaffold: the right model class from day one (not a toy), in-script validation, a challenge-requirements map, fixed work plan, honest grouped CV with the exact metric. Use after the contract, audit, plan and validation exist.
---

# /eris-baseline

Start from `.claude/scripts/solution_template.py` (already: argv contract, seeds, fixed constants, time logging only, validator, atomic write, requirements map). Fill the plan's **lean primary**
at the cheapest adequate capacity rung (frozen probe or GBDT first). Defaults: tabular → LightGBM grouped 5-fold; text → strongest affordable encoder with a trained head (LP-FT if tiny
data); vision → frozen DINOv2/ConvNeXt probe + per-label shrinkage, then LP-FT; retrieval → bi-encoder retrieve + trained reranker/listwise GBDT; structured → exact enumeration with potentials.
If weights are banned or the machine is CPU-only (description), follow that.

Fill the **challenge requirements map** in the docstring first: each explicit requirement/ban of the description with where the code satisfies it (the platform's Prompt Compliance check reads the code against the challenge text).

Non-negotiables: no clock/hardware/env branches; seeds fixed; reads only `sys.argv`; outputs validated against the sample and the description's grammar; in-script `metric_self_test`; no test statistics;
comments explain *why*. After writing: `python3 .claude/scripts/compliance_scan.py solution.py` must show no ERROR; run `python3 .claude/scripts/local_run.py <challenge dir>`.

Record in `CHALLENGE_NOTES.md`: exact-metric OOF mean ± std (per fold, per slice), constant baseline, frozen-probe score if relevant, runtime profile, which plan item this implements.
Hand off to `/eris-experiment`.
