---
name: eris-plateau
description: Use when CV has not improved beyond noise for three experiments. Forces a different direction: structural limitation, a different architecture or training strategy, an unused signal, or a diversity ensemble. Gives one concrete next direction with first steps.
---

# /eris-plateau

Gather: last three log rows, best CV/LB, `§ Error analysis`, the ceiling diagnostics. Answer:
1. **Structural limitation** of the current approach that tuning cannot fix (e.g. independent per-slot heads cannot represent a joint constraint; weak representation at the needed resolution; decode not matched to the metric).
2. **A different approach** named specifically (representation swap, joint enumeration, exact marginalisation, coupling, capacity-ladder rung, different assumption family).
3. **An unused signal**: structural invariants, symmetries, taxonomy recoverable from train, label-derived statistics done out-of-fold, auxiliary targets, cross-field features.
Check `eris-playbook` (metric-and-decoding, features, ensembling) for levers independent solvers converged on that this solution lacks. Before recommending an ensemble check OOF diversity (correlation < ~0.9)
and the union oracle. Output **one** direction with first steps, then `/eris-implement`.
