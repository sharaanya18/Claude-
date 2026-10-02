# Research index (2026-10-02)
Public-source research done to strengthen the Eris playbook. Evidence tags: [S] primary source read, [S-snip] search snippet only, [R] recalled. Nothing here was run on a real Eris challenge unless stated; promoted items live in `.claude/skills/eris-playbook/references/` (see the "Research additions" sections); unpromoted hypotheses stay in `references/proposals.md`.

| File | Scope | Highest-value takeaways |
|---|---|---|
| A_hidden_lb_generalisation.md | Private-LB generalisation | public score as z-score; corrected paired t; winner's curse arithmetic; label-shuffle + id-only audits; compliant adversarial validation; group-OOF calibration |
| B_tabular_and_forecasting.md | Tabular / forecasting | meta-tuned GBDT defaults; cross-fitted features; shrunk-ridge blends; TabM/RealMLP from scratch; booster determinism; do not ship TFMs |
| C_nlp_structured_generation.md | NLP / weak supervision | outcome-verified self-training controls; MBR selection; generation speed pitfalls; target-only loss; base-model KV/gating arithmetic |
| D_vision_audio_multimodal.md | Vision / audio / multimodal | LP-FT; WiSE-FT averaging; weight provenance + licences (torchvision download rule); group-aware CV; runtime formula |
| E_agent_baselines_and_benchmark_design.md | AI baseline + benchmark design | baseline is probably AIDE-like; agent failure modes; hidden-shift unit taxonomy; train-only diagnostics list |
| F_own_retrospective.md | Own evidence (3 challenges) | process errors and the template that would have saved the most time |

Out of bounds and not read: public repos holding other solvers' Eris solutions; leaderboard-probing papers.
