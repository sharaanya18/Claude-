# Shared Adam Memory Repair: experiment log (scene-grouped, train only)
Metric proxy: pick quality = exact-simulated quality (learner.assess semantics) of the chosen order; random orders ~0.34, canonical ~0.34.
| id | change | pick quality |
|---|---|---|
| E1 | HistGBR scorer on physics features, 400 random candidates, grouped 4-fold | 0.41 |
| E2 | Linear-Gaussian state posterior (prior: J^T J span), true v, mean state | 0.53 |
| E3 | + 6-16 posterior draws, expected quality | 0.58-0.59 |
| E4 | prior = span of the 52 per-image gradients (zero mean), true v | 0.64 |
| E5 | learned v-hat (LightGBM, grouped) instead of true v | 0.55 |
| E6 | + draws with v-noise, top-128 re-scoring (solution.py), end-to-end 4-fold by scene, 144 requests | 0.604 (mean-only 0.543, canonical 0.343, random 0.334) |
Findings: only the response of the two Adam buffers on the request images matters (min-norm state from exact responses -> 0.93); v matters (scale and shape errors cost ~0.05 each); linearised diagnostics have 5-25% error (IEKF gave +0.01 at small noise); hyper-parameters are on a plateau.
Public leaderboard (user screenshots): top ~0.572, AI baseline 0.3698.
