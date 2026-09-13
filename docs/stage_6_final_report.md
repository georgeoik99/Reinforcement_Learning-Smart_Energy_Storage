# Stage 6 Final Report

The project is complete and ready for GitHub review. The environment, dataset,
reward, benchmark strategies and Stage 1–5 scientific artifacts were preserved.
Only seeds 7 and 123 were trained; seed 42 was reused. No seed was selected.

## Authoritative 2025 comparison

| Strategy | Adjusted cost (€) | Savings (€) | Savings (%) | PV self-use (%) | PV surplus captured (%) | EFC/day | Peak import (kW) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Battery | 29,881.607 | 0.000 | 0.000 | 91.926 | 0.000 | 0.000 | 71.364 |
| Rule-Based | 28,684.041 | 1,197.566 | 4.008 | 92.358 | 5.346 | 0.803 | 115.833 |
| Q-Learning | 29,881.667 | -0.060 | -0.000 | 91.926 | 0.000 | 0.001 | 71.364 |
| DQN | 28,250.404 | 1,631.203 | 5.459 | 95.498 | 44.238 | 1.514 | 116.395 |

## Three-seed DQN robustness

| seed | Adjusted cost (€) | Savings (€) | Savings (%) | PV self-use (%) | Surplus captured (%) | EFC/day | Peak (kW) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 28,250.404 | 1,631.203 | 5.459 | 95.498 | 44.238 | 1.514 | 116.395 |
| 7.000 | 28,741.318 | 1,140.289 | 3.816 | 94.186 | 27.989 | 0.748 | 121.027 |
| 123.000 | 28,547.788 | 1,333.819 | 4.464 | 94.040 | 26.183 | 0.989 | 117.059 |

Adjusted savings: mean 4.580%, sample SD 0.828 pp,
range 3.816–5.459%. This is sensitivity evidence,
not statistical proof.

## Final conclusion

DQN improves terminal-SOC-adjusted electricity cost and renewable utilization,
but its additional savings come with more battery cycling and peak-demand
exposure. Q-Learning adds complexity without useful operational value in this
experiment. More advanced AI is valuable only when the improvement justifies
the operational trade-offs.

## Delivery

- Seven curated figures: `outputs/portfolio_figures/`.
- Full business discussion: `docs/final_results_and_business_interpretation.md`.
- Exact robustness results: `outputs/results/dqn_seed_robustness.csv`.
- Exact seed-42 comparison: `outputs/results/final_model_comparison.csv`.
- Change inventory: `outputs/results/stage6_completion_report.json`.
- Tests: 59/59 passed.
- README: recruiter-focused and GitHub-readable.
- No Git operation was performed; Stage 6 stops here.
