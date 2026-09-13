# Stage 4 — Tabular Q-Learning after Stage 4.5 refinement

The final selected Q-table was trained from scratch for 200 complete 2023
episodes. Alpha=0.10, gamma=0.95, epsilon=1.0 initially, decay=0.98 per episode,
minimum=0.05, seed=42. Validation is full-year 2024; test is full-year 2025.
The final episode is selected without tuning or checkpoint selection.

The state has 3 price bins, 4 net-load bins, 4 SOC bins, 4 time blocks and
2 weekend values: 384 states. Net-load bin zero explicitly represents <=0;
positive P33/P66 and price P33/P66 are fitted only on 2023. The other bins and
actions remain interpretable. Environment actions are -1 discharge, 0 hold,
+1 charge; Q-table action indices are separate array indices.

The update remains Q(s,a) <- Q(s,a) + alpha *
[reward + gamma * max Q(s_next,.) - Q(s,a)], with zero bootstrap on terminal
transitions. Reward is negative raw grid cost plus throughput penalty inside
the negative sign. Terminal inventory valuation is a separate evaluation KPI.

## Commands

```bash
python -m unittest discover -s tests -v
python -m src.training.train_q_learning
python -m src.evaluation.evaluate_q_learning
```

Training writes readable model files and greedy validation results. Evaluation
loads the completed model, checks provenance hashes and performs frozen greedy
test inference. If the completed model already exists, run only evaluation;
the training command starts a fresh run and is not a checkpoint-resume command.
Optional `--verify-reproducibility` repeats the full training. That option was
not used for this refinement; seeded determinism is covered by repeated small
complete runs in the test suite. The legacy full-run repeat is not evidence for
the new experiment.

## Current results

| Strategy | raw_electricity_cost_eur | terminal_soc_adjusted_cost_eur | adjusted_cost_savings_pct | pv_self_consumption_pct | efc_per_day |
| --- | --- | --- | --- | --- | --- |
| No Battery | 29,881.606847 | 29,881.606847 | 0.000000 | 91.926349 | 0.000000 |
| Rule-Based | 28,679.986474 | 28,684.041001 | 4.007702 | 92.357980 | 0.802565 |
| Q-Learning | 29,877.612552 | 29,881.667079 | -0.000202 | 91.926349 | 0.000521 |

Q-Learning does not charge or capture PV in validation/test. It uses 38 kWh
from initial inventory and ends at minimum SOC. Its adjusted test cost is
0.060232 EUR above No Battery. It does not beat Rule-Based on purchasing cost
or PV capture, although it cycles less and avoids the baseline's higher peak.
The model and hyperparameters were retained despite poor performance; no test
tuning was used. All 37 tests passed, and frozen replays matched exported
validation/test outputs. This is one synthetic scenario and seed.

See [Stage 4.5](stage_4_5_refinement.md) for full validation/test metrics,
training history, action versus flow analysis, exact formulas and limitations.
Models are in `outputs/models/`, results in `outputs/results/`, and seven shared
figures in `outputs/figures/`. Old incompatible artifacts are archived. DQN is
not implemented.
