# Stage 5 — Vanilla Deep Q-Network

## Outcome

Stage 5 adds a standard Vanilla DQN while retaining the Stage 4.5 dataset,
environment, Rule-Based controller, Q-Learning model, reward and accounting.
The fixed final network lowers the 2025 raw and terminal-SOC-adjusted electricity
cost relative to all three benchmarks. It also cycles much more aggressively.
After the existing throughput penalty is included, DQN costs EUR
85.62 more than Rule-Based.
The added neural-network complexity therefore shows measurable bill value in
this synthetic test, but not an unqualified business improvement.

## DQN in plain language

Tabular Q-Learning converts continuous observations into 384 bins and stores
three values for each bin. DQN accepts seven normalized continuous values and a
small neural network approximates `Q(state, action)` for discharge, hold and
charge. This lets nearby observations share learned structure without making
the controller continuous-action or forecast-based.

```text
Tabular: continuous values -> bins -> 384-state Q-table -> 3 actions
DQN:    normalized continuous values -> 7-64-64-3 network -> 3 actions
```

The Q-network has 4,867 trainable parameters: Linear 7→64, ReLU, Linear
64→64, ReLU, Linear 64→3. Output indices map to environment actions as
0→discharge (-1), 1→hold (0), 2→charge (+1).

Uniform replay stores past transitions and samples minibatches to reduce
sequential correlation. A frozen target network supplies Vanilla DQN targets;
the same target network selects and values the maximum action, so this is not
Double DQN. Adam performs minibatch updates. Epsilon-greedy behavior explores
during training and is disabled for validation/test.

## State and normalization

The state is demand kWh, PV generation kWh, EUR/kWh price, SOC fraction,
hour sine, hour cosine and weekend flag. Demand, PV and price use z-scores fitted
only on 2023. SOC, cyclical hour variables and weekend remain in their bounded
forms. The fitted normalizer is serialized and checked against a fresh 2023-only
fit before test. Validation/test never fit or mutate it.

## Fixed configuration and training

- Training: 100 complete 2023 episodes, 876,000 environment steps.
- Network: 7→64→64→3, ReLU, CPU, PyTorch 2.14.0+cpu, seed 42.
- Gamma 0.95; Adam learning rate 0.001; Huber loss.
- Replay capacity 50,000; warm-up 2,000; batch 64; one update every 4 steps.
- 218,501 gradient updates; gradient-norm clipping at 10.
- Target synchronization every 1,000 steps; 876 synchronizations.
- Linear epsilon 1.00→0.05 over 350,400 steps (40 episodes), then 0.05.
- Reward remains `-(raw grid cost + 0.01 EUR/kWh throughput)`.
- The final episode was preselected. No checkpoint selection or 2025 tuning.

The first exploratory reward was EUR -31,849.23;
the final was EUR -29,527.97; the final
10-episode mean was EUR -29,446.66.
The final 10-episode mean Huber loss was
2.286408. Losses and Q-values stayed
finite; maximum recorded |Q| was 108.423.
All three actions remained present. These are training diagnostics rather than
evidence of greedy economic performance.

The episode-boundary checkpoint contained episode 100, model/target weights,
optimizer, full replay buffer and Python/NumPy/PyTorch RNG states. Its weights
matched the final model exactly. It was already complete, so continuation did
not retrain or resume any episode.

## Frozen 2024 validation

Validation was completed with epsilon zero and no changes to network, target,
optimizer, replay buffer, normalizer or RNG. Reloading the saved model reproduced
the hourly output exactly. Validation used only to screen for implementation or
training failure; the fixed configuration was accepted without changes.

| Strategy | raw_electricity_cost_eur | terminal_soc_adjusted_cost_eur | adjusted_cost_savings_eur | adjusted_cost_savings_pct | total_grid_energy_purchased_kwh | pv_self_consumption_pct | efc_per_day | peak_grid_import_kwh_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Battery | 29,929.207 | 29,929.207 | 0.000 | 0.000 | 293,147.373 | 91.920 | 0.000 | 74.455 |
| Rule-Based | 28,626.055 | 28,630.110 | 1,299.097 | 4.341 | 295,754.375 | 92.329 | 0.835 | 116.907 |
| Q-Learning | 29,925.229 | 29,929.284 | -0.077 | -0.000 | 293,109.373 | 91.920 | 0.001 | 74.455 |
| DQN | 28,340.381 | 28,344.435 | 1,584.771 | 5.295 | 294,811.562 | 95.199 | 1.495 | 119.282 |

DQN captures 40.578% of available
validation surplus and has 1.4947
EFC/day. Its adjusted cost is EUR
-285.67
relative to Rule-Based; negative means lower DQN cost.

## Frozen 2025 test comparison

The 2025 test was opened only after the validation gate recorded
`accepted_no_changes`. Evaluation used epsilon zero, no gradient/replay/target
updates and no model selection. A second model reload reproduced every hourly
result exactly. Existing three-strategy hourly files were revalidated and their
metrics matched the preserved Stage 4.5 comparison.

| Strategy | raw_electricity_cost_eur | terminal_soc_adjusted_cost_eur | adjusted_cost_savings_eur | adjusted_cost_savings_pct | total_grid_energy_purchased_kwh | pv_self_consumption_pct | efc_per_day | peak_grid_import_kwh_per_hour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Battery | 29,881.607 | 29,881.607 | 0.000 | 0.000 | 292,736.997 | 91.926 | 0.000 | 71.364 |
| Rule-Based | 28,679.986 | 28,684.041 | 1,197.566 | 4.008 | 295,188.527 | 92.358 | 0.803 | 115.833 |
| Q-Learning | 29,877.613 | 29,881.667 | -0.060 | -0.000 | 292,698.997 | 91.926 | 0.001 | 71.364 |
| DQN | 28,246.350 | 28,250.404 | 1,631.203 | 5.459 | 294,132.114 | 95.498 | 1.514 | 116.395 |

Full-precision metrics are in `outputs/results/stage5_final_comparison.csv`.
DQN raw cost is EUR 28,246.35; the EUR
4.054526 terminal shortfall adjustment produces
EUR 28,250.40. Adjusted savings versus No Battery
are EUR 1,631.20 (5.4589%).
DQN adjusted cost is EUR 433.64
below Rule-Based and EUR 1,631.26
below Q-Learning.

## Test behavior

Low price means below the training P33, EUR
0.087490/kWh. In
2,953 low-price hours, DQN requests charge
2,079 times (70.40%),
with 867 active flows and
32,418.08 kWh of grid charging.

High price means at or above training P66, EUR
0.104769/kWh. DQN requests discharge
1,593 times in
2,994 high-price hours; 1,037
requests deliver 33,522.15 kWh to load.

The test has 574 PV-surplus hours and
9,558.66 kWh available surplus. DQN requests charge
in 347 of those hours and stores
4,228.55 kWh, capturing
44.24%. It curtails
5,330.11 kWh and also takes
53,839.87 kWh from the grid. Positive physical
PV-to-battery flow supports the conclusion that DQN learned solar-storage
behavior in this synthetic test; it did not capture all surplus.

DQN issues 7,301 non-hold commands;
3,145 (43.08%)
produce non-zero energy flow. Zero-flow commands comprise
3,020 charge and
1,136 discharge requests. Of these,
4,070 are caused by SOC boundaries
and 86 are no-load
discharge requests outside a boundary. Requests and physical flows are reported
separately.

## Business interpretation

- DQN reduces both raw and terminal-adjusted cost versus No Battery, Rule-Based and Q-Learning.
- It captures meaningful PV surplus and responds strongly to low/high training-price groups.
- It buys 294,132.11 kWh from the grid, more than No Battery, because it grid-charges.
- It raises peak hourly import to 116.395 kW versus 71.364 kW without storage and 115.833 kW for Rule-Based.
- It reaches 552.566 EFC, or 1.5139/day, nearly twice Rule-Based; this is aggressive cycling.
- All battery strategies begin at 50 kWh and DQN ends at 10 kWh. Terminal adjustment prevents its inventory shortfall from appearing as free savings.
- DQN throughput penalty is EUR 1,105.13; raw cost plus penalty is EUR 29,351.48, versus EUR 29,265.86 for Rule-Based. On this declared reward-related measure, Rule-Based is better.

The extra AI complexity produces a measurable reduction in the synthetic energy
bill and meaningful PV capture. It also produces more cycling, more grid energy,
a slightly higher peak than Rule-Based, many infeasible-at-boundary requests and
a worse cost-plus-throughput-penalty result. Deployment value would require
battery-aging, demand-charge and hardware studies that this experiment omits.

## Verification and artifacts

All 54 tests passed: 37 preserved Stage 3–4.5 tests and 17 Stage-5 tests.
They cover dimensions, normalization isolation, replay behavior, Vanilla targets,
target synchronization, deterministic training/checkpoint continuation, exact
model/normalizer serialization, frozen inference, finite values, physical limits,
energy balance and action/flow accounting. Seven figures were generated and
visually reviewed. No Stage 4.5 artifact was overwritten.

The full pre-Stage-5 repository state contains 98 files and is preserved with
SHA-256 hashes in `outputs/archive/pre_stage5_repository.zip` and
`pre_stage5_manifest.json`. Exact created/modified/unchanged files are recorded
in `outputs/results/stage5_preservation_report.json`.

## Limitations and stopping point

This is one deterministic seed and one predeclared configuration trained by
replaying one synthetic calendar year 100 times. Current-hour demand/PV/price are
observed before action; no forecast, export, demand charge, capex, detailed
degradation, battery temperature or hardware constraint model is present.
Actions request full feasible discrete charge/discharge, which contributes to
boundary commands. Terminal valuation and EFC are proxies. Results do not prove
generalization to measured sites, other years or market regimes.

Stage 5 stops here. Stage 6, advanced DQN variants, PPO/SAC and Git operations
were not performed.
