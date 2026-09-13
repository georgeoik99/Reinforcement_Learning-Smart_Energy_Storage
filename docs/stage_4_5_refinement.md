# Stage 4.5 — Environment & Evaluation Refinement

This experiment supersedes the original Stage-3/Stage-4 numerical results.
The objective is a better common simulation before DQN, not an improved RL ranking.
The Q-Learning update, action mapping, Rule-Based rules and battery configuration
are retained. No DQN or Git operation is part of this stage.

## Why the experiment changed

The legacy one-year experiment had only three surplus-PV hours in total and none
in validation/test. Its 70/15/15 split put test in November–December. Consequently
100% PV self-consumption provided no evidence of battery-based solar utilization.
The environment also removed more energy than the building could use during
discharge. Finally, ending with less stored energy artificially lowered a raw
bill comparison unless the change in inventory was reported and valued.

## Data: a full seasonal cycle for every split

The generator now produces 2023, 2024 and 2025 with a **100 kWp rooftop PV**
assumption. Demand retains the commercial/light-industrial activity, weekday/
weekend, seasonality and noise concepts. Dynamic prices retain hourly/seasonal
patterns, noise, spikes and occasional negative prices. The same PV solar-angle,
seasonal and cloud process is scaled from 47 to 100 kWp and capped at nameplate
energy for a one-hour timestep. Surplus rows are not inserted, forced or tuned.

Each year has an independent reproducible stream using
`numpy.random.SeedSequence([seed, year])`, with generator seed 42. Seasonal phase
uses the year's 365 or 366 days. Hourly labels are naive simulation time, without
DST gaps or repeated local hours. This is a synthetic business scenario, not
measured HEnEx, PVGIS, customer load or site-engineering data. Cloud and price
noise are simplified; the annual price generator's PV percentile is part of the
exogenous data-generating process, not a learned controller threshold.

The active file is `data/processed/energy_hourly_2023_2025.csv`:

- 2023: training, 8,760 hours.
- 2024: validation, 8,784 hours including 29 February.
- 2025: test, 8,760 hours.

The split rejects incomplete years, shuffled data, duplicate timestamps and gaps.
All controller price/net-load thresholds and the inventory valuation price are
fitted on 2023 only. Validation/test are never used to adjust parameters or bins.
The old `energy_hourly_2025.csv` is retained unchanged as legacy reference, with
an adjacent `LEGACY_DATA.md`; it is not the new 2025 test slice.

## Physical discharge and unchanged charge priorities

`BatteryModel.step` accepts an optional bus-side discharge cap. The environment
always supplies current residual building demand before the SOC transition:

```text
residual = max(demand - PV, 0)
discharge_to_bus = min(power_limit * timestep,
                       (SOC - SOC_min) * eta_discharge,
                       residual)
SOC_next = SOC - discharge_to_bus / eta_discharge
```

The energy retained when load is small remains in the battery. At zero residual
load a discharge action causes no SOC change, no throughput and no penalty.
`unused_discharge_kwh` remains in the shared schema and is asserted zero.
Standalone battery calls without a cap preserve their old physical API; all
environment/controller runs pass the cap. No export or energy destruction is used.

Charge remains the full feasible discrete-action amount:

```text
charge_input = min(charge_power * timestep, (SOC_max - SOC) / eta_charge)
PV_to_battery = min(max(PV - demand, 0), charge_input)
grid_to_battery = charge_input - PV_to_battery
SOC_next = SOC + eta_charge * charge_input
```

PV supplies demand before charging. A charge command can draw from the grid even
when PV surplus triggered it. This deliberately preserves the simple Stage-3
policy/accounting. Capacity remains 100 kWh; SOC bounds 10–95%; initial SOC 50%;
power limits 50 kW per direction; efficiencies 0.95 per direction; timestep 1 h.
Each evaluation resets once, not once per day or plotting week.

## Q-Learning: a distinct surplus state

The state is `(price_bin, net_load_bin, soc_bin, time_block, weekend)`.

| Component | Interpretation |
|---|---|
| Price, 3 bins | Training P33/P66; equality goes into the higher bin. |
| Net load, 4 bins | 0 = net load <= 0; 1/2/3 = low/medium/high positive net load. P33/P66 use positive training loads only. |
| SOC, 4 bins | Equal intervals within usable SOC: 10–31.25–52.5–73.75–95%. |
| Time, 4 bins | Night 00–05, morning 06–11, afternoon 12–17, evening 18–23. |
| Weekend, 2 bins | Weekday 0; Saturday/Sunday 1. |

This yields **384 states / 1,152 Q-values**. The saved legacy 288-state table is
incompatible and archived. The new agent is trained from zeros, without warm
starting from the old table. No additional state forecasts or continuous actions
are introduced. Observations are current-hour price/demand/PV and pre-action SOC,
using the same information assumption as previous stages.

The hyperparameters are unchanged: **200 episodes, alpha 0.10, gamma 0.95,
epsilon start 1.0, decay 0.98 per episode, minimum 0.05, seed 42**. Each episode
processes the 2023 sequence chronologically. The final episode is selected
without tuning. Terminal transitions do not bootstrap into the next reset.
Greedy evaluation does not modify Q-values, visits, epsilon or RNG state.

## Raw and terminal-SOC-adjusted costs

Let `p_ref` be the **2023 median price in EUR/kWh**, `S0` the initial stored
energy and `ST` the final stored energy. Define `delta = S0 - ST`:

```text
adjustment = p_ref * delta / eta_charge       when delta >= 0
adjustment = p_ref * delta * eta_discharge    when delta < 0
terminal_SOC_adjusted_cost = raw_grid_cost + adjustment
```

A shortfall is charged at the grid energy needed to replenish it. An excess
receives the value of useful future discharge into load. This asymmetric
convention includes conversion losses; credits are avoided future purchases,
not export revenue. Equal initial/final SOC gives zero adjustment. No terminal
charge/discharge is physically simulated, and no test-period price chooses the
valuation. The reference is fixed before seeing the new test-policy results.

The raw bill is still reported. Adjusted savings use the No-Battery raw bill as
reference because No Battery has zero inventory adjustment. Zero-reference
percentage conventions remain 0%. This is a reference-price proxy, not a cash
trade, optimum salvage value or battery-investment return. Depending on the
reference price and small terminal inventory relative to a full year's energy,
the adjustment can be small; its importance is methodological fairness.

The training reward remains `-(raw grid cost + throughput penalty)`. Terminal
valuation is deliberately a separate evaluation KPI, not a reward reshaping or
test-based training objective. Both raw-plus-penalty and adjusted-plus-penalty
costs are also exported to distinguish cash cost, cycling and inventory.

## Cycling and solar metrics

Throughput is bus-side charge input plus useful discharge output, in kWh.
`EFC = throughput / (2 * nominal_capacity_kwh)` and
`EFC_per_day = EFC / (number_of_hours * timestep_hours / 24)`.
Validation therefore uses 366 evaluation days, training/test 365. This is a
utilization/cycling proxy, not an electrochemical battery-health model.
The throughput penalty remains **0.01 EUR per kWh of throughput**.

PV surplus is `max(PV - demand, 0)` before battery dispatch. PV self-consumption
is `(direct PV + PV input to battery) / total PV`, before storage losses and
including PV left in final inventory. It is capture on site, not load-delivered
solar energy. Curtailed PV is uncaptured surplus. Peak import is kWh per hourly
timestep, numerically hourly average kW; subhourly peaks are not modeled.

Behavior reports distinguish requested actions from actual flows. A no-load
discharge request can be zero-flow even above minimum SOC; it is not attributed
to an SOC boundary. Boundary-caused zero flow and no-load-caused zero flow are
reported separately. PV-capture conclusions use actual PV-to-battery kWh, not
just charge requests during surplus hours.

## Reproduce this experiment

```bash
python -m src.data.generate_synthetic_energy_data
python -m src.evaluation.dataset_report
python -m unittest discover -s tests -v
python -m src.evaluation.compare_baselines
python -m src.training.train_q_learning
python -m src.evaluation.evaluate_q_learning
```

The main training command saves a fresh model and greedy 2024 validation.
The separate evaluation command loads that model, verifies input/model hashes,
and compares all three policies on 2025. Existing baseline outputs must come
from the same refined environment; they are reproduced and checked numerically.
Optional `--verify-reproducibility` repeats a full training run. This stage's
default run trains once; deterministic seeded training is tested with repeated
complete small episodes in the test suite, not claimed as a second full run.

## Verification and preservation

The **37 passing tests** comprise the 27 useful prior tests (adapted only for
changed data/dispatch/bin assumptions) plus 10 refinement tests. Coverage:

- Three-year contiguous chronology, complete year splits and leap day.
- Reproducible generation, independent annual variation, naturally occurring
  daytime surplus in every split, no NaN/infinity, no gap or duplicate timestamp.
- Dedicated surplus state and training-only price/net-load/valuation fitting.
- Demand-capped discharge, zero unused output, SOC/power bounds, efficiencies,
  charge priorities and bus/inventory energy conservation.
- Both terminal valuation branches, equal-SOC case, raw/adjusted separation,
  unchanged No-Battery valuation and EFC/day (including 366-day validation).
- Q-update arithmetic, terminal bootstrapping, frozen greedy evaluation,
  deterministic seeded training and exact readable model serialization.
- Future evaluation edits/truncation do not change past decisions or state.

Hourly accounting assertions run again on all validation/test strategies.
CSV reloads and all comparison metrics are checked after export.

Before changes, all **60 pre-refinement files** were inventoried with SHA-256
hashes and placed in `outputs/archive/pre_stage4_5_repository.zip`. Old results,
models and figures were relocated under `outputs/archive/pre_stage4_5/` so active
output folders cannot silently mix incompatible experiments. The archive manifest
is `outputs/archive/pre_stage4_5_manifest.json`. The EDA notebook now points to
the active dataset; its old outputs/execution counts were cleared as obsolete,
with its original version preserved in the snapshot. It has not been re-executed.

Seven focused figures are produced: demand/PV, cumulative raw cost, cumulative
SOC-adjusted cost, Q-Learning SOC, actions, actual PV-surplus capture, and training
reward. The example is the **first full Monday–Sunday week in June 2025**,
2–8 June, fixed by calendar before evaluating performance. Cumulative adjusted
cost adds the inventory correction at each hour-end to the cumulative raw bill;
it does not sum repeated inventory valuations. No additional baseline figures
are generated by the Stage-3 command to avoid duplicate visuals.

## Completed experiment and interpretation

The 200-episode training finished once. The existing completed model was reused
on continuation; no completed training was repeated. Both greedy evaluations
finished with epsilon zero and no Q updates. A final independent replay of the
saved model reproduced all validation/test comparison metrics and saved hourly
Q results; test baselines also matched their Stage-3 CSVs. This replay is
evaluation only. Seven exported figures were visually inspected.

**The Q-Learning agent did not learn to capture PV surplus in this experiment.**
It makes no charge requests in validation or test. It delivers just 38 kWh from
the initial 50 kWh inventory, finishes at 10 kWh and then has no stored energy
above its minimum. Test raw savings are EUR 3.994295; charging for the terminal
shortfall adds EUR 4.054526, leaving adjusted cost EUR 0.060232 above No Battery.
Validation similarly has adjusted cost about EUR 0.077 above No Battery.
This is effectively no operational benefit, not successful solar arbitrage.

Rule-Based has test adjusted savings of EUR 1,197.565846 (4.007702%), captures
511.022571 kWh of PV in storage and cycles at 0.802565 EFC/day. It also buys more
grid energy and raises peak hourly grid import to 115.833 kW versus 71.364 kW
without storage. Cost savings do not establish superiority on every KPI.
Only about 5.35% of available test PV surplus is captured even by Rule-Based;
its unchanged grid-charging behavior can leave little headroom for solar.

In high-price hours Q-Learning requests discharge 41.148965% of the time,
but only one such hour has positive discharge. Low-price charging frequency
is 0%. There are 574 surplus hours, zero charging actions during surplus,
zero PV-to-battery and zero grid-to-battery energy. There are 4,783 zero-flow
commands at the SOC boundary. Requests must not be confused with useful flow.
319 of 384 states were visited during training; three test hours use unseen
states. The reward trend during exploratory training is not evidence that
the final greedy policy is good. Coarse aggregation and finite training may
contribute, but their causal effect was not tested. No post-test tuning was done.

### Dataset statistics

All energy columns below are kWh; prices are EUR/kWh.

| split | year | rows | demand_kwh | pv_generation_kwh | pv_surplus_hours | pv_surplus_kwh | pv_direct_consumption_pct | price_min_eur_kwh | price_mean_eur_kwh | price_max_eur_kwh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training | 2023 | 8760 | 401,681.429000 | 118,879.196000 | 585 | 10,427.017000 | 91.228897 | -0.025000 | 0.098048 | 0.291080 |
| Validation | 2024 | 8784 | 402,567.072000 | 119,038.170000 | 594 | 9,618.471000 | 91.919843 | -0.025000 | 0.098179 | 0.313070 |
| Test | 2025 | 8760 | 401,571.621000 | 118,393.285000 | 574 | 9,558.661000 | 91.926349 | -0.025000 | 0.098106 | 0.291340 |

### Greedy validation — 2024

| Metric | No Battery | Rule-Based | Q-Learning |
| --- | --- | --- | --- |
| total_electricity_cost_eur | 29,929.206531 | 28,626.055349 | 29,925.229089 |
| cost_savings_eur | 0.000000 | 1,303.151182 | 3.977442 |
| cost_savings_pct | 0.000000 | 4.354112 | 0.013290 |
| total_grid_energy_purchased_kwh | 293,147.373000 | 295,754.375380 | 293,109.373000 |
| pv_self_consumption_pct | 91.919843 | 92.328726 | 91.919843 |
| battery_throughput_kwh | 0.000000 | 61,146.869169 | 38.000000 |
| equivalent_full_cycles | 0.000000 | 305.734346 | 0.190000 |
| peak_grid_import_kwh_per_hour | 74.455000 | 116.907000 | 74.455000 |
| initial_soc_kwh | 0.000000 | 50.000000 | 50.000000 |
| final_soc_kwh | 0.000000 | 10.000000 | 10.000000 |
| net_soc_change_kwh | 0.000000 | -40.000000 | -40.000000 |
| pv_to_battery_kwh | 0.000000 | 486.726789 | 0.000000 |
| grid_to_battery_kwh | 0.000000 | 31,633.572380 | 0.000000 |
| unused_discharge_kwh | 0.000000 | 0.000000 | 0.000000 |
| pv_curtailed_kwh | 9,618.471000 | 9,131.744211 | 9,618.471000 |
| battery_losses_kwh | 0.000000 | 3,133.729169 | 2.000000 |
| degradation_penalty_eur | 0.000000 | 611.468692 | 0.380000 |
| raw_electricity_cost_eur | 29,929.206531 | 28,626.055349 | 29,925.229089 |
| terminal_reference_price_eur_kwh | 0.096295 | 0.096295 | 0.096295 |
| terminal_soc_adjustment_eur | 0.000000 | 4.054526 | 4.054526 |
| terminal_soc_adjusted_cost_eur | 29,929.206531 | 28,630.109876 | 29,929.283616 |
| adjusted_cost_savings_eur | 0.000000 | 1,299.096655 | -0.077084 |
| adjusted_cost_savings_pct | 0.000000 | 4.340565 | -0.000258 |
| pv_generation_kwh | 119,038.170000 | 119,038.170000 | 119,038.170000 |
| pv_surplus_kwh | 9,618.471000 | 9,618.471000 | 9,618.471000 |
| pv_surplus_hours | 594.000000 | 594.000000 | 594.000000 |
| evaluation_days | 366.000000 | 366.000000 | 366.000000 |
| efc_per_day | 0.000000 | 0.835340 | 0.000519 |
| total_rl_reward_eur | -29,929.206531 | -29,237.524041 | -29,925.609089 |
| cost_plus_penalty_eur | 29,929.206531 | 29,237.524041 | 29,925.609089 |
| adjusted_cost_plus_penalty_eur | 29,929.206531 | 29,241.578567 | 29,929.663616 |
| cost_difference_vs_rule_based_eur | 1,303.151182 | 0.000000 | 1,299.173740 |

### Final test — 2025

| Metric | No Battery | Rule-Based | Q-Learning |
| --- | --- | --- | --- |
| total_electricity_cost_eur | 29,881.606847 | 28,679.986474 | 29,877.612552 |
| cost_savings_eur | 0.000000 | 1,201.620373 | 3.994295 |
| cost_savings_pct | 0.000000 | 4.021271 | 0.013367 |
| total_grid_energy_purchased_kwh | 292,736.997000 | 295,188.526504 | 292,698.997000 |
| pv_self_consumption_pct | 91.926349 | 92.357980 | 91.926349 |
| battery_throughput_kwh | 0.000000 | 58,587.234075 | 38.000000 |
| equivalent_full_cycles | 0.000000 | 292.936170 | 0.190000 |
| peak_grid_import_kwh_per_hour | 71.364000 | 115.833000 | 71.364000 |
| initial_soc_kwh | 0.000000 | 50.000000 | 50.000000 |
| final_soc_kwh | 0.000000 | 10.000000 | 10.000000 |
| net_soc_change_kwh | 0.000000 | -40.000000 | -40.000000 |
| pv_to_battery_kwh | 0.000000 | 511.022571 | 0.000000 |
| grid_to_battery_kwh | 0.000000 | 30,263.870504 | 0.000000 |
| unused_discharge_kwh | 0.000000 | 0.000000 | 0.000000 |
| pv_curtailed_kwh | 9,558.661000 | 9,047.638429 | 9,558.661000 |
| battery_losses_kwh | 0.000000 | 3,002.552075 | 2.000000 |
| degradation_penalty_eur | 0.000000 | 585.872341 | 0.380000 |
| raw_electricity_cost_eur | 29,881.606847 | 28,679.986474 | 29,877.612552 |
| terminal_reference_price_eur_kwh | 0.096295 | 0.096295 | 0.096295 |
| terminal_soc_adjustment_eur | 0.000000 | 4.054526 | 4.054526 |
| terminal_soc_adjusted_cost_eur | 29,881.606847 | 28,684.041001 | 29,881.667079 |
| adjusted_cost_savings_eur | 0.000000 | 1,197.565846 | -0.060232 |
| adjusted_cost_savings_pct | 0.000000 | 4.007702 | -0.000202 |
| pv_generation_kwh | 118,393.285000 | 118,393.285000 | 118,393.285000 |
| pv_surplus_kwh | 9,558.661000 | 9,558.661000 | 9,558.661000 |
| pv_surplus_hours | 574.000000 | 574.000000 | 574.000000 |
| evaluation_days | 365.000000 | 365.000000 | 365.000000 |
| efc_per_day | 0.000000 | 0.802565 | 0.000521 |
| total_rl_reward_eur | -29,881.606847 | -29,265.858815 | -29,877.992552 |
| cost_plus_penalty_eur | 29,881.606847 | 29,265.858815 | 29,877.992552 |
| adjusted_cost_plus_penalty_eur | 29,881.606847 | 29,269.913341 | 29,882.047079 |
| cost_difference_vs_rule_based_eur | 1,201.620373 | 0.000000 | 1,197.626078 |

### Test action frequencies by training price bin

| price_bin | price_label | hours | charge_hours | hold_hours | discharge_hours | active_charge_hours | active_discharge_hours | non_hold_zero_flow_hours | charge_pct | discharge_pct | charge_from_bus_kwh | discharge_to_load_kwh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | low | 2953 | 0 | 1494 | 1459 | 0 | 0 | 1459 | 0.000000 | 49.407382 | 0.000000 | 0.000000 |
| 1 | medium | 2813 | 0 | 719 | 2094 | 0 | 1 | 2093 | 0.000000 | 74.440100 | 0.000000 | 9.018000 |
| 2 | high | 2994 | 0 | 1762 | 1232 | 0 | 1 | 1231 | 0.000000 | 41.148965 | 0.000000 | 28.982000 |

### Training trace

| episode | cumulative_reward | electricity_cost_eur | grid_import_kwh | battery_throughput_kwh | epsilon | final_soc_kwh |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | -31,803.923465 | 30,357.771844 | 299,486.769745 | 144,615.162078 | 1.000000 | 10.000000 |
| 200 | -30,207.953220 | 29,725.391920 | 293,906.164645 | 48,256.129958 | 0.050000 | 10.000000 |

Final 10-episode mean reward: -30,524.123109 EUR.
Training rewards include exploration and the throughput penalty.

## Limitations and stopping point

Full-season splits and genuine surplus make the evaluation more informative,
but this is still synthetic data, coarse state aggregation, a repeated single
training year and one declared hyperparameter configuration/seed. No claim of
optimality or reliable real-site performance follows. Raw savings, adjusted
savings, PV capture, grid import, cycling and peak demand can disagree. Findings
must be reported across those metrics rather than from a single ranking.

Stage 4.5 ends with the refined common environment and rerun baselines/Q-Learning.
DQN is not implemented. No Git repository was initialized or pushed.
