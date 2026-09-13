# Stage 3 — Baselines on the Stage 4.5 environment

The pre-refinement results are archived and are not numerically comparable.
Run from the repository root after generating the active three-year dataset:

```bash
python -m src.evaluation.compare_baselines
```

The No-Battery baseline directly consumes PV, imports residual demand, curtails
surplus and has no battery inventory or cycling. Rule-Based retains its original
rules: charge when there is PV surplus and headroom, otherwise charge below
training P25; discharge above training P75 when residual load and stored energy
allow; otherwise hold. Thresholds are fitted only on full-year 2023:
P25 = 0.08281750 and P75 = 0.11045000 EUR/kWh.

Charge uses surplus PV first and fills the remaining requested input from grid.
Discharge is now capped by residual demand before SOC changes. No export is
allowed and unused discharge is zero. Each battery run starts at 50 kWh, with
100 kWh capacity, 10–95 kWh SOC bounds, 50 kW charge/discharge limits and 0.95
efficiency in either direction. The throughput penalty remains 0.01 EUR/kWh.

The Stage-3 command saves test baselines for 2025. The Stage-4 training command
also compares all three strategies on 2024 validation using the same rules.
Both raw bill and terminal-SOC-adjusted cost are retained. A 40 kWh inventory
shortfall incurs 40 / 0.95 * 0.096295 = 4.054526 EUR, with the reference price
fixed at the training median. This is a valuation, not a simulated grid trade.

## Current 2025 results

| Strategy | raw_electricity_cost_eur | terminal_soc_adjusted_cost_eur | adjusted_cost_savings_pct | pv_self_consumption_pct | efc_per_day |
| --- | --- | --- | --- | --- | --- |
| No Battery | 29,881.606847 | 29,881.606847 | 0.000000 | 91.926349 | 0.000000 |
| Rule-Based | 28,679.986474 | 28,684.041001 | 4.007702 | 92.357980 | 0.802565 |

Rule-Based saves money but increases purchased grid energy, cycling and peak
import. EFC/day is throughput / (2 * 100 kWh) / 365 days; it is a utilization
proxy, not battery-life prediction. PV self-consumption counts direct use plus
PV input into storage before losses. Initial/final SOC is 50/10 kWh for
Rule-Based. See [Stage 4.5](stage_4_5_refinement.md) for all metrics, exact
accounting, validation results, seven shared figures and 37 passing tests.

Active CSVs are in `outputs/results/`; previous results and original files are
preserved under `outputs/archive/`. Baseline results are reproduced by the frozen
Q-Learning evaluation and checked numerically within 1e-8.
