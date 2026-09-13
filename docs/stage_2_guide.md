# Stage 2 — EDA & Battery Environment

## Stage 2A — EDA
Run `notebooks/01_energy_data_eda.ipynb`.

Understand demand, PV, electricity-price and net-load patterns.

## Stage 2B — Battery Model
File: `src/environment/battery_model.py`

It handles battery capacity, SOC limits, charge/discharge limits and efficiencies.

Actions:
- `-1` = discharge
- `0` = hold
- `+1` = charge

## Stage 2C — Energy Environment
File: `src/environment/energy_storage_env.py`

State:
`[demand, PV, price, SOC, hour_sin, hour_cos, weekend]`

Reward:
`reward = -(grid cost + battery-throughput penalty)`

## Run the demo
From the project root:

```bash
python -m src.environment.demo_environment
```

## Understand before Stage 3
1. What SOC means.
2. Why SOC changes after every action.
3. Why battery efficiency is below 100%.
4. How grid import is calculated.
5. Why reward is negative cost.
6. Why charge / hold / discharge are actions.

Next: Stage 3 = No-Battery baseline + Rule-Based controller.
