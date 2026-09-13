# Data Dictionary

## Active dataset: `energy_hourly_2023_2025.csv`

| Column | Unit | Description |
|---|---|---|
| `timestamp` | naive simulation hourly timestamp | Observation time |
| `demand_kwh` | kWh | Commercial facility energy demand during the hour |
| `pv_generation_kwh` | kWh | Rooftop PV energy produced during the hour |
| `electricity_price_eur_mwh` | EUR/MWh | Dynamic wholesale-style electricity price |
| `electricity_price_eur_kwh` | EUR/kWh | Price converted for direct cost calculation |
| `net_load_before_battery_kwh` | kWh | Demand minus PV generation before battery dispatch |
| `hour` | 0-23 | Hour of day |
| `day_of_week` | 0-6 | Monday=0, Sunday=6 |
| `is_weekend` | 0/1 | Weekend indicator |
| `month` | 1-12 | Calendar month |
| `day_of_year` | 1-365/366 | Calendar day number |

## Important modeling note

`battery_soc` is deliberately absent from the dataset.

State of charge depends on previous battery actions and therefore belongs to the
reinforcement-learning environment state, not the exogenous input dataset.

## Data provenance

The Stage 1 dataset is realistic synthetic data generated with deterministic
seasonality, commercial operating-hour effects, PV daylight/seasonality effects,
price time-of-day effects, random volatility and occasional price spikes.

It is not actual customer, HEnEx or PVGIS data.


## Stage 4.5 calendar and PV refinement

The active generator now creates 26,304 contiguous hours: 2023 training (8,760),
2024 validation (8,784, including leap day), and 2025 test (8,760). Timestamps
are naive simulation labels without DST discontinuities. Demand and dynamic
price concepts are retained; rooftop PV assumes 100 kWp. Seed 42 and independent
`SeedSequence([seed, year])` streams give reproducible annual variation.
Surplus emerges from generated load and PV, without editing observations.
All learned thresholds and the terminal inventory reference price use 2023 only.

The retained `energy_hourly_2025.csv` is obsolete legacy data and is not the
2025 slice of the active dataset. The EDA notebook points to the active file;
obsolete execution outputs were cleared and the notebook was not re-executed.
See [the refinement report](stage_4_5_refinement.md) for annual statistics,
validation rules, synthetic-data limitations and exact evaluation formulas.
