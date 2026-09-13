# Final Results and Business Interpretation

## Decision summary

The seed-42 DQN has the lowest terminal-SOC-adjusted electricity cost in the
authoritative 2025 comparison and captures substantially more PV surplus than
the other battery controllers. Its savings require much more cycling and a
higher grid-import peak. Across three fixed-configuration seeds, adjusted
savings range from 3.816% to 5.459%. This supports a
lightweight robustness claim, not statistical proof or a deployment decision.

| Strategy | Adjusted cost (€) | Savings (€) | Savings (%) | PV self-use (%) | PV surplus captured (%) | EFC/day | Peak import (kW) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Battery | 29,881.607 | 0.000 | 0.000 | 91.926 | 0.000 | 0.000 | 71.364 |
| Rule-Based | 28,684.041 | 1,197.566 | 4.008 | 92.358 | 5.346 | 0.803 | 115.833 |
| Q-Learning | 29,881.667 | -0.060 | -0.000 | 91.926 | 0.000 | 0.001 | 71.364 |
| DQN | 28,250.404 | 1,631.203 | 5.459 | 95.498 | 44.238 | 1.514 | 116.395 |

## Economic Performance

No Battery costs EUR 29,881.61 and
sets the zero-savings reference. Rule-Based saves EUR
1,197.57 (4.008%).
Tabular Q-Learning is EUR 0.06 worse than
No Battery after valuing its terminal SOC; its coarse 384-state representation
failed to learn useful charging behavior. Seed-42 DQN saves EUR
1,631.20 (5.459%), with
adjusted cost EUR 433.64
below Rule-Based and EUR
1,631.26
below Q-Learning.

The cost ranking does not include capital expenditure, demand charges or a
detailed degradation model. The existing throughput penalty is reported
separately: DQN raw cost plus penalty is EUR
29,351.48, versus EUR
29,265.86 for
Rule-Based. On that simplified reward-related measure, Rule-Based is better.

## Renewable Utilization

No Battery directly consumes 91.93% of PV.
Rule-Based raises self-consumption only to 92.36%
and captures 5.35% of available surplus. Seed-42
DQN reaches 95.50% PV self-consumption and stores
4,228.55 kWh, or 44.24% of
available surplus. Positive PV-to-battery flow supports the conclusion that the
DQN learned solar-storage behavior in the synthetic test.

## Battery Utilization

Rule-Based performs 0.803 equivalent full cycles per day.
Seed-42 DQN performs 1.514 EFC/day, almost twice as much. Higher
cycling can shorten battery life and increase replacement cost. This project
uses EFC and a simple throughput penalty as proxies; it does not implement an
electrochemical degradation, temperature or warranty model. The DQN bill
savings must therefore be treated as gross simulated operating savings.

## Peak Demand

Peak hourly import is 71.36 kW with
No Battery, 115.83 kW with Rule-Based and
116.39 kW with seed-42 DQN. Full-rate grid
charging can coincide with facility demand and raise the site peak. A tariff
with demand charges could reduce or reverse the DQN advantage, but demand
charges are outside this model.

## AI Complexity vs Business Value

Q-Learning is more complex than static rules yet provides essentially no useful
operational value here: it does not learn charging or PV capture. DQN uses
continuous normalized inputs and neural-network function approximation, and it
delivers measurable cost reduction and renewable utilization. It also operates
the battery more intensely, creates many boundary-limited commands and raises
peak exposure. More advanced AI is valuable only when the improvement justifies
the operational trade-offs.

## Lightweight multi-seed robustness

Seed 42 is retained for the historical main comparison. Seeds 7 and 123 use the
same 7–64–64–3 network, training schedule, data split, reward and normalization;
only the random seed changes. No seed is selected or hidden.

| seed | Adjusted cost (€) | Savings (€) | Savings (%) | PV self-use (%) | Surplus captured (%) | EFC/day | Peak (kW) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.000 | 28,250.404 | 1,631.203 | 5.459 | 95.498 | 44.238 | 1.514 | 116.395 |
| 7.000 | 28,741.318 | 1,140.289 | 3.816 | 94.186 | 27.989 | 0.748 | 121.027 |
| 123.000 | 28,547.788 | 1,333.819 | 4.464 | 94.040 | 26.183 | 0.989 | 117.059 |

Summary statistics use sample standard deviation (`ddof=1`) across three seeds.

| Statistic | Adjusted cost (€) | Savings (€) | Savings (%) | Surplus captured (%) | EFC/day | Peak (kW) |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 28,513.170 | 1,368.437 | 4.580 | 32.803 | 1.084 | 118.160 |
| std | 247.281 | 247.281 | 0.828 | 9.944 | 0.392 | 2.505 |
| min | 28,250.404 | 1,140.289 | 3.816 | 26.183 | 0.748 | 116.395 |
| max | 28,741.318 | 1,631.203 | 5.459 | 44.238 | 1.514 | 121.027 |

Mean adjusted savings are 4.580% with a 0.828
percentage-point sample standard deviation. Mean PV-surplus capture is
32.80% and mean cycling is 1.084 EFC/day. Three seeds are
too few for statistical confidence, but they expose sensitivity that a single
run cannot show.
