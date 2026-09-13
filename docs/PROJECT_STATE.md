# Project continuation checkpoint

- Status: Stage 1 Dataset ✅; Stage 2 EDA/Environment ✅; Stage 3 Baselines ✅; Stage 4 Q-Learning ✅; Stage 4.5 Refinement ✅; Stage 5 DQN ✅; Stage 6 Robustness/Portfolio ✅.
- Goal/architecture: synthetic commercial demand+100 kWp PV+prices → unchanged 100 kWh battery environment → No Battery, Rule-Based, 384-state Q-Learning, Vanilla DQN → common cost/PV/grid/cycling/peak evaluation.
- Split: 2023 train and fit; 2024 validation; 2025 frozen test. No shuffle, test tuning, environment/reward/benchmark changes or new algorithms.
- Battery/reward: SOC 10–95 kWh, initial 50, 50 kW, eta 0.95, one-hour steps, no export, demand-capped discharge, PV-first charge then grid. Reward `-(grid cost + 0.01 EUR/kWh throughput)`; terminal inventory valued separately at 2023 median price.
- DQN: continuous normalized [demand, PV, price, SOC fraction, hour sin/cos, weekend], 7→64→64→3, ReLU, uniform replay50k/warmup2k/batch64, Adam.001, gamma.95, update/4 steps, target/1000, epsilon1→.05 over40 episodes, CPU. 100 episodes/876,000 steps per seed.
- Authoritative seed-42 test adjusted cost/savings: No Battery EUR 29,881.61/0%; Rule-Based EUR 28,684.04/4.008%; Q-Learning EUR 29,881.67/-0.0002%; DQN EUR 28,250.40/5.459% (EUR 1,631.20).
- Seed-42 DQN behavior: PV self-use 95.50%, surplus capture 44.24%, EFC/day 1.514, peak 116.395 kW, SOC 50→10. Lowest adjusted bill, but higher cycling/peak; cost+throughput penalty remains worse than Rule-Based.
- Robustness seeds 42/7/123: adjusted savings mean 4.580%, sample SD 0.828 pp, range 3.816–5.459%; mean surplus capture 32.80%; mean EFC/day 1.084. All seeds reported; no selection. Lightweight sensitivity only.
- Tests: 59/59 passed after Stage 6. Frozen model reloads reproduce inference; robustness configuration differs only by seed; aggregation uses sample SD.
- Outputs: authoritative `final_model_comparison.csv`, `dqn_seed_robustness.csv`, seven curated files in `outputs/portfolio_figures/`, final business interpretation and recruiter README. Prior scientific outputs retained; pre-Stage-6 active manifest records 100 files.
- Limitations: synthetic single facility, simplified aging, no demand charge/export/capex/forecast uncertainty, current-hour information, discrete full-flow actions, one DQN configuration and three seeds. Terminal valuation/EFC are proxies.
- Repository is ready for GitHub presentation. No Git init/add/commit/push occurred. Project development is complete unless a genuine issue or explicit new task is provided.
