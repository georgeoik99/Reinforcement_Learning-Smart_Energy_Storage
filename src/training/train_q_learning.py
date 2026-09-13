"""Train on the calendar year 2023, then greedy validation. Never evaluate test here.

Run: python -m src.training.train_q_learning [--episodes 200] [--seed 42]
Use --verify-reproducibility to repeat the complete training run with the seed.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.agents.q_learning_agent import ENV_ACTIONS, QLearningAgent, QLearningConfig
from src.agents.state_discretizer import StateDiscretizer
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import chronological_split, load_battery_config, validate_hourly_data
from src.evaluation.terminal_valuation import TerminalValuation


ROOT = Path(__file__).resolve().parents[2]


def train_agent(training_data, battery_config, penalty, settings, progress=True):
    """Each episode resets the common environment once; terminal target=r.

    Q updates use only (current observation, action, observed reward, next
    observation). The terminal transition never bootstraps into another episode.
    Fixed final episode is selected; no checkpoint/test-performance selection.
    """
    data = validate_hourly_data(training_data)
    discretizer = StateDiscretizer.fit(data, battery_config)
    agent = QLearningAgent(settings)
    env = EnergyStorageEnv(data, battery_config, penalty)
    observations = [
        (row.electricity_price_eur_kwh, row.demand_kwh - row.pv_generation_kwh,
         row.hour, row.is_weekend) for row in data.itertuples(index=False)
    ]

    def encode(index):
        price, net, hour, weekend = observations[index]
        return discretizer.encode(price, net, env.battery.soc_kwh / battery_config.capacity_kwh,
                                  hour, weekend)

    history = []
    for episode in range(1, settings.episodes + 1):
        env.reset()
        state = encode(0)
        totals = dict(cumulative_reward=0., electricity_cost_eur=0., grid_import_kwh=0.,
                      battery_throughput_kwh=0.)
        epsilon = agent.epsilon
        for index in range(len(data)):
            action = agent.select_action(state)
            _, reward, done, info = env.step(ENV_ACTIONS[action])
            next_state = None if done else encode(index + 1)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            totals["cumulative_reward"] += reward
            totals["electricity_cost_eur"] += info["grid_cost_eur"]
            totals["grid_import_kwh"] += info["grid_import_kwh"]
            totals["battery_throughput_kwh"] += info["battery_throughput_kwh"]
        history.append({"episode": episode, **totals, "epsilon": epsilon,
                        "final_soc_kwh": env.battery.soc_kwh})
        agent.decay_epsilon()
        if progress and (episode == 1 or episode % 10 == 0 or episode == settings.episodes):
            print(f"Episode {episode}/{settings.episodes}: reward={totals['cumulative_reward']:.2f}, "
                  f"cost={totals['electricity_cost_eur']:.2f}, epsilon={epsilon:.4f}", flush=True)
    history = pd.DataFrame(history)
    if not np.isfinite(history.to_numpy()).all():
        raise AssertionError("Nonfinite training history.")
    np.testing.assert_allclose(history.cumulative_reward,
        -(history.electricity_cost_eur + penalty * history.battery_throughput_kwh), atol=1e-7)
    return agent, discretizer, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default, kind in (("episodes", 200, int), ("alpha", .1, float),
            ("gamma", .95, float), ("epsilon-start", 1., float),
            ("epsilon-decay", .98, float), ("epsilon-min", .05, float), ("seed", 42, int)):
        parser.add_argument("--" + name, default=default, type=kind)
    parser.add_argument("--verify-reproducibility", action="store_true")
    args = parser.parse_args()
    settings = QLearningConfig(**{k: v for k, v in vars(args).items() if k != "verify_reproducibility"})
    path = ROOT / "data/processed/energy_hourly_2023_2025.csv"
    train, validation, _ = chronological_split(pd.read_csv(path, parse_dates=["timestamp"]))
    battery, penalty = load_battery_config(ROOT / "config/battery_config.json")
    agent, discretizer, history = train_agent(train, battery, penalty, settings)
    reproduced = False
    if args.verify_reproducibility:
        print("Repeating full training with identical seed; no validation/test fitting.", flush=True)
        repeat, repeat_disc, repeat_history = train_agent(train, battery, penalty, settings)
        np.testing.assert_array_equal(agent.q_table, repeat.q_table)
        np.testing.assert_array_equal(agent.visits, repeat.visits)
        pd.testing.assert_frame_equal(history, repeat_history, check_exact=True)
        assert discretizer == repeat_disc
        reproduced = True
    from src.evaluation.evaluate_q_learning import compare_period
    frames, comparison = compare_period(train, validation, agent, discretizer, battery, penalty)
    models, results = ROOT / "outputs/models", ROOT / "outputs/results"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    agent.save(models / "q_table.csv")
    discretizer.save(models / "state_discretizer.json")
    # Verify human-readable serialization preserves the policy exactly.
    restored = QLearningAgent.load(models / "q_table.csv", settings)
    np.testing.assert_array_equal(agent.q_table, restored.q_table)
    history.to_csv(results / "q_learning_training_history.csv", index=False)
    frames["Q-Learning"].to_csv(results / "q_learning_validation_results.csv", index=False)
    comparison.to_csv(results / "q_learning_validation_comparison.csv", index=False)
    metadata = {
        "simulation_version": "stage_4_5",
        "split_rule": "2023=training; 2024=validation; 2025=test",
        "terminal_reference_price_eur_kwh": TerminalValuation.fit(train).reference_price_eur_kwh,
        "terminal_valuation_in_training_reward": False,
        "hyperparameters": asdict(settings), "battery_config": asdict(battery),
        "throughput_penalty_eur_per_kwh": penalty,
        "training_rows": len(train), "validation_rows": len(validation),
        "training_start": str(train.timestamp.iloc[0]), "training_end": str(train.timestamp.iloc[-1]),
        "validation_start": str(validation.timestamp.iloc[0]), "validation_end": str(validation.timestamp.iloc[-1]),
        "model_selection": "Single predeclared configuration; final episode; no tuning or checkpoint selection",
        "test_used_for_training_or_selection": False,
        "full_training_reproducibility_verified": reproduced,
        "q_table_sha256": hashlib.sha256((models / "q_table.csv").read_bytes()).hexdigest(),
        "discretizer_sha256": hashlib.sha256((models / "state_discretizer.json").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256((ROOT / "src/environment/energy_storage_env.py").read_bytes()).hexdigest(),
        "battery_model_sha256": hashlib.sha256((ROOT / "src/environment/battery_model.py").read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (models / "q_learning_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Greedy validation (no Q updates):", flush=True)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"Saved selected model. Full-seed reproducibility: {reproduced}", flush=True)


if __name__ == "__main__":
    main()
