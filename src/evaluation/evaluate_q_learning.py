"""Frozen greedy evaluation and Stage-3-compatible three-strategy comparison.

Run after training: python -m src.evaluation.evaluate_q_learning
This command loads the selected model, never updates it, and evaluates test.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.q_learning_agent import ENV_ACTIONS, QLearningAgent, QLearningConfig
from src.agents.state_discretizer import STATE_COLUMNS, StateDiscretizer
from src.agents.rule_based_controller import RuleBasedController, run_rule_based_controller
from src.environment.energy_storage_env import EnergyStorageEnv
from .data_utils import chronological_split, load_battery_config, validate_battery_config, validate_hourly_data
from .metrics import validate_results
from .terminal_valuation import TerminalValuation, calculate_refined_metrics
from .no_battery_baseline import run_no_battery_baseline


ROOT = Path(__file__).resolve().parents[2]


def evaluate_agent(data, agent, discretizer, config, penalty):
    data = validate_hourly_data(data)
    validate_battery_config(config)
    if data.timestamp.iloc[0] <= pd.Timestamp(discretizer.training_end):
        raise ValueError("Evaluation must be later than discretizer training.")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Invalid throughput penalty.")
    q_before, visits_before = agent.q_table.copy(), agent.visits.copy()
    rng_before, epsilon_before = deepcopy(agent.rng.bit_generator.state), agent.epsilon
    env = EnergyStorageEnv(data, config, penalty)
    env.reset()
    rows = []
    for row in data.itertuples(index=False):
        soc_before = env.battery.soc_kwh
        state = discretizer.encode(row.electricity_price_eur_kwh,
            row.demand_kwh - row.pv_generation_kwh, soc_before / config.capacity_kwh,
            row.hour, row.is_weekend)
        index = agent.select_action(state, explore=False)  # effective epsilon = 0
        _, _, _, info = env.step(ENV_ACTIONS[index])
        charge = info["pv_to_battery_kwh"] + info["grid_to_battery_kwh"]
        discharge = info["battery_throughput_kwh"] - charge
        info.update({
            "electricity_price_eur_kwh": row.electricity_price_eur_kwh,
            "action_index": index, "soc_before_kwh": soc_before,
            "charge_from_bus_kwh": charge, "discharge_to_bus_kwh": discharge,
            "unused_discharge_kwh": discharge - info["discharge_to_load_kwh"],
            "pv_curtailed_kwh": row.pv_generation_kwh - info["pv_used_directly_kwh"] - info["pv_to_battery_kwh"],
            "battery_losses_kwh": charge * (1 - config.charge_efficiency) + discharge * (1 / config.discharge_efficiency - 1),
            "state_seen_in_training": bool(agent.visits[state].sum() > 0),
            "chosen_action_seen_in_training": bool(agent.visits[(*state, index)] > 0),
            **dict(zip(STATE_COLUMNS, state)),
        })
        rows.append(info)
    np.testing.assert_array_equal(agent.q_table, q_before)
    np.testing.assert_array_equal(agent.visits, visits_before)
    if agent.rng.bit_generator.state != rng_before or agent.epsilon != epsilon_before:
        raise AssertionError("Evaluation mutated agent state.")
    result = pd.DataFrame(rows)
    validate_results(result, config, True, penalty)
    return result


def compare_period(training, period, agent, discretizer, config, penalty):
    frames = {
        "No Battery": run_no_battery_baseline(period),
        "Rule-Based": run_rule_based_controller(period, RuleBasedController.fit(training), config, penalty),
        "Q-Learning": evaluate_agent(period, agent, discretizer, config, penalty),
    }
    baseline_cost = float(frames["No Battery"].grid_cost_eur.sum())
    rule_cost = float(frames["Rule-Based"].grid_cost_eur.sum())
    valuation = TerminalValuation.fit(training)
    rows = []
    for name, frame in frames.items():
        for col in ("timestamp", "demand_kwh", "pv_generation_kwh", "electricity_price_eur_kwh"):
            pd.testing.assert_series_equal(frame[col], frames["No Battery"][col])
        validate_results(frame, config, name != "No Battery", penalty)
        metrics = calculate_refined_metrics(frame, baseline_cost, config, valuation)
        rows.append({"Strategy": name, **metrics,
            "cost_difference_vs_rule_based_eur": metrics["total_electricity_cost_eur"] - rule_cost,
            "total_rl_reward_eur": float(frame.reward.sum()),
            "cost_plus_penalty_eur": float(frame.grid_cost_eur.sum() + frame.degradation_penalty_eur.sum()),
        })
    return frames, pd.DataFrame(rows)


def behaviour_summary(result):
    """Conditional action frequencies, not causal claims about learned intent."""
    rows = []
    for price_bin, label in enumerate(("low", "medium", "high")):
        sample = result[result.price_bin == price_bin]
        count = len(sample)
        rows.append({"price_bin": price_bin, "price_label": label, "hours": count,
            "charge_hours": int((sample.action == 1).sum()),
            "hold_hours": int((sample.action == 0).sum()),
            "discharge_hours": int((sample.action == -1).sum()),
            "active_charge_hours": int((sample.charge_from_bus_kwh > 1e-8).sum()),
            "active_discharge_hours": int((sample.discharge_to_bus_kwh > 1e-8).sum()),
            "non_hold_zero_flow_hours": int(((sample.action != 0) &
                (sample.battery_throughput_kwh <= 1e-8)).sum()),
            "charge_pct": float((sample.action == 1).mean() * 100) if count else 0.,
            "discharge_pct": float((sample.action == -1).mean() * 100) if count else 0.,
            "charge_from_bus_kwh": float(sample.charge_from_bus_kwh.sum()),
            "discharge_to_load_kwh": float(sample.discharge_to_load_kwh.sum()),
        })
    return pd.DataFrame(rows)


def surplus_behaviour(result, config):
    """Distinguish boundary-limited commands from no-load discharge commands."""
    surplus = (result.pv_generation_kwh - result.demand_kwh).clip(lower=0)
    zero = result.battery_throughput_kwh <= 1e-8
    boundary = zero & (((result.action == 1) & (result.soc_before_kwh >= config.max_soc_kwh - 1e-8)) |
                       ((result.action == -1) & (result.soc_before_kwh <= config.min_soc_kwh + 1e-8)))
    no_load = zero & (result.action == -1) & (result.demand_kwh <= result.pv_generation_kwh) & ~boundary
    return {
        "pv_surplus_hours": int((surplus > 0).sum()),
        "charge_actions_during_surplus": int(((surplus > 0) & (result.action == 1)).sum()),
        "pv_surplus_kwh": float(surplus.sum()),
        "pv_to_battery_kwh": float(result.pv_to_battery_kwh.sum()),
        "surplus_captured_pct": float(result.pv_to_battery_kwh.sum() / surplus.sum() * 100) if surplus.sum() else 0.,
        "grid_to_battery_kwh": float(result.grid_to_battery_kwh.sum()),
        "discharge_to_load_kwh": float(result.discharge_to_load_kwh.sum()),
        "zero_flow_soc_boundary_commands": int(boundary.sum()),
        "zero_flow_no_load_commands_excluding_boundary": int(no_load.sum()),
        "all_zero_flow_non_hold_commands": int((zero & (result.action != 0)).sum()),
    }


def main():
    models, results = ROOT / "outputs/models", ROOT / "outputs/results"
    metadata = json.loads((models / "q_learning_config.json").read_text(encoding="utf-8"))
    for name, key in (("q_table.csv", "q_table_sha256"), ("state_discretizer.json", "discretizer_sha256")):
        if hashlib.sha256((models / name).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f"Selected model artifact changed: {name}")
    dataset = ROOT / "data/processed/energy_hourly_2023_2025.csv"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Dataset changed since model training.")
    for path, key in (("src/environment/energy_storage_env.py", "environment_sha256"),
                      ("src/environment/battery_model.py", "battery_model_sha256")):
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError("Simulation changed since model training.")
    train, _, test = chronological_split(pd.read_csv(dataset, parse_dates=["timestamp"]))
    config, penalty = load_battery_config(ROOT / "config/battery_config.json")
    if asdict(config) != metadata["battery_config"] or penalty != metadata["throughput_penalty_eur_per_kwh"]:
        raise ValueError("Battery configuration changed since training.")
    agent = QLearningAgent.load(models / "q_table.csv", QLearningConfig(**metadata["hyperparameters"]))
    discretizer = StateDiscretizer.load(models / "state_discretizer.json")
    if discretizer != StateDiscretizer.fit(train, config):
        raise AssertionError("Discretizer differs from training-only fit.")
    frames, comparison = compare_period(train, test, agent, discretizer, config, penalty)
    frames["Q-Learning"].to_csv(results / "q_learning_test_results.csv", index=False)
    comparison.to_csv(results / "q_learning_comparison.csv", index=False)
    behaviour_summary(frames["Q-Learning"]).to_csv(results / "q_learning_behaviour.csv", index=False)
    # Recomputed baselines must match the refined Stage-3 artifacts.
    for name, filename in (("No Battery", "no_battery_test_results.csv"),
                           ("Rule-Based", "rule_based_test_results.csv")):
        old = pd.read_csv(results / filename, parse_dates=["timestamp"])
        pd.testing.assert_frame_equal(old, frames[name], check_dtype=False, atol=1e-8, rtol=1e-10)
    from .q_learning_plots import make_q_learning_plots
    history = pd.read_csv(results / "q_learning_training_history.csv")
    valuation = TerminalValuation.fit(train)
    make_q_learning_plots(history, frames, config, ROOT / "outputs/figures", valuation)
    q = frames["Q-Learning"]
    audit = {
        "test_start": str(test.timestamp.iloc[0]), "test_end": str(test.timestamp.iloc[-1]),
        "test_rows": len(test), "effective_epsilon": 0, "q_updates_during_evaluation": 0,
        "q_table_sha256": metadata["q_table_sha256"],
        "training_states_visited": int((agent.visits.sum(axis=-1) > 0).sum()), "possible_states": int(np.prod(agent.q_table.shape[:-1])),
        "unseen_test_state_hours": int((~q.state_seen_in_training).sum()),
        "unseen_chosen_action_hours": int((~q.chosen_action_seen_in_training).sum()),
        "baseline_reproduction": "identical to refined Stage-3 results within 1e-8",
        "soc_treatment": "Common initial SOC; raw cost and separate training-price terminal valuation",
        "test_policy_tuned": False,
    }
    audit["terminal_reference_price_eur_kwh"] = valuation.reference_price_eur_kwh
    audit["pv_surplus_behaviour"] = surplus_behaviour(q, config)
    (results / "stage4_evaluation_metadata.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
