"""Stage-3 checks: python -m unittest discover -s tests -v.

Analytical cases, preserved-environment replay, physical constraints and causal
perturbation checks. No optimizer or performance target is used in these tests.
"""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.agents.rule_based_controller import RuleBasedController, run_rule_based_controller
from src.environment.battery_model import BatteryConfig, BatteryModel
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.compare_baselines import evaluate_baselines, select_example_week
from src.evaluation.data_utils import chronological_split, load_battery_config, validate_hourly_data
from src.evaluation.metrics import calculate_metrics, validate_results
from src.data.generate_synthetic_energy_data import generate_energy_dataset
from src.evaluation.no_battery_baseline import run_no_battery_baseline


ROOT = Path(__file__).resolve().parents[1]


def hourly(demand, pv, price, start="2025-01-01"):
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(demand), freq="h"),
        "demand_kwh": demand, "pv_generation_kwh": pv,
        "electricity_price_eur_kwh": price,
    })


def fixed_controller():
    return RuleBasedController.fit(hourly([1] * 4, [0] * 4, [.1, .1, .3, .3], "2024-01-01"))


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        self.config = BatteryConfig()
        self.controller = fixed_controller()

    def test_no_battery_hand_calculated(self):
        data = hourly([10, 10, 0], [4, 20, 0], [.2, .1, -.1])
        result = run_no_battery_baseline(data)
        np.testing.assert_allclose(result.pv_used_directly_kwh, [4, 10, 0])
        np.testing.assert_allclose(result.grid_import_kwh, [6, 0, 0])
        np.testing.assert_allclose(result.grid_cost_eur, [1.2, 0, 0])
        np.testing.assert_allclose(result.pv_curtailed_kwh, [0, 10, 0])
        validate_results(result, self.config, False)
        metrics = calculate_metrics(result, 1.2, 100)
        self.assertAlmostEqual(metrics["pv_self_consumption_pct"], 14 / 24 * 100)
        self.assertAlmostEqual(metrics["total_electricity_cost_eur"], 1.2)
        self.assertEqual(metrics["equivalent_full_cycles"], 0)
        self.assertEqual(metrics["peak_grid_import_kwh_per_hour"], 6)

    def test_zero_denominators_are_finite_zero(self):
        result = run_no_battery_baseline(hourly([0, 0], [0, 0], [.1, .2]))
        metrics = calculate_metrics(result, 0, 100)
        self.assertEqual(metrics["cost_savings_pct"], 0)
        self.assertEqual(metrics["pv_self_consumption_pct"], 0)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        with self.assertRaises(ValueError):
            calculate_metrics(result, 0, 0)

    def test_negative_prices_and_no_export_revenue(self):
        result = run_no_battery_baseline(hourly([2, 0], [0, 10], [-.1, .5]))
        np.testing.assert_allclose(result.grid_cost_eur, [-.2, 0])
        validate_results(result, self.config, False)

    def test_calendar_split_and_boundaries(self):
        training, validation, test = chronological_split(generate_energy_dataset())
        self.assertEqual([len(training), len(validation), len(test)], [8760, 8784, 8760])
        self.assertEqual(training.timestamp.iloc[-1], pd.Timestamp("2023-12-31 23:00"))
        self.assertEqual(validation.timestamp.iloc[0], pd.Timestamp("2024-01-01"))
        self.assertEqual(test.timestamp.iloc[0], pd.Timestamp("2025-01-01"))
        with self.assertRaises(ValueError):
            chronological_split(training)

    def test_policy_priority_threshold_equalities_and_soc_limits(self):
        battery = BatteryModel(self.config)
        cases = [
            ((10, 20, .5), (1, "pv_surplus")),
            ((10, 0, .05), (1, "low_price")),
            ((10, 0, .4), (-1, "high_price")),
            ((0, 0, .4), (0, "hold")),
            ((10, 0, .1), (0, "hold")),
            ((10, 0, .3), (0, "hold")),
        ]
        for observation, expected in cases:
            self.assertEqual(self.controller.decide(*observation, battery), expected)
        battery.soc_kwh = self.config.max_soc_kwh
        self.assertEqual(self.controller.decide(0, 10, .05, battery), (0, "hold"))
        battery.soc_kwh = self.config.min_soc_kwh
        self.assertEqual(self.controller.decide(10, 0, .4, battery), (0, "hold"))

    def test_analytical_charge_top_up_losses_and_capped_discharge(self):
        data = hourly([10, 5, 100], [30, 0, 0], [.4, .4, .4])
        result = run_rule_based_controller(data, self.controller, self.config)
        np.testing.assert_array_equal(result.action, [1, -1, -1])
        charge = 45 / .95
        np.testing.assert_allclose(result.charge_from_bus_kwh, [charge, 0, 0])
        np.testing.assert_allclose(result.discharge_to_bus_kwh, [0, 5, 50])
        np.testing.assert_allclose(result.unused_discharge_kwh, [0, 0, 0])
        np.testing.assert_allclose(result.grid_to_battery_kwh, [charge - 20, 0, 0])
        np.testing.assert_allclose(result.grid_import_kwh, [charge - 20, 0, 50])
        np.testing.assert_allclose(result.soc_kwh, [95, 95 - 5 / .95, 95 - 55 / .95])
        validate_results(result, self.config, True)
        metrics = calculate_metrics(result, 42, 100)
        expected_cost = (charge - 20 + 50) * .4
        self.assertAlmostEqual(metrics["total_electricity_cost_eur"], expected_cost)
        self.assertAlmostEqual(metrics["cost_savings_eur"], 42 - expected_cost)
        self.assertAlmostEqual(metrics["cost_savings_pct"], (42 - expected_cost) / 42 * 100)
        self.assertAlmostEqual(metrics["battery_throughput_kwh"], charge + 55)
        self.assertAlmostEqual(metrics["equivalent_full_cycles"], (charge + 55) / 200)
        self.assertAlmostEqual(metrics["pv_self_consumption_pct"], 100)
        self.assertAlmostEqual(metrics["battery_losses_kwh"], charge * .05 + 55 * (1 / .95 - 1))
        self.assertAlmostEqual(metrics["net_soc_change_kwh"], 45 - 55 / .95)

    def test_replay_matches_unmodified_environment_and_battery(self):
        rng = np.random.default_rng(301)
        data = validate_hourly_data(hourly(
            rng.uniform(0, 100, 400), rng.uniform(0, 100, 400), rng.uniform(-.1, .6, 400)
        ))
        result = run_rule_based_controller(data, self.controller, self.config)
        env = EnergyStorageEnv(data, self.config)
        env.reset()
        battery = BatteryModel(self.config)
        for row in result.itertuples(index=False):
            state, reward, done, info = env.step(row.action)
            for key, value in info.items():
                if key == "timestamp":
                    self.assertEqual(value, row.timestamp)
                else:
                    self.assertAlmostEqual(value, getattr(row, key))
            physical = battery.step(row.action, max_discharge_to_bus_kwh=max(row.demand_kwh - row.pv_generation_kwh, 0))
            for key, value in physical.items():
                self.assertAlmostEqual(value, getattr(row, key))
        self.assertTrue(done)
        self.assertIsNone(state)
        validate_results(result, self.config, True)

    def test_repeated_actions_respect_limits_and_efficiencies(self):
        config = replace(self.config, max_charge_power_kw=7, max_discharge_power_kw=9,
                         charge_efficiency=.8, discharge_efficiency=.9)
        data = hourly([100] * 40, [0] * 40, [.05] * 20 + [.4] * 20)
        result = run_rule_based_controller(data, self.controller, config)
        validate_results(result, config, True)
        self.assertAlmostEqual(result.soc_kwh.max(), 95)
        self.assertAlmostEqual(result.soc_kwh.min(), 10)
        self.assertAlmostEqual(result.charge_from_bus_kwh.max(), 7)
        self.assertAlmostEqual(result.discharge_to_bus_kwh.max(), 9)

    def test_evaluation_resets_once_and_is_repeatable(self):
        data = hourly([20] * 8, [0] * 8, [.4] * 8)
        before = data.copy(deep=True)
        first = run_rule_based_controller(data, self.controller, self.config)
        second = run_rule_based_controller(data, self.controller, self.config)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(data, before)
        self.assertEqual(first.soc_before_kwh.iloc[0], 50)
        self.assertEqual(first.soc_kwh.iloc[-1], 10)
        self.assertAlmostEqual(first.soc_before_kwh.iloc[1], 50 - 20 / .95)

    def test_training_only_quantiles_ignore_validation_and_test(self):
        data = generate_energy_dataset()
        original = evaluate_baselines(data, self.config)
        changed = data.copy()
        changed.loc[8760:, "electricity_price_eur_kwh"] = 1000.0
        modified = evaluate_baselines(changed, self.config)
        self.assertEqual(original[4].thresholds, modified[4].thresholds)
        expected = data.iloc[:8760].electricity_price_eur_kwh.quantile([.25, .75])
        self.assertEqual(original[4].thresholds.low_eur_kwh, expected.loc[.25])
        self.assertEqual(original[4].thresholds.high_eur_kwh, expected.loc[.75])
        self.assertEqual(original[4].thresholds.training_rows, 8760)
        validation_changed = data.copy()
        validation_changed.loc[8760:17543, "electricity_price_eur_kwh"] = -500.0
        validation_run = evaluate_baselines(validation_changed, self.config)
        pd.testing.assert_frame_equal(original[1], validation_run[1])

    def test_future_perturbation_and_truncation_leave_past_unchanged(self):
        data = hourly([20] * 20, [0] * 20, [.05, .4] * 10)
        original = run_rule_based_controller(data, self.controller, self.config)
        changed = data.copy()
        changed.loc[10:, "electricity_price_eur_kwh"] = -100.0
        changed.loc[10:, "demand_kwh"] = 0.0
        changed.loc[10:, "pv_generation_kwh"] = 200.0
        future_run = run_rule_based_controller(changed, self.controller, self.config)
        truncated = run_rule_based_controller(data.iloc[:10], self.controller, self.config)
        pd.testing.assert_frame_equal(original.iloc[:10], future_run.iloc[:10])
        pd.testing.assert_frame_equal(original.iloc[:10], truncated)

    def test_reject_evaluation_overlap_with_training(self):
        training = hourly([10] * 8, [0] * 8, [.1] * 8)
        controller = RuleBasedController.fit(training)
        with self.assertRaises(ValueError):
            run_rule_based_controller(training.iloc[-2:], controller, self.config)

    def test_reject_invalid_input_and_configuration(self):
        good = hourly([10] * 4, [0] * 4, [.1] * 4)
        for invalid in (good.iloc[::-1], good.iloc[[0, 0]], good.iloc[[0, 2]], good.iloc[:0]):
            with self.assertRaises(ValueError):
                validate_hourly_data(invalid)
        for column, value in (("demand_kwh", -1), ("pv_generation_kwh", np.nan),
                              ("electricity_price_eur_kwh", np.inf)):
            invalid = good.copy()
            invalid[column] = invalid[column].astype(float)
            invalid.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_hourly_data(invalid)
        for config in (replace(self.config, capacity_kwh=0),
                       replace(self.config, charge_efficiency=0),
                       replace(self.config, initial_soc_fraction=.99),
                       replace(self.config, timestep_hours=.5)):
            with self.assertRaises(ValueError):
                run_rule_based_controller(good, self.controller, config)

    def test_assertions_detect_corrupted_results(self):
        result = run_rule_based_controller(
            hourly([10] * 3, [0] * 3, [.05, .4, .4]), self.controller, self.config
        )
        for column, value in (("soc_kwh", 101), ("grid_import_kwh", -1),
                              ("charge_from_bus_kwh", 60), ("grid_cost_eur", np.nan),
                              ("unused_discharge_kwh", -5)):
            damaged = result.copy()
            damaged.loc[0, column] = value
            with self.assertRaises(AssertionError):
                validate_results(damaged, self.config, True)

    def test_week_selection_depends_only_on_calendar(self):
        data = hourly([10] * 24 * 21, [0] * 24 * 21, [.2] * 24 * 21, "2025-11-07 06:00")
        week = select_example_week(data)
        self.assertEqual(len(week), 168)
        self.assertEqual(week.timestamp.iloc[0], pd.Timestamp("2025-11-10 00:00"))
        self.assertEqual(week.timestamp.iloc[-1], pd.Timestamp("2025-11-16 23:00"))
        with self.assertRaises(ValueError):
            select_example_week(data.iloc[:24])

    def test_actual_dataset_integration(self):
        data = pd.read_csv(ROOT / "data/processed/energy_hourly_2023_2025.csv", parse_dates=["timestamp"])
        config, penalty = load_battery_config(ROOT / "config/battery_config.json")
        no_battery, rule_based, comparison, metrics, controller, splits = evaluate_baselines(
            data, config, penalty
        )
        self.assertEqual([splits[name]["rows"] for name in ("training", "validation", "test")],
                         [8760, 8784, 8760])
        self.assertEqual(len(no_battery), 8760)
        pd.testing.assert_series_equal(no_battery.timestamp, rule_based.timestamp)
        self.assertEqual(rule_based.timestamp.iloc[0], pd.Timestamp("2025-01-01 00:00"))
        self.assertEqual(rule_based.timestamp.iloc[-1], pd.Timestamp("2025-12-31 23:00"))
        self.assertFalse(comparison.isna().any().any())
        self.assertTrue(np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all())
        self.assertEqual(config.capacity_kwh, 100)
        self.assertEqual(config.min_soc_fraction, .1)
        self.assertEqual(config.max_soc_fraction, .95)
        self.assertEqual(penalty, .01)


if __name__ == "__main__":
    unittest.main()
