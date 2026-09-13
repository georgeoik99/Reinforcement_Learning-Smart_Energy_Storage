"""Calendar data, physical dispatch and training-priced terminal inventory."""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd

from src.data.generate_synthetic_energy_data import generate_energy_dataset
from src.environment.battery_model import BatteryConfig, BatteryModel
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import chronological_split, validate_hourly_data
from src.evaluation.dataset_report import dataset_statistics
from src.evaluation.terminal_valuation import TerminalValuation, calculate_refined_metrics
from src.evaluation.no_battery_baseline import run_no_battery_baseline
from src.agents.state_discretizer import StateDiscretizer, STATE_SHAPE


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_energy_dataset()
        cls.training, cls.validation, cls.test = chronological_split(cls.data)
        cls.config = BatteryConfig()

    def test_calendar_chronology_and_leap_day(self):
        self.assertEqual(len(self.data), 26304)
        self.assertEqual([len(self.training), len(self.validation), len(self.test)], [8760, 8784, 8760])
        self.assertEqual(len(self.validation[self.validation.timestamp.dt.strftime("%m-%d") == "02-29"]), 24)
        self.assertTrue(self.data.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all())
        self.assertFalse(self.data.timestamp.duplicated().any())
        self.assertTrue(np.isfinite(self.data.select_dtypes(include="number").to_numpy()).all())

    def test_reject_missing_extra_and_legacy_years(self):
        for frame in (self.data.iloc[1:], self.data.iloc[:-1], self.test, self.data.iloc[::-1]):
            with self.assertRaises(ValueError):
                chronological_split(frame)

    def test_seeded_generator_and_independent_year_variation(self):
        pd.testing.assert_frame_equal(self.data, generate_energy_dataset(), check_exact=True)
        self.assertFalse(np.array_equal(self.training.demand_kwh, self.test.demand_kwh))
        changed = generate_energy_dataset(seed=43)
        self.assertFalse(np.array_equal(self.data.demand_kwh, changed.demand_kwh))
        self.assertLessEqual(self.data.pv_generation_kwh.max(), 100)

    def test_surplus_emerges_in_all_annual_splits(self):
        for part in (self.training, self.validation, self.test):
            surplus = part[part.pv_generation_kwh > part.demand_kwh]
            self.assertGreater(len(surplus), 0)
            self.assertGreater(surplus.timestamp.dt.date.nunique(), 1)
            self.assertTrue(surplus.hour.between(6, 19).all())
        report = dataset_statistics(self.data)
        self.assertTrue((report.pv_surplus_kwh > 0).all())
        self.assertTrue(report.pv_direct_consumption_pct.between(0, 100, inclusive="neither").all())

    def test_dedicated_surplus_bin_training_positive_edges(self):
        disc = StateDiscretizer.fit(self.training, self.config)
        self.assertEqual(int(np.prod(STATE_SHAPE)), 384)
        self.assertEqual(disc.encode(.1, -1, .5, 12, 0)[1], 0)
        self.assertEqual(disc.encode(.1, 0, .5, 12, 0)[1], 0)
        self.assertEqual(disc.encode(.1, .0001, .5, 12, 0)[1], 1)
        self.assertEqual(disc.encode(.1, disc.net_load_edges[0], .5, 12, 0)[1], 2)
        self.assertEqual(disc.encode(.1, disc.net_load_edges[1], .5, 12, 0)[1], 3)

    def test_battery_caps_before_soc_transition(self):
        battery = BatteryModel(self.config)
        result = battery.step(-1, max_discharge_to_bus_kwh=5)
        self.assertAlmostEqual(result["discharge_to_bus_kwh"], 5)
        self.assertAlmostEqual(result["soc_kwh"], 50 - 5 / .95)
        result = battery.step(-1, max_discharge_to_bus_kwh=0)
        self.assertEqual(result["discharge_to_bus_kwh"], 0)
        self.assertAlmostEqual(result["soc_kwh"], 50 - 5 / .95)
        for cap in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                battery.step(-1, max_discharge_to_bus_kwh=cap)

    def test_surplus_discharge_does_not_destroy_energy(self):
        frame = self.test.iloc[:2].copy()
        frame.demand_kwh = [5., 100.]
        frame.pv_generation_kwh = [10., 0.]
        env = EnergyStorageEnv(frame, self.config)
        env.reset()
        _, _, _, first = env.step(-1)
        self.assertEqual(first["soc_kwh"], 50)
        self.assertEqual(first["battery_throughput_kwh"], 0)
        _, _, _, second = env.step(-1)
        self.assertAlmostEqual(second["discharge_to_load_kwh"], 38)
        self.assertEqual(second["soc_kwh"], 10)

    def test_terminal_replenishment_credit_equal_soc(self):
        value = TerminalValuation(.1, "2023-01-01", "2023-12-31 23:00")
        self.assertAlmostEqual(value.adjustment(50, 10, self.config), 40 / .95 * .1)
        self.assertAlmostEqual(value.adjustment(50, 90, self.config), -40 * .95 * .1)
        self.assertEqual(value.adjustment(50, 50, self.config), 0)
        other = replace(self.config, charge_efficiency=.8, discharge_efficiency=.9)
        self.assertAlmostEqual(value.adjustment(50, 10, other), 5)
        self.assertAlmostEqual(value.adjustment(50, 90, other), -3.6)

    def test_terminal_valuation_and_bins_ignore_future_prices(self):
        value = TerminalValuation.fit(self.training)
        disc = StateDiscretizer.fit(self.training, self.config)
        changed = self.data.copy()
        changed.loc[changed.timestamp.dt.year > 2023, "electricity_price_eur_kwh"] = 999
        train, _, _ = chronological_split(changed)
        self.assertEqual(value, TerminalValuation.fit(train))
        self.assertEqual(disc, StateDiscretizer.fit(train, self.config))
        self.assertEqual(value.reference_price_eur_kwh, self.training.electricity_price_eur_kwh.median())

    def test_refined_metrics_efc_day_and_raw_separation(self):
        result = run_no_battery_baseline(self.validation)
        value = TerminalValuation.fit(self.training)
        raw = float(result.grid_cost_eur.sum())
        metrics = calculate_refined_metrics(result, raw, self.config, value)
        self.assertEqual(metrics["evaluation_days"], 366)
        self.assertEqual(metrics["efc_per_day"], 0)
        self.assertEqual(metrics["terminal_soc_adjusted_cost_eur"], raw)
        sample = result.iloc[:24].copy()
        sample["battery_throughput_kwh"] = 10.
        sample["soc_before_kwh"] = 50.
        sample["soc_kwh"] = 10.
        # A metrics arithmetic fixture, not a physical trajectory.
        metrics = calculate_refined_metrics(sample, raw, self.config, value)
        self.assertAlmostEqual(metrics["efc_per_day"], 240 / 200)
        self.assertAlmostEqual(metrics["terminal_soc_adjusted_cost_eur"] - metrics["raw_electricity_cost_eur"],
                               40 / .95 * value.reference_price_eur_kwh)


if __name__ == "__main__":
    unittest.main()
