import unittest
from dataclasses import asdict
import numpy as np

from src.agents.dqn_agent import DQNConfig
from src.evaluation.dqn_robustness import METRICS, seed_row, summarize_seed_rows
from src.evaluation.final_portfolio import build_final_comparison, FINAL_COLUMNS
from src.training.train_dqn_seed import config_for_seed

class Stage6RobustnessTests(unittest.TestCase):
    def test_seed_configs_change_only_seed(self):
        one, two = asdict(config_for_seed(7)), asdict(config_for_seed(123))
        self.assertEqual(one.pop('seed'), 7)
        self.assertEqual(two.pop('seed'), 123)
        self.assertEqual(one, two)
        reference = asdict(DQNConfig())
        reference.pop('seed')
        self.assertEqual(one, reference)

    def test_seed_row_keeps_physical_capture_separate(self):
        metrics = {key: i + 1. for i, key in enumerate(METRICS)}
        behavior = {'pv_surplus_captured_pct': 44.2}
        row = seed_row(42, metrics, behavior)
        self.assertEqual(row['seed'], 42)
        self.assertEqual(row['pv_surplus_captured_pct'], 44.2)
        self.assertEqual(row['row_type'], 'seed')

    def test_summary_has_three_seeds_and_sample_statistics(self):
        rows = []
        for index, seed in enumerate((42, 7, 123), start=1):
            metrics = {key: float(index) for key in METRICS}
            rows.append(seed_row(seed, metrics, {'pv_surplus_captured_pct': float(index)}))
        result = summarize_seed_rows(rows)
        self.assertEqual(result.seed_or_statistic.tolist(), ['42', '7', '123', 'mean', 'std', 'min', 'max'])
        self.assertEqual(result.loc[result.seed_or_statistic == 'mean', METRICS[0]].iloc[0], 2.)
        self.assertEqual(result.loc[result.seed_or_statistic == 'std', METRICS[0]].iloc[0], 1.)
        self.assertTrue(np.isfinite(result[list(METRICS)].to_numpy()).all())

    def test_summary_rejects_missing_or_duplicate_seeds(self):
        metrics = {key: 1. for key in METRICS}
        row = seed_row(42, metrics, {'pv_surplus_captured_pct': 1.})
        with self.assertRaises(ValueError):
            summarize_seed_rows([row, row, row])

    def test_final_comparison_calculates_physical_pv_capture(self):
        import pandas as pd
        frame = pd.DataFrame({
            'Strategy': ['No Battery', 'Rule-Based', 'Q-Learning', 'DQN'],
            **{column: [1., 2., 3., 4.] for column in FINAL_COLUMNS[1:]
               if column != 'pv_surplus_captured_pct'},
            'pv_surplus_kwh': [10., 10., 0., 20.],
            'pv_to_battery_kwh': [0., 2., 0., 5.],
        })
        result = build_final_comparison(frame)
        self.assertEqual(tuple(result.columns), FINAL_COLUMNS)
        self.assertEqual(result.pv_surplus_captured_pct.tolist(), [0., 20., 0., 25.])

if __name__ == '__main__':
    unittest.main()
