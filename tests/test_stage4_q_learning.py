"""Q update, discretization, causality, physical replay and reproducibility."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.agents.q_learning_agent import ENV_ACTIONS, QLearningAgent, QLearningConfig
from src.agents.state_discretizer import STATE_SHAPE, StateDiscretizer
from src.environment.battery_model import BatteryConfig
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import chronological_split, validate_hourly_data
from src.evaluation.evaluate_q_learning import evaluate_agent
from src.training.train_q_learning import train_agent


def data(n=96, start="2025-01-01"):
    rng = np.random.default_rng(314)
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="h"),
        "demand_kwh": rng.uniform(0, 70, n), "pv_generation_kwh": rng.uniform(0, 50, n),
        "electricity_price_eur_kwh": rng.uniform(-.05, .3, n)})


class Stage4Tests(unittest.TestCase):
    def setUp(self):
        self.config = BatteryConfig()
        self.training = data()
        self.disc = StateDiscretizer.fit(self.training, self.config)
        self.state = (0, 0, 0, 0, 0)

    def test_q_update_and_terminal_no_bootstrap(self):
        agent = QLearningAgent(QLearningConfig(alpha=.1, gamma=.95))
        next_state = (1, 1, 1, 1, 1)
        agent.q_table[next_state] = [10, 5, 0]
        agent.q_table[self.state] = [2, 0, 0]
        agent.update(self.state, 0, -3, next_state, False)
        self.assertAlmostEqual(agent.q_table[(*self.state, 0)], 2 + .1 * (-3 + .95 * 10 - 2))
        agent.update(self.state, 1, -3, None, True)
        self.assertAlmostEqual(agent.q_table[(*self.state, 1)], -.3)
        self.assertEqual(agent.visits[self.state].sum(), 2)

    def test_action_mapping_and_ties(self):
        self.assertEqual(ENV_ACTIONS, (-1, 0, 1))
        agent = QLearningAgent()
        self.assertEqual(agent.greedy_action(self.state), 1)
        agent.q_table[self.state] = [-2, -3, -1]
        self.assertEqual(ENV_ACTIONS[agent.select_action(self.state, explore=False)], 1)

    def test_exploration_seed_and_decay(self):
        settings = QLearningConfig(seed=8, epsilon_decay=.5, epsilon_min=.1)
        first, second = QLearningAgent(settings), QLearningAgent(settings)
        actions = [first.select_action(self.state) for _ in range(100)]
        self.assertEqual(actions, [second.select_action(self.state) for _ in range(100)])
        self.assertEqual(set(actions), {0, 1, 2})
        for _ in range(20):
            first.decay_epsilon()
        self.assertEqual(first.epsilon, .1)

    def test_discretization_edges_surplus_and_time(self):
        self.assertEqual(int(np.prod(STATE_SHAPE)), 384)
        self.assertEqual(self.disc.encode(-10, -100, .1, 0, 0), (0, 0, 0, 0, 0))
        self.assertEqual(self.disc.encode(10, 1000, .95, 23, 1), (2, 3, 3, 3, 1))
        for hour, block in ((5, 0), (6, 1), (11, 1), (12, 2), (17, 2), (18, 3)):
            self.assertEqual(self.disc.encode(.1, 10, .5, hour, 0)[3], block)
        self.assertEqual(self.disc.encode(self.disc.price_edges[0], 0, self.disc.soc_edges[0], 0, 0)[0], 1)
        self.assertEqual(self.disc.encode(.1, 0, self.disc.soc_edges[0], 0, 0)[2], 1)

    def test_training_only_thresholds(self):
        original = data(26304, start="2023-01-01")
        train, _, _ = chronological_split(original)
        original_disc = StateDiscretizer.fit(train, self.config)
        changed = original.copy()
        changed.loc[8760:, ["electricity_price_eur_kwh", "demand_kwh", "pv_generation_kwh"]] = 500
        changed_train, _, _ = chronological_split(changed)
        self.assertEqual(original_disc, StateDiscretizer.fit(changed_train, self.config))
        np.testing.assert_allclose(original_disc.price_edges, train.electricity_price_eur_kwh.quantile([.33, .66]))
        net = train.demand_kwh - train.pv_generation_kwh
        np.testing.assert_allclose(original_disc.net_load_edges, net[net > 0].quantile([.33, .66]))

    def test_surplus_only_training_and_configured_soc_range(self):
        sample = self.training.copy()
        sample.demand_kwh = 0.
        config = replace(self.config, min_soc_fraction=.2, max_soc_fraction=.8)
        disc = StateDiscretizer.fit(sample, config)
        self.assertEqual(disc.net_load_edges, (0., 0.))
        self.assertEqual(disc.encode(.1, -1, .2, 0, 0)[1], 0)
        self.assertEqual(disc.encode(.1, 1, .8, 0, 0)[2], 3)

    def test_complete_small_training_reproducible(self):
        settings = QLearningConfig(episodes=5, seed=92)
        one, disc1, hist1 = train_agent(self.training, self.config, .01, settings, False)
        two, disc2, hist2 = train_agent(self.training, self.config, .01, settings, False)
        np.testing.assert_array_equal(one.q_table, two.q_table)
        np.testing.assert_array_equal(one.visits, two.visits)
        pd.testing.assert_frame_equal(hist1, hist2, check_exact=True)
        self.assertEqual(disc1, disc2)
        self.assertEqual(one.visits.sum(), 5 * len(self.training))
        self.assertTrue(np.isfinite(one.q_table).all())

    def test_greedy_evaluation_is_frozen_and_matches_environment(self):
        agent, disc, _ = train_agent(self.training, self.config, .01, QLearningConfig(episodes=3), False)
        table = agent.q_table.copy()
        visits = agent.visits.copy()
        sample = data(start="2025-06-01")
        result = evaluate_agent(sample, agent, disc, self.config, .01)
        np.testing.assert_array_equal(table, agent.q_table)
        np.testing.assert_array_equal(visits, agent.visits)
        env = EnergyStorageEnv(validate_hourly_data(sample), self.config)
        env.reset()
        for row in result.itertuples(index=False):
            _, _, _, info = env.step(row.action)
            for key, expected in info.items():
                if key == "timestamp":
                    self.assertEqual(getattr(row, key), expected)
                else:
                    self.assertAlmostEqual(getattr(row, key), expected)
        self.assertFalse(result.isna().any().any())

    def test_future_evaluation_changes_cannot_change_past(self):
        agent, disc, _ = train_agent(self.training, self.config, .01, QLearningConfig(episodes=2), False)
        sample = data(start="2025-06-01")
        original = evaluate_agent(sample, agent, disc, self.config, .01)
        sample.loc[48:, "electricity_price_eur_kwh"] = 100
        sample.loc[48:, "pv_generation_kwh"] = 500
        changed = evaluate_agent(sample, agent, disc, self.config, .01)
        short = evaluate_agent(sample.iloc[:48], agent, disc, self.config, .01)
        pd.testing.assert_frame_equal(original.iloc[:48], changed.iloc[:48])
        pd.testing.assert_frame_equal(original.iloc[:48], short)

    def test_serialization_exact_and_complete(self):
        agent, disc, _ = train_agent(self.training, self.config, .01, QLearningConfig(episodes=2), False)
        with tempfile.TemporaryDirectory() as temp:
            qfile, dfile = Path(temp) / "q.csv", Path(temp) / "disc.json"
            agent.save(qfile)
            disc.save(dfile)
            restored = QLearningAgent.load(qfile)
            np.testing.assert_array_equal(agent.q_table, restored.q_table)
            np.testing.assert_array_equal(agent.visits, restored.visits)
            self.assertEqual(disc, StateDiscretizer.load(dfile))
            self.assertEqual(len(pd.read_csv(qfile)), 384)

    def test_invalid_inputs_rejected(self):
        for settings in ({"alpha": 0}, {"gamma": 2}, {"episodes": 0}, {"epsilon_min": 2}):
            with self.assertRaises(ValueError):
                QLearningConfig(**settings)
        for observation in ((np.nan, 0, .5, 0, 0), (.1, 0, .99, 0, 0), (.1, 0, .5, 24, 0)):
            with self.assertRaises(ValueError):
                self.disc.encode(*observation)
        with self.assertRaises(ValueError):
            evaluate_agent(self.training, QLearningAgent(), self.disc, self.config, .01)
        with self.assertRaises(ValueError):
            QLearningAgent().update(self.state, 1, np.nan, None, True)


if __name__ == "__main__":
    unittest.main()
