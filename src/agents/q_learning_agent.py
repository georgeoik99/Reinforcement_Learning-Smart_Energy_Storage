"""Standard tabular Q-Learning. Q values represent discounted reward in EUR."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .state_discretizer import STATE_COLUMNS, STATE_SHAPE


# Internal action index -> preserved environment action. No action masking.
ENV_ACTIONS = (-1, 0, 1)
ACTION_NAMES = ("discharge", "hold", "charge")


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 200
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.98
    epsilon_min: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.episodes, int) or self.episodes < 1:
            raise ValueError("episodes must be a positive integer.")
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid alpha/gamma.")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1 or not 0 < self.epsilon_decay <= 1:
            raise ValueError("Invalid epsilon schedule.")


class QLearningAgent:
    def __init__(self, config=None):
        self.config = config or QLearningConfig()
        self.q_table = np.zeros((*STATE_SHAPE, 3), dtype=np.float64)
        self.visits = np.zeros((*STATE_SHAPE, 3), dtype=np.int64)
        self.rng = np.random.default_rng(self.config.seed)
        self.epsilon = self.config.epsilon_start

    def greedy_action(self, state):
        """Deterministic ties: hold, then discharge, then charge.

        Zero-initialized unseen states therefore hold. Evaluation never draws
        random numbers or creates table entries. Negative learned values can
        still make unvisited actions optimistic; visits are exported explicitly.
        """
        values = self.q_table[state]
        best = float(values.max())
        return next(action for action in (1, 0, 2) if values[action] == best)

    def select_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        if action not in (0, 1, 2) or not np.isfinite(reward):
            raise ValueError("Invalid action or nonfinite reward.")
        target = reward if done else reward + self.config.gamma * self.q_table[next_state].max()
        key = (*state, action)
        self.q_table[key] += self.config.alpha * (target - self.q_table[key])
        self.visits[key] += 1
        if not np.isfinite(self.q_table[key]):
            raise AssertionError("Nonfinite Q value.")

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def to_frame(self):
        rows = []
        for state in product(*(range(n) for n in STATE_SHAPE)):
            action = self.greedy_action(state)
            rows.append({
                **dict(zip(STATE_COLUMNS, state)),
                **dict(zip(("q_discharge", "q_hold", "q_charge"), self.q_table[state])),
                **dict(zip(("visits_discharge", "visits_hold", "visits_charge"), self.visits[state])),
                "state_visits": int(self.visits[state].sum()),
                "best_action": ACTION_NAMES[action], "best_action_index": action,
                "best_environment_action": ENV_ACTIONS[action],
            })
        return pd.DataFrame(rows)

    def save(self, path):
        self.to_frame().to_csv(path, index=False)

    @classmethod
    def load(cls, path, config=None):
        frame = pd.read_csv(path, float_precision="round_trip")
        agent = cls(config)
        if len(frame) != int(np.prod(STATE_SHAPE)) or frame.duplicated(list(STATE_COLUMNS)).any():
            raise ValueError("Q-table must contain exactly one row per state.")
        for row in frame.itertuples(index=False):
            state = tuple(int(getattr(row, name)) for name in STATE_COLUMNS)
            if any(not 0 <= value < size for value, size in zip(state, STATE_SHAPE)):
                raise ValueError("Invalid state index in saved table.")
            agent.q_table[state] = (row.q_discharge, row.q_hold, row.q_charge)
            agent.visits[state] = (row.visits_discharge, row.visits_hold, row.visits_charge)
        if not np.isfinite(agent.q_table).all() or (agent.visits < 0).any():
            raise ValueError("Invalid Q-table values.")
        return agent
