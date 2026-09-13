"""Training-only, interpretable 3 x 4 x 4 x 4 x 2 state space (384 states)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from src.evaluation.data_utils import validate_battery_config, validate_hourly_data


STATE_COLUMNS = ("price_bin", "net_load_bin", "soc_bin", "time_block", "weekend")
STATE_SHAPE = (3, 4, 4, 4, 2)


@dataclass(frozen=True)
class StateDiscretizer:
    price_edges: tuple
    net_load_edges: tuple
    soc_edges: tuple
    min_soc: float
    max_soc: float
    training_start: str
    training_end: str
    training_rows: int

    @classmethod
    def fit(cls, training_data, config):
        """Caller supplies training ONLY; no fitting in evaluation.

        Price: training P33/P66. Net load: positive-training-net-load P33/P66;
        all negative/zero net load falls in a dedicated surplus bin (0). If no positive load
        exists, both cut points are zero. Equality goes in the higher bin.
        SOC: four equal intervals within the configured usable SOC range.
        """
        data = validate_hourly_data(training_data)
        validate_battery_config(config)
        if config.min_soc_fraction >= config.max_soc_fraction:
            raise ValueError("Discretization requires a nonzero usable SOC range.")
        net = data.demand_kwh - data.pv_generation_kwh
        positive = net[net > 0]
        return cls(
            tuple(data.electricity_price_eur_kwh.quantile([.33, .66]).astype(float)),
            tuple(positive.quantile([.33, .66]).astype(float)) if len(positive) else (0., 0.),
            tuple(np.linspace(config.min_soc_fraction, config.max_soc_fraction, 5)[1:-1]),
            config.min_soc_fraction, config.max_soc_fraction,
            str(data.timestamp.iloc[0]), str(data.timestamp.iloc[-1]), len(data),
        )

    def encode(self, price, net_load, soc_fraction, hour, weekend):
        if not np.isfinite([price, net_load, soc_fraction, hour, weekend]).all():
            raise ValueError("State values must be finite.")
        if not self.min_soc - 1e-8 <= soc_fraction <= self.max_soc + 1e-8:
            raise ValueError("SOC outside configured physical limits.")
        if int(hour) != hour or not 0 <= hour < 24 or weekend not in (0, 1):
            raise ValueError("Invalid hour or weekend flag.")
        # Direct comparisons make scalar encoding cheap; equal edges are valid.
        price_bin = sum(price >= edge for edge in self.price_edges)
        net_bin = 0 if net_load <= 0 else 1 + sum(net_load >= edge for edge in self.net_load_edges)
        soc_bin = sum(soc_fraction >= edge for edge in self.soc_edges)
        return int(price_bin), int(net_bin), int(soc_bin), int(hour) // 6, int(weekend)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for name in ("price_edges", "net_load_edges", "soc_edges"):
            raw[name] = tuple(raw[name])
        return cls(**raw)
