"""Input validation, chronological splitting and existing JSON configuration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.battery_model import BatteryConfig


ENERGY_COLUMNS = [
    "demand_kwh", "pv_generation_kwh", "electricity_price_eur_kwh"
]


def validate_hourly_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy; reject gaps, duplicates and disorder instead of sorting.

    The simulation treats each row as one hour and observes the current hour's demand,
    PV and price before dispatch. Negative prices are allowed; energy is not.
    """
    missing = {"timestamp", *ENERGY_COLUMNS} - set(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Hourly data must not be empty.")
    result = data.copy().reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    times = result["timestamp"]
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be non-null, unique and chronological.")
    if not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("The simulation requires contiguous hourly observations.")
    values = result[ENERGY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Demand, PV and price must be finite and non-null.")
    if (values[:, :2] < 0).any():
        raise ValueError("Demand and PV must be nonnegative.")
    # The preserved environment needs these deterministic, current-time features.
    result["hour"] = times.dt.hour
    result["is_weekend"] = (times.dt.dayofweek >= 5).astype(int)
    return result


def chronological_split(data: pd.DataFrame):
    """Full calendar years: 2023 train, 2024 validation, 2025 test.

    Reject incomplete years and legacy percentage splits explicitly. Hourly
    labels are naive simulation time (no DST), including 2024-02-29's 24 hours.
    """
    data = validate_hourly_data(data)
    expected = pd.date_range("2023-01-01", "2026-01-01", freq="h", inclusive="left")
    if len(data) != len(expected) or not np.array_equal(data.timestamp.to_numpy(), expected.to_numpy()):
        raise ValueError("Require complete hourly 2023-2025 data; legacy 70/15/15 is obsolete.")
    return tuple(data.loc[data.timestamp.dt.year == year].copy().reset_index(drop=True)
                 for year in (2023, 2024, 2025))


def validate_battery_config(config: BatteryConfig):
    values = np.array(list(vars(config).values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Battery configuration must be finite.")
    if config.capacity_kwh <= 0 or config.timestep_hours != 1:
        raise ValueError("The simulation requires positive capacity and a one-hour timestep.")
    if min(config.max_charge_power_kw, config.max_discharge_power_kw) < 0:
        raise ValueError("Power limits must be nonnegative.")
    if not (0 < config.charge_efficiency <= 1 and 0 < config.discharge_efficiency <= 1):
        raise ValueError("Efficiencies must be in (0, 1].")
    if not (0 <= config.min_soc_fraction <= config.initial_soc_fraction
            <= config.max_soc_fraction <= 1):
        raise ValueError("Require 0 <= minimum SOC <= initial SOC <= maximum SOC <= 1.")


def load_battery_config(path: Path):
    """Map the existing JSON keys without rewriting the Stage-2 config/model."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    config = BatteryConfig(
        capacity_kwh=raw["battery_capacity_kwh"],
        max_charge_power_kw=raw["max_charge_power_kw"],
        max_discharge_power_kw=raw["max_discharge_power_kw"],
        charge_efficiency=raw["charge_efficiency"],
        discharge_efficiency=raw["discharge_efficiency"],
        min_soc_fraction=raw["minimum_soc_fraction"],
        max_soc_fraction=raw["maximum_soc_fraction"],
        initial_soc_fraction=raw["initial_soc_fraction"],
        timestep_hours=raw["time_step_hours"],
    )
    validate_battery_config(config)
    penalty = float(raw["battery_throughput_penalty_eur_per_kwh"])
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Throughput penalty must be finite and nonnegative.")
    return config, penalty
