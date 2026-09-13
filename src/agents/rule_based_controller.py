"""A fixed, causal Stage-3 policy using the Stage-4.5 BatteryModel."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.environment.battery_model import BatteryConfig, BatteryModel
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import validate_battery_config, validate_hourly_data


@dataclass(frozen=True)
class PriceThresholds:
    low_eur_kwh: float
    high_eur_kwh: float
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    training_rows: int


class RuleBasedController:
    """Priority: surplus-PV charge, cheap charge, expensive discharge, hold.

    Actions match both current classes: -1 discharge, 0 hold, 1 charge.
    The README's proposed 0/1/2 encoding is not used by the existing code.
    No price forecasts, rolling refits, lookahead, or test-period optimization.
    """

    def __init__(self, thresholds: PriceThresholds):
        if not np.isfinite([thresholds.low_eur_kwh, thresholds.high_eur_kwh]).all():
            raise ValueError("Price thresholds must be finite.")
        if thresholds.low_eur_kwh > thresholds.high_eur_kwh:
            raise ValueError("Low threshold must not exceed high threshold.")
        self.thresholds = thresholds

    @classmethod
    def fit(cls, training_data: pd.DataFrame):
        """Fit ONLY on the training frame supplied by chronological_split.

        Pandas quantiles use linear interpolation. Equality to a threshold
        does not trigger a price-based charge/discharge.
        """
        training = validate_hourly_data(training_data)
        quantiles = training.electricity_price_eur_kwh.quantile(
            [0.25, 0.75], interpolation="linear"
        )
        return cls(PriceThresholds(
            float(quantiles.loc[0.25]), float(quantiles.loc[0.75]),
            training.timestamp.iloc[0], training.timestamp.iloc[-1], len(training)
        ))

    def decide(self, demand_kwh: float, pv_kwh: float, price_eur_kwh: float,
               battery: BatteryModel):
        """Use only current observations, fixed training thresholds and SOC."""
        can_charge = battery.soc_kwh < battery.config.max_soc_kwh
        if can_charge and pv_kwh > demand_kwh:
            return 1, "pv_surplus"
        if can_charge and price_eur_kwh < self.thresholds.low_eur_kwh:
            return 1, "low_price"
        if (demand_kwh > pv_kwh and price_eur_kwh > self.thresholds.high_eur_kwh
                and battery.soc_kwh > battery.config.min_soc_kwh):
            return -1, "high_price"
        return 0, "hold"


def run_rule_based_controller(data: pd.DataFrame, controller: RuleBasedController,
                              config: BatteryConfig,
                              throughput_penalty_eur_per_kwh: float = 0.01):
    """Reset once at evaluation start and dispatch through EnergyStorageEnv.

    The existing BatteryModel chooses the full feasible charge/discharge
    amount. PV surplus supplies charge first; grid supplies any remainder,
    even if a surplus (rather than a low price) triggered the action. Discharge
    is capped by residual load before SOC is updated; unused output is zero.
    Grid cost excludes the environment's separate cycling penalty.
    """
    data = validate_hourly_data(data)
    validate_battery_config(config)
    if data.timestamp.iloc[0] <= controller.thresholds.training_end:
        raise ValueError("Evaluation must begin strictly after the training period.")
    if not np.isfinite(throughput_penalty_eur_per_kwh) or throughput_penalty_eur_per_kwh < 0:
        raise ValueError("Throughput penalty must be finite and nonnegative.")
    env = EnergyStorageEnv(data, config, throughput_penalty_eur_per_kwh)
    env.reset()
    records = []
    for row in data.itertuples(index=False):
        soc_before = env.battery.soc_kwh
        action, reason = controller.decide(
            row.demand_kwh, row.pv_generation_kwh,
            row.electricity_price_eur_kwh, env.battery
        )
        _, _, _, info = env.step(action)
        charge = info["pv_to_battery_kwh"] + info["grid_to_battery_kwh"]
        discharge = info["battery_throughput_kwh"] - charge
        info.update({
            "electricity_price_eur_kwh": row.electricity_price_eur_kwh,
            "action_reason": reason,
            "soc_before_kwh": soc_before,
            "charge_from_bus_kwh": charge,
            "discharge_to_bus_kwh": discharge,
            "unused_discharge_kwh": discharge - info["discharge_to_load_kwh"],
            "pv_curtailed_kwh": (row.pv_generation_kwh - info["pv_used_directly_kwh"]
                                  - info["pv_to_battery_kwh"]),
            "battery_losses_kwh": (charge * (1 - config.charge_efficiency)
                                    + discharge * (1 / config.discharge_efficiency - 1)),
        })
        records.append(info)
    return pd.DataFrame.from_records(records)
