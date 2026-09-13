import numpy as np
import pandas as pd

from .battery_model import BatteryConfig, BatteryModel


class EnergyStorageEnv:
    """
    Energy environment, refined in Stage 4.5: discharge cannot exceed residual demand.

    Actions:
        -1 = discharge
         0 = hold
         1 = charge
    """

    def __init__(self, data: pd.DataFrame, battery_config=None, throughput_penalty_eur_per_kwh=0.01):
        self.data = data.reset_index(drop=True).copy()
        self.battery = BatteryModel(battery_config or BatteryConfig())
        self.throughput_penalty = throughput_penalty_eur_per_kwh
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.battery.reset()
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        hour = float(row["hour"])

        return np.array([
            float(row["demand_kwh"]),
            float(row["pv_generation_kwh"]),
            float(row["electricity_price_eur_kwh"]),
            float(self.battery.soc_kwh / self.battery.config.capacity_kwh),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            float(row["is_weekend"]),
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]

        demand = float(row["demand_kwh"])
        pv = float(row["pv_generation_kwh"])
        price = float(row["electricity_price_eur_kwh"])

        direct_pv = min(demand, pv)
        residual_demand = max(demand - pv, 0.0)
        pv_surplus = max(pv - demand, 0.0)

        battery_result = self.battery.step(action, max_discharge_to_bus_kwh=residual_demand)
        charge = battery_result["charge_from_bus_kwh"]
        discharge = battery_result["discharge_to_bus_kwh"]

        pv_to_battery = min(pv_surplus, charge)
        grid_to_battery = max(charge - pv_to_battery, 0.0)

        useful_discharge = min(discharge, residual_demand)
        grid_import = max(residual_demand - useful_discharge, 0.0) + grid_to_battery

        grid_cost = grid_import * price
        throughput = charge + discharge
        penalty = throughput * self.throughput_penalty
        reward = -(grid_cost + penalty)

        info = {
            "timestamp": row["timestamp"],
            "action": action,
            "demand_kwh": demand,
            "pv_generation_kwh": pv,
            "pv_used_directly_kwh": direct_pv,
            "pv_to_battery_kwh": pv_to_battery,
            "grid_to_battery_kwh": grid_to_battery,
            "discharge_to_load_kwh": useful_discharge,
            "grid_import_kwh": grid_import,
            "grid_cost_eur": grid_cost,
            "battery_throughput_kwh": throughput,
            "degradation_penalty_eur": penalty,
            "soc_kwh": battery_result["soc_kwh"],
            "soc_fraction": battery_result["soc_fraction"],
            "reward": reward,
        }

        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = None if done else self._get_state()

        return next_state, reward, done, info
