from dataclasses import dataclass
import math


@dataclass
class BatteryConfig:
    capacity_kwh: float = 100.0
    max_charge_power_kw: float = 50.0
    max_discharge_power_kw: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc_fraction: float = 0.10
    max_soc_fraction: float = 0.95
    initial_soc_fraction: float = 0.50
    timestep_hours: float = 1.0

    @property
    def min_soc_kwh(self):
        return self.capacity_kwh * self.min_soc_fraction

    @property
    def max_soc_kwh(self):
        return self.capacity_kwh * self.max_soc_fraction

    @property
    def initial_soc_kwh(self):
        return self.capacity_kwh * self.initial_soc_fraction


class BatteryModel:
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.soc_kwh = config.initial_soc_kwh

    def reset(self):
        self.soc_kwh = self.config.initial_soc_kwh
        return self.soc_kwh

    def step(self, action: int, max_discharge_to_bus_kwh=None):
        # The environment passes residual demand as an additional bus-side cap.
        # None preserves standalone BatteryModel usage and its existing API.
        if max_discharge_to_bus_kwh is not None:
            if not math.isfinite(max_discharge_to_bus_kwh) or max_discharge_to_bus_kwh < 0:
                raise ValueError("Discharge cap must be nonnegative and finite.")
        if action not in (-1, 0, 1):
            raise ValueError("Action must be -1, 0 or 1.")

        charge_from_bus_kwh = 0.0
        discharge_to_bus_kwh = 0.0

        if action == 1:
            max_input = self.config.max_charge_power_kw * self.config.timestep_hours
            room = self.config.max_soc_kwh - self.soc_kwh
            charge_from_bus_kwh = min(max_input, room / self.config.charge_efficiency)
            self.soc_kwh += charge_from_bus_kwh * self.config.charge_efficiency

        elif action == -1:
            max_output = self.config.max_discharge_power_kw * self.config.timestep_hours
            removable = self.soc_kwh - self.config.min_soc_kwh
            discharge_to_bus_kwh = min(max_output, removable * self.config.discharge_efficiency)
            if max_discharge_to_bus_kwh is not None:
                discharge_to_bus_kwh = min(discharge_to_bus_kwh, max_discharge_to_bus_kwh)
            self.soc_kwh -= discharge_to_bus_kwh / self.config.discharge_efficiency

        self.soc_kwh = min(max(self.soc_kwh, self.config.min_soc_kwh), self.config.max_soc_kwh)

        return {
            "soc_kwh": self.soc_kwh,
            "soc_fraction": self.soc_kwh / self.config.capacity_kwh,
            "charge_from_bus_kwh": charge_from_bus_kwh,
            "discharge_to_bus_kwh": discharge_to_bus_kwh,
        }
