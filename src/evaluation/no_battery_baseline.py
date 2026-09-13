"""No-storage reference: PV serves demand first; surplus is curtailed."""

import numpy as np
import pandas as pd

from .data_utils import ENERGY_COLUMNS, validate_hourly_data


def run_no_battery_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Hourly bus-side kWh and grid purchasing cost in EUR, with no export.

    Zero battery columns provide a common accounting schema. SOC=0 here means
    no battery is installed; the configured battery SOC limits do not apply.
    """
    data = validate_hourly_data(data)
    result = data[["timestamp", *ENERGY_COLUMNS]].copy()
    result["pv_used_directly_kwh"] = np.minimum(data.demand_kwh, data.pv_generation_kwh)
    result["grid_import_kwh"] = np.maximum(data.demand_kwh - data.pv_generation_kwh, 0)
    result["grid_cost_eur"] = result.grid_import_kwh * data.electricity_price_eur_kwh
    result["pv_curtailed_kwh"] = data.pv_generation_kwh - result.pv_used_directly_kwh
    result["action"] = 0
    result["action_reason"] = "no_battery"
    for column in (
        "pv_to_battery_kwh", "grid_to_battery_kwh", "charge_from_bus_kwh",
        "discharge_to_bus_kwh", "discharge_to_load_kwh", "unused_discharge_kwh",
        "battery_throughput_kwh", "battery_losses_kwh", "degradation_penalty_eur",
        "soc_before_kwh", "soc_kwh", "soc_fraction",
    ):
        result[column] = 0.0
    result["reward"] = -result.grid_cost_eur
    return result
