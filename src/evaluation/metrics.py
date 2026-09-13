"""Business KPIs and hourly accounting checks for the two Stage-3 baselines.

All energies are bus-side kWh per hourly timestep unless explicitly SOC.
Throughput = charge from bus + ALL discharge to bus (including unused output).
EFC = sum(throughput) / (2 * nominal battery capacity); this matches Stage 2's
throughput convention, not an electrochemical degradation estimate.
PV self-consumption = 100 * sum(direct PV + PV input to battery) / sum(PV).
It measures PV captured before storage losses, including PV left in final SOC;
it does not claim that all captured PV was delivered to load during the test.
No PV => 0%; zero reference cost => 0% savings (an undefined ratio convention).
Electricity cost is energy purchasing only, excluding cycling penalties, capex,
fixed charges and any initial/final stored-energy valuation.
"""

import numpy as np
import pandas as pd

from src.environment.battery_model import BatteryConfig


def calculate_metrics(results: pd.DataFrame, baseline_cost_eur: float,
                      battery_capacity_kwh: float) -> dict:
    """Return full-precision KPIs; round only when formatting tables/plots.

    Savings (EUR) = reference cost - strategy cost. Savings (%) divides by the
    reference cost; its usual business interpretation assumes a positive bill.
    Peak grid import is kWh/hourly row, numerically kW for the one-hour timestep.
    """
    if results.empty or results.isna().any().any():
        raise ValueError("Metrics require nonempty results without missing values.")
    if not np.isfinite(results.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Metrics require finite numeric results.")
    if not np.isfinite(battery_capacity_kwh) or battery_capacity_kwh <= 0:
        raise ValueError("Nominal battery capacity must be positive and finite.")
    if not np.isfinite(baseline_cost_eur):
        raise ValueError("Baseline cost must be finite.")
    cost = float(results.grid_cost_eur.sum())
    savings = float(baseline_cost_eur - cost)
    pv = float(results.pv_generation_kwh.sum())
    captured = float((results.pv_used_directly_kwh + results.pv_to_battery_kwh).sum())
    throughput = float(results.battery_throughput_kwh.sum())
    return {
        "total_electricity_cost_eur": cost,
        "cost_savings_eur": savings,
        "cost_savings_pct": 100 * savings / baseline_cost_eur if baseline_cost_eur != 0 else 0.0,
        "total_grid_energy_purchased_kwh": float(results.grid_import_kwh.sum()),
        "pv_self_consumption_pct": 100 * captured / pv if pv > 0 else 0.0,
        "battery_throughput_kwh": throughput,
        "equivalent_full_cycles": throughput / (2 * battery_capacity_kwh),
        "peak_grid_import_kwh_per_hour": float(results.grid_import_kwh.max()),
        "initial_soc_kwh": float(results.soc_before_kwh.iloc[0]),
        "final_soc_kwh": float(results.soc_kwh.iloc[-1]),
        "net_soc_change_kwh": float(results.soc_kwh.iloc[-1] - results.soc_before_kwh.iloc[0]),
        "pv_to_battery_kwh": float(results.pv_to_battery_kwh.sum()),
        "grid_to_battery_kwh": float(results.grid_to_battery_kwh.sum()),
        "unused_discharge_kwh": float(results.unused_discharge_kwh.sum()),
        "pv_curtailed_kwh": float(results.pv_curtailed_kwh.sum()),
        "battery_losses_kwh": float(results.battery_losses_kwh.sum()),
        "degradation_penalty_eur": float(results.degradation_penalty_eur.sum()),
    }


def validate_results(results: pd.DataFrame, config: BatteryConfig,
                     has_battery: bool, penalty_eur_per_kwh: float = 0.01):
    """Assert bounds, energy conservation, costs and continuity at 1e-8 kWh.

    These checks use explicit AssertionError so they remain active with python -O.
    """
    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def close(left, right):
        np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-10)

    require(not results.empty, "Results must not be empty.")
    require(not results.isna().any().any(), "Results contain NaN.")
    require(np.isfinite(results.select_dtypes(include="number").to_numpy()).all(),
            "Results contain infinite values.")
    nonnegative = [c for c in results if c.endswith("_kwh") and c != "electricity_price_eur_kwh"]
    require((results[nonnegative].to_numpy() >= -1e-8).all(), "Negative energy flow or SOC.")
    times = pd.to_datetime(results.timestamp)
    require(times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(), "Nonhourly results.")
    require(results.action.isin([-1, 0, 1]).all(), "Unknown action.")
    c, d = results.charge_from_bus_kwh, results.discharge_to_bus_kwh
    close(c * d, 0)
    close(c[results.action != 1], 0)
    close(d[results.action != -1], 0)
    close(results.pv_used_directly_kwh, np.minimum(results.demand_kwh, results.pv_generation_kwh))
    close(results.pv_to_battery_kwh, np.minimum(
        (results.pv_generation_kwh - results.demand_kwh).clip(lower=0), c))
    close(c, results.pv_to_battery_kwh + results.grid_to_battery_kwh)
    close(d, results.discharge_to_load_kwh + results.unused_discharge_kwh)
    close(results.unused_discharge_kwh, 0)
    require((d <= (results.demand_kwh - results.pv_generation_kwh).clip(lower=0) + 1e-8).all(),
            "Discharge exceeds residual load.")
    close(results.discharge_to_load_kwh, np.minimum(
        d, (results.demand_kwh - results.pv_generation_kwh).clip(lower=0)))
    close(results.pv_generation_kwh,
          results.pv_used_directly_kwh + results.pv_to_battery_kwh + results.pv_curtailed_kwh)
    close(results.grid_import_kwh,
          results.demand_kwh - results.pv_used_directly_kwh
          - results.discharge_to_load_kwh + results.grid_to_battery_kwh)
    close(results.pv_generation_kwh + results.grid_import_kwh + d,
          results.demand_kwh + c + results.pv_curtailed_kwh + results.unused_discharge_kwh)
    close(results.grid_cost_eur, results.grid_import_kwh * results.electricity_price_eur_kwh)
    close(results.battery_throughput_kwh, c + d)
    close(results.degradation_penalty_eur, (c + d) * penalty_eur_per_kwh)
    close(results.reward, -(results.grid_cost_eur + results.degradation_penalty_eur))
    if has_battery:
        for soc in (results.soc_before_kwh, results.soc_kwh):
            require((soc >= config.min_soc_kwh - 1e-8).all(), "SOC below minimum.")
            require((soc <= config.max_soc_kwh + 1e-8).all(), "SOC above maximum.")
        require((c <= config.max_charge_power_kw * config.timestep_hours + 1e-8).all(),
                "Charge exceeds power limit.")
        require((d <= config.max_discharge_power_kw * config.timestep_hours + 1e-8).all(),
                "Discharge exceeds power limit.")
        close(results.soc_kwh - results.soc_before_kwh,
              c * config.charge_efficiency - d / config.discharge_efficiency)
        close(results.soc_before_kwh.iloc[0], config.initial_soc_kwh)
        close(results.soc_before_kwh.iloc[1:].to_numpy(), results.soc_kwh.iloc[:-1].to_numpy())
        close(results.soc_fraction, results.soc_kwh / config.capacity_kwh)
        close(results.battery_losses_kwh,
              c * (1 - config.charge_efficiency) + d * (1 / config.discharge_efficiency - 1))
    else:
        close(c + d + results.soc_kwh + results.soc_before_kwh + results.soc_fraction, 0)
        close(results.battery_losses_kwh, 0)
