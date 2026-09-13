"""Training-only terminal inventory valuation, separate from the cash grid bill.

Shortfall is priced at the grid input required to replenish it (delta / eta_c).
Excess is credited as future avoided load (excess * eta_d), not export revenue.
The asymmetric convention includes losses and does not invent a terminal trade.
"""
from dataclasses import dataclass
import numpy as np

from .data_utils import validate_hourly_data
from .metrics import calculate_metrics


@dataclass(frozen=True)
class TerminalValuation:
    reference_price_eur_kwh: float
    training_start: str
    training_end: str

    @classmethod
    def fit(cls, training_data):
        training = validate_hourly_data(training_data)
        return cls(float(training.electricity_price_eur_kwh.median()),
                   str(training.timestamp.iloc[0]), str(training.timestamp.iloc[-1]))

    def adjustment(self, initial_soc_kwh, final_soc_kwh, config):
        if not np.isfinite([self.reference_price_eur_kwh, initial_soc_kwh, final_soc_kwh]).all():
            raise ValueError("Terminal valuation inputs must be finite.")
        if not 0 < config.charge_efficiency <= 1 or not 0 < config.discharge_efficiency <= 1:
            raise ValueError("Terminal valuation requires efficiencies in (0, 1].")
        delta = initial_soc_kwh - final_soc_kwh
        bus_equivalent = delta / config.charge_efficiency if delta >= 0 else delta * config.discharge_efficiency
        return float(bus_equivalent * self.reference_price_eur_kwh)


def calculate_refined_metrics(results, baseline_cost_eur, config, valuation):
    """Stage-3 metrics plus explicit raw/adjusted cost, PV surplus and EFC/day.

    No Battery has zero initial/final SOC, so its adjusted cost equals raw cost.
    Reference price must be supplied explicitly; no implicit zero-value default.
    Total reward remains raw grid cost plus the original throughput penalty.
    """
    metrics = calculate_metrics(results, baseline_cost_eur, config.capacity_kwh)
    correction = valuation.adjustment(metrics["initial_soc_kwh"], metrics["final_soc_kwh"], config)
    raw = metrics["total_electricity_cost_eur"]
    adjusted = raw + correction
    days = len(results) * config.timestep_hours / 24
    metrics.update({
        "raw_electricity_cost_eur": raw,
        "terminal_reference_price_eur_kwh": valuation.reference_price_eur_kwh,
        "terminal_soc_adjustment_eur": correction,
        "terminal_soc_adjusted_cost_eur": adjusted,
        "adjusted_cost_savings_eur": baseline_cost_eur - adjusted,
        "adjusted_cost_savings_pct": 100 * (baseline_cost_eur - adjusted) / baseline_cost_eur if baseline_cost_eur else 0.,
        "pv_generation_kwh": float(results.pv_generation_kwh.sum()),
        "pv_surplus_kwh": float((results.pv_generation_kwh - results.demand_kwh).clip(lower=0).sum()),
        "pv_surplus_hours": int((results.pv_generation_kwh > results.demand_kwh).sum()),
        "evaluation_days": days,
        "efc_per_day": metrics["equivalent_full_cycles"] / days,
        "total_rl_reward_eur": float(results.reward.sum()),
        "cost_plus_penalty_eur": raw + metrics["degradation_penalty_eur"],
        "adjusted_cost_plus_penalty_eur": adjusted + metrics["degradation_penalty_eur"],
    })
    return metrics
