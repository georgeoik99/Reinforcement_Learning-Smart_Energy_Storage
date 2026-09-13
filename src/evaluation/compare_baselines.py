"""Run from the project root: python -m src.evaluation.compare_baselines."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.agents.rule_based_controller import RuleBasedController, run_rule_based_controller
from .data_utils import chronological_split, load_battery_config
from .metrics import validate_results
from .terminal_valuation import TerminalValuation, calculate_refined_metrics
from .no_battery_baseline import run_no_battery_baseline


ROOT = Path(__file__).resolve().parents[2]


def evaluate_baselines(data, config, penalty_eur_per_kwh=0.01):
    """Fit on the calendar year 2023; evaluate both strategies on the identical calendar year 2025.

    No validation/test values enter fitting. Validation is left unused. Each
    test run resets storage once to configured initial SOC, with no warm-up,
    terminal liquidation or forced end-of-test charge. Inventory valuation is a separate KPI.
    """
    training, validation, test = chronological_split(data)
    controller = RuleBasedController.fit(training)
    no_battery = run_no_battery_baseline(test)
    rule_based = run_rule_based_controller(test, controller, config, penalty_eur_per_kwh)
    pd.testing.assert_series_equal(no_battery.timestamp, rule_based.timestamp)
    for column in ("demand_kwh", "pv_generation_kwh", "electricity_price_eur_kwh"):
        np.testing.assert_array_equal(no_battery[column], rule_based[column])
    validate_results(no_battery, config, False, penalty_eur_per_kwh)
    validate_results(rule_based, config, True, penalty_eur_per_kwh)
    valuation = TerminalValuation.fit(training)
    baseline_cost = float(no_battery.grid_cost_eur.sum())
    metrics = pd.DataFrame([
        {"Strategy": name, **calculate_refined_metrics(frame, baseline_cost, config, valuation)}
        for name, frame in (("No Battery", no_battery), ("Rule-Based", rule_based))
    ])
    comparison = metrics[[
        "Strategy", "total_electricity_cost_eur", "cost_savings_pct",
        "terminal_soc_adjusted_cost_eur", "adjusted_cost_savings_pct",
        "total_grid_energy_purchased_kwh", "pv_self_consumption_pct", "equivalent_full_cycles", "efc_per_day"
    ]].rename(columns={
        "terminal_soc_adjusted_cost_eur": "Adjusted Cost €", "adjusted_cost_savings_pct": "Adjusted Savings %",
        "efc_per_day": "EFC/day", "total_electricity_cost_eur": "Cost €", "cost_savings_pct": "Savings %",
        "total_grid_energy_purchased_kwh": "Grid Import kWh",
        "pv_self_consumption_pct": "PV Self-Consumption %", "equivalent_full_cycles": "EFC"
    })
    splits = {
        name: {"rows": len(frame), "start": str(frame.timestamp.iloc[0]),
               "end": str(frame.timestamp.iloc[-1])}
        for name, frame in (("training", training), ("validation", validation), ("test", test))
    }
    return no_battery, rule_based, comparison, metrics, controller, splits


def select_example_week(results):
    """First complete Monday 00:00 to Sunday 23:00 wholly within test."""
    times = pd.to_datetime(results.timestamp)
    candidates = times[(times.dt.dayofweek == 0) & (times.dt.hour == 0)]
    for start in candidates:
        week = results.loc[(times >= start) & (times < start + pd.Timedelta(days=7))]
        if len(week) == 168:
            return week.copy().reset_index(drop=True)
    raise ValueError("Test period must contain a full Monday-Sunday week for plots.")


def make_plots(no_battery, rule_based, config, figure_dir):
    """Three reproducible portfolio figures; no performance-based selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure_dir.mkdir(parents=True, exist_ok=True)
    week = select_example_week(rule_based)
    start = week.timestamp.iloc[0]
    end = week.timestamp.iloc[-1] + pd.Timedelta(hours=1)
    navy, teal, amber = "#263F58", "#007F78", "#B97917"
    style = {
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 17,
        "axes.titleweight": "bold", "axes.labelcolor": "#334155",
        "text.color": "#24364B", "xtick.color": "#526174", "ytick.color": "#526174",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.grid": True, "grid.color": "#E4E9EE", "grid.linewidth": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 180,
    }

    def finish(fig, ax, filename, note, left=0.095):
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=8)
        fig.subplots_adjust(left=left, right=0.96, top=0.80, bottom=0.23)
        fig.text(left, 0.065, note, fontsize=9, color="#66758A", va="bottom")
        fig.savefig(figure_dir / filename, facecolor="white")
        plt.close(fig)

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        for frame, name, color in ((no_battery, "No Battery", navy),
                                    (rule_based, "Rule-Based", teal)):
            # Cost booked at the END of each hour, beginning at zero.
            x = [frame.timestamp.iloc[0], *(frame.timestamp + pd.Timedelta(hours=1))]
            y = [0.0, *frame.grid_cost_eur.cumsum()]
            ax.plot(x, y, label=f"{name}  ·  €{y[-1]:,.2f}", color=color, linewidth=2.2)
        savings = no_battery.grid_cost_eur.sum() - rule_based.grid_cost_eur.sum()
        pct = savings / no_battery.grid_cost_eur.sum() * 100 if no_battery.grid_cost_eur.sum() else 0
        fig.suptitle("Electricity purchasing cost", x=0.095, ha="left", y=0.97,
                     fontsize=18, fontweight="bold")
        ax.set_title(f"Test period · Rule-Based savings: €{savings:,.2f} ({pct:.2f}%)",
                     loc="left", fontsize=11, fontweight="normal", pad=16)
        ax.set_ylabel("Cumulative cost (€)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(loc="upper left", frameon=False)
        ax.set_ylim(bottom=0)
        finish(fig, ax, "stage3_cumulative_electricity_cost.png",
               "Synthetic full-year 2025 test · Identical held-out test hours · Energy charges only\n"
               "Raw bill shown; terminal-SOC-adjusted costs are reported separately.")

        period = f"{start:%d %b}–{end - pd.Timedelta(days=1):%d %b %Y}"
        fig, ax = plt.subplots(figsize=(11.5, 5.0))
        x = [start, *(week.timestamp + pd.Timedelta(hours=1))]
        y = [week.soc_before_kwh.iloc[0] / config.capacity_kwh * 100,
             *(week.soc_fraction * 100)]
        ax.step(x, y, where="post", color=teal, linewidth=1.8)
        ax.axhline(config.min_soc_fraction * 100, color=amber, linestyle="--", linewidth=1)
        ax.axhline(config.max_soc_fraction * 100, color=amber, linestyle="--", linewidth=1,
                   label=f"SOC limits: {config.min_soc_fraction:.0%}–{config.max_soc_fraction:.0%}")
        ax.set_title("Battery state of charge", loc="left", pad=35)
        ax.text(0, 1.035, f"Rule-Based · {period}", transform=ax.transAxes, fontsize=11)
        ax.set_ylabel("State of charge (%)")
        ax.set_ylim(0, 103)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        finish(fig, ax, "stage3_battery_soc_example_week.png",
               "First complete Monday–Sunday week in the test period; no reset at the week boundary.\n"
               "SOC shown at hourly boundaries, with each transition booked at the end of the hour.")

        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        edges = [*week.timestamp, end]
        ax.stairs(week.action.to_numpy(), mdates.date2num(edges), baseline=None,
                  color=navy, linewidth=1.1)
        for action, color in ((-1, navy), (0, "#A8B1BC"), (1, teal)):
            selected = week.action == action
            ax.scatter(week.timestamp[selected] + pd.Timedelta(minutes=30),
                       week.action[selected], color=color, s=13, zorder=3)
        ax.set_yticks([-1, 0, 1], ["Discharge (−1)", "Hold (0)", "Charge (+1)"])
        ax.set_ylim(-1.45, 1.45)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        ax.set_title("Hourly battery decisions", loc="left", pad=35)
        ax.text(0, 1.035, f"Rule-Based · {period}", transform=ax.transAxes, fontsize=11)
        finish(fig, ax, "stage3_controller_actions_example_week.png",
               "Same fixed example week as the SOC chart · One decision per hour\n"
               "Priority: surplus PV → charge; low price → charge; high price + residual demand → discharge.",
               left=0.15)
    return {"selection_rule": "First complete Monday-Sunday week wholly inside test",
            "start": str(start), "end_exclusive": str(end), "rows": len(week)}


def main():
    data_path = ROOT / "data/processed/energy_hourly_2023_2025.csv"
    config_path = ROOT / "config/battery_config.json"
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    config, penalty = load_battery_config(config_path)
    no_battery, rule_based, comparison, metrics, controller, splits = evaluate_baselines(
        data, config, penalty
    )
    output = ROOT / "outputs"
    example = select_example_week(rule_based.loc[rule_based.timestamp.dt.month == 6])
    week = {"selection_rule": "First full Monday-Sunday week in June",
            "start": str(example.timestamp.iloc[0]),
            "end_exclusive": str(example.timestamp.iloc[-1] + pd.Timedelta(hours=1))}
    # Stage-4.5 figures are generated together by evaluate_q_learning to avoid duplication.
    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("no_battery_test_results.csv", no_battery),
        ("rule_based_test_results.csv", rule_based),
        ("baseline_comparison.csv", comparison),
        ("baseline_metrics.csv", metrics),
    ):
        frame.to_csv(result_dir / name, index=False, encoding="utf-8-sig")
    thresholds = asdict(controller.thresholds)
    thresholds["training_start"] = str(thresholds["training_start"])
    thresholds["training_end"] = str(thresholds["training_end"])
    import matplotlib
    metadata = {
        "stage": "4.5",
        "dataset": "data/processed/energy_hourly_2023_2025.csv",
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (data_path, config_path, ROOT / "src/environment/battery_model.py",
                         ROOT / "src/environment/energy_storage_env.py")
        },
        "split_rule": "2023=training; 2024=validation; 2025=test",
        "splits": splits,
        "thresholds": thresholds,
        "terminal_reference_price_eur_kwh": TerminalValuation.fit(chronological_split(data)[0]).reference_price_eur_kwh,
        "quantile_interpolation": "linear",
        "validation_used_for_tuning": False,
        "current_hour_observations_available_before_dispatch": True,
        "battery_config": asdict(config),
        "throughput_penalty_eur_per_kwh": penalty,
        "terminal_soc_treatment": "Raw and adjusted KPIs; training-median replenishment cost / avoided-load credit",
        "initial_soc_treatment": "Reset once to configured initial SOC; no purchase charge for initial inventory",
        "dispatch_accounting": "Residual-load-capped discharge; no export or unused discharge",
        "example_week": week,
        "hourly_accounting_assertions": "passed for both test strategies",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    (result_dir / "stage3_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Stage 3: identical held-out TEST period; all hourly assertions passed.")
    for name, split in splits.items():
        print(f"{name:10s}: {split['rows']} rows | {split['start']} -> {split['end']}")
    print(f"Training-only price thresholds: P25={controller.thresholds.low_eur_kwh:.8f}, "
          f"P75={controller.thresholds.high_eur_kwh:.8f} EUR/kWh")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"Rule-Based SOC: {rule_based.soc_before_kwh.iloc[0]:.2f} -> "
          f"{rule_based.soc_kwh.iloc[-1]:.2f} kWh; raw and terminal-adjusted KPIs reported separately.")
    print(f"Example week: {week['start']} -> {week['end_exclusive']} (exclusive)")
    print(f"Results: {result_dir}; run Q-Learning evaluation for the common figures.")


if __name__ == "__main__":
    main()
