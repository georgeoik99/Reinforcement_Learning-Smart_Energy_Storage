"""Seven focused Stage-4.5 figures; week chosen by calendar, not performance."""
import pandas as pd
from .compare_baselines import select_example_week


def make_q_learning_plots(history, frames, config, destination, valuation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    destination.mkdir(parents=True, exist_ok=True)
    colors = {"No Battery": "#263F58", "Rule-Based": "#007F78", "Q-Learning": "#B66D14"}
    q = frames["Q-Learning"]
    # First full Monday-Sunday week in June, fixed before examining performance.
    week = select_example_week(q.loc[q.timestamp.dt.month == 6])
    start, end = week.timestamp.iloc[0], week.timestamp.iloc[-1] + pd.Timedelta(hours=1)
    note = (f"First complete June week: {start:%d %b}–{end - pd.Timedelta(days=1):%d %b %Y}\n"
            "Calendar-selected summer example · No weekly battery reset · Synthetic 100 kWp PV scenario")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.left": False, "axes.spines.bottom": False,
            "axes.grid": True, "grid.alpha": .22, "axes.titleweight": "bold",
            "axes.titlesize": 17, "figure.facecolor": "white"}):
        def save(fig, ax, name, title, footer, left=.12):
            ax.set_title(title, loc="left", pad=22)
            ax.set_axisbelow(True)
            ax.tick_params(length=0, pad=8)
            fig.subplots_adjust(left=left, right=.96, top=.83, bottom=.25)
            fig.text(left, .06, footer, fontsize=9, color="#526174")
            fig.savefig(destination / name, dpi=180, facecolor="white")
            plt.close(fig)

        def format_week(ax):
            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))

        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax.plot(history.episode, history.cumulative_reward, color="#ADB8C6", lw=1,
                label="Exploratory episode reward")
        ax.plot(history.episode, history.cumulative_reward.rolling(10, min_periods=1).mean(),
                color=colors["Q-Learning"], lw=2, label="10-episode trailing mean")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Reward (€)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(frameon=False, loc="lower right")
        save(fig, ax, "stage4_training_reward.png", "Q-Learning training reward · 2023",
             "200 episodes · Higher is better · Includes unchanged €0.01/kWh throughput penalty\n"
             "Terminal valuation is a separate evaluation KPI, not a training reward change.")

        for adjusted in (False, True):
            fig, ax = plt.subplots(figsize=(11.5, 5.4))
            for name, result in frames.items():
                cost = result.grid_cost_eur.cumsum().to_numpy()
                if adjusted:
                    initial = float(result.soc_before_kwh.iloc[0])
                    cost = cost + [valuation.adjustment(initial, float(soc), config) for soc in result.soc_kwh]
                ax.plot([result.timestamp.iloc[0], *(result.timestamp + pd.Timedelta(hours=1))],
                        [0., *cost], color=colors[name], lw=2, label=f"{name} · €{cost[-1]:,.2f}")
            locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.set_ylabel("Cumulative cost (€)")
            ax.legend(frameon=False, loc="upper left")
            save(fig, ax, "stage4_5_cumulative_adjusted_cost.png" if adjusted else "stage4_test_cumulative_cost.png",
                 "SOC-adjusted economic cost · 2025" if adjusted else "Raw electricity purchasing cost · 2025",
                 ("Raw cumulative bill plus inventory adjustment at each hour-end (not accumulated repeatedly).\n"
                  f"2023 median reference: €{valuation.reference_price_eur_kwh:.5f}/kWh · Cycling penalty excluded")
                 if adjusted else "Same full-year 2025 test · Frozen policies\nRaw cash purchasing cost; terminal valuation and cycling penalty reported separately.")

        for kind in ("soc", "actions"):
            fig, ax = plt.subplots(figsize=(11.5, 5.3))
            if kind == "soc":
                ax.step([start, *(week.timestamp + pd.Timedelta(hours=1))],
                        [week.soc_before_kwh.iloc[0] / config.capacity_kwh * 100, *(week.soc_fraction * 100)],
                        where="post", color=colors["Q-Learning"], lw=1.7)
                for bound in (config.min_soc_fraction, config.max_soc_fraction):
                    ax.axhline(bound * 100, linestyle="--", color="#007F78", lw=1)
                ax.set_ylim(0, 103)
                ax.set_ylabel("Battery SOC (%)")
                title, left = "Q-Learning battery state of charge", .12
            else:
                ax.stairs(week.action, mdates.date2num([*week.timestamp, end]), baseline=None,
                          color=colors["Q-Learning"], lw=1.5)
                ax.set_yticks([-1, 0, 1], ["Discharge (−1)", "Hold (0)", "Charge (+1)"])
                ax.set_ylim(-1.4, 1.4)
                title, left = "Q-Learning hourly decisions", .16
            format_week(ax)
            save(fig, ax, f"stage4_test_{kind}_week.png", title, note, left)

        fig, ax = plt.subplots(figsize=(11.5, 5.3))
        ax.plot(week.timestamp, week.demand_kwh, color=colors["No Battery"], label="Building demand")
        ax.plot(week.timestamp, week.pv_generation_kwh, color=colors["Rule-Based"], label="PV generation")
        ax.fill_between(week.timestamp, week.demand_kwh, week.pv_generation_kwh,
                        where=week.pv_generation_kwh > week.demand_kwh, interpolate=True,
                        color="#8FC9A1", alpha=.6, label="PV surplus")
        ax.set_ylabel("Hourly energy (kWh)")
        ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
        format_week(ax)
        save(fig, ax, "stage4_5_demand_pv_week.png", "A real solar-storage opportunity in the simulation", note)

        fig, ax = plt.subplots(figsize=(11.5, 5.3))
        surplus = (week.pv_generation_kwh - week.demand_kwh).clip(lower=0)
        ax.fill_between(week.timestamp, 0, surplus, color="#A2CBBF", alpha=.6, label="Available PV surplus")
        ax.plot(week.timestamp, week.pv_to_battery_kwh, color=colors["Q-Learning"], lw=2, label="Actual PV into battery")
        ax.set_ylabel("Hourly energy (kWh)")
        ax.legend(frameon=False, loc="upper left", fontsize=10)
        format_week(ax)
        save(fig, ax, "stage4_5_pv_capture_week.png", "Q-Learning capture of surplus PV", note)
