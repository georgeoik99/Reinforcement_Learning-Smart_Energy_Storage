"""Create the authoritative Stage-6 comparison and curated portfolio figures."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from .compare_baselines import select_example_week

ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = ('No Battery', 'Rule-Based', 'Q-Learning', 'DQN')
FINAL_COLUMNS = (
    'Strategy', 'raw_electricity_cost_eur', 'terminal_soc_adjusted_cost_eur',
    'adjusted_cost_savings_eur', 'adjusted_cost_savings_pct',
    'total_grid_energy_purchased_kwh', 'pv_self_consumption_pct',
    'pv_surplus_captured_pct', 'pv_curtailed_kwh', 'pv_to_battery_kwh',
    'grid_to_battery_kwh', 'battery_throughput_kwh',
    'equivalent_full_cycles', 'efc_per_day',
    'peak_grid_import_kwh_per_hour', 'initial_soc_kwh', 'final_soc_kwh',
    'degradation_penalty_eur', 'total_rl_reward_eur',
)

def build_final_comparison(stage5):
    if tuple(stage5.Strategy) != STRATEGIES:
        raise ValueError('Expected the authoritative seed-42 Stage-5 strategy order')
    result = stage5.copy()
    surplus = result.pv_surplus_kwh.astype(float)
    result['pv_surplus_captured_pct'] = np.where(
        surplus > 0, 100 * result.pv_to_battery_kwh / surplus, 0.)
    result = result[list(FINAL_COLUMNS)]
    if result.isna().any().any() or not np.isfinite(result.select_dtypes('number').to_numpy()).all():
        raise AssertionError('Invalid final comparison')
    return result

def make_figures(comparison, robustness, history, dqn_test, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    destination.mkdir(parents=True, exist_ok=True)
    existing = ROOT / 'outputs/figures'
    for source, target in (
        ('stage4_5_demand_pv_week.png', '01_energy_profile.png'),
        ('stage5_test_cumulative_cost.png', '02_cumulative_cost_comparison.png'),
        ('stage5_pv_capture_week.png', '04_pv_surplus_capture.png'),
    ):
        shutil.copyfile(existing / source, destination / target)

    colors = {'No Battery': '#52677D', 'Rule-Based': '#00867D',
              'Q-Learning': '#B77723', 'DQN': '#6B4BA5'}
    palette = [colors[name] for name in STRATEGIES]
    week = select_example_week(dqn_test.loc[dqn_test.timestamp.dt.month == 6])
    start, end = week.timestamp.iloc[0], week.timestamp.iloc[-1] + pd.Timedelta(hours=1)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18,
                         'axes.axisbelow': True, 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 height_ratios=[1.4, 1])
        axes[0].set_title('DQN dispatch during a representative summer week',
                          loc='left', fontsize=16, fontweight='bold', pad=14)
        axes[0].step([start, *(week.timestamp + pd.Timedelta(hours=1))],
                     [week.soc_before_kwh.iloc[0], *(week.soc_fraction * 100)],
                     where='post', color=colors['DQN'], lw=1.7)
        axes[0].axhline(10, color='#52677D', ls='--', lw=1)
        axes[0].axhline(95, color='#52677D', ls='--', lw=1)
        axes[0].set_ylabel('Battery SOC (%)'); axes[0].set_ylim(0, 100)
        axes[1].step([*week.timestamp, end], [*week.action, week.action.iloc[-1]],
                     where='post', color=colors['DQN'], lw=1.5)
        axes[1].set_yticks([-1, 0, 1], ['Discharge', 'Hold', 'Charge'])
        axes[1].set_ylabel('Requested action'); axes[1].set_ylim(-1.3, 1.3)
        axes[1].set_xlim(start, end); axes[1].xaxis.set_major_locator(mdates.DayLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
        fig.subplots_adjust(left=.13, right=.97, top=.88, bottom=.15, hspace=.16)
        fig.text(.13, .035, '2–8 June 2025 · Calendar-selected week · No weekly battery reset',
                 color='#526174', fontsize=9)
        fig.savefig(destination / '03_dqn_soc_actions.png', dpi=180, facecolor='white')
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        specs = [
            ('adjusted_cost_savings_pct', 'Adjusted electricity-cost savings', '%', True),
            ('pv_self_consumption_pct', 'PV self-consumption', '%', False),
            ('efc_per_day', 'Battery cycling intensity', 'EFC / day', False),
            ('peak_grid_import_kwh_per_hour', 'Peak hourly grid import', 'kW', False),
        ]
        x = np.arange(4)
        for ax, (column, title, unit, zero) in zip(axes.flat, specs):
            values = comparison[column].to_numpy(float)
            bars = ax.bar(x, values, color=palette, width=.68)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.set_ylabel(unit); ax.set_xticks(x, STRATEGIES, rotation=12, ha='right')
            if zero: ax.axhline(0, color='#526174', lw=1)
            if column == 'pv_self_consumption_pct': ax.set_ylim(88, 100)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        fig.suptitle('Economic gains come with operational trade-offs · 2025',
                     x=.08, ha='left', fontsize=17, fontweight='bold')
        fig.subplots_adjust(left=.09, right=.97, top=.87, bottom=.14,
                            hspace=.45, wspace=.28)
        fig.savefig(destination / '05_final_kpi_comparison.png', dpi=180,
                    facecolor='white')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(history.episode, history.cumulative_reward,
                     color='#BCAED4', lw=1, label='Episode')
        axes[0].plot(history.episode,
                     history.cumulative_reward.rolling(5, min_periods=1).mean(),
                     color=colors['DQN'], lw=2, label='5-episode mean')
        axes[0].set_title('Training reward', loc='left', fontweight='bold')
        axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward (€)')
        axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[0].legend(frameon=False)
        axes[1].plot(history.episode, history.mean_training_loss,
                     color='#BCAED4', lw=1, label='Episode mean')
        axes[1].plot(history.episode,
                     history.mean_training_loss.rolling(5, min_periods=1).mean(),
                     color=colors['DQN'], lw=2, label='5-episode mean')
        axes[1].set_title('Minibatch Huber loss', loc='left', fontweight='bold')
        axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Loss')
        axes[1].legend(frameon=False)
        fig.suptitle('DQN learning progression · seed 42 · 2023',
                     x=.08, ha='left', fontsize=17, fontweight='bold')
        fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.18, wspace=.25)
        fig.text(.08, .035, '100 fixed episodes · Test data not used for training or selection',
                 color='#526174', fontsize=9)
        fig.savefig(destination / '06_dqn_training.png', dpi=180, facecolor='white')
        plt.close(fig)

        seeds = robustness.loc[robustness.row_type == 'seed'].copy()
        mean = float(robustness.loc[robustness.seed_or_statistic == 'mean',
                                    'adjusted_cost_savings_pct'].iloc[0])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        values = seeds.adjusted_cost_savings_pct.to_numpy(float)
        bars = ax.bar(seeds.seed.astype(int).astype(str), values,
                      color=['#6B4BA5', '#8B72BB', '#B2A2D0'], width=.62)
        ax.axhline(mean, color='#00867D', ls='--', lw=1.6,
                   label=f'Three-seed mean: {mean:.2f}%')
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value, f'{value:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        ax.set_title('DQN adjusted savings vary by random seed', loc='left',
                     fontsize=16, fontweight='bold', pad=14)
        ax.set_xlabel('Training seed'); ax.set_ylabel('Adjusted savings vs No Battery (%)')
        ax.legend(frameon=False)
        fig.subplots_adjust(left=.13, right=.96, top=.84, bottom=.18)
        fig.text(.13, .035, 'Same 7–64–64–3 DQN and 100-episode schedule · Lightweight robustness check',
                 color='#526174', fontsize=9)
        fig.savefig(destination / '07_dqn_seed_robustness.png', dpi=180,
                    facecolor='white')
        plt.close(fig)

def main():
    results = ROOT / 'outputs/results'
    comparison = build_final_comparison(pd.read_csv(results / 'stage5_final_comparison.csv'))
    comparison.to_csv(results / 'final_model_comparison.csv', index=False)
    robustness = pd.read_csv(results / 'dqn_seed_robustness.csv')
    history = pd.read_csv(results / 'dqn_training_history.csv')
    dqn_test = pd.read_csv(results / 'dqn_test_results.csv', parse_dates=['timestamp'])
    make_figures(comparison, robustness, history, dqn_test,
                 ROOT / 'outputs/portfolio_figures')
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

if __name__ == '__main__':
    main()
