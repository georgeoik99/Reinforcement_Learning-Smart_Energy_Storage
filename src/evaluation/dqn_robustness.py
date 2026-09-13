"""Build the three-seed DQN robustness table without selecting a seed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 7, 123)
METRICS = (
    'terminal_soc_adjusted_cost_eur', 'adjusted_cost_savings_eur',
    'adjusted_cost_savings_pct', 'total_grid_energy_purchased_kwh',
    'pv_self_consumption_pct', 'pv_to_battery_kwh',
    'pv_surplus_captured_pct', 'efc_per_day',
    'peak_grid_import_kwh_per_hour', 'final_soc_kwh', 'total_rl_reward_eur',
)

def seed_row(seed, metrics, behavior):
    row = {'row_type': 'seed', 'seed_or_statistic': str(seed), 'seed': seed}
    for key in METRICS:
        source = behavior if key == 'pv_surplus_captured_pct' else metrics
        value = source[key]
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite robustness metric: {key}')
        row[key] = float(value)
    return row

def summarize_seed_rows(seed_rows):
    frame = pd.DataFrame(seed_rows)
    if set(frame.seed.astype(int)) != set(SEEDS) or len(frame) != len(SEEDS):
        raise ValueError('Robustness summary requires exactly seeds 42, 7 and 123')
    rows = list(seed_rows)
    for statistic in ('mean', 'std', 'min', 'max'):
        row = {'row_type': 'summary', 'seed_or_statistic': statistic, 'seed': np.nan}
        for key in METRICS:
            values = frame[key].astype(float)
            if statistic == 'mean': value = values.mean()
            elif statistic == 'std': value = values.std(ddof=1)
            elif statistic == 'min': value = values.min()
            else: value = values.max()
            row[key] = float(value)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not np.isfinite(result[list(METRICS)].to_numpy()).all():
        raise AssertionError('Nonfinite robustness summary')
    return result

def load_seed_42():
    results = ROOT / 'outputs/results'
    metrics = pd.read_csv(results / 'stage5_final_comparison.csv')
    metrics = metrics.loc[metrics.Strategy == 'DQN'].iloc[0].to_dict()
    behavior = json.loads((results / 'dqn_test_behaviour.json').read_text())
    return seed_row(42, metrics, behavior)

def load_additional_seed(seed):
    folder = ROOT / 'outputs/results/dqn_seed_runs'
    metrics = json.loads((folder / f'dqn_seed_{seed}_test_metrics.json').read_text())
    behavior = json.loads((folder / f'dqn_seed_{seed}_test_behaviour.json').read_text())
    return seed_row(seed, metrics, behavior)

def main():
    rows = [load_seed_42(), load_additional_seed(7), load_additional_seed(123)]
    result = summarize_seed_rows(rows)
    path = ROOT / 'outputs/results/dqn_seed_robustness.csv'
    result.to_csv(path, index=False)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

if __name__ == '__main__':
    main()
