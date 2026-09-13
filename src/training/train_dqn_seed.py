"""Train one missing robustness seed with the frozen Stage-5 configuration.

This script accepts only seeds 7 and 123. It writes isolated episode checkpoints,
resumes valid incomplete work, and evaluates fixed episode 100 on 2024 and 2025.
It never selects the best seed or checkpoint.
"""
import argparse, hashlib, json, os, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.agents.state_normalizer import StateNormalizer
from src.evaluation.data_utils import chronological_split, load_battery_config
from src.evaluation.evaluate_dqn import evaluate_dqn, dqn_behaviour
from src.evaluation.terminal_valuation import TerminalValuation, calculate_refined_metrics
from src.training.train_dqn import train_episode

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_equivalent(value):
    return json.loads(json.dumps(value))

def config_for_seed(seed):
    baseline = json.loads((ROOT / 'outputs/models/dqn_config.json').read_text())['hyperparameters']
    baseline['hidden_layers'] = tuple(baseline['hidden_layers'])
    return replace(DQNConfig(**baseline), seed=seed)

def metric_dict(result, training, battery, comparison_filename):
    reference = float(pd.read_csv(ROOT / 'outputs/results' / comparison_filename)
                      .loc[lambda x: x.Strategy == 'No Battery', 'raw_electricity_cost_eur'].iloc[0])
    return calculate_refined_metrics(result, reference, battery, TerminalValuation.fit(training))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(7, 123))
    args = parser.parse_args()
    seed = args.seed
    model_dir = ROOT / 'outputs/models/dqn_seed_runs'
    result_dir = ROOT / 'outputs/results/dqn_seed_runs'
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'dqn_seed_{seed}'
    checkpoint_path = model_dir / f'{prefix}_checkpoint.pt'
    model_path = model_dir / f'{prefix}_model.pt'
    metadata_path = model_dir / f'{prefix}_config.json'
    history_path = result_dir / f'{prefix}_training_history.csv'
    config = config_for_seed(seed)
    source_paths = [
        'data/processed/energy_hourly_2023_2025.csv', 'config/battery_config.json',
        'src/environment/battery_model.py', 'src/environment/energy_storage_env.py',
        'src/agents/dqn_agent.py', 'src/agents/replay_buffer.py',
        'src/agents/state_normalizer.py', 'src/training/train_dqn.py',
        'src/training/train_dqn_seed.py', 'src/evaluation/evaluate_dqn.py',
        'outputs/models/dqn_state_normalizer.json',
    ]
    provenance = {path: digest(ROOT / path) for path in source_paths}
    if model_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get('complete') and existing.get('provenance') == provenance:
            print(f'Seed {seed} already complete; refusing redundant training.')
            return
    dataset = pd.read_csv(ROOT / 'data/processed/energy_hourly_2023_2025.csv',
                          parse_dates=['timestamp'])
    training, validation, test = chronological_split(dataset)
    battery, penalty = load_battery_config(ROOT / 'config/battery_config.json')
    normalizer = StateNormalizer.load(ROOT / 'outputs/models/dqn_state_normalizer.json')
    if normalizer != StateNormalizer.fit(training):
        raise AssertionError('Saved normalizer differs from a 2023-only fit')
    agent = DQNAgent(config)
    history = []
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if (json_equivalent(saved['config']) != json_equivalent(asdict(config))
                or saved['provenance'] != provenance):
            raise ValueError('Seed checkpoint configuration or provenance is invalid')
        history = saved['history']
        if len(history) > config.episodes:
            raise ValueError('Seed checkpoint contains too many episodes')
        agent.restore_training(saved['agent'])
        if agent.steps != len(history) * len(training):
            raise ValueError('Seed checkpoint step/episode count is inconsistent')
        print(f'Resuming seed {seed} after episode {len(history)}.', flush=True)
    started = time.monotonic()
    for _ in range(len(history), config.episodes):
        summary = train_episode(agent, normalizer, training, battery, penalty)
        history.append(summary)
        payload = {'config': asdict(config), 'provenance': provenance,
                   'agent': agent.training_state(), 'history': history}
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"seed={seed} episode={summary['episode']}/{config.episodes} "
              f"reward={summary['cumulative_reward']:.2f} "
              f"loss={summary['mean_training_loss']:.5f} "
              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    agent.save(model_path)
    validation_result = evaluate_dqn(validation, agent, normalizer, battery, penalty)
    test_result = evaluate_dqn(test, agent, normalizer, battery, penalty)
    restored = DQNAgent.load(model_path)
    pd.testing.assert_frame_equal(
        validation_result,
        evaluate_dqn(validation, restored, normalizer, battery, penalty),
        check_exact=True)
    pd.testing.assert_frame_equal(
        test_result,
        evaluate_dqn(test, restored, normalizer, battery, penalty),
        check_exact=True)
    validation_metrics = metric_dict(validation_result, training, battery,
                                     'q_learning_validation_comparison.csv')
    test_metrics = metric_dict(test_result, training, battery,
                               'q_learning_comparison.csv')
    for period, result, metrics in (
        ('validation', validation_result, validation_metrics),
        ('test', test_result, test_metrics),
    ):
        (result_dir / f'{prefix}_{period}_metrics.json').write_text(
            json.dumps(metrics, indent=2) + '\n')
        (result_dir / f'{prefix}_{period}_behaviour.json').write_text(
            json.dumps(dqn_behaviour(result, training, battery), indent=2) + '\n')
    metadata = {
        'seed': seed, 'complete': True, 'episodes': config.episodes,
        'training_steps': agent.steps, 'gradient_updates': agent.updates,
        'target_syncs': agent.target_syncs, 'configuration': asdict(config),
        'configuration_difference_from_seed_42': {'seed': [42, seed]},
        'model_selection': 'Fixed episode 100; no best-seed or checkpoint selection',
        'validation_year': 2024, 'test_year': 2025, 'test_tuning': False,
        'frozen_reload_exact': True, 'provenance': provenance,
        'model_sha256': digest(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Seed {seed} complete: adjusted test savings '
          f"{test_metrics['adjusted_cost_savings_pct']:.4f}%", flush=True)

if __name__ == '__main__':
    main()
