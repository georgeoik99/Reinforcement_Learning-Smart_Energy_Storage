"""One predefined 2023 DQN run; episode-boundary checkpoints resume without repeats.

Run python -m src.training.train_dqn. A completed model is never overwritten.
The local trusted checkpoint includes optimizer/replay/RNG state, not just weights.
"""
import json,hashlib,time,os
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from src.agents.dqn_agent import DQNAgent,DQNConfig,ENV_ACTIONS
from src.agents.state_normalizer import StateNormalizer
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import chronological_split,validate_hourly_data,load_battery_config
from src.evaluation.evaluate_dqn import evaluate_dqn,dqn_behaviour,comparison_with_dqn

ROOT=Path(__file__).resolve().parents[2]

def train_episode(agent,normalizer,training,battery,penalty):
    env=EnergyStorageEnv(training,battery,penalty)
    state=normalizer.transform(env.reset())
    totals=dict(cumulative_reward=0.,electricity_cost_eur=0.,grid_import_kwh=0.,battery_throughput_kwh=0.)
    diagnostics=[];counts=[0,0,0];epsilon_start=agent.epsilon
    for _ in range(len(training)):
        action=agent.select_action(state)
        nxt,reward,done,info=env.step(ENV_ACTIONS[action])
        next_state=None if done else normalizer.transform(nxt)
        diagnostic=agent.observe(state,action,reward,next_state,done)
        if diagnostic is not None:diagnostics.append(diagnostic)
        state=next_state;counts[action]+=1
        for key,source in [('cumulative_reward','reward'),('electricity_cost_eur','grid_cost_eur'),('grid_import_kwh','grid_import_kwh'),('battery_throughput_kwh','battery_throughput_kwh')]:totals[key]+=info[source]
    summary={'episode':agent.steps//len(training),**totals,'epsilon_start':epsilon_start,'epsilon':agent.epsilon,
       'final_soc_kwh':env.battery.soc_kwh,'training_steps':agent.steps,'gradient_updates':agent.updates,
       'target_syncs':agent.target_syncs,'mean_training_loss':float(np.mean([d['loss'] for d in diagnostics])) if diagnostics else 0.,
       'max_training_loss':max((d['loss'] for d in diagnostics),default=0.),
       'max_abs_q':max((d['max_abs_q'] for d in diagnostics),default=0.),
       'max_abs_target':max((d['max_abs_target'] for d in diagnostics),default=0.),
       'max_gradient_norm_before_clip':max((d['gradient_norm'] for d in diagnostics),default=0.),
       'discharge_requests':counts[0],'hold_requests':counts[1],'charge_requests':counts[2]}
    if not np.isfinite(list(summary.values())).all():raise FloatingPointError('Nonfinite training diagnostics')
    np.testing.assert_allclose(totals['cumulative_reward'],-totals['electricity_cost_eur']-penalty*totals['battery_throughput_kwh'],atol=1e-7)
    return summary

def main():
    models=ROOT/'outputs/models';results=ROOT/'outputs/results'
    if (models/'dqn_model.pt').exists() and (models/'dqn_config.json').exists() and json.loads((models/'dqn_config.json').read_text()).get('validation_frozen_state_verified'):
        print('Completed DQN model exists: refusing redundant training. Use saved validation/test outputs.');return
    dataset=ROOT/'data/processed/energy_hourly_2023_2025.csv'
    train,validation,_=chronological_split(pd.read_csv(dataset,parse_dates=['timestamp']))
    config=DQNConfig();battery,penalty=load_battery_config(ROOT/'config/battery_config.json')
    normalizer=StateNormalizer.fit(train);agent=DQNAgent(config)
    paths=['data/processed/energy_hourly_2023_2025.csv','config/battery_config.json',
      'src/environment/energy_storage_env.py','src/environment/battery_model.py',
      'src/agents/dqn_agent.py','src/agents/replay_buffer.py','src/agents/state_normalizer.py',
      'src/training/train_dqn.py','outputs/models/q_table.csv',
      'outputs/results/q_learning_comparison.csv','outputs/results/q_learning_validation_comparison.csv']
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    metadata={'stage':5,'algorithm':'Vanilla DQN','architecture':[7,64,64,3],
      'hyperparameters':asdict(config),'loss':'Huber (smooth L1), beta=1','device':'cpu','torch_threads':1,
      'torch_version':str(torch.__version__),'numpy_version':np.__version__,'pandas_version':pd.__version__,
      'training_year':2023,'validation_year':2024,'test_year':2025,
      'model_selection':'Fixed final episode 100; one predefined configuration; no test tuning',
      'validation_review_status':'pending','provenance':provenance,'reward_changed':False,
      'battery_config':asdict(battery),'throughput_penalty_eur_per_kwh':penalty}
    checkpoint=models/'dqn_training_checkpoint.pt';history=[]
    if checkpoint.exists():
        # Only load this project's own local checkpoint. Never load an untrusted pickle.
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['provenance']!=provenance or saved['config']!=asdict(config):raise ValueError('Checkpoint inputs/configuration changed')
        agent.restore_training(saved['agent']);history=saved['history']
        print(f'Resuming after episode {len(history)}',flush=True)
    normalizer.save(models/'dqn_state_normalizer.json')
    (models/'dqn_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.monotonic()
    for _ in range(len(history),config.episodes):
        summary=train_episode(agent,normalizer,train,battery,penalty);history.append(summary)
        payload={'config':asdict(config),'provenance':provenance,'agent':agent.training_state(),'history':history}
        temp=checkpoint.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,checkpoint)
        pd.DataFrame(history).to_csv(results/'dqn_training_history.csv',index=False)
        print(f"Episode {summary['episode']}/{config.episodes}: reward={summary['cumulative_reward']:.2f} loss={summary['mean_training_loss']:.5f} eps={agent.epsilon:.3f} qmax={summary['max_abs_q']:.2f} elapsed={time.monotonic()-started:.0f}s",flush=True)
    # Final configuration fixed before test; validation diagnosis is a separate gate.
    agent.save(models/'dqn_model.pt')
    result=evaluate_dqn(validation,agent,normalizer,battery,penalty)
    restored=DQNAgent.load(models/'dqn_model.pt')
    repeated=evaluate_dqn(validation,restored,StateNormalizer.load(models/'dqn_state_normalizer.json'),battery,penalty)
    pd.testing.assert_frame_equal(result,repeated,check_exact=True)
    comparison=comparison_with_dqn(pd.read_csv(results/'q_learning_validation_comparison.csv'),result,train,battery)
    result.to_csv(results/'dqn_validation_results.csv',index=False)
    comparison.to_csv(results/'dqn_validation_comparison.csv',index=False)
    (results/'dqn_validation_behaviour.json').write_text(json.dumps(dqn_behaviour(result,train,battery),indent=2)+'\n')
    metadata.update({'training_steps':agent.steps,'gradient_updates':agent.updates,'target_syncs':agent.target_syncs,
      'model_sha256':hashlib.sha256((models/'dqn_model.pt').read_bytes()).hexdigest(),
      'normalizer_sha256':hashlib.sha256((models/'dqn_state_normalizer.json').read_bytes()).hexdigest(),
      'validation_frozen_state_verified':True,'validation_reload_exact':True})
    (models/'dqn_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Training and greedy validation complete. Review validation diagnostics before final test.',flush=True)
    print(comparison[['Strategy','terminal_soc_adjusted_cost_eur','efc_per_day']].to_string(index=False),flush=True)

if __name__=='__main__':main()
