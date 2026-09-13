"""Frozen DQN inference using the unchanged Stage-4.5 dispatch and accounting."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.agents.dqn_agent import DQNAgent,ENV_ACTIONS,fingerprint
from src.agents.state_normalizer import StateNormalizer
from src.environment.energy_storage_env import EnergyStorageEnv
from .data_utils import validate_hourly_data,validate_battery_config,load_battery_config,chronological_split
from .metrics import validate_results
from .terminal_valuation import TerminalValuation,calculate_refined_metrics

ROOT=Path(__file__).resolve().parents[2]

def evaluate_dqn(data,agent,normalizer,config,penalty):
    data=validate_hourly_data(data);validate_battery_config(config)
    if data.timestamp.iloc[0]<=pd.Timestamp(normalizer.training_end):raise ValueError('Evaluation must follow training')
    if not np.isfinite(penalty) or penalty<0:raise ValueError('Invalid penalty')
    before=fingerprint(agent.training_state());norm_before=fingerprint(vars(normalizer))
    env=EnergyStorageEnv(data,config,penalty)
    state=normalizer.transform(env.reset());rows=[]
    for row in data.itertuples(index=False):
        soc=env.battery.soc_kwh
        q=agent.q_values(state)
        action=agent.select_action(state,explore=False)
        nxt,_,done,info=env.step(ENV_ACTIONS[action])
        charge=info['pv_to_battery_kwh']+info['grid_to_battery_kwh']
        discharge=info['discharge_to_load_kwh']
        info.update({'electricity_price_eur_kwh':row.electricity_price_eur_kwh,
            'action_index':action,'soc_before_kwh':soc,'charge_from_bus_kwh':charge,
            'discharge_to_bus_kwh':discharge,'unused_discharge_kwh':0.,
            'pv_curtailed_kwh':row.pv_generation_kwh-info['pv_used_directly_kwh']-info['pv_to_battery_kwh'],
            'battery_losses_kwh':charge*(1-config.charge_efficiency)+discharge*(1/config.discharge_efficiency-1),
            'q_discharge':float(q[0]),'q_hold':float(q[1]),'q_charge':float(q[2])})
        rows.append(info)
        if not done:state=normalizer.transform(nxt)
    if before!=fingerprint(agent.training_state()) or norm_before!=fingerprint(vars(normalizer)):
        raise AssertionError('Greedy evaluation mutated model/optimizer/replay/normalizer/RNG')
    result=pd.DataFrame(rows);validate_results(result,config,True,penalty)
    return result

def dqn_behaviour(result,training,config):
    low,high=training.electricity_price_eur_kwh.quantile([.33,.66]).to_numpy()
    cheap=result.electricity_price_eur_kwh<low;expensive=result.electricity_price_eur_kwh>=high
    surplus=(result.pv_generation_kwh-result.demand_kwh).clip(lower=0)
    charge=result.action.eq(1);discharge=result.action.eq(-1)
    flow=result.battery_throughput_kwh>1e-8;commands=charge|discharge
    boundary=(charge & (result.soc_before_kwh>=config.max_soc_kwh-1e-8)) | (discharge & (result.soc_before_kwh<=config.min_soc_kwh+1e-8))
    return {
      'price_threshold_definition':'2023 P33/P66, identical price groups to tabular analysis',
      'low_price_threshold_eur_kwh':float(low),'high_price_threshold_eur_kwh':float(high),
      'low_price_hours':int(cheap.sum()),'low_price_charge_requests':int((cheap&charge).sum()),
      'low_price_charge_request_pct':float((cheap&charge).sum()/cheap.sum()*100) if cheap.any() else 0.,
      'low_price_active_charge_events':int((cheap&charge&flow).sum()),
      'low_price_grid_to_battery_kwh':float(result.loc[cheap,'grid_to_battery_kwh'].sum()),
      'high_price_hours':int(expensive.sum()),'high_price_discharge_requests':int((expensive&discharge).sum()),
      'high_price_useful_discharge_events':int((expensive&discharge&flow).sum()),
      'high_price_useful_discharge_kwh':float(result.loc[expensive,'discharge_to_load_kwh'].sum()),
      'pv_surplus_hours':int((surplus>0).sum()),'pv_surplus_kwh':float(surplus.sum()),
      'charge_requests_during_surplus':int(((surplus>0)&charge).sum()),
      'pv_to_battery_kwh':float(result.pv_to_battery_kwh.sum()),
      'pv_surplus_captured_pct':float(result.pv_to_battery_kwh.sum()/surplus.sum()*100) if surplus.sum() else 0.,
      'pv_curtailed_kwh':float(result.pv_curtailed_kwh.sum()),
      'grid_to_battery_kwh':float(result.grid_to_battery_kwh.sum()),
      'discharge_to_load_kwh':float(result.discharge_to_load_kwh.sum()),
      'charge_requests':int(charge.sum()),'hold_requests':int((~commands).sum()),'discharge_requests':int(discharge.sum()),
      'non_hold_commands':int(commands.sum()),'non_hold_commands_with_flow':int((commands&flow).sum()),
      'non_hold_commands_with_flow_pct':float((commands&flow).sum()/commands.sum()*100) if commands.any() else 0.,
      'zero_flow_charge_commands':int((charge&~flow).sum()),'zero_flow_discharge_commands':int((discharge&~flow).sum()),
      'zero_flow_soc_boundary_commands':int((commands&~flow&boundary).sum()),
      'zero_flow_no_load_discharge_excluding_boundary':int((discharge&~flow&~boundary).sum()),
      'max_abs_q':float(result[['q_discharge','q_hold','q_charge']].abs().to_numpy().max()),
    }

def comparison_with_dqn(previous,result,training,config):
    previous=previous.copy()
    base=float(previous.loc[previous.Strategy=='No Battery','raw_electricity_cost_eur'].iloc[0])
    row={'Strategy':'DQN',**calculate_refined_metrics(result,base,config,TerminalValuation.fit(training))}
    comparison=pd.concat([previous,pd.DataFrame([row])],ignore_index=True)
    for label,key in [('Rule-Based','rule_based'),('Q-Learning','q_learning')]:
        for cost,prefix in [('raw_electricity_cost_eur','raw'),('terminal_soc_adjusted_cost_eur','adjusted')]:
            benchmark=float(comparison.loc[comparison.Strategy==label,cost].iloc[0])
            comparison[f'{prefix}_cost_difference_vs_{key}_eur']=comparison[cost]-benchmark
    # Preserve original raw-difference column's interpretation for all four rows.
    comparison['cost_difference_vs_rule_based_eur']=comparison['raw_cost_difference_vs_rule_based_eur']
    if comparison.isna().any().any():raise AssertionError('Incomplete comparison')
    return comparison

def main():
    models=ROOT/'outputs/models';results=ROOT/'outputs/results'
    metadata=json.loads((models/'dqn_config.json').read_text())
    if metadata.get('validation_review_status')!='accepted_no_changes':
        raise ValueError('Review validation diagnostics and record selection before final test')
    for rel,expected in metadata['provenance'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Provenance changed: '+rel)
    for filename,key in [('dqn_model.pt','model_sha256'),('dqn_state_normalizer.json','normalizer_sha256')]:
        if hashlib.sha256((models/filename).read_bytes()).hexdigest()!=metadata[key]:raise ValueError(filename+' changed')
    train,_,test=chronological_split(pd.read_csv(ROOT/'data/processed/energy_hourly_2023_2025.csv',parse_dates=['timestamp']))
    config,penalty=load_battery_config(ROOT/'config/battery_config.json')
    norm=StateNormalizer.load(models/'dqn_state_normalizer.json')
    if norm!=StateNormalizer.fit(train):raise AssertionError('Normalizer differs from training-only fit')
    agent=DQNAgent.load(models/'dqn_model.pt')
    result=evaluate_dqn(test,agent,norm,config,penalty)
    repeat=evaluate_dqn(test,DQNAgent.load(models/'dqn_model.pt'),StateNormalizer.load(models/'dqn_state_normalizer.json'),config,penalty)
    pd.testing.assert_frame_equal(result,repeat,check_exact=True)
    prior=pd.read_csv(results/'q_learning_comparison.csv')
    # Recompute metrics from all old hourly artifacts; never rewrite them.
    frames={}
    for label,filename in [('No Battery','no_battery_test_results.csv'),('Rule-Based','rule_based_test_results.csv'),('Q-Learning','q_learning_test_results.csv')]:
        frame=pd.read_csv(results/filename,parse_dates=['timestamp'])
        validate_results(frame,config,label!='No Battery',penalty)
        for col in ('timestamp','demand_kwh','pv_generation_kwh','electricity_price_eur_kwh'):
            pd.testing.assert_series_equal(frame[col],result[col],check_names=False)
        metrics=calculate_refined_metrics(frame,float(prior.iloc[0].raw_electricity_cost_eur),config,TerminalValuation.fit(train))
        for k,v in metrics.items():np.testing.assert_allclose(v,prior.loc[prior.Strategy==label,k].iloc[0],atol=1e-8,rtol=1e-10)
        frames[label]=frame
    comparison=comparison_with_dqn(prior,result,train,config)
    result.to_csv(results/'dqn_test_results.csv',index=False)
    comparison.to_csv(results/'stage5_final_comparison.csv',index=False)
    behaviour=dqn_behaviour(result,train,config)
    (results/'dqn_test_behaviour.json').write_text(json.dumps(behaviour,indent=2)+'\n')
    (results/'stage5_evaluation_metadata.json').write_text(json.dumps({'epsilon':0,'gradient_updates':0,
       'frozen_full_training_state_verified':True,'model_reload_exact_reproduction':True,
       'baseline_csv_metrics_reverified':True,'test_year':2025,'model_sha256':metadata['model_sha256'],
       'test_tuning':False,'differences_sign':'DQN minus benchmark; negative is lower cost'},indent=2)+'\n')
    from .dqn_plots import make_dqn_plots
    frames['DQN']=result
    make_dqn_plots(pd.read_csv(results/'dqn_training_history.csv'),frames,comparison,ROOT/'outputs/figures')
    print(comparison[['Strategy','raw_electricity_cost_eur','terminal_soc_adjusted_cost_eur','adjusted_cost_savings_pct','efc_per_day']].to_string(index=False))
    print(json.dumps(behaviour,indent=2))

if __name__=='__main__':main()
