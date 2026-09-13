import unittest,tempfile
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from src.agents.dqn_agent import DQNAgent,DQNConfig,QNetwork,ENV_ACTIONS,fingerprint
from src.agents.state_normalizer import StateNormalizer
from src.agents.replay_buffer import ReplayBuffer
from src.environment.battery_model import BatteryConfig
from src.environment.energy_storage_env import EnergyStorageEnv
from src.evaluation.data_utils import validate_hourly_data,chronological_split
from src.evaluation.evaluate_dqn import evaluate_dqn,dqn_behaviour
from src.training.train_dqn import train_episode

def data(year=2023,n=96):
    rng=np.random.default_rng(5)
    return validate_hourly_data(pd.DataFrame({'timestamp':pd.date_range(f'{year}-01-01',periods=n,freq='h'),
       'demand_kwh':rng.uniform(1,70,n),'pv_generation_kwh':rng.uniform(0,100,n),
       'electricity_price_eur_kwh':rng.uniform(-.02,.3,n)}))

class DQNTests(unittest.TestCase):
    def setUp(self):
        self.settings=replace(DQNConfig(),episodes=2,batch_size=8,minimum_replay_size=16,replay_buffer_size=128,target_sync_steps=20)
        self.train=data();self.norm=StateNormalizer.fit(self.train);self.battery=BatteryConfig()

    def test_network_dimensions_and_finite_output(self):
        network=QNetwork()
        self.assertEqual(network(torch.zeros(11,7)).shape,(11,3))
        self.assertTrue(torch.isfinite(network(torch.randn(11,7))).all())
        self.assertEqual(sum(p.numel() for p in network.parameters()),4867)

    def test_normalizer_statistics_and_unchanged_bounded_features(self):
        env=EnergyStorageEnv(self.train,self.battery);raw=env.reset();out=self.norm.transform(raw)
        np.testing.assert_array_equal(raw[3:],out[3:])
        np.testing.assert_allclose(self.norm.mean,self.train[['demand_kwh','pv_generation_kwh','electricity_price_eur_kwh']].mean())
        with self.assertRaises(ValueError):StateNormalizer.fit(data(2024))
        with self.assertRaises(ValueError):StateNormalizer.fit(pd.concat([self.train,data(2025)]))

    def test_future_rows_cannot_fit_normalizer(self):
        frame=data(2023,26304)
        original=StateNormalizer.fit(chronological_split(frame)[0])
        frame.loc[8760:,'demand_kwh']=9999
        self.assertEqual(original,StateNormalizer.fit(chronological_split(frame)[0]))

    def test_normalizer_serialization_and_constant_feature(self):
        sample=self.train.copy();sample.pv_generation_kwh=0
        norm=StateNormalizer.fit(sample)
        self.assertEqual(norm.scale[1],1.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'norm.json';norm.save(path)
            restored=StateNormalizer.load(path)
            self.assertEqual(norm,restored)
            np.testing.assert_array_equal(norm.transform(np.ones(7)),restored.transform(np.ones(7)))

    def test_invalid_state_and_settings(self):
        for state in [np.zeros(6),np.full(7,np.nan)]:
            with self.assertRaises(ValueError):self.norm.transform(state)
        for kwargs in [{'gamma':2},{'batch_size':50001},{'train_every':0},{'epsilon_min':2}]:
            with self.assertRaises(ValueError):DQNConfig(**kwargs)

    def test_replay_fifo_and_terminal_state(self):
        buffer=ReplayBuffer(3)
        for i in range(5):buffer.add(np.full(7,i),i%3,float(i),None,True)
        self.assertEqual(len(buffer),3);self.assertEqual(set(buffer.rewards),{2.,3.,4.})
        self.assertTrue((buffer.next_states==0).all())
        sample=buffer.sample(3);self.assertEqual(sample[0].shape,(3,7))
        with self.assertRaises(ValueError):buffer.sample(4)

    def test_replay_sampling_and_restore_deterministic(self):
        buffer=ReplayBuffer(9)
        for i in range(9):buffer.add(np.full(7,i),0,float(i),np.ones(7),False)
        restored=ReplayBuffer(9);restored.load_state_dict(deepcopy(buffer.state_dict()))
        for a,b in zip(buffer.sample(5),restored.sample(5)):np.testing.assert_array_equal(a,b)

    def test_vanilla_target_and_terminal_no_bootstrap(self):
        agent=DQNAgent(self.settings)
        with torch.no_grad():
            for p in agent.target.parameters():p.zero_()
            agent.target[-1].bias.copy_(torch.tensor([2.,8.,4.]))
            for p in agent.online.parameters():p.zero_()
            agent.online[-1].bias.copy_(torch.tensor([30.,0.,0.]))
        target=agent.td_targets(torch.tensor([-3.,-3.]),torch.zeros(2,7),torch.tensor([True,False]))
        np.testing.assert_allclose(target.numpy(),[-3.,-3.+.95*8],atol=1e-6)

    def test_target_periodic_sync_only(self):
        agent=DQNAgent(replace(self.settings,target_sync_steps=100))
        target=fingerprint(agent.target.state_dict())
        for i in range(99):agent.observe(np.ones(7),1,-1.,np.zeros(7),False)
        self.assertEqual(target,fingerprint(agent.target.state_dict()))
        self.assertNotEqual(target,fingerprint(agent.online.state_dict()))
        agent.observe(np.ones(7),1,-1.,None,True)
        self.assertEqual(fingerprint(agent.target.state_dict()),fingerprint(agent.online.state_dict()))

    def test_epsilon_step_schedule_and_action_mapping(self):
        agent=DQNAgent(self.settings)
        self.assertEqual(ENV_ACTIONS,(-1,0,1));self.assertEqual(agent.epsilon,1.)
        agent.steps=self.settings.epsilon_decay_steps
        self.assertAlmostEqual(agent.epsilon,.05)
        agent.steps*=2;self.assertAlmostEqual(agent.epsilon,.05)
        with torch.no_grad():
            for p in agent.online.parameters():p.zero_()
        self.assertEqual(agent.select_action(np.zeros(7),explore=False),1)

    def test_training_finite_and_seeded_determinism(self):
        one=DQNAgent(self.settings);hist1=train_episode(one,self.norm,self.train,self.battery,.01)
        weights=fingerprint(one.online.state_dict())
        two=DQNAgent(self.settings);hist2=train_episode(two,self.norm,self.train,self.battery,.01)
        self.assertEqual(weights,fingerprint(two.online.state_dict()));self.assertEqual(hist1,hist2)
        self.assertGreater(hist1['mean_training_loss'],0)
        self.assertTrue(np.isfinite(list(hist1.values())).all())

    def test_checkpoint_resumes_exact_training(self):
        one=DQNAgent(self.settings);train_episode(one,self.norm,self.train,self.battery,.01)
        state=deepcopy(one.training_state())
        expected=train_episode(one,self.norm,self.train,self.battery,.01)
        expected_weights=fingerprint(one.online.state_dict())
        restored=DQNAgent(self.settings);restored.restore_training(state)
        actual=train_episode(restored,self.norm,self.train,self.battery,.01)
        self.assertEqual(actual,expected);self.assertEqual(expected_weights,fingerprint(restored.online.state_dict()))

    def test_validation_and_test_frozen_including_replay(self):
        agent=DQNAgent(self.settings);train_episode(agent,self.norm,self.train,self.battery,.01)
        before=fingerprint(agent.training_state())
        for year in [2024,2025]:evaluate_dqn(data(year),agent,self.norm,self.battery,.01)
        self.assertEqual(before,fingerprint(agent.training_state()))

    def test_model_reload_exact_predictions_and_hourly_inference(self):
        agent=DQNAgent(self.settings);train_episode(agent,self.norm,self.train,self.battery,.01)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';agent.save(path);restored=DQNAgent.load(path)
            np.testing.assert_array_equal(agent.q_values(np.zeros(7)),restored.q_values(np.zeros(7)))
            first=evaluate_dqn(data(2025),agent,self.norm,self.battery,.01)
            second=evaluate_dqn(data(2025),restored,self.norm,self.battery,.01)
            pd.testing.assert_frame_equal(first,second,check_exact=True)

    def test_physical_flows_match_environment_for_both_directions(self):
        agent=DQNAgent(self.settings)
        for index in [0,2]:
            with torch.no_grad():
                for p in agent.online.parameters():p.zero_()
                agent.online[-1].bias[index]=1.
            sample=data(2025);result=evaluate_dqn(sample,agent,self.norm,self.battery,.01)
            env=EnergyStorageEnv(sample,self.battery);env.reset()
            for row in result.itertuples(index=False):
                _,_,_,info=env.step(row.action)
                for key in ('soc_kwh','grid_import_kwh','pv_to_battery_kwh','reward'):
                    self.assertAlmostEqual(getattr(row,key),info[key])
            self.assertTrue((result.grid_import_kwh>=0).all())
            self.assertTrue(result.soc_fraction.between(.1,.95).all())

    def test_future_changes_do_not_change_past_actions(self):
        agent=DQNAgent(self.settings);sample=data(2025)
        first=evaluate_dqn(sample,agent,self.norm,self.battery,.01)
        sample.loc[48:,'electricity_price_eur_kwh']=999
        second=evaluate_dqn(sample,agent,self.norm,self.battery,.01)
        pd.testing.assert_frame_equal(first.iloc[:48],second.iloc[:48],check_exact=True)

    def test_behaviour_counts_requests_separately_from_flow(self):
        agent=DQNAgent(self.settings)
        with torch.no_grad():
            for p in agent.online.parameters():p.zero_()
            agent.online[-1].bias[0]=1
        result=evaluate_dqn(data(2025),agent,self.norm,self.battery,.01)
        stats=dqn_behaviour(result,self.train,self.battery)
        self.assertEqual(stats['discharge_requests'],96)
        self.assertGreater(stats['zero_flow_discharge_commands'],0)
        self.assertLess(stats['non_hold_commands_with_flow_pct'],100)

if __name__=='__main__':unittest.main()
