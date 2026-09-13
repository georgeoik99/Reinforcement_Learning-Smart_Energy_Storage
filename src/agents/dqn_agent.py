"""Vanilla DQN: uniform replay and max over the same target network (not Double DQN)."""
from dataclasses import dataclass,asdict
from copy import deepcopy
import hashlib,random
import numpy as np
import torch
from torch import nn
from .replay_buffer import ReplayBuffer

ENV_ACTIONS=(-1,0,1)

@dataclass(frozen=True)
class DQNConfig:
    episodes:int=100
    gamma:float=.95
    learning_rate:float=.001
    batch_size:int=64
    replay_buffer_size:int=50000
    minimum_replay_size:int=2000
    epsilon_start:float=1.
    epsilon_min:float=.05
    epsilon_decay_steps:int=350400
    train_every:int=4
    target_sync_steps:int=1000
    hidden_layers:tuple=(64,64)
    gradient_clip_norm:float=10.
    seed:int=42

    def __post_init__(self):
        for name in ('episodes','batch_size','replay_buffer_size','minimum_replay_size','epsilon_decay_steps','train_every','target_sync_steps'):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1: raise ValueError(name)
        if not 0<=self.gamma<=1 or not 0<self.learning_rate or not np.isfinite(self.learning_rate): raise ValueError('Invalid learning settings')
        if not self.batch_size<=self.minimum_replay_size<=self.replay_buffer_size: raise ValueError('Invalid replay settings')
        if not 0<=self.epsilon_min<=self.epsilon_start<=1: raise ValueError('Invalid exploration settings')
        if tuple(self.hidden_layers)!=(64,64): raise ValueError('This stage fixes two 64-unit hidden layers')
        if not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm<=0: raise ValueError('Invalid gradient clip')

class QNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))

def fingerprint(value):
    """Stable digest for tensors, optimizer, replay arrays and RNG states."""
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor): visit(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray): h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str): visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()

class DQNAgent:
    def __init__(self,config=None):
        self.config=config or DQNConfig()
        random.seed(self.config.seed);np.random.seed(self.config.seed);torch.manual_seed(self.config.seed)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        self.rng=np.random.default_rng(self.config.seed)
        self.online=QNetwork().cpu()
        self.target=deepcopy(self.online).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.config.learning_rate)
        self.replay=ReplayBuffer(self.config.replay_buffer_size,self.config.seed+1)
        self.steps=0;self.updates=0;self.target_syncs=0

    @property
    def epsilon(self):
        c=self.config
        return c.epsilon_start+(c.epsilon_min-c.epsilon_start)*min(self.steps/c.epsilon_decay_steps,1.)

    def q_values(self,state):
        with torch.no_grad():
            q=self.online(torch.as_tensor(state,dtype=torch.float32)).cpu().numpy()
        if not np.isfinite(q).all(): raise FloatingPointError('Nonfinite Q values')
        return q

    def select_action(self,state,explore=True):
        if explore and self.rng.random()<self.epsilon:return int(self.rng.integers(3))
        q=self.q_values(state)
        if q.shape!=(3,):raise ValueError('Expected one observation')
        return next(a for a in (1,0,2) if q[a]==q.max())

    def td_targets(self,rewards,next_states,dones):
        with torch.no_grad():
            return rewards+self.config.gamma*(~dones).float()*self.target(next_states).max(dim=1).values

    def learn(self):
        states,actions,rewards,nxt,dones=[torch.from_numpy(x) for x in self.replay.sample(self.config.batch_size)]
        q=self.online(states).gather(1,actions[:,None]).squeeze(1)
        target=self.td_targets(rewards,nxt,dones)
        loss=nn.functional.smooth_l1_loss(q,target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=nn.utils.clip_grad_norm_(self.online.parameters(),self.config.gradient_clip_norm,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        return {'loss':float(loss.detach()),'max_abs_q':float(q.detach().abs().max()),
                'max_abs_target':float(target.abs().max()),'gradient_norm':float(norm)}

    def observe(self,state,action,reward,next_state,done):
        self.replay.add(state,action,reward,next_state,done);self.steps+=1
        metrics=None
        if len(self.replay)>=self.config.minimum_replay_size and self.steps%self.config.train_every==0:metrics=self.learn()
        if self.steps%self.config.target_sync_steps==0:
            self.target.load_state_dict(self.online.state_dict());self.target_syncs+=1
        return metrics

    def training_state(self):
        return {'online':self.online.state_dict(),'target':self.target.state_dict(),
                'optimizer':self.optimizer.state_dict(),'replay':self.replay.state_dict(),
                'steps':self.steps,'updates':self.updates,'target_syncs':self.target_syncs,
                'rng':self.rng.bit_generator.state,'python_rng':random.getstate(),
                'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'online_training':self.online.training,'target_training':self.target.training}

    def restore_training(self,state):
        self.online.load_state_dict(state['online']);self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer']);self.replay.load_state_dict(state['replay'])
        for k in ('steps','updates','target_syncs'):setattr(self,k,state[k])
        self.rng.bit_generator.state=state['rng'];random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])

    def save(self,path):
        torch.save({'config':asdict(self.config),'online':self.online.state_dict(),
                    'target':self.target.state_dict(),'steps':self.steps,'updates':self.updates,
                    'target_syncs':self.target_syncs},path)

    @classmethod
    def load(cls,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        agent=cls(DQNConfig(**payload['config']))
        agent.online.load_state_dict(payload['online']);agent.target.load_state_dict(payload['target'])
        for k in ('steps','updates','target_syncs'):setattr(agent,k,payload[k])
        return agent
