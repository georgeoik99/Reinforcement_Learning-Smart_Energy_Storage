"""Uniform FIFO replay, sampled without replacement within each minibatch."""
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, seed=42):
        if not isinstance(capacity,int) or capacity<1: raise ValueError('Positive integer capacity required')
        self.capacity=capacity; self.position=0; self.size=0
        self.rng=np.random.default_rng(seed)
        self.states=np.zeros((capacity,7),np.float32)
        self.next_states=np.zeros_like(self.states)
        self.actions=np.zeros(capacity,np.int64)
        self.rewards=np.zeros(capacity,np.float32)
        self.dones=np.zeros(capacity,bool)

    def __len__(self): return self.size

    def add(self,state,action,reward,next_state,done):
        state=np.asarray(state,dtype=np.float32)
        nxt=np.zeros(7,np.float32) if done else np.asarray(next_state,dtype=np.float32)
        if state.shape!=(7,) or nxt.shape!=(7,) or not np.isfinite(state).all() or not np.isfinite(nxt).all() or not np.isfinite(reward) or action not in (0,1,2):
            raise ValueError('Invalid replay transition')
        i=self.position
        self.states[i]=state; self.next_states[i]=nxt; self.actions[i]=action
        self.rewards[i]=reward; self.dones[i]=bool(done)
        self.position=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)

    def sample(self,batch_size):
        if batch_size<1 or batch_size>self.size: raise ValueError('Insufficient replay observations')
        idx=self.rng.choice(self.size,size=batch_size,replace=False)
        return tuple(x[idx] for x in (self.states,self.actions,self.rewards,self.next_states,self.dones))

    def state_dict(self):
        return {k:getattr(self,k) for k in ('capacity','position','size','states','next_states','actions','rewards','dones')} | {'rng_state':self.rng.bit_generator.state}

    def load_state_dict(self,state):
        if state['capacity']!=self.capacity: raise ValueError('Replay capacity differs')
        for key,value in state.items():
            if key=='rng_state': self.rng.bit_generator.state=value
            else: setattr(self,key,value)
