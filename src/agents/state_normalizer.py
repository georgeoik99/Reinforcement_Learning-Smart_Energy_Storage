"""Training-2023-only z-scores for the first three continuous observations."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from src.evaluation.data_utils import validate_hourly_data, ENERGY_COLUMNS

STATE_NAMES = ('demand_kwh', 'pv_generation_kwh', 'electricity_price_eur_kwh',
               'soc_fraction', 'hour_sin', 'hour_cos', 'weekend_flag')

@dataclass(frozen=True)
class StateNormalizer:
    mean: tuple
    scale: tuple
    training_start: str
    training_end: str
    training_rows: int

    def __post_init__(self):
        if len(self.mean)!=3 or len(self.scale)!=3 or not np.isfinite([self.mean,self.scale]).all() or min(self.scale)<=0:
            raise ValueError('Invalid normalizer parameters')

    @classmethod
    def fit(cls, data):
        data=validate_hourly_data(data)
        if not data.timestamp.dt.year.eq(2023).all():
            raise ValueError('Normalizer fitting permits only training year 2023')
        values=data[ENERGY_COLUMNS].to_numpy(dtype=float)
        scale=values.std(axis=0,ddof=0)
        scale=np.where(scale<1e-12,1.,scale)
        return cls(tuple(values.mean(axis=0)),tuple(scale),str(data.timestamp.iloc[0]),str(data.timestamp.iloc[-1]),len(data))

    def transform(self, state):
        result=np.asarray(state,dtype=np.float32).copy()
        if result.ndim not in (1,2) or result.shape[-1]!=7 or not np.isfinite(result).all():
            raise ValueError('Expected finite (...,7) observations')
        result[...,:3]=(result[...,:3]-np.asarray(self.mean))/np.asarray(self.scale)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite normalized state')
        return result

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        data['mean']=tuple(data['mean']); data['scale']=tuple(data['scale'])
        return cls(**data)
