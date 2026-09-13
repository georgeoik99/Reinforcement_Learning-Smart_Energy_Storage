from pathlib import Path
import pandas as pd

from src.environment import BatteryConfig, EnergyStorageEnv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "energy_hourly_2023_2025.csv"

df = pd.read_csv(DATA, parse_dates=["timestamp"])
sample = df.iloc[:48].copy()

env = EnergyStorageEnv(sample, BatteryConfig())
state = env.reset()

print("Initial state:", state)

for action in [0, 1, -1]:
    next_state, reward, done, info = env.step(action)
    print(
        f"{info['timestamp']} | action={action:+d} | "
        f"SOC={info['soc_fraction']:.1%} | "
        f"grid={info['grid_import_kwh']:.2f} kWh | "
        f"cost=EUR {info['grid_cost_eur']:.2f} | "
        f"reward={reward:.3f}"
    )
