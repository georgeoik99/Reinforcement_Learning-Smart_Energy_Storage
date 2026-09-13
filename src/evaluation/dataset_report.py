"""Generate an auditable per-year report from the new, generated input dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data_utils import chronological_split

ROOT = Path(__file__).resolve().parents[2]


def dataset_statistics(data):
    records = []
    for name, part in zip(("Training", "Validation", "Test"), chronological_split(data)):
        pv = part.pv_generation_kwh
        surplus = (pv - part.demand_kwh).clip(lower=0)
        records.append({
            "split": name, "year": int(part.timestamp.dt.year.iloc[0]), "rows": len(part),
            "demand_kwh": float(part.demand_kwh.sum()), "pv_generation_kwh": float(pv.sum()),
            "pv_surplus_hours": int((surplus > 0).sum()), "pv_surplus_kwh": float(surplus.sum()),
            "pv_direct_consumption_pct": float(np.minimum(pv, part.demand_kwh).sum() / pv.sum() * 100) if pv.sum() else 0.,
            "price_min_eur_kwh": float(part.electricity_price_eur_kwh.min()),
            "price_mean_eur_kwh": float(part.electricity_price_eur_kwh.mean()),
            "price_max_eur_kwh": float(part.electricity_price_eur_kwh.max()),
        })
    return pd.DataFrame(records)


def main():
    data = pd.read_csv(ROOT / "data/processed/energy_hourly_2023_2025.csv", parse_dates=["timestamp"])
    result = dataset_statistics(data)
    folder = ROOT / "outputs/results"
    folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(folder / "stage4_5_dataset_statistics.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
