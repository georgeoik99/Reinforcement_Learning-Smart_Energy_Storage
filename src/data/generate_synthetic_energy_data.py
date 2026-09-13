from pathlib import Path
import numpy as np
import pandas as pd


def _generate_year(year: int, seed: int, pv_capacity_kwp: float) -> pd.DataFrame:
    """
    Generate a calendar year with an independent, reproducible annual random stream.

    IMPORTANT:
    - This is realistic synthetic data for portfolio/RL development.
    - It is NOT actual HEnEx or PVGIS data.
    - Battery state-of-charge is intentionally excluded because SOC
      is generated dynamically inside the RL environment.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, year]))
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    n = len(idx)
    days_in_year = n / 24

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    is_weekend = (dow >= 5).astype(int)

    # 1. Commercial demand
    business_activity = np.where(
        (hour >= 8) & (hour <= 18),
        1.0,
        np.where(
            (hour >= 6) & (hour < 8),
            0.55,
            np.where((hour > 18) & (hour <= 22), 0.45, 0.25),
        ),
    )
    weekday_factor = np.where(is_weekend == 1, 0.68, 1.0)
    seasonal_demand = (
        1.0
        + 0.13 * np.cos(2 * np.pi * (doy - 20) / days_in_year)
        + 0.10 * np.cos(2 * np.pi * (doy - 205) / days_in_year)
    )
    demand = (
        18
        + 47 * business_activity * weekday_factor * seasonal_demand
        + rng.normal(0, 3.2, n)
    )
    demand = np.clip(demand, 10, None)

    # 2. Configurable rooftop PV (100 kWp default), same seasonal/daylight process.
    day_length_factor = 0.78 + 0.28 * np.sin(2 * np.pi * (doy - 80) / days_in_year)
    solar_angle = np.sin(np.pi * np.clip((hour - 6.0) / 13.0, 0, 1))
    solar_angle[(hour < 6) | (hour > 19)] = 0
    seasonal_pv = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / days_in_year)
    seasonal_pv = np.clip(seasonal_pv, 0.18, 1.0)
    cloud_factor = np.clip(rng.beta(5, 2, n) * 1.12, 0.20, 1.0)

    pv = pv_capacity_kwp * solar_angle * seasonal_pv * day_length_factor * cloud_factor
    pv = np.clip(pv, 0, pv_capacity_kwp)

    # 3. Dynamic electricity price (EUR/MWh)
    hourly_price_shape = (
        16 * np.exp(-0.5 * ((hour - 9) / 2.2) ** 2)
        + 38 * np.exp(-0.5 * ((hour - 20) / 2.0) ** 2)
        - 18 * np.exp(-0.5 * ((hour - 14) / 2.4) ** 2)
    )
    seasonal_price = (
        12 * np.cos(2 * np.pi * (doy - 15) / days_in_year)
        + 6 * np.cos(4 * np.pi * (doy - 15) / days_in_year)
    )
    weekend_price = np.where(is_weekend == 1, -7, 0)

    price = (
        92
        + hourly_price_shape
        + seasonal_price
        + weekend_price
        + rng.normal(0, 10, n)
    )

    spike_mask = rng.random(n) < 0.012
    price[spike_mask] += rng.uniform(60, 170, spike_mask.sum())

    positive_pv = pv[pv > 0]
    pv_threshold = np.percentile(positive_pv, 65)
    low_price_mask = (
        (hour >= 12)
        & (hour <= 16)
        & (pv > pv_threshold)
        & (rng.random(n) < 0.025)
    )
    price[low_price_mask] -= rng.uniform(55, 105, low_price_mask.sum())
    price = np.clip(price, -25, 320)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "demand_kwh": np.round(demand, 3),
            "pv_generation_kwh": np.round(pv, 3),
            "electricity_price_eur_mwh": np.round(price, 2),
        }
    )

    df["electricity_price_eur_kwh"] = np.round(
        df["electricity_price_eur_mwh"] / 1000, 5
    )
    df["net_load_before_battery_kwh"] = np.round(
        df["demand_kwh"] - df["pv_generation_kwh"], 3
    )
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    return df


def generate_energy_dataset(seed: int = 42, pv_capacity_kwp: float = 100.0) -> pd.DataFrame:
    """2023 training, 2024 validation, 2025 test; naive local hourly labels.

    No daylight-saving transitions are simulated. Year-specific independent
    streams retain the same demand, PV and price generating process. The annual
    seasonal phase accounts for 366 days in 2024. No rows are hand-adjusted.
    """
    if not np.isfinite(pv_capacity_kwp) or pv_capacity_kwp <= 0:
        raise ValueError("PV nameplate capacity must be positive and finite.")
    return pd.concat([_generate_year(year, seed, pv_capacity_kwp)
                      for year in (2023, 2024, 2025)], ignore_index=True)


if __name__ == "__main__":
    output_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "energy_hourly_2023_2025.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_energy_dataset(seed=42)
    data.to_csv(output_path, index=False)

    print(f"Saved {len(data):,} hourly rows to: {output_path}")
    print(data.head())
