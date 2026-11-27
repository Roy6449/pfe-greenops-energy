from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

RAW_PATH = Path("data") / "raw" / "bear_3_buildings_raw.csv"
PROCESSED_DIR = Path("data") / "processed"
REJECTED_DIR = Path("data") / "rejected"


def run_etl():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Chargement des données brutes...")
    df = pd.read_csv(RAW_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    # 2. Combler les 6 trous isolés de température par interpolation par bâtiment
    df["temperature_c"] = df.groupby("building_id")["temperature_c"].transform(
        lambda s: s.interpolate(method="linear").bfill().ffill()
    )

    # 3. Dead Letter Queue (DLQ) : isolation des relevés de consommation manquants ou aberrants
    invalid_mask = (
        df["consumption_kwh"].isna()
        | (df["consumption_kwh"] < 0)
        | (df["temperature_c"] < -30)
        | (df["temperature_c"] > 55)
    )

    df_rejected = df[invalid_mask].copy()
    df_rejected["rejection_reason"] = "missing_or_out_of_bounds_value"
    rejected_path = REJECTED_DIR / "rejected_records.csv"
    df_rejected.to_csv(rejected_path, index=False)
    print(f"2. Dead Letter Queue : {len(df_rejected)} lignes rejetées -> {rejected_path}")

    df_clean = df[~invalid_mask].copy()

    # 4. Validation stricte du contrat de données avec Pandera
    schema = pa.DataFrameSchema(
        {
            "timestamp": Column("datetime64[ns]", nullable=False),
            "building_id": Column(str, nullable=False),
            "consumption_kwh": Column(float, Check.ge(0), nullable=False),
            "temperature_c": Column(float, Check.in_range(-30, 55), nullable=False),
        }
    )
    df_clean = schema.validate(df_clean)
    print(f"3. Validation Pandera OK : {len(df_clean)} lignes valides conservées.")

    # 5. Feature Engineering (variables calendaires + inertie thermique / historique)
    df_clean["hour"] = df_clean["timestamp"].dt.hour
    df_clean["day_of_week"] = df_clean["timestamp"].dt.dayofweek
    df_clean["is_weekend"] = (df_clean["day_of_week"] >= 5).astype(int)
    df_clean["month"] = df_clean["timestamp"].dt.month

    df_clean["lag_24h"] = df_clean.groupby("building_id")["consumption_kwh"].shift(24)
    df_clean["rolling_mean_24h"] = (
        df_clean.groupby("building_id")["consumption_kwh"]
        .transform(lambda s: s.shift(1).rolling(window=24, min_periods=12).mean())
    )

    # Suppression des premières heures sans historique lag_24h
    df_model = df_clean.dropna(subset=["lag_24h", "rolling_mean_24h"]).reset_index(drop=True)

    output_path = PROCESSED_DIR / "bear_model_ready.csv"
    df_model.to_csv(output_path, index=False)
    print(f"4. Dataset prêt pour l'IA : {len(df_model)} lignes sauvegardées -> {output_path}")


if __name__ == "__main__":
    run_etl()
