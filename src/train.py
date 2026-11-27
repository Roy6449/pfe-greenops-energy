from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data") / "processed" / "bear_model_ready.csv"
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data") / "processed"


def run_training():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["building_code"] = df["building_id"].astype("category").cat.codes
    building_mapping = dict(
        enumerate(df["building_id"].astype("category").cat.categories)
    )

    features = [
        "building_code",
        "temperature_c",
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "lag_24h",
        "rolling_mean_24h",
    ]
    target = "consumption_kwh"

    # Découpage chronologique strict (80% passé, 20% futur)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    mlflow.set_experiment("pfe_greenops_energy")
    with mlflow.start_run(run_name="lightgbm_anomaly_comparison"):
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[features], train_df[target])

        test_df["predicted_kwh"] = model.predict(test_df[features])
        test_df["residual"] = test_df[target] - test_df["predicted_kwh"]

        mae = mean_absolute_error(test_df[target], test_df["predicted_kwh"])
        rmse = np.sqrt(mean_squared_error(test_df[target], test_df["predicted_kwh"]))

        # Initialisation des 3 colonnes de comparaison d'anomalies
        test_df["anomaly_zscore"] = False
        test_df["anomaly_iqr"] = False
        test_df["anomaly_iforest"] = False
        thresholds = {}

        for b_id, group in test_df.groupby("building_id"):
            idx = group.index
            res = group["residual"]

            # 1. Approche Z-Score (2.5 écarts-types)
            z_thresh = 2.5 * res.std()
            thresholds[b_id] = float(z_thresh)
            test_df.loc[idx, "anomaly_zscore"] = res.abs() > z_thresh

            # 2. Approche IQR robuste (Interquartile Range)
            q1, q3 = res.quantile(0.25), res.quantile(0.75)
            iqr = q3 - q1
            iqr_mask = (res < (q1 - 2.5 * iqr)) | (res > (q3 + 2.5 * iqr))
            test_df.loc[idx, "anomaly_iqr"] = iqr_mask

            # 3. Approche Isolation Forest (sur résidu + contexte thermique/horaire)
            iso = IsolationForest(contamination=0.025, random_state=42)
            iso_features = group[["residual", "consumption_kwh", "temperature_c", "hour"]]
            preds_iso = iso.fit_predict(iso_features)
            test_df.loc[idx, "anomaly_iforest"] = preds_iso == -1

        # Logs MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mae_kwh", mae)
        mlflow.log_metric("rmse_kwh", rmse)
        mlflow.log_metric("anomalies_zscore", int(test_df["anomaly_zscore"].sum()))
        mlflow.log_metric("anomalies_iqr", int(test_df["anomaly_iqr"].sum()))
        mlflow.log_metric("anomalies_iforest", int(test_df["anomaly_iforest"].sum()))

        joblib.dump(model, MODEL_DIR / "lgbm_energy.joblib")
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(
                {"building_mapping": building_mapping, "thresholds": thresholds, "features": features},
                f,
                indent=2,
            )

        test_df.to_csv(PROCESSED_DIR / "predictions.csv", index=False)

        print(f"✅ Entraînement et comparaison terminés !")
        print(f"   - MAE : {mae:.2f} kWh | RMSE : {rmse:.2f} kWh")
        print(f"   - Anomalies Z-Score (2.5 std) : {test_df['anomaly_zscore'].sum()}")
        print(f"   - Anomalies IQR (robuste)     : {test_df['anomaly_iqr'].sum()}")
        print(f"   - Anomalies Isolation Forest  : {test_df['anomaly_iforest'].sum()}")


if __name__ == "__main__":
    run_training()
