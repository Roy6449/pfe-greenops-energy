from pathlib import Path
import pandas as pd

# Configuration des chemins (compatible Mac/Linux/Windows)
RAW_DIR = Path("data") / "raw"
TARGET_BUILDINGS = [
    "Bear_education_Liliana",
    "Bear_education_Nanette",
    "Bear_education_Destiny",
]
SITE_ID = "Bear"


def run_ingestion():
    print("1. Chargement des fichiers sources BDG2...")
    # Utilise electricity.csv (ou electricity_cleaned.csv selon le fichier que tu as téléchargé)
    elec_file = RAW_DIR / "electricity_full.csv"
    if not elec_file.exists():
        elec_file = RAW_DIR / "electricity_cleaned.csv"

    df_elec = pd.read_csv(elec_file, usecols=["timestamp"] + TARGET_BUILDINGS)
    df_weather = pd.read_csv(RAW_DIR / "weather_full.csv")

    print("2. Passage au schéma pivot (long format)...")
    df_long = df_elec.melt(
        id_vars=["timestamp"],
        value_vars=TARGET_BUILDINGS,
        var_name="building_id",
        value_name="consumption_kwh",
    )

    print("3. Fusion avec la météo du site Bear...")
    weather_bear = df_weather[df_weather["site_id"] == SITE_ID][
        ["timestamp", "airTemperature"]
    ].rename(columns={"airTemperature": "temperature_c"})

    df_merged = pd.merge(df_long, weather_bear, on="timestamp", how="left")

    # Sauvegarde idempotente (écrase le fichier existant sans créer de doublons)
    output_path = RAW_DIR / "bear_3_buildings_raw.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Ingestion terminée : {len(df_merged)} lignes sauvegardées dans {output_path}")


if __name__ == "__main__":
    run_ingestion()
