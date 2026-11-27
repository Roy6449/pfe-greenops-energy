from pathlib import Path
import pandas as pd

RAW_PATH = Path("data") / "raw" / "bear_3_buildings_raw.csv"

def run_eda():
    df = pd.read_csv(RAW_PATH)
    print("--- APERÇU ---")
    print(df.info())
    print("\n--- VALEURS MANQUANTES ---")
    print(df.isna().sum())
    print("\n--- STATS PAR BÂTIMENT ---")
    print(df.groupby("building_id")["consumption_kwh"].describe())

if __name__ == "__main__":
    run_eda()
