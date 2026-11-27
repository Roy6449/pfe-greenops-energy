from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="GreenOps Energy Monitor",
    page_icon="🌿",
    layout="wide",
)

DATA_PATH = Path("data") / "processed" / "predictions.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.title("🌿 Data-Driven Energy — Monitoring & Détection d'Anomalies")
st.caption("Démarche LeanOps & GreenOps • Prédiction LightGBM & Analyse comparative des dérives")

if not DATA_PATH.exists():
    st.error("Fichier predictions.csv introuvable. Lance d'abord `python src/train.py`.")
    st.stop()

df = load_data()

# --- BARRE LATÉRALE (FILTRES) ---
st.sidebar.header("⚙️ Paramètres d'analyse")

buildings = sorted(df["building_id"].unique())
selected_building = st.sidebar.selectbox("Bâtiment surveillé", buildings)

method_map = {
    "Z-Score (2.5 σ — Sensible aux pics)": "anomaly_zscore",
    "IQR Robuste (Conservateur — Zéro faux positif)": "anomaly_iqr",
    "Isolation Forest (ML Contextuel — Heure/Météo)": "anomaly_iforest",
}
selected_method_label = st.sidebar.selectbox(
    "Méthode de détection d'anomalies", list(method_map.keys())
)
anomaly_col = method_map[selected_method_label]

df_b = df[df["building_id"] == selected_building].sort_values("timestamp").copy()

# Filtre temporel (30 derniers jours par défaut pour une lecture fluide)
min_date = df_b["timestamp"].min().date()
max_date = df_b["timestamp"].max().date()
default_start = max(min_date, (df_b["timestamp"].max() - pd.Timedelta(days=21)).date())

date_range = st.sidebar.date_input(
    "Période affichée",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    mask_date = (df_b["timestamp"].dt.date >= date_range[0]) & (
        df_b["timestamp"].dt.date <= date_range[1]
    )
    df_view = df_b[mask_date].copy()
else:
    df_view = df_b.copy()

# --- CARTES KPI ---
anomalies_df = df_view[df_view[anomaly_col]]
mae_building = (df_view["consumption_kwh"] - df_view["predicted_kwh"]).abs().mean()
overconsumption = anomalies_df[anomalies_df["residual"] > 0]["residual"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Consommation moyenne", f"{df_view['consumption_kwh'].mean():.1f} kWh")
col2.metric("Erreur modèle (MAE)", f"{mae_building:.1f} kWh")
col3.metric(
    "Anomalies détectées",
    f"{len(anomalies_df)}",
    f"{(len(anomalies_df) / max(len(df_view), 1) * 100):.1f} % des relevés",
    delta_color="inverse",
)
col4.metric("Surconsommation détectée", f"{overconsumption:.0f} kWh")

st.divider()

# --- GRAPHIQUE INTERACTIF PLOTLY ---
st.subheader(f"📈 Courbe de charge & Alertes — {selected_building}")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_view["timestamp"],
        y=df_view["consumption_kwh"],
        mode="lines",
        name="Consommation Réelle (kWh)",
        line=dict(color="#1B4332", width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_view["timestamp"],
        y=df_view["predicted_kwh"],
        mode="lines",
        name="Prédiction IA (LightGBM)",
        line=dict(color="#52B788", width=1.8, dash="dot"),
    )
)

fig.add_trace(
    go.Scatter(
        x=anomalies_df["timestamp"],
        y=anomalies_df["consumption_kwh"],
        mode="markers",
        name="Anomalie détectée",
        marker=dict(color="#D62828", size=9, symbol="circle", line=dict(width=1, color="white")),
    )
)

fig.update_layout(
    template="plotly_white",
    height=460,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Date & Heure",
    yaxis_title="Consommation électrique (kWh)",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# --- COMPARATIF DES 3 APPROCHES & JOURNAL DES ALERTES ---
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("⚖️ Comparaison des approches")
    comp_data = pd.DataFrame(
        {
            "Méthode": ["Z-Score (2.5 σ)", "IQR Robuste", "Isolation Forest"],
            "Alertes (période)": [
                int(df_view["anomaly_zscore"].sum()),
                int(df_view["anomaly_iqr"].sum()),
                int(df_view["anomaly_iforest"].sum()),
            ],
            "Alertes (total test)": [
                int(df_b["anomaly_zscore"].sum()),
                int(df_b["anomaly_iqr"].sum()),
                int(df_b["anomaly_iforest"].sum()),
            ],
        }
    )
    st.dataframe(comp_data, use_container_width=True, hide_index=True)
    st.info(
        "**Lecture métier :** L'IQR filtre uniquement les dérives sévères (idéal pour éviter la fatigue d'alerte), "
        "tandis qu'Isolation Forest repère les anomalies contextuelles (ex. surconsommation nocturne ou week-end)."
    )

with col_right:
    st.subheader("🚨 Dernières alertes à investiguer")
    if len(anomalies_df) > 0:
        display_cols = ["timestamp", "consumption_kwh", "predicted_kwh", "residual", "temperature_c"]
        alerts_table = (
            anomalies_df[display_cols]
            .sort_values("timestamp", ascending=False)
            .head(10)
            .rename(
                columns={
                    "timestamp": "Horodatage",
                    "consumption_kwh": "Réel (kWh)",
                    "predicted_kwh": "Prédit (kWh)",
                    "residual": "Écart (kWh)",
                    "temperature_c": "Temp. (°C)",
                }
            )
        )
        st.dataframe(alerts_table.round(1), use_container_width=True, hide_index=True)
    else:
        st.success("Aucune anomalie détectée sur cette période avec cette méthode.")
