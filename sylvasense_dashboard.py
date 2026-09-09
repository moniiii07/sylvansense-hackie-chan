"""Streamlit dashboard for the checked-in SylvaSense demonstration artifacts."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from biomass_model import load_agbd_model, predict_agbd

ROOT = Path(__file__).resolve().parent
FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "VV", "VH", "VV_minus_VH"]

st.set_page_config(page_title="SylvaSense | Amazon Forest Intelligence", page_icon="🌳", layout="wide")
st.markdown('<h1>🌳 SylvaSense</h1><p>AI-powered Amazon forest intelligence using Sentinel-1, Sentinel-2, GEDI LiDAR and tree-canopy detection</p>', unsafe_allow_html=True)


def artifact(name: str) -> Path:
    return ROOT / name


def show_map(filename: str) -> None:
    path = artifact(filename)
    if path.exists():
        components.html(path.read_text(encoding="utf-8"), height=680, scrolling=True)
    else:
        st.warning(f"Missing file: {filename}")


@st.cache_data
def training_medians() -> dict[str, float]:
    data = pd.read_csv(artifact("agbd_training_samples.csv"))
    return {feature: float(data[feature].median()) for feature in FEATURES}


@st.cache_data
def validation_report() -> str:
    return artifact("agbd_validation_report.txt").read_text(encoding="utf-8")


report = validation_report()
metrics = dict(line.split(": ", 1) for line in report.splitlines() if ": " in line)
with st.sidebar:
    st.header("🌎 SylvaSense")
    st.markdown("**Forest Intelligence Platform**\n\n- 🛰️ Sentinel-2\n- 📡 Sentinel-1 SAR\n- 🌲 GEDI LiDAR\n- 🤖 Machine learning\n- 🌳 Tree-canopy detection")
    st.success("Amazon target region loaded")
    st.caption("SylvaSense • Hackathon Prototype")
canopy_path = artifact("deepforest_canopy_detections.geojson")
canopy_count = len(json.loads(canopy_path.read_text()).get("features", [])) if canopy_path.exists() else 0
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forest NDVI change", "-0.022")
col2.metric("Strong-loss area", "295.01 km²")
col3.metric("GEDI AGB model R²", metrics.get("R²", "Unavailable"))
col4.metric("Canopy detections", canopy_count)
st.info("Amazon change and biomass analysis use Sentinel-1, Sentinel-2, and GEDI. "
        "The DeepForest canopy export is a high-resolution technical demo.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌿 Forest change", "🛰️ Optical + SAR", "🌲 Biomass & carbon", "🌳 Tree intelligence", "🤖 Model insights",
])

with tab1:
    st.subheader("Amazon vegetation change")
    show_map("ndvi_comparison.html")

with tab2:
    st.subheader("Sentinel-2 optical and Sentinel-1 radar layers")
    show_map("target_satellite_layers.html")

with tab3:
    st.subheader("GEDI LiDAR aboveground biomass density")
    show_map("gedi_biomass.html")

with tab4:
    st.subheader("DeepForest canopy-detection export")
    geojson_path = artifact("deepforest_canopy_detections.geojson")
    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        st.success(f"{len(geojson['features'])} canopy detections exported to GeoJSON.")
        st.download_button("Download canopy GeoJSON", geojson_path.read_bytes(),
                           file_name=geojson_path.name, mime="application/geo+json")
        st.json(geojson["features"][:2])
    else:
        st.warning("Run deepforest_geojson_export.py first.")

with tab5:
    st.subheader("🤖 Local random-forest AGBD estimate")
    st.caption("Enter values in the same units used for training. Defaults are training-set medians.")
    defaults = training_medians()
    columns = st.columns(2)
    values = {}
    for index, feature in enumerate(FEATURES):
        with columns[index % 2]:
            values[feature] = st.number_input(feature, value=defaults[feature], format="%.6f")
    prediction, tree_spread = predict_agbd(values)
    metric_col, spread_col = st.columns(2)
    metric_col.metric("Estimated AGBD", f"{prediction:.1f} Mg/ha")
    spread_col.metric("Tree-to-tree spread", f"{tree_spread:.1f} Mg/ha")
    st.caption("Spread reflects variation among random-forest trees; it is not a calibrated confidence interval.")
    st.markdown("#### Validation")
    st.code(report, language="text")
    st.warning("R² is 0.162, so this is a prototype estimate and not suitable for operational carbon accounting.")
    importance_path = artifact("agbd_feature_importance.csv")
    st.bar_chart(pd.read_csv(importance_path).set_index("feature"))
    st.download_button("Download model validation report", artifact("agbd_validation_report.txt").read_bytes(),
                       file_name="agbd_validation_report.txt", mime="text/plain")
    st.download_button("Download feature importance", importance_path.read_bytes(),
                       file_name=importance_path.name, mime="text/csv")
    bundle = load_agbd_model(artifact("agbd_random_forest_model.joblib"))
    st.caption(f"Loaded checked-in model artifact with {len(bundle['features'])} predictors.")
