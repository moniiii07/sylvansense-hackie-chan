import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="SylvaSense", layout="wide")

st.title("SylvaSense")
st.caption("Amazon forest monitoring • optical + SAR + LiDAR + canopy detection")

col1, col2, col3 = st.columns(3)
col1.metric("Forest NDVI change", "-0.022")
col2.metric("Strong-loss area", "295.01 km²")
col3.metric("GEDI AGB model R²", "0.162")

st.info(
    "Amazon change and biomass analysis use Sentinel-1, Sentinel-2, and GEDI. "
    "The DeepForest canopy export is a high-resolution technical demo."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Forest change",
    "Optical + SAR",
    "GEDI biomass",
    "Canopy GeoJSON",
])


def show_map(filename):
    path = Path(filename)

    if path.exists():
        components.html(path.read_text(), height=680, scrolling=True)
    else:
        st.warning(f"Missing file: {filename}")


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

    geojson_path = Path("deepforest_canopy_detections.geojson")

    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text())
        count = len(geojson["features"])

        st.success(f"{count} canopy detections exported to GeoJSON.")
        st.download_button(
            "Download canopy GeoJSON",
            data=geojson_path.read_bytes(),
            file_name="deepforest_canopy_detections.geojson",
            mime="application/geo+json",
        )

        st.json(geojson["features"][:2])
    else:
        st.warning("Run deepforest_geojson_export.py first.")