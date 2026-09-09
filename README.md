# SylvaSense

Amazon forest-monitoring prototype combining Sentinel-2 optical imagery,
Sentinel-1 radar, GEDI biomass observations, and a DeepForest canopy demo.

## Run the dashboard

Install `requirements.txt`, then run `streamlit run sylvasense_dashboard.py`.
It uses checked-in maps and the saved model, so it starts without Earth Engine access.

## AGBD model status

The model is a 300-tree random-forest regressor trained on 3,000 GEDI-labelled
samples using six Sentinel-2 bands, NDVI, Sentinel-1 VV/VH, and VV-minus-VH.

| Metric | Value |
| --- | ---: |
| MAE | 65.36 Mg/ha |
| RMSE | 84.72 Mg/ha |
| R² | 0.162 |

The dashboard shows these values beside every local prediction. The low R²
means the model is a prototype, not suitable for operational carbon accounting
until it is improved and independently validated.

## Regenerate Earth Engine artifacts

Earth Engine scripts require authenticated credentials and the configured GCP
project. Run `create_target_polygon.py`, `prepare_satellite_layers.py`,
`prepare_gedi_biomass.py`, `build_agbd_training_data.py`, and
`train_agbd_model.py` in that order. `live_agbd_inference.py` produces a
separate Earth Engine prototype map.

Run `pytest` to validate the local model bundle. Remote Earth Engine scripts
are excluded from test discovery.
