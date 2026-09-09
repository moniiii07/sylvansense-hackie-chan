from pathlib import Path

import pytest

from biomass_model import load_agbd_model, predict_agbd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "agbd_random_forest_model.joblib"
SAMPLE = {"B2": 327.5, "B3": 509.0, "B4": 303.0, "B8": 2930.75, "B11": 1580.0,
          "B12": 655.5, "NDVI": 0.814965, "VV": -8.197785, "VH": -14.301755,
          "VV_minus_VH": 6.102407}


def test_model_artifact_has_expected_features():
    assert load_agbd_model(MODEL)["features"] == list(SAMPLE)


def test_prediction_is_finite_and_has_tree_spread():
    prediction, spread = predict_agbd(SAMPLE, MODEL)
    assert prediction >= 0
    assert spread >= 0


def test_prediction_rejects_missing_feature():
    incomplete = SAMPLE.copy()
    incomplete.pop("VV")
    with pytest.raises(ValueError, match="VV"):
        predict_agbd(incomplete, MODEL)
