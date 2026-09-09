"""Utilities for running the versioned local AGBD regression artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).with_name("agbd_random_forest_model.joblib")


def load_agbd_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the model bundle and validate its artifact structure."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or {"model", "features"} - bundle.keys():
        raise ValueError("AGBD model artifact must contain 'model' and 'features'.")
    if not bundle["features"]:
        raise ValueError("AGBD model artifact has no predictor features.")
    return bundle


def predict_agbd(values: Mapping[str, float], model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[float, float]:
    """Return estimated AGBD and random-forest tree-to-tree spread in Mg/ha."""
    bundle = load_agbd_model(model_path)
    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing model inputs: {', '.join(missing)}")
    row = pd.DataFrame([{feature: float(values[feature]) for feature in features}])
    matrix = row.to_numpy()
    estimator = bundle["model"]
    prediction = float(estimator.predict(matrix)[0])
    if not hasattr(estimator, "estimators_"):
        return prediction, float("nan")
    tree_predictions = np.asarray([tree.predict(matrix)[0] for tree in estimator.estimators_])
    return prediction, float(tree_predictions.std(ddof=0))
