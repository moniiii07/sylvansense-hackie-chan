import math
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("agbd_training_samples.csv")

features = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "NDVI", "VV", "VH", "VV_minus_VH",
]

X = data[features]
y = data["agbd"]

if len(data) < 30:
    raise ValueError(
        f"Only {len(data)} GEDI samples were found. "
        "At least 30 are needed for a meaningful first model."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

joblib.dump(
    {"model": model, "features": features},
    "agbd_random_forest_model.joblib",
)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv("agbd_feature_importance.csv", index=False)

report = (
    "SylvaSense AGB regression validation\n"
    "===================================\n"
    f"Total GEDI-labelled samples: {len(data)}\n"
    f"Training samples: {len(X_train)}\n"
    f"Test samples: {len(X_test)}\n"
    f"MAE: {mae:.2f} Mg/ha\n"
    f"RMSE: {rmse:.2f} Mg/ha\n"
    f"R²: {r2:.3f}\n"
)

with open("agbd_validation_report.txt", "w") as file:
    file.write(report)

print(report)
print("Created agbd_random_forest_model.joblib")
print("Created agbd_feature_importance.csv")
print("Created agbd_validation_report.txt")