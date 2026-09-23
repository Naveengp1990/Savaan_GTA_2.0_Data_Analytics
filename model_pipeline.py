# ============================================================
# Advertising Sales Prediction
# Model Training & Serialization
# ============================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Import custom transformer from a separate module
from custom_transformer import AddTotalAdsTransformer


# ============================================================
# 1. Configuration
# ============================================================

DATA_FILE = "Advertising_Data.xlsx"
MODEL_FILE = "advertising_model.pkl"
METADATA_FILE = "model_metadata.pkl"

FEATURES = [
    "tv_ads",
    "social_media_ads",
    "print_ads"
]

TARGET = "sales"


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_excel(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. Define Features and Target
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 5. Build ML Pipeline
# ============================================================

pipeline = Pipeline([
    
    (
        "add_total_ads",
        AddTotalAdsTransformer()
    ),

    (
        "poly_features",
        PolynomialFeatures(
            degree=3,
            include_bias=False
        )
    ),

    (
        "linear_regression",
        LinearRegression()
    )
])


# ============================================================
# 6. Train
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 7. Predictions
# ============================================================

y_pred_train = pipeline.predict(X_train)

y_pred_test = pipeline.predict(X_test)


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate_model(y_true, y_pred, name):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n--- {name} Metrics ---")

    print(f"MAE :  {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²  :  {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


train_metrics = evaluate_model(
    y_train,
    y_pred_train,
    "Training"
)

test_metrics = evaluate_model(
    y_test,
    y_pred_test,
    "Testing"
)


# ============================================================
# 9. Save Complete Pipeline
# ============================================================

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    f"\nModel saved successfully: {MODEL_FILE}"
)


# ============================================================
# 10. Save Metadata
# ============================================================

metadata = {
    "features": FEATURES,
    "target": TARGET,
    "polynomial_degree": 3,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics
}

joblib.dump(
    metadata,
    METADATA_FILE
)

print(
    f"Metadata saved successfully: {METADATA_FILE}"
)


# ============================================================
# 11. Verify Saved Model
# ============================================================

loaded_model = joblib.load(
    MODEL_FILE
)


sample_input = pd.DataFrame({
    "tv_ads": [100],
    "social_media_ads": [50],
    "print_ads": [20]
})


prediction = loaded_model.predict(
    sample_input
)


print("\n--- Model Verification ---")

print("\nInput:")
print(sample_input)

print(
    f"\nPredicted Sales: {prediction[0]:.4f}"
)

print(
    "\nModel saved and loaded successfully!"
)
