# model_pipeline.py
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Custom Transformer to handle feature engineering inside the pipeline
class AddTotalAdsTransformer(BaseEstimator, TransformerMixin):
    """Adds a 'total_ads_expenses' feature by summing the 3 input ad columns."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to numpy array for consistent mathematical operations
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        # Sum across columns (axis=1) and reshape to a column vector
        total_ads = np.sum(X_arr, axis=1).reshape(-1, 1)
        
        # Concatenate original 3 features with the new 'total_ads' feature
        return np.hstack([X_arr, total_ads])

# 2. Load Dataset
# Ensure 'Advertising_Data.xlsx' is in the same folder as this script
df = pd.read_excel('Advertising_Data.xlsx')

# 3. Define Features (X) and Target (y)
# FIX: Only use the 3 raw input features. Removed target-leaked 'expenses_per_unit'.
X = df[['tv_ads', 'social_media_ads', 'print_ads']]
y = df['sales']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build the End-to-End Pipeline
pipeline = Pipeline([
    ('add_total_ads', AddTotalAdsTransformer()),          # Step 1: Feature Engineering
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)), # Step 2: Preprocessing
    ('lin_reg', LinearRegression())                       # Step 3: Modeling
])

# 6. Train the Pipeline
pipeline.fit(X_train, y_train)

# 7. Evaluate Metrics
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

print("--- Training Metrics ---")
print(f"MAE:  {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"R2:   {r2_score(y_train, y_pred_train):.4f}\n")

print("--- Testing Metrics ---")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"R2:   {r2_score(y_test, y_pred_test):.4f}")
