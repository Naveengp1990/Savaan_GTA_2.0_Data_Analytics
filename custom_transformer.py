import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class AddTotalAdsTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a total advertising expenditure feature.

    total_ads_expenses =
        tv_ads + social_media_ads + print_ads
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Convert input to NumPy array
        X_array = np.asarray(X, dtype=float)

        # Calculate total advertising expenditure
        total_ads = X_array.sum(
            axis=1,
            keepdims=True
        )

        # Add total_ads to original features
        return np.hstack([
            X_array,
            total_ads
        ])
