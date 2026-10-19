"""Small, serializable transformations used in feature experiments."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeCycle(BaseEstimator, TransformerMixin):
    """24-hour phase relative to the dataset's first transaction."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        seconds = np.asarray(X, dtype=float).reshape(-1)
        phase = 2 * np.pi * seconds / 86400
        return np.column_stack((np.sin(phase), np.cos(phase)))
