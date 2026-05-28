"""
DecisionTreeClassifier
"""

from __future__ import annotations
import numpy as np
from .utils import as_2d_array, as_1d_array


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        # Simple baseline model state (placeholder)
        self.classes_ = None
        self.majority_class_ = None

    def partial_fit(self, X_chunk, y_chunk):
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_class_ = values[int(np.argmax(counts))]
        return self

    def predict(self, X):
        X = as_2d_array(X, "X")
        if self.majority_class_ is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet.")
        return np.full((X.shape[0],), self.majority_class_, dtype=int)