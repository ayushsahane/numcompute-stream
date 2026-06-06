"""
In this i'm using: 
EnsembleClassifier: manages multiple decision trees with bootstrap aggregating (Bagging).

- Manages N decision trees with at least one ensemble method
- .partial_fit() and .predict() must support streaming adaptation

Ensemble method: BAGGING 
- For each chunk, each tree trains on a bootstrap sample of the chunk
- This is how Random Forest-like ensembles work online according to me
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .tree import DecisionTreeClassifier
from .utils import as_2d_array, as_1d_array


class EnsembleClassifier:
    """
    Bagging ensemble of decision trees.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        random_state: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")

        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        # Create ensemble of trees
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=None if random_state is None else random_state + i,
            )
            for i in range(n_estimators)
        ]

        self.rng_ = np.random.default_rng(random_state)
        self.classes_ = None

    def partial_fit(self, X_chunk, y_chunk):
        """
        Stream training: for each tree, train on a bootstrap sample of the chunk.

        X_chunk: (n_samples, n_features)
        y_chunk: (n_samples,)
        """
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk").astype(int, copy=False)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        # Track classes
        self.classes_ = np.unique(y) if self.classes_ is None else np.unique(
            np.concatenate([self.classes_, np.unique(y)])
        )

        # Bootstrap aggregating: each tree gets a random sample with replacement
        for est in self.estimators_:
            indices = self.rng_.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            est.partial_fit(X_boot, y_boot)

        return self

    def fit(self, X, y):
        """Non-streaming fit = partial_fit once."""
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i,
            )
            for i in range(self.n_estimators)
        ]
        return self.partial_fit(X, y)

    def predict(self, X):
        """
        Predict by majority vote among trees.

        X: (n_samples, n_features)
        Returns: (n_samples,) array of predictions
        """
        X = as_2d_array(X, "X")
        if len(self.estimators_) == 0 or self.estimators_[0].root_ is None:
            raise ValueError("EnsembleClassifier is not fitted yet.")

        # Get predictions from all trees
        all_preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)

        # Majority vote (vectorised)
        out = np.empty((X.shape[0],), dtype=int)
        for i in range(X.shape[0]):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            max_count = np.max(counts)
            winners = values[counts == max_count]
            out[i] = int(np.min(winners))  # break ties by smallest label

        return out