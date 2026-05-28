"""
EnsembleClassifier 
"""

from __future__ import annotations
import numpy as np
from .tree import DecisionTreeClassifier
from .utils import as_2d_array, as_1d_array


class EnsembleClassifier:
    def __init__(self, n_estimators: int = 5, random_state: int | None = None, **tree_kwargs):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0.")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.tree_kwargs = dict(tree_kwargs)

        self.estimators_ = [
            DecisionTreeClassifier(random_state=random_state, **self.tree_kwargs)
            for _ in range(n_estimators)
        ]

    def partial_fit(self, X_chunk, y_chunk):
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")
        for est in self.estimators_:
            est.partial_fit(X, y)
        return self

    def predict(self, X):
        X = as_2d_array(X, "X")
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)
        # majority vote
        # For simplicity, assume class labels are integers.
        out = []
        for i in range(preds.shape[1]):
            values, counts = np.unique(preds[:, i], return_counts=True)
            out.append(values[int(np.argmax(counts))])
        return np.array(out, dtype=int)