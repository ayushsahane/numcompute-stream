"""
- supports streaming via .partial_fit(...)
- transformers may have: partial_fit, fit, transform
- model may have: partial_fit, fit, predict

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np

from .utils import as_2d_array, as_1d_array


@dataclass
class Pipeline:
    """
            Pipeline([
                ("scale", StandardScaler()),
                ("model", EnsembleClassifier(...))
            ])
    """
    steps: List[Tuple[str, Any]]

    def _split(self):
        if not self.steps:
            raise ValueError("Pipeline.steps is empty.")
        *transformers, (model_name, model) = self.steps
        return transformers, (model_name, model)

    def partial_fit(self, X, y=None):
        """
        Streaming training: update each transformer then update the model.
        """
        X = as_2d_array(X, "X")
        if y is not None:
            y = as_1d_array(y, "y")

        transformers, (_, model) = self._split()

        Xt = X
        # 1) update transformers
        for _, t in transformers:
            if hasattr(t, "partial_fit"):
                t.partial_fit(Xt, y)
            elif hasattr(t, "fit"):
                # fallback for non-streaming transformer (still acceptable for small chunks)
                t.fit(Xt, y)
            if hasattr(t, "transform"):
                Xt = t.transform(Xt)

        # 2) update model
        if hasattr(model, "partial_fit"):
            model.partial_fit(Xt, y)
        elif hasattr(model, "fit"):
            model.fit(Xt, y)
        else:
            raise TypeError("Final pipeline step must be a model with partial_fit() or fit().")

        return self

    def predict(self, X):
        """Run transformers then model.predict(X)."""
        X = as_2d_array(X, "X")
        transformers, (_, model) = self._split()

        Xt = X
        for _, t in transformers:
            if hasattr(t, "transform"):
                Xt = t.transform(Xt)

        if not hasattr(model, "predict"):
            raise TypeError("Final pipeline step must implement predict().")
        return model.predict(Xt)