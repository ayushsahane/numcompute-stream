"""
pipeline.py

- support partial_fit() for models and transformers
- Support incremental transformation + prediction in chained pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .utils import as_2d_array, as_1d_array


@dataclass
class Pipeline:
    steps: List[Tuple[str, Any]]

    def __post_init__(self):
        if not isinstance(self.steps, list) or len(self.steps) == 0:
            raise ValueError("Pipeline.steps must be a non-empty list of (name, obj) tuples.")
        for step in self.steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise ValueError("Each pipeline step must be a tuple: (name, obj).")

    def _split(self):
        if len(self.steps) < 1:
            raise ValueError("Pipeline must have at least one step.")
        *transformers, (model_name, model) = self.steps
        return transformers, (model_name, model)

    def partial_fit(self, X, y=None):
        """
        Stream-training on a chunk.

        X: (n_samples, n_features)
        y: (n_samples,) or None

        Returns self.
        """
        X = as_2d_array(X, "X")
        if y is not None:
            y = as_1d_array(y, "y")

        transformers, (_, model) = self._split()

        Xt = X

        # 1) Update + transform using transformers
        for name, t in transformers:
            # Update transformer
            if hasattr(t, "partial_fit"):
                t.partial_fit(Xt, y)
            elif hasattr(t, "fit"):
                t.fit(Xt, y)
            else:
                raise TypeError(
                    f"Transformer step '{name}' must implement partial_fit() or fit()."
                )

            # Transform
            if not hasattr(t, "transform"):
                raise TypeError(f"Transformer step '{name}' must implement transform().")
            Xt = t.transform(Xt)

            Xt = as_2d_array(Xt, f"Output of transformer '{name}'")

        # 2) Update model
        if y is None:
            raise ValueError("Pipeline.partial_fit requires y for supervised classification.")

        if hasattr(model, "partial_fit"):
            model.partial_fit(Xt, y)
        elif hasattr(model, "fit"):
            model.fit(Xt, y)
        else:
            raise TypeError("Final pipeline step (model) must implement partial_fit() or fit().")

        return self

    def predict(self, X):
        """Transform X through transformers then call model.predict(X)."""
        X = as_2d_array(X, "X")
        transformers, (_, model) = self._split()

        Xt = X
        for name, t in transformers:
            if not hasattr(t, "transform"):
                raise TypeError(f"Transformer step '{name}' must implement transform().")
            Xt = t.transform(Xt)
            Xt = as_2d_array(Xt, f"Output of transformer '{name}'")

        if not hasattr(model, "predict"):
            raise TypeError("Final pipeline step (model) must implement predict().")
        y_pred = model.predict(Xt)
        y_pred = as_1d_array(y_pred, "y_pred")
        return y_pred