"""
StreamTrainer: trains chunk-by-chunk and logs metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np

from .metrics import Accuracy
from .utils import as_2d_array, as_1d_array


@dataclass
class StreamTrainer:
    pipeline: Any
    metric: Any = field(default_factory=Accuracy)
    history_: List[Dict[str, float]] = field(default_factory=list)

    def fit_chunk(self, X_chunk, y_chunk) -> Dict[str, float]:
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")

        self.pipeline.partial_fit(X, y)
        y_pred = self.pipeline.predict(X)

        self.metric.update(y, y_pred)
        info = {
            "chunk_accuracy": float(np.mean(y == y_pred)),
            "cumulative_accuracy": float(self.metric.result()),
            "n_samples_seen": float(len(y)),
        }
        self.history_.append(info)
        return info

    def score_chunk(self, X_chunk, y_chunk) -> float:
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")
        y_pred = self.pipeline.predict(X)
        return float(np.mean(y == y_pred))