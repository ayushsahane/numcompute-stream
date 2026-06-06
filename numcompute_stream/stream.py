"""
Here: 
StreamTrainer: trains a pipeline chunk-by-chunk and logs metrics + memory footprint.

- Implements StreamTrainer managing model + pipeline + logging
- Supports .fit_chunk(X, y) and .score_chunk(X, y) logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import sys

from .metrics import Accuracy
from .utils import as_2d_array, as_1d_array


def _estimate_memory_bytes(obj) -> int:
    """Rough estimate of object memory usage in bytes."""
    try:
        return sys.getsizeof(obj)
    except:
        return 0


@dataclass
class StreamTrainer:
    """
    Trainer for streaming learning pipelines.

    Manages:
    - A pipeline (transformers + model)
    - A metric object (for accumulating results)
    - History of per-chunk metrics
    - Memory footprint tracking

    """

    pipeline: Any
    metric: Any = field(default_factory=Accuracy)
    history_: List[Dict[str, float]] = field(default_factory=list)

    def fit_chunk(self, X_chunk, y_chunk) -> Dict[str, float]:
        """
        Train on one chunk and log metrics.
        """
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        # 1) Train on chunk
        self.pipeline.partial_fit(X, y)

        # 2) Predict on chunk
        y_pred = self.pipeline.predict(X)

        # 3) Update accumulating metric
        self.metric.update(y, y_pred)

        # 4) Compute per-chunk metrics
        chunk_acc = float(np.mean(y == y_pred))
        cumulative_acc = float(self.metric.result())
        n_samples = float(X.shape[0])

        # 5) Estimate memory (rough)
        mem_bytes = _estimate_memory_bytes(self.pipeline)

        info = {
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": cumulative_acc,
            "n_samples_in_chunk": n_samples,
            "memory_pipeline_bytes": float(mem_bytes),
        }

        self.history_.append(info)
        return info

    def score_chunk(self, X_chunk, y_chunk) -> float:
        """
        Evaluate on a test chunk (no training, no metric update).

        Returns accuracy on this chunk only.
        """
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")
        y_pred = self.pipeline.predict(X)
        return float(np.mean(y == y_pred))

    def reset_metric(self):
        """Reset the accumulating metric (but keep history)."""
        if hasattr(self.metric, "reset"):
            self.metric.reset()