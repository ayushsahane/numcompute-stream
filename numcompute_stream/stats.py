"""
streaming stats.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import as_2d_array


@dataclass
class StreamingStats:
    """
    Keep running statistics for streaming data.

    Attributes updated via update_stats(X_chunk):
    - n_seen
    - mean
    - var
    """
    n_seen: int = 0
    mean: np.ndarray | None = None
    var: np.ndarray | None = None

    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        X = as_2d_array(X_chunk, "X_chunk")
        self.n_seen += X.shape[0]
        self.mean = np.nanmean(X, axis=0)
        self.var = np.nanvar(X, axis=0)
        return self