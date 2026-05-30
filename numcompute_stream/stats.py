"""
Streaming statistics utilities.

- StreamingStats: running mean/variance (per-feature)
- StreamingQuantile: approximate streaming quantiles using a fixed-size reservoir
- StreamingHistogram: fixed-bin histogram that can be updated per chunk
- Welford's algorithm lets us update mean/variance without storing all past data.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .utils import as_2d_array


def _nan_count(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Count non-NaN values along an axis (vectorised)."""
    return np.sum(~np.isnan(X), axis=axis)


@dataclass
class StreamingStats:
    """Running per-feature mean/variance using streaming Welford-style updates.

    This class is the base building block for StandardScaler.

    For scaling, population variance is commonly used.
    """

    n_seen_: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    M2_: Optional[np.ndarray] = None

    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        """Updates running mean/variance using a new chunk.
        """
        X = as_2d_array(X_chunk, "X_chunk")
        n_features = X.shape[1]

        if self.n_seen_ is None:
            # First call: initialise state.
            self.n_seen_ = np.zeros((n_features,), dtype=np.int64)
            self.mean_ = np.zeros((n_features,), dtype=float)
            self.M2_ = np.zeros((n_features,), dtype=float)

        # Chunk stats (ignoring NaNs per feature)
        nB = _nan_count(X, axis=0).astype(np.int64)

        meanB = np.nanmean(X, axis=0)
        varB = np.nanvar(X, axis=0)  # population variance

        # If a whole feature column is NaN in this chunk, nanmean/nanvar returns NaN.
        # Replace with 0; we will skip those features using mask (nB>0).
        meanB = np.where(np.isnan(meanB), 0.0, meanB)
        varB = np.where(np.isnan(varB), 0.0, varB)

        M2B = varB * nB

        nA = self.n_seen_
        meanA = self.mean_
        M2A = self.M2_

        n = nA + nB
        mask = nB > 0  # only update features that had at least 1 value in this chunk

        delta = meanB - meanA

        new_mean = meanA.copy()
        new_mean[mask] = (nA[mask] * meanA[mask] + nB[mask] * meanB[mask]) / n[mask]

        new_M2 = M2A.copy()
        new_M2[mask] = (
            M2A[mask]
            + M2B[mask]
            + (delta[mask] ** 2) * (nA[mask] * nB[mask]) / n[mask]
        )

        self.n_seen_ = n
        self.mean_ = new_mean
        self.M2_ = new_M2
        return self

    def variance(self) -> np.ndarray:
        """Return population variance per feature (safe for n=0)."""
        if self.n_seen_ is None:
            raise ValueError("StreamingStats is not initialised. Call update_stats first.")
        n = self.n_seen_.astype(float)
        return np.where(n > 0, self.M2_ / n, 0.0)


class StreamingQuantile:
    """Approximate streaming quantiles using a reservoir sample.
    """

    def __init__(self, reservoir_size: int = 2048, random_state: int | None = None):
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be > 0")
        self.reservoir_size = int(reservoir_size)
        self.rng = np.random.default_rng(random_state)
        self._reservoir = np.empty((0,), dtype=float)
        self._n_seen = 0

    def reset(self):
        self._reservoir = np.empty((0,), dtype=float)
        self._n_seen = 0

    def update(self, X_chunk: np.ndarray):
        x = np.asarray(X_chunk, dtype=float).ravel()
        x = x[~np.isnan(x)]
        if x.size == 0:
            return

        # Reservoir sampling loop (simple and correct).
        for val in x:
            self._n_seen += 1
            if self._reservoir.size < self.reservoir_size:
                self._reservoir = np.append(self._reservoir, val)
            else:
                j = self.rng.integers(0, self._n_seen)
                if j < self.reservoir_size:
                    self._reservoir[j] = val

    def result(self, q: float | Iterable[float]):
        if self._reservoir.size == 0:
            if isinstance(q, Iterable):
                return np.array([np.nan for _ in q], dtype=float)
            return np.nan
        return np.quantile(self._reservoir, q)


class StreamingHistogram:
    """Streaming histogram with fixed bin edges.
    """

    def __init__(self, bins: int = 10, range: tuple[float, float] = (0.0, 1.0)):
        if bins <= 0:
            raise ValueError("bins must be > 0")
        self.bins = int(bins)
        self.range = (float(range[0]), float(range[1]))
        self.edges_ = np.linspace(self.range[0], self.range[1], self.bins + 1)
        self.counts_ = np.zeros((self.bins,), dtype=np.int64)

    def reset(self):
        self.counts_[:] = 0

    def update(self, X_chunk: np.ndarray):
        x = np.asarray(X_chunk, dtype=float).ravel()
        x = x[~np.isnan(x)]
        if x.size == 0:
            return
        counts, _ = np.histogram(x, bins=self.edges_)
        self.counts_ += counts.astype(np.int64)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (counts, edges)."""
        return self.counts_.copy(), self.edges_.copy()