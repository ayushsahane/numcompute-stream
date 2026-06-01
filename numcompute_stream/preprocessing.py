"""
- StandardScaler:
    * partial_fit ignores NaNs in statistics
    * transform keeps NaNs as NaNs (it does not magically fill missing values)
    * You can use Imputer before StandardScaler in a Pipeline if you want no NaNs.
- Imputer:
    * strategy = "mean" only (simple, stable, common)
    * partial_fit updates running mean using StreamingStats
- OneHotEncoder:
    * keeps an ordered list of categories per column
    * partial_fit adds new categories when they appear in later chunks
    * transform outputs a consistent number/order of columns based on learned categories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np

from .utils import as_2d_array
from .stats import StreamingStats


class StandardScaler:
    """
    StandardScaler for streaming data.

    transform(X) applies:
        (X - mean_) / sqrt(var_ + eps)
    """
    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)

        self._stats = StreamingStats()

        # Public fitted attributes (set after partial_fit)
        self.n_seen_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")
        self._stats.update_stats(X)

        self.n_seen_ = self._stats.n_seen_.copy()
        self.mean_ = self._stats.mean_.copy()
        self.var_ = self._stats.variance().copy()
        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.mean_ is None or self.var_ is None:
            raise ValueError("StandardScaler is not fitted yet. Call partial_fit first.")

        Z = X.astype(float, copy=True)

        # Keep NaNs as NaNs. Operations with NaN stay NaN, which is what we want.
        if self.with_mean:
            Z = Z - self.mean_
        if self.with_std:
            Z = Z / np.sqrt(self.var_ + self.eps)
        return Z


class Imputer:
    """
    Streaming mean imputer (NaN -> running mean).

    partial_fit updates running mean (ignoring NaNs).
    transform fills NaNs using current running mean.
    """
    def __init__(self, strategy: str = "mean"):
        if strategy != "mean":
            raise ValueError("Only strategy='mean' is supported for this assignment implementation.")
        self.strategy = strategy
        self._stats = StreamingStats()

        self.fill_: Optional[np.ndarray] = None

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")
        self._stats.update_stats(X)
        self.fill_ = self._stats.mean_.copy()
        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.fill_ is None:
            raise ValueError("Imputer is not fitted yet. Call partial_fit first.")

        Z = X.astype(float, copy=True)
        mask = np.isnan(Z)
        if np.any(mask):
            # mask is 2D; mask indices for columns tell us which feature mean to use
            cols = np.where(mask)[1]
            Z[mask] = np.take(self.fill_, cols)
        return Z


class OneHotEncoder:
    """
    Incremental OneHotEncoder.

    We treat each feature column as categorical.
    - partial_fit collects new categories across chunks.
    - transform returns concatenated one-hot blocks (one block per column).

    Handling NaNs:
    - NaN is treated as "missing category" and encoded as all zeros for that column.
      (Simple and usually acceptable.)
    """
    def __init__(self, dtype=float):
        self.dtype = dtype
        self.categories_: Optional[List[np.ndarray]] = None

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")

        if self.categories_ is None:
            self.categories_ = [np.array([], dtype=float) for _ in range(X.shape[1])]

        if len(self.categories_) != X.shape[1]:
            raise ValueError("Number of features changed between chunks for OneHotEncoder.")

        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            uniq = np.unique(col)

            # Merge existing categories with new ones, keep sorted unique order
            merged = np.unique(np.concatenate([self.categories_[j], uniq]))
            self.categories_[j] = merged

        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted yet. Call partial_fit first.")
        if len(self.categories_) != X.shape[1]:
            raise ValueError("Number of features in transform does not match fitted encoder.")

        blocks = []
        for j, cats in enumerate(self.categories_):
            n = X.shape[0]
            k = cats.size
            block = np.zeros((n, k), dtype=self.dtype)

            if k == 0:
                blocks.append(block)
                continue

            col = X[:, j]
            # For each category value, mark equality (vectorised per category)
            # This uses a small loop over number of categories (usually not huge).
            for idx, v in enumerate(cats):
                block[:, idx] = (col == v).astype(self.dtype)

            # NaNs become all zeros automatically because (NaN == v) is False.
            blocks.append(block)

        return np.concatenate(blocks, axis=1) if blocks else np.zeros((X.shape[0], 0), dtype=self.dtype)