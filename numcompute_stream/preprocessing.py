"""
- StandardScaler (streaming mean/var later)
- Imputer (streaming fill values later)
- OneHotEncoder (incremental categories later)
"""

from __future__ import annotations
import numpy as np
from .utils import as_2d_array


class StandardScaler:
    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.n_seen_ = 0
        self.mean_ = None
        self.var_ = None

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")
        # Placeholder: store simple batch stats (will be replaced with true streaming update)
        self.n_seen_ += X.shape[0]
        self.mean_ = np.nanmean(X, axis=0)
        self.var_ = np.nanvar(X, axis=0)
        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.mean_ is None:
            raise ValueError("StandardScaler is not fitted yet. Call partial_fit first.")
        Z = X.copy()
        if self.with_mean:
            Z = Z - self.mean_
        if self.with_std:
            Z = Z / np.sqrt(self.var_ + self.eps)
        return Z


class Imputer:
    def __init__(self, strategy: str = "mean"):
        if strategy not in {"mean"}:
            raise ValueError("Only strategy='mean' ")
        self.strategy = strategy
        self.fill_ = None

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")
        self.fill_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.fill_ is None:
            raise ValueError("Imputer is not fitted yet. Call partial_fit first.")
        Z = X.copy()
        mask = np.isnan(Z)
        Z[mask] = np.take(self.fill_, np.where(mask)[1])
        return Z


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None  # list of arrays per column

    def partial_fit(self, X, y=None):
        X = as_2d_array(X, "X")
        # treat columns as categorical strings/numbers; gather unique values per column
        cats = []
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col[~np.isnan(col)])
            cats.append(uniq)
        self.categories_ = cats
        return self

    def transform(self, X):
        X = as_2d_array(X, "X")
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted yet. Call partial_fit first.")
        # Placeholder: minimal encoding (will be improved)
        outs = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            col_out = np.zeros((X.shape[0], len(cats)), dtype=float)
            for k, v in enumerate(cats):
                col_out[:, k] = (col == v).astype(float)
            outs.append(col_out)
        return np.concatenate(outs, axis=1) if outs else np.zeros((X.shape[0], 0), dtype=float)