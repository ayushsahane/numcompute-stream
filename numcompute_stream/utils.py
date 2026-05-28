"""
Small internal helpers used across the codebase.
"""

from __future__ import annotations
import numpy as np


def as_2d_array(X, name: str = "X") -> np.ndarray:
    """
    Ensure X is a 2D NumPy float array.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features). Got ndim={X.ndim}.")
    return X.astype(float, copy=False)


def as_1d_array(y, name: str = "y") -> np.ndarray:
    """
    Ensure y is a 1D NumPy array.

    Expected shape: (n_samples,)
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of shape (n_samples,). Got ndim={y.ndim}.")
    return y