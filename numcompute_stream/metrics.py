"""
Accuracy + ConfusionMatrix skeleton.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import as_1d_array


@dataclass
class Accuracy:
    correct_: int = 0
    total_: int = 0

    def reset(self):
        self.correct_ = 0
        self.total_ = 0

    def update(self, y_true, y_pred):
        y_true = as_1d_array(y_true, "y_true")
        y_pred = as_1d_array(y_pred, "y_pred")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        self.correct_ += int(np.sum(y_true == y_pred))
        self.total_ += int(y_true.shape[0])

    def result(self) -> float:
        return float(self.correct_ / self.total_) if self.total_ > 0 else 0.0


@dataclass
class ConfusionMatrix:
    n_classes: int
    cm_: np.ndarray | None = None

    def reset(self):
        self.cm_ = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def update(self, y_true, y_pred):
        y_true = as_1d_array(y_true, "y_true").astype(int, copy=False)
        y_pred = as_1d_array(y_pred, "y_pred").astype(int, copy=False)
        if self.cm_ is None:
            self.reset()
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                self.cm_[t, p] += 1

    def result(self) -> np.ndarray:
        if self.cm_ is None:
            self.reset()
        return self.cm_