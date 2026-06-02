"""
- Metrics support streaming updates via update(y_true_chunk, y_pred_chunk)
- reset(), result() methods
- Confusion matrix and AUC accumulate over time
- Include support for rolling-window metrics
- Multiclass metrics: accuracy + confusion matrix + macro precision/recall/F1
- Binary metrics: precision/recall/F1 + AUC (requires scores)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Deque, Tuple, Any
from collections import deque

import numpy as np

from .utils import as_1d_array


def _safe_div(num: float, den: float) -> float:
    """Return num/den, but if den==0 return 0.0 (avoids crashes)."""
    return float(num / den) if den != 0 else 0.0



# 1) Basic streaming metrics
@dataclass
class Accuracy:
    """Streaming accuracy = correct / total."""
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
        return _safe_div(self.correct_, self.total_)


@dataclass
class ConfusionMatrix:
    """
    Streaming confusion matrix for multiclass classification.
    """
    n_classes: int
    cm_: Optional[np.ndarray] = None

    def reset(self):
        self.cm_ = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, y_true, y_pred):
        y_true = as_1d_array(y_true, "y_true").astype(int, copy=False)
        y_pred = as_1d_array(y_pred, "y_pred").astype(int, copy=False)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        if self.cm_ is None:
            self.reset()

        # Filter out-of-range safely
        mask = (
            (y_true >= 0) & (y_true < self.n_classes) &
            (y_pred >= 0) & (y_pred < self.n_classes)
        )
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Vectorised bincount:
        # index = true * n_classes + pred
        idx = y_true * self.n_classes + y_pred
        counts = np.bincount(idx, minlength=self.n_classes * self.n_classes)
        self.cm_ += counts.reshape(self.n_classes, self.n_classes)

    def result(self) -> np.ndarray:
        if self.cm_ is None:
            self.reset()
        return self.cm_.copy()




# 2) Precision / Recall / F1 (binary + macro)
@dataclass
class BinaryPrecisionRecallF1:
    """
    Streaming precision/recall/F1 for binary classification.

    Assumes labels are 0/1.
    """
    tp_: int = 0
    fp_: int = 0
    fn_: int = 0

    def reset(self):
        self.tp_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def update(self, y_true, y_pred):
        y_true = as_1d_array(y_true, "y_true").astype(int, copy=False)
        y_pred = as_1d_array(y_pred, "y_pred").astype(int, copy=False)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        self.tp_ += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.fp_ += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.fn_ += int(np.sum((y_true == 1) & (y_pred == 0)))

    def precision(self) -> float:
        return _safe_div(self.tp_, (self.tp_ + self.fp_))

    def recall(self) -> float:
        return _safe_div(self.tp_, (self.tp_ + self.fn_))

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return _safe_div(2 * p * r, (p + r))

    def result(self) -> Tuple[float, float, float]:
        """Returns (precision, recall, f1)."""
        return (self.precision(), self.recall(), self.f1())


@dataclass
class MacroPrecisionRecallF1:
    """
    Macro-averaged precision/recall/F1 for multiclass streaming classification.

    We keep a confusion matrix and compute macro metrics from it.
    """
    n_classes: int
    cm_: ConfusionMatrix = None

    def __post_init__(self):
        self.cm_ = ConfusionMatrix(self.n_classes)

    def reset(self):
        self.cm_.reset()

    def update(self, y_true, y_pred):
        self.cm_.update(y_true, y_pred)

    def result(self) -> Tuple[float, float, float]:
        cm = self.cm_.result().astype(float)

        # For each class i:
        # TP_i = cm[i,i]
        # FP_i = sum over rows except i in column i
        # FN_i = sum over cols except i in row i
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        precision_i = np.array([_safe_div(tp[i], tp[i] + fp[i]) for i in range(self.n_classes)])
        recall_i = np.array([_safe_div(tp[i], tp[i] + fn[i]) for i in range(self.n_classes)])
        f1_i = np.array([_safe_div(2 * precision_i[i] * recall_i[i], precision_i[i] + recall_i[i])
                         for i in range(self.n_classes)])

        return (float(np.mean(precision_i)), float(np.mean(recall_i)), float(np.mean(f1_i)))



# 3) Rolling window wrapper
class RollingMetric:
    """
    - reset()
    - update(y_true, y_pred)
    - result()
    - keep last N pairs (y_true, y_pred)
    - recompute metric from scratch over the window when result() is called
    """
    def __init__(self, metric_factory: Any, window_size: int = 200):
        """
        metric_factory: callable that returns a NEW metric instance
        """
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.metric_factory = metric_factory
        self.window_size = int(window_size)
        self._buf_true: Deque[int] = deque(maxlen=self.window_size)
        self._buf_pred: Deque[int] = deque(maxlen=self.window_size)

    def reset(self):
        self._buf_true.clear()
        self._buf_pred.clear()

    def update(self, y_true, y_pred):
        y_true = as_1d_array(y_true, "y_true")
        y_pred = as_1d_array(y_pred, "y_pred")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        for t, p in zip(y_true, y_pred):
            self._buf_true.append(int(t))
            self._buf_pred.append(int(p))

    def result(self):
        m = self.metric_factory()
        m.reset()
        if len(self._buf_true) == 0:
            return m.result()
        m.update(np.array(self._buf_true), np.array(self._buf_pred))
        return m.result()



# 4) Streaming AUC (binary)
class StreamingAUC:
    """
    - update(y_true_chunk, y_score_chunk)
    - reset()
    - result() -> float
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._y_true = np.empty((0,), dtype=int)
        self._y_score = np.empty((0,), dtype=float)

    def update(self, y_true, y_score):
        y_true = as_1d_array(y_true, "y_true").astype(int, copy=False)
        y_score = as_1d_array(y_score, "y_score").astype(float, copy=False)
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score must have the same length.")
        self._y_true = np.concatenate([self._y_true, y_true])
        self._y_score = np.concatenate([self._y_score, y_score])

    def result(self) -> float:
        if self._y_true.size == 0:
            return 0.0

        y_true = self._y_true
        y_score = self._y_score

        # Guard: binary only
        uniq = np.unique(y_true)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError("StreamingAUC supports only binary labels {0,1}.")

        P = int(np.sum(y_true == 1))
        N = int(np.sum(y_true == 0))
        if P == 0 or N == 0:
            return 0.0

        # Sort by score descending
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        tpr = tps / P
        fpr = fps / N

        # prepend (0,0) for trapezoid integration
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])

        return float(np.trapezoid(tpr, fpr))