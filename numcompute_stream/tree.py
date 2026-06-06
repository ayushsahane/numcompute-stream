"""
DecisionTreeClassifier.

- depth-limited
- split criterion: gini or entropy
- partial_fit(X_chunk, y_chunk) for streaming settings
- support config: max_depth, min_samples_split, max_features

- partial_fit appends data to an internal buffer
- then we rebuild the tree from the buffered data
  (this is still "streaming" because you train chunk-by-chunk)

- NaNs:
  - while evaluating splits for a feature, we ignore rows where that feature is NaN
  - during prediction, if the split feature is NaN, we send the row to the
    majority side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import as_2d_array, as_1d_array


def _gini(y: np.ndarray) -> float:
    """Gini impurity for labels y."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


def _entropy(y: np.ndarray) -> float:
    """Entropy impurity for labels y."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # add tiny epsilon to avoid log2(0)
    eps = 1e-12
    return float(-np.sum(p * np.log2(p + eps)))


def _majority_class(y: np.ndarray) -> int:
    """Return the majority class. Tie breaks by choosing the smallest label."""
    values, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    winners = values[counts == max_count]
    return int(np.min(winners))


@dataclass
class _Node:
    # Leaf fields
    is_leaf: bool
    pred_class: int

    # Split fields (used if not leaf)
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    # If feature value is NaN at prediction time, choose this side:
    # 0 => left, 1 => right
    nan_go_to: int = 0


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be >= 1 or None.")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.buffer_size = buffer_size  # optional cap for streaming buffer

        self.rng_ = np.random.default_rng(self.random_state)

        # Model state
        self.root_: Optional[_Node] = None
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

        # Streaming buffer (stores all seen chunks unless buffer_size is set)
        self._X_buf = None
        self._y_buf = None

    def partial_fit(self, X_chunk, y_chunk):
        X = as_2d_array(X_chunk, "X_chunk")
        y = as_1d_array(y_chunk, "y_chunk").astype(int, copy=False)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features changed between chunks.")

        # Update classes seen so far
        self.classes_ = np.unique(y) if self.classes_ is None else np.unique(np.concatenate([self.classes_, np.unique(y)]))

        # Append to buffer
        if self._X_buf is None:
            self._X_buf = X.copy()
            self._y_buf = y.copy()
        else:
            self._X_buf = np.vstack([self._X_buf, X])
            self._y_buf = np.concatenate([self._y_buf, y])

        # Optional: keep only the most recent buffer_size samples (sliding memory)
        if self.buffer_size is not None and self._y_buf.size > self.buffer_size:
            excess = self._y_buf.size - self.buffer_size
            self._X_buf = self._X_buf[excess:]
            self._y_buf = self._y_buf[excess:]

        # Rebuild tree from buffered data
        self.root_ = self._build_tree(self._X_buf, self._y_buf, depth=0)
        return self

    def fit(self, X, y):
        """Non-streaming fit = reset buffer then partial_fit once."""
        self._X_buf = None
        self._y_buf = None
        self.root_ = None
        self.n_features_in_ = None
        self.classes_ = None
        return self.partial_fit(X, y)

    def predict(self, X):
        X = as_2d_array(X, "X")
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet.")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than the fitted tree.")

        out = np.empty((X.shape[0],), dtype=int)
        for i in range(X.shape[0]):
            out[i] = self._predict_one(self.root_, X[i])
        return out


    # Internal tree construction


    def _impurity(self, y: np.ndarray) -> float:
        return _gini(y) if self.criterion == "gini" else _entropy(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float, int]:
        """
        Find the best split.

        Returns
        -------
        best_feature, best_threshold, best_gain, nan_go_to

        gain is impurity decrease.
        """
        n_samples, n_features = X.shape
        parent_imp = self._impurity(y)

        if parent_imp == 0.0:
            return -1, 0.0, 0.0, 0

        # Choose feature subset if max_features is set (like RandomForest)
        feat_idx = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feat_idx = self.rng_.choice(feat_idx, size=self.max_features, replace=False)

        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_nan_go_to = 0

        for j in feat_idx:
            col = X[:, j]
            not_nan = ~np.isnan(col)
            if np.sum(not_nan) < self.min_samples_split:
                continue

            xj = col[not_nan]
            yj = y[not_nan]

            # Sort by feature value so we can consider thresholds efficiently
            order = np.argsort(xj)
            x_sorted = xj[order]
            y_sorted = yj[order]

            # Candidate thresholds are midpoints between unique values
            uniq = np.unique(x_sorted)
            if uniq.size <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            # Evaluate each threshold (loop is acceptable; core ops are vectorised where possible)
            for thr in thresholds:
                left_mask = x_sorted <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                y_left = y_sorted[left_mask]
                y_right = y_sorted[right_mask]

                # Weighted impurity
                nL = y_left.size
                nR = y_right.size
                child_imp = (nL / (nL + nR)) * self._impurity(y_left) + (nR / (nL + nR)) * self._impurity(y_right)

                gain = parent_imp - child_imp
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(j)
                    best_threshold = float(thr)

                    # Decide where NaNs should go during prediction:
                    # send NaNs to the side with MORE training samples (more stable)
                    best_nan_go_to = 0 if nL >= nR else 1

        return best_feature, best_threshold, best_gain, best_nan_go_to

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # Leaf prediction is always the majority class at this node
        pred = _majority_class(y)

        # Stopping conditions
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(is_leaf=True, pred_class=pred)
        if y.size < self.min_samples_split:
            return _Node(is_leaf=True, pred_class=pred)
        if np.unique(y).size == 1:
            return _Node(is_leaf=True, pred_class=pred)

        feat, thr, gain, nan_go_to = self._best_split(X, y)
        if feat == -1 or gain <= 0.0:
            return _Node(is_leaf=True, pred_class=pred)

        col = X[:, feat]
        # For training split: NaNs are sent to the majority side too (same as prediction rule)
        nan_mask = np.isnan(col)
        left_mask = (col <= thr) & ~nan_mask
        right_mask = (col > thr) & ~nan_mask

        if nan_go_to == 0:
            left_mask = left_mask | nan_mask
        else:
            right_mask = right_mask | nan_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # If split is degenerate, return leaf
        if y_left.size == 0 or y_right.size == 0:
            return _Node(is_leaf=True, pred_class=pred)

        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)

        return _Node(
            is_leaf=False,
            pred_class=pred,
            feature_index=feat,
            threshold=thr,
            left=left_node,
            right=right_node,
            nan_go_to=nan_go_to,
        )

    def _predict_one(self, node: _Node, x_row: np.ndarray) -> int:
        while not node.is_leaf:
            v = x_row[node.feature_index]
            if np.isnan(v):
                node = node.left if node.nan_go_to == 0 else node.right
            else:
                node = node.left if v <= node.threshold else node.right
        return int(node.pred_class)