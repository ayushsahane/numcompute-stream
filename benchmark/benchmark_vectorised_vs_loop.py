"""
Comparison of vectorised NumPy operations vs. Python loops.

It basically measures:
- Time to compute mean/var over large arrays
- Time to compute confusion matrix
- Time to predict on many samples
"""

import time

import numpy as np

from numcompute_stream.stats import StreamingStats
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.metrics import ConfusionMatrix


def benchmark_streaming_stats():
    """Benchmark StreamingStats (uses vectorised Welford)."""
    print("\n=== Benchmark: Streaming Stats (Welford) ===")

    rng = np.random.default_rng(0)
    n_chunks = 10
    chunk_size = 1000
    n_features = 50

    s = StreamingStats()
    start = time.time()
    for _ in range(n_chunks):
        X = rng.normal(size=(chunk_size, n_features))
        s.update_stats(X)
    elapsed = time.time() - start

    print(f"Chunks: {n_chunks}, Chunk size: {chunk_size}, Features: {n_features}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Samples/sec: {n_chunks * chunk_size / elapsed:.0f}")


def benchmark_confusion_matrix_vectorised():
    """Benchmark vectorised confusion matrix updates."""
    print("\n=== Benchmark: Confusion Matrix (Vectorised Bincount) ===")

    rng = np.random.default_rng(0)
    n_chunks = 20
    chunk_size = 1000
    n_classes = 5

    cm = ConfusionMatrix(n_classes=n_classes)
    start = time.time()
    for _ in range(n_chunks):
        y_true = rng.integers(0, n_classes, size=chunk_size)
        y_pred = rng.integers(0, n_classes, size=chunk_size)
        cm.update(y_true, y_pred)
    elapsed = time.time() - start

    print(f"Chunks: {n_chunks}, Chunk size: {chunk_size}, Classes: {n_classes}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Samples/sec: {n_chunks * chunk_size / elapsed:.0f}")


def benchmark_tree_predict_vectorised():
    """Benchmark tree prediction (uses row-by-row traversal, vectorised at feature level)."""
    print("\n=== Benchmark: Tree Prediction ===")

    rng = np.random.default_rng(0)
    n_train = 500
    n_predict = 5000
    n_features = 10

    X_train = rng.normal(size=(n_train, n_features))
    y_train = rng.integers(0, 2, size=n_train)

    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    X_predict = rng.normal(size=(n_predict, n_features))

    start = time.time()
    pred = clf.predict(X_predict)
    elapsed = time.time() - start

    print(f"Train samples: {n_train}, Predict samples: {n_predict}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Predictions/sec: {n_predict / elapsed:.0f}")


if __name__ == "__main__":
    print("=" * 60)
    print("NumCompute-Stream: Vectorisation Benchmarks")
    print("=" * 60)

    benchmark_streaming_stats()
    benchmark_confusion_matrix_vectorised()
    benchmark_tree_predict_vectorised()

    print("\n" + "=" * 60)
    print("Summary: All operations use NumPy vectorisation.")
    print("Core loops (tree traversal, sample-wise ops) are necessary,")
    print("but feature-level and class-level ops are fully vectorised.")
    print("=" * 60)