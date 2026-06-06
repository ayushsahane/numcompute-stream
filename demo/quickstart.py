"""
Shows the full streaming ML pipeline:
1. Load data (simulate with random)
2. Split into chunks
3. Train incrementally with partial_fit
4. Log metrics and visualise
"""

import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import Accuracy, ConfusionMatrix
from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
)


def main():
    print("=" * 70)
    print("NumCompute-Stream: Quickstart Demo")
    print("=" * 70)

    # 1. Generate synthetic streaming data 
    print("\n[1] Generating synthetic data...")
    rng = np.random.default_rng(42)
    n_samples_total = 2000
    n_features = 5
    n_chunks = 8
    chunk_size = n_samples_total // n_chunks

    X = rng.normal(loc=0.5, scale=2.0, size=(n_samples_total, n_features))
    # Add some NaNs to test robustness
    nan_mask = rng.random((n_samples_total, n_features)) < 0.05
    X[nan_mask] = np.nan

    # Binary classification: based on first feature
    y = (X[:, 0] > 0.5).astype(int)

    print(f"   Total samples: {n_samples_total}")
    print(f"   Features: {n_features}")
    print(f"   Chunks: {n_chunks}")
    print(f"   Chunk size: {chunk_size}")

    # 2. Build pipeline
    print("\n[2] Building streaming pipeline...")
    pipe_single = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=3, max_depth=4, random_state=0)),
    ])

    pipe_ensemble = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=5, max_depth=4, random_state=1)),
    ])

    # 3. Train on chunks
    print("\n[3] Training on chunks...")

    trainer_single = StreamTrainer(pipe_single, metric=Accuracy())
    trainer_ensemble = StreamTrainer(pipe_ensemble, metric=Accuracy())

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        X_chunk = X[start:end]
        y_chunk = y[start:end]

        info_s = trainer_single.fit_chunk(X_chunk, y_chunk)
        info_e = trainer_ensemble.fit_chunk(X_chunk, y_chunk)

        print(f"   Chunk {chunk_idx + 1}/{n_chunks}: "
              f"Single Acc={info_s['cumulative_accuracy']:.3f}, "
              f"Ensemble Acc={info_e['cumulative_accuracy']:.3f}")

    # 4. Extract metrics
    print("\n[4] Extracting metrics...")
    acc_single = [h["cumulative_accuracy"] for h in trainer_single.history_]
    acc_ensemble = [h["cumulative_accuracy"] for h in trainer_ensemble.history_]
    chunk_acc_single = [h["chunk_accuracy"] for h in trainer_single.history_]

    # 5. Visualise
    print("\n[5] Visualising results...")

    print("   - Plotting single model accuracy over time...")
    plot_metric_over_time(
        acc_single,
        title="Single Ensemble: Cumulative Accuracy Over Chunks",
        ylabel="Accuracy",
        save_path="plots/single_accuracy.png",
        show=False,
    )

    print("   - Comparing two ensemble models...")
    compare_models(
        acc_single,
        acc_ensemble,
        labels=("3-tree Ensemble", "5-tree Ensemble"),
        title="Ensemble Size Comparison",
        ylabel="Cumulative Accuracy",
        save_path="plots/ensemble_comparison.png",
        show=False,
    )

    print("   - Plotting chunk accuracy...")
    plot_metric_over_time(
        chunk_acc_single,
        title="Per-Chunk Accuracy",
        ylabel="Chunk Accuracy",
        save_path="plots/chunk_accuracy.png",
        show=False,
    )

    # 6. Final evaluation 
    print("\n[6] Final evaluation...")
    y_pred_single = pipe_single.predict(X)
    y_pred_ensemble = pipe_ensemble.predict(X)

    final_acc_single = np.mean(y_pred_single == y)
    final_acc_ensemble = np.mean(y_pred_ensemble == y)

    print(f"   Single Ensemble Final Accuracy: {final_acc_single:.4f}")
    print(f"   Multi Ensemble Final Accuracy: {final_acc_ensemble:.4f}")

    # Confusion matrices
    cm_single = ConfusionMatrix(n_classes=2)
    cm_ensemble = ConfusionMatrix(n_classes=2)

    cm_single.update(y, y_pred_single)
    cm_ensemble.update(y, y_pred_ensemble)

    print("\n   Confusion Matrix (Single):")
    print(cm_single.result())
    print("\n   Confusion Matrix (Ensemble):")
    print(cm_ensemble.result())

    print("\n" + "=" * 70)
    print("Demo complete! Check 'plots/' folder for visualisations.")
    print("=" * 70)


if __name__ == "__main__":
    main()