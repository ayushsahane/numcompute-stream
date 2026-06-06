"""
Reusable matplotlib plotting functions for streaming metrics.
- plot_metric_over_time(metric_values, title, ylabel)
- compare_models(metric1, metric2, labels)
- plot_predictions_vs_ground_truth(y_true, y_pred)
- Support options for saving to file or inline display
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values: Sequence[float],
    title: str,
    ylabel: str,
    xlabel: str = "chunk",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a metric value across chunks.
    """
    x = np.arange(len(metric_values))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, metric_values, marker="o", linewidth=2, markersize=6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def compare_models(
    metric1: Sequence[float],
    metric2: Sequence[float],
    labels: Sequence[str] = ("Model 1", "Model 2"),
    title: str = "Model Comparison",
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Compare two models on a streaming metric.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x1 = np.arange(len(metric1))
    x2 = np.arange(len(metric2))

    ax.plot(x1, metric1, marker="o", label=labels[0], linewidth=2, markersize=6)
    ax.plot(x2, metric2, marker="s", label=labels[1], linewidth=2, markersize=6)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("chunk", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs Ground Truth",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot predictions vs actual labels on the latest chunk.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y_true, marker="o", label="Ground Truth", linewidth=2, markersize=6)
    ax.plot(x, y_pred, marker="x", label="Prediction", linewidth=2, markersize=8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


def plot_confusion_matrix(
    cm,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a confusion matrix as a heatmap.

    """
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax