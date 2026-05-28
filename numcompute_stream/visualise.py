"""
Reusable matplotlib plotting helpers.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values: Sequence[float], title: str, ylabel: str):
    x = np.arange(len(metric_values))
    plt.figure()
    plt.plot(x, metric_values, marker="o")
    plt.title(title)
    plt.xlabel("chunk")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    return plt.gca()


def compare_models(metric1: Sequence[float], metric2: Sequence[float], labels=("Model 1", "Model 2")):
    x = np.arange(max(len(metric1), len(metric2)))
    plt.figure()
    plt.plot(np.arange(len(metric1)), metric1, marker="o", label=labels[0])
    plt.plot(np.arange(len(metric2)), metric2, marker="o", label=labels[1])
    plt.title("Streaming metric comparison")
    plt.xlabel("chunk")
    plt.ylabel("metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gca()


def plot_predictions_vs_ground_truth(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure()
    plt.plot(y_true, label="y_true", marker="o")
    plt.plot(y_pred, label="y_pred", marker="x")
    plt.title("Predictions vs Ground Truth (latest chunk)")
    plt.xlabel("index")
    plt.ylabel("class")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gca()