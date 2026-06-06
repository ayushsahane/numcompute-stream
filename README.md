# NumCompute-Stream

A streaming decision tree–based machine learning framework built using **only NumPy and matplotlib**.
---

## Features

### Streaming Learning
- All components support incremental `.partial_fit(X_chunk, y_chunk)` updates
- Chunk-by-chunk training simulating online learning
- Numerical stability via Welford-style streaming statistics

### Decision Trees & Ensembles
- `DecisionTreeClassifier`: depth-limited, Gini/Entropy split criterion
- `EnsembleClassifier`: Bagging with multiple trees and majority voting
- Streaming adaptation per chunk via bootstrap sampling

### Preprocessing Pipeline
- `StandardScaler`: streaming running mean/variance (NaN-safe)
- `Imputer`: online missing-value imputation (mean strategy)
- `OneHotEncoder`: incremental categorical expansion
- `Pipeline`: chainable transformers → model with `.partial_fit()`

### Metrics & Logging
- **Accuracy**, **Precision/Recall/F1** (binary + macro multiclass)
- **Confusion Matrix** (accumulating)
- **AUC** for binary classification (via score curves)
- **Rolling window** metrics for recent-sample evaluation
- `StreamTrainer`: logs per-chunk metrics, cumulative accuracy, memory footprint

### Visualisation
- `plot_metric_over_time()`: track accuracy/loss per chunk
- `compare_models()`: side-by-side model comparison
- `plot_predictions_vs_ground_truth()`: visualise predictions on latest chunk
- `plot_confusion_matrix()`: heatmap of classification errors
- Save plots to file or display inline

---

## Installation

### Requirements
- Python 3.10+
- NumPy 2.0+ (or 1.x with compatibility fixes)
- Matplotlib 3.5+
- Pytest (for tests)

### Setup
```bash
git clone https://github.com/yourusername/numcompute-stream.git
cd numcompute-stream

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows

# Install dependencies
pip install -U pip
pip install numpy matplotlib pytest

# For testing the assignment & running the benchmark
pytest / pytest -v
python -m benchmark.benchmark_vectorised_vs_loop 