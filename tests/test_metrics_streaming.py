import numpy as np

from numcompute_stream.metrics import (
    Accuracy,
    ConfusionMatrix,
    BinaryPrecisionRecallF1,
    MacroPrecisionRecallF1,
    RollingMetric,
    StreamingAUC,
)


def test_accuracy_streaming_matches_batch():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = rng.integers(0, 3, size=200)

    a = Accuracy()
    a.update(y_true[:50], y_pred[:50])
    a.update(y_true[50:120], y_pred[50:120])
    a.update(y_true[120:], y_pred[120:])

    assert np.isclose(a.result(), np.mean(y_true == y_pred))


def test_confusion_matrix_multiclass_shape_and_counts():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 1, 0])

    cm = ConfusionMatrix(n_classes=3)
    cm.update(y_true[:3], y_pred[:3])
    cm.update(y_true[3:], y_pred[3:])

    M = cm.result()
    assert M.shape == (3, 3)
    assert M.sum() == len(y_true)
    assert M[0, 0] == 2  # two correct zeros


def test_binary_precision_recall_f1_known_case():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    # tp=2 (positions 0,2), fp=1 (pos 3), fn=1 (pos1)

    m = BinaryPrecisionRecallF1()
    m.update(y_true, y_pred)
    p, r, f1 = m.result()

    assert np.isclose(p, 2 / 3)
    assert np.isclose(r, 2 / 3)
    assert np.isclose(f1, 2 / 3)


def test_macro_precision_recall_f1_multiclass_runs():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 1, 0])

    m = MacroPrecisionRecallF1(n_classes=3)
    m.update(y_true[:3], y_pred[:3])
    m.update(y_true[3:], y_pred[3:])

    p, r, f1 = m.result()
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_rolling_metric_accuracy_window():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    roll = RollingMetric(metric_factory=lambda: Accuracy(), window_size=3)
    roll.update(y_true[:2], y_pred[:2])      # window: [1,1] vs [1,0]
    roll.update(y_true[2:], y_pred[2:])      # now last 3 pairs are indices 2,3,4

    # last 3 true = [0,0,1], pred=[0,1,1] => 2 correct out of 3
    assert np.isclose(roll.result(), 2 / 3)


def test_streaming_auc_basic_separable():
    # Perfectly separable: positives have higher scores than negatives => AUC ~ 1
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    auc = StreamingAUC()
    auc.update(y_true[:2], y_score[:2])
    auc.update(y_true[2:], y_score[2:])

    assert auc.result() > 0.99