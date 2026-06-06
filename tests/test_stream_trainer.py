import numpy as np

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.metrics import Accuracy


def test_stream_trainer_logs_history():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3)),
    ])
    trainer = StreamTrainer(pipe, metric=Accuracy())

    X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
    y = np.array([0, 1, 0, 1])

    info1 = trainer.fit_chunk(X[:2], y[:2])
    info2 = trainer.fit_chunk(X[2:], y[2:])

    assert len(trainer.history_) == 2
    assert "chunk_accuracy" in info1
    assert "cumulative_accuracy" in info1
    assert "n_samples_in_chunk" in info1
    assert "memory_pipeline_bytes" in info1

    assert info1["cumulative_accuracy"] <= 1.0
    assert info2["cumulative_accuracy"] <= 1.0


def test_stream_trainer_score_chunk_no_train():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    trainer = StreamTrainer(pipe, metric=Accuracy())

    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([[0.5], [2.5]])
    y_test = np.array([0, 1])

    trainer.fit_chunk(X_train, y_train)
    test_acc = trainer.score_chunk(X_test, y_test)

    assert 0.0 <= test_acc <= 1.0
    # trainer.history should only have train chunk, not test
    assert len(trainer.history_) == 1
    
def test_stream_trainer_reset_metric():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    trainer = StreamTrainer(pipe, metric=Accuracy())

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    trainer.fit_chunk(X[:2], y[:2])
    acc1 = trainer.metric.result()
    assert acc1 > 0.0

    trainer.reset_metric()
    acc_after_reset = trainer.metric.result()
    assert acc_after_reset == 0.0

    # History should still be there
    assert len(trainer.history_) == 1
    
def test_stream_trainer_empty_history_start():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    trainer = StreamTrainer(pipe)
    assert len(trainer.history_) == 0


def test_stream_trainer_memory_footprint_nonzero():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    trainer = StreamTrainer(pipe)
    
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    
    info = trainer.fit_chunk(X, y)
    assert info["memory_pipeline_bytes"] > 0