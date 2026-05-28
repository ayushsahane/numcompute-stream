import numpy as np

from numcompute_stream import (
    Pipeline,
    StandardScaler,
    DecisionTreeClassifier,
    StreamTrainer,
)


def test_pipeline_import_and_basic_fit_predict():
    # Tiny toy data
    X = np.array([[0.0, 1.0], [1.0, 1.0], [10.0, 0.0]])
    y = np.array([0, 0, 1])

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])

    pipe.partial_fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape


def test_stream_trainer_logs_history():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    trainer = StreamTrainer(pipe)

    info = trainer.fit_chunk(X, y)
    assert "chunk_accuracy" in info
    assert "cumulative_accuracy" in info
    assert len(trainer.history_) == 1