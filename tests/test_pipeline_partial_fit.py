import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_partial_fit_and_predict_runs():
    X = np.array([[1.0, np.nan], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("imp", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])

    pipe.partial_fit(X[:2], y[:2])
    pipe.partial_fit(X[2:], y[2:])

    pred = pipe.predict(X)
    assert pred.shape == y.shape


def test_pipeline_raises_if_y_missing():
    X = np.array([[1.0], [2.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier())])
    with pytest.raises(ValueError):
        pipe.partial_fit(X, None)


def test_pipeline_requires_transformer_transform():
    class BadTransformer:
        def partial_fit(self, X, y=None):
            return self

    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    pipe = Pipeline([("bad", BadTransformer()), ("model", DecisionTreeClassifier())])
    with pytest.raises(TypeError):
        pipe.partial_fit(X, y)


def test_pipeline_requires_model_predict():
    class BadModel:
        def partial_fit(self, X, y):
            return self

    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    pipe = Pipeline([("scale", StandardScaler()), ("badmodel", BadModel())])
    pipe.partial_fit(X, y)
    with pytest.raises(TypeError):
        pipe.predict(X)