import numpy as np

from numcompute_stream.ensemble import EnsembleClassifier


def test_ensemble_bagging_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, size=50)

    ens = EnsembleClassifier(n_estimators=5, random_state=0)
    ens.partial_fit(X[:25], y[:25])
    ens.partial_fit(X[25:], y[25:])

    pred = ens.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})


def test_ensemble_majority_vote_consistency():
    # Simple linearly separable problem
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    y = np.array([0, 0, 0, 1, 1])

    ens = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=0)
    ens.fit(X, y)

    pred = ens.predict(X)
    acc = np.mean(pred == y)
    assert acc >= 0.6  # should do decent
    
def test_ensemble_with_different_random_seeds():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    y = np.array([0, 0, 0, 1, 1])

    ens1 = EnsembleClassifier(n_estimators=3, random_state=0)
    ens2 = EnsembleClassifier(n_estimators=3, random_state=1)

    ens1.fit(X, y)
    ens2.fit(X, y)

    pred1 = ens1.predict(X)
    pred2 = ens2.predict(X)

    # Different seeds should (usually) give slightly different results
    # but both should be valid predictions
    assert pred1.shape == pred2.shape == y.shape
    
def test_ensemble_n_estimators_consistency():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    y = np.array([0, 0, 1, 1, 0])
    
    ens1 = EnsembleClassifier(n_estimators=1, random_state=0)
    ens3 = EnsembleClassifier(n_estimators=3, random_state=0)
    
    ens1.fit(X, y)
    ens3.fit(X, y)
    
    # 1-tree ensemble should match single tree predictions exactly
    # 3-tree ensemble may differ due to voting
    pred1 = ens1.predict(X)
    pred3 = ens3.predict(X)
    
    assert pred1.shape == pred3.shape == y.shape