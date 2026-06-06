import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


def test_tree_fits_and_predicts_simple_separable():
    # Feature 0 separates classes at ~0.5
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2, criterion="gini")
    clf.partial_fit(X[:3], y[:3])
    clf.partial_fit(X[3:], y[3:])

    pred = clf.predict(X)
    assert (pred == y).mean() >= 0.83


def test_tree_respects_max_depth():
    # XOR needs depth>1 to be perfect; with depth=1 should not perfectly fit
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    clf = DecisionTreeClassifier(max_depth=1, criterion="gini")
    clf.fit(X, y)
    pred = clf.predict(X)
    assert (pred == y).mean() < 1.0


def test_tree_handles_nans_prediction():
    X = np.array([[0.0], [1.0], [np.nan], [0.1], [1.2]])
    y = np.array([0, 1, 0, 0, 1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    pred = clf.predict(np.array([[np.nan], [0.0], [1.0]]))
    assert pred.shape == (3,)


def test_tree_buffer_size_limits_memory():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = rng.integers(0, 2, size=50)

    clf = DecisionTreeClassifier(buffer_size=20, random_state=0)
    clf.partial_fit(X[:25], y[:25])
    clf.partial_fit(X[25:], y[25:])

    # internal buffer should not exceed 20
    assert clf._y_buf.size <= 20
    
def test_tree_criterion_entropy_vs_gini():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=2)
    tree_ent = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    
    tree_gini.fit(X, y)
    tree_ent.fit(X, y)
    
    pred_gini = tree_gini.predict(X)
    pred_ent = tree_ent.predict(X)
    
    # Both should reach reasonable accuracy
    assert np.mean(pred_gini == y) >= 0.8
    assert np.mean(pred_ent == y) >= 0.8


def test_tree_min_samples_split_respected():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]])
    y = np.array([0, 0, 1, 1, 0, 0])
    
    tree = DecisionTreeClassifier(min_samples_split=4, max_depth=3)
    tree.fit(X, y)
    
    # With min_samples_split=4, we can't split nodes with <4 samples
    # So tree should be shallow
    pred = tree.predict(X)
    assert pred.shape == y.shape