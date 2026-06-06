import numpy as np

from numcompute_stream.preprocessing import StandardScaler, Imputer, OneHotEncoder


def test_standard_scaler_streaming_matches_batch_no_nans():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))

    sc = StandardScaler()
    sc.partial_fit(X[:50])
    sc.partial_fit(X[50:140])
    sc.partial_fit(X[140:])

    # Running stats should match batch stats (no NaNs)
    assert np.allclose(sc.mean_, X.mean(axis=0))
    assert np.allclose(sc.var_, X.var(axis=0))


def test_standard_scaler_ignores_nans_in_partial_fit():
    X1 = np.array([[1.0, np.nan], [3.0, 10.0]])
    X2 = np.array([[np.nan, 14.0], [5.0, np.nan]])

    sc = StandardScaler().partial_fit(X1).partial_fit(X2)

    # Feature 0 values seen: [1,3,5]
    assert np.isclose(sc.mean_[0], np.mean([1.0, 3.0, 5.0]))
    # Feature 1 values seen: [10,14]
    assert np.isclose(sc.mean_[1], np.mean([10.0, 14.0]))


def test_imputer_fills_nans_with_running_mean():
    X1 = np.array([[1.0, np.nan], [3.0, 10.0]])
    X2 = np.array([[5.0, 14.0]])

    imp = Imputer().partial_fit(X1).partial_fit(X2)

    X_test = np.array([[np.nan, np.nan], [3.0, np.nan]])
    Z = imp.transform(X_test)

    # Column means from observed values:
    # col0: [1,3,5] mean=3
    # col1: [10,14] mean=12
    assert np.allclose(Z[0], [3.0, 12.0])
    assert np.allclose(Z[1], [3.0, 12.0])


def test_onehotencoder_incremental_categories_expand():
    X1 = np.array([[0.0], [1.0], [1.0]])
    X2 = np.array([[2.0], [0.0]])

    enc = OneHotEncoder()
    enc.partial_fit(X1)
    assert enc.categories_[0].tolist() == [0.0, 1.0]

    enc.partial_fit(X2)
    assert enc.categories_[0].tolist() == [0.0, 1.0, 2.0]

    Z = enc.transform(np.array([[2.0], [1.0], [0.0]]))
    # 3 categories => 3 columns
    assert Z.shape == (3, 3)
    assert (Z[0] == np.array([0.0, 0.0, 1.0])).all()
    assert (Z[1] == np.array([0.0, 1.0, 0.0])).all()
    assert (Z[2] == np.array([1.0, 0.0, 0.0])).all()


def test_onehotencoder_nan_encodes_all_zeros():
    enc = OneHotEncoder().partial_fit(np.array([[0.0], [1.0]]))
    Z = enc.transform(np.array([[np.nan], [0.0]]))
    assert (Z[0] == np.array([0.0, 0.0])).all()
    assert (Z[1] == np.array([1.0, 0.0])).all()
    
def test_imputer_zero_variance_feature():
    X1 = np.array([[5.0, 1.0], [5.0, 2.0]])
    X2 = np.array([[5.0, np.nan]])
    
    imp = Imputer().partial_fit(X1).partial_fit(X2)
    Z = imp.transform(np.array([[5.0, np.nan]]))
    
    # Feature 0 is constant (all 5), feature 1 has mean of [1,2] = 1.5
    assert np.isclose(Z[0, 0], 5.0)
    assert np.isclose(Z[0, 1], 1.5)


def test_scaler_with_constant_feature():
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    sc = StandardScaler().partial_fit(X)
    Z = sc.transform(X)
    
    # First feature is constant, variance = 0
    # After scaling: (1-1)/sqrt(0+eps) = 0
    assert np.allclose(Z[:, 0], 0.0, atol=1e-6)