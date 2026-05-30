import numpy as np

from numcompute_stream.stats import StreamingStats, StreamingQuantile, StreamingHistogram


def test_streaming_stats_matches_batch_mean_var_no_nans():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))

    s = StreamingStats()
    s.update_stats(X[:50])
    s.update_stats(X[50:120])
    s.update_stats(X[120:])

    assert np.allclose(s.mean_, X.mean(axis=0))
    assert np.allclose(s.variance(), X.var(axis=0))


def test_streaming_stats_ignores_nans_per_feature():
    X = np.array(
        [
            [1.0, np.nan],
            [3.0, 10.0],
            [np.nan, 14.0],
            [5.0, np.nan],
        ]
    )

    s = StreamingStats().update_stats(X[:2]).update_stats(X[2:])

    # Feature 0 uses values [1,3,5]
    assert np.isclose(s.mean_[0], np.mean([1.0, 3.0, 5.0]))
    # Feature 1 uses values [10,14]
    assert np.isclose(s.mean_[1], np.mean([10.0, 14.0]))


def test_streaming_quantile_basic():
    q = StreamingQuantile(reservoir_size=128, random_state=0)
    X = np.arange(1000, dtype=float)
    q.update(X[:500])
    q.update(X[500:])
    # Should be near the true median ~ 499.5 (approx because reservoir)
    med = q.result(0.5)
    assert 350 < med < 650


def test_streaming_histogram_counts_sum_to_n():
    h = StreamingHistogram(bins=5, range=(0.0, 1.0))
    X = np.linspace(0.0, 1.0, 101)
    h.update(X)
    counts, edges = h.result()
    assert counts.sum() == 101
    assert edges.shape == (6,)