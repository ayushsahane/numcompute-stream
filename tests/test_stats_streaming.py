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
    
def test_streaming_quantile_empty_reset():
    q = StreamingQuantile()
    result1 = q.result(0.5)
    assert np.isnan(result1)
    
    q.reset()
    result2 = q.result(0.5)
    assert np.isnan(result2)


def test_streaming_histogram_bins_consistent():
    h1 = StreamingHistogram(bins=10, range=(0.0, 1.0))
    h2 = StreamingHistogram(bins=10, range=(0.0, 1.0))

    X = np.linspace(0, 1, 100)
    h1.update(X[:50])
    h1.update(X[50:])
    
    h2.update(X)
    
    counts1, edges1 = h1.result()
    counts2, edges2 = h2.result()
    
    assert np.allclose(counts1, counts2)
    assert np.allclose(edges1, edges2)