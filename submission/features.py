"""Label-free features computed from the candidates of one query."""
import numpy as np

from submission.data import group_sizes

PREFIX = "rankfusion_qrank_v1_"


def query_percentiles(X):
    """Midrank percentiles; equal values receive equal ranks, including constants."""
    X = np.asarray(X, dtype=np.float32)
    n, width = X.shape
    # Bound sorting temporaries even for a single very large query.
    if width > 16:
        result = np.empty_like(X)
        for start in range(0, width, 16):
            result[:, start:start + 16] = query_percentiles(X[:, start:start + 16])
        return result
    if n < 2:
        return np.where(np.isnan(X), np.nan, np.float32(0.5)).astype(np.float32)
    order = np.argsort(X, axis=0, kind="stable")
    ordered = np.take_along_axis(X, order, axis=0)
    positions = np.arange(n)[:, None]
    starts = np.ones((n, width), dtype=bool)
    ends = np.ones((n, width), dtype=bool)
    starts[1:] = ordered[1:] != ordered[:-1]
    ends[:-1] = starts[1:]
    left = np.maximum.accumulate(np.where(starts, positions, 0), axis=0)
    right = np.minimum.accumulate(np.where(ends, positions, n - 1)[::-1], axis=0)[::-1]
    finite_count = np.isfinite(X).sum(axis=0)
    ranked = ((left + right) / (2.0 * np.maximum(finite_count - 1, 1))).astype(np.float32)
    result = np.empty_like(X)
    np.put_along_axis(result, order, ranked, axis=0)
    result[:, finite_count < 2] = 0.5
    result[np.isnan(X)] = np.nan
    return result


def tree_features(X, qids=None, mode="raw"):
    X = np.asarray(X, dtype=np.float32)
    if mode == "raw":
        return X
    if mode != "query_rank":
        raise ValueError(f"unknown tree feature mode: {mode}")
    ranks = np.empty_like(X)
    start = 0
    for size in (group_sizes(qids) if qids is not None else [len(X)]):
        ranks[start:start + size] = query_percentiles(X[start:start + size])
        start += size
    return np.concatenate([X, ranks], axis=1)


def tree_feature_names(width, mode):
    if mode == "raw":
        return "auto"
    return ([f"{PREFIX}raw_{i}" for i in range(width)] +
            [f"{PREFIX}rank_{i}" for i in range(width)])


def features_for_model(model, X, qids=None):
    mode = "query_rank" if model.feature_name()[0].startswith(PREFIX) else "raw"
    return tree_features(X, qids, mode)
