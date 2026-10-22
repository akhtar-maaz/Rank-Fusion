"""Label-free features computed from the candidates of one query."""
import numpy as np

from submission.data import group_sizes

PREFIX = "rankfusion_qrank_v1_"
NORMALIZED_PREFIX = "rankfusion_qnorm_v1_"
ALL_PREFIX = "rankfusion_qall_v1_"
PERCENTILE_PREFIX = "rankfusion_qpercentile_v1_"


def query_minmax(X):
    """Query-local [0, 1] values; constants are 0.5 and NaNs stay missing."""
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X)
    low = np.min(np.where(finite, X, np.inf), axis=0, initial=np.inf)
    high = np.max(np.where(finite, X, -np.inf), axis=0, initial=-np.inf)
    # Float64 subtraction also handles opposite, very large finite float32s.
    low = np.where(np.isfinite(low), low, 0).astype(np.float64)
    span = high.astype(np.float64) - low
    variable = np.isfinite(span) & (span > 0)
    result = np.full(X.shape, 0.5, dtype=np.float32)
    np.divide(X.astype(np.float64) - low, np.where(variable, span, 1),
              out=result, where=finite & variable)
    result[~finite] = np.nan
    return result


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
    if mode not in ("query_rank", "query_normalized", "query_all", "query_percentile"):
        raise ValueError(f"unknown tree feature mode: {mode}")
    ranks = np.empty_like(X)
    normalized = np.empty_like(X) if mode in ("query_normalized", "query_all") else None
    start = 0
    for size in (group_sizes(qids) if qids is not None else [len(X)]):
        ranks[start:start + size] = query_percentiles(X[start:start + size])
        if normalized is not None:
            normalized[start:start + size] = query_minmax(X[start:start + size])
        start += size
    if mode == "query_percentile":
        return ranks
    if mode == "query_all":
        return np.concatenate([X, ranks, normalized], axis=1)
    return np.concatenate([X if mode == "query_rank" else normalized, ranks], axis=1)


def tree_feature_names(width, mode):
    if mode == "raw":
        return "auto"
    if mode == "query_percentile":
        return [f"{PERCENTILE_PREFIX}rank_{i}" for i in range(width)]
    if mode == "query_all":
        return [f"{ALL_PREFIX}{block}_{i}" for block in ("raw", "rank", "minmax") for i in range(width)]
    if mode not in ("query_rank", "query_normalized"):
        raise ValueError(f"unknown tree feature mode: {mode}")
    prefix = PREFIX if mode == "query_rank" else NORMALIZED_PREFIX
    values = "raw" if mode == "query_rank" else "minmax"
    return ([f"{prefix}{values}_{i}" for i in range(width)] +
            [f"{prefix}rank_{i}" for i in range(width)])


def feature_mode_for_model(model):
    name = model.feature_name()[0]
    return ("query_percentile" if name.startswith(PERCENTILE_PREFIX) else
            "query_rank" if name.startswith(PREFIX) else
            "query_all" if name.startswith(ALL_PREFIX) else
            "query_normalized" if name.startswith(NORMALIZED_PREFIX) else "raw")


def features_for_model(model, X, qids=None):
    return tree_features(X, qids, feature_mode_for_model(model))
