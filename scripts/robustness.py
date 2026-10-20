"""Public, label-free TRAIN diagnostics; these are not staff C2 versions."""
import hashlib

import numpy as np

from submission.data import group_sizes

SCENARIOS = ("clean", "affine", "feature_dropout", "noise")


def perturb_features(X, qids, scenario, seed=1729, *, strength=1.0):
    """Transform each query independently, deterministically from its ID.

    Affine: positive feature scales in [0.5, 2] and shifts within 25% of
    the feature's range. Dropout: 10% chance of replacing a whole query's
    feature column by zero. Noise: Gaussian noise, SD 5% of query feature SD.
    strength=2 doubles the log-scale/shift range, dropout probability and
    noise SD for the broader training augmentation. Default diagnostics
    remain unchanged. Missing entries remain missing; no labels accepted.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if not np.isfinite(strength) or not 0 < strength <= 10:
        raise ValueError("strength must be finite and in (0, 10]")
    X = np.asarray(X, dtype=np.float32)
    if scenario == "clean":
        return X.copy()
    result = X.copy()
    start = 0
    for size in group_sizes(qids):
        stop = start + size
        identity = f"{seed}:{qids[start]}".encode()
        rng = np.random.RandomState(int.from_bytes(hashlib.sha256(identity).digest()[:4], "little"))
        block = X[start:stop].astype(np.float64)
        missing = ~np.isfinite(block)
        if scenario == "feature_dropout":
            block[:, rng.rand(X.shape[1]) < 0.1 * strength] = 0
        else:
            count = np.maximum((~missing).sum(axis=0), 1)
            mean = np.where(missing, 0, block).sum(axis=0) / count
            if scenario == "noise":
                variance = np.where(missing, 0, (block - mean) ** 2).sum(axis=0) / count
                block += rng.normal(size=block.shape) * (0.05 * strength * np.sqrt(variance))
            else:
                low = np.min(np.where(missing, np.inf, block), axis=0)
                high = np.max(np.where(missing, -np.inf, block), axis=0)
                span = np.where(np.isfinite(high - low), high - low, 0)
                block = (block * np.exp(rng.uniform(-np.log(2) * strength, np.log(2) * strength, X.shape[1])) +
                         rng.uniform(-0.25 * strength, 0.25 * strength, X.shape[1]) * span)
        block[missing] = np.nan
        result[start:stop] = block
        start = stop
    return result
