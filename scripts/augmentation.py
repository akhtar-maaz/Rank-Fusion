"""TRAIN-only whole-query augmentation with a seed distinct from diagnostics."""
import hashlib

import numpy as np

from scripts.robustness import perturb_features
from submission.data import group_sizes


def augment_training(X, y, qids, config):
    """Keep originals; append one hash-selected domain or all three domains.

    The opaque query ID chooses whether to copy and which perturbation to use.
    Balanced mode appends each of affine/dropout/noise for every query.
    Labels are only copied, never inspected. Call this after splitting TRAIN.
    Copies have separate group IDs so pairs never cross query versions.
    """
    balanced = config in ({"name": "balanced_query_domains", "seed": 20261024, "strength": 1.0},
                          {"name": "balanced_query_domains", "seed": 20261024, "strength": 2.0})
    if not balanced and config not in ({"name": "half_query_dropout_noise", "seed": 20261019},
                      {"name": "half_query_mixed", "seed": 20261020},
                      {"name": "full_query_mixed", "seed": 20261021},
                      {"name": "full_query_stress", "seed": 20261022},
                      {"name": "full_query_mixed_v8", "seed": 20261023}):
        raise ValueError("unsupported augmentation configuration")
    X, y, qids = np.asarray(X), np.asarray(y), np.asarray(qids)
    if len(X) != len(y) or len(y) != len(qids):
        raise ValueError("training arrays have different lengths")
    blocks, labels, identities = [X], [y], [np.array([f"original:{q}" for q in qids], dtype=object)]
    if balanced:
        # Every query gets all three versions. Labels are copied, never inspected.
        for scenario in ("affine", "feature_dropout", "noise"):
            blocks.append(perturb_features(X, qids, scenario, seed=config["seed"], strength=config["strength"]))
            labels.append(y)
            identities.append(np.array([f"augmented:{scenario}:{q}" for q in qids], dtype=object))
        return np.concatenate(blocks), np.concatenate(labels), np.concatenate(identities)
    start = 0
    for size in group_sizes(qids):
        stop = start + size
        digest = hashlib.sha256(f"augment:{config['seed']}:{qids[start]}".encode()).digest()
        if config["name"].startswith("full_query_") or digest[0] % 2 == 0:
            if config["name"] == "half_query_dropout_noise":
                scenario = "feature_dropout" if digest[1] % 2 == 0 else "noise"
            else:
                scenario = ("affine", "feature_dropout", "noise")[digest[1] % 3]
            options = {"strength": 2.0} if config["name"] == "full_query_stress" else {}
            blocks.append(perturb_features(X[start:stop], qids[start:stop], scenario, seed=config["seed"], **options))
            labels.append(y[start:stop])
            identities.append(np.full(size, f"augmented:{qids[start]}", dtype=object))
        start = stop
    return np.concatenate(blocks), np.concatenate(labels), np.concatenate(identities)
