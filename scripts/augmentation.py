"""TRAIN-only whole-query augmentation with a seed distinct from diagnostics."""
import hashlib

import numpy as np

from scripts.robustness import perturb_features
from submission.data import group_sizes


def augment_training(X, y, qids, config):
    """Keep originals and append one copy of a fixed half or every query.

    The opaque query ID chooses whether to copy and which perturbation to use.
    Labels are only copied, never inspected. Call this after splitting TRAIN.
    Copies have separate group IDs so pairs never cross query versions.
    """
    if config not in ({"name": "half_query_dropout_noise", "seed": 20261019},
                      {"name": "half_query_mixed", "seed": 20261020},
                      {"name": "full_query_mixed", "seed": 20261021}):
        raise ValueError("unsupported augmentation configuration")
    X, y, qids = np.asarray(X), np.asarray(y), np.asarray(qids)
    if len(X) != len(y) or len(y) != len(qids):
        raise ValueError("training arrays have different lengths")
    blocks, labels, identities = [X], [y], [np.array([f"original:{q}" for q in qids], dtype=object)]
    start = 0
    for size in group_sizes(qids):
        stop = start + size
        digest = hashlib.sha256(f"augment:{config['seed']}:{qids[start]}".encode()).digest()
        if config["name"] == "full_query_mixed" or digest[0] % 2 == 0:
            if config["name"] == "half_query_dropout_noise":
                scenario = "feature_dropout" if digest[1] % 2 == 0 else "noise"
            else:
                scenario = ("affine", "feature_dropout", "noise")[digest[1] % 3]
            blocks.append(perturb_features(X[start:stop], qids[start:stop], scenario, seed=config["seed"]))
            labels.append(y[start:stop])
            identities.append(np.full(size, f"augmented:{qids[start]}", dtype=object))
        start = stop
    return np.concatenate(blocks), np.concatenate(labels), np.concatenate(identities)
