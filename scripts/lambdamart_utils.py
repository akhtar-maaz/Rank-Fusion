"""Shared LightGBM LambdaMART helpers for train_model.py and cv_lambdamart.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from submission.data import group_sizes


def default_params(num_leaves=31, learning_rate=0.05, min_data_in_leaf=50, lambda_l2=1.0, seed=0, **extra):
    p = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "num_leaves": num_leaves,
        "learning_rate": learning_rate, "min_data_in_leaf": min_data_in_leaf, "lambda_l2": lambda_l2,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 1,
    }
    p.update(extra)
    return p


def fit(X, y, qids, params, num_trees):
    import lightgbm as lgb
    return lgb.train(params, lgb.Dataset(X, label=y, group=group_sizes(qids)), num_boost_round=num_trees)


def ndcg10(model, X, y, qids, num_iteration=None):
    return mean_ndcg_from_scores(model.predict(X, num_iteration=num_iteration, num_threads=1), y, qids, k=10)
