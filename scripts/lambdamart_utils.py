"""Shared LightGBM LambdaMART helpers for train_model.py and cv_lambdamart.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from submission.data import group_sizes
from submission.features import features_for_model, tree_feature_names, tree_features


def default_params(num_leaves=31, learning_rate=0.05, min_data_in_leaf=50, lambda_l2=1.0, seed=0, **extra):
    p = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "num_leaves": num_leaves,
        "learning_rate": learning_rate, "min_data_in_leaf": min_data_in_leaf, "lambda_l2": lambda_l2,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 1,
    }
    p.update(extra)
    return p


def fit(X, y, qids, params, num_trees, feature_mode="raw", prepared=False):
    import lightgbm as lgb
    if not prepared:
        X = tree_features(X, qids, feature_mode)
    multiplier = 1 if feature_mode == "raw" else 3 if feature_mode == "query_all" else 2
    width = X.shape[1] // multiplier
    names = tree_feature_names(width, feature_mode)
    return lgb.train(params, lgb.Dataset(X, label=y, group=group_sizes(qids), feature_name=names),
                     num_boost_round=num_trees)


def fit_selected(X, y, qids, selected):
    if "members" not in selected:
        return fit(X, y, qids, selected["params"], selected["trees"], selected.get("feature_mode", "raw"))
    from submission.tree_ensemble import combine
    models = [fit(X, y, qids, member["params"], member["trees"], member["feature_mode"])
              for member in selected["members"]]
    return combine(models, [member["weight"] for member in selected["members"]])


def ndcg10(model, X, y, qids, num_iteration=None, prepared=False):
    if not prepared:
        X = features_for_model(model, X, qids)
    threads = model.params.get("num_threads", 1)
    return mean_ndcg_from_scores(model.predict(X, num_iteration=num_iteration, num_threads=threads), y, qids, k=10)
