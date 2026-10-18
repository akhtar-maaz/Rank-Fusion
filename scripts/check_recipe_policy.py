"""TRAIN-derived, nested holdout diagnostics for the adaptive C1 policy.

No public/hidden benchmark labels or DEV are read. These constructed domains
are method checks, not unbiased estimates for an unknown C1 benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from scripts.lambdamart_utils import ndcg10
from submission.data import load_ltr
from submission.recipe import fit_arrays

DOMAINS = ((300, 40, 4), (1200, 90, 4), (4000, 136, 2))


def baseline_fit(X, y, q, Xv, yv, qv):
    from submission.data import group_sizes
    from submission.features import tree_feature_names, tree_features
    width, nq = X.shape[1], len(set(q))
    names = tree_feature_names(width, "query_rank")
    training = lgb.Dataset(tree_features(X, q, "query_rank"), label=y, group=group_sizes(q), feature_name=names)
    validation = lgb.Dataset(tree_features(Xv, qv, "query_rank"), label=yv, group=group_sizes(qv),
                             reference=training, feature_name=names)
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], num_leaves=15 if nq < 200 else 31,
                  learning_rate=0.05, min_data_in_leaf=max(20, min(200, len(y) // 2000)), lambda_l2=10.0,
                  feature_fraction=0.9 if width < 100 else 0.8, bagging_fraction=0.9, bagging_freq=1, seed=0,
                  lambdarank_truncation_level=15, deterministic=True, force_row_wise=True, verbose=-1, num_threads=4)
    return lgb.train(params, training, num_boost_round=500, valid_sets=[validation],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--domain", type=int, choices=range(3), required=True)
    parser.add_argument("--out-dir", default="runs/recipe_v3")
    args = parser.parse_args()
    X, y, q = load_ltr(args.train)[0]
    queries, width, max_label = DOMAINS[args.domain]
    rng = np.random.RandomState(2026 + args.domain)
    selected_queries = rng.choice(np.unique(q), min(queries, len(np.unique(q))), replace=False)
    mask = np.isin(q, selected_queries)
    columns = rng.choice(X.shape[1], min(width, X.shape[1]), replace=False)
    X, y, q = X[mask][:, columns], np.minimum(y[mask], max_label), q[mask]
    # Unnamed, shuffled, rescaled features; both methods see exactly the same inputs.
    X *= np.power(10.0, rng.uniform(-2, 2, X.shape[1])).astype(np.float32)
    development, heldout = grouped_kfold(q, k=5, seed=0)[0]
    inner_train, inner_validation = grouped_kfold(q[development], k=5, seed=1)[0]
    tr, va = development[inner_train], development[inner_validation]
    report = {"domain": args.domain, "arguments": vars(args), "rows": len(y), "queries": len(set(q)),
              "features": X.shape[1], "max_label": max_label,
              "fit_queries": len(set(q[tr])), "validation_queries": len(set(q[va])),
              "outer_queries": len(set(q[heldout])), "models": {}}
    for name in ("baseline_v2", "adaptive"):
        started = time.perf_counter()
        if name == "baseline_v2":
            model, history = baseline_fit(X[tr], y[tr], q[tr], X[va], y[va], q[va]), None
        else:
            model, history = fit_arrays(X[tr], y[tr], q[tr], X[va], y[va], q[va])
        elapsed = time.perf_counter() - started
        report["models"][name] = {"seconds": elapsed, "trees": model.best_iteration,
                                  "leaves": model.params["num_leaves"], "history": history,
                                  "outer_ndcg@10": ndcg10(model, X[heldout], y[heldout], q[heldout])}
        print(json.dumps(report["models"][name]), flush=True)
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"domain_{args.domain}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
