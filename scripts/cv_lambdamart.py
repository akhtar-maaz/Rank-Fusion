"""Grouped 5-fold cross-validation sweep over the number of trees (assignment Section 4.3).

    python scripts/cv_lambdamart.py --train data/train --trees 25 50 100 200 400

For each tree count we train on 4 folds and score the 5th. NO early stopping is used on the reported fold.
The gap between the train-fold and validation-fold nDCG@10 is your local estimate of how much a model loses on
queries it has not seen: a model whose gap widens with more trees is overfitting. Pick the tree count where validation stops improving.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import itertools
import json
import time

import numpy as np

from harness.cv import grouped_kfold
from submission.data import load_ltr
from scripts.lambdamart_utils import default_params, fit, ndcg10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--trees", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--num-leaves", type=int, nargs="+", default=[15, 31])
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, nargs="+", default=[50, 150])
    ap.add_argument("--lambda-l2", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/lambdamart_cv.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = os.path.join(args.train, "train.txt") if os.path.isdir(args.train) else args.train
    X, y, q = load_ltr(path)[0]
    folds = grouped_kfold(q, k=args.folds, seed=args.seed)
    trees = sorted(set(args.trees))
    if min(trees) < 1:
        ap.error("tree counts must be positive")
    report = {"arguments": vars(args), "rows": len(y), "queries": len(set(q)), "results": []}
    started = time.perf_counter()
    for leaves, min_leaf, regularization in itertools.product(args.num_leaves, args.min_data_in_leaf, args.lambda_l2):
        params = default_params(leaves, args.learning_rate, min_leaf, lambda_l2=regularization,
                                seed=args.seed, feature_fraction=args.feature_fraction, num_threads=args.threads)
        measurements = {n: [] for n in trees}
        for fold, (tr_idx, va_idx) in enumerate(folds):
            # A model's first n trees equal a fixed n-tree fit with this seed.
            # The reported fold never affects training or stopping decisions.
            model = fit(X[tr_idx], y[tr_idx], q[tr_idx], params, max(trees))
            for n in trees:
                measurements[n].append({"fold": fold,
                    "train_ndcg@10": ndcg10(model, X[tr_idx], y[tr_idx], q[tr_idx], n),
                    "validation_ndcg@10": ndcg10(model, X[va_idx], y[va_idx], q[va_idx], n)})
            print(f"leaves={leaves} min_leaf={min_leaf} l2={regularization}: fold {fold + 1}/{args.folds}", flush=True)
        for n, rows in measurements.items():
            tr = np.array([r["train_ndcg@10"] for r in rows])
            va = np.array([r["validation_ndcg@10"] for r in rows])
            entry = {"trees": n, "params": params, "train_mean": float(tr.mean()),
                     "validation_mean": float(va.mean()), "validation_std": float(va.std(ddof=1)),
                     "validation_train_ratio": float(va.mean() / tr.mean()) if tr.mean() else 0.0,
                     "folds": rows}
            report["results"].append(entry)
            print(f"trees={n:4d} train={tr.mean():.6f} validation={va.mean():.6f} +/- {va.std(ddof=1):.6f}", flush=True)
        report["selected"] = max(report["results"], key=lambda r: (r["validation_mean"], -r["trees"], -r["params"]["num_leaves"]))
        report["seconds"] = time.perf_counter() - started
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as stream:
            json.dump(report, stream, indent=2)
    print(f"CV saved to {args.out}; select on TRAIN folds only", flush=True)


if __name__ == "__main__":
    main()
