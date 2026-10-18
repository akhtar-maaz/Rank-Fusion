"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --out submission/model.txt --trees 300

--train takes a split file or a directory holding train.txt. Pass --use-dev to ALSO train on the dev split;
that raises your dev nDCG (Tracks A and B) without helping on queries the model has not seen, so your public
leaderboard position will overstate where you stand (assignment Section 7). Whatever you do, declare it in your
submission notes (Section 10).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json

import numpy as np

from submission.data import load_ltr
from scripts.lambdamart_utils import default_params, fit, fit_selected, ndcg10


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; used for reporting only unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--feature-mode", choices=["raw", "query_rank", "query_normalized"], default="raw")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--truncation", type=int, default=None)
    ap.add_argument("--config", help="CV JSON; use its selected configuration and tree count")
    ap.add_argument("--use-dev", action="store_true", help="also train on dev (lowers Track C; declare it)")
    args = ap.parse_args()

    train_path = _resolve(args.train, "train.txt")
    dev_path = _resolve(args.dev, "vali.txt") if args.dev else os.path.join(args.train, "vali.txt")
    paths = [train_path] + ([dev_path] if os.path.isfile(dev_path) else [])
    if args.dev and not os.path.isfile(dev_path):
        raise SystemExit(f"missing dev file: {dev_path}")
    splits = load_ltr(*paths)
    train, dev = splits[0], splits[1] if len(splits) > 1 else None
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs a dev file")
        # Namespace IDs so a query ID reused in another split cannot be merged.
        X, y = np.vstack([train[0], dev[0]]), np.concatenate([train[1], dev[1]])
        q = np.concatenate([[f"train:{v}" for v in train[2]], [f"dev:{v}" for v in dev[2]]])
    else:
        X, y, q = train
    params = default_params(args.num_leaves, args.learning_rate, args.min_data_in_leaf, seed=args.seed,
                            lambda_l2=args.lambda_l2, feature_fraction=args.feature_fraction, num_threads=args.threads)
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.truncation is not None:
        params["lambdarank_truncation_level"] = args.truncation
    selected = None
    if args.config:
        with open(args.config) as stream:
            selected = json.load(stream)["selected"]
        params, args.trees = selected["params"], selected["trees"]
        args.feature_mode = selected.get("feature_mode", "raw")
    model = (fit_selected(X, y, q, selected) if selected is not None else
             fit(X, y, q, params, args.trees, args.feature_mode))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(set(q))} queries ({'train+dev' if args.use_dev else 'train only'}) -> {args.out}")
    print(f"train nDCG@10 = {ndcg10(model, X, y, q):.4f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = dev
        print(f"dev   nDCG@10 = {ndcg10(model, Xd, yd, qd):.4f}")


if __name__ == "__main__":
    main()
