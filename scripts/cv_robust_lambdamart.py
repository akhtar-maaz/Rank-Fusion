"""Round-three fixed-tree grouped TRAIN CV and explicit robustness diagnostics.

Run independent candidates via PBS, never read DEV, and retain out-of-fold
scores for reproducible comparisons. No early stopping on reported folds.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.lambdamart_utils import default_params, fit
from scripts.robustness import SCENARIOS, perturb_features
from submission.data import load_ltr
from submission.features import tree_features


def candidates():
    baseline = default_params(num_leaves=127, min_data_in_leaf=100, lambda_l2=10, num_threads=4)
    return [
        {"name": "v2_control", "feature_mode": "query_rank", "params": baseline, "trees": [200, 400, 800]},
        {"name": "query_normalized", "feature_mode": "query_normalized", "params": baseline,
         "trees": [200, 400, 800]},
        {"name": "regularized", "feature_mode": "query_rank", "trees": [400, 800, 1200],
         "params": dict(baseline, num_leaves=63, min_data_in_leaf=200, lambda_l2=20,
                        feature_fraction_bynode=0.5, path_smooth=20, lambdarank_truncation_level=13)},
        {"name": "random_thresholds", "feature_mode": "query_rank", "trees": [400, 800, 1600],
         "params": dict(baseline, extra_trees=True, path_smooth=10, lambdarank_truncation_level=13)},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--candidate", type=int, choices=range(len(candidates())), required=True)
    parser.add_argument("--out-dir", default="runs/cv_v3")
    args = parser.parse_args()
    selected = candidates()[args.candidate]
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    X, y, q = load_ltr(args.train)[0]
    prepared = tree_features(X, q, selected["feature_mode"])
    folds = grouped_kfold(q, k=5, seed=0)
    measurements = {n: [] for n in selected["trees"]}
    scores = np.empty((len(selected["trees"]), len(SCENARIOS), len(y)), dtype=np.float64)
    for fold, (tr, va) in enumerate(folds):
        model = fit(prepared[tr], y[tr], q[tr], selected["params"], max(selected["trees"]),
                    selected["feature_mode"], prepared=True)
        model.save_model(str(destination / f"candidate_{args.candidate}_fold_{fold}.txt"))
        entries = {}
        for count in selected["trees"]:
            prediction = model.predict(prepared[tr], num_iteration=count, num_threads=4)
            entries[count] = {"fold": fold, "train_ndcg@10": mean_ndcg_from_scores(prediction, y[tr], q[tr]),
                              "scenarios": {}}
        for scenario_index, scenario in enumerate(SCENARIOS):
            features = (prepared[va] if scenario == "clean" else
                        tree_features(perturb_features(X[va], q[va], scenario), q[va], selected["feature_mode"]))
            for count_index, count in enumerate(selected["trees"]):
                prediction = model.predict(features, num_iteration=count, num_threads=4)
                scores[count_index, scenario_index, va] = prediction
                entries[count]["scenarios"][scenario] = mean_ndcg_from_scores(prediction, y[va], q[va])
            del features
        for count in selected["trees"]:
            measurements[count].append(entries[count])
        print(json.dumps({"candidate": selected["name"], "fold": fold, "measurements": entries}), flush=True)
    results = []
    for count, entries in measurements.items():
        train_mean = float(np.mean([entry["train_ndcg@10"] for entry in entries]))
        values = {scenario: [entry["scenarios"][scenario] for entry in entries] for scenario in SCENARIOS}
        means = {scenario: float(np.mean(value)) for scenario, value in values.items()}
        diagnostic_mean = float(np.mean([means[name] for name in SCENARIOS if name != "clean"]))
        results.append({"name": selected["name"], "trees": count, "params": selected["params"],
                        "feature_mode": selected["feature_mode"], "train_mean": train_mean,
                        "validation_mean": means["clean"], "validation_std": float(np.std(values["clean"], ddof=1)),
                        "validation_train_ratio": means["clean"] / train_mean if train_mean else 0,
                        "scenario_means": means, "diagnostic_mean": diagnostic_mean,
                        "selection_score": 0.5 * (means["clean"] + diagnostic_mean), "folds": entries})
    report = {"arguments": vars(args), "candidate": selected, "rows": len(y), "queries": len(set(q)),
              "scenarios": SCENARIOS, "results": results, "seconds": time.perf_counter() - started}
    (destination / f"part_{args.candidate}.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(destination / f"oof_{args.candidate}.npz", scores=scores, labels=y, qids=q.astype(str),
                        trees=selected["trees"], scenarios=SCENARIOS)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
