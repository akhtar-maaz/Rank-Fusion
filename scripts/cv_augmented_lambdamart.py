"""Fourth-round fixed-count TRAIN CV, with optional training augmentation."""
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
from scripts.lambdamart_utils import default_params, fit, fit_specification
from scripts.robustness import SCENARIOS, perturb_features
from submission.data import load_ltr
from submission.features import tree_features


def candidates():
    params = default_params(num_leaves=127, min_data_in_leaf=100, lambda_l2=10,
                            lambdarank_truncation_level=13, num_threads=4)
    augmentation = {"name": "half_query_dropout_noise", "seed": 20261019}
    return [dict(name=name, params=params.copy(), feature_mode=mode, augmentation=aug,
                 trees=[200, 400, 800]) for name, mode, aug in (
        ("rank_focus", "query_rank", None),
        ("normalized_focus", "query_normalized", None),
        ("normalized_augmented", "query_normalized", augmentation.copy()),
        ("rank_augmented", "query_rank", augmentation.copy()),
    )]


def summarize(name, members, entries):
    means = {scenario: float(np.mean([fold["scenarios"][scenario] for fold in entries]))
             for scenario in SCENARIOS}
    training = float(np.mean([fold["train_ndcg@10"] for fold in entries]))
    diagnostic = float(np.mean([means[scenario] for scenario in SCENARIOS[1:]]))
    result = dict(name=name, trees=sum(member["trees"] for member in members),
                  params=members[0]["params"], feature_mode="query_all" if len(members) > 1 else members[0]["feature_mode"],
                  train_mean=training, validation_mean=means["clean"],
                  validation_std=float(np.std([fold["scenarios"]["clean"] for fold in entries], ddof=1)),
                  validation_train_ratio=means["clean"] / training if training else 0,
                  scenario_means=means, diagnostic_mean=diagnostic,
                  selection_score=0.5 * (means["clean"] + diagnostic), folds=entries)
    if len(members) > 1:
        result["members"] = members
    else:
        result["augmentation"] = members[0].get("augmentation")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--candidate", type=int, choices=range(4), required=True)
    parser.add_argument("--out-dir", default="runs/cv_v4")
    args = parser.parse_args()
    chosen = candidates()[args.candidate]
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    X, y, q = load_ltr(args.train)[0]
    prepared = tree_features(X, q, chosen["feature_mode"])
    measurements = {count: [] for count in chosen["trees"]}
    scores = np.empty((len(chosen["trees"]), len(SCENARIOS), len(y)), dtype=np.float64)
    for fold, (tr, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        fit_start = time.perf_counter()
        if chosen["augmentation"] is None:
            model = fit(prepared[tr], y[tr], q[tr], chosen["params"], max(chosen["trees"]),
                        chosen["feature_mode"], prepared=True)
        else:
            model = fit_specification(X[tr], y[tr], q[tr], dict(chosen, trees=max(chosen["trees"])))
        model.save_model(str(destination / f"candidate_{args.candidate}_fold_{fold}.txt"))
        entries = {}
        for count in chosen["trees"]:
            prediction = model.predict(prepared[tr], num_iteration=count, num_threads=4)
            entries[count] = {"fold": fold, "scenarios": {},
                              "train_ndcg@10": mean_ndcg_from_scores(prediction, y[tr], q[tr])}
        for scenario_index, scenario in enumerate(SCENARIOS):
            features = (prepared[va] if scenario == "clean" else
                        tree_features(perturb_features(X[va], q[va], scenario), q[va], chosen["feature_mode"]))
            for count_index, count in enumerate(chosen["trees"]):
                prediction = model.predict(features, num_iteration=count, num_threads=4)
                scores[count_index, scenario_index, va] = prediction
                entries[count]["scenarios"][scenario] = mean_ndcg_from_scores(prediction, y[va], q[va])
            del features
        for count in chosen["trees"]:
            measurements[count].append(entries[count])
        print(json.dumps(dict(candidate=chosen["name"], fold=fold, seconds=time.perf_counter() - fit_start,
                              measurements=entries)), flush=True)
    results = [summarize(chosen["name"], [dict(chosen, trees=count)], measurements[count])
               for count in chosen["trees"]]
    report = dict(arguments=vars(args), candidate=chosen, rows=len(y), queries=len(set(q)),
                  scenarios=SCENARIOS, results=results, seconds=time.perf_counter() - started)
    # Publish the report only after a complete, atomically renamed OOF archive.
    temporary = destination / f"oof_{args.candidate}.pending.npz"
    np.savez_compressed(temporary, scores=scores, labels=y, qids=q.astype(str),
                        trees=chosen["trees"], scenarios=SCENARIOS)
    temporary.replace(destination / f"oof_{args.candidate}.npz")
    temporary = destination / f"part_{args.candidate}.pending.json"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination / f"part_{args.candidate}.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
