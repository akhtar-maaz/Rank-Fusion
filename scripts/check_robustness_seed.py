"""Post-selection stress check on TRAIN validation folds, never a selector."""
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
from harness.metrics import mean_ndcg_from_scores
from scripts.robustness import SCENARIOS, perturb_features
from submission.data import load_ltr
from submission.features import feature_mode_for_model, tree_features
from submission.tree_ensemble import combine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--selection", default="results/lambdamart_cv_v5.json")
    parser.add_argument("--out", default="results/robustness_seed_v5.json")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--control-key", default="v4_control")
    parser.add_argument("--control-label", default="v4")
    parser.add_argument("--selected-label", default="v5")
    args = parser.parse_args()
    if args.seed in (1729, 20261019, 20261020):
        parser.error("use a seed distinct from training and model selection")
    started = time.perf_counter()
    selection = json.loads(Path(args.selection).read_text())
    if not selection["promoted"]:
        raise ValueError("check only the frozen promoted configuration")
    rows = {args.control_label: selection[args.control_key], args.selected_label: selection["selected"]}
    versions = {source[0] for row in rows.values() for source in row["sources"]}
    directories = {version: Path(selection.get("arguments", {}).get(f"{version}_dir", f"runs/cv_{version}"))
                   for version in versions}
    X, y, q = load_ltr(args.train)[0]
    entries = []
    for fold, (_, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        scorers = {}
        for name, row in rows.items():
            models = [lgb.Booster(model_file=str(directories[version] / f"candidate_{candidate}_fold_{fold}.txt"))
                      for version, candidate, _ in row["sources"]]
            counts = [source[2] for source in row["sources"]]
            if len(models) == 1:
                scorers[name] = (models[0], counts[0])
            else:
                scorers[name] = (combine(models, [member["weight"] for member in row["members"]], counts), None)
        result = dict(fold=fold, scenarios={})
        for scenario in SCENARIOS[1:]:
            changed = perturb_features(X[va], q[va], scenario, seed=args.seed)
            all_features = tree_features(changed, q[va], "query_all")
            width = changed.shape[1]
            result["scenarios"][scenario] = {}
            for name, (model, count) in scorers.items():
                mode = feature_mode_for_model(model)
                features = (all_features if mode == "query_all" else all_features[:, :2 * width]
                            if mode == "query_rank" else np.concatenate([all_features[:, 2 * width:], all_features[:, width:2 * width]], axis=1)
                            if mode == "query_normalized" else changed)
                scores = model.predict(features, num_iteration=count, num_threads=4)
                if not np.isfinite(scores).all():
                    raise ValueError("non-finite stress-check prediction")
                result["scenarios"][scenario][name] = mean_ndcg_from_scores(scores, y[va], q[va])
        entries.append(result)
        print(json.dumps(result), flush=True)
    means = {name: {scenario: float(np.mean([fold["scenarios"][scenario][name] for fold in entries]))
                    for scenario in SCENARIOS[1:]} for name in rows}
    report = dict(arguments=vars(args), purpose="Post-selection TRAIN-fold stress check; does not choose or change the artifact",
                  rows=len(y), queries=len(set(q)), selection_seed=1729, diagnostic_seed=args.seed,
                  scenarios=SCENARIOS[1:], scenario_means=means,
                  diagnostic_means={name: float(np.mean(list(scores.values()))) for name, scores in means.items()},
                  folds=entries, seconds=time.perf_counter() - started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
