"""Score two fixed blends from saved TRAIN-fold models and OOF predictions."""
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
from submission.data import load_ltr
from submission.features import tree_features
from submission.tree_ensemble import combine

BLENDS = ((400, 200), (800, 200))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--out-dir", default="runs/cv_v3")
    args = parser.parse_args()
    destination = Path(args.out_dir)
    started = time.perf_counter()
    reports = [json.loads((destination / f"part_{index}.json").read_text()) for index in (0, 1)]
    arrays = [np.load(destination / f"oof_{index}.npz") for index in (0, 1)]
    X, y, q = load_ltr(args.train)[0]
    for report, array in zip(reports, arrays):
        if ((report["rows"], report["queries"]) != (len(y), len(set(q))) or
                not np.array_equal(array["labels"], y) or not np.array_equal(array["qids"], q) or
                not np.isfinite(array["scores"]).all()):
            raise ValueError("incomplete or incompatible OOF predictions")
    source_scores = [array["scores"] for array in arrays]
    prepared = tree_features(X, q, "query_all")
    del X
    folds = grouped_kfold(q, k=5, seed=0)
    measurements = {counts: [] for counts in BLENDS}
    for fold, (tr, va) in enumerate(folds):
        models = [lgb.Booster(model_file=str(destination / f"candidate_{index}_fold_{fold}.txt")) for index in (0, 1)]
        for counts in BLENDS:
            ensemble = combine(models, [0.5, 0.5], counts)
            train_scores = ensemble.predict(prepared[tr], num_threads=4)
            indices = [list(array["trees"]).index(count) for array, count in zip(arrays, counts)]
            expected = 0.5 * (source_scores[0][indices[0], :, va].T + source_scores[1][indices[1], :, va].T)
            # The exported single native booster must score identically to the
            # arithmetic average used in OOF comparisons, within summation error.
            actual = ensemble.predict(prepared[va], num_threads=4)
            error = float(np.max(np.abs(actual - expected[0])))
            if error > 5e-12:
                raise ValueError(f"native ensemble differed from explicit average: {error}")
            scores = {str(scenario): mean_ndcg_from_scores(expected[index], y[va], q[va])
                      for index, scenario in enumerate(arrays[0]["scenarios"])}
            entry = {"fold": fold, "train_ndcg@10": mean_ndcg_from_scores(train_scores, y[tr], q[tr]),
                     "scenarios": scores, "native_max_abs_error": error}
            measurements[counts].append(entry)
            print(json.dumps({"counts": counts, **entry}), flush=True)
    results = []
    scenarios = list(reports[0]["scenarios"])
    for counts, entries in measurements.items():
        means = {name: float(np.mean([entry["scenarios"][name] for entry in entries])) for name in scenarios}
        train_mean = float(np.mean([entry["train_ndcg@10"] for entry in entries]))
        diagnostic = float(np.mean([means[name] for name in scenarios if name != "clean"]))
        members = [dict(params=report["candidate"]["params"], trees=count, feature_mode=report["candidate"]["feature_mode"],
                        weight=0.5) for report, count in zip(reports, counts)]
        results.append(dict(name=f"blend_{counts[0]}_{counts[1]}", trees=sum(counts), feature_mode="query_all",
                            params=members[0]["params"], members=members, train_mean=train_mean,
                            validation_mean=means["clean"], validation_std=float(np.std([entry["scenarios"]["clean"] for entry in entries], ddof=1)),
                            validation_train_ratio=means["clean"] / train_mean, scenario_means=means,
                            diagnostic_mean=diagnostic, selection_score=0.5 * (means["clean"] + diagnostic), folds=entries))
    report = dict(arguments=vars(args), rows=len(y), queries=len(set(q)), scenarios=scenarios, results=results,
                  seconds=time.perf_counter() - started)
    (destination / "blends.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
