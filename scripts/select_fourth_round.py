"""Bounded fourth-round averages and selection using complete TRAIN folds."""
import argparse
import copy
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
from scripts.cv_augmented_lambdamart import summarize
from scripts.robustness import SCENARIOS
from submission.data import load_ltr
from submission.features import tree_features
from submission.tree_ensemble import combine


def specifications(completed=range(4)):
    settings = []
    for rank_trees in (400, 800):
        for normalized_trees in (200, 400):
            for rank_weight in (0.25, 0.5, 0.75):
                settings.append(dict(name=f"cached_{rank_trees}_{normalized_trees}_{rank_weight}",
                                     sources=[("v3", 0, rank_trees), ("v3", 1, normalized_trees)],
                                     weights=[rank_weight, 1 - rank_weight]))
    for candidate in completed:
        for trees in (200, 400, 800):
            settings.append(dict(name=f"candidate_{candidate}_{trees}",
                                 sources=[("v4", candidate, trees)], weights=[1.0]))
            settings.append(dict(name=f"v3_plus_{candidate}_{trees}",
                                 sources=[("v3", 0, 800), ("v3", 1, 200), ("v4", candidate, trees)],
                                 weights=[0.25, 0.25, 0.5]))
    return settings


def select(rows, baseline):
    if not rows or len(set(row["name"] for row in rows)) != len(rows):
        raise ValueError("need distinct complete settings")
    for row in rows:
        if sorted(fold["fold"] for fold in row["folds"]) != list(range(5)):
            raise ValueError("every setting requires five distinct folds")
        if not all(np.isfinite(fold["train_ndcg@10"]) and
                   all(np.isfinite(fold["scenarios"][name]) for name in SCENARIOS) for fold in row["folds"]):
            raise ValueError("non-finite fold metric")
        expected = summarize(row["name"], row.get("members", [row]), row["folds"])
        for name in ("train_mean", "validation_mean", "validation_std", "validation_train_ratio",
                     "diagnostic_mean", "selection_score"):
            if not np.isfinite(row[name]) or abs(row[name] - expected[name]) > 1e-12:
                raise ValueError(f"inconsistent aggregate: {name}")
    controls = [row for row in rows if row["name"] == "cached_800_200_0.5"]
    if len(controls) != 1:
        raise ValueError("missing v3 control")
    control = controls[0]
    old = baseline["selected"]
    if control.get("members") != old["members"]:
        # v3 members omit the optional null augmentation field.
        normalized = copy.deepcopy(control["members"])
        for member in normalized:
            if member.get("augmentation") is None:
                member.pop("augmentation", None)
        if normalized != old["members"]:
            raise ValueError("v3 control specifications changed")
    for name in ("train_mean", "validation_mean", "diagnostic_mean", "selection_score"):
        if abs(control[name] - old[name]) > 1e-12:
            raise ValueError("v3 control does not reproduce previous TRAIN CV")
    best_clean = max(row["validation_mean"] for row in rows)
    eligible = [row for row in rows if row["validation_mean"] >= best_clean - 0.0025]
    winner = max(eligible, key=lambda row: (row["selection_score"], -row["trees"]))
    promoted = winner["selection_score"] >= control["selection_score"] + 0.0005
    return dict(results=rows, selected=winner if promoted else control, promoted=promoted,
                best_clean_mean=best_clean, v3_control=control,
                selection_protocol=dict(split="TRAIN only", folds=5, seed=0, clean_tolerance=0.0025,
                                        minimum_mixture_improvement=0.0005,
                                        score="0.5 clean + 0.5 mean(affine, feature_dropout, noise)",
                                        staff_c2_reconstruction=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--v3-dir", default="runs/cv_v3")
    parser.add_argument("--v4-dir", default="runs/cv_v4")
    parser.add_argument("--baseline", default="runs/baseline_v3/lambdamart_cv.json")
    parser.add_argument("--out", default="results/lambdamart_cv_v4.json")
    parser.add_argument("--excluded-candidates", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if len(set(args.excluded_candidates)) != len(args.excluded_candidates) or not set(args.excluded_candidates) <= set(range(4)):
        parser.error("exclusions must name distinct candidates 0..3")
    completed = sorted(set(range(4)) - set(args.excluded_candidates))
    started = time.perf_counter()
    baseline = json.loads(Path(args.baseline).read_text())
    X, y, q = load_ltr(args.train)[0]
    reports, arrays, predictions = {}, {}, {}
    directories = dict(v3=Path(args.v3_dir), v4=Path(args.v4_dir))
    for version, candidates in (("v3", (0, 1)), ("v4", completed)):
        for candidate in candidates:
            key = (version, candidate)
            report = json.loads((directories[version] / f"part_{candidate}.json").read_text())
            with np.load(directories[version] / f"oof_{candidate}.npz") as array:
                if ((report["rows"], report["queries"]) != (len(y), len(set(q))) or
                        not np.array_equal(array["labels"], y) or not np.array_equal(array["qids"], q) or
                        tuple(array["scenarios"]) != SCENARIOS or
                        not np.array_equal(array["trees"], report["candidate"]["trees"])):
                    raise ValueError("incompatible OOF data")
                score = array["scores"]
            if score.shape != (3, len(SCENARIOS), len(y)) or not np.isfinite(score).all():
                raise ValueError("invalid OOF scores")
            for row in report["results"]:
                if sorted(fold["fold"] for fold in row["folds"]) != list(range(5)):
                    raise ValueError("incomplete source folds")
            reports[key], arrays[key] = report, score
            for position, trees in enumerate(report["candidate"]["trees"]):
                predictions[(version, candidate, trees)] = score[position]
    settings = specifications(completed)
    sources = sorted(set(source for setting in settings for source in setting["sources"]))
    members = {}
    for source in sources:
        version, candidate, count = source
        specification = reports[(version, candidate)]["candidate"]
        members[source] = dict(params=specification["params"], trees=count,
                               feature_mode=specification["feature_mode"])
        if specification.get("augmentation") is not None:
            members[source]["augmentation"] = specification["augmentation"]
    prepared = tree_features(X, q, "query_all")
    width = X.shape[1]
    del X
    measurements = {setting["name"]: [] for setting in settings}
    for fold, (tr, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        models, training = {}, {}
        for mode in ("query_rank", "query_normalized"):
            features = (prepared[tr, :2 * width] if mode == "query_rank" else
                        np.concatenate([prepared[tr, 2 * width:], prepared[tr, width:2 * width]], axis=1))
            for version, candidate, count in sources:
                source = (version, candidate, count)
                if members[source]["feature_mode"] != mode:
                    continue
                key = (version, candidate)
                if key not in models:
                    models[key] = lgb.Booster(model_file=str(directories[version] / f"candidate_{candidate}_fold_{fold}.txt"))
                training[source] = models[key].predict(features, num_iteration=count, num_threads=4)
            del features
        for setting in settings:
            train_scores = sum(weight * training[source] for source, weight in zip(setting["sources"], setting["weights"]))
            scores = sum(weight * predictions[source][:, va] for source, weight in zip(setting["sources"], setting["weights"]))
            entry = {"fold": fold, "train_ndcg@10": mean_ndcg_from_scores(train_scores, y[tr], q[tr]),
                     "scenarios": {name: mean_ndcg_from_scores(scores[index], y[va], q[va])
                                   for index, name in enumerate(SCENARIOS)}}
            measurements[setting["name"]].append(entry)
        print(json.dumps(dict(fold=fold, settings=len(settings), seconds=time.perf_counter() - started)), flush=True)
    rows = []
    for setting in settings:
        constituent_specs = [dict(members[source], weight=weight)
                             for source, weight in zip(setting["sources"], setting["weights"])]
        row = summarize(setting["name"], constituent_specs, measurements[setting["name"]])
        row["sources"] = setting["sources"]
        rows.append(row)
    result = select(rows, baseline)
    # Verify the exact exported winner against its explicit OOF average on
    # every held-out TRAIN query before freezing selection for final fitting.
    chosen = next(setting for setting in settings if setting["name"] == result["selected"]["name"])
    equivalence = []
    for fold, (_, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        if len(chosen["sources"]) == 1:
            equivalence.append(dict(fold=fold, native_max_abs_error=0.0, single_model=True))
            continue
        models = [lgb.Booster(model_file=str(directories[version] / f"candidate_{candidate}_fold_{fold}.txt"))
                  for version, candidate, _ in chosen["sources"]]
        native = combine(models, chosen["weights"], [source[2] for source in chosen["sources"]])
        expected = sum(weight * predictions[source][0, va] for source, weight in zip(chosen["sources"], chosen["weights"]))
        error = float(np.max(np.abs(native.predict(prepared[va], num_threads=4) - expected)))
        if error > 5e-12:
            raise ValueError(f"native winner differs from explicit average: {error}")
        equivalence.append(dict(fold=fold, native_max_abs_error=error))
    result.update(arguments=vars(args), rows=len(y), queries=len(set(q)), scenarios=SCENARIOS,
                  completed_candidates=completed, runtime_excluded_candidates=args.excluded_candidates,
                  native_equivalence=equivalence, seconds=time.perf_counter() - started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(promoted=result["promoted"], selected=result["selected"])), flush=True)


if __name__ == "__main__":
    main()
