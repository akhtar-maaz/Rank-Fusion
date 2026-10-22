"""Measure fixed averages of saved TRAIN-fold scorers without refitting them."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from scripts.cv_augmented_lambdamart import summarize
from scripts.robustness import SCENARIOS
from scripts.select_fourth_round import predict_prefixes
from submission.data import load_ltr
from submission.features import tree_features
from submission.tree_ensemble import combine


def measure(train_path, directories, settings):
    """Return complete five-fold rows and data needed to verify the winner."""
    X, y, q = load_ltr(train_path)[0]
    sources = sorted({source for setting in settings for source in setting["sources"]})
    reports, predictions, members = {}, {}, {}
    for version, candidate in sorted({source[:2] for source in sources}):
        folder = Path(directories[version])
        report = json.loads((folder / f"part_{candidate}.json").read_text())
        with np.load(folder / f"oof_{candidate}.npz") as array:
            if ((report["rows"], report["queries"]) != (len(y), len(set(q))) or
                    not np.array_equal(array["labels"], y) or not np.array_equal(array["qids"], q) or
                    tuple(array["scenarios"]) != SCENARIOS or
                    not np.array_equal(array["trees"], report["candidate"]["trees"])):
                raise ValueError("incompatible source OOF data")
            scores = array["scores"]
        if scores.shape != (len(report["candidate"]["trees"]), len(SCENARIOS), len(y)) or not np.isfinite(scores).all():
            raise ValueError("invalid source scores")
        if len(report["results"]) != len(report["candidate"]["trees"]):
            raise ValueError("incomplete source checkpoints")
        for row in report["results"]:
            if sorted(fold["fold"] for fold in row["folds"]) != list(range(5)):
                raise ValueError("incomplete source folds")
        specification = report["candidate"]
        reports[(version, candidate)] = report
        for index, count in enumerate(specification["trees"]):
            source = (version, candidate, count)
            if source not in sources:
                continue
            predictions[source] = scores[index]
            members[source] = dict(params=specification["params"], trees=count, feature_mode=specification["feature_mode"])
            if specification.get("augmentation") is not None:
                members[source]["augmentation"] = specification["augmentation"]
    prepared = tree_features(X, q, "query_all")
    width = X.shape[1]
    del X
    measurements = {setting["name"]: [] for setting in settings}
    for fold, (tr, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        training = {}
        for mode in ("query_rank", "query_normalized", "query_all", "query_percentile"):
            keys = [key for key, report in reports.items() if report["candidate"]["feature_mode"] == mode]
            if not keys:
                continue
            features = (prepared[tr, width:2 * width] if mode == "query_percentile" else
                        prepared[tr] if mode == "query_all" else prepared[tr, :2 * width] if mode == "query_rank" else
                        np.concatenate([prepared[tr, 2 * width:], prepared[tr, width:2 * width]], axis=1))
            for version, candidate in keys:
                counts = [count for source_version, source_candidate, count in sources
                          if (source_version, source_candidate) == (version, candidate)]
                model = lgb.Booster(model_file=str(Path(directories[version]) / f"candidate_{candidate}_fold_{fold}.txt"))
                for count, scores in predict_prefixes(model, features, counts).items():
                    training[(version, candidate, count)] = scores
            del features
        for setting in settings:
            weights = np.asarray(setting["weights"], dtype=float)
            if (len(weights) != len(setting["sources"]) or not np.isfinite(weights).all() or
                    np.any(weights <= 0) or abs(weights.sum() - 1) > 1e-12):
                raise ValueError("invalid fixed weights")
            train_scores = sum(weight * training[source] for source, weight in zip(setting["sources"], weights))
            validation = sum(weight * predictions[source][:, va] for source, weight in zip(setting["sources"], weights))
            measurements[setting["name"]].append({
                "fold": fold, "train_ndcg@10": mean_ndcg_from_scores(train_scores, y[tr], q[tr]),
                "scenarios": {name: mean_ndcg_from_scores(validation[index], y[va], q[va])
                              for index, name in enumerate(SCENARIOS)}})
        print(json.dumps(dict(fold=fold, settings=len(settings))), flush=True)
    rows = []
    for setting in settings:
        specifications = [dict(members[source], weight=weight) for source, weight in zip(setting["sources"], setting["weights"])]
        row = summarize(setting["name"], specifications, measurements[setting["name"]])
        row["sources"] = setting["sources"]
        rows.append(row)
    context = dict(prepared=prepared, labels=y, qids=q, predictions=predictions, directories=directories,
                   members=members, rows=len(y), queries=len(set(q)))
    return rows, context


def verify_native(setting, context):
    records = []
    for fold, (_, va) in enumerate(grouped_kfold(context["qids"], k=5, seed=0)):
        models = [lgb.Booster(model_file=str(Path(context["directories"][version]) / f"candidate_{candidate}_fold_{fold}.txt"))
                  for version, candidate, _ in setting["sources"]]
        counts = [source[2] for source in setting["sources"]]
        if len(models) == 1:
            from submission.features import features_for_model
            model = models[0]
            features = features_for_model(model, context["prepared"][va, :context["prepared"].shape[1] // 3], context["qids"][va])
            actual = model.predict(features, num_iteration=counts[0], num_threads=4)
        else:
            model = combine(models, setting["weights"], counts)
            actual = model.predict(context["prepared"][va], num_threads=4)
        expected = sum(weight * context["predictions"][source][0, va]
                       for source, weight in zip(setting["sources"], setting["weights"]))
        error = float(np.max(np.abs(actual - expected)))
        actual_ndcg = mean_ndcg_from_scores(actual, context["labels"][va], context["qids"][va])
        expected_ndcg = mean_ndcg_from_scores(expected, context["labels"][va], context["qids"][va])
        if error > 5e-12 or abs(actual_ndcg - expected_ndcg) > 1e-12:
            raise ValueError("native winner differs from the explicit average")
        records.append(dict(fold=fold, native_max_abs_error=error, native_ndcg_10=actual_ndcg,
                            explicit_ndcg_10=expected_ndcg))
    return records
