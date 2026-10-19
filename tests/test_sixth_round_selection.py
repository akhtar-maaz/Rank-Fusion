import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.cv_augmented_lambdamart import summarize
from scripts.select_pairwise_v6 import CONTROL_FOLD0, select as select_neural
from scripts.select_sixth_round import CONTROL, select, specifications


def tree_row(name, clean, diagnostic):
    members = [dict(params={"num_leaves": 127}, trees=800, feature_mode="query_all", weight=weight)
               for weight in (0.125, 0.125, 0.25, 0.5)]
    entries = [dict(fold=fold, **{"train_ndcg@10": 0.75},
                    scenarios=dict(clean=clean, affine=diagnostic, feature_dropout=diagnostic, noise=diagnostic))
               for fold in range(5)]
    return summarize(name, members, entries)


def test_sixth_round_plan_and_exact_control():
    settings = specifications()
    assert len(settings) == len({row["name"] for row in settings}) == 49
    assert len(specifications([0, 3])) == 25
    for row in settings:
        assert len(row["sources"]) == len(row["weights"])
        assert sum(row["weights"]) == 1 and min(row["weights"]) > 0
    control = tree_row(CONTROL, 0.487, 0.474)
    baseline = dict(selected=copy.deepcopy(control))
    candidate = tree_row("new", 0.488, 0.476)
    result = select([control, candidate], baseline)
    assert result["promoted"] and result["selected"] == candidate
    assert result["v5_control"] == control and "v4_control" not in result
    baseline["selected"]["members"][0]["weight"] = 0.2
    with pytest.raises(ValueError, match="specifications"):
        select([control, candidate], baseline)


def neural_reports():
    specs = json.loads((Path(__file__).resolve().parents[1]/"configs/v6_neural_candidates.json").read_text())
    scores = [CONTROL_FOLD0, 0.43, 0.44, 0.435, 0.438]
    reports = []
    for index, spec in enumerate(specs):
        folds = [dict(fold=fold, train_ndcg_10=0.6, validation_ndcg_10=score,
                      history=[{"epoch": epoch, "train_ndcg@10": 0.6} for epoch in range(1, spec["epochs"]+1)])
                 for fold, score in enumerate(scores)]
        reports.append(dict(arguments=dict(candidate=index, train="TRAIN"), candidate=spec, rows=1000, queries=100,
                            folds=folds, train_mean=0.6, validation_mean=float(np.mean(scores)),
                            validation_std=float(np.std(scores, ddof=1)), validation_train_ratio=float(np.mean(scores)/0.6)))
    return reports, specs


def change_scores(report, deltas):
    for fold, delta in zip(report["folds"], deltas):
        fold["validation_ndcg_10"] += delta
    scores = [fold["validation_ndcg_10"] for fold in report["folds"]]
    report.update(validation_mean=float(np.mean(scores)), validation_std=float(np.std(scores, ddof=1)),
                  validation_train_ratio=float(np.mean(scores)/report["train_mean"]))


def test_neural_promotion_requires_mean_gain_and_three_improved_folds():
    reports, specs = neural_reports()
    assert not select_neural(reports, specs)["promoted"]
    change_scores(reports[1], [0.002]*5)
    assert select_neural(reports, specs)["selected_index"] == 1
    # Highest mean with just two improved folds fails the frozen guard;
    # it does not trigger an adaptive search for a different winner.
    change_scores(reports[2], [0.01, 0.01, 0, 0, 0])
    assert not select_neural(reports, specs)["promoted"]


def test_neural_selection_rejects_partial_changed_or_early_stopped_reports():
    reports, specs = neural_reports()
    reports[1]["folds"].pop()
    with pytest.raises(ValueError, match="five"):
        select_neural(reports, specs)
    reports, specs = neural_reports()
    reports[2]["folds"][0]["history"].pop()
    with pytest.raises(ValueError, match="fixed-epoch"):
        select_neural(reports, specs)
    reports, specs = neural_reports()
    change_scores(reports[0], [0.01, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="reproduce"):
        select_neural(reports, specs)
    reports, specs = neural_reports()
    reports[3]["candidate"] = dict(reports[3]["candidate"], seed=1)
    with pytest.raises(ValueError, match="specification"):
        select_neural(reports, specs)
