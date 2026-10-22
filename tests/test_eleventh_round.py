import copy
import json
from pathlib import Path

import numpy as np
import pytest

from harness.metrics import average_precision, ndcg_at_k
from scripts.select_eleventh_round import specifications, select
from scripts.select_fusion_v11 import CHANNELS, configurations, decision, query_metrics, query_scores
from submission.features import tree_features
from submission.fusion import combmnz_scores, weighted_combsum_scores, weighted_rrf_scores


def test_percentiles_are_query_local_monotonic_invariant_and_ties_safe():
    X = np.array([[1, 7, np.nan], [3, 7, 4], [5, 7, 8], [2, 9, 1], [4, 9, 1]], np.float32)
    q = np.array(["a"] * 3 + ["b"] * 2)
    ranks = tree_features(X, q, "query_percentile")
    assert np.allclose(ranks[:3], [[0, .5, np.nan], [.5, .5, 0], [1, .5, 1]], equal_nan=True)
    assert np.array_equal(ranks[3:], tree_features(X[3:], mode="query_percentile"))
    assert np.allclose(ranks, tree_features(np.exp(X), q, "query_percentile"), equal_nan=True)


def test_frozen_search_contains_exact_control_and_33_tree_settings():
    control = json.loads(Path("configs/v11_fusion_control.json").read_text())
    configs = configurations(control)
    assert len(configs) == len({json.dumps(c, sort_keys=True) for c in configs}) == 1146
    assert configs[-1] == control
    rows = specifications()
    assert len(rows) == 33
    assert all(sum(r["weights"]) == 1 for r in rows)
    candidates = json.loads(Path("configs/v11_candidates.json").read_text())
    assert [r["feature_mode"] for r in candidates] == ["query_percentile"] * 2 + ["query_normalized"] * 2
    assert all(r["trees"] == [400, 800] for r in candidates)


def test_vectorized_fusion_matches_hand_code_and_exact_harness_metrics():
    raw = np.array([[2, 0, 3, -10], [2, 0, 4, -10], [1, 0, 4, -8], [0, 0, 0, -7]], float)
    labels = np.array([3, 0, 1, 2])
    ids = [str(i) for i in range(len(raw))]
    base = {doc: dict(zip(CHANNELS, row)) for doc, row in zip(ids, raw)}
    control = dict(method="weighted_combsum", weights=dict(zip(CHANNELS, [.25, .25, .5, 0])))
    configs = configurations(control)
    scores = query_scores(raw, configs)
    metrics = query_metrics(scores, labels)
    qrels = dict(zip(ids, labels.tolist()))
    for i, config in enumerate(configs):
        if config["method"] == "weighted_rrf":
            values = weighted_rrf_scores(base, config["weights"], config["c"])
        elif config["method"] == "weighted_combsum":
            values = weighted_combsum_scores(base, config["weights"])
        else:
            values = combmnz_scores(base)
        assert np.array_equal(scores[i], list(values.values()))
        order = sorted(ids, key=lambda d: -values[d])[:10]
        assert metrics[i, 0] == pytest.approx(ndcg_at_k(order, qrels, 10), abs=1e-15)
        assert metrics[i, 1] == pytest.approx(average_precision(order, qrels), abs=1e-15)
    assert np.all(query_metrics(scores, np.zeros(len(raw))) == 0)


@pytest.mark.parametrize("weights,c", [({"a": -1}, 0), ({"a": np.nan}, 0), ({"a": np.inf}, 0), ({}, -1), ({}, np.nan)])
def test_weighted_rrf_rejects_invalid_parameters(weights, c):
    with pytest.raises(ValueError):
        weighted_rrf_scores({}, weights, c)


def test_fusion_promotion_requires_gain_and_paired_folds():
    metrics = np.full((2, 10, 2), .3)
    groups = np.repeat(np.arange(5), 2)
    metrics[0, :4] += .01
    assert not decision(metrics, groups)[0]["promoted"]
    metrics[0, 4:6] += .01
    assert decision(metrics, groups)[0]["promoted"]
    metrics[0] = metrics[1] + .0001
    assert not decision(metrics, groups)[0]["promoted"]


def test_tree_control_requires_full_historical_identity():
    # Archived data can validate selection gates without reading any real feature rows.
    historical = json.loads(Path("results/lambdamart_cv_v10.json").read_text())
    row = copy.deepcopy(historical["selected"])
    row["name"] = "cached_v10_control"
    result = select([row], historical)
    assert not result["promoted"]
    bad = copy.deepcopy(row)
    bad["members"][0]["trees"] += 1
    with pytest.raises(ValueError, match="specifications"):
        select([bad], historical)
