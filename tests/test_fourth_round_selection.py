import copy

import numpy as np
import pytest

from scripts.cv_augmented_lambdamart import summarize
from scripts.lambdamart_utils import default_params, fit
from scripts.select_fourth_round import predict_prefixes, select, specifications


def row(name, clean, diagnostic, trees=200, members=None):
    if members is None:
        members = [dict(params={"num_leaves": 127}, trees=trees, feature_mode="query_rank")]
    folds = [{"fold": fold, "train_ndcg@10": 0.7,
              "scenarios": dict(clean=clean, affine=diagnostic, feature_dropout=diagnostic, noise=diagnostic)}
             for fold in range(5)]
    return summarize(name, members, folds)


def fixture():
    members = [dict(params={"num_leaves": 127}, trees=800, feature_mode="query_rank", weight=0.5),
               dict(params={"num_leaves": 127}, trees=200, feature_mode="query_normalized", weight=0.5)]
    control = row("cached_800_200_0.5", 0.485, 0.468, members=members)
    return control, dict(selected=copy.deepcopy(control))


def test_fourth_round_finite_search_and_positive_weights():
    settings = specifications()
    assert len(settings) == 36
    assert len(set(setting["name"] for setting in settings)) == 36
    for setting in settings:
        assert len(setting["sources"]) == len(setting["weights"])
        assert all(weight > 0 for weight in setting["weights"])
        assert sum(setting["weights"]) == 1
    assert len(specifications([0, 1])) == 24


def test_fourth_round_clean_guard_and_incremental_threshold():
    control, baseline = fixture()
    weak = row("small_gain", 0.485, 0.4688)
    result = select([control, weak], baseline)
    assert not result["promoted"] and result["selected"] == control
    robust = row("robust", 0.484, 0.474)
    excessive_clean_loss = row("bad_clean", 0.48, 0.49)
    result = select([control, weak, robust, excessive_clean_loss], baseline)
    assert result["promoted"] and result["selected"] == robust


def test_fourth_round_rejects_partial_metrics_and_changed_control():
    control, baseline = fixture()
    candidate = row("candidate", 0.485, 0.475)
    candidate["folds"].pop()
    with pytest.raises(ValueError, match="five"):
        select([control, candidate], baseline)
    candidate = row("candidate", 0.485, 0.475)
    candidate["selection_score"] += 0.001
    with pytest.raises(ValueError, match="aggregate"):
        select([control, candidate], baseline)
    baseline["selected"]["validation_mean"] += 0.001
    with pytest.raises(ValueError, match="reproduce"):
        select([control], baseline)


def test_disjoint_tree_predictions_match_full_prefixes():
    rng = np.random.RandomState(12)
    X = rng.normal(size=(160, 4)).astype(np.float32)
    q = np.repeat(np.arange(20), 8)
    y = rng.randint(0, 5, len(X))
    model = fit(X, y, q, default_params(7, min_data_in_leaf=2), 24)
    for count, prediction in predict_prefixes(model, X, [6, 12, 24]).items():
        expected = model.predict(X, num_iteration=count, num_threads=1)
        assert np.allclose(prediction, expected, atol=1e-12)
        assert np.array_equal(np.argsort(prediction, kind="stable"), np.argsort(expected, kind="stable"))
