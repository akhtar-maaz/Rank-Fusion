import numpy as np
import pytest

from scripts.lambdamart_utils import default_params, fit
from scripts.robustness import SCENARIOS, perturb_features
from submission.features import features_for_model, query_minmax, tree_features
from submission import ranking
from scripts.select_robust_lambdamart import select


def test_query_normalization_affine_invariant_and_missing_safe():
    X = np.array([[1, 7, np.nan], [3, 7, 4], [5, 7, 8]], np.float32)
    expected = [[0, 0.5, np.nan], [0.5, 0.5, 0], [1, 0.5, 1]]
    assert np.allclose(query_minmax(X), expected, equal_nan=True)
    original = tree_features(X, mode="query_normalized")
    changed = tree_features(X * [2, 3, 4] + [10, -5, 7], mode="query_normalized")
    assert np.allclose(original, changed, equal_nan=True)
    assert np.isnan(query_minmax(np.array([[np.nan]], np.float32))).all()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_diagnostics_are_query_local_deterministic_and_preserve_input(scenario):
    X = np.arange(120, dtype=np.float32).reshape(12, 10)
    X[0, 3] = np.nan
    q = np.repeat(["first", "second"], 6)
    original = X.copy()
    changed = perturb_features(X, q, scenario)
    assert np.array_equal(X, original, equal_nan=True)
    assert np.array_equal(changed, perturb_features(X, q, scenario), equal_nan=True)
    assert np.array_equal(changed[6:], perturb_features(X[6:], q[6:], scenario), equal_nan=True)
    assert np.array_equal(np.isnan(changed), np.isnan(X))
    order = np.r_[6:12, 0:6]
    assert np.array_equal(changed[order], perturb_features(X[order], q[order], scenario), equal_nan=True)


def test_normalized_model_roundtrips_and_ranking_uses_metadata(tmp_path):
    import lightgbm as lgb
    rng = np.random.RandomState(3)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    q = np.repeat(np.arange(20), 6)
    y = (X[:, 0] > 0).astype(int)
    model = fit(X, y, q, default_params(num_leaves=7, min_data_in_leaf=2), 12, "query_normalized")
    path = str(tmp_path / "model.txt")
    model.save_model(path)
    restored = lgb.Booster(model_file=path)
    assert np.array_equal(features_for_model(restored, X, q), tree_features(X, q, "query_normalized"))
    ranking.prepare(path)
    ids = [str(i) for i in range(6)]
    actual = ranking.rerank_with_ltr("q", ids, dict(zip(ids, X[:6].tolist())), 6)
    scores = restored.predict(features_for_model(restored, X[:6]), num_threads=1)
    assert [doc for doc, _ in actual] == [ids[i] for i in np.argsort(-scores, kind="stable")]


def selection_fixture():
    reports = []
    for index, (name, clean, diagnostic) in enumerate([
            ("v2_control", 0.48, 0.42), ("normalized", 0.479, 0.45),
            ("regularized", 0.47, 0.5), ("random", 0.478, 0.43)]):
        means = dict(clean=clean, affine=diagnostic, feature_dropout=diagnostic, noise=diagnostic)
        row = dict(name=name, trees=800, params={"num_leaves": 127}, feature_mode="query_rank",
                   validation_mean=clean, diagnostic_mean=diagnostic, selection_score=0.5 * (clean + diagnostic),
                   folds=[dict(fold=fold, scenarios=means.copy()) for fold in range(5)])
        reports.append(dict(arguments=dict(candidate=index, train="train.txt"), rows=100, queries=10,
                            scenarios=SCENARIOS, results=[row]))
    baseline = dict(rows=100, queries=10, selected=reports[0]["results"][0].copy())
    return reports, baseline


def test_promotion_requires_clean_guard_and_diagnostic_improvement():
    reports, baseline = selection_fixture()
    result = select(reports, baseline)
    assert result["promoted"] and result["selected"]["name"] == "normalized"
    reports = [reports[0]] + [dict(report, results=[dict(report["results"][0],
               **{key: reports[0]["results"][0][key] for key in
                  ("validation_mean", "diagnostic_mean", "selection_score", "folds")})]) for report in reports[1:]]
    result = select(reports, baseline)
    assert not result["promoted"] and result["selected"]["name"] == "v2_control"


def test_selector_rejects_partial_folds_and_changed_control():
    reports, baseline = selection_fixture()
    reports[1]["results"][0]["folds"] = reports[1]["results"][0]["folds"][:-1]
    with pytest.raises(ValueError, match="five"):
        select(reports, baseline)
    reports, baseline = selection_fixture()
    baseline["selected"]["validation_mean"] += 0.01
    with pytest.raises(ValueError, match="reproduce"):
        select(reports, baseline)
