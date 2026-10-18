import numpy as np
import lightgbm as lgb

from scripts.lambdamart_utils import default_params, fit, ndcg10
from submission.features import features_for_model, query_percentiles, tree_features
import submission.ranking as ranking
from submission.fusion import combsum_scores, weighted_combsum_scores


def test_percentiles_ties_and_permutation():
    X = np.array([[2, 1], [0, 1], [2, 1], [4, 1]], dtype=np.float32)
    ranks = query_percentiles(X)
    assert np.allclose(ranks[:, 0], [0.5, 0, 0.5, 1])
    assert np.all(ranks[:, 1] == 0.5)
    order = [2, 0, 3, 1]
    assert np.array_equal(query_percentiles(X[order]), ranks[order])
    assert np.array_equal(query_percentiles(X[:1]), [[0.5, 0.5]])


def test_transform_is_query_local():
    X = np.array([[0], [2], [100], [101]], dtype=np.float32)
    result = tree_features(X, ["a", "a", "b", "b"], "query_rank")
    assert np.array_equal(result[:, 0], X[:, 0])
    assert np.array_equal(result[:, 1], [0, 1, 0, 1])


def test_weighted_fusion_is_separate_from_textbook_combsum():
    base = {"a": {"x": 1, "y": 10}, "b": {"x": 3, "y": 20}, "c": {"x": 2, "y": 30}}
    scores = weighted_combsum_scores(base, {"x": 0.75, "y": 0.25})
    assert scores == {"a": 0.0, "b": 0.875, "c": 0.625}
    assert combsum_scores(base) == {"a": 0.0, "b": 1.5, "c": 1.5}


def test_engineered_artifact_roundtrip_through_ranking(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    q = np.repeat(np.arange(20), 6)
    params = default_params(num_leaves=7, min_data_in_leaf=2)
    model = fit(X, y, q, params, 20, "query_rank")
    path = tmp_path / "qrank.txt"
    model.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    assert restored.num_feature() == 6
    assert ndcg10(restored, X, y, q) > 0.8
    ranking.prepare(str(path))
    ids = [str(i) for i in range(6)]
    features = dict(zip(ids, X[:6].tolist()))
    actual = ranking.rerank_with_ltr("q", ids, features, 6)
    scores = restored.predict(features_for_model(restored, X[:6]), num_threads=1)
    assert [d for d, _ in actual] == [ids[i] for i in np.argsort(-scores, kind="stable")]
