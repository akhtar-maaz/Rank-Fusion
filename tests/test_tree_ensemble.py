import lightgbm as lgb
import numpy as np
import pytest

from scripts.lambdamart_utils import default_params, fit, fit_selected
from submission import ranking
from submission.features import features_for_model, tree_features
from submission.tree_ensemble import combine


@pytest.mark.parametrize("first_mode", ["raw", "query_rank"])
def test_native_average_matches_both_transforms_after_save(tmp_path, first_mode):
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    q = np.repeat(np.arange(20), 6)
    y = (X[:, 0] > 0).astype(int)
    params = default_params(num_leaves=7, min_data_in_leaf=2)
    first = fit(X, y, q, params, 12, first_mode)
    second = fit(X, y, q, params, 15, "query_normalized")
    expected = (0.7 * first.predict(tree_features(X, q, first_mode), num_iteration=8, num_threads=1) +
                0.3 * second.predict(tree_features(X, q, "query_normalized"), num_iteration=10, num_threads=1))
    ensemble = combine([first, second], [0.7, 0.3], [8, 10])
    path = str(tmp_path / "model.txt")
    ensemble.save_model(path)
    restored = lgb.Booster(model_file=path)
    assert restored.num_trees() == 18 and restored.num_feature() == 9
    assert np.allclose(restored.predict(features_for_model(restored, X, q), num_threads=1), expected, atol=1e-12)
    ranking.prepare(path)
    ids = [str(i) for i in range(6)]
    actual = ranking.rerank_with_ltr("q", ids, dict(zip(ids, X[:6].tolist())), 6)
    assert [doc for doc, _ in actual] == [ids[i] for i in np.argsort(-expected[:6], kind="stable")]


def test_native_average_reproducible_from_selected_config():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 2)).astype(np.float32)
    q = np.repeat(np.arange(10), 6)
    y = (X[:, 0] > 0).astype(int)
    selected = {"members": [dict(params=default_params(7, min_data_in_leaf=2), trees=5,
                                 feature_mode=mode, weight=0.5) for mode in ("query_rank", "query_normalized")]}
    first, second = fit_selected(X, y, q, selected), fit_selected(X, y, q, selected)
    assert first.model_to_string() == second.model_to_string()
    with pytest.raises(ValueError, match="positive"):
        combine([first, second], [0.5, -0.5])
