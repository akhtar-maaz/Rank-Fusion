import numpy as np
import pytest

from harness.cv import grouped_kfold
from scripts.augmentation import augment_training
from scripts.cv_augmented_lambdamart import candidates
from scripts.lambdamart_utils import default_params, fit_selected, fit_specification
from submission.data import group_sizes
from submission.features import features_for_model

CONFIG = {"name": "half_query_dropout_noise", "seed": 20261019}
MIXED = {"name": "half_query_mixed", "seed": 20261020}
FULL_MIXED = {"name": "full_query_mixed", "seed": 20261021}


def sample():
    rng = np.random.RandomState(21)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    X[3, 4] = np.nan
    y = rng.randint(0, 5, len(X))
    q = np.repeat([f"q{i}" for i in range(50)], 6)
    return X, y, q


@pytest.mark.parametrize("config", [CONFIG, MIXED, FULL_MIXED])
def test_augmentation_keeps_originals_and_query_labels_without_leaking_validation(config):
    X, y, q = sample()
    original = X.copy()
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    augmented, labels, ids = augment_training(X[tr], y[tr], q[tr], config)
    assert np.array_equal(augmented[:len(tr)], X[tr], equal_nan=True)
    assert np.array_equal(labels[:len(tr)], y[tr])
    if config == FULL_MIXED:
        assert len(augmented) == 2 * len(tr)
    else:
        assert len(tr) < len(augmented) < 2 * len(tr)
    assert all(size == 6 for size in group_sizes(ids))
    assert set(identity.split(":", 1)[1] for identity in ids).isdisjoint(q[va])
    for identity in dict.fromkeys(ids):
        source = identity.split(":", 1)[1]
        assert np.array_equal(labels[ids == identity], y[q == source])
        assert np.array_equal(np.isnan(augmented[ids == identity]), np.isnan(X[q == source]))
    assert np.array_equal(X, original, equal_nan=True)
    alternative_labels = np.zeros_like(y[tr])
    changed, _, changed_ids = augment_training(X[tr], alternative_labels, q[tr], config)
    assert np.array_equal(augmented, changed, equal_nan=True)
    assert np.array_equal(ids, changed_ids)


@pytest.mark.parametrize("config", [CONFIG, MIXED, FULL_MIXED])
def test_augmentation_is_query_local_and_repeatable(config):
    X, y, q = sample()
    whole, _, ids = augment_training(X, y, q, config)
    for query in dict.fromkeys(q):
        rows = q == query
        single, _, single_ids = augment_training(X[rows], y[rows], q[rows], config)
        for identity in dict.fromkeys(single_ids):
            assert np.array_equal(single[single_ids == identity], whole[ids == identity], equal_nan=True)
    with pytest.raises(ValueError, match="configuration"):
        augment_training(X, y, q, dict(CONFIG, seed=1729))


def test_augmented_selected_fit_reproduces_direct_fit_and_native_average():
    X, y, q = sample()
    base = dict(params=default_params(7, min_data_in_leaf=2), trees=8,
                feature_mode="query_normalized", augmentation=CONFIG)
    direct = fit_specification(X, y, q, base)
    selected = fit_selected(X, y, q, base)
    assert direct.model_to_string() == selected.model_to_string()
    members = [dict(base, weight=0.5), dict(base, feature_mode="query_rank", weight=0.5)]
    combined = fit_selected(X, y, q, dict(members=members))
    expected = sum(0.5 * model.predict(features_for_model(model, X, q), num_threads=1)
                   for model in (fit_specification(X, y, q, member) for member in members))
    assert np.allclose(combined.predict(features_for_model(combined, X, q), num_threads=1), expected, atol=1e-12)
    assert all(candidate["params"]["objective"] == "lambdarank" for candidate in candidates())


def test_mixed_augmentation_exercises_affine_dropout_and_noise(monkeypatch):
    import scripts.augmentation as augmentation
    observed = []
    original = augmentation.perturb_features

    def track(X, qids, scenario, seed):
        observed.append((scenario, seed))
        return original(X, qids, scenario, seed)

    monkeypatch.setattr(augmentation, "perturb_features", track)
    augmentation.augment_training(*sample(), MIXED)
    assert {name for name, _ in observed} == {"affine", "feature_dropout", "noise"}
    assert all(seed != 1729 for _, seed in observed)
