import copy
import json
from pathlib import Path

import numpy as np
import pytest

from harness.cv import grouped_kfold
from scripts.augmentation import augment_training
from scripts.cv_pairwise_ensemble_v12 import specifications as neural_specs
from scripts.lambdamart_utils import default_params, fit_specification
from scripts.robustness import perturb_features
from scripts.select_twelfth_round import specifications, select
from submission.data import group_sizes


@pytest.mark.parametrize("strength", [1., 2.])
def test_balanced_copies_cover_all_domains_are_query_local_and_label_independent(strength):
    rng = np.random.RandomState(8)
    X = rng.normal(size=(120, 12)).astype(np.float32)
    X[1, 4] = np.nan
    y = rng.randint(0, 5, len(X))
    q = np.repeat([f"opaque:{i}" for i in range(20)], 6)
    tr, va = grouped_kfold(q)[0]
    config = dict(name="balanced_query_domains", seed=20261024, strength=strength)
    features, labels, ids = augment_training(X[tr], y[tr], q[tr], config)
    assert len(features) == 4 * len(tr) and np.all(group_sizes(ids) == np.array([6] * (4 * 16)))
    assert len(set(ids)) == 4 * len(set(q[tr]))
    assert np.array_equal(features[:len(tr)], X[tr], equal_nan=True)
    assert np.array_equal(labels, np.tile(y[tr], 4))
    for i, scenario in enumerate(("affine", "feature_dropout", "noise"), 1):
        block = slice(i * len(tr), (i + 1) * len(tr))
        assert np.array_equal(features[block], perturb_features(X[tr], q[tr], scenario, seed=20261024, strength=strength), equal_nan=True)
        assert set(ids[block]) == {f"augmented:{scenario}:{v}" for v in set(q[tr])}
    changed, _, changed_ids = augment_training(X[tr], np.zeros_like(y[tr]), q[tr], config)
    assert np.array_equal(changed, features, equal_nan=True) and np.array_equal(changed_ids, ids)
    assert not any(identity.endswith(":" + query) for query in q[va] for identity in set(ids))
    query = q[tr][0]
    single, _, single_ids = augment_training(X[q == query], y[q == query], q[q == query], config)
    for identity in set(single_ids):
        assert np.array_equal(single[single_ids == identity], features[ids == identity], equal_nan=True)


def test_balanced_rejects_unplanned_settings_and_actual_fit_is_reproducible():
    X = np.random.RandomState(2).normal(size=(100, 4)).astype(np.float32)
    y, q = np.tile([0, 1, 2, 3, 0], 20), np.repeat(np.arange(20), 5)
    config = dict(name="balanced_query_domains", seed=20261024, strength=1.)
    for bad in (dict(config, seed=1729), dict(config, strength=3), dict(config, copies=2)):
        with pytest.raises(ValueError, match="configuration"):
            augment_training(X, y, q, bad)
    spec = dict(params=default_params(7, min_data_in_leaf=2), trees=8,
                feature_mode="query_all", augmentation=config)
    assert fit_specification(X, y, q, spec).model_to_string() == fit_specification(X, y, q, spec).model_to_string()


def test_v12_frozen_families_and_exact_baseline_control():
    tree = json.loads(Path('configs/v12_candidates.json').read_text())
    assert len(tree) == 4 and all(row['trees'] == [200, 400] and row['feature_mode'] == 'query_all' for row in tree)
    assert [(r['augmentation']['strength'], r['params']['feature_fraction']) for r in tree] == [(1., .8), (2., .8), (1., .5), (2., .5)]
    assert len(specifications()) == 33
    neural = neural_specs()
    assert neural[0] == json.loads(Path('results/neural_selection_v8.json').read_text())['selected']['candidate']
    baseline = json.loads(Path('results/lambdamart_cv_release_v11.json').read_text())
    row = copy.deepcopy(baseline['selected']); row['name'] = 'cached_v11_control'
    assert not select([row], baseline)['promoted']
