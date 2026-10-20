import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.select_ninth_round import CONTROL, select, specifications
from scripts.lambdamart_utils import default_params, fit
from submission.features import features_for_model, tree_features
from submission.tree_ensemble import combine


def test_ninth_round_exact_plan_and_evaluated_control():
    settings = specifications()
    assert len(settings) == len({s['name'] for s in settings}) == 49
    assert len(specifications([1, 3])) == 25
    baseline = json.loads((Path(__file__).parents[1] / 'results/lambdamart_cv_v8.json').read_text())
    row = dict(copy.deepcopy(baseline['selected']), name=CONTROL)
    assert row['trees'] == 3800
    assert [tuple(s) for s in row['sources']] == settings[0]['sources']
    assert [m['weight'] for m in row['members']] == settings[0]['weights']
    result = select([row], baseline)
    assert not result['promoted'] and result['v8_control'] == row
    for setting in settings[1:]:
        new = [s for s in setting['sources'] if s[0] == 'v9']
        assert len(new) == 1 and new[0][2] in (400, 800, 1200)
        assert sum(setting['weights']) == 1


def test_ninth_round_does_not_rescue_v8_rejected_setting():
    baseline = json.loads((Path(__file__).parents[1] / 'results/lambdamart_cv_v8.json').read_text())
    control = dict(copy.deepcopy(baseline['v7_control']), name=CONTROL)
    proposed = next(r for r in baseline['results'] if r['name'] == baseline['promotion_gates']['candidate'])
    result = select([control, proposed], baseline)
    assert not result['promoted'] and not result['promotion_gates']['diagnostic_passed']
    assert result['selected'] == control


def test_ninth_round_frozen_candidate_pairs():
    rows = json.loads((Path(__file__).parents[1] / 'configs/v9_candidates.json').read_text())
    assert len(rows) == 4 and [r['augmentation'] is None for r in rows] == [True, False, True, False]
    assert rows[0]['params'] == rows[1]['params'] and rows[2]['params'] == rows[3]['params']
    assert rows[0]['params']['objective'] == 'rank_xendcg'
    assert rows[0]['params']['objective_seed'] == 0
    assert rows[2]['params']['objective'] == 'lambdarank'
    assert (rows[2]['params']['num_leaves'], rows[2]['params']['max_depth'], rows[2]['params']['path_smooth']) == (31, 5, 100)
    assert all(r['trees'] == [400, 800, 1200] and r['feature_mode'] == 'query_all' for r in rows)


@pytest.mark.parametrize('first_objective', ['lambdarank', 'rank_xendcg'])
def test_mixed_ranking_objectives_export_explicit_scores(tmp_path, first_objective):
    import lightgbm as lgb
    rng = np.random.RandomState(29)
    X = rng.normal(size=(240, 5)).astype(np.float32)
    q = np.repeat(np.arange(30), 8)
    y = rng.randint(0, 5, len(q))
    first = fit(X, y, q, default_params(7, min_data_in_leaf=2, objective=first_objective), 12, 'query_rank')
    second = fit(X, y, q, default_params(7, min_data_in_leaf=2, objective='rank_xendcg', objective_seed=0), 15, 'query_normalized')
    expected = (.75 * first.predict(tree_features(X, q, 'query_rank'), num_iteration=8, num_threads=1) +
                .25 * second.predict(tree_features(X, q, 'query_normalized'), num_iteration=10, num_threads=1))
    model = combine([first, second], [.75, .25], [8, 10])
    model.save_model(str(tmp_path / 'model.txt'))
    restored = lgb.Booster(model_file=str(tmp_path / 'model.txt'))
    actual = restored.predict(features_for_model(restored, X, q), num_threads=1)
    assert restored.num_trees() == 18 and restored.num_feature() == 15
    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=0)
    assert np.array_equal(np.argsort(-actual, kind='stable'), np.argsort(-expected, kind='stable'))


def test_export_still_rejects_nonranking_objective():
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    q = np.repeat(np.arange(10), 4)
    m = fit(X, np.tile([0, 1, 2, 3], 10), q, default_params(3, min_data_in_leaf=2, objective='regression'), 3)
    with pytest.raises(ValueError, match='ordinary, single-output'):
        combine([m, m], [.5, .5])
