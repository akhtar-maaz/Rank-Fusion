import copy
import json
from pathlib import Path

import pytest

from scripts.cv_augmented_lambdamart import summarize
from scripts.select_fifth_round import CONTROL, select, specifications


def row(name, clean, affine, dropout, noise, members=None):
    if members is None:
        members = [dict(params={"num_leaves": 63}, trees=400, feature_mode="query_all")]
    folds = [{"fold": fold, "train_ndcg@10": 0.7,
              "scenarios": dict(clean=clean, affine=affine, feature_dropout=dropout, noise=noise)}
             for fold in range(5)]
    return summarize(name, members, folds)


def fixture():
    members = [dict(params={"num_leaves": 127}, trees=800, feature_mode="query_rank", weight=0.25),
               dict(params={"num_leaves": 127}, trees=200, feature_mode="query_normalized", weight=0.25),
               dict(params={"num_leaves": 127}, trees=800, feature_mode="query_rank", weight=0.5,
                    augmentation={"name": "half_query_dropout_noise", "seed": 20261019})]
    control = row(CONTROL, 0.486, 0.467, 0.475, 0.469, members)
    return control, dict(selected=copy.deepcopy(control))


def test_fifth_round_fixed_finite_search_and_frozen_specs():
    settings = specifications()
    assert len(settings) == len({setting["name"] for setting in settings}) == 54
    for setting in settings:
        assert len(setting["sources"]) == len(setting["weights"])
        assert sum(setting["weights"]) == 1
        assert all(weight > 0 for weight in setting["weights"])
    assert len(specifications([0, 1])) == 30
    path = Path(__file__).resolve().parents[1] / "configs/v5_candidates.json"
    candidates = json.loads(path.read_text())
    assert len(candidates) == 4
    assert all(candidate["params"]["objective"] == "lambdarank" and
               candidate["params"]["lambdarank_truncation_level"] == 30 for candidate in candidates)


def test_fifth_round_guard_rejects_a_large_single_diagnostic_regression():
    control, baseline = fixture()
    balanced = row("balanced", 0.486, 0.47, 0.476, 0.47)
    unbalanced = row("unbalanced", 0.486, 0.46, 0.49, 0.49)
    result = select([control, balanced, unbalanced], baseline)
    assert result["promoted"] and result["selected"]["name"] == "balanced"
    assert "unbalanced" not in result["eligible_settings"]


def test_fifth_round_retains_v4_when_no_candidate_passes_both_guards():
    control, baseline = fixture()
    clean_loss = row("clean_loss", 0.48, 0.48, 0.49, 0.49)
    weak = row("weak", 0.486, 0.467, 0.4751, 0.469)
    result = select([control, clean_loss, weak], baseline)
    assert not result["promoted"] and result["selected"] == control
    high_clean = row("high_clean", 0.5, 0.4, 0.4, 0.4)
    result = select([control, high_clean], baseline)
    assert not result["eligible_settings"] and result["selected"] == control


def test_fifth_round_rejects_partial_results_and_changed_control():
    control, baseline = fixture()
    candidate = row("candidate", 0.486, 0.47, 0.476, 0.47)
    candidate["folds"].pop()
    with pytest.raises(ValueError, match="five"):
        select([control, candidate], baseline)
    control, baseline = fixture()
    baseline["selected"]["members"][0]["trees"] = 400
    with pytest.raises(ValueError, match="specifications"):
        select([control], baseline)
    control, baseline = fixture()
    control["scenario_means"]["noise"] += 0.01
    with pytest.raises(ValueError, match="diagnostic means"):
        select([control], baseline)
