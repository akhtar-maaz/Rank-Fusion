import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer
from harness.preprocessing import Standardizer
from scripts.cv_pairwise_ensemble_v7 import select, specifications
from scripts.pairwise_ensemble import combine_scorers, same_standardizer


def member(seed, hidden, width=136, standardizer=None):
    torch.manual_seed(seed)
    network = ScoringNetwork(width, hidden)
    if standardizer is None:
        standardizer = Standardizer(np.zeros(width), np.ones(width))
    return network, standardizer


@pytest.mark.parametrize("different_statistics", [False, True])
def test_compiled_average_loads_with_pristine_harness_and_matches_scores(tmp_path, different_statistics):
    rng = np.random.RandomState(1729)
    X = rng.normal(size=(137, 136)).astype(np.float32)
    standardizer = Standardizer(rng.normal(size=136), rng.uniform(0.5, 2.0, size=136))
    other = Standardizer(rng.normal(size=136), rng.uniform(0.5, 2.0, size=136)) if different_statistics else standardizer
    members = [member(0, [128, 64], standardizer=standardizer), member(17, [128, 64], standardizer=other)]
    network, common = combine_scorers(members, [0.5, 0.5])
    assert network.hidden_sizes == [256, 128]
    assert len(network.hidden_sizes) <= MAX_HIDDEN_LAYERS and max(network.hidden_sizes) <= MAX_WIDTH
    first_weights = members[0][0].net[0].weight.detach().clone()
    torch.testing.assert_close(network.net[0].weight[:128], first_weights, rtol=0, atol=0)
    if not different_statistics:
        torch.testing.assert_close(network.net[0].weight[128:], members[1][0].net[0].weight, rtol=0, atol=0)
    path = tmp_path / "pairwise_model.pt"
    save_scorer(str(path), network, common)
    blob = torch.load(path, weights_only=True)
    assert set(blob) == {"config", "state_dict", "standardizer"}
    assert blob["config"] == {"n_features": 136, "hidden": [256, 128]}
    actual = load_scorer(str(path))(X)
    expected = 0.5 * make_scorer(*members[0])(X) + 0.5 * make_scorer(*members[1])(X)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert same_standardizer(common, standardizer)
    torch.testing.assert_close(members[0][0].net[0].weight, first_weights, rtol=0, atol=0)


def test_compilation_handles_three_layers_unequal_widths_and_normalized_weights():
    members = [member(0, [11, 7, 5], width=4), member(41, [9, 8, 3], width=4)]
    X = np.random.RandomState(0).normal(size=(19, 4)).astype(np.float32)
    rng_before = torch.random.get_rng_state().clone()
    network, standardizer = combine_scorers(members, [2, 6])
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert network.hidden_sizes == [20, 15, 8]
    expected = 0.25 * make_scorer(*members[0])(X) + 0.75 * make_scorer(*members[1])(X)
    np.testing.assert_allclose(make_scorer(network, standardizer)(X), expected, atol=2e-6, rtol=2e-5)


def test_compilation_rejects_incompatible_or_out_of_bounds_members():
    with pytest.raises(ValueError, match="at least one"):
        combine_scorers([])
    for weights in ([1], [0, 1], [-1, 2], [np.nan, 1]):
        with pytest.raises(ValueError, match="positive finite weight"):
            combine_scorers([member(0, [8]), member(17, [8])], weights)
    with pytest.raises(ValueError, match="bounds"):
        combine_scorers([member(0, [129]), member(17, [128])])
    with pytest.raises(ValueError, match="depth"):
        combine_scorers([member(0, [8]), member(17, [8, 4])])
    with pytest.raises(ValueError, match="input width"):
        combine_scorers([member(0, [8], width=3), member(17, [8], width=4)])
    with pytest.raises(ValueError, match="positive finite"):
        combine_scorers([member(0, [8], width=2, standardizer=Standardizer(np.zeros(2), np.array([1, 0])))])


def fold_reports():
    specs = specifications()
    context = dict(train_sha256="synthetic", rows=100, queries=20, features=136, folds=5, split_seed=0)
    reports = []
    for fold in range(5):
        rows = []
        for index, candidate in enumerate(specs):
            score = 0.4 + 0.005 * index
            rows.append(dict(fold=fold, train_ndcg_10=0.6, validation_ndcg_10=score,
                             standardizers_identical=True, control_reference_checked=False,
                             train_compilation=dict(exported_ndcg_10=0.6),
                             validation_compilation=dict(exported_ndcg_10=score),
                             members=[dict(specification=spec, history=[dict(epoch=epoch, **{"train_ndcg@10": 0.6})
                                                                      for epoch in range(1, spec["epochs"]+1)])
                                      for spec in candidate["members"]]))
        reports.append(dict(fold=fold, context=dict(context, fold=fold, train_indices_sha256=str(fold),
                                                   validation_indices_sha256=str(fold)),
                            specifications=specs, results=rows))
    return reports, specs


def test_frozen_plan_and_selection_use_all_exported_fold_scores():
    path = Path(__file__).resolve().parents[1] / "configs/v7_neural_candidates.json"
    assert json.loads(path.read_text()) == specifications()
    reports, specs = fold_reports()
    result = select(reports, specs)
    assert result["promoted"] and result["selected_index"] == 2
    assert not result["selection_protocol"]["released_dev_used"]
    assert result["proposed_improved_folds"] == 5
    for report in reports:
        report["results"][1]["validation_ndcg_10"] = 0.4
        report["results"][1]["validation_compilation"]["exported_ndcg_10"] = 0.4
        report["results"][2]["validation_ndcg_10"] = 0.4005
        report["results"][2]["validation_compilation"]["exported_ndcg_10"] = 0.4005
    assert not select(reports, specs)["promoted"]
    # A high average concentrated in two folds fails the fixed wins guard.
    for fold, report in enumerate(reports):
        score = 0.5 if fold < 2 else 0.4
        report["results"][2]["validation_ndcg_10"] = score
        report["results"][2]["validation_compilation"]["exported_ndcg_10"] = score
    assert not select(reports, specs)["promoted"]


def test_selection_rejects_missing_folds_changed_fits_and_non_exported_metrics():
    reports, specs = fold_reports()
    with pytest.raises(ValueError, match="five distinct"):
        select(reports[:-1], specs)
    damaged = copy.deepcopy(reports)
    damaged[1]["context"]["train_sha256"] = "other TRAIN"
    with pytest.raises(ValueError, match="different TRAIN"):
        select(damaged, specs)
    damaged = copy.deepcopy(reports)
    damaged[1]["results"][1]["members"][0]["history"].pop()
    with pytest.raises(ValueError, match="fixed-epoch"):
        select(damaged, specs)
    damaged = copy.deepcopy(reports)
    damaged[1]["results"][1]["validation_ndcg_10"] = 0.9
    with pytest.raises(ValueError, match="exported plain MLP"):
        select(damaged, specs)
    damaged = copy.deepcopy(reports)
    for report in damaged:
        report["context"].update(rows=723412, queries=6000, features=136)
    with pytest.raises(ValueError, match="prior fold"):
        select(damaged, specs)
