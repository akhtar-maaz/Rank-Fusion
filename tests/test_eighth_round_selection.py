import copy
import json
from pathlib import Path

import pytest

from scripts.cv_augmented_lambdamart import summarize
from scripts.select_eighth_round import CONTROL, main, select, specifications


def row(name, clean=0.487, diagnostic=0.475, *, clean_deltas=None, diagnostic_deltas=None):
    members = [dict(params={"num_leaves": 127}, trees=count, feature_mode="query_all", weight=weight)
               for count, weight in zip((800, 200, 800, 800, 800, 400),
                                        (0.0703125, 0.0703125, 0.140625, 0.28125, 0.1875, 0.25))]
    clean_deltas = [0] * 5 if clean_deltas is None else clean_deltas
    diagnostic_deltas = [0] * 5 if diagnostic_deltas is None else diagnostic_deltas
    folds = [dict(fold=fold, **{"train_ndcg@10": 0.75},
                  scenarios=dict(clean=clean + clean_deltas[fold],
                                 affine=diagnostic + diagnostic_deltas[fold],
                                 feature_dropout=diagnostic + diagnostic_deltas[fold],
                                 noise=diagnostic + diagnostic_deltas[fold]))
             for fold in range(5)]
    return summarize(name, members, folds)


def fixture():
    control = row(CONTROL)
    return control, dict(selected=copy.deepcopy(control))


def test_eighth_round_exact_frozen_fit_configurations():
    path = Path(__file__).resolve().parents[1] / "configs/v8_candidates.json"
    actual = json.loads(path.read_text())
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], num_leaves=127,
                  learning_rate=0.05, min_data_in_leaf=100, lambda_l2=10, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True,
                  force_row_wise=True, verbose=-1, num_threads=4, lambdarank_truncation_level=30,
                  extra_trees=True, extra_seed=0)
    expected = []
    for name, overrides, augmentation in (
            ("cross_aug_extra", dict(extra_seed=17), dict(name="full_query_mixed_v8", seed=20261023)),
            ("query_bagging_extra", dict(bagging_by_query=True), dict(name="full_query_mixed", seed=20261021)),
            ("node_subsample_extra", dict(feature_fraction_bynode=0.6), dict(name="full_query_mixed", seed=20261021)),
            ("regularized_extra", dict(num_leaves=63, min_data_in_leaf=200, lambda_l2=20),
             dict(name="full_query_mixed", seed=20261021))):
        expected.append(dict(name=name, feature_mode="query_all", params=dict(params, **overrides),
                             augmentation=augmentation, trees=[200, 400, 600]))
    assert actual == expected


def test_eighth_round_finite_plan_reproduces_exact_v7_sources():
    settings = specifications()
    assert len(settings) == len({setting["name"] for setting in settings}) == 49
    assert settings[0] == dict(name=CONTROL,
                              sources=[("v3", 0, 800), ("v3", 1, 200), ("v4", 3, 800),
                                       ("v5", 1, 800), ("v6", 3, 800), ("v7", 2, 400)],
                              weights=[0.0703125, 0.0703125, 0.140625, 0.28125, 0.1875, 0.25])
    assert len(specifications([0, 3])) == 25
    assert specifications([]) == [settings[0]]
    for setting in settings:
        assert len(setting["sources"]) == len(setting["weights"])
        assert sum(setting["weights"]) == 1 and min(setting["weights"]) > 0
        new = [source for source in setting["sources"] if source[0] == "v8"]
        if setting["name"] != CONTROL:
            assert len(new) == 1 and new[0][1] in range(4) and new[0][2] in (200, 400, 600)
    blend = next(setting for setting in settings if setting["name"] == "v7_plus_2_600_weight0.5")
    assert blend["sources"] == settings[0]["sources"] + [("v8", 2, 600)]
    assert blend["weights"] == [value * 0.5 for value in settings[0]["weights"]] + [0.5]


def test_eighth_round_accepts_actual_archived_v7_control_without_refitting():
    path = Path(__file__).resolve().parents[1] / "results/lambdamart_cv_v7.json"
    baseline = json.loads(path.read_text())
    control = dict(copy.deepcopy(baseline["selected"]), name=CONTROL)
    plan = specifications()[0]
    assert control["trees"] == 3800
    assert [tuple(source) for source in control["sources"]] == plan["sources"]
    assert [member["weight"] for member in control["members"]] == plan["weights"]
    result = select([control], baseline)
    assert not result["promoted"] and result["selected"] == control
    assert result["v7_control"] == control and result["promotion_gates"]["improved_folds"] == 0


def test_eighth_round_promotion_records_every_gate_and_fold_gain():
    control, baseline = fixture()
    candidate = row("candidate", 0.488, 0.477)
    result = select([control, candidate], baseline)
    assert result["promoted"] and result["selected"] == candidate
    assert result["v7_control"] == control and "v6_control" not in result
    gates = result["promotion_gates"]
    assert gates["candidate"] == "candidate" and gates["passed"]
    assert gates["mixture_passed"] and gates["diagnostic_passed"] and gates["paired_folds_passed"]
    assert gates["mixture_gain"] == pytest.approx(0.0015)
    assert gates["diagnostic_gain"] == pytest.approx(0.002)
    assert gates["improved_folds"] == 5
    assert [fold["fold"] for fold in gates["fold_mixture_gains"]] == list(range(5))
    assert all(fold["mixture_gain"] == pytest.approx(0.0015) for fold in gates["fold_mixture_gains"])
    protocol = result["selection_protocol"]
    assert (protocol["clean_tolerance"], protocol["per_diagnostic_tolerance"],
            protocol["minimum_mixture_improvement"], protocol["minimum_diagnostic_improvement"],
            protocol["minimum_improved_folds"]) == (0.0025, 0.0025, 0.0005, 0.001, 3)


def test_eighth_round_diagnostic_gate_retains_control_without_runner_up_search():
    control, baseline = fixture()
    balanced = row("balanced", 0.4877, 0.4762)
    high_clean = row("high_clean", 0.490, 0.4754)
    result = select([control, balanced, high_clean], baseline)
    assert "balanced" in result["eligible_settings"]
    assert result["promotion_gates"]["candidate"] == "high_clean"
    assert result["promotion_gates"]["mixture_passed"]
    assert not result["promotion_gates"]["diagnostic_passed"]
    assert not result["promoted"] and result["selected"] == control


def test_eighth_round_paired_gate_rejects_two_fold_gain_without_runner_up_search():
    control, baseline = fixture()
    balanced = row("balanced", diagnostic=0.4771)
    concentrated = row("concentrated", diagnostic_deltas=[0.008, 0.008, -0.001, -0.001, -0.001])
    result = select([control, balanced, concentrated], baseline)
    gates = result["promotion_gates"]
    assert gates["candidate"] == "concentrated"
    assert gates["mixture_passed"] and gates["diagnostic_passed"]
    assert gates["improved_folds"] == 2 and not gates["paired_folds_passed"]
    assert not result["promoted"] and result["selected"] == control


def test_eighth_round_pairs_folds_by_identity_and_zero_gain_is_not_positive():
    control = row(CONTROL, clean_deltas=[-0.02, -0.01, 0, 0.01, 0.02])
    baseline = dict(selected=copy.deepcopy(control))
    candidate = row("candidate", clean_deltas=[-0.02, -0.01, 0, 0.01, 0.02], diagnostic=0.477)
    candidate["folds"].reverse()
    result = select([control, candidate], baseline)
    assert result["promoted"] and result["promotion_gates"]["improved_folds"] == 5
    control, baseline = fixture()
    two_positive = row("two_positive", diagnostic_deltas=[0.006, 0.006, 0, 0, 0])
    result = select([control, two_positive], baseline)
    assert result["promotion_gates"]["improved_folds"] == 2
    assert not result["promoted"]


def test_eighth_round_preserves_clean_and_single_diagnostic_guards():
    control, baseline = fixture()
    balanced = row("balanced", 0.488, 0.477)
    clean_loss = row("clean_loss", 0.48, 0.49)
    unbalanced = row("unbalanced", 0.488, 0.49)
    for fold in unbalanced["folds"]:
        fold["scenarios"]["affine"] = 0.47
    unbalanced = summarize(unbalanced["name"], unbalanced["members"], unbalanced["folds"])
    result = select([control, balanced, clean_loss, unbalanced], baseline)
    assert result["promoted"] and result["selected"] == balanced
    assert "clean_loss" not in result["eligible_settings"]
    assert "unbalanced" not in result["eligible_settings"]
    weak = row("weak", 0.487, 0.4751)
    result = select([control, weak], baseline)
    assert not result["promotion_gates"]["mixture_passed"]
    assert not result["promoted"] and result["selected"] == control


def test_eighth_round_rejects_incomplete_or_changed_control_results():
    control, baseline = fixture()
    candidate = row("candidate", 0.488, 0.477)
    candidate["folds"].pop()
    with pytest.raises(ValueError, match="five"):
        select([control, candidate], baseline)
    control, baseline = fixture()
    baseline["selected"]["members"][0]["weight"] = 0.1
    with pytest.raises(ValueError, match="specifications"):
        select([control], baseline)
    control, baseline = fixture()
    baseline["selected"]["validation_mean"] += 0.001
    with pytest.raises(ValueError, match="reproduce"):
        select([control], baseline)
    control, baseline = fixture()
    control["scenario_means"]["noise"] += 0.01
    with pytest.raises(ValueError, match="diagnostic means"):
        select([control], baseline)


@pytest.mark.parametrize("exclusions", ([0, 0], [4]))
def test_eighth_round_cli_rejects_invalid_runtime_exclusions(monkeypatch, capsys, exclusions):
    monkeypatch.setattr("sys.argv", ["select_eighth_round.py", "--excluded-candidates", *map(str, exclusions)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "exclusions must name distinct candidates 0..3" in capsys.readouterr().err
