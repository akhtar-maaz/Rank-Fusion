import copy

import pytest

from scripts.cv_augmented_lambdamart import summarize
from scripts.select_seventh_round import CONTROL, select, specifications


def row(name, clean=0.487, diagnostic=0.475, *, clean_deltas=None, diagnostic_deltas=None):
    members = [dict(params={"num_leaves": 127}, trees=count, feature_mode="query_all", weight=weight)
               for count, weight in zip((800, 200, 800, 800, 800),
                                        (0.09375, 0.09375, 0.1875, 0.375, 0.25))]
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


def test_seventh_round_finite_plan_reproduces_exact_v6_sources():
    settings = specifications()
    assert len(settings) == len({setting["name"] for setting in settings}) == 49
    assert settings[0] == dict(name=CONTROL,
                              sources=[("v3", 0, 800), ("v3", 1, 200), ("v4", 3, 800),
                                       ("v5", 1, 800), ("v6", 3, 800)],
                              weights=[0.09375, 0.09375, 0.1875, 0.375, 0.25])
    assert len(specifications([0, 3])) == 25
    assert specifications([]) == [settings[0]]
    for setting in settings:
        assert len(setting["sources"]) == len(setting["weights"])
        assert sum(setting["weights"]) == 1 and min(setting["weights"]) > 0
        new = [source for source in setting["sources"] if source[0] == "v7"]
        if setting["name"] != CONTROL:
            assert len(new) == 1 and new[0][1] in range(4) and new[0][2] in (200, 400, 800)
    blend = next(setting for setting in settings if setting["name"] == "v6_plus_2_400_weight0.5")
    assert blend["sources"] == settings[0]["sources"] + [("v7", 2, 400)]
    assert blend["weights"] == [value * 0.5 for value in settings[0]["weights"]] + [0.5]


def test_seventh_round_promotion_records_every_gate_and_fold_gain():
    control, baseline = fixture()
    candidate = row("candidate", 0.488, 0.477)
    result = select([control, candidate], baseline)
    assert result["promoted"] and result["selected"] == candidate
    assert result["v6_control"] == control and "v4_control" not in result
    gates = result["promotion_gates"]
    assert gates["candidate"] == "candidate" and gates["passed"]
    assert gates["mixture_passed"] and gates["diagnostic_passed"] and gates["paired_folds_passed"]
    assert gates["mixture_gain"] == pytest.approx(0.0015)
    assert gates["diagnostic_gain"] == pytest.approx(0.002)
    assert gates["improved_folds"] == 5
    assert [fold["fold"] for fold in gates["fold_mixture_gains"]] == list(range(5))
    assert all(fold["mixture_gain"] == pytest.approx(0.0015) for fold in gates["fold_mixture_gains"])


def test_seventh_round_diagnostic_gate_retains_control_without_searching_runner_up():
    control, baseline = fixture()
    balanced = row("balanced", 0.4877, 0.4762)
    high_clean = row("high_clean", 0.490, 0.4754)
    result = select([control, balanced, high_clean], baseline)
    assert "balanced" in result["eligible_settings"]
    assert result["promotion_gates"]["candidate"] == "high_clean"
    assert result["promotion_gates"]["mixture_passed"]
    assert not result["promotion_gates"]["diagnostic_passed"]
    assert not result["promoted"] and result["selected"] == control


def test_seventh_round_paired_gate_rejects_two_fold_gain_without_runner_up_search():
    control, baseline = fixture()
    balanced = row("balanced", diagnostic=0.4771)
    concentrated = row("concentrated", diagnostic_deltas=[0.008, 0.008, -0.001, -0.001, -0.001])
    result = select([control, balanced, concentrated], baseline)
    gates = result["promotion_gates"]
    assert gates["candidate"] == "concentrated"
    assert gates["mixture_passed"] and gates["diagnostic_passed"]
    assert gates["improved_folds"] == 2 and not gates["paired_folds_passed"]
    assert not result["promoted"] and result["selected"] == control


def test_seventh_round_pairs_folds_by_identity_and_zero_gain_is_not_positive():
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


def test_seventh_round_preserves_clean_and_single_diagnostic_guards():
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


def test_seventh_round_rejects_incomplete_or_changed_control_results():
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
