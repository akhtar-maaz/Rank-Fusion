"""Frozen neural v8 plan, cache isolation, control proof and staff artifacts."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harness.neural import ScoringNetwork, load_scorer, make_scorer, save_scorer
from harness.preprocessing import Standardizer
from scripts.cv_pairwise_ensemble_v8 import (check_control, digest_file, fit_key, fit_member,
                                            fit_specification, read_reference, run_fit, runtime,
                                            select, source_fingerprints, specifications,
                                            state_fingerprint, statistics_fingerprint)
from scripts.pairwise_ensemble import combine_scorers


ROOT = Path(__file__).resolve().parents[1]


def reports_fixture():
    specs = specifications()
    base = dict(train_sha256="synthetic", rows=100, queries=20, features=136, folds=5, split_seed=0,
                source_sha256=source_fingerprints(), runtime=runtime())
    reports = []
    for fold in range(5):
        context = dict(base, fold=fold, train_indices_sha256="train%d" % fold,
                       validation_indices_sha256="validation%d" % fold)
        rows = []
        for index, candidate in enumerate(specs):
            score = 0.44 + 0.0015 * index
            members = []
            for spec in candidate["members"]:
                members.append(dict(specification=copy.deepcopy(spec), cache_key=fit_key(spec),
                                    identity=dict(context=copy.deepcopy(context), specification=fit_specification(spec)),
                                    history=[dict(epoch=epoch, train_loss=0.3, **{"train_ndcg@10": 0.6})
                                             for epoch in range(1, 3)],
                                    artifact="missing_v8_fixture/fold%d/%s/seed_%d.pt" % (fold, fit_key(spec), spec["seed"]),
                                    artifact_sha256="a" * 64, state_sha256="b" * 64,
                                    standardizer_sha256="c" * 64))
            rows.append(dict(fold=fold, train_ndcg_10=0.6, validation_ndcg_10=score,
                             standardizers_identical=True, control_reference_checked=False,
                             exported_config=dict(n_features=136, hidden=[256, 128]),
                             train_compilation=dict(exported_ndcg_10=0.6),
                             validation_compilation=dict(exported_ndcg_10=score), members=members))
        reports.append(dict(fold=fold, context=context, results=rows, specifications=copy.deepcopy(specs),
                            distinct_member_fits=9))
    reference = dict(candidate=copy.deepcopy(specs[0]),
                     folds=[copy.deepcopy(report["results"][0]) for report in reports])
    return reports, specs, reference


def change_score(report, candidate, score):
    report["results"][candidate]["validation_ndcg_10"] = score
    report["results"][candidate]["validation_compilation"]["exported_ndcg_10"] = score


def test_frozen_plan_has_nine_disjoint_complete_fit_keys():
    specs = specifications()
    assert json.loads((ROOT / "configs/v8_neural_candidates.json").read_text()) == specs
    assert [len(candidate["members"]) for candidate in specs] == [2, 2, 2, 2, 1]
    members = [member for candidate in specs for member in candidate["members"]]
    assert len(members) == len({fit_key(member) for member in members}) == 9
    assert len({fit_key(member) for member in members if member["seed"] == 0}) == 5
    assert fit_key(dict(members[0], weight=0.25)) == fit_key(members[0])
    assert all(member["epochs"] == 2 and member["pair_weighting"] == "ndcg" for member in members)
    assert [candidate["members"][0]["warmup_epochs"] for candidate in specs] == [10, 14, 20, 10, 10]


def test_fit_cache_separates_same_seed_training_specs_and_rejects_changed_sources(tmp_path):
    rng = np.random.RandomState(2)
    data = (rng.normal(size=(48, 12)).astype(np.float32), np.tile(np.arange(4), 12), np.repeat(np.arange(8), 6))
    context = dict(split="synthetic", source_sha256={"trainer": "frozen"})
    first_spec = specifications()[0]["members"][0]
    other_spec = specifications()[1]["members"][0]
    first = fit_member(data, first_spec, context, tmp_path)
    other = fit_member(data, other_spec, context, tmp_path)
    assert Path(first[2]["artifact"]).name == Path(other[2]["artifact"]).name == "seed_0.pt"
    assert Path(first[2]["artifact"]).parent != Path(other[2]["artifact"]).parent
    assert first[2]["state_sha256"] != other[2]["state_sha256"]
    replay = fit_member(data, first_spec, context, tmp_path)
    assert replay[2] == first[2]
    assert all(torch.equal(a, b) for a, b in zip(first[0].parameters(), replay[0].parameters()))
    with pytest.raises(ValueError, match="source"):
        fit_member(data, first_spec, dict(context, source_sha256={"trainer": "changed"}), tmp_path)


def test_control_checks_all_member_identities_histories_hashes_and_fold_metrics():
    reports, _, reference = reports_fixture()
    report = reports[0]
    proof = check_control(report["results"][0], reference, report["context"])
    assert proof["candidate_equal"] and proof["exported_metrics_equal"]
    assert all(member["artifact_sha256_checked"] for member in proof["member_checks"])
    mutations = [
        (lambda row: row.update(validation_ndcg_10=0.5), "fold metrics"),
        (lambda row: row["members"][0]["identity"]["specification"].update(seed=17), "specification"),
        (lambda row: row["members"][0]["identity"]["context"].update(train_indices_sha256="other"), "row identity"),
        (lambda row: row["members"][0]["history"][0].update(train_loss=0.4), "history"),
        (lambda row: row["members"][0].update(artifact_sha256="d" * 64), "artifact hash"),
    ]
    for mutate, error in mutations:
        damaged = copy.deepcopy(report["results"][0])
        mutate(damaged)
        with pytest.raises(ValueError, match=error):
            check_control(damaged, reference, report["context"])


def test_control_compares_archived_state_and_standardizer_when_available(tmp_path):
    reports, _, reference = reports_fixture()
    row = reports[0]["results"][0]
    for index, (current, archived) in enumerate(zip(row["members"], reference["folds"][0]["members"])):
        torch.manual_seed(index)
        network = ScoringNetwork(136, [128, 64])
        standardizer = Standardizer(np.zeros(136), np.ones(136))
        path = tmp_path / ("seed_%d.pt" % current["specification"]["seed"])
        save_scorer(str(path), network, standardizer)
        archived["artifact"] = str(path)
        current["artifact_sha256"] = archived["artifact_sha256"] = digest_file(path)
        current["state_sha256"] = state_fingerprint(network)
        current["standardizer_sha256"] = statistics_fingerprint(standardizer)
    proof = check_control(row, reference, reports[0]["context"])
    assert all(member["archived_state_checked"] for member in proof["member_checks"])
    row["members"][0]["state_sha256"] = "changed"
    with pytest.raises(ValueError, match="state or fitted statistics"):
        check_control(row, reference, reports[0]["context"])


def test_all_five_folds_and_fixed_mean_and_win_guards_select_exported_model():
    reports, specs, _ = reports_fixture()
    selected = select(reports, specs)
    assert selected["promoted"] and selected["selected_index"] == 4
    assert selected["selection_protocol"]["total_member_fits"] == 45
    assert not selected["selection_protocol"]["released_dev_used"]
    for report in reports:
        for index in range(1, 5):
            change_score(report, index, 0.4409)
    assert not select(reports, specs)["promoted"]
    # Highest mean but only two wins must preserve control, even if another
    # candidate could satisfy the gate: no adaptive fallback among candidates.
    for fold, report in enumerate(reports):
        change_score(report, 3, 0.442)
        change_score(report, 4, 0.46 if fold < 2 else 0.44)
    result = select(reports, specs)
    assert result["proposed_index"] == 4 and not result["promoted"]
    for fold, report in enumerate(reports):
        change_score(report, 4, 0.443 if fold < 3 else 0.44)
    # Candidate3 now has the larger mean; it wins all folds and promotes.
    assert select(reports, specs)["selected_index"] == 3


def test_selection_rejects_partial_changed_non_exported_and_unverified_controls():
    reports, specs, _ = reports_fixture()
    with pytest.raises(ValueError, match="five distinct"):
        select(reports[:-1], specs)
    mutations = [
        (lambda report: report["context"]["source_sha256"].update(trainer="changed"), "different TRAIN"),
        (lambda report: report["results"][1].update(validation_ndcg_10=0.9), "exported plain MLP"),
        (lambda report: report["results"][1]["members"][0].update(cache_key="wrong"), "full fit specification"),
        (lambda report: report["results"][1]["members"][0]["history"].pop(), "fixed-epoch"),
        (lambda report: report.update(distinct_member_fits=8), "five-candidate"),
        (lambda report: report["results"][4]["exported_config"].update(hidden=[512, 128]), "bounded"),
    ]
    for mutate, error in mutations:
        damaged = copy.deepcopy(reports)
        mutate(damaged[0])
        with pytest.raises(ValueError, match=error):
            select(damaged, specs)
    for report in reports:
        report["context"].update(rows=723412, queries=6000, features=136)
        for row in report["results"]:
            for member in row["members"]:
                member["identity"]["context"] = copy.deepcopy(report["context"])
    with pytest.raises(ValueError, match="archived v7 folds"):
        select(reports, specs)


def test_reference_requires_exact_candidate_and_matching_official_dataset(tmp_path):
    reports, specs, reference = reports_fixture()
    context = reports[0]["context"]
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(dict(context=context, selected=reference)))
    assert read_reference(path, context, specs) == reference
    reference["candidate"] = specs[1]
    path.write_text(json.dumps(dict(context=context, selected=reference)))
    with pytest.raises(ValueError, match="exact control"):
        read_reference(path, context, specs)
    with pytest.raises(ValueError, match="matching archived"):
        read_reference(tmp_path / "missing.json", dict(context, rows=723412, queries=6000, features=136), specs)


def test_single_wide_export_is_plain_staff_loadable_and_exact(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.normal(size=(31, 136)).astype(np.float32)
    standardizer = Standardizer().fit(X)
    network = ScoringNetwork(136, [256, 128])
    merged, fitted = combine_scorers([(network, standardizer)], [1.0])
    path = tmp_path / "pairwise_model.pt"
    save_scorer(str(path), merged, fitted)
    blob = torch.load(path, weights_only=True)
    assert set(blob) == {"config", "state_dict", "standardizer"}
    assert blob["config"] == {"n_features": 136, "hidden": [256, 128]}
    assert sum(parameter.numel() for parameter in merged.parameters()) == 68097
    np.testing.assert_array_equal(load_scorer(str(path))(X), make_scorer(network, standardizer)(X))


def test_no_promotion_copies_the_exact_baseline_without_loading_training_data(tmp_path):
    reports, specs, _ = reports_fixture()
    for report in reports:
        for index in range(1, 5):
            change_score(report, index, 0.44)
    selection = select(reports, specs)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"exact archived baseline bytes")
    output = tmp_path / "output.pt"
    args = argparse.Namespace(selection=str(path), specs=str(ROOT / "configs/v8_neural_candidates.json"),
                              baseline=str(baseline), out=str(output), history_out=str(tmp_path / "history.json"),
                              train="must-not-load-training-data", cache_dir=str(tmp_path / "cache"), fresh=True)
    run_fit(args)
    assert output.read_bytes() == baseline.read_bytes()
