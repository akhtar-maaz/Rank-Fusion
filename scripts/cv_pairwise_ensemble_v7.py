"""Three frozen neural candidates, evaluated using query-grouped TRAIN only.

cv fits each of seeds 0, 17 and 41 once per fold, reusing seed 0 for both
fixed averages. select consumes all five complete fold reports. fit refits
only the already frozen winner, with no released DEV access in any mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from harness.cv import grouped_kfold
from harness.metrics import mean_ndcg_from_scores
from harness.neural import ScoringNetwork, load_scorer, make_scorer, save_scorer
from harness.preprocessing import Standardizer
from scripts.cv_pairwise_v6 import fit_arguments
from scripts.pairwise_ensemble import combine_scorers, same_standardizer
from submission.data import load_ltr
from submission.pointwise_pairwise import train_model

FIT_KEYS = ("hidden", "warmup_epochs", "epochs", "lr", "pairs_per_query",
            "pair_weighting", "batch_size", "seed")


def specifications():
    base = dict(hidden=[128, 64], warmup_epochs=10, epochs=2, lr=0.0001,
                pairs_per_query=300, pair_weighting="ndcg", batch_size=1024)
    return [dict(name=name, members=[dict(base, seed=seed, weight=1.0 / len(seeds)) for seed in seeds])
            for name, seeds in (("v6_pairwise_control", [0]),
                                ("pairwise_seeds_0_17", [0, 17]),
                                ("pairwise_seeds_0_41", [0, 41]))]


def read_specifications(path):
    specs = json.loads(Path(path).read_text())
    if specs != specifications():
        raise ValueError("candidate list differs from the frozen three-candidate plan")
    return specs


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_member(path):
    # Same class, keys and strict state loading as pristine staff loader.
    blob = torch.load(path, weights_only=True)
    network = ScoringNetwork(blob["config"]["n_features"], blob["config"]["hidden"])
    network.load_state_dict(blob["state_dict"], strict=True)
    network.eval()
    return network, Standardizer.from_dict(blob["standardizer"])


def validate_history(history, spec):
    if ([row["epoch"] for row in history] != list(range(1, spec["epochs"] + 1)) or
            any("dev_ndcg@10" in row for row in history) or
            any(not np.isfinite(row["train_ndcg@10"]) for row in history)):
        raise ValueError("member must contain all final fixed-epoch TRAIN history, without DEV")


def fit_member(data, spec, context, directory, fresh=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("seed_%d.pt" % spec["seed"])
    metadata_path = path.with_suffix(".json")
    identity = dict(context=context, specification={key: spec[key] for key in FIT_KEYS})
    if not fresh and path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity or metadata["artifact_sha256"] != digest_file(path):
            raise ValueError("cached member does not match the frozen TRAIN fit")
        network, standardizer = load_member(path)
        validate_history(metadata["history"], spec)
        return network, standardizer, metadata
    started = time.perf_counter()
    network, standardizer, history = train_model("pairwise", data, None, patience=0,
                                                verbose=True, **fit_arguments(spec))
    validate_history(history, spec)
    # Independently confirm fit statistics use exactly this training matrix.
    expected = Standardizer().fit(data[0])
    if not same_standardizer(standardizer, expected):
        raise ValueError("member standardizer differs from a deterministic fit on TRAIN rows")
    save_scorer(str(path), network, standardizer)
    metadata = dict(identity=identity, history=history, seconds=time.perf_counter()-started,
                    artifact_sha256=digest_file(path), artifact=str(path))
    write_json(metadata_path, metadata)
    return network, standardizer, metadata


def compare_export(network, standardizer, members, weights, data, path):
    """Score the exported staff artifact; expose float32 compilation error."""
    save_scorer(str(path), network, standardizer)
    exported = load_scorer(str(path))(data[0])
    direct = np.zeros(len(data[1]), dtype=np.float64)
    for (member, member_standardizer), weight in zip(members, weights):
        direct += float(weight) * make_scorer(member, member_standardizer)(data[0]).astype(np.float64)
    if not np.isfinite(exported).all() or not np.isfinite(direct).all():
        raise ValueError("compiled artifact produced non-finite scores")
    if not np.allclose(exported, direct, atol=2e-6, rtol=2e-5):
        raise ValueError("compiled artifact differs from arithmetic score average beyond float32 tolerance")
    return dict(exported_ndcg_10=mean_ndcg_from_scores(exported, data[1], data[2], k=10),
                arithmetic_average_ndcg_10=mean_ndcg_from_scores(direct, data[1], data[2], k=10),
                maximum_absolute_score_error=float(np.max(np.abs(exported - direct))),
                mean_absolute_score_error=float(np.mean(np.abs(exported - direct))),
                score_atol=2e-6, score_rtol=2e-5)


def evaluate_fold(data, fold, tr, va, specs, context, directory, reference=None, fresh=False):
    fit_data = tuple(array[tr] for array in data)
    validation_data = tuple(array[va] for array in data)
    fold_context = dict(context, fold=fold, train_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
                        validation_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest())
    members = {}
    for candidate in specs:
        for spec in candidate["members"]:
            if spec["seed"] not in members:
                members[spec["seed"]] = fit_member(fit_data, spec, fold_context,
                                                     Path(directory) / ("fold_%d" % fold), fresh)
    shared = next(iter(members.values()))[1]
    if not all(same_standardizer(shared, member[1]) for member in members.values()):
        raise ValueError("same-fold seed fits must have byte-identical standardizer statistics")
    rows = []
    for index, candidate in enumerate(specs):
        pairs = [(members[spec["seed"]][0], members[spec["seed"]][1]) for spec in candidate["members"]]
        weights = [spec["weight"] for spec in candidate["members"]]
        network, standardizer = combine_scorers(pairs, weights)
        export = Path(directory) / ("fold_%d" % fold) / ("candidate_%d.pt" % index)
        train_comparison = compare_export(network, standardizer, pairs, weights, fit_data, export)
        validation_comparison = compare_export(network, standardizer, pairs, weights, validation_data, export)
        row = dict(fold=fold, train_ndcg_10=train_comparison["exported_ndcg_10"],
                   validation_ndcg_10=validation_comparison["exported_ndcg_10"],
                   train_rows=len(tr), validation_rows=len(va),
                   train_queries=len(set(fit_data[2])), validation_queries=len(set(validation_data[2])),
                   standardizers_identical=True, exported_config=dict(n_features=data[0].shape[1],
                                                                     hidden=network.hidden_sizes),
                   exported_sha256=digest_file(export), train_compilation=train_comparison,
                   validation_compilation=validation_comparison,
                   members=[dict(specification=spec, **members[spec["seed"]][2]) for spec in candidate["members"]],
                   control_reference_checked=False)
        if index == 0 and reference is not None:
            expected = next(item for item in reference["folds"] if item["fold"] == fold)
            if any(abs(row[key]-expected[key]) > 1e-10 for key in ("train_ndcg_10", "validation_ndcg_10")):
                raise ValueError("v6 seed-0 control did not reproduce prior five-fold TRAIN scores")
            row["control_reference_checked"] = True
        rows.append(row)
        print(json.dumps(dict(candidate=candidate["name"], **row)), flush=True)
    return dict(fold=fold, context=fold_context, results=rows)


def run_cv(args):
    specs = read_specifications(args.specs)
    data = load_ltr(args.train)[0]
    context = dict(train_sha256=digest_file(args.train), rows=len(data[1]), queries=len(set(data[2])),
                   features=data[0].shape[1], folds=5, split_seed=0)
    reference = None
    if (context["rows"], context["queries"], context["features"]) == (723412, 6000, 136):
        previous = json.loads(Path(args.control_report).read_text())
        reference = previous["selected"]
        control_spec = {key: specs[0]["members"][0][key] for key in FIT_KEYS}
        if control_spec != {key: reference["candidate"][key] for key in FIT_KEYS}:
            raise ValueError("v6 control fit parameters differ")
        context["control_report_sha256"] = digest_file(args.control_report)
    started = time.perf_counter()
    folds = grouped_kfold(data[2], k=5, seed=0)
    for fold, (tr, va) in enumerate(folds):
        if args.fold is not None and fold != args.fold:
            continue
        report = evaluate_fold(data, fold, tr, va, specs, context, args.cache_dir, reference, args.fresh)
        report.update(arguments=vars(args), specifications=specs, seconds=time.perf_counter()-started)
        write_json(Path(args.out_dir) / ("part_fold%d.json" % fold), report)


def summarize(candidate, folds, context):
    train = [fold["train_ndcg_10"] for fold in folds]
    validation = [fold["validation_ndcg_10"] for fold in folds]
    return dict(candidate=candidate, folds=folds, rows=context["rows"], queries=context["queries"],
                train_mean=float(np.mean(train)), validation_mean=float(np.mean(validation)),
                validation_std=float(np.std(validation, ddof=1)),
                validation_train_ratio=float(np.mean(validation) / np.mean(train)))


def select(reports, specs):
    if specs != specifications():
        raise ValueError("candidate list differs from frozen specification")
    if sorted(report["fold"] for report in reports) != list(range(5)):
        raise ValueError("all five distinct complete TRAIN folds are required")
    reports = sorted(reports, key=lambda report: report["fold"])
    contexts = [{key: value for key, value in report["context"].items()
                 if key not in ("fold", "train_indices_sha256", "validation_indices_sha256")} for report in reports]
    if any(context != contexts[0] for context in contexts):
        raise ValueError("fold reports use different TRAIN datasets or protocols")
    if contexts[0]["folds"] != 5 or contexts[0]["split_seed"] != 0:
        raise ValueError("fold reports differ from the frozen five-fold seed-0 protocol")
    for report in reports:
        if report["specifications"] != specs or len(report["results"]) != len(specs):
            raise ValueError("incomplete or changed candidate specification")
        for index, row in enumerate(report["results"]):
            if row["fold"] != report["fold"] or not row["standardizers_identical"]:
                raise ValueError("inconsistent fold or standardizer proof")
            for key, comparison in (("train_ndcg_10", "train_compilation"),
                                    ("validation_ndcg_10", "validation_compilation")):
                if not np.isfinite(row[key]) or row[key] != row[comparison]["exported_ndcg_10"]:
                    raise ValueError("selection metric must come from the exported plain MLP")
            if len(row["members"]) != len(specs[index]["members"]):
                raise ValueError("incomplete candidate members")
            for member, spec in zip(row["members"], specs[index]["members"]):
                if member["specification"] != spec:
                    raise ValueError("member differs from frozen specification")
                validate_history(member["history"], spec)
        if ((contexts[0]["rows"], contexts[0]["queries"], contexts[0]["features"]) == (723412, 6000, 136)
                and not report["results"][0]["control_reference_checked"]):
            raise ValueError("official TRAIN control must reproduce all prior fold scores")
    results = [summarize(candidate, [report["results"][index] for report in reports], contexts[0])
               for index, candidate in enumerate(specs)]
    control = results[0]
    winner = max(range(1, len(specs)), key=lambda index: (results[index]["validation_mean"], -index))
    wins = sum(current["validation_ndcg_10"] > baseline["validation_ndcg_10"]
               for current, baseline in zip(results[winner]["folds"], control["folds"]))
    gain = results[winner]["validation_mean"] - control["validation_mean"]
    promoted = gain >= 0.001 and wins >= 3
    chosen = winner if promoted else 0
    return dict(promoted=promoted, selected_index=chosen, selected=results[chosen], control=control,
                results=results, context=contexts[0], proposed_index=winner, proposed_mean_gain=gain,
                proposed_improved_folds=wins,
                selection_protocol=dict(split="TRAIN only", folds=5, seed=0, fixed_epochs=True,
                                        candidate_count=3, distinct_seed_fits_per_fold=3,
                                        minimum_mean_gain=0.001, minimum_improved_folds=3,
                                        released_dev_used=False, weights=[0.5, 0.5],
                                        selection_metric="exported plain-MLP nDCG@10"))


def run_select(args):
    specs = read_specifications(args.specs)
    reports = [json.loads((Path(args.out_dir) / ("part_fold%d.json" % fold)).read_text()) for fold in range(5)]
    result = select(reports, specs)
    result["arguments"] = vars(args)
    write_json(args.selection, result)
    print(json.dumps(dict(promoted=result["promoted"], candidate=result["selected"]["candidate"],
                          validation_mean=result["selected"]["validation_mean"])), flush=True)


def run_fit(args):
    selection = json.loads(Path(args.selection).read_text())
    specs = read_specifications(args.specs)
    candidate = selection["selected"]["candidate"]
    if candidate not in specs or selection["selection_protocol"]["released_dev_used"]:
        raise ValueError("full fit requires a frozen TRAIN-only selection")
    started = time.perf_counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = dict(candidate=candidate, promoted=selection["promoted"], arguments=vars(args))
    if not selection["promoted"]:
        shutil.copyfile(args.baseline, args.out)
        report["trained_on"] = "unchanged v6 TRAIN artifact"
        report["members"] = []
    else:
        data = load_ltr(args.train)[0]
        context = dict(train_sha256=digest_file(args.train), rows=len(data[1]), queries=len(set(data[2])),
                       features=data[0].shape[1], split="full TRAIN")
        if context["train_sha256"] != selection["context"]["train_sha256"]:
            raise ValueError("full fit TRAIN input differs from the selected CV data")
        fitted = [fit_member(data, spec, context, Path(args.cache_dir) / "full", args.fresh)
                  for spec in candidate["members"]]
        if not all(same_standardizer(fitted[0][1], member[1]) for member in fitted):
            raise ValueError("full-TRAIN seed standardizers differ")
        pairs = [(member[0], member[1]) for member in fitted]
        weights = [spec["weight"] for spec in candidate["members"]]
        network, standardizer = combine_scorers(pairs, weights)
        report.update(trained_on="TRAIN only", standardizers_identical=True,
                      members=[dict(specification=spec, **member[2]) for spec, member in zip(candidate["members"], fitted)],
                      exported_config=dict(n_features=data[0].shape[1], hidden=network.hidden_sizes),
                      compilation=compare_export(network, standardizer, pairs, weights, data, args.out))
    report.update(artifact_sha256=digest_file(args.out), seconds=time.perf_counter()-started)
    write_json(args.history_out, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cv", "select", "fit"), default="cv")
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--specs", default="configs/v7_neural_candidates.json")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--cache-dir", default="runs/neural_v7/cache")
    parser.add_argument("--out-dir", default="runs/neural_v7")
    parser.add_argument("--control-report", default="results/neural_selection_v6.json")
    parser.add_argument("--selection", default="results/neural_selection_v7.json")
    parser.add_argument("--baseline", default="runs/baseline_v6/pairwise_model.pt")
    parser.add_argument("--out", default="submission/pairwise_model.pt")
    parser.add_argument("--history-out", default="results/pairwise_refit_v7.json")
    parser.add_argument("--fresh", action="store_true", help="refit members even when a valid cache exists")
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"cv": run_cv, "select": run_select, "fit": run_fit}[args.mode](args)


if __name__ == "__main__":
    main()
