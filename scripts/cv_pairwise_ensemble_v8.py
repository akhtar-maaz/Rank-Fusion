"""Five frozen bounded neural candidates; all assessment and fitting use TRAIN.

Each fold freshly fits nine distinct specifications: the two-member v7
control, two longer-warmup averages, a higher-pair-cap average, and one
unstructured wider network. Caches use complete fit specifications and
source fingerprints, never seeds alone. No mode accepts a DEV input.
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
from harness.neural import save_scorer
from harness.preprocessing import Standardizer
from scripts.cv_pairwise_ensemble_v7 import (FIT_KEYS, compare_export, digest_file, load_member,
                                            summarize, validate_history, write_json)
from scripts.cv_pairwise_v6 import fit_arguments
from scripts.pairwise_ensemble import combine_scorers, same_standardizer
from submission.data import load_ltr
from submission.pointwise_pairwise import train_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("submission/pointwise_pairwise.py", "submission/data.py",
                "scripts/cv_pairwise_ensemble_v8.py", "scripts/cv_pairwise_ensemble_v7.py",
                "scripts/cv_pairwise_v6.py", "scripts/pairwise_ensemble.py", "harness/neural.py",
                "harness/preprocessing.py", "harness/cv.py", "harness/metrics.py")
OFFICIAL_SHAPE = (723412, 6000, 136)


def specifications():
    base = dict(hidden=[128, 64], warmup_epochs=10, epochs=2, lr=0.0001,
                pairs_per_query=300, pair_weighting="ndcg", batch_size=1024)
    candidates = []
    for name, warmup, pairs, hidden, seeds in (
            ("pairwise_seeds_0_41", 10, 300, [128, 64], [0, 41]),
            ("warmup14_seeds_0_41", 14, 300, [128, 64], [0, 41]),
            ("warmup20_seeds_0_41", 20, 300, [128, 64], [0, 41]),
            ("pairs1000_seeds_0_41", 10, 1000, [128, 64], [0, 41]),
            ("direct_wide_seed0", 10, 300, [256, 128], [0])):
        members = [dict(base, hidden=hidden, warmup_epochs=warmup, pairs_per_query=pairs,
                        seed=seed, weight=1.0 / len(seeds)) for seed in seeds]
        candidates.append(dict(name=name, members=members))
    return candidates


def read_specifications(path):
    specs = json.loads(Path(path).read_text())
    if specs != specifications():
        raise ValueError("candidate list differs from the frozen five-candidate plan")
    return specs


def fit_specification(spec):
    return {key: spec[key] for key in FIT_KEYS}


def fit_key(spec):
    """All training choices, including seed, width, warmup and cap, identify a fit."""
    encoded = json.dumps(fit_specification(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_fingerprints():
    return {path: digest_file(ROOT / path) for path in SOURCE_PATHS}


def runtime():
    return dict(torch=str(torch.__version__), numpy=np.__version__, cpu_threads=4,
                deterministic_algorithms=True)


def state_fingerprint(network):
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def statistics_fingerprint(standardizer):
    encoded = json.dumps(standardizer.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def fit_member(data, spec, context, directory, fresh=False):
    key = fit_key(spec)
    destination = Path(directory) / key
    destination.mkdir(parents=True, exist_ok=True)
    # Keep v7 member basenames, permitting meaningful byte-hash comparison.
    path = destination / ("seed_%d.pt" % spec["seed"])
    metadata_path = path.with_suffix(".json")
    identity = dict(context=context, specification=fit_specification(spec))
    if not fresh and path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata["identity"] != identity or metadata["cache_key"] != key or
                metadata["artifact_sha256"] != digest_file(path)):
            raise ValueError("cached member differs from the frozen TRAIN fit or source")
        network, standardizer = load_member(path)
        if (metadata["state_sha256"] != state_fingerprint(network) or
                metadata["standardizer_sha256"] != statistics_fingerprint(standardizer)):
            raise ValueError("cached state or standardizer identity differs")
        validate_history(metadata["history"], spec)
        return network, standardizer, metadata
    started = time.perf_counter()
    network, standardizer, history = train_model("pairwise", data, None, patience=0,
                                                verbose=True, **fit_arguments(spec))
    validate_history(history, spec)
    if not same_standardizer(standardizer, Standardizer().fit(data[0])):
        raise ValueError("member standardizer differs from the deterministic TRAIN-row fit")
    save_scorer(str(path), network, standardizer)
    metadata = dict(identity=identity, cache_key=key, history=history,
                    seconds=time.perf_counter()-started, artifact=str(path),
                    artifact_sha256=digest_file(path), state_sha256=state_fingerprint(network),
                    standardizer_sha256=statistics_fingerprint(standardizer))
    write_json(metadata_path, metadata)
    return network, standardizer, metadata


def is_official(context):
    return tuple(context[key] for key in ("rows", "queries", "features")) == OFFICIAL_SHAPE


def read_reference(path, context, specs):
    path = Path(path)
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is None or previous.get("context", {}).get("train_sha256") != context["train_sha256"]:
        if is_official(context):
            raise ValueError("official TRAIN needs the matching archived v7 control reference")
        return None
    reference = previous["selected"]
    if reference["candidate"] != specs[0]:
        raise ValueError("v7 control candidate differs from the frozen exact control")
    for key in ("rows", "queries", "features", "folds", "split_seed"):
        if previous["context"][key] != context[key]:
            raise ValueError("v7 control reference uses a different TRAIN protocol")
    if sorted(row["fold"] for row in reference["folds"]) != list(range(5)):
        raise ValueError("v7 control reference needs all five folds")
    return reference


def check_control(row, reference, context):
    """Compare freshly fitted control to every archived v7 fold/member."""
    if reference["candidate"] != specifications()[0]:
        raise ValueError("v7 control candidate identity differs")
    expected = next(item for item in reference["folds"] if item["fold"] == row["fold"])
    for key in ("train_ndcg_10", "validation_ndcg_10"):
        if not np.isfinite(expected[key]) or abs(row[key]-expected[key]) > 1e-10:
            raise ValueError("v7 control did not reproduce archived exported fold metrics")
    if len(row["members"]) != 2 or len(expected["members"]) != 2:
        raise ValueError("v7 control must contain both original members")
    checks = []
    for current, archived, specification in zip(row["members"], expected["members"], specifications()[0]["members"]):
        if (current["specification"] != specification or archived["specification"] != specification or
                current["identity"]["specification"] != fit_specification(specification) or
                archived["identity"]["specification"] != fit_specification(specification)):
            raise ValueError("v7 control member specification identity differs")
        archived_context = archived["identity"]["context"]
        for key in ("train_sha256", "rows", "queries", "features", "folds", "split_seed", "fold",
                    "train_indices_sha256", "validation_indices_sha256"):
            if archived_context[key] != context[key] or current["identity"]["context"][key] != context[key]:
                raise ValueError("v7 control member TRAIN row identity differs")
        validate_history(current["history"], specification)
        validate_history(archived["history"], specification)
        if len(current["history"]) != len(archived["history"]):
            raise ValueError("v7 control member history differs")
        for actual, prior in zip(current["history"], archived["history"]):
            if (set(actual) != set(prior) or
                    any(not np.isfinite(prior[key]) or abs(actual[key]-prior[key]) > 1e-10 for key in prior)):
                raise ValueError("v7 control member fixed-epoch history differs")
        current_path, archived_path = Path(current["artifact"]), Path(archived["artifact"])
        hash_checked = current_path.name == archived_path.name
        if hash_checked and current["artifact_sha256"] != archived["artifact_sha256"]:
            raise ValueError("v7 control member artifact hash differs despite the same basename")
        state_checked = False
        if archived_path.is_file():
            if digest_file(archived_path) != archived["artifact_sha256"]:
                raise ValueError("archived v7 control member artifact is corrupt")
            old_network, old_standardizer = load_member(archived_path)
            if (current["state_sha256"] != state_fingerprint(old_network) or
                    current["standardizer_sha256"] != statistics_fingerprint(old_standardizer)):
                raise ValueError("v7 control member state or fitted statistics differ")
            state_checked = True
        checks.append(dict(seed=specification["seed"], history_equal=True,
                           same_artifact_basename=hash_checked, artifact_sha256_checked=hash_checked,
                           artifact_sha256_equal=True if hash_checked else None,
                           archived_state_checked=state_checked,
                           state_and_statistics_equal=True if state_checked else None))
    return dict(exported_metric_tolerance=1e-10, exported_metrics_equal=True,
                candidate_equal=True, member_checks=checks)


def evaluate_fold(data, fold, tr, va, specs, context, directory, reference=None, fresh=False):
    fit_data = tuple(array[tr] for array in data)
    validation_data = tuple(array[va] for array in data)
    fold_context = dict(context, fold=fold, train_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
                        validation_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest())
    members = {}
    for candidate in specs:
        for spec in candidate["members"]:
            key = fit_key(spec)
            if key not in members:
                members[key] = fit_member(fit_data, spec, fold_context,
                                           Path(directory) / ("fold_%d" % fold), fresh)
    if len(members) != 9:
        raise ValueError("frozen plan must fit exactly nine distinct members per fold")
    shared = next(iter(members.values()))[1]
    if not all(same_standardizer(shared, member[1]) for member in members.values()):
        raise ValueError("all same-fold specifications must share identical TRAIN statistics")
    rows = []
    for index, candidate in enumerate(specs):
        fitted = [members[fit_key(spec)] for spec in candidate["members"]]
        pairs = [(member[0], member[1]) for member in fitted]
        weights = [spec["weight"] for spec in candidate["members"]]
        network, standardizer = combine_scorers(pairs, weights)
        export = Path(directory) / ("fold_%d" % fold) / ("candidate_%d.pt" % index)
        training = compare_export(network, standardizer, pairs, weights, fit_data, export)
        validation = compare_export(network, standardizer, pairs, weights, validation_data, export)
        row = dict(fold=fold, train_ndcg_10=training["exported_ndcg_10"],
                   validation_ndcg_10=validation["exported_ndcg_10"],
                   train_rows=len(tr), validation_rows=len(va),
                   train_queries=len(set(fit_data[2])), validation_queries=len(set(validation_data[2])),
                   standardizers_identical=True,
                   exported_config=dict(n_features=data[0].shape[1], hidden=network.hidden_sizes),
                   exported_sha256=digest_file(export), train_compilation=training,
                   validation_compilation=validation,
                   members=[dict(specification=spec, **member[2]) for spec, member in zip(candidate["members"], fitted)],
                   control_reference_checked=False)
        if index == 0 and reference is not None:
            row["control_reference"] = check_control(row, reference, fold_context)
            row["control_reference_checked"] = True
        rows.append(row)
        print(json.dumps(dict(candidate=candidate["name"], fold=fold, train_ndcg_10=row["train_ndcg_10"],
                              validation_ndcg_10=row["validation_ndcg_10"],
                              control_reference_checked=row["control_reference_checked"])), flush=True)
    return dict(fold=fold, context=fold_context, results=rows, distinct_member_fits=9)


def run_cv(args):
    specs = read_specifications(args.specs)
    data = load_ltr(args.train)[0]
    context = dict(train_sha256=digest_file(args.train), rows=len(data[1]), queries=len(set(data[2])),
                   features=data[0].shape[1], folds=5, split_seed=0,
                   source_sha256=source_fingerprints(), runtime=runtime())
    reference = read_reference(args.control_report, context, specs)
    if reference is not None:
        context["control_report_sha256"] = digest_file(args.control_report)
    started = time.perf_counter()
    for fold, (tr, va) in enumerate(grouped_kfold(data[2], k=5, seed=0)):
        if args.fold is not None and fold != args.fold:
            continue
        report = evaluate_fold(data, fold, tr, va, specs, context, args.cache_dir, reference, args.fresh)
        report.update(arguments=vars(args), specifications=specs, seconds=time.perf_counter()-started)
        write_json(Path(args.out_dir) / ("part_fold%d.json" % fold), report)


def validate_row(row, spec, fold):
    if row["fold"] != fold or not row["standardizers_identical"]:
        raise ValueError("inconsistent fold or TRAIN standardizer proof")
    for key, comparison in (("train_ndcg_10", "train_compilation"),
                            ("validation_ndcg_10", "validation_compilation")):
        if not np.isfinite(row[key]) or row[key] != row[comparison]["exported_ndcg_10"]:
            raise ValueError("selection metric must come from the exported plain MLP")
    if len(row["members"]) != len(spec["members"]):
        raise ValueError("candidate is missing members")
    for member, expected in zip(row["members"], spec["members"]):
        if (member["specification"] != expected or member["cache_key"] != fit_key(expected) or
                member["identity"]["specification"] != fit_specification(expected)):
            raise ValueError("member differs from the frozen full fit specification")
        validate_history(member["history"], expected)
    widths = [sum(member["hidden"][layer] for member in spec["members"]) for layer in range(2)]
    if row["exported_config"]["hidden"] != widths or max(widths) > 256:
        raise ValueError("exported network differs from the bounded plain-MLP candidate")


def select(reports, specs, reference=None):
    if specs != specifications():
        raise ValueError("candidate list differs from frozen specification")
    if sorted(report["fold"] for report in reports) != list(range(5)):
        raise ValueError("all five distinct complete TRAIN folds are required")
    reports = sorted(reports, key=lambda report: report["fold"])
    contexts = [{key: value for key, value in report["context"].items()
                 if key not in ("fold", "train_indices_sha256", "validation_indices_sha256")} for report in reports]
    if any(context != contexts[0] for context in contexts):
        raise ValueError("folds use different TRAIN inputs, source fingerprints or protocols")
    if contexts[0]["folds"] != 5 or contexts[0]["split_seed"] != 0:
        raise ValueError("folds differ from the frozen five-fold seed-0 protocol")
    for report in reports:
        if (report["specifications"] != specs or len(report["results"]) != 5 or
                report["distinct_member_fits"] != 9):
            raise ValueError("incomplete or changed five-candidate plan")
        for row, spec in zip(report["results"], specs):
            validate_row(row, spec, report["fold"])
            for member in row["members"]:
                if member["identity"]["context"] != report["context"]:
                    raise ValueError("member cache uses different TRAIN rows or source fingerprints")
        control = report["results"][0]
        if reference is not None:
            check_control(control, reference, report["context"])
        if is_official(contexts[0]) and not control["control_reference_checked"]:
            raise ValueError("official TRAIN control must reproduce the archived v7 folds")
        if control["control_reference_checked"]:
            proof = control.get("control_reference", {})
            if not proof.get("candidate_equal") or not proof.get("exported_metrics_equal"):
                raise ValueError("missing exact v7 control proof")
    results = [summarize(candidate, [report["results"][index] for report in reports], contexts[0])
               for index, candidate in enumerate(specs)]
    control = results[0]
    winner = max(range(1, 5), key=lambda index: (results[index]["validation_mean"], -index))
    wins = sum(current["validation_ndcg_10"] > baseline["validation_ndcg_10"]
               for current, baseline in zip(results[winner]["folds"], control["folds"]))
    gain = results[winner]["validation_mean"] - control["validation_mean"]
    promoted = gain >= 0.001 and wins >= 3
    chosen = winner if promoted else 0
    return dict(promoted=promoted, selected_index=chosen, selected=results[chosen], control=control,
                results=results, context=contexts[0], proposed_index=winner, proposed_mean_gain=gain,
                proposed_improved_folds=wins,
                selection_protocol=dict(split="TRAIN only", folds=5, seed=0, fixed_epochs=True,
                                        candidate_count=5, new_candidate_count=4,
                                        distinct_member_fits_per_fold=9, total_member_fits=45,
                                        minimum_mean_gain=0.001, minimum_improved_folds=3,
                                        released_dev_used=False, control_metric_tolerance=1e-10,
                                        selection_metric="exported plain-MLP nDCG@10"))


def run_select(args):
    specs = read_specifications(args.specs)
    reports = [json.loads((Path(args.out_dir) / ("part_fold%d.json" % fold)).read_text()) for fold in range(5)]
    context = reports[0]["context"]
    if context["source_sha256"] != source_fingerprints() or context["runtime"] != runtime():
        raise ValueError("selection source or runtime differs from frozen CV fits")
    reference = read_reference(args.control_report, context, specs)
    result = select(reports, specs, reference)
    result["arguments"] = vars(args)
    write_json(args.selection, result)
    print(json.dumps(dict(promoted=result["promoted"], candidate=result["selected"]["candidate"],
                          validation_mean=result["selected"]["validation_mean"])), flush=True)


def run_fit(args):
    selection = json.loads(Path(args.selection).read_text())
    specs = read_specifications(args.specs)
    index = selection["selected_index"]
    if (index not in range(5) or selection["selected"]["candidate"] != specs[index] or
            selection["selection_protocol"]["released_dev_used"] or
            bool(selection["promoted"]) != (index != 0)):
        raise ValueError("full fit requires a frozen TRAIN-only selection")
    if (selection["context"]["source_sha256"] != source_fingerprints() or
            selection["context"]["runtime"] != runtime()):
        raise ValueError("full fit source or runtime differs from the selected CV fits")
    candidate = specs[index]
    started = time.perf_counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = dict(candidate=candidate, promoted=selection["promoted"], arguments=vars(args),
                  source_sha256=source_fingerprints(), runtime=runtime())
    if not selection["promoted"]:
        shutil.copyfile(args.baseline, args.out)
        report.update(trained_on="unchanged v7.1 TRAIN artifact", members=[],
                      baseline_sha256=digest_file(args.baseline))
    else:
        data = load_ltr(args.train)[0]
        context = dict(train_sha256=digest_file(args.train), rows=len(data[1]), queries=len(set(data[2])),
                       features=data[0].shape[1], split="full TRAIN",
                       source_sha256=source_fingerprints(), runtime=runtime())
        if any(context[key] != selection["context"][key] for key in ("train_sha256", "rows", "queries", "features")):
            raise ValueError("full fit TRAIN input differs from the selected CV data")
        fitted = [fit_member(data, spec, context, Path(args.cache_dir) / "full", args.fresh)
                  for spec in candidate["members"]]
        if not all(same_standardizer(fitted[0][1], member[1]) for member in fitted):
            raise ValueError("full-TRAIN member standardizers differ")
        pairs = [(member[0], member[1]) for member in fitted]
        weights = [spec["weight"] for spec in candidate["members"]]
        network, standardizer = combine_scorers(pairs, weights)
        report.update(trained_on="TRAIN only", standardizers_identical=True,
                      members=[dict(specification=spec, **member[2]) for spec, member in zip(candidate["members"], fitted)],
                      exported_config=dict(n_features=data[0].shape[1], hidden=network.hidden_sizes),
                      compilation=compare_export(network, standardizer, pairs, weights, data, args.out))
    report.update(artifact_sha256=digest_file(args.out), seconds=time.perf_counter()-started)
    write_json(args.history_out, report)
    print(json.dumps(dict(promoted=report["promoted"], artifact_sha256=report["artifact_sha256"],
                          seconds=report["seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cv", "select", "fit"), default="cv")
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--specs", default="configs/v8_neural_candidates.json")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--cache-dir", default="runs/neural_v8/cache")
    parser.add_argument("--out-dir", default="runs/neural_v8")
    parser.add_argument("--control-report", default="results/neural_selection_v7.json")
    parser.add_argument("--selection", default="results/neural_selection_v8.json")
    parser.add_argument("--baseline", default="runs/baseline_v7_1/pairwise_model.pt")
    parser.add_argument("--out", default="submission/pairwise_model.pt")
    parser.add_argument("--history-out", default="results/pairwise_refit_v8.json")
    parser.add_argument("--fresh", action="store_true", help="refit even if a valid v8 full-spec cache exists")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    {"cv": run_cv, "select": run_select, "fit": run_fit}[args.mode](args)


if __name__ == "__main__":
    main()
