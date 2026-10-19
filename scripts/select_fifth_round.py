"""Fifth-round fixed comparisons with a per-diagnostic regression guard."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.cv_augmented_lambdamart import summarize
from scripts.ensemble_comparison import measure, verify_native
from scripts.robustness import SCENARIOS

CONTROL = "cached_aug800_weight0.5"


def specifications(completed=range(4)):
    base = [("v3", 0, 800), ("v3", 1, 200)]
    settings = []
    for count in (400, 800):
        for weight in (0.25, 0.5, 0.75):
            settings.append(dict(name=f"cached_aug{count}_weight{weight}", sources=base + [("v4", 3, count)],
                                 weights=[0.5 * (1 - weight), 0.5 * (1 - weight), weight]))
    for candidate in completed:
        for count in (200, 400, 800):
            new = ("v5", candidate, count)
            settings.append(dict(name=f"single_{candidate}_{count}", sources=[new], weights=[1.0]))
            for weight in (0.25, 0.5, 0.75):
                settings.append(dict(name=f"v4_plus_{candidate}_{count}_weight{weight}",
                                     sources=base + [("v4", 3, 800), new],
                                     weights=[0.25 * (1 - weight), 0.25 * (1 - weight), 0.5 * (1 - weight), weight]))
    return settings


def select(rows, baseline, *, control_name=CONTROL, control_key="v4_control"):
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("need distinct complete settings")
    for row in rows:
        if sorted(fold["fold"] for fold in row["folds"]) != list(range(5)):
            raise ValueError("every setting requires five distinct folds")
        if not all(np.isfinite(fold["train_ndcg@10"]) for fold in row["folds"]):
            raise ValueError("non-finite training metric")
        expected = summarize(row["name"], row.get("members", [row]), row["folds"])
        for key in ("train_mean", "validation_mean", "validation_std", "validation_train_ratio", "diagnostic_mean", "selection_score"):
            if not np.isfinite(row[key]) or abs(row[key] - expected[key]) > 1e-12:
                raise ValueError(f"inconsistent aggregate: {key}")
        if any(not np.isfinite(value) for fold in row["folds"] for value in fold["scenarios"].values()):
            raise ValueError("non-finite diagnostic")
        if row["scenario_means"] != expected["scenario_means"]:
            raise ValueError("inconsistent diagnostic means")
    controls = [row for row in rows if row["name"] == control_name]
    if len(controls) != 1:
        raise ValueError("missing exact baseline control")
    control = controls[0]
    old = baseline["selected"]
    if control.get("members") != old["members"]:
        raise ValueError("baseline control specifications changed")
    for key in ("train_mean", "validation_mean", "diagnostic_mean", "selection_score"):
        if abs(control[key] - old[key]) > 1e-12:
            raise ValueError("baseline control does not reproduce previous TRAIN CV")
    best_clean = max(row["validation_mean"] for row in rows)
    eligible = [row for row in rows if row["validation_mean"] >= best_clean - 0.0025 and
                all(row["scenario_means"][name] >= control["scenario_means"][name] - 0.0025 for name in SCENARIOS[1:])]
    winner = max(eligible, key=lambda row: (row["selection_score"], -row["trees"])) if eligible else control
    promoted = bool(eligible) and winner["selection_score"] >= control["selection_score"] + 0.0005
    return dict(results=rows, selected=winner if promoted else control, promoted=promoted,
                **{control_key: control}, best_clean_mean=best_clean, eligible_settings=[row["name"] for row in eligible],
                selection_protocol=dict(split="TRAIN only", folds=5, seed=0, clean_tolerance=0.0025,
                                        per_diagnostic_tolerance=0.0025, minimum_mixture_improvement=0.0005,
                                        score="0.5 clean + 0.5 mean(affine, feature_dropout, noise)",
                                        staff_c2_reconstruction=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--v3-dir", default="runs/cv_v3")
    parser.add_argument("--v4-dir", default="runs/cv_v4")
    parser.add_argument("--v5-dir", default="runs/cv_v5")
    parser.add_argument("--baseline", default="runs/baseline_v4/lambdamart_cv.json")
    parser.add_argument("--out", default="results/lambdamart_cv_v5.json")
    parser.add_argument("--excluded-candidates", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if len(set(args.excluded_candidates)) != len(args.excluded_candidates) or not set(args.excluded_candidates) <= set(range(4)):
        parser.error("exclusions must name distinct candidates 0..3")
    completed = sorted(set(range(4)) - set(args.excluded_candidates))
    started = time.perf_counter()
    settings = specifications(completed)
    rows, context = measure(args.train, dict(v3=args.v3_dir, v4=args.v4_dir, v5=args.v5_dir), settings)
    result = select(rows, json.loads(Path(args.baseline).read_text()))
    setting = next(setting for setting in settings if setting["name"] == result["selected"]["name"])
    result.update(arguments=vars(args), rows=context["rows"], queries=context["queries"], scenarios=SCENARIOS,
                  completed_candidates=completed, runtime_excluded_candidates=args.excluded_candidates,
                  native_equivalence=verify_native(setting, context), seconds=time.perf_counter() - started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(promoted=result["promoted"], selected=result["selected"])), flush=True)


if __name__ == "__main__":
    main()
