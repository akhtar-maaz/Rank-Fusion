"""Eighth-round fixed TRAIN comparisons, preserving seventh-round gates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ensemble_comparison import measure, verify_native
from scripts.robustness import SCENARIOS
from scripts.select_fifth_round import select as guarded_select
from scripts.select_seventh_round import fold_mixture

CONTROL = "cached_v7_control"


def specifications(completed=range(4)):
    base = [("v3", 0, 800), ("v3", 1, 200), ("v4", 3, 800),
            ("v5", 1, 800), ("v6", 3, 800), ("v7", 2, 400)]
    weights = [0.0703125, 0.0703125, 0.140625, 0.28125, 0.1875, 0.25]
    settings = [dict(name=CONTROL, sources=base, weights=weights)]
    for candidate in completed:
        for count in (200, 400, 600):
            new = ("v8", candidate, count)
            settings.append(dict(name=f"single_{candidate}_{count}", sources=[new], weights=[1.0]))
            for weight in (0.25, 0.5, 0.75):
                settings.append(dict(name=f"v7_plus_{candidate}_{count}_weight{weight}",
                                     sources=base + [new],
                                     weights=[value * (1 - weight) for value in weights] + [weight]))
    return settings


def select(rows, baseline):
    result = guarded_select(rows, baseline, control_name=CONTROL, control_key="v7_control")
    control = result["v7_control"]
    eligible = [row for row in rows if row["name"] in result["eligible_settings"]]
    winner = max(eligible, key=lambda row: (row["selection_score"], -row["trees"])) if eligible else control
    control_folds = {fold["fold"]: fold for fold in control["folds"]}
    gains = [{"fold": fold["fold"],
              "mixture_gain": fold_mixture(fold) - fold_mixture(control_folds[fold["fold"]])}
             for fold in sorted(winner["folds"], key=lambda fold: fold["fold"])]
    improved_folds = sum(fold["mixture_gain"] > 0 for fold in gains)
    diagnostic_gain = winner["diagnostic_mean"] - control["diagnostic_mean"]
    gates = dict(candidate=winner["name"],
                 mixture_gain=winner["selection_score"] - control["selection_score"],
                 diagnostic_gain=diagnostic_gain, improved_folds=improved_folds,
                 fold_mixture_gains=gains, mixture_passed=result["promoted"],
                 diagnostic_passed=diagnostic_gain >= 0.001,
                 paired_folds_passed=improved_folds >= 3)
    gates["passed"] = all(gates[key] for key in ("mixture_passed", "diagnostic_passed", "paired_folds_passed"))
    result.update(promotion_gates=gates, promoted=gates["passed"],
                  selected=winner if gates["passed"] else control)
    result["selection_protocol"].update(
        minimum_diagnostic_improvement=0.001, minimum_improved_folds=3,
        paired_fold_score="0.5 clean + 0.5 mean(affine, feature_dropout, noise)",
        additional_gate_failure="retain exact v7 control; do not search another setting")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    for version in ("v3", "v4", "v5", "v6", "v7", "v8"):
        parser.add_argument(f"--{version}-dir", default=f"runs/cv_{version}")
    parser.add_argument("--baseline", default="runs/baseline_v7_1/lambdamart_cv.json")
    parser.add_argument("--out", default="results/lambdamart_cv_v8.json")
    parser.add_argument("--excluded-candidates", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if len(set(args.excluded_candidates)) != len(args.excluded_candidates) or not set(args.excluded_candidates) <= set(range(4)):
        parser.error("exclusions must name distinct candidates 0..3")
    completed = sorted(set(range(4)) - set(args.excluded_candidates))
    started = time.perf_counter()
    settings = specifications(completed)
    directories = {version: getattr(args, f"{version}_dir") for version in ("v3", "v4", "v5", "v6", "v7", "v8")}
    rows, context = measure(args.train, directories, settings)
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
