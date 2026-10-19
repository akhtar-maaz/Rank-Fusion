"""Sixth-round fixed TRAIN comparisons using the unchanged promotion rule."""
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

CONTROL = "cached_v5_control"


def specifications(completed=range(4)):
    base = [("v3", 0, 800), ("v3", 1, 200), ("v4", 3, 800), ("v5", 1, 800)]
    weights = [0.125, 0.125, 0.25, 0.5]
    settings = [dict(name=CONTROL, sources=base, weights=weights)]
    for candidate in completed:
        for count in (200, 400, 800):
            new = ("v6", candidate, count)
            settings.append(dict(name=f"single_{candidate}_{count}", sources=[new], weights=[1.0]))
            for weight in (0.25, 0.5, 0.75):
                settings.append(dict(name=f"v5_plus_{candidate}_{count}_weight{weight}",
                                     sources=base+[new], weights=[value*(1-weight) for value in weights]+[weight]))
    return settings


def select(rows, baseline):
    return guarded_select(rows, baseline, control_name=CONTROL, control_key="v5_control")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    for version in ("v3", "v4", "v5", "v6"):
        parser.add_argument(f"--{version}-dir", default=f"runs/cv_{version}")
    parser.add_argument("--baseline", default="runs/baseline_v5/lambdamart_cv.json")
    parser.add_argument("--out", default="results/lambdamart_cv_v6.json")
    parser.add_argument("--excluded-candidates", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if len(set(args.excluded_candidates)) != len(args.excluded_candidates) or not set(args.excluded_candidates) <= set(range(4)):
        parser.error("exclusions must name distinct candidates 0..3")
    completed = sorted(set(range(4))-set(args.excluded_candidates))
    started = time.perf_counter()
    settings = specifications(completed)
    directories = {version: getattr(args, f"{version}_dir") for version in ("v3", "v4", "v5", "v6")}
    rows, context = measure(args.train, directories, settings)
    result = select(rows, json.loads(Path(args.baseline).read_text()))
    setting = next(setting for setting in settings if setting["name"] == result["selected"]["name"])
    result.update(arguments=vars(args), rows=context["rows"], queries=context["queries"], scenarios=SCENARIOS,
                  completed_candidates=completed, runtime_excluded_candidates=args.excluded_candidates,
                  native_equivalence=verify_native(setting, context), seconds=time.perf_counter()-started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(promoted=result["promoted"], selected=result["selected"])), flush=True)


if __name__ == "__main__":
    main()
