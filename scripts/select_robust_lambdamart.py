"""Apply the documented round-three promotion rule to complete TRAIN CV."""
import argparse
import json
from pathlib import Path

import numpy as np

SCENARIOS = ("clean", "affine", "feature_dropout", "noise")


def select(reports, baseline):
    if sorted(report["arguments"]["candidate"] for report in reports) != [0, 1, 2, 3]:
        raise ValueError("all four distinct candidates must complete")
    rows = []
    for report in reports:
        if (report["rows"], report["queries"]) != (baseline["rows"], baseline["queries"]):
            raise ValueError("candidate and baseline dataset sizes differ")
        if tuple(report["scenarios"]) != SCENARIOS:
            raise ValueError("diagnostic scenarios differ")
        if report["arguments"]["train"] != reports[0]["arguments"]["train"]:
            raise ValueError("TRAIN paths differ")
        for row in report["results"]:
            if sorted(fold["fold"] for fold in row["folds"]) != list(range(5)):
                raise ValueError("every result needs all five distinct folds")
            means = {scenario: float(np.mean([fold["scenarios"][scenario] for fold in row["folds"]]))
                     for scenario in SCENARIOS}
            diagnostic = float(np.mean([means[name] for name in SCENARIOS[1:]]))
            score = 0.5 * (means["clean"] + diagnostic)
            for key, expected in (("validation_mean", means["clean"]), ("diagnostic_mean", diagnostic),
                                  ("selection_score", score)):
                if not np.isfinite(row[key]) or abs(row[key] - expected) > 1e-12:
                    raise ValueError(f"inconsistent aggregate: {key}")
            rows.append(row)
    controls = [row for row in rows if row["name"] == "v2_control" and
                row["trees"] == baseline["selected"]["trees"]]
    if len(controls) != 1:
        raise ValueError("missing unique control at the v2 selected tree count")
    control = controls[0]
    if (control["params"] != baseline["selected"]["params"] or
            control["feature_mode"] != baseline["selected"].get("feature_mode", "raw") or
            abs(control["validation_mean"] - baseline["selected"]["validation_mean"]) > 1e-12):
        raise ValueError("the v2 control does not reproduce the prior TRAIN-CV result")
    best_clean = max(row["validation_mean"] for row in rows)
    eligible = [row for row in rows if row["validation_mean"] >= best_clean - 0.0025]
    winner = max(eligible, key=lambda row: (row["selection_score"], -row["trees"], -row["params"]["num_leaves"]))
    promoted = winner["selection_score"] >= control["selection_score"] + 0.001
    return {"rows": baseline["rows"], "queries": baseline["queries"], "results": rows,
            "selected": winner if promoted else control, "promoted": promoted,
            "selection_protocol": {"split": "TRAIN only", "folds": 5, "seed": 0,
                                   "clean_tolerance": 0.0025, "minimum_mixture_improvement": 0.001,
                                   "score": "0.5 clean + 0.5 mean(affine, feature_dropout, noise)",
                                   "staff_c2_reconstruction": False},
            "best_clean_mean": best_clean, "v2_control": control}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+")
    parser.add_argument("--baseline", default="results/lambdamart_cv_v2.json")
    parser.add_argument("--out", default="results/lambdamart_cv.json")
    args = parser.parse_args()
    result = select([json.loads(Path(path).read_text()) for path in args.parts],
                    json.loads(Path(args.baseline).read_text()))
    result["parts"] = args.parts
    result["baseline"] = args.baseline
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"promoted": result["promoted"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
