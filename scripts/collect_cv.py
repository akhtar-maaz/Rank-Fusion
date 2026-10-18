"""Collect disjoint CV configurations, selecting on TRAIN folds only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+")
    parser.add_argument("--out", default="runs/lambdamart_cv.json")
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.parts]
    for report in reports[1:]:
        for key in ("seed", "folds", "train"):
            if report["arguments"][key] != reports[0]["arguments"][key]:
                raise ValueError(f"incompatible CV parts: {key}")
    rows = [row for report in reports for row in report["results"]]
    result = {"parts": args.parts, "results": rows,
              "selected": max(rows, key=lambda r: (r["validation_mean"], -r["trees"], -r["params"]["num_leaves"]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
