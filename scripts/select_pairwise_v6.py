"""Select among four fixed-epoch five-fold neural TRAIN reports."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CONTROL_FOLD0 = 0.44476016845475064


def select(reports, specifications):
    if sorted(report["arguments"]["candidate"] for report in reports) != list(range(4)):
        raise ValueError("all four distinct candidates must complete")
    reports = sorted(reports, key=lambda report: report["arguments"]["candidate"])
    for index, report in enumerate(reports):
        if report["candidate"] != specifications[index]:
            raise ValueError("candidate differs from frozen specification")
        if ((report["rows"], report["queries"]) != (reports[0]["rows"], reports[0]["queries"]) or
                report["arguments"]["train"] != reports[0]["arguments"]["train"]):
            raise ValueError("candidate TRAIN datasets differ")
        if sorted(fold["fold"] for fold in report["folds"]) != list(range(5)):
            raise ValueError("each candidate needs five distinct complete folds")
        for fold in report["folds"]:
            history = fold["history"]
            if ([row["epoch"] for row in history] != list(range(1, specifications[index]["epochs"]+1)) or
                    any("dev_ndcg@10" in row for row in history) or
                    abs(history[-1]["train_ndcg@10"]-fold["train_ndcg_10"]) > 1e-12):
                raise ValueError("reported fold must use final fixed-epoch weights")
        training = [fold["train_ndcg_10"] for fold in report["folds"]]
        validation = [fold["validation_ndcg_10"] for fold in report["folds"]]
        expected = dict(train_mean=np.mean(training), validation_mean=np.mean(validation),
                        validation_std=np.std(validation, ddof=1), validation_train_ratio=np.mean(validation)/np.mean(training))
        if any(not np.isfinite(value) for value in training+validation):
            raise ValueError("non-finite fold metric")
        if any(not np.isfinite(report[key]) or abs(report[key]-value) > 1e-12 for key, value in expected.items()):
            raise ValueError("inconsistent aggregate")
    control = reports[0]
    fold0 = next(fold for fold in control["folds"] if fold["fold"] == 0)
    if abs(fold0["validation_ndcg_10"]-CONTROL_FOLD0) > 1e-10:
        raise ValueError("control fold 0 did not reproduce prior TRAIN result")
    eligible = []
    for index, report in enumerate(reports[1:], 1):
        control_scores = {fold["fold"]: fold["validation_ndcg_10"] for fold in control["folds"]}
        wins = sum(fold["validation_ndcg_10"] > control_scores[fold["fold"]] for fold in report["folds"])
        if report["validation_mean"] >= control["validation_mean"]+0.001 and wins >= 3:
            eligible.append(index)
    winner = max(range(1, 4), key=lambda index: (reports[index]["validation_mean"],
                                               -specifications[index]["epochs"], -index))
    promoted = winner in eligible
    chosen = winner if promoted else 0
    return dict(promoted=promoted, selected_index=chosen, selected=reports[chosen],
                control=control, results=reports, eligible_candidates=eligible,
                selection_protocol=dict(split="TRAIN only", folds=5, seed=0, fixed_epochs=True,
                                        minimum_mean_gain=0.001, minimum_improved_folds=3,
                                        control_fold0_reference=CONTROL_FOLD0, released_dev_used=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", default="runs/neural_v6")
    parser.add_argument("--specs", default="configs/v6_neural_candidates.json")
    parser.add_argument("--out", default="results/neural_selection_v6.json")
    args = parser.parse_args()
    reports = [json.loads((Path(args.parts_dir)/f"part_{index}.json").read_text()) for index in range(4)]
    result = select(reports, json.loads(Path(args.specs).read_text()))
    result["arguments"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(promoted=result["promoted"], selected=result["selected"]["candidate"],
                          validation_mean=result["selected"]["validation_mean"])), flush=True)


if __name__ == "__main__":
    main()
