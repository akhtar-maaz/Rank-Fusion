"""Promote a pairwise artifact using comparable internal TRAIN histories only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def choose_candidate(reports):
    reference = reports[0]["arguments"]
    scores = []
    for report in reports:
        args = report["arguments"]
        if args.get("dev") is not None:
            raise ValueError("candidate reports must not evaluate released DEV during selection")
        for key in ("train", "seed", "folds", "hidden"):
            if args[key] != reference[key]:
                raise ValueError(f"incomparable neural candidates: {key}")
        model = report["models"]["pairwise"]
        best = max(model["selection_history"], key=lambda row: row["dev_ndcg@10"])
        if model["best_epoch"] != best["epoch"] or len(model["full_train_history"]) != model["best_epoch"]:
            raise ValueError("candidate refit does not match its selected epoch")
        scores.append(best["dev_ndcg@10"])
    return max(range(len(reports)), key=lambda i: (scores[i], -reports[i]["models"]["pairwise"]["best_epoch"], -i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", nargs="+", required=True)
    parser.add_argument("--baseline-history", default="results/neural_history_v1.json")
    parser.add_argument("--baseline-model", default="runs/baseline_v1/pairwise_model.pt")
    parser.add_argument("--model-out", default="submission/pairwise_model.pt")
    parser.add_argument("--history-out", default="results/neural_history.json")
    parser.add_argument("--out", default="results/neural_selection_v2.json")
    args = parser.parse_args()
    paths = [Path(args.baseline_history)] + [Path(d) / "history.json" for d in args.candidate_dir]
    models = [Path(args.baseline_model)] + [Path(d) / "pairwise_model.pt" for d in args.candidate_dir]
    reports = [json.loads(path.read_text()) for path in paths]
    selected = choose_candidate(reports)
    import torch
    blob = torch.load(models[selected], weights_only=True)
    if blob["config"] != {"n_features": 136, "hidden": reports[0]["hidden"]}:
        raise ValueError("selected artifact must keep the pointwise network architecture")
    shutil.copy2(models[selected], args.model_out)
    combined = {key: reports[0][key] for key in ("seed", "trained_on", "selected_on", "hidden")}
    combined["model_arguments"] = {"pointwise": reports[0]["arguments"], "pairwise": reports[selected]["arguments"]}
    combined["models"] = {"pointwise": reports[0]["models"]["pointwise"],
                          "pairwise": reports[selected]["models"]["pairwise"]}
    Path(args.history_out).write_text(json.dumps(combined, indent=2) + "\n")
    saved = Path(args.out).parent / "neural_candidates"
    saved.mkdir(parents=True, exist_ok=True)
    for index, report in enumerate(reports[1:]):
        (saved / f"candidate_{index}.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [{"history": str(path), "model": str(model), "arguments": report["arguments"],
             "best_epoch": report["models"]["pairwise"]["best_epoch"],
             "TRAIN_selection_ndcg@10": max(row["dev_ndcg@10"] for row in report["models"]["pairwise"]["selection_history"])}
            for path, model, report in zip(paths, models, reports)]
    record = {"selected_on": "TRAIN internal fold 0 only", "candidates": rows,
              "selected_index": selected, "selected": rows[selected],
              "artifact_sha256": hashlib.sha256(Path(args.model_out).read_bytes()).hexdigest()}
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
