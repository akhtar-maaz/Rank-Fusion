"""Fixed-epoch five-fold TRAIN assessment; no released DEV is loaded."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from harness.cv import grouped_kfold
from submission.data import load_ltr
from submission.pointwise_pairwise import dev_ndcg, make_scorer, train_model


def fit_arguments(specification):
    return {key: specification[key] for key in ("hidden", "warmup_epochs", "epochs", "lr", "pairs_per_query",
                                                 "pair_weighting", "batch_size", "seed")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--candidate", type=int, choices=range(4), required=True)
    parser.add_argument("--specs", default="configs/v6_neural_candidates.json")
    parser.add_argument("--out-dir", default="runs/neural_v6")
    args = parser.parse_args()
    torch.set_num_threads(4)
    specifications = json.loads(Path(args.specs).read_text())
    chosen = specifications[args.candidate]
    data = load_ltr(args.train)[0]
    started = time.perf_counter()
    entries = []
    for fold, (tr, va) in enumerate(grouped_kfold(data[2], k=5, seed=0)):
        fit_data = tuple(array[tr] for array in data)
        validation_data = tuple(array[va] for array in data)
        # None forces a fixed-epoch fit: validation cannot influence weights.
        net, standardizer, history = train_model("pairwise", fit_data, None, patience=0,
                                                 verbose=True, **fit_arguments(chosen))
        scorer = make_scorer(net, standardizer)
        entry = dict(fold=fold, train_ndcg_10=history[-1]["train_ndcg@10"],
                     validation_ndcg_10=dev_ndcg(scorer, *validation_data), history=history)
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    train_mean = float(np.mean([entry["train_ndcg_10"] for entry in entries]))
    mean = float(np.mean([entry["validation_ndcg_10"] for entry in entries]))
    report = dict(arguments=vars(args), candidate=chosen, rows=len(data[1]), queries=len(set(data[2])),
                  folds=entries, train_mean=train_mean, validation_mean=mean,
                  validation_std=float(np.std([entry["validation_ndcg_10"] for entry in entries], ddof=1)),
                  validation_train_ratio=mean / train_mean, seconds=time.perf_counter()-started)
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/f"part_{args.candidate}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
