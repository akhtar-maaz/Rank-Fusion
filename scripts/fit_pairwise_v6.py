"""Refit a frozen neural TRAIN choice, or preserve the v5 artifact exactly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.cv_pairwise_v6 import fit_arguments
from submission.data import load_ltr
from submission.pointwise_pairwise import save_scorer, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--selection", default="results/neural_selection_v6.json")
    parser.add_argument("--baseline", default="runs/baseline_v5/pairwise_model.pt")
    parser.add_argument("--out", default="submission/pairwise_model.pt")
    parser.add_argument("--history-out", default="results/pairwise_refit_v6.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    selection = json.loads(Path(args.selection).read_text())
    specification = selection["selected"]["candidate"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    if selection["promoted"]:
        train = load_ltr(args.train)[0]
        network, standardizer, history = train_model("pairwise", train, None, patience=0, verbose=True,
                                                      **fit_arguments(specification))
        save_scorer(args.out, network, standardizer)
    else:
        shutil.copy2(args.baseline, args.out)
    report = dict(arguments=vars(args), promoted=selection["promoted"], specification=specification,
                  history=history, artifact_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
                  seconds=time.perf_counter()-started, trained_on="TRAIN only" if selection["promoted"] else "unchanged v5 TRAIN artifact")
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
