"""Train the pointwise and/or pairwise (RankNet) MLP scorers and save pointwise_model.pt / pairwise_model.pt.

    python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission

Select stopping epochs on one query-grouped fold inside TRAIN, then refit on
all TRAIN queries. --dev is optional final evaluation only. Both networks use
the same architecture; the objective and the training examples differ.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import json

from harness.cv import grouped_kfold
from submission.data import load_ltr
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="optional final evaluation ONLY; never used to tune or stop")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--history-out", default="runs/neural_history.json")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pairs-per-query", type=int, default=300)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64],
                    help="hidden layer widths, e.g. --hidden 128 64 (at most 3 layers of at most 256 units)")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    paths = [_resolve(args.train, "train.txt")]
    if args.dev:
        paths.append(_resolve(args.dev, "vali.txt"))
    splits = load_ltr(*paths)
    train = splits[0]
    fit_rows, selection_rows = grouped_kfold(train[2], k=args.folds, seed=args.seed)[0]
    fit_data = tuple(a[fit_rows] for a in train)
    selection_data = tuple(a[selection_rows] for a in train)
    results = {"arguments": vars(args), "seed": args.seed, "trained_on": "TRAIN", "selected_on": "TRAIN internal fold 0",
               "hidden": args.hidden, "models": {}}
    os.makedirs(args.out_dir, exist_ok=True)
    for kind in (["pointwise", "pairwise"] if args.kind == "both" else [args.kind]):
        _, _, hist = train_model(kind, fit_data, selection_data, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                                     lr=args.lr, pairs_per_query=args.pairs_per_query, patience=args.patience,
                                     hidden=args.hidden, verbose=True)
        best_epoch = max(hist, key=lambda row: row["dev_ndcg@10"])["epoch"]
        print(f"{kind}: selected {best_epoch} epochs on TRAIN fold; refitting full TRAIN", flush=True)
        net, std, final_hist = train_model(kind, train, None, seed=args.seed, epochs=best_epoch,
                                          batch_size=args.batch_size, lr=args.lr,
                                          pairs_per_query=args.pairs_per_query, patience=0,
                                          hidden=args.hidden, verbose=True)
        path = os.path.join(args.out_dir, f"{kind}_model.pt")
        save_scorer(path, net, std, args.hidden)
        entry = {"selection_history": hist, "best_epoch": best_epoch, "full_train_history": final_hist}
        if args.dev:
            entry["final_dev_ndcg@10"] = dev_ndcg(make_scorer(net, std), *splits[1])
            print(f"{kind}: final DEV nDCG@10 = {entry['final_dev_ndcg@10']:.6f}", flush=True)
        results["models"][kind] = entry
        os.makedirs(os.path.dirname(args.history_out) or ".", exist_ok=True)
        with open(args.history_out, "w") as stream:
            json.dump(results, stream, indent=2)
        print(f"{kind}: saved {path}", flush=True)


if __name__ == "__main__":
    main()
