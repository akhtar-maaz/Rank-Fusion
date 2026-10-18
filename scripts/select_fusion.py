"""Compare a small, predeclared set of fusion rules on TRAIN only."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from harness.feature_names import base_scores_from_row
from harness.metrics import average_precision, ndcg_at_k
from submission.data import group_sizes, load_ltr
from submission.fusion import rrf_scores, combsum_scores, combmnz_scores


def select(train_path):
    X, y, q = load_ltr(train_path)[0]
    configs = [{"method": "rrf", "c": c} for c in (60.0, 0.0, 10.0, 30.0, 100.0)]
    configs += [{"method": "combsum"}, {"method": "combmnz"}]
    totals = np.zeros((len(configs), 2))
    start, sizes = 0, group_sizes(q)
    for size in sizes:
        ids = [str(i) for i in range(size)]
        base = {d: base_scores_from_row(row) for d, row in zip(ids, X[start:start + size])}
        qrels = dict(zip(ids, map(int, y[start:start + size])))
        for i, config in enumerate(configs):
            method = config["method"]
            if method == "rrf":
                scores = rrf_scores(base, c=config["c"])
            else:
                scores = (combsum_scores if method == "combsum" else combmnz_scores)(base)
            ranked = sorted(ids, key=lambda d: -scores[d])[:10]
            totals[i] += ndcg_at_k(ranked, qrels, 10), average_precision(ranked, qrels)
        start += size
    rows = [{"config": c, "ndcg@10": float(t[0] / len(sizes)), "map@10": float(t[1] / len(sizes))}
            for c, t in zip(configs, totals)]
    # Local proxy for Track A; class percentiles cannot be computed locally.
    best = max(rows, key=lambda r: 0.8 * r["ndcg@10"] + 0.2 * r["map@10"])
    return {"source": train_path, "selected_on": "TRAIN only", "queries": len(sizes),
            "results": rows, "selected": best["config"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--out", default="runs/fusion_selection.json")
    parser.add_argument("--config-out", default="submission/fusion_config.json")
    args = parser.parse_args()
    report = select(args.train)
    for path, value in ((args.out, report), (args.config_out, report["selected"])):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
