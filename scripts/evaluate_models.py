"""Final DEV evaluation through the submission interface, with bounded memory."""
import argparse
import importlib.metadata
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from harness.feature_names import BASE_RANKER_CHANNELS, base_scores_from_row
from harness.metrics import average_precision, mean_ndcg_from_scores, ndcg_at_k
from harness.neural import load_scorer
from submission.data import group_sizes, load_ltr
import submission.ranking as ranking


def evaluate(features_path, model_dir):
    X, y, q = load_ltr(features_path)[0]
    before = time.perf_counter()
    ranking.prepare(os.path.join(model_dir, "model.txt"))
    load_seconds = time.perf_counter() - before
    totals = np.zeros(3)
    maximum_fusion, maximum_ltr = 0.0, 0.0
    sizes, start = group_sizes(q), 0
    for size in sizes:
        ids = [f"{q[start]}:{i}" for i in range(size)]
        rows, labels = X[start:start + size], y[start:start + size]
        qrels = dict(zip(ids, map(int, labels)))
        base = {d: base_scores_from_row(row) for d, row in zip(ids, rows)}
        features = dict(zip(ids, rows.tolist()))
        before = time.perf_counter()
        fused = ranking.fuse_candidates(str(q[start]), ids, base, 10)
        maximum_fusion = max(maximum_fusion, time.perf_counter() - before)
        before = time.perf_counter()
        reranked = ranking.rerank_with_ltr(str(q[start]), ids, features, 10)
        maximum_ltr = max(maximum_ltr, time.perf_counter() - before)
        for results in (fused, reranked):
            returned = [d for d, _ in results]
            if len(returned) > 10 or len(set(returned)) != len(returned) or not set(returned) <= set(ids):
                raise ValueError("submission violated candidate contract")
        fused_ids = [d for d, _ in fused]
        totals += ndcg_at_k(fused_ids, qrels), average_precision(fused_ids, qrels), ndcg_at_k([d for d, _ in reranked], qrels)
        start += size
    totals /= len(sizes)
    result = {"features": features_path, "queries": len(sizes), "rows": len(y),
              "fusion_ndcg@10": float(totals[0]), "fusion_map@10": float(totals[1]),
              "lambdamart_ndcg@10": float(totals[2]),
              "bm25_ndcg@10": mean_ndcg_from_scores(X[:, BASE_RANKER_CHANNELS["bm25_whole_document"] - 1], y, q),
              "timing": {"prepare_seconds": load_seconds, "maximum_fusion_seconds": maximum_fusion,
                         "maximum_rerank_seconds": maximum_ltr},
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "lightgbm", "torch")}}
    for kind in ("pointwise", "pairwise"):
        scorer = load_scorer(os.path.join(model_dir, f"{kind}_model.pt"))
        scores = np.concatenate([scorer(X[start:start + 4096]) for start in range(0, len(X), 4096)])
        result[f"{kind}_ndcg@10"] = mean_ndcg_from_scores(scores, y, q)
    result["pairwise_beats_bm25"] = result["pairwise_ndcg@10"] > result["bm25_ndcg@10"]
    result["within_per_call_limits"] = load_seconds < 60 and maximum_fusion < 2 and maximum_ltr < 2
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/dev/vali.txt")
    parser.add_argument("--model-dir", default="submission")
    parser.add_argument("--out", default="results/dev_metrics.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = evaluate(args.features, args.model_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pairwise_beats_bm25"] or not result["within_per_call_limits"]:
        raise SystemExit("final model did not meet an assignment acceptance check; see saved metrics")


if __name__ == "__main__":
    main()
