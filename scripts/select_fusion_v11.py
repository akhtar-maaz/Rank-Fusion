"""Fixed finite TRAIN-only fusion comparison, with vectorized hand calculations."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from harness.cv import grouped_kfold
from harness.feature_names import BASE_RANKER_CHANNELS
from submission.data import group_sizes, load_ltr

CHANNELS = tuple(BASE_RANKER_CHANNELS)


def configurations(control):
    weights = [dict(zip(CHANNELS, (n / 10 for n in counts)))
               for counts in itertools.product(range(11), repeat=4) if sum(counts) == 10]
    return ([dict(method="weighted_rrf", c=c, weights=w) for c in (0., 10., 60.) for w in weights]
            + [dict(method="weighted_combsum", weights=w) for w in weights]
            + [dict(method="combmnz"), control])


def query_scores(raw, configs):
    """raw is documents x four channels, in candidate order; scores are rules x docs."""
    raw = np.asarray(raw, dtype=np.float64)
    low, high = raw.min(axis=0), raw.max(axis=0)
    normalized = (raw - low) / np.where(high > low, high - low, 1.)
    ranks = np.empty(raw.shape, dtype=np.int64)
    for channel in range(4):
        order = np.argsort(-raw[:, channel], kind="stable")
        ranks[order, channel] = np.arange(1, len(raw) + 1)
    scores = np.zeros((len(configs), len(raw)), dtype=np.float64)
    for method in ("weighted_rrf", "weighted_combsum"):
        indices = [i for i, config in enumerate(configs) if config["method"] == method]
        if not indices:
            continue
        weights = np.asarray([[configs[i]["weights"].get(c, 1.) for c in CHANNELS] for i in indices])
        constants = np.asarray([configs[i].get("c", 0.) for i in indices])[:, None]
        # Use the same channel accumulation order and division as hand-written inference.
        for channel in range(4):
            scores[indices] += (weights[:, channel, None] / (constants + ranks[:, channel])
                                if method == "weighted_rrf" else
                                weights[:, channel, None] * normalized[:, channel])
    for i, config in enumerate(configs):
        if config["method"] == "combmnz":
            scores[i] = normalized.sum(axis=1) * (normalized > 0).sum(axis=1)
        elif config["method"] not in ("weighted_rrf", "weighted_combsum"):
            raise ValueError("unsupported fixed fusion method")
    return scores


def query_metrics(scores, labels):
    labels = np.asarray(labels)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    relevance = labels[top]
    discount = 1. / np.log2(np.arange(relevance.shape[1]) + 2)
    ideal = np.sort(labels)[::-1][:10]
    idcg = np.sum((2. ** ideal - 1) * discount)
    ndcg = np.sum((2. ** relevance - 1) * discount, axis=1) / (idcg if idcg else 1.)
    relevant = relevance > 0
    ap = np.sum(np.cumsum(relevant, axis=1) / np.arange(1, relevance.shape[1] + 1) * relevant, axis=1)
    ap /= max(1, np.count_nonzero(labels > 0))
    return np.stack([ndcg, ap], axis=1)


def decision(metrics, query_folds):
    means = metrics.mean(axis=1)
    proxy = .8 * means[:, 0] + .2 * means[:, 1]
    winner, control = int(np.argmax(proxy)), len(proxy) - 1
    folds = np.stack([metrics[:, query_folds == fold].mean(axis=1) for fold in range(5)], axis=1)
    fold_proxy = .8 * folds[:, :, 0] + .2 * folds[:, :, 1]
    gain = float(proxy[winner] - proxy[control])
    gains = fold_proxy[winner] - fold_proxy[control]
    passed = bool(gain >= .0002 and np.count_nonzero(gains > 0) >= 3)
    return dict(winner=winner, control=control, selected=winner if passed else control,
                promoted=passed, proxy_gain=gain, paired_fold_gains=gains.tolist(),
                improved_folds=int(np.count_nonzero(gains > 0))), means, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train/train.txt")
    parser.add_argument("--control", default="configs/v11_fusion_control.json")
    parser.add_argument("--out", default="results/fusion_selection_v11.json")
    args = parser.parse_args()
    started = time.perf_counter()
    control = json.loads(Path(args.control).read_text())
    configs = configurations(control)
    assert len(configs) == len({json.dumps(c, sort_keys=True) for c in configs}) == 1146
    X, y, q = load_ltr(args.train)[0]
    columns = [i - 1 for i in BASE_RANKER_CHANNELS.values()]
    sizes = group_sizes(q)
    metrics = np.empty((len(configs), len(sizes), 2), dtype=np.float64)
    row_folds = np.empty(len(q), dtype=int)
    for fold, (_, va) in enumerate(grouped_kfold(q, k=5, seed=0)):
        row_folds[va] = fold
    query_folds, start = [], 0
    for query, size in enumerate(sizes):
        end = start + size
        metrics[:, query] = query_metrics(query_scores(X[start:end, columns], configs), y[start:end])
        query_folds.append(row_folds[start])
        start = end
    if not np.isfinite(metrics).all():
        raise ValueError("nonfinite fusion metrics")
    gates, means, folds = decision(metrics, np.asarray(query_folds))
    report = dict(arguments=vars(args), selected_on="released TRAIN only", rows=len(y), queries=len(sizes),
                  fold_seed=0, proxy="0.8 nDCG@10 + 0.2 MAP@10; not class percentiles",
                  selected=configs[gates["selected"]], promotion_gates=gates,
                  results=[dict(config=config, ndcg_10=float(means[i, 0]), map_10=float(means[i, 1]),
                                folds=[dict(fold=f, ndcg_10=float(folds[i, f, 0]), map_10=float(folds[i, f, 1]))
                                       for f in range(5)]) for i, config in enumerate(configs)],
                  seconds=time.perf_counter() - started)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pending.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({key: report[key] for key in ("selected", "promotion_gates", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
