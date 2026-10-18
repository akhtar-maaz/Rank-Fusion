"""Compute/memory check on a wide, large domain derived only from TRAIN."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np

from harness.cv import grouped_kfold
from submission.data import group_sizes, load_ltr
from submission.features import features_for_model
from submission.recipe import fit_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/recipe_v3")
    args = parser.parse_args()
    started = time.perf_counter()
    X, y, q = load_ltr("data/train/train.txt")[0]
    tr, va = grouped_kfold(q, k=5, seed=0)[0]
    cumulative = np.cumsum(group_sizes(q[tr]))
    count = cumulative[np.searchsorted(cumulative, 500000, side="right") - 1]
    tr = tr[:count]
    width = 300
    columns = np.arange(width) % X.shape[1]
    scale = np.exp(np.random.RandomState(173).uniform(-1, 1, width)).astype(np.float32)
    Xt = X[tr][:, columns] * scale
    Xv = X[va][:, columns] * scale
    yt, yv, qt, qv = y[tr], y[va], q[tr], q[va]
    del X
    fitting = time.perf_counter()
    model, history = fit_arrays(Xt, yt, qt, Xv, yv, qv, seed=0)
    fit_seconds = time.perf_counter() - fitting
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "budget_model.txt"
    model.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    example = Xt[:group_sizes(qt)[0]]
    before = model.predict(features_for_model(model, example), num_threads=1)
    after = restored.predict(features_for_model(restored, example), num_threads=1)
    if not np.array_equal(before, after):
        raise ValueError("recipe save/load prediction differed")
    record = {"source": "Fold1 TRAIN-derived size/dimension stress check; no new dataset labels",
              "train_rows": len(yt), "validation_rows": len(yv), "raw_features": width,
              "model_features": restored.num_feature(), "train_queries": len(set(qt)),
              "validation_queries": len(set(qv)), "fit_seconds": fit_seconds,
              "load_prepare_fit_save_seconds": time.perf_counter() - started,
              "process_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "history": history, "save_load_bit_identical_predictions": True,
              "note": "300 derived columns repeat and rescale the 136 TRAIN columns; a compute check, not a new benchmark. File parsing uses the original TRAIN file; expanded columns are constructed in memory."}
    (destination / "budget.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
