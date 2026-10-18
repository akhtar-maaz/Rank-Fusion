"""Check final native/staff loaders, protected artifacts and content hashes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import torch

from harness.neural import load_scorer
from submission import ranking
from submission.features import feature_mode_for_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/artifacts.json")
    parser.add_argument("--hpc-manifest")
    args = parser.parse_args()
    torch.set_num_threads(4)
    previous = json.loads((ROOT / "results/artifacts_v2.json").read_text())["artifacts"]
    hpc = json.loads(Path(args.hpc_manifest).read_text())["artifacts"] if args.hpc_manifest else None
    records = {}
    for name in ("model.txt", "pointwise_model.pt", "pairwise_model.pt", "fusion_config.json"):
        path = ROOT / "submission" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        unchanged = digest == previous[name]["sha256"]
        if name != "model.txt" and not unchanged:
            raise ValueError(f"protected v2 artifact changed: {name}")
        records[name] = dict(bytes=path.stat().st_size, sha256=digest, unchanged_from_v2=unchanged)
        if hpc is not None:
            if digest != hpc[name]["sha256"]:
                raise ValueError(f"local/HPC artifact differed: {name}")
            records[name]["matches_hpc_copy"] = True
    model = lgb.Booster(model_file=str(ROOT / "submission/model.txt"))
    selected = json.loads((ROOT / "results/lambdamart_cv.json").read_text())["selected"]
    mode = selected.get("feature_mode", "raw")
    multiplier = 1 if mode == "raw" else 3 if mode == "query_all" else 2
    if model.num_trees() != selected["trees"] or model.num_feature() != 136 * multiplier or feature_mode_for_model(model) != mode:
        raise ValueError("artifact does not match the frozen TRAIN selection")
    records["model.txt"].update(features=model.num_feature(), input_features=136, feature_mode=mode, trees=model.num_trees())
    X = np.random.RandomState(2026).normal(size=(16, 136)).astype(np.float32)
    for kind in ("pointwise", "pairwise"):
        path = ROOT / "submission" / f"{kind}_model.pt"
        blob = torch.load(path, weights_only=True)
        if blob["config"] != {"n_features": 136, "hidden": [128, 64]}:
            raise ValueError("staff neural architecture changed")
        scorer = load_scorer(str(path))
        scores = scorer(X)
        if scores.shape != (16,) or not np.isfinite(scores).all() or not np.array_equal(scores, scorer(X)):
            raise ValueError("staff neural loading failed")
        records[path.name].update(config=blob["config"], staff_loader_passed=True,
                                  staff_loader_python312_passed=sys.version_info[:2] == (3, 12))
    ids = [f"q:{i}" for i in range(len(X))]
    features = dict(zip(ids, X.tolist()))
    ranking.prepare(str(ROOT / "submission/model.txt"))
    first = ranking.rerank_with_ltr("q", ids, features, 10)
    if (first != ranking.rerank_with_ltr("q", ids, features, 10) or len(first) != 10 or
            len({doc for doc, _ in first}) != 10 or not {doc for doc, _ in first} <= set(ids)):
        raise ValueError("ranking contract or determinism failed")
    record = dict(artifacts=records, local_python=platform.python_version(),
                  local_versions={name: importlib.metadata.version(name) for name in ("numpy", "lightgbm", "torch")},
                  ranking_deterministic=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
