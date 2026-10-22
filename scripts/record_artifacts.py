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
    parser.add_argument("--pairwise-refit", help="full-TRAIN refit record for a frozen promoted pairwise choice")
    parser.add_argument("--baseline-manifest", help="protected checkpoint hashes; defaults to the original v2 artifacts")
    parser.add_argument("--fusion-selection", help="frozen TRAIN-only selection authorizing a changed fusion configuration")
    args = parser.parse_args()
    torch.set_num_threads(4)
    previous = json.loads((ROOT / "results/artifacts_v2.json").read_text())["artifacts"]
    protected = json.loads(Path(args.baseline_manifest).read_text())["artifacts"] if args.baseline_manifest else previous
    hpc = json.loads(Path(args.hpc_manifest).read_text())["artifacts"] if args.hpc_manifest else None
    permitted_pairwise = None
    permitted_fusion = None
    if args.fusion_selection:
        selection = json.loads(Path(args.fusion_selection).read_text())
        if selection["selected_on"] != "released TRAIN only":
            raise ValueError("fusion selection must use TRAIN only")
        actual_config = json.loads((ROOT / "submission/fusion_config.json").read_text())
        if actual_config != selection["selected"]:
            raise ValueError("fusion artifact differs from frozen selection")
        permitted_fusion = hashlib.sha256((ROOT / "submission/fusion_config.json").read_bytes()).hexdigest()
    pairwise_hidden = protected["pairwise_model.pt"].get("config", {}).get("hidden", [128, 64])
    if args.pairwise_refit:
        refit = json.loads(Path(args.pairwise_refit).read_text())
        selection = json.loads(Path(refit["arguments"]["selection"]).read_text())
        specification = refit.get("specification", refit.get("candidate"))
        if refit["promoted"] != selection["promoted"] or specification != selection["selected"]["candidate"]:
            raise ValueError("pairwise refit does not match its frozen selection")
        if refit["promoted"]:
            permitted_pairwise = refit["artifact_sha256"]
            if "members" in specification:
                hidden = [member["hidden"] for member in specification["members"]]
                if len({len(sizes) for sizes in hidden}) != 1:
                    raise ValueError("frozen pairwise members have different depths")
                pairwise_hidden = [sum(sizes[layer] for sizes in hidden) for layer in range(len(hidden[0]))]
            else:
                pairwise_hidden = specification["hidden"]
    records = {}
    for name in ("model.txt", "pointwise_model.pt", "pairwise_model.pt", "fusion_config.json"):
        path = ROOT / "submission" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        unchanged = digest == previous[name]["sha256"]
        unchanged_protected = digest == protected[name]["sha256"]
        if (name != "model.txt" and not unchanged_protected and
                not (name == "pairwise_model.pt" and digest == permitted_pairwise) and
                not (name == "fusion_config.json" and digest == permitted_fusion)):
            raise ValueError(f"protected checkpoint artifact changed: {name}")
        records[name] = dict(bytes=path.stat().st_size, sha256=digest, unchanged_from_v2=unchanged)
        if args.baseline_manifest:
            records[name]["unchanged_from_baseline"] = unchanged_protected
        if hpc is not None:
            if digest != hpc[name]["sha256"]:
                raise ValueError(f"local/HPC artifact differed: {name}")
            records[name]["matches_hpc_copy"] = True
    model = lgb.Booster(model_file=str(ROOT / "submission/model.txt"))
    selected = json.loads((ROOT / "results/lambdamart_cv.json").read_text())["selected"]
    mode = selected.get("feature_mode", "raw")
    multiplier = 1 if mode in ("raw", "query_percentile") else 3 if mode == "query_all" else 2
    if model.num_trees() != selected["trees"] or model.num_feature() != 136 * multiplier or feature_mode_for_model(model) != mode:
        raise ValueError("artifact does not match the frozen TRAIN selection")
    records["model.txt"].update(features=model.num_feature(), input_features=136, feature_mode=mode, trees=model.num_trees())
    X = np.random.RandomState(2026).normal(size=(16, 136)).astype(np.float32)
    for kind in ("pointwise", "pairwise"):
        path = ROOT / "submission" / f"{kind}_model.pt"
        blob = torch.load(path, weights_only=True)
        expected_hidden = [128, 64] if kind == "pointwise" else pairwise_hidden
        if blob["config"] != {"n_features": 136, "hidden": expected_hidden}:
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
