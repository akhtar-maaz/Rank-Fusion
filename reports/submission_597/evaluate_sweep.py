"""Report-only DEV prefix diagnostic of the exact submission 597 archive.

This never fits, selects, or overwrites a model. Run with the project venv.
The three tree counts were swept in the archived TRAIN-only V3 experiment;
the DEV values are retrospective in-sample diagnostics for the report.
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from harness.metrics import mean_ndcg_from_scores
from submission.data import load_ltr
from submission.features import features_for_model


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = ROOT / "dist/rank-fusion-submission-v12.zip"
    expected_archive = "e42c56344bcce528e48b803ba2fb265aa3adb93cddc53791876dbd9f373b6e28"
    assert sha(archive.read_bytes()) == expected_archive
    with zipfile.ZipFile(archive) as z:
        for name in ["submission/data.py", "submission/features.py", "submission/tree_ensemble.py", "harness/metrics.py"]:
            assert z.read(name) == (ROOT / name).read_bytes(), name
        model_bytes = z.read("submission/model.txt")
        assert sha(model_bytes) == "f2fa40faac67aaab067bb43eb169fd7d7db5a13c5eeb0a6d248c88f941e44864"
        fit = json.loads(z.read("results/tree_fit_v10.json"))
        first = fit["selected"]["members"][0]
        assert first["trees"] == 800 and first["feature_mode"] == "query_rank"
        assert first["weight"] == 0.0703125
        cv = json.loads(z.read("results/lambdamart_cv_v3.json"))
        historical = {a["trees"]: a for a in cv["results"] if a["name"] == "v2_control"}
        dev_reference = json.loads(z.read("results/dev_metrics_v12.json"))
    dev_path = ROOT / "data/dev/vali.txt"
    # Verify the already released file, not an official external download.
    dev_hash = sha(dev_path.read_bytes())
    assert dev_hash == next(a for a in fit["inputs"] if a["split"] == "DEV")["sha256"]
    print("Verified V12 archive, model, source and released DEV identities.", flush=True)
    X, y, q = load_ltr(dev_path)[0]
    model = lgb.Booster(model_str=model_bytes.decode("utf-8"))
    assert model.num_trees() == 3800 and model.num_feature() == 408
    prepared = features_for_model(model, X, q)
    rows = []
    for trees in [200, 400, 800]:
        pred = model.predict(prepared, start_iteration=0, num_iteration=trees, num_threads=4)
        score = mean_ndcg_from_scores(pred, y, q, k=10)
        original = historical[trees]
        rows.append({
            "trees": trees,
            "dev_ndcg_10_in_sample": score,
            "train_cv_training_mean": original["train_mean"],
            "train_cv_validation_mean": original["validation_mean"],
            "train_cv_validation_std": original["validation_std"],
            "train_cv_diagnostic_mean": original["diagnostic_mean"],
        })
        print(rows[-1], flush=True)
    # Cross-platform inference must still agree with the evaluated V12 model.
    full = mean_ndcg_from_scores(model.predict(prepared, num_threads=4), y, q, k=10)
    assert abs(full - dev_reference["lambdamart_ndcg@10"]) < 1e-12
    result = {
        "submission": 597,
        "report_diagnostic_date": "2026-10-24",
        "archive": "dist/rank-fusion-submission-v12.zip",
        "archive_sha256": expected_archive,
        "model_sha256": sha(model_bytes),
        "dev_sha256": dev_hash,
        "rows": len(y), "queries": 2000,
        "lightgbm_version": lgb.__version__,
        "description": "First unaugmented query_rank member of the shipped V12 six-member model, prefixes 200/400/800. Its positive export weight changes score scale but not ranking. DEV was part of fitting: all DEV values are in-sample. This retrospective reporting diagnostic is not a model-selection run and changes no submitted artifact.",
        "historical_train_cv_source": "V12 ZIP: results/lambdamart_cv_v3.json, name=v2_control",
        "full_submitted_model_dev_ndcg_10": full,
        "points": rows,
    }
    out = Path(__file__).with_name("tree_count_sweep.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
