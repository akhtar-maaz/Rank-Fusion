"""Require complete frozen TRAIN-only selection evidence before final DEV work.

Ignored orchestration helper. Importing it has no side effects.
"""
import hashlib
import json
import math
import numbers
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_twelfth_round import select, specifications
from scripts.cv_pairwise_ensemble_v12 import select as neural_select, specifications as neural_specifications, source_fingerprints

PROTOCOL = "3b8b205741db27e371e4365f15f0e6e92bc2eabc"
SOURCE = "c8c1f1a47e133caa66eb2187b721026bcb49bab0"
BASELINE = "1bbeefc9f258dc2da8b945db6df43bd31c34e49b"


def read(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def compare_recomputed(actual, expected, path="result"):
    """Audit derived numeric summaries across NumPy/platform versions.

    Decisions, keys, candidate identities and booleans must agree exactly.
    This does not change selector thresholds or its 1e-12 control check.
    Local/HPC mean arithmetic can differ by one final float64 bit.
    """
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "keys differ: " + path)
        return max((compare_recomputed(actual[key], value, path + "." + key)
                    for key, value in expected.items()), default=0.0)
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), "sequence differs: " + path)
        return max((compare_recomputed(a, b, path + f"[{i}]")
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if isinstance(expected, numbers.Real) and not isinstance(expected, bool):
        require(isinstance(actual, numbers.Real) and not isinstance(actual, bool) and
                math.isfinite(actual) and math.isfinite(expected), "non-finite/wrong numeric type: " + path)
        error = abs(actual - expected)
        require(error <= 1e-12, "derived numeric summary differs: " + path)
        return float(error)
    require(actual == expected and type(actual) is type(expected), "decision/identity differs: " + path)
    return 0.0


def audit():
    metadata = read("runs/v12_jobs.json")
    require(not metadata.get("pending_submission"), "unresolved scheduler intent")
    require((metadata["protocol_commit"], metadata["workflow_commit"], metadata["baseline_commit"]) ==
            (PROTOCOL, SOURCE, BASELINE), "frozen identities differ")
    require(all(metadata.get(k) for k in ("cv_array", "neural_selection", "tree_selection")),
            "V12 first-stage jobs have not been submitted")
    jobs = read("runs/v12_job_metadata.json")
    ids = [metadata["cv_array"].replace("[]", f"[{i}]") for i in range(4)]
    ids += [metadata[key] for key in ("neural_selection", "tree_selection")]
    require(len(set(ids)) == 6, "need six distinct first-stage job records")
    for job in ids:
        require(jobs.get(job, {}).get("job_state") == "F" and
                jobs[job].get("Exit_status") == "0", "selection job not successful: " + job)
    require(metadata["source_ci"]["conclusion"] == "success" and metadata["hpc_source_tests"]["passed"] == 176,
            "source checks incomplete")
    require(digest("results/lambdamart_cv_release_v11.json") == metadata["baseline_configuration_sha256"],
            "archived TRAIN control changed")
    for name, expected in metadata["baseline_hashes"].items():
        require(digest("submission/" + name) == expected, "selection audit requires intact baseline: " + name)
    for path, expected in metadata["source_sha256"].items():
        require(digest(path) == expected, "frozen source changed: " + path)
    for path in ("configs/v12_candidates.json", "configs/v12_neural_candidates.json"):
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", SOURCE + ":" + path])
        require((ROOT / path).read_bytes() == original, "candidate grid changed: " + path)
    identities = read("runs/v12_identity_preflight.json")
    require(identities["all_source_hashes_match_committed_bundle"] and
            identities["input_sha256"] == metadata["fit_input_sha256"] and
            identities["all_protected_artifacts_unchanged"], "server identities incomplete")

    tree = read("results/lambdamart_cv_v12.json")
    require((tree["rows"], tree["queries"], tree["completed_candidates"], tree["runtime_excluded_candidates"]) ==
            (723412, 6000, list(range(4)), []), "tree data/candidates incomplete")
    specs = specifications()
    require([r["name"] for r in tree["results"]] == [r["name"] for r in specs], "33 fixed settings differ")
    for row, setting in zip(tree["results"], specs):
        require(row["sources"] == [list(s) for s in setting["sources"]], "tree sources changed")
        require([m["weight"] for m in row.get("members", [dict(weight=1)])] == setting["weights"],
                "tree weights changed")
    frozen_candidates = read("configs/v12_candidates.json")
    for candidate in range(4):
        part = read(f"runs/cv_v12/part_{candidate}.json")
        require(part["candidate"] == frozen_candidates[candidate] and
                (part["rows"], part["queries"]) == (723412, 6000), "candidate report differs")
        require([r["trees"] for r in part["results"]] == [200, 400], "missing checkpoints")
        require(all(sorted(f["fold"] for f in r["folds"]) == list(range(5)) for r in part["results"]),
                "missing candidate folds")
    expected = select(tree["results"], read("results/lambdamart_cv_release_v11.json"))
    tree_recomputation_error = max(compare_recomputed(tree[key], value, "tree." + key)
                                   for key, value in expected.items())
    native = tree["native_equivalence"]
    require(sorted(r["fold"] for r in native) == list(range(5)), "need five native checks")
    require(all(r["native_max_abs_error"] <= 5e-12 and
                abs(r["native_ndcg_10"] - r["explicit_ndcg_10"]) <= 1e-12 for r in native),
            "native arithmetic differs")

    neural = read("results/neural_selection_v12.json")
    neural_specs = neural_specifications()
    require(read("configs/v12_neural_candidates.json") == neural_specs, "neural grid changed")
    reports = [read(f"runs/neural_v12/part_fold{fold}.json") for fold in range(5)]
    reference = read("results/neural_selection_v8.json")["selected"]
    expected_neural = neural_select(reports, neural_specs, reference)
    neural_recomputation_error = max(compare_recomputed(neural[key], value, "neural." + key)
                                     for key, value in expected_neural.items())
    context = neural["context"]
    require((context["rows"], context["queries"], context["features"]) == (723412, 6000, 136) and
            context["train_sha256"] == metadata["fit_input_sha256"]["TRAIN"] and
            context["source_sha256"] == source_fingerprints(), "neural input/source differs")
    require(neural["control"]["candidate"] == reference["candidate"], "wrong neural control")
    require(neural["selection_protocol"]["total_member_fits"] == 50, "need 50 fresh member fits")
    for report in reports:
        proof = report["results"][0]["control_reference"]
        require(proof["candidate_equal"] and proof["exported_metrics_equal"] and
                all(p["artifact_sha256_checked"] and p["artifact_sha256_equal"] and
                    (not p["archived_state_checked"] or p["state_and_statistics_equal"]) and
                    p["history_equal"] for p in proof["member_checks"]), "fresh neural control proof incomplete")
    return dict(protocol_commit=PROTOCOL, workflow_commit=SOURCE, baseline_commit=BASELINE,
                selected_on="released TRAIN only", new_dev_measurement_inspected=False,
                first_stage_jobs={job: jobs[job] for job in ids}, tree_settings=33,
                tree_promoted=tree["promoted"], tree_selected=tree["selected"],
                tree_promotion_gates=tree["promotion_gates"], tree_native_equivalence=native,
                neural_candidates=5, fresh_neural_member_fits=50,
                neural_promoted=neural["promoted"], neural_selected=neural["selected"]["candidate"],
                neural_mean_gain=neural["proposed_mean_gain"], neural_improved_folds=neural["proposed_improved_folds"],
                numeric_recomputation_audit=dict(absolute_tolerance=1e-12,
                    tree_max_abs_error=tree_recomputation_error, neural_max_abs_error=neural_recomputation_error,
                    decisions_and_identities_exact=True, promotion_thresholds_unchanged=True,
                    reason="Local/HPC NumPy/platform float64 mean arithmetic can differ by final bits; all selector gates, control tolerances, candidates and decisions are unchanged."),
                protected_sha256=metadata["baseline_hashes"],
                part_sha256={path: digest(path) for path in ([f"runs/cv_v12/part_{i}.json" for i in range(4)] + [f"runs/neural_v12/part_fold{i}.json" for i in range(5)])},
                identity_preflight=identities, source_sha256=metadata["source_sha256"],
                selection_sha256={path: digest(path) for path in
                                  ("results/lambdamart_cv_v12.json", "results/neural_selection_v12.json")})



if __name__ == "__main__":
    evidence = audit()
    target = ROOT / "results/v12_selection_audit.json"
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(tree_promoted=evidence["tree_promoted"],
                          neural_promoted=evidence["neural_promoted"],
                          audit=str(target))))
