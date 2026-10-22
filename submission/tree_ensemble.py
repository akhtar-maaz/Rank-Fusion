"""Export an arithmetic average of numeric LambdaMARTs as one native model.

Pinned to the LightGBM v4 text format produced by our LightGBM 4.6 fits.
Only feature indices, tree indices, and output values change. Grading loads
one ordinary LightGBM text booster; no constituent files are needed.
"""
import re

import numpy as np

from submission.features import feature_mode_for_model, tree_feature_names


def combine(models, weights, iterations=None):
    import lightgbm as lgb

    weights = np.asarray(weights, dtype=np.float64)
    if len(models) < 2 or len(models) != len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("need at least two models and matching positive finite weights")
    weights = weights / weights.sum()
    if iterations is None:
        iterations = [None] * len(models)
    if len(iterations) != len(models):
        raise ValueError("one iteration limit is required per model")
    blocks, union_info, first_header, footer, width = [], None, None, None, None
    for model, weight, count in zip(models, weights, iterations):
        mode = feature_mode_for_model(model)
        multiplier = 1 if mode in ("raw", "query_percentile") else 3 if mode == "query_all" else 2
        current_width, remainder = divmod(model.num_feature(), multiplier)
        if remainder or (width is not None and current_width != width):
            raise ValueError("constituents have different raw input widths")
        width = current_width
        if count is not None and not 1 <= count <= model.num_trees():
            raise ValueError("invalid constituent tree count")
        source = model.model_to_string(num_iteration=count)
        head = source[:source.index("\nTree=0\n")]
        metadata = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
        if (metadata.get("version") != "v4" or metadata.get("objective") not in {"lambdarank", "rank_xendcg"} or
                metadata.get("num_class") != "1" or metadata.get("num_tree_per_iteration") != "1" or
                "average_output" in head):
            raise ValueError("only ordinary, single-output numeric LambdaMART models are supported")
        if first_header is None:
            first_header = head
            footer = source[source.index("\nparameters:\n"):]
            union_info = ["none"] * (3 * width)
        mapping = list(range(model.num_feature()))
        if mode == "query_percentile":
            mapping = list(range(width, 2 * width))
        if mode == "query_normalized":
            mapping[:width] = range(2 * width, 3 * width)
        for old, info in enumerate(metadata["feature_infos"].split()):
            union_info[mapping[old]] = info
        end = source.index("\nend of trees\n")
        starts = [match.start() for match in re.finditer(r"(?m)^Tree=\d+\n", source[:end])]
        for start, stop in zip(starts, starts[1:] + [end]):
            lines = source[start:stop].strip().splitlines()
            rewritten = []
            for line in lines:
                key, separator, value = line.partition("=")
                if key == "Tree":
                    value = str(len(blocks))
                elif key == "split_feature":
                    value = " ".join(str(mapping[int(index)]) for index in value.split())
                elif key in ("leaf_value", "internal_value", "shrinkage"):
                    value = " ".join(format(float(number) * weight, ".17g") for number in value.split())
                elif (key == "is_linear" and value != "0") or (key == "num_cat" and value != "0"):
                    raise ValueError("linear or categorical trees are not supported")
                rewritten.append(key + separator + value)
            blocks.append("\n".join(rewritten) + "\n\n\n")
    overrides = {"max_feature_idx": str(3 * width - 1),
                 "feature_names": " ".join(tree_feature_names(width, "query_all")),
                 "feature_infos": " ".join(union_info),
                 "tree_sizes": " ".join(str(len(block.encode())) for block in blocks)}
    header = []
    for line in first_header.splitlines():
        key = line.partition("=")[0]
        header.append(key + "=" + overrides[key] if key in overrides else line)
    footer = re.sub(r"(?m)^\[num_iterations: \d+\]$", f"[num_iterations: {len(blocks)}]", footer)
    native = "\n".join(header).rstrip() + "\n\n" + "".join(blocks) + "end of trees\n" + footer
    result = lgb.Booster(model_str=native)
    if result.num_trees() != len(blocks) or result.num_feature() != 3 * width:
        raise ValueError("native model roundtrip lost trees or features")
    return result
