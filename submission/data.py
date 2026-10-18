"""Bounded-memory SVMLight loading for training, without changing staff code.

Two passes avoid retaining Python tuples for every feature in a large file.
All inputs share one inferred width; rows are grouped by first-seen query.
"""
import numpy as np


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.partition("#")[0].split()
            if fields:
                if len(fields) < 2 or not fields[1].startswith("qid:") or not fields[1][4:]:
                    raise ValueError(f"{path}:{line_no}: expected label qid:<id> features")
                yield line_no, fields


def load_ltr(*paths, allow_nan=False):
    """Return one (float32 X, integer labels, qids) tuple per input path.

    Tree recipes may preserve NaNs for native missing-value handling; neural
    training requires finite features because its standardizer is arithmetic.
    """
    metadata, width = [], 0
    for path in paths:
        count = 0
        for line_no, fields in _rows(path):
            count += 1
            for token in fields[2:]:
                try:
                    index, _ = token.split(":", 1)
                    index = int(index)
                    if index < 1:
                        raise ValueError("feature indices must be positive")
                    width = max(width, index)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid feature {token!r}") from exc
        if not count:
            raise ValueError(f"{path}: empty LTR input")
        metadata.append(count)
    if width < 1:
        raise ValueError("LTR inputs must contain at least one feature")
    output = []
    for path, count in zip(paths, metadata):
        X = np.zeros((count, width), dtype=np.float32)
        labels = np.empty(count, dtype=np.int64)
        codes = np.empty(count, dtype=np.int32)
        queries = {}
        for row, (line_no, fields) in enumerate(_rows(path)):
            try:
                label = float(fields[0])
                if not label.is_integer() or not 0 <= label <= 4:
                    raise ValueError("labels must be integers in 0..4")
                labels[row] = int(label)
                qid = fields[1][4:]
                codes[row] = queries.setdefault(qid, len(queries))
                for token in fields[2:]:
                    index, value = token.split(":", 1)
                    X[row, int(index) - 1] = float(value)
                invalid = np.isinf(X[row]) if allow_nan else ~np.isfinite(X[row])
                if invalid.any():
                    raise ValueError("features must be finite")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
        names = np.asarray(list(queries), dtype=object)
        if np.any(codes[1:] < codes[:-1]):
            order = np.argsort(codes, kind="stable")
            X, labels, codes = X[order], labels[order], codes[order]
        output.append((X, labels, names[codes]))
    return tuple(output)


def group_sizes(qids):
    """LightGBM group sizes; reject a query appearing in separate blocks."""
    qids = np.asarray(qids)
    if not len(qids):
        return []
    starts = np.r_[0, np.flatnonzero(qids[1:] != qids[:-1]) + 1]
    if len(set(qids[starts])) != len(starts):
        raise ValueError("query rows must be contiguous")
    return np.diff(np.r_[starts, len(qids)]).tolist()
