#!/usr/bin/env bash
# Test a temporary copy so smoke tests cannot overwrite submitted artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
rank_python="${RANK_FUSION_PYTHON:-python}"
rank_python="$("$rank_python" -c 'import sys; print(sys.executable)')"
if [ "$(uname -s)" = Darwin ]; then
    # PyTorch bundles OpenMP. Have LightGBM use that same runtime on macOS.
    rank_omp="$("$rank_python" -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
    export DYLD_LIBRARY_PATH="$rank_omp${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
rank_scratch="$(mktemp -d)"
trap 'rm -rf "$rank_scratch"' EXIT
cp -R submission harness scripts tests configs results conftest.py "$rank_scratch/"
mkdir "$rank_scratch/data"
cp -R data/toy data/toy_transfer "$rank_scratch/data/"
cd "$rank_scratch"
"$rank_python" scripts/train_model.py --train data/toy --out submission/model.txt --trees 30
"$rank_python" -m harness.run_harness --features data/toy/vali.txt --model submission/model.txt
"$rank_python" -m pytest tests -q
