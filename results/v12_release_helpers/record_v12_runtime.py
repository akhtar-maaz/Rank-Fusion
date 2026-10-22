"""Operational runtime receipt; does not fit, seed, or change training settings."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

path = Path(sys.argv[1])
assert not path.exists(), 'do not replace an existing fit runtime receipt'
path.write_text(json.dumps(dict(python=platform.python_version(),
    versions={name: importlib.metadata.version(name) for name in ('numpy','torch','lightgbm')},
    environment={name: os.environ.get(name) for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}), indent=2)+'\n')
