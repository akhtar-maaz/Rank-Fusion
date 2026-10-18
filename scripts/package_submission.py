"""Package the repository layout and artifacts, excluding local training data."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile

REQUIRED = {"submission/ranking.py", "submission/fusion.py", "submission/pointwise_pairwise.py",
            "submission/recipe.py", "submission/model.txt", "submission/pointwise_model.pt",
            "submission/pairwise_model.pt", "submission/fusion_config.json", "submission/features.py"}
FORBIDDEN = (("data", "train"), ("data", "dev"), ("data", "downloads"), (".venv",), ("runs",), ("dist",))


def build_archive(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    names = [name for name in tracked if name]
    if not REQUIRED <= set(names):
        raise ValueError(f"required submission files are untracked or missing: {sorted(REQUIRED - set(names))}")
    for name in names:
        parts, path = Path(name).parts, root / name
        if any(parts[:len(prefix)] == prefix for prefix in FORBIDDEN):
            raise ValueError(f"local-only file is tracked; refusing to package: {name}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing file or unsupported symlink: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            archive.write(root / name, arcname=name)
    return {"files": len(names), "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/rank-fusion-submission.zip")
    args = parser.parse_args()
    print(build_archive(Path(__file__).resolve().parents[1], args.out))


if __name__ == "__main__":
    main()
