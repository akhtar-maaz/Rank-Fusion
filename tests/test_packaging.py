"""The exported assignment must never include local real-data files."""
import subprocess
import zipfile

import pytest

from scripts.package_submission import REQUIRED, build_archive


def test_package_preserves_layout_without_local_data(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in REQUIRED | {"README.md", "data/toy/train.txt"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    real = root / "data/train/train.txt"
    real.parent.mkdir(parents=True)
    real.write_text("local data")
    destination = tmp_path / "submission.zip"
    build_archive(root, destination)
    with zipfile.ZipFile(destination) as archive:
        assert REQUIRED <= set(archive.namelist())
        assert "data/toy/train.txt" in archive.namelist()
        assert "data/train/train.txt" not in archive.namelist()
    subprocess.run(["git", "-C", str(root), "add", "data/train/train.txt"], check=True)
    with pytest.raises(ValueError, match="local-only"):
        build_archive(root, destination)
