"""Copy MSLR-WEB10K Fold1 into the layout the repo expects. There is no scriptable download URL: get
MSLR-WEB10K.zip from Microsoft Research, unzip it, then:

    python scripts/prepare_mslr.py --fold-dir /path/to/MSLR-WEB10K/Fold1 --out data

Creates data/train/train.txt (Fold1 train.txt) and data/dev/vali.txt (Fold1 vali.txt). The Fold1 test.txt is the
withheld held-out split: staff keep it; do NOT copy it here (assignment Section 6).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import shutil
import zipfile
from pathlib import PurePosixPath


def prepare_archive(archive_path, out):
    """Read ONLY Fold1 train.txt and vali.txt; never open any other member."""
    with zipfile.ZipFile(archive_path) as archive:
        selected = {}
        for source, destination in (("train.txt", "train/train.txt"), ("vali.txt", "dev/vali.txt")):
            matches = [info for info in archive.infolist()
                       if PurePosixPath(info.filename).parts[-2:] == ("Fold1", source)]
            if len(matches) != 1:
                raise ValueError(f"archive must contain exactly one Fold1/{source}; found {len(matches)}")
            selected[destination] = matches[0]
        for destination, member in selected.items():
            path = os.path.join(out, destination)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with archive.open(member) as source, open(path, "wb") as target:
                shutil.copyfileobj(source, target)
            print(f"{member.filename} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--fold-dir")
    source.add_argument("--archive", help="official ZIP: open only the two permitted Fold1 members")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    if args.archive:
        prepare_archive(args.archive, args.out)
        return
    for src, dst in (("train.txt", "train/train.txt"), ("vali.txt", "dev/vali.txt")):
        s = os.path.join(args.fold_dir, src)
        if not os.path.isfile(s):
            raise SystemExit(f"missing {s}")
        d = os.path.join(args.out, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        print(f"{s} -> {d}")


if __name__ == "__main__":
    main()
