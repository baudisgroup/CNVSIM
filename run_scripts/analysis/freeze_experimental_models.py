from __future__ import annotations

"""Freeze compact experimental-model application artifacts."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.experimental_models import freeze_experimental_model_projection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the experimental-model projection application."
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    outputs = freeze_experimental_model_projection(
        atlas_dir=atlas_dir,
        work_dir=(
            Path(args.work_dir).expanduser().resolve()
            if args.work_dir
            else None
        ),
        out_dir=(
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else None
        ),
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] frozen experimental-model application")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
