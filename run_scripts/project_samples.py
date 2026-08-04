from __future__ import annotations

"""Project external CNV segments into the frozen CNVSIM atlas."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.projection import project_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project samples.tsv and 0-based half-open GRCh38 segments.tsv "
            "into the frozen CNVSIM reference."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = project_samples(
        bundle_dir=Path(args.bundle_dir),
        samples_path=Path(args.samples),
        segments_path=Path(args.segments),
        out_dir=Path(args.out_dir),
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] external projection complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
