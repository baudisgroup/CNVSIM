from __future__ import annotations

"""Build the final calibrated second-stage conditional reference."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.conditional_projection import (
    build_conditional_projection_reference,
)
from cnvsim.analysis.config import load_analysis_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument(
        "--analysis-config", default=Path("configs/analysis_v1.yaml"), type=Path
    )
    parser.add_argument("--analysis-work-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = args.atlas_dir.expanduser().resolve()
    config = load_analysis_config(args.analysis_config)
    analysis_work_dir = (
        args.analysis_work_dir.expanduser().resolve()
        if args.analysis_work_dir is not None
        else atlas_dir / "work" / config.run_name / "analysis"
    )
    validation_dir = (
        args.validation_dir.expanduser().resolve()
        if args.validation_dir is not None
        else atlas_dir
        / "work"
        / "validation"
        / "conditional_contexts_repeated_v1"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else analysis_work_dir / "conditional_reference"
    )
    outputs = build_conditional_projection_reference(
        analysis_work_dir=analysis_work_dir,
        validation_dir=validation_dir,
        out_dir=out_dir,
        params=config.projection,
        overwrite=args.overwrite,
    )
    print("[CNVSIM] conditional projection reference complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
