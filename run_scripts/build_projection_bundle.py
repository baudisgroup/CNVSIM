from __future__ import annotations

"""Build the portable fixed-atlas projection reference bundle."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.config import load_analysis_config
from cnvsim.config import load_preprocess_config
from cnvsim.projection import build_projection_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen CNVSIM projection reference bundle."
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument(
        "--preprocess-config", default="configs/preprocess_baseline.yaml"
    )
    parser.add_argument("--analysis-config", default="configs/analysis_v1.yaml")
    parser.add_argument("--preprocess-release-dir")
    parser.add_argument("--release-candidate-dir")
    parser.add_argument("--conditional-reference-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    preprocess_config = load_preprocess_config(args.preprocess_config)
    analysis_config = load_analysis_config(args.analysis_config)
    analysis_run_dir = atlas_dir / "work" / analysis_config.run_name / "analysis"
    conditional_reference_dir = (
        Path(args.conditional_reference_dir).expanduser().resolve()
        if args.conditional_reference_dir
        else analysis_run_dir / "conditional_reference"
    )
    outputs = build_projection_bundle(
        atlas_dir=atlas_dir,
        preprocess_release_dir=(
            Path(args.preprocess_release_dir).expanduser().resolve()
            if args.preprocess_release_dir
            else atlas_dir / "preprocessing" / "baseline"
        ),
        analysis_run_dir=analysis_run_dir,
        release_candidate_dir=(
            Path(args.release_candidate_dir).expanduser().resolve()
            if args.release_candidate_dir
            else atlas_dir / "release_candidate"
        ),
        out_dir=(
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else atlas_dir / "projection_bundle_candidate"
        ),
        low_fraction_threshold=float(preprocess_config.encoding.low_frac_threshold),
        high_fraction_threshold=float(preprocess_config.encoding.high_frac_threshold),
        top_candidates=int(analysis_config.projection.top_candidates),
        conditional_reference_dir=(
            conditional_reference_dir
            if conditional_reference_dir.exists()
            else None
        ),
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] projection bundle complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
