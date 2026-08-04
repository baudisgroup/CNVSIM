from __future__ import annotations

"""Evaluate candidate projection distance thresholds without rerunning KNN."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.config import load_analysis_config
from cnvsim.analysis.projection_thresholds import evaluate_projection_thresholds


def _quantiles(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-calibrate anchor projection and compare query outcomes "
            "across candidate distance quantiles."
        )
    )
    parser.add_argument("--projection-dir", required=True)
    parser.add_argument("--reference-contexts-dir", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--distance-quantiles", default="0.90,0.95,0.99")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=2701)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_analysis_config(args.analysis_config)
    outputs = evaluate_projection_thresholds(
        projection_dir=Path(args.projection_dir).expanduser().resolve(),
        reference_contexts_dir=(
            Path(args.reference_contexts_dir).expanduser().resolve()
        ),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        params=config.projection,
        distance_quantiles=_quantiles(args.distance_quantiles),
        n_folds=int(args.cv_folds),
        random_seed=int(args.random_seed),
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] projection-threshold sensitivity complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
