from __future__ import annotations

"""Validate intermittent coherent communities as optional conditional contexts."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.config import load_analysis_config
from cnvsim.validation.conditional_contexts import (
    ConditionalContextValidationParams,
    freeze_conditional_context_validation,
    validate_conditional_contexts,
    validate_conditional_contexts_repeated,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=["validate", "freeze"], default="validate"
    )
    parser.add_argument("--analysis-config", type=Path)
    parser.add_argument("--analysis-work-dir", type=Path)
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--conditional-reference-dir", type=Path)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--minimum-test-samples", type=int, default=30)
    parser.add_argument("--minimum-rescue-eligible-samples", type=int, default=20)
    parser.add_argument("--minimum-assignment-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-conditional-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-operational-precision", type=float, default=0.90)
    parser.add_argument("--maximum-core-control-acceptance", type=float, default=0.05)
    parser.add_argument("--core-controls-per-root", type=int, default=2000)
    parser.add_argument("--minimum-platform-test-samples", type=int, default=10)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of fixed-label hold-out repeats; values above 1 enable repeated validation.",
    )
    parser.add_argument(
        "--minimum-repeat-pass-rate",
        type=float,
        default=0.80,
        help="Minimum fraction of repeats passing all single-split gates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = args.atlas_dir.expanduser().resolve()
    if args.stage == "freeze":
        outputs = freeze_conditional_context_validation(
            atlas_dir=atlas_dir,
            source_dir=args.out_dir,
            conditional_reference_dir=args.conditional_reference_dir,
            out_dir=args.freeze_dir,
            overwrite=args.overwrite,
        )
        print("[CNVSIM] frozen conditional-context validation")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        return
    if args.analysis_config is None:
        parser.error("--analysis-config is required for --stage validate")
    analysis_work_dir = (
        args.analysis_work_dir.expanduser().resolve()
        if args.analysis_work_dir is not None
        else atlas_dir / "work" / "analysis_v1" / "analysis"
    )
    diagnostic_dir = (
        args.diagnostic_dir.expanduser().resolve()
        if args.diagnostic_dir is not None
        else atlas_dir / "work" / "validation" / "diagnostic_concordance_v2"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else atlas_dir
        / "work"
        / "validation"
        / (
            "conditional_contexts_repeated_v1"
            if args.repeats > 1
            else "conditional_contexts_v1"
        )
    )
    config = load_analysis_config(args.analysis_config)
    validation_params = ConditionalContextValidationParams(
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
        minimum_test_samples=args.minimum_test_samples,
        minimum_rescue_eligible_samples=args.minimum_rescue_eligible_samples,
        minimum_assignment_coverage=args.minimum_assignment_coverage,
        minimum_conditional_accuracy=args.minimum_conditional_accuracy,
        minimum_operational_precision=args.minimum_operational_precision,
        maximum_core_control_acceptance=args.maximum_core_control_acceptance,
        core_controls_per_root=args.core_controls_per_root,
        minimum_platform_test_samples=args.minimum_platform_test_samples,
    )
    common = {
        "atlas_dir": atlas_dir,
        "analysis_work_dir": analysis_work_dir,
        "diagnostic_dir": diagnostic_dir,
        "metadata_path": args.metadata,
        "out_dir": out_dir,
        "projection_params": config.projection,
        "validation_params": validation_params,
        "overwrite": args.overwrite,
    }
    if args.repeats > 1:
        outputs = validate_conditional_contexts_repeated(
            **common,
            repeats=args.repeats,
            minimum_repeat_pass_rate=args.minimum_repeat_pass_rate,
        )
        label = "repeated conditional-context validation"
    else:
        outputs = validate_conditional_contexts(**common)
        label = "conditional-context validation"
    print(f"[CNVSIM] {label} complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
