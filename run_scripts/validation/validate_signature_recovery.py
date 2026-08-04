from __future__ import annotations

"""Validate fixed-parameter recovery of released core CNV signatures."""

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

native_threads = os.environ.get("CNVSIM_NATIVE_THREADS", "1")
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[variable] = native_threads

import pandas as pd

from cnvsim.analysis.config import load_analysis_config
from cnvsim.validation.config import load_validation_config
from cnvsim.validation.signature_recovery import (
    extract_fold_signatures,
    rebuild_fixed_parameter_partitions,
    resolve_signature_recovery_inputs,
    summarize_signature_recovery,
)
from cnvsim.validation.signature_recovery_release import (
    freeze_signature_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument(
        "--analysis-config",
        default="configs/analysis_v1.yaml",
        type=Path,
    )
    parser.add_argument(
        "--validation-config",
        default="configs/validation_v1.yaml",
        type=Path,
    )
    parser.add_argument(
        "--stage",
        choices=["rebuild", "signatures", "summarize", "freeze", "all"],
        default="all",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = args.atlas_dir.expanduser().resolve()
    if args.stage == "freeze":
        source_dir = (
            args.source_dir.expanduser().resolve()
            if args.source_dir is not None
            else (
                atlas_dir
                / "work"
                / "validation"
                / "signature_recovery_v1"
            )
        )
        outputs = freeze_signature_recovery(
            atlas_dir=atlas_dir,
            source_dir=source_dir,
            out_dir=args.freeze_dir,
            overwrite=args.overwrite,
        )
        print("[CNVSIM] frozen five-fold signature recovery")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        return

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (
            atlas_dir
            / "work"
            / "validation"
            / "signature_recovery_v1"
        )
    )
    inputs = resolve_signature_recovery_inputs(
        atlas_dir=atlas_dir,
        analysis_dir=args.analysis_dir,
    )
    analysis_config = load_analysis_config(args.analysis_config)
    params = load_validation_config(
        args.validation_config
    ).signature_recovery

    outputs: dict[str, Path] = {}
    if args.stage in {"rebuild", "all"}:
        outputs.update(
            rebuild_fixed_parameter_partitions(
                inputs=inputs,
                out_dir=out_dir,
                analysis_config=analysis_config,
                params=params,
                overwrite=args.overwrite,
            )
        )
    if args.stage in {"signatures", "all"}:
        outputs.update(
            extract_fold_signatures(
                inputs=inputs,
                out_dir=out_dir,
                analysis_config=analysis_config,
                params=params,
                overwrite=args.overwrite,
            )
        )
    if args.stage in {"summarize", "all"}:
        outputs.update(
            summarize_signature_recovery(
                inputs=inputs,
                out_dir=out_dir,
                params=params,
            )
        )

    summary_path = out_dir / "signature_recovery_summary.tsv"
    if summary_path.exists():
        print("\n[CNVSIM] five-fold signature recovery summary")
        print(
            pd.read_csv(summary_path, sep="\t").to_string(index=False)
        )
    print("\n[CNVSIM] five-fold signature recovery complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
