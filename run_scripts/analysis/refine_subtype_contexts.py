from __future__ import annotations

"""Merge unsupported initial communities and validate the refined contexts."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from cnvsim.analysis.config import load_analysis_config
from cnvsim.analysis.context_refinement import refine_contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine selected Leiden communities using stability and signature evidence."
    )
    parser.add_argument("--parameter-dir", required=True)
    parser.add_argument("--provisional-partitions-dir", required=True)
    parser.add_argument("--provisional-signatures-dir", required=True)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--roots")
    args = parser.parse_args()

    parameter_dir = Path(args.parameter_dir).expanduser().resolve()
    signature_dir = Path(args.provisional_signatures_dir).expanduser().resolve()
    config = load_analysis_config(args.analysis_config)
    roots = (
        [root.strip() for root in args.roots.split(",") if root.strip()]
        if args.roots
        else None
    )

    required = {
        "selected_parameters": parameter_dir / "selected_parameters.tsv",
        "subsampling_runs": parameter_dir / "subsampling_runs.tsv",
        "subsampling_contexts": parameter_dir / "subsampling_by_subtype.tsv",
        "overlap_cells": parameter_dir / "subsampling_overlap_cells.tsv",
        "context_diagnostics": parameter_dir / "selected_context_diagnostics.tsv",
        "track1_profiles": signature_dir / "track1_context_profiles.parquet",
        "track1_supported": signature_dir / "track1_supported_events.parquet",
        "track2_stats": signature_dir / "track2_marker_stats.parquet",
        "track2_strict": signature_dir / "track2_strict_evidence.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing refinement inputs: {missing}")

    outputs = refine_contexts(
        selected_parameters=pd.read_csv(required["selected_parameters"], sep="\t"),
        subsampling_runs=pd.read_csv(required["subsampling_runs"], sep="\t"),
        subsampling_contexts=pd.read_csv(required["subsampling_contexts"], sep="\t"),
        overlap_cells=pd.read_csv(required["overlap_cells"], sep="\t"),
        provisional_partitions_dir=Path(args.provisional_partitions_dir).expanduser().resolve(),
        context_diagnostics=pd.read_csv(required["context_diagnostics"], sep="\t"),
        track1_context_profiles=pd.read_parquet(required["track1_profiles"]),
        track1_supported_events=pd.read_parquet(required["track1_supported"]),
        track2_marker_stats=pd.read_parquet(required["track2_stats"]),
        track2_strict_evidence=pd.read_parquet(required["track2_strict"]),
        ir_dir=Path(args.ir_dir).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        refinement=config.refinement,
        subsampling=config.subsampling,
        roots=roots,
    )
    print("[CNVSIM] context refinement complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
