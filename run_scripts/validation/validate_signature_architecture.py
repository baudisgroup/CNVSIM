from __future__ import annotations

"""Validate released Track 2 events against raw segment architecture."""

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.validation.signature_architecture import (
    SignatureArchitectureParams,
    freeze_signature_architecture,
    validate_signature_architecture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=["validate", "freeze"],
        default="validate",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--signature-events", type=Path)
    parser.add_argument("--reference-contexts", type=Path)
    parser.add_argument("--analysis-assignments", type=Path)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--gene-intervals", type=Path)
    parser.add_argument("--focal-fraction-chromosome", type=float, default=0.02)
    parser.add_argument("--broad-fraction-chromosome", type=float, default=0.50)
    parser.add_argument("--minimum-segment-carrier-fraction", type=float, default=0.05)
    parser.add_argument("--low-event-gene-coverage", type=float, default=0.10)
    parser.add_argument("--high-event-gene-coverage", type=float, default=0.20)
    parser.add_argument("--minimum-gene-coverage", type=float, default=0.50)
    parser.add_argument("--dominance-fraction", type=float, default=0.60)
    parser.add_argument("--mixed-fraction", type=float, default=0.25)
    parser.add_argument(
        "--max-negligible-raw-prevalence-delta",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-segment-encoding-abs-delta",
        type=float,
        default=0.02,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = args.atlas_dir.expanduser().resolve()
    if args.stage == "freeze":
        source_dir = (
            args.source_dir.expanduser().resolve()
            if args.source_dir is not None
            else atlas_dir / "work" / "validation" / "signature_architecture_v1"
        )
        outputs = freeze_signature_architecture(
            atlas_dir=atlas_dir,
            source_dir=source_dir,
            out_dir=args.freeze_dir,
            overwrite=args.overwrite,
        )
        print("[CNVSIM] frozen signature-architecture validation")
    else:
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else atlas_dir / "work" / "validation" / "signature_architecture_v1"
        )
        outputs = validate_signature_architecture(
            atlas_dir=atlas_dir,
            out_dir=out_dir,
            signatures_path=args.signature_events,
            references_path=args.reference_contexts,
            assignments_path=args.analysis_assignments,
            segments_path=args.segments,
            gene_intervals_path=args.gene_intervals,
            params=SignatureArchitectureParams(
                focal_fraction_chromosome=args.focal_fraction_chromosome,
                broad_fraction_chromosome=args.broad_fraction_chromosome,
                minimum_segment_carrier_fraction=(
                    args.minimum_segment_carrier_fraction
                ),
                low_event_gene_coverage=args.low_event_gene_coverage,
                high_event_gene_coverage=args.high_event_gene_coverage,
                minimum_gene_coverage=args.minimum_gene_coverage,
                dominance_fraction=args.dominance_fraction,
                mixed_fraction=args.mixed_fraction,
                max_negligible_raw_prevalence_delta=(
                    args.max_negligible_raw_prevalence_delta
                ),
                max_segment_encoding_abs_delta=(
                    args.max_segment_encoding_abs_delta
                ),
            ),
            overwrite=args.overwrite,
        )
        print("[CNVSIM] signature-architecture validation complete")
        summary = pd.read_csv(outputs["summary"], sep="\t", low_memory=False)
        overall = summary[
            summary["scope_type"].eq("all")
            & summary["scope_value"].eq("all")
        ]
        if not overall.empty:
            print("\n[CNVSIM] signature-architecture summary")
            print(overall.to_string(index=False))
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
