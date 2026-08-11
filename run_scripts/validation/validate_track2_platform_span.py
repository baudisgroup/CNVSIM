from __future__ import annotations

"""Audit platform-associated segment-span variation in mixed Track 2 events."""

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.validation.track2_platform_span import (
    Track2PlatformSpanParams,
    validate_track2_platform_span,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--signature-events", type=Path)
    parser.add_argument("--reference-contexts", type=Path)
    parser.add_argument("--analysis-assignments", type=Path)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--gene-intervals", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--minimum-carriers-per-platform", type=int, default=10)
    parser.add_argument("--minimum-gene-coverage", type=float, default=0.50)
    parser.add_argument("--focal-fraction-chromosome", type=float, default=0.02)
    parser.add_argument("--broad-fraction-chromosome", type=float, default=0.50)
    parser.add_argument("--dominance-fraction", type=float, default=0.60)
    parser.add_argument("--mixed-fraction", type=float, default=0.25)
    parser.add_argument("--association-fdr", type=float, default=0.05)
    parser.add_argument("--minimum-epsilon-squared", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = args.atlas_dir.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else atlas_dir / "work" / "validation" / "track2_platform_span_v1"
    )
    outputs = validate_track2_platform_span(
        atlas_dir=atlas_dir,
        out_dir=out_dir,
        signatures_path=args.signature_events,
        references_path=args.reference_contexts,
        assignments_path=args.analysis_assignments,
        segments_path=args.segments,
        gene_intervals_path=args.gene_intervals,
        metadata_path=args.metadata,
        params=Track2PlatformSpanParams(
            minimum_carriers_per_platform=args.minimum_carriers_per_platform,
            minimum_gene_coverage=args.minimum_gene_coverage,
            focal_fraction_chromosome=args.focal_fraction_chromosome,
            broad_fraction_chromosome=args.broad_fraction_chromosome,
            dominance_fraction=args.dominance_fraction,
            mixed_fraction=args.mixed_fraction,
            association_fdr=args.association_fdr,
            minimum_epsilon_squared=args.minimum_epsilon_squared,
        ),
        overwrite=args.overwrite,
    )
    print("[CNVSIM] Track 2 platform-span validation complete")
    summary = pd.read_csv(outputs["summary"], sep="\t", low_memory=False)
    overall = summary[
        summary["scope_type"].eq("all") & summary["scope_value"].eq("all")
    ]
    if not overall.empty:
        print("\n[CNVSIM] Track 2 platform-span summary")
        print(overall.to_string(index=False))
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
