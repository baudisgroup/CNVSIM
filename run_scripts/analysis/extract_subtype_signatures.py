from __future__ import annotations

"""Extract chromosome-scale and gene-level evidence for CNV contexts."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.config import load_analysis_config
from cnvsim.analysis.signatures import extract_context_signatures


def _parse_roots(value: str | None) -> list[str] | None:
    if not value:
        return None
    roots = [root.strip() for root in value.split(",") if root.strip()]
    return roots or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    assignment_source = parser.add_mutually_exclusive_group(required=True)
    assignment_source.add_argument("--assignments-dir", type=Path)
    assignment_source.add_argument(
        "--reference-contexts-dir",
        type=Path,
        help=(
            "Reference-context output containing anchor_partitions and "
            "sample_reference_status.parquet. Signatures are defined by "
            "anchors while non-anchor samples remain in the same-root rest group."
        ),
    )
    parser.add_argument("--chr44-profiles", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--roots")
    parser.add_argument("--min-cluster-size", type=int)
    args = parser.parse_args()

    outputs = extract_context_signatures(
        source_dir=args.source_dir,
        release_dir=args.release_dir,
        assignments_dir=args.assignments_dir,
        reference_contexts_dir=args.reference_contexts_dir,
        chr44_profiles_path=args.chr44_profiles,
        out_dir=args.out_dir,
        config=load_analysis_config(args.analysis_config),
        roots=_parse_roots(args.roots),
        min_context_size=args.min_cluster_size,
    )
    print(f"[CNVSIM] context evidence: {args.out_dir.expanduser().resolve()}")
    print(f"  manifest: {outputs['manifest']}")


if __name__ == "__main__":
    main()
