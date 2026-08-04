from __future__ import annotations

"""Build the reusable sample-level chr44 profile table from raw CNV segments."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from cnvsim.analysis.chromosome_architecture import (
    build_chr44_profiles,
    normalize_segments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert whole-genome CNV segments into sample-level chr44 profiles."
    )
    parser.add_argument("--segments", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-analyses", type=int)
    args = parser.parse_args()

    segments_path = Path(args.segments).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(metadata_path)
    if "analysis_id" not in metadata.columns:
        raise KeyError("Metadata is missing analysis_id")
    analysis_ids = metadata["analysis_id"].astype(str).drop_duplicates().tolist()
    if args.expected_analyses is not None and len(analysis_ids) != int(
        args.expected_analyses
    ):
        raise ValueError(
            f"Expected {args.expected_analyses} analyses; found {len(analysis_ids)}"
        )

    raw_segments = pd.read_parquet(segments_path)
    segments, normalization_qc = normalize_segments(raw_segments)
    profiles = build_chr44_profiles(segments, analysis_ids=analysis_ids)
    if len(profiles) != len(analysis_ids):
        raise RuntimeError("chr44 profile row count does not match metadata")

    profile_path = out_dir / "sample_chr44_profiles.parquet"
    profiles.to_parquet(profile_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "genome_build": "GRCh38",
        "segments": str(segments_path),
        "metadata": str(metadata_path),
        "profile": str(profile_path),
        "n_analyses": int(len(profiles)),
        "n_features": int(len(profiles.columns) - 1),
        "normalization_qc": normalization_qc,
        "coverage_definition": "union segment coverage divided by chromosome length",
        "event_groups": ["gain", "loss"],
        "chromosomes": [str(index) for index in range(1, 23)],
    }
    manifest_path = out_dir / "sample_chr44_profiles.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[CNVSIM] chr44 profiles: {profile_path}")
    print(f"[CNVSIM] analyses: {len(profiles)}")


if __name__ == "__main__":
    main()
