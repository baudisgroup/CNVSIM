from __future__ import annotations

"""Build stable reference anchors from refined CNV communities."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from cnvsim.analysis.config import load_analysis_config
from cnvsim.analysis.reference_contexts import build_reference_contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify refined communities by repeated-subsampling recovery and "
            "build anchor-only assignment tables."
        )
    )
    parser.add_argument("--refinement-dir", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--roots")
    args = parser.parse_args()

    refinement_dir = Path(args.refinement_dir).expanduser().resolve()
    stability_path = refinement_dir / "merged_context_stability.tsv"
    partitions_dir = refinement_dir / "final_partitions"
    for required in (stability_path, partitions_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    config = load_analysis_config(args.analysis_config)
    roots = (
        [root.strip() for root in args.roots.split(",") if root.strip()]
        if args.roots
        else None
    )
    outputs = build_reference_contexts(
        final_partitions_dir=partitions_dir,
        stability=pd.read_csv(stability_path, sep="\t"),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        params=config.reference_contexts,
        expected_repeats=config.subsampling.repeats,
        roots=roots,
    )
    print("[CNVSIM] reference-context classification complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
