from __future__ import annotations

"""Convert manually selected bycon analyses into CNVSIM projection inputs."""

import argparse
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bycon import prepare_projection_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a candidate table with selected=yes rows and create samples.tsv "
            "and segments.tsv for one or more analyses. CNVSIM overlap annotations "
            "are retained for audit and never prevent projection."
        )
    )
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selection_path = Path(args.selection).expanduser().resolve()
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    selection = pd.read_csv(selection_path, sep="\t", dtype=str, keep_default_na=False)
    outputs = prepare_projection_inputs(
        selection=selection,
        out_dir=Path(args.out_dir),
        overwrite=bool(args.overwrite),
    )
    samples = pd.read_csv(outputs["samples"], sep="\t")
    segments = pd.read_csv(outputs["segments"], sep="\t")
    audit = pd.read_csv(outputs["audit"], sep="\t")
    print("[CNVSIM] projection inputs prepared")
    print(f"  analyses: {len(samples):,}")
    print(f"  autosomal CNV segments: {len(segments):,}")
    skipped = int(audit["n_invalid_coordinate_segments_skipped"].sum())
    if skipped:
        print(f"  non-positive CNV intervals skipped: {skipped:,}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
