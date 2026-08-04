from __future__ import annotations

"""Project the frozen experimental-model cohort into CNVSIM references."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.experimental_models import project_experimental_models
from cnvsim.preprocess.gene_panel_store import resolve_gene_fraction_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project frozen experimental models into the CNVSIM atlas."
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument("--bundle-dir")
    parser.add_argument("--metadata")
    parser.add_argument(
        "--gene-fraction-store",
        "--gene-panel-cache",
        dest="gene_fraction_store",
        help=(
            "Compact public gene-CNV store or legacy internal cache. "
            "The old --gene-panel-cache name remains an alias."
        ),
    )
    parser.add_argument("--model-map")
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--reuse-analysis-projection",
        action="store_true",
        help=(
            "Reuse the existing projection/ outputs and rebuild only model-level "
            "summaries. Requires --overwrite."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    outputs = project_experimental_models(
        bundle_dir=(
            Path(args.bundle_dir).expanduser().resolve()
            if args.bundle_dir
            else atlas_dir / "projection_bundle_candidate"
        ),
        metadata_path=(
            Path(args.metadata).expanduser().resolve()
            if args.metadata
            else atlas_dir
            / "source_file"
            / "metadata"
            / "experimental_model_metadata.parquet"
        ),
        gene_fraction_store=resolve_gene_fraction_store(
            Path(args.gene_fraction_store).expanduser().resolve()
            if args.gene_fraction_store
            else atlas_dir
        ),
        model_map_path=(
            Path(args.model_map).expanduser().resolve()
            if args.model_map
            else None
        ),
        reuse_analysis_projection=bool(args.reuse_analysis_projection),
        out_dir=(
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else atlas_dir
            / "work"
            / "applications"
            / "experimental_model_projection_v1"
        ),
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] experimental-model projection complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
