#!/usr/bin/env python3
from __future__ import annotations

"""Run CNVSIM preprocessing from a frozen ``source_file`` package."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.config import load_preprocess_config
from cnvsim.preprocess.gene_panel_store import resolve_gene_fraction_store


def _parse_roots(value: str | None) -> list[str] | None:
    if not value:
        return None
    roots = [root.strip() for root in value.split(",") if root.strip()]
    return roots or None


def _resolve_source_dir(path: Path) -> Path:
    """Accept a release root, source_data, or source_file directory."""

    path = path.expanduser().resolve()
    candidates = [
        path,
        path / "source_file",
        path / "source_data" / "source_file",
    ]
    for candidate in candidates:
        if (candidate / "metadata" / "tumour_analysis_metadata.parquet").exists():
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "No CNVSIM source_file package was found. Checked:\n  " + checked
    )


def _required_source_paths(source_dir: Path) -> dict[str, Path]:
    return {
        "tumour metadata": (
            source_dir / "metadata" / "tumour_analysis_metadata.parquet"
        ),
        "source manifest": source_dir / "release_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build root-specific CNV comparison spaces from a CNVSIM "
            "source_file package"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "preprocess_baseline.yaml",
        help="Scientific settings YAML (default: configs/preprocess_baseline.yaml)",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Path to the source-data package used for preprocessing",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Directory for intermediate preprocessing outputs",
    )
    parser.add_argument(
        "--release-dir",
        required=True,
        type=Path,
        help="Directory for the exported preprocessing reference space",
    )
    parser.add_argument(
        "--roots",
        help="Optional comma-separated grouped ICD-O-T roots for a partial run",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Optional run-directory name override; useful for a pilot that "
            "must not replace the configured baseline run"
        ),
    )
    parser.add_argument(
        "--min-altered-genes-per-sample",
        type=int,
        help=(
            "Optional encoding override applied after stable-gene filtering"
        ),
    )
    parser.add_argument(
        "--rerun-adapter",
        action="store_true",
        help="Rebuild meta_cleaned.tsv from the frozen source metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement where a stage requires explicit permission",
    )
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    if args.run_name is not None:
        run_name = str(args.run_name).strip()
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("--run-name must be one non-empty path component")
        cfg = replace(cfg, run_name=run_name)
    if args.min_altered_genes_per_sample is not None:
        minimum = int(args.min_altered_genes_per_sample)
        if minimum < 0:
            raise ValueError("--min-altered-genes-per-sample must be >= 0")
        cfg = replace(
            cfg,
            encoding=replace(
                cfg.encoding,
                min_altered_genes_per_sample=minimum,
            ),
        )
    source_dir = _resolve_source_dir(args.source_dir)
    run_dir = (args.work_dir.expanduser().resolve() / cfg.run_name)
    release_dir = args.release_dir.expanduser().resolve()
    roots = _parse_roots(args.roots)

    missing = [
        f"{label}: {path}"
        for label, path in _required_source_paths(source_dir).items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The source package is incomplete:\n  " + "\n  ".join(missing)
        )

    from cnvsim.preprocess.export_reference_space import export_reference_space
    from cnvsim.preprocess.pipeline import (
        materialize_final_space_from_stage2a,
        run_encode,
        run_firstpass_stage1,
        run_stage2a_all,
    )
    from cnvsim.progenetix_reference.preprocess_reference import (
        preprocess_progenetix_reference,
    )

    meta_path = run_dir / "meta_cleaned.tsv"
    if args.rerun_adapter or not meta_path.exists():
        print("\n[CNVSIM] metadata adapter", flush=True)
        preprocess_progenetix_reference(
            cfg=cfg,
            source_dir=source_dir,
            run_dir=run_dir,
            overwrite=bool(args.rerun_adapter),
        )

    gene_fraction_store = resolve_gene_fraction_store(source_dir)

    print("\n[CNVSIM] encode gene-level CNV features", flush=True)
    run_encode(
        cfg=cfg,
        run_dir=run_dir,
        gene_fraction_store=gene_fraction_store,
        roots=roots,
    )

    print("\n[CNVSIM] filter likely technical groups", flush=True)
    run_firstpass_stage1(cfg=cfg, run_dir=run_dir, roots=roots)

    print("\n[CNVSIM] construct root-specific comparison spaces", flush=True)
    run_stage2a_all(cfg=cfg, run_dir=run_dir, roots=roots)

    materialize_final_space_from_stage2a(
        cfg=cfg, run_dir=run_dir, roots=roots
    )

    print("\n[CNVSIM] export preprocessing reference space", flush=True)
    export_reference_space(
        cfg=cfg,
        run_dir=run_dir,
        release_dir=release_dir,
        roots=roots,
        include_final_space=True,
    )

    print("\n[CNVSIM] complete")
    print(f"  mode: {cfg.pipeline.mode}")
    print(f"  run directory: {run_dir}")
    print(f"  release directory: {release_dir}")


if __name__ == "__main__":
    main()
