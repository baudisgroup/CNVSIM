from __future__ import annotations

"""Adapt the frozen CNVSIM source package to the preprocessing contract."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cnvsim.config import PreprocessConfig
from cnvsim.preprocess.engine import run_preprocess_engine
from cnvsim.preprocess.roots import icdot_root_grouped


TUMOUR_METADATA = Path("metadata/tumour_analysis_metadata.parquet")
SOURCE_MANIFEST = Path("release_manifest.json")


def _load_source_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"CNVSIM source manifest not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _validate_tumour_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    required = {
        "analysis_id",
        "icdo_topography_id",
        "platform_family_refined",
        "source_group_refined",
    }
    missing = sorted(required - set(meta.columns))
    if missing:
        raise KeyError(f"Tumour metadata missing columns: {missing}")

    out = meta.copy()
    out["analysis_id"] = out["analysis_id"].astype(str)
    if out["analysis_id"].duplicated().any():
        examples = out.loc[
            out["analysis_id"].duplicated(keep=False), "analysis_id"
        ].unique()[:5]
        raise ValueError(
            "Tumour metadata must contain one row per analysis; "
            f"duplicates include {examples.tolist()}"
        )

    if "icdot_root_grouped" not in out.columns:
        out["icdot_root_grouped"] = out["icdo_topography_id"].apply(
            icdot_root_grouped
        )
    return out


def preprocess_progenetix_reference(
    *,
    cfg: PreprocessConfig,
    source_dir: Path,
    run_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Materialize ``meta_cleaned.tsv`` from a frozen CNVSIM source package.

    Platform harmonization, source assignment, tumour selection, and matching
    to the gene-panel cache are completed when ``source_file`` is built. This
    adapter deliberately does not repeat those release-preparation decisions.
    """

    source_dir = Path(source_dir).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    metadata_path = source_dir / TUMOUR_METADATA
    manifest_path = source_dir / SOURCE_MANIFEST

    if not metadata_path.exists():
        raise FileNotFoundError(f"Tumour metadata not found: {metadata_path}")
    if (run_dir / "meta_cleaned.tsv").exists() and not overwrite:
        raise FileExistsError(
            "Preprocessing metadata already exists; pass --rerun-adapter to "
            f"replace it: {run_dir / 'meta_cleaned.tsv'}"
        )

    source_manifest = _load_source_manifest(manifest_path)
    meta = _validate_tumour_metadata(pd.read_parquet(metadata_path))

    expected = source_manifest.get("counts", {}).get("tumour_analyses")
    if expected is not None and len(meta) != int(expected):
        raise ValueError(
            "Tumour metadata count does not match source manifest: "
            f"metadata={len(meta)}, manifest={expected}"
        )

    return run_preprocess_engine(
        meta_tumour=meta,
        run_dir=run_dir,
        inputs={
            "source_package": source_dir.name,
            "tumour_metadata": TUMOUR_METADATA.as_posix(),
            "source_manifest": SOURCE_MANIFEST.as_posix(),
            "source_schema_version": source_manifest.get("schema_version"),
        },
        extra_counts={
            "source_release_tumour_analyses": int(len(meta)),
        },
        extra_manifest={
            "run_name": cfg.run_name,
            "config": {
                "pipeline": asdict(cfg.pipeline),
                "filters": asdict(cfg.filters),
                "encoding": asdict(cfg.encoding),
            },
        },
    )
