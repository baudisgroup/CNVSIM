from __future__ import annotations

"""Write standardized tumour metadata and its provenance manifest."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cnvsim.preprocess.contract import assert_meta_contract, validate_meta_contract


ID_COL = "analysis_id"


def _nunique(df: pd.DataFrame, col: str = ID_COL) -> int:
    return int(df[col].astype(str).nunique()) if (df is not None and col in df.columns) else 0


def run_preprocess_engine(
    *,
    meta_tumour: pd.DataFrame,
    run_dir: Path,
    inputs: dict[str, Any] | None = None,
    extra_counts: dict[str, Any] | None = None,
    extra_breakdown: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
    require_platform_family: bool = True,
) -> Path:
    """Materialize the metadata input consumed by feature encoding.

    A dataset adapter must first select tumour analyses and assign any
    dataset-specific fields such as source group, platform family, and grouped
    topography. This function validates that table, writes
    ``meta_cleaned.tsv``, and records input provenance, sample counts, and
    metadata composition in ``run_manifest.json``.

    Args:
        meta_tumour: Tumour-only table with one row per ``analysis_id``.
        run_dir: Output directory (will be created).
        inputs: Optional dict with provenance (paths, versions).
        extra_counts: Optional dict merged into manifest['counts'].
        extra_breakdown: Optional dict merged into manifest['breakdown'].
        extra_manifest: Optional top-level manifest fields supplied by the
            dataset adapter.
        require_platform_family: Whether ``platform_family_refined`` must be
            present before writing outputs.

    Returns:
        Path to run_dir.
    """

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # strict contract validation (with clear error messages)
    assert_meta_contract(
        meta_tumour,
        require_topography=False,  # engine is generic; topography/root may be added by adapters
        require_platform_family=bool(require_platform_family),
        require_morphology=False,
    )

    meta_out = run_dir / "meta_cleaned.tsv"
    meta_tumour.to_csv(meta_out, sep="\t", index=False)

    contract = validate_meta_contract(
        meta_tumour,
        require_topography=False,
        require_platform_family=bool(require_platform_family),
        require_morphology=False,
    )

    # default breakdown if columns exist
    breakdown: dict[str, Any] = {}
    for col in ["source_group_refined", "platform_family_refined", "icdot_root_grouped"]:
        if col in meta_tumour.columns:
            counts = (
                meta_tumour[[ID_COL, col]]
                .drop_duplicates(subset=[ID_COL])
                .groupby(col)[ID_COL]
                .nunique()
                .sort_values(ascending=False)
                .to_dict()
            )
            breakdown[col] = {str(k): int(v) for k, v in counts.items()}

    if extra_breakdown:
        breakdown.update(extra_breakdown)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs or {},
        "counts": {
            "tumour_analyses": _nunique(meta_tumour),
            **(extra_counts or {}),
        },
        "contract": {
            "meta": {
                "required_missing": list(contract.missing_required),
                "optional_missing": list(contract.missing_optional),
                "morphology_available": bool(contract.morphology_available),
            }
        },
        "breakdown": breakdown,
        "outputs": {
            "run_dir": str(run_dir),
            "meta_cleaned": str(meta_out),
        },
    }

    if extra_manifest:
        # shallow merge: caller can add e.g. 'config', extra output pointers, etc.
        manifest.update(extra_manifest)

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    return run_dir
