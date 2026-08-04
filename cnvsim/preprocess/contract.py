from __future__ import annotations

"""Validate the sample metadata columns consumed by preprocessing stages."""

from dataclasses import dataclass

import pandas as pd


ID_COL = "analysis_id"


@dataclass(frozen=True)
class MetaContractReport:
    """Result of checking required and optional metadata columns."""

    ok: bool
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    morphology_available: bool


def validate_meta_contract(
    meta: pd.DataFrame,
    *,
    require_topography: bool = True,
    require_platform_family: bool = True,
    require_morphology: bool = False,
) -> MetaContractReport:
    """Check whether a sample table contains the requested CNVSIM fields.

    ``analysis_id`` is always required because it links metadata to CNV
    features. Topography is represented either by a raw ICD-O-T code or by an
    already grouped root. Platform family is required when platform QC is
    enabled. Morphology is optional unless explicitly requested; stages that
    use morphology purity return an undefined metric when it is unavailable.

    Args:
        meta: One-row-per-analysis metadata table.
        require_topography: require `icdo_topography_id` OR `icdot_root_grouped`.
        require_platform_family: require `platform_family_refined`.
        require_morphology: require `icdo_morphology_id`.

    Returns:
        MetaContractReport
    """

    required = [ID_COL]

    if require_platform_family:
        required.append("platform_family_refined")

    if require_topography:
        # accept either raw topography or pre-mapped root group
        if ("icdo_topography_id" not in meta.columns) and ("icdot_root_grouped" not in meta.columns):
            # represent the OR-requirement as a pseudo-column for reporting
            missing_topo = ("icdo_topography_id|icdot_root_grouped",)
        else:
            missing_topo = tuple()
    else:
        missing_topo = tuple()

    missing_required = [c for c in required if c not in meta.columns]
    missing_required.extend(list(missing_topo))

    optional = ["icdo_morphology_id"]
    missing_optional = [c for c in optional if c not in meta.columns]

    morphology_available = ("icdo_morphology_id" in meta.columns) and (not meta["icdo_morphology_id"].isna().all())

    if require_morphology and (not morphology_available):
        # treat as required failure if explicitly required
        if "icdo_morphology_id" not in missing_required:
            missing_required.append("icdo_morphology_id")

    ok = len(missing_required) == 0

    return MetaContractReport(
        ok=bool(ok),
        missing_required=tuple(missing_required),
        missing_optional=tuple(missing_optional),
        morphology_available=bool(morphology_available),
    )


def assert_meta_contract(
    meta: pd.DataFrame,
    *,
    require_topography: bool = True,
    require_platform_family: bool = True,
    require_morphology: bool = False,
) -> MetaContractReport:
    """Return the contract report or raise ``ValueError`` for missing fields."""

    rep = validate_meta_contract(
        meta,
        require_topography=require_topography,
        require_platform_family=require_platform_family,
        require_morphology=require_morphology,
    )

    if not rep.ok:
        msg = (
            "Meta contract failed. Missing required columns: "
            f"{', '.join(rep.missing_required)}. "
            "(For single-platform cohorts set platform_family_refined='CohortPlatform'.)"
        )
        raise ValueError(msg)

    return rep
