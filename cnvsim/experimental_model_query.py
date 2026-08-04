from __future__ import annotations

"""Read-only queries over the public CNVSIM cell-line resource."""

import json
from pathlib import Path

import pandas as pd


CELL_LINE_FILES = {
    "profiles": "model_projection_profiles.parquet",
    "models": "cell_lines.tsv",
    "availability": "subtype_model_availability.tsv",
}

EVIDENCE_LABELS = {
    "replicate_supported": "Replicate-supported",
    "single_analysis_supported": "Single-profile assignment",
    "provisional_consensus": "Provisional consensus",
    "discordant_across_analyses": "Discordant profiles",
    "not_projected": "No confident match",
}

STATUS_EXPLANATIONS = {
    "projected_consensus": "Model profiles support a common CNVSIM reference.",
    "ambiguous": "Competing subtype votes prevented a confident assignment.",
    "unresolved": "The nearest subtype was outside calibrated distance limits.",
    "no_autosomal_gene_cnv_signal": (
        "No autosomal cancer-gene CNV signal was available for projection."
    ),
}

CONTEXT_RESULT_COLUMNS = [
    "model_rank",
    "model_id",
    "model_label",
    "root",
    "assigned_reference_key",
    "reference_level",
    "recommendation_tier",
    "evidence_label",
    "n_profiles",
    "assignment_consistency",
    "assignment_coverage",
    "calibrated_distance_ratio",
    "representative_analysis_id",
]

MODEL_RESULT_COLUMNS = [
    "model_id",
    "model_label",
    "root",
    "model_assignment_status",
    "assigned_reference_key",
    "reference_level",
    "recommendation_tier",
    "evidence_label",
    "n_profiles",
    "n_reference_assigned",
    "n_consensus_support",
    "assignment_consistency",
    "assignment_coverage",
    "calibrated_distance_ratio",
    "representative_analysis_id",
    "multi_root_model",
    "status_explanation",
]


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def load_experimental_model_application(
    application_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load and validate the public cell-line query tables."""
    application_dir = Path(application_dir).expanduser().resolve()
    if not application_dir.is_dir():
        raise FileNotFoundError(application_dir)

    manifest_path = application_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "cnvsim_public_cell_line_v1":
        raise ValueError("Unsupported CNVSIM cell-line resource schema")

    paths = {
        name: application_dir / filename
        for name, filename in CELL_LINE_FILES.items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CNVSIM cell-line resource is incomplete: " + ", ".join(missing)
        )

    tables = {
        "profiles": pd.read_parquet(paths["profiles"]),
        "models": pd.read_csv(paths["models"], sep="\t"),
        "availability": pd.read_csv(paths["availability"], sep="\t"),
    }
    _require_columns(
        tables["profiles"],
        {
            "analysis_id",
            "model_id",
            "root",
            "platform_family",
            "assignment_status",
            "assignment_layer",
            "assigned_reference_key",
            "descriptive_candidate_reference_key",
            "candidate_interpretation",
            "status_reason",
        },
        label="model projection profiles",
    )
    _require_columns(
        tables["models"],
        {
            "model_id",
            "model_label",
            "root",
            "n_profiles",
            "assigned_reference_key",
            "reference_layer",
            "assignment_status",
            "evidence_status",
            "recommendation_tier",
            "is_recommended",
            "model_rank",
        },
        label="cell lines",
    )
    _require_columns(
        tables["availability"],
        {"reference_key", "root", "reference_layer", "coverage_status"},
        label="subtype model availability",
    )
    return tables


def _with_display_fields(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    evidence = output["evidence_status"].astype(str)
    output["evidence_label"] = evidence.map(EVIDENCE_LABELS).fillna(evidence)
    return output


def query_models_for_context(
    application_dir: str | Path,
    reference_key: str,
    *,
    top_n: int = 5,
    recommended_only: bool = True,
) -> pd.DataFrame:
    """Return ranked Cellosaurus models for one CNVSIM subtype."""
    if int(top_n) < 1:
        raise ValueError("top_n must be at least 1")

    tables = load_experimental_model_application(application_dir)
    availability = tables["availability"]
    matched_reference = availability[
        availability["reference_key"].astype(str).eq(str(reference_key))
    ]
    if matched_reference.empty:
        raise KeyError(f"Unknown CNVSIM reference_key: {reference_key}")

    result = tables["models"]
    result = result[
        result["assigned_reference_key"].astype(str).eq(str(reference_key))
    ].copy()
    if recommended_only:
        result = result[result["is_recommended"].astype("boolean").fillna(False)]
    result = _with_display_fields(result)
    result = result.sort_values(["model_rank", "model_id"]).head(int(top_n))
    result = result.rename(
        columns={
            "reference_layer": "reference_level",
            "distance_to_limit": "calibrated_distance_ratio",
        }
    )
    return result.reindex(columns=CONTEXT_RESULT_COLUMNS).reset_index(drop=True)


def _canonical_model_id(
    model_ids: pd.Series,
    requested_model_id: str,
) -> str:
    available = model_ids.dropna().astype(str).drop_duplicates()
    requested = str(requested_model_id).strip()
    lookup = {value.casefold(): value for value in available}
    candidates = [requested]
    if requested.upper().startswith("CVCL_"):
        candidates.append(f"cellosaurus:{requested.upper()}")
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise KeyError(f"Unknown cell-line model_id: {requested_model_id}")


def query_context_for_model(
    application_dir: str | Path,
    model_id: str,
    *,
    root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return model-level matches and their profile-level audit rows."""
    tables = load_experimental_model_application(application_dir)
    models = tables["models"]
    canonical_id = _canonical_model_id(models["model_id"], model_id)
    summary = models[models["model_id"].astype(str).eq(canonical_id)].copy()
    if root is not None:
        summary = summary[summary["root"].astype(str).eq(str(root))].copy()
        if summary.empty:
            raise KeyError(
                f"Model {canonical_id} has no model-root group for {root}"
            )

    summary = _with_display_fields(summary)
    summary["status_explanation"] = (
        summary["assignment_status"]
        .astype(str)
        .map(STATUS_EXPLANATIONS)
        .fillna("See profile-level projection details.")
    )
    summary = summary.rename(
        columns={
            "assignment_status": "model_assignment_status",
            "reference_layer": "reference_level",
            "distance_to_limit": "calibrated_distance_ratio",
        }
    )
    summary = summary.reindex(columns=MODEL_RESULT_COLUMNS).sort_values("root")

    profiles = tables["profiles"]
    profiles = profiles[profiles["model_id"].astype(str).eq(canonical_id)].copy()
    if root is not None:
        profiles = profiles[profiles["root"].astype(str).eq(str(root))].copy()
    profiles = profiles.rename(
        columns={
            "platform_family": "platform_family_refined",
            "vote_fraction": "top_vote_fraction",
            "distance_limit": "max_mean_neighbor_distance",
        }
    )
    detail_columns = [
        "analysis_id",
        "model_id",
        "model_label",
        "root",
        "platform_family_refined",
        "assignment_status",
        "assignment_layer",
        "assigned_reference_key",
        "descriptive_candidate_reference_key",
        "candidate_interpretation",
        "status_reason",
        "top_vote_fraction",
        "vote_margin",
        "mean_neighbor_distance",
        "max_mean_neighbor_distance",
        "distance_to_limit",
        "primary_assignment_status",
        "conditional_assignment_status",
    ]
    profiles = profiles.reindex(columns=detail_columns).sort_values(
        ["root", "analysis_id"]
    )
    return summary.reset_index(drop=True), profiles.reset_index(drop=True)
