from __future__ import annotations

"""Project frozen Progenetix experimental models into CNVSIM references."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cnvsim.preprocess.gene_panel_store import load_gene_panel_subset_from_store
from cnvsim.projection import EVENT_COLUMNS, project_gene_profiles


MODEL_CONSISTENCY_THRESHOLD = 0.60
MODEL_ASSIGNMENT_COVERAGE_THRESHOLD = 0.50
MODEL_MIN_REPLICATE_SUPPORT = 2
RECOMMENDED_MODEL_EVIDENCE = {
    "replicate_supported",
    "single_analysis_supported",
}
MODEL_EVIDENCE_ORDER = {
    "replicate_supported": 0,
    "single_analysis_supported": 1,
    "provisional_consensus": 2,
    "discordant_across_analyses": 3,
    "not_projected": 4,
}
CATALOG_MODEL_GROUPING_BASIS = "external_model_map"
CATALOG_REFERENCE_LEVELS = {"subtype", "conditional"}
CATALOG_EXCLUSION_DESCRIPTIONS = {
    "unverified_model_identity": (
        "No curated external model mapping is available; a Progenetix "
        "individual_id fallback is not accepted as a public model identity."
    ),
    "root_only_reference": (
        "The analysis belongs to a descriptive root-only reference and is not "
        "eligible for subtype or conditional model recommendations."
    ),
    "out_of_scope_root": (
        "The analysis belongs to a root outside the public experimental-model "
        "projection catalogue."
    ),
}
FROZEN_APPLICATION_FILES = {
    "analysis_assignments": "analysis_assignments.parquet",
    "model_assignments": "model_assignments.tsv",
    "context_model_candidates": "context_model_candidates.tsv",
    "context_model_availability": "context_model_availability.tsv",
    "root_summary": "root_summary.tsv",
}


def _annotate_catalog_eligibility(
    frame: pd.DataFrame,
    *,
    eligible_roots: set[str],
    root_only_roots: set[str],
) -> pd.DataFrame:
    """Mark rows eligible for the public Cellosaurus model catalogue."""

    output = frame.copy()
    grouping_basis = output.get(
        "model_grouping_basis",
        pd.Series(pd.NA, index=output.index, dtype="object"),
    ).astype(str)
    roots = output["root"].astype(str)
    externally_mapped = grouping_basis.eq(CATALOG_MODEL_GROUPING_BASIS)
    in_catalog_root = roots.isin(eligible_roots)
    output["catalog_eligible"] = externally_mapped & in_catalog_root
    output["catalog_exclusion_reason"] = np.select(
        [
            output["catalog_eligible"],
            ~externally_mapped,
            roots.isin(root_only_roots),
        ],
        [
            "",
            "unverified_model_identity",
            "root_only_reference",
        ],
        default="out_of_scope_root",
    )
    return output


def _summarize_catalog_exclusions(
    analysis: pd.DataFrame,
    models: pd.DataFrame,
) -> dict[str, object]:
    """Describe every class excluded from the public model catalogue."""

    excluded_analysis = analysis.loc[~analysis["catalog_eligible"]].copy()
    excluded_models = models.loc[~models["catalog_eligible"]].copy()
    by_reason: dict[str, object] = {}
    for reason in sorted(excluded_analysis["catalog_exclusion_reason"].unique()):
        reason_analysis = excluded_analysis.loc[
            excluded_analysis["catalog_exclusion_reason"].eq(reason)
        ]
        reason_models = excluded_models.loc[
            excluded_models["catalog_exclusion_reason"].eq(reason)
        ]
        entry: dict[str, object] = {
            "description": CATALOG_EXCLUSION_DESCRIPTIONS.get(
                str(reason), "Excluded by the public model catalogue policy."
            ),
            "n_analyses": int(len(reason_analysis)),
            "n_distinct_model_ids": int(reason_analysis["model_id"].nunique()),
            "n_model_root_groups": int(len(reason_models)),
            "roots": {
                str(root): int(count)
                for root, count in reason_analysis["root"].value_counts().items()
            },
        }
        if reason == "unverified_model_identity":
            entry["analysis_ids"] = sorted(
                reason_analysis["analysis_id"].astype(str).tolist()
            )
            entry["provisional_model_ids"] = sorted(
                reason_analysis["model_id"].astype(str).unique().tolist()
            )
        by_reason[str(reason)] = entry

    return {
        "n_excluded_analyses": int(len(excluded_analysis)),
        "n_distinct_model_ids_in_excluded_rows": int(
            excluded_analysis["model_id"].nunique()
        ),
        "n_excluded_model_root_groups": int(len(excluded_models)),
        "excluded_from": [
            "context_model_candidates",
            "context_model_availability",
            "experimental_model_query",
        ],
        "retained_for_audit_in": [
            "analysis_assignments.parquet",
            "model_assignments.tsv",
            "root_summary.tsv",
        ],
        "audit_columns": ["catalog_eligible", "catalog_exclusion_reason"],
        "by_reason": by_reason,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_model_map(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator, dtype=str)
    if "analysis_id" not in frame.columns:
        raise KeyError("Model map must contain analysis_id")
    id_columns = [
        column
        for column in ("model_id", "cellosaurus_id", "group_cellosaurus_id")
        if column in frame.columns
    ]
    if not id_columns:
        raise KeyError(
            "Model map must contain model_id, cellosaurus_id, or "
            "group_cellosaurus_id"
        )
    model_id_column = id_columns[0]
    label_columns = [
        column
        for column in ("model_label", "cellosaurus_label")
        if column in frame.columns
    ]
    keep = ["analysis_id", model_id_column, *label_columns]
    frame = frame[keep].rename(columns={model_id_column: "mapped_model_id"})
    if label_columns:
        frame = frame.rename(columns={label_columns[0]: "mapped_model_label"})
    else:
        frame["mapped_model_label"] = pd.NA
    frame["analysis_id"] = frame["analysis_id"].astype(str)
    if frame["analysis_id"].duplicated().any():
        raise ValueError("Model map contains duplicate analysis_id values")
    return frame


def _load_bundle_references(bundle_dir: Path) -> pd.DataFrame:
    catalog_columns = [
        "reference_key",
        "root",
        "reference_level",
        "projection_mode",
        "n_definition_members",
        "n_projected_members",
        "n_total_assigned",
        "evidence_class",
        "n_track1_events",
        "n_track2_events",
    ]
    primary_path = bundle_dir / "reference_contexts.tsv"
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    primary = pd.read_csv(primary_path, sep="\t")
    required = {"reference_key", "root"}
    missing = sorted(required - set(primary.columns))
    if missing:
        raise KeyError(f"Projection references are missing columns: {missing}")
    primary = primary.copy()
    if "reference_level" not in primary.columns:
        primary["reference_level"] = np.where(
            primary["reference_key"].astype(str).str.endswith("-ROOT"),
            "root",
            "subtype",
        )
    for column in catalog_columns:
        if column not in primary.columns:
            primary[column] = pd.NA
    primary = primary[catalog_columns]

    conditional_path = (
        bundle_dir / "conditional" / "conditional_contexts.tsv"
    )
    tables = [primary]
    if conditional_path.exists():
        conditional = pd.read_csv(conditional_path, sep="\t")
        key_column = (
            "reference_key"
            if "reference_key" in conditional.columns
            else "conditional_reference_key"
        )
        missing = sorted({"root", key_column} - set(conditional.columns))
        if missing:
            raise KeyError(
                f"Conditional projection references are missing columns: {missing}"
            )
        definition_counts = pd.to_numeric(
            conditional.get(
                "n_full_members",
                pd.Series(0, index=conditional.index),
            ),
            errors="coerce",
        ).fillna(0).astype(int)
        tables.append(
            pd.DataFrame(
                {
                    "reference_key": conditional[key_column].astype(str),
                    "root": conditional["root"].astype(str),
                    "reference_level": "conditional",
                    "projection_mode": "conditional_knn",
                    "n_definition_members": definition_counts,
                    "n_projected_members": 0,
                    "n_total_assigned": definition_counts,
                    "evidence_class": conditional.get(
                        "evidence_class",
                        pd.Series(pd.NA, index=conditional.index),
                    ),
                    "n_track1_events": conditional.get(
                        "n_track1_events",
                        pd.Series(pd.NA, index=conditional.index),
                    ),
                    "n_track2_events": conditional.get(
                        "n_track2_markers",
                        pd.Series(pd.NA, index=conditional.index),
                    ),
                }
            )
        )

    references = pd.concat(tables, ignore_index=True, sort=False)
    references["reference_key"] = references["reference_key"].astype(str)
    references["root"] = references["root"].astype(str)
    if references["reference_key"].duplicated().any():
        duplicated = references.loc[
            references["reference_key"].duplicated(keep=False),
            "reference_key",
        ].tolist()
        raise ValueError(f"Projection bundle contains duplicate references: {duplicated}")
    return references


def _use_effective_projection_metrics(assignments: pd.DataFrame) -> pd.DataFrame:
    """Expose the metrics from the layer that made the final assignment."""

    output = assignments.copy()
    conditional = output.get(
        "assignment_layer",
        pd.Series("", index=output.index, dtype="string"),
    ).astype(str).eq("conditional")
    metric_columns = {
        "top_vote_fraction": "conditional_top_vote_fraction",
        "vote_margin": "conditional_vote_margin",
        "mean_neighbor_distance": "conditional_mean_neighbor_distance",
        "max_mean_neighbor_distance": "conditional_max_mean_neighbor_distance",
    }
    for target, source in metric_columns.items():
        if source not in output.columns:
            continue
        if target not in output.columns:
            output[target] = np.nan
        output.loc[conditional, target] = pd.to_numeric(
            output.loc[conditional, source], errors="coerce"
        )
    return output


def _attach_model_ids(
    metadata: pd.DataFrame,
    model_map: pd.DataFrame,
) -> pd.DataFrame:
    output = metadata.copy()
    if model_map.empty:
        output["mapped_model_id"] = pd.NA
        output["mapped_model_label"] = pd.NA
    else:
        output = output.merge(
            model_map,
            on="analysis_id",
            how="left",
            validate="one_to_one",
        )
    fallback = output["individual_id"].fillna(output["analysis_id"]).astype(str)
    mapped = output["mapped_model_id"].notna() & output[
        "mapped_model_id"
    ].astype(str).str.strip().ne("")
    output["model_id"] = output["mapped_model_id"].where(mapped, fallback)
    output["model_grouping_basis"] = np.where(
        mapped,
        "external_model_map",
        "progenetix_individual_id",
    )
    default_label = output.get(
        "analysis_label", pd.Series(pd.NA, index=output.index)
    )
    output["model_label"] = output["mapped_model_label"].where(
        output["mapped_model_label"].notna(), default_label
    )
    return output.drop(columns=["mapped_model_id", "mapped_model_label"])


def _deterministic_mode(values: pd.Series) -> object:
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return pd.NA
    counts = cleaned.value_counts()
    maximum = int(counts.max())
    return sorted(counts[counts.eq(maximum)].index.astype(str))[0]


def _classify_model_evidence(
    *,
    n_analyses: int,
    n_reference_assigned: int,
    n_consensus_support: int,
    assignment_consistency: float,
    assignment_coverage: float,
) -> str:
    if n_consensus_support <= 0 or n_reference_assigned <= 0:
        return "not_projected"
    if n_analyses == 1:
        return "single_analysis_supported"
    if (
        n_consensus_support >= MODEL_MIN_REPLICATE_SUPPORT
        and assignment_consistency >= MODEL_CONSISTENCY_THRESHOLD
        and assignment_coverage >= MODEL_ASSIGNMENT_COVERAGE_THRESHOLD
    ):
        return "replicate_supported"
    if assignment_consistency > 0.5:
        return "provisional_consensus"
    return "discordant_across_analyses"


def _summarize_models(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assigned_statuses = {"projected_assignment", "root_only_reference"}
    for (model_id, root), group in analysis.groupby(
        ["model_id", "root"], sort=True, dropna=False
    ):
        assigned = group[
            group["assignment_status"].astype(str).isin(assigned_statuses)
            & group["assigned_reference_key"].notna()
        ].copy()
        consensus = _deterministic_mode(assigned["assigned_reference_key"])
        if pd.isna(consensus):
            model_status = str(_deterministic_mode(group["assignment_status"]))
            supporting = assigned
            consistency = np.nan
        else:
            supporting = assigned[
                assigned["assigned_reference_key"].astype(str).eq(str(consensus))
            ]
            consistency = float(len(supporting) / len(assigned))
            model_status = (
                "root_only_reference"
                if str(consensus).endswith("-ROOT")
                else "projected_consensus"
            )
        n_analyses = int(len(group))
        n_reference_assigned = int(len(assigned))
        n_consensus_support = int(len(supporting))
        assignment_coverage = float(n_reference_assigned / n_analyses)
        evidence_status = _classify_model_evidence(
            n_analyses=n_analyses,
            n_reference_assigned=n_reference_assigned,
            n_consensus_support=n_consensus_support,
            assignment_consistency=float(consistency),
            assignment_coverage=assignment_coverage,
        )
        distance_ratio = pd.to_numeric(
            supporting.get("mean_neighbor_distance"), errors="coerce"
        ) / pd.to_numeric(
            supporting.get("max_mean_neighbor_distance"), errors="coerce"
        )
        representative_pool = supporting.copy()
        if representative_pool.empty:
            representative_pool = group.copy()
        representative_pool["_distance"] = pd.to_numeric(
            representative_pool.get("mean_neighbor_distance"), errors="coerce"
        ).fillna(np.inf)
        representative_id = str(
            representative_pool.sort_values(
                ["_distance", "analysis_id"]
            ).iloc[0]["analysis_id"]
        )
        rows.append(
            {
                "model_id": str(model_id),
                "model_label": _deterministic_mode(group["model_label"]),
                "model_grouping_basis": _deterministic_mode(
                    group["model_grouping_basis"]
                ),
                "root": str(root),
                "n_analyses": n_analyses,
                "n_reference_assigned": n_reference_assigned,
                "n_consensus_support": n_consensus_support,
                "assigned_reference_key": consensus,
                "model_assignment_status": model_status,
                "assignment_consistency": consistency,
                "assignment_coverage": assignment_coverage,
                "model_evidence_status": evidence_status,
                "representative_analysis_id": representative_id,
                "median_vote_fraction": pd.to_numeric(
                    supporting.get("top_vote_fraction"), errors="coerce"
                ).median(),
                "median_vote_margin": pd.to_numeric(
                    supporting.get("vote_margin"), errors="coerce"
                ).median(),
                "median_mean_neighbor_distance": pd.to_numeric(
                    supporting.get("mean_neighbor_distance"), errors="coerce"
                ).median(),
                "median_distance_to_limit_ratio": distance_ratio.median(),
            }
        )
    return pd.DataFrame(rows)


def _rank_context_models(models: pd.DataFrame) -> pd.DataFrame:
    candidates = models[
        models["assigned_reference_key"].notna()
        & models["model_assignment_status"].isin(
            ["projected_consensus", "root_only_reference"]
        )
    ].copy()
    candidates["_distance"] = pd.to_numeric(
        candidates["median_distance_to_limit_ratio"], errors="coerce"
    ).fillna(np.inf)
    candidates["_consistency"] = pd.to_numeric(
        candidates["assignment_consistency"], errors="coerce"
    ).fillna(0.0)
    candidates["_vote"] = pd.to_numeric(
        candidates["median_vote_margin"], errors="coerce"
    ).fillna(0.0)
    candidates["_evidence_rank"] = (
        candidates["model_evidence_status"]
        .map(MODEL_EVIDENCE_ORDER)
        .fillna(max(MODEL_EVIDENCE_ORDER.values()) + 1)
        .astype(int)
    )
    candidates["is_recommended_model"] = candidates[
        "model_evidence_status"
    ].isin(RECOMMENDED_MODEL_EVIDENCE)
    candidates = candidates.sort_values(
        [
            "assigned_reference_key",
            "_evidence_rank",
            "_consistency",
            "_distance",
            "_vote",
            "model_id",
        ],
        ascending=[True, True, False, True, False, True],
    )
    candidates["model_rank"] = (
        candidates.groupby("assigned_reference_key", sort=False).cumcount() + 1
    )
    return candidates.drop(
        columns=["_distance", "_consistency", "_vote", "_evidence_rank"]
    )


def _existing_projection_outputs(projection_dir: Path) -> dict[str, Path]:
    outputs = {
        "assignments": projection_dir / "projection_assignments.tsv",
        "candidates": projection_dir / "projection_candidates.tsv",
        "coordinates": projection_dir / "query_coordinates.parquet",
        "manifest": projection_dir / "projection_manifest.json",
    }
    missing = [str(path) for path in outputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Existing analysis projection is incomplete: " + ", ".join(missing)
        )
    return outputs


def _validate_projection_assignments(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "root", "assignment_status"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise KeyError(f"Projection assignments are missing columns: {missing}")

    output = assignments.copy()
    output["sample_id"] = output["sample_id"].astype(str)
    if output["sample_id"].duplicated().any():
        raise ValueError("Projection assignments contain duplicate sample IDs")

    expected = metadata[["analysis_id", "icdot_root_grouped"]].copy()
    expected["analysis_id"] = expected["analysis_id"].astype(str)
    expected_ids = set(expected["analysis_id"])
    observed_ids = set(output["sample_id"])
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "Existing projection does not match frozen experimental-model metadata; "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )

    expected_roots = expected.set_index("analysis_id")["icdot_root_grouped"].astype(str)
    observed_roots = output.set_index("sample_id")["root"].astype(str)
    root_mismatches = expected_roots.index[
        expected_roots.ne(observed_roots.reindex(expected_roots.index))
    ].tolist()
    if root_mismatches:
        raise ValueError(
            "Existing projection roots do not match frozen metadata: "
            f"{root_mismatches[:5]}"
        )
    return output


def project_experimental_models(
    *,
    bundle_dir: Path,
    metadata_path: Path,
    gene_fraction_store: Path,
    out_dir: Path,
    model_map_path: Path | None = None,
    reuse_analysis_projection: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Project all frozen experimental-model analyses and summarize models."""

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    gene_fraction_store = Path(gene_fraction_store).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if reuse_analysis_projection:
        if not out_dir.exists():
            raise FileNotFoundError(
                f"Experimental-model output does not exist for reuse: {out_dir}"
            )
        if not overwrite:
            raise FileExistsError(
                "Reusing an analysis projection rewrites model summaries; "
                "pass overwrite=True"
            )
    else:
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Experimental-model output exists: {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    metadata = pd.read_parquet(metadata_path)
    required = {"analysis_id", "individual_id", "icdot_root_grouped"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"Experimental-model metadata is missing columns: {missing}")
    metadata = metadata.copy()
    metadata["analysis_id"] = metadata["analysis_id"].astype(str)
    if metadata["analysis_id"].duplicated().any():
        raise ValueError("Experimental-model metadata contains duplicate analyses")
    metadata = _attach_model_ids(metadata, _read_model_map(model_map_path))
    if reuse_analysis_projection:
        projection_outputs = _existing_projection_outputs(out_dir / "projection")
    else:
        analysis_ids = metadata["analysis_id"].tolist()
        gene_fractions = load_gene_panel_subset_from_store(
            store_dir=gene_fraction_store,
            analysis_ids=analysis_ids,
            columns=["analysis_id", "gene_symbol", *EVENT_COLUMNS],
        )
        observed_ids = set(gene_fractions["analysis_id"].astype(str))
        missing_profiles = sorted(set(analysis_ids) - observed_ids)
        if missing_profiles:
            raise ValueError(
                "Experimental models are missing gene-panel profiles: "
                f"{missing_profiles[:5]}"
            )

        samples = metadata[["analysis_id", "icdot_root_grouped"]].rename(
            columns={"analysis_id": "sample_id", "icdot_root_grouped": "root"}
        )
        samples["genome_build"] = "GRCh38"
        projection_outputs = project_gene_profiles(
            bundle_dir=bundle_dir,
            samples=samples,
            gene_fractions=gene_fractions,
            out_dir=out_dir / "projection",
            input_provenance={
                "metadata": str(metadata_path),
                "gene_fraction_store": str(gene_fraction_store),
            },
        )
    assignments = _validate_projection_assignments(
        pd.read_csv(projection_outputs["assignments"], sep="\t"),
        metadata,
    )
    assignments = _use_effective_projection_metrics(assignments)
    assignments = assignments.rename(columns={"sample_id": "analysis_id"})
    metadata_columns = [
        "analysis_id",
        "biosample_id",
        "individual_id",
        "analysis_label",
        "icdo_topography_id",
        "icdo_morphology_id",
        "platform_id",
        "platform_family_refined",
        "model_id",
        "model_label",
        "model_grouping_basis",
    ]
    analysis = metadata[
        [column for column in metadata_columns if column in metadata.columns]
    ].merge(assignments, on="analysis_id", how="left", validate="one_to_one")
    if analysis["assignment_status"].isna().any():
        raise ValueError("Some experimental models lack projection accounting")
    references = _load_bundle_references(bundle_dir)
    catalog_references = references[
        references["reference_level"].astype(str).isin(CATALOG_REFERENCE_LEVELS)
    ].copy()
    eligible_roots = set(catalog_references["root"].astype(str))
    root_only_roots = set(
        references.loc[
            references["reference_level"].astype(str).eq("root"), "root"
        ].astype(str)
    )
    analysis = _annotate_catalog_eligibility(
        analysis,
        eligible_roots=eligible_roots,
        root_only_roots=root_only_roots,
    )
    models = _summarize_models(analysis)
    roots_per_model = models.groupby("model_id")["root"].nunique()
    models["n_roots_for_model"] = (
        models["model_id"].map(roots_per_model).astype(int)
    )
    models["multi_root_model"] = models["n_roots_for_model"].gt(1)
    reference_levels = references.set_index("reference_key")["reference_level"]
    models["reference_level"] = models["assigned_reference_key"].map(
        reference_levels
    )
    models = _annotate_catalog_eligibility(
        models,
        eligible_roots=eligible_roots,
        root_only_roots=root_only_roots,
    )
    catalog_models = models[models["catalog_eligible"]].copy()
    candidates = _rank_context_models(catalog_models)
    unknown_references = (
        set(candidates["assigned_reference_key"].dropna().astype(str))
        - set(references["reference_key"])
    )
    if unknown_references:
        raise ValueError(
            "Experimental models were assigned to references absent from the "
            f"projection bundle catalog: {sorted(unknown_references)}"
        )
    availability = (
        candidates.groupby("assigned_reference_key", as_index=False)
        .agg(
            n_model_groups=("model_id", "nunique"),
            n_analysis_ids=("n_analyses", "sum"),
            n_consensus_supporting_analyses=("n_consensus_support", "sum"),
            best_model_id=("model_id", "first"),
            best_model_evidence_status=("model_evidence_status", "first"),
        )
        .rename(columns={"assigned_reference_key": "reference_key"})
    )
    evidence_counts = (
        candidates.groupby(
            ["assigned_reference_key", "model_evidence_status"]
        )["model_id"]
        .nunique()
        .unstack(fill_value=0)
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"assigned_reference_key": "reference_key"})
    )
    evidence_columns = {
        "replicate_supported": "n_replicate_supported_models",
        "single_analysis_supported": "n_single_analysis_supported_models",
        "provisional_consensus": "n_provisional_consensus_models",
        "discordant_across_analyses": "n_discordant_models",
    }
    evidence_counts = evidence_counts.rename(columns=evidence_columns)
    for column in evidence_columns.values():
        if column not in evidence_counts.columns:
            evidence_counts[column] = 0
    availability = availability.merge(
        evidence_counts[
            ["reference_key", *evidence_columns.values()]
        ],
        on="reference_key",
        how="left",
        validate="one_to_one",
    )
    recommended = candidates[candidates["is_recommended_model"]].copy()
    best_recommended = (
        recommended.groupby("assigned_reference_key", as_index=False)
        .agg(
            best_recommended_model_id=("model_id", "first"),
            best_recommended_model_evidence_status=(
                "model_evidence_status",
                "first",
            ),
        )
        .rename(columns={"assigned_reference_key": "reference_key"})
    )
    availability = availability.merge(
        best_recommended,
        on="reference_key",
        how="left",
        validate="one_to_one",
    )
    availability = catalog_references.merge(
        availability, on="reference_key", how="left", validate="one_to_one"
    )
    availability["n_model_groups"] = availability["n_model_groups"].fillna(0).astype(int)
    availability["n_analysis_ids"] = availability["n_analysis_ids"].fillna(0).astype(int)
    availability["n_consensus_supporting_analyses"] = availability[
        "n_consensus_supporting_analyses"
    ].fillna(0).astype(int)
    for column in evidence_columns.values():
        availability[column] = availability[column].fillna(0).astype(int)
    availability["n_recommended_models"] = (
        availability["n_replicate_supported_models"]
        + availability["n_single_analysis_supported_models"]
    )
    availability["model_coverage_status"] = np.where(
        availability["n_model_groups"].gt(0), "model_available", "not_covered"
    )
    availability["recommended_model_coverage_status"] = np.select(
        [
            availability["n_recommended_models"].gt(0),
            availability["n_model_groups"].gt(0),
        ],
        ["recommended_model_available", "provisional_model_only"],
        default="not_covered",
    )
    root_summary = (
        analysis.groupby(
            ["catalog_eligible", "catalog_exclusion_reason", "root", "assignment_status"],
            as_index=False,
        )
        .agg(n_analyses=("analysis_id", "size"))
    )

    catalog_analysis = analysis[analysis["catalog_eligible"]].copy()
    catalog_roots_per_model = catalog_models.groupby("model_id")["root"].nunique()
    audit_exclusions = (
        analysis.loc[
            ~analysis["catalog_eligible"], "catalog_exclusion_reason"
        ]
        .value_counts()
        .to_dict()
    )
    catalog_exclusions = _summarize_catalog_exclusions(analysis, models)

    outputs = {
        "analysis_assignments": out_dir / "analysis_assignments.parquet",
        "model_assignments": out_dir / "model_assignments.tsv",
        "context_model_candidates": out_dir / "context_model_candidates.tsv",
        "context_model_availability": out_dir / "context_model_availability.tsv",
        "root_summary": out_dir / "root_summary.tsv",
    }
    analysis.to_parquet(outputs["analysis_assignments"], index=False)
    models.to_csv(outputs["model_assignments"], sep="\t", index=False)
    candidates.to_csv(outputs["context_model_candidates"], sep="\t", index=False)
    availability.to_csv(outputs["context_model_availability"], sep="\t", index=False)
    root_summary.to_csv(outputs["root_summary"], sep="\t", index=False)
    manifest = {
        "schema_version": "cnvsim_experimental_model_projection_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_projection_mode": (
            "reused_existing" if reuse_analysis_projection else "computed"
        ),
        "bundle": str(bundle_dir),
        "metadata": str(metadata_path),
        "gene_fraction_store": str(gene_fraction_store),
        "model_map": str(Path(model_map_path).expanduser().resolve())
        if model_map_path is not None
        else None,
        "model_evidence_policy": {
            "consistency_threshold": MODEL_CONSISTENCY_THRESHOLD,
            "assignment_coverage_threshold": MODEL_ASSIGNMENT_COVERAGE_THRESHOLD,
            "minimum_replicate_support": MODEL_MIN_REPLICATE_SUPPORT,
            "recommended_evidence": sorted(RECOMMENDED_MODEL_EVIDENCE),
        },
        "catalog_policy": {
            "model_grouping_basis": CATALOG_MODEL_GROUPING_BASIS,
            "reference_levels": sorted(CATALOG_REFERENCE_LEVELS),
            "excluded_records_remain_in_audit_tables": True,
        },
        "catalog_exclusions": catalog_exclusions,
        "counts": {
            "analyses": int(len(catalog_analysis)),
            "models": int(catalog_analysis["model_id"].nunique()),
            "model_groups": int(len(catalog_models)),
            "multi_root_models": int(catalog_roots_per_model.gt(1).sum()),
            "model_grouping_basis": {
                str(basis): int(count)
                for basis, count in catalog_analysis[
                    ["model_id", "model_grouping_basis"]
                ]
                .drop_duplicates("model_id")["model_grouping_basis"]
                .value_counts()
                .items()
            },
            "model_evidence_status": {
                str(status): int(count)
                for status, count in catalog_models[
                    "model_evidence_status"
                ].value_counts().items()
            },
            "references": int(len(catalog_references)),
            "reference_level": {
                str(level): int(count)
                for level, count in catalog_references[
                    "reference_level"
                ].value_counts().items()
            },
            "references_with_models": int(availability["n_model_groups"].gt(0).sum()),
            "references_with_recommended_models": int(
                availability["n_recommended_models"].gt(0).sum()
            ),
            "references_with_provisional_models_only": int(
                availability["recommended_model_coverage_status"]
                .eq("provisional_model_only")
                .sum()
            ),
            "assignment_status": {
                str(status): int(count)
                for status, count in catalog_analysis[
                    "assignment_status"
                ].value_counts().items()
            },
            "assignment_layer": {
                str(layer): int(count)
                for layer, count in catalog_analysis.get(
                    "assignment_layer",
                    pd.Series("none", index=catalog_analysis.index),
                ).value_counts().items()
            },
        },
        "audit_counts": {
            "input_analyses": int(len(analysis)),
            "input_models": int(analysis["model_id"].nunique()),
            "input_model_groups": int(len(models)),
            "input_multi_root_models": int(roots_per_model.gt(1).sum()),
            "model_grouping_basis": {
                str(basis): int(count)
                for basis, count in analysis[
                    ["model_id", "model_grouping_basis"]
                ]
                .drop_duplicates("model_id")["model_grouping_basis"]
                .value_counts()
                .items()
            },
            "catalog_exclusion_reason": {
                str(reason): int(count)
                for reason, count in audit_exclusions.items()
            },
            "projection_references": int(len(references)),
            "projection_reference_level": {
                str(level): int(count)
                for level, count in references[
                    "reference_level"
                ].value_counts().items()
            },
            "assignment_status": {
                str(status): int(count)
                for status, count in analysis[
                    "assignment_status"
                ].value_counts().items()
            },
        },
        "outputs": {key: path.name for key, path in outputs.items()},
        "projection_outputs": {
            key: str(path.relative_to(out_dir))
            for key, path in projection_outputs.items()
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**outputs, "manifest": manifest_path}


def freeze_experimental_model_projection(
    *,
    atlas_dir: Path,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Freeze compact experimental-model application tables outside work/."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    source_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else atlas_dir
        / "work"
        / "applications"
        / "experimental_model_projection_v1"
    )
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else atlas_dir / "applications" / "experimental_model_projection_v1"
    )
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("analysis_projection_mode") not in {
        "computed",
        "reused_existing",
    }:
        raise ValueError(
            "Freeze requires a computed or explicitly reused analysis projection"
        )

    source_paths = {
        name: source_dir / filename
        for name, filename in FROZEN_APPLICATION_FILES.items()
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experimental-model application is incomplete: " + ", ".join(missing)
        )

    analysis = pd.read_parquet(source_paths["analysis_assignments"])
    models = pd.read_csv(source_paths["model_assignments"], sep="\t")
    candidates = pd.read_csv(
        source_paths["context_model_candidates"], sep="\t"
    )
    availability = pd.read_csv(
        source_paths["context_model_availability"], sep="\t"
    )
    root_summary = pd.read_csv(source_paths["root_summary"], sep="\t")
    required_analysis = {
        "analysis_id",
        "model_id",
        "catalog_eligible",
        "catalog_exclusion_reason",
        "assignment_status",
        "assignment_layer",
        "assigned_reference_key",
    }
    required_models = {
        "model_id",
        "root",
        "catalog_eligible",
        "catalog_exclusion_reason",
        "model_evidence_status",
        "reference_level",
        "n_roots_for_model",
        "multi_root_model",
    }
    required_availability = {
        "reference_key",
        "reference_level",
        "n_model_groups",
        "n_recommended_models",
        "model_coverage_status",
        "recommended_model_coverage_status",
    }
    for name, frame, required in (
        ("analysis assignments", analysis, required_analysis),
        ("model assignments", models, required_models),
        ("model availability", availability, required_availability),
    ):
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise KeyError(f"{name} are missing columns: {missing_columns}")
    if analysis["analysis_id"].astype(str).duplicated().any():
        raise ValueError("Frozen analysis assignments contain duplicate analysis IDs")
    if models[["model_id", "root"]].astype(str).duplicated().any():
        raise ValueError("Frozen model assignments contain duplicate model-root groups")
    if availability["reference_key"].astype(str).duplicated().any():
        raise ValueError("Frozen availability contains duplicate references")
    if not candidates.empty and candidates[
        ["assigned_reference_key", "model_rank"]
    ].astype(str).duplicated().any():
        raise ValueError("Frozen candidates contain duplicate reference ranks")

    counts = source_manifest.get("counts", {})
    catalog_analysis = analysis[
        analysis["catalog_eligible"].fillna(False).astype(bool)
    ]
    catalog_models = models[
        models["catalog_eligible"].fillna(False).astype(bool)
    ]
    observed = {
        "analyses": int(len(catalog_analysis)),
        "models": int(catalog_analysis["model_id"].nunique()),
        "model_groups": int(len(catalog_models)),
        "references": int(len(availability)),
        "references_with_models": int(
            pd.to_numeric(
                availability["n_model_groups"], errors="coerce"
            ).fillna(0).gt(0).sum()
        ),
        "references_with_recommended_models": int(
            pd.to_numeric(
                availability["n_recommended_models"], errors="coerce"
            ).fillna(0).gt(0).sum()
        ),
    }
    for name, value in observed.items():
        if int(counts.get(name, -1)) != value:
            raise ValueError(
                f"Application count disagrees with source manifest: {name}"
            )
    if int(pd.to_numeric(root_summary["n_analyses"], errors="raise").sum()) != len(
        analysis
    ):
        raise ValueError("Root summary does not account for every analysis")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Frozen experimental-model application exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    outputs = {
        name: output_dir / path.name for name, path in source_paths.items()
    }
    for name, source_path in source_paths.items():
        shutil.copy2(source_path, outputs[name])

    manifest = {
        "schema_version": "cnvsim_experimental_model_application_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "frozen",
        "source_application": {
            "manifest_sha256": _sha256(source_manifest_path),
        },
        "analysis_projection_mode": source_manifest["analysis_projection_mode"],
        "model_evidence_policy": source_manifest.get("model_evidence_policy", {}),
        "catalog_policy": source_manifest.get("catalog_policy", {}),
        "catalog_exclusions": source_manifest.get("catalog_exclusions", {}),
        "counts": counts,
        "audit_counts": source_manifest.get("audit_counts", {}),
        "outputs": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksum_path = output_dir / "checksums.sha256"
    checksum_paths = [*outputs.values(), manifest_path]
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    return {**outputs, "manifest": manifest_path, "checksums": checksum_path}
