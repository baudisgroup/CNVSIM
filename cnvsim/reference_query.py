from __future__ import annotations

"""Link projected CNV profiles to reference cases, molecular context and models.

The query layer is read-only. It projects segments with the released bundle,
then looks up published reference definitions, optional aggregate
sequence-variant tables and the cell-line catalogue. It does not train,
download or run statistical tests.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from cnvsim.experimental_model_query import load_experimental_model_application
from cnvsim.reference_query_context import (
    ModelSignatureContext, MutationContext, MUTATION_FILES, optional_context,
)
from cnvsim.projection import _canonical_samples, project_samples


ACCEPTED_STATUS = "projected_assignment"
RECOMMENDED_EVIDENCE = {"replicate_supported", "single_analysis_supported"}
DEFINITION_STATUSES = {"anchor_member", "conditional_definition_member"}
COHORT_COLUMNS = [
    "retrieval_rank",
    "analysis_id",
    "biosample_id",
    "individual_id",
    "assigned_reference_key",
    "pipeline_status",
    "histological_diagnosis_id",
    "histological_diagnosis_label",
    "icdo_morphology_id",
    "icdo_morphology_label",
    "cohorts",
    "cohort_labels",
    "source_group",
    "source_group_refined",
    "platform_family_refined",
    "distance",
    "distance_metric",
]

SUPPORT_FIELDS = [
    ("vote_fraction", "top_vote_fraction"),
    ("vote_margin", "vote_margin"),
    ("mean_neighbor_distance", "mean_neighbor_distance"),
    ("distance_limit", "max_mean_neighbor_distance"),
    ("n_neighbors_used", "n_neighbors_used"),
]


def _clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return _clean(frame.to_dict("records"))


def _canonical_model_id(value: object) -> str:
    text = str(value or "").strip()
    accession = text.split(":")[-1].upper()
    return f"cellosaurus:{accession}" if accession.startswith("CVCL_") else text


def coverage_category(models: pd.DataFrame) -> str:
    """Summarise catalogue evidence for one subtype."""
    if models.empty:
        return "no_catalogue_match"
    evidence = models["evidence_status"]
    if evidence.isin(RECOMMENDED_EVIDENCE).any():
        return "recommended_available"
    if evidence.eq("provisional_consensus").any():
        return "provisional_only"
    if evidence.eq("discordant_across_analyses").all():
        return "discordant_only"
    return "no_recommended_candidate_other_evidence"


def support_metrics(assignment: dict) -> dict:
    """Return support metrics from the last layer that scored the query."""
    routed = assignment.get("conditional_assignment_status") in {
        "projected_assignment",
        "ambiguous",
        "unresolved",
    }
    conditional = assignment.get("assignment_layer") == "conditional" or routed
    prefix = "conditional_" if conditional else ""
    result = {
        name: _clean(assignment.get(prefix + column))
        for name, column in SUPPORT_FIELDS
    }
    if result["vote_fraction"] is None:
        result["scoring_layer"] = "none"
    else:
        result["scoring_layer"] = "conditional" if conditional else "primary"
    return result


class ReferenceRelease:
    """Published CNVSIM tables needed for reference-case retrieval."""

    def __init__(self, release_dir: str | Path):
        release = Path(release_dir).expanduser().resolve()
        self.release_dir = release
        self.version = json.loads((release / "manifest.json").read_text())[
            "release_version"
        ]
        self.bundle_dir = release / "projection_bundle"
        self.subtypes = pd.read_csv(release / "atlas/subtypes.tsv", sep="\t")
        self.signatures = pd.read_csv(
            release / "atlas/signature_events.tsv", sep="\t"
        )
        metadata_dir = release / "source_data/source_file/metadata"
        tumour = pd.read_parquet(metadata_dir / "tumour_analysis_metadata.parquet")
        cell_line = pd.read_parquet(
            metadata_dir / "cell_line_metadata.parquet",
            columns=["analysis_id", "individual_id"],
        )
        assignments = pd.read_parquet(release / "atlas/analysis_assignments.parquet")
        definition = assignments[assignments["pipeline_status"].isin(DEFINITION_STATUSES)]
        self.definition = definition.merge(
            tumour.drop(columns=["biosample_id"]),
            on="analysis_id",
            validate="one_to_one",
        )
        identity = pd.concat(
            [tumour[["analysis_id", "individual_id"]], cell_line]
        ).drop_duplicates()
        if identity["analysis_id"].duplicated().any():
            raise ValueError("Conflicting analysis-to-individual mapping")
        self.individual_of = identity.set_index("analysis_id")["individual_id"]
        self.coordinates = {
            "primary": pd.read_parquet(
                self.bundle_dir / "anchor_coordinates.parquet"
            ).set_index("analysis_id"),
            "conditional": pd.read_parquet(
                self.bundle_dir
                / "conditional/conditional_reference_coordinates.parquet"
            ).set_index("analysis_id"),
        }
        self.model_index = {
            "primary": pd.read_csv(
                self.bundle_dir / "model_index.tsv", sep="\t"
            ).set_index("root"),
            "conditional": pd.read_csv(
                self.bundle_dir / "conditional/model_index.tsv", sep="\t"
            ).set_index("root"),
        }
        models = load_experimental_model_application(release / "cell_line")
        self.models = models["models"]
        self.profiles = models["profiles"]
        recommended = self.models["evidence_status"].isin(RECOMMENDED_EVIDENCE)
        if not self.models["is_recommended"].eq(recommended).all():
            raise ValueError("Unexpected model recommendation/evidence mapping")

    def query_identity(self, sample: dict) -> dict:
        """Resolve analysis, individual and model identities for exclusions."""
        analysis_ids = {str(sample["sample_id"])}
        for column in ("source_analysis_id", "analysis_id"):
            if sample.get(column):
                analysis_ids.add(str(sample[column]))
        individuals = set(
            self.individual_of.reindex(sorted(analysis_ids)).dropna().astype(str)
        )
        supplied = sample.get("individual_id")
        if supplied and individuals and supplied not in individuals:
            raise ValueError(
                "Supplied individual_id disagrees with released analysis metadata"
            )
        if supplied:
            individuals.add(supplied)
        models = set(
            self.profiles.loc[
                self.profiles["analysis_id"].isin(analysis_ids), "model_id"
            ].dropna()
        )
        model = _canonical_model_id(sample.get("model_id", ""))
        if model and models and model not in models:
            raise ValueError("Supplied model_id disagrees with released profile metadata")
        if model:
            models.add(model)
        member = self.definition["analysis_id"].isin(analysis_ids).any()
        same_individual = self.definition["individual_id"].isin(individuals).any()
        return {
            "analysis_ids": sorted(analysis_ids),
            "individual_ids": sorted(individuals),
            "model_ids": sorted(models),
            "query_is_definition_member": bool(member),
            "query_individual_has_definition_records": bool(same_individual),
            "independent_of_reference_definition": not bool(member or same_individual),
            "independence_note": (
                "Metadata overlap check only; cross-source duplicates may be "
                "unresolved. Released assignment is unchanged by retrieval exclusions."
            ),
        }

    def reference_cohort(
        self,
        assignment: dict,
        identity: dict,
        query_coordinates: pd.Series,
    ) -> tuple[dict, pd.DataFrame]:
        """Same-subtype definition records ranked by stored-coordinate distance."""
        key = assignment["assigned_reference_key"]
        layer = "conditional" if assignment["assignment_layer"] == "conditional" else "primary"
        pool = self.definition[self.definition["assigned_reference_key"].eq(key)]
        expected = int(
            self.subtypes.loc[self.subtypes["reference_key"].eq(key), "n_definition_samples"].iloc[0]
        )
        if len(pool) != expected:
            raise ValueError("Reference cohort does not reproduce published definition count")
        same_analysis = pool["analysis_id"].isin(identity["analysis_ids"])
        same_individual = pool["individual_id"].isin(identity["individual_ids"])
        excluded = same_analysis | same_individual
        cohort = pool[~excluded].copy()

        settings = self.model_index[layer].loc[assignment["root"]]
        dimensions = [f"D{index}" for index in range(1, int(settings["n_pcs"]) + 1)]
        reference = self.coordinates[layer].loc[cohort["analysis_id"], dimensions]
        cohort["distance"] = (
            pairwise_distances(
                query_coordinates[dimensions].to_numpy(np.float32).reshape(1, -1),
                reference.to_numpy(np.float32),
                metric=str(settings["metric"]),
            )[0]
            if len(reference)
            else []
        )
        cohort["distance_metric"] = str(settings["metric"])
        cohort = cohort.sort_values(["distance", "analysis_id"])
        cohort["retrieval_rank"] = np.arange(1, len(cohort) + 1)
        cohort = cohort[COHORT_COLUMNS]
        for column in ("cohorts", "cohort_labels"):
            cohort[column] = cohort[column].map(
                lambda value: "; ".join(map(str, value))
                if isinstance(value, (list, np.ndarray))
                else value
            )
        summary = {
            "reference_key": key,
            "definition_records_before_exclusion": len(pool),
            "excluded_analysis_matches": int(same_analysis.sum()),
            "excluded_individual_matches": int(same_individual.sum()),
            "excluded_records_union": int(excluded.sum()),
            "eligible_definition_records": len(cohort),
            "unique_individual_ids": int(cohort["individual_id"].nunique()),
            "source_counts": cohort["source_group_refined"]
            .fillna("Unknown")
            .value_counts()
            .to_dict(),
            "ranking": (
                "Same assigned subtype, released stored coordinates, ascending "
                "distance then analysis_id"
            ),
            "caveat": (
                "Analysis records, not population prevalence. Repository groups "
                "are not independent studies; cross-source patient deduplication "
                "is incomplete."
            ),
        }
        return summary, cohort

    def model_candidates(
        self,
        reference_key: str,
        identity: dict,
    ) -> tuple[dict, pd.DataFrame]:
        """Catalogue matches for a subtype, excluding the query's own model."""
        matches = self.models[self.models["assigned_reference_key"].eq(reference_key)]
        own_model = matches["model_id"].isin(identity["model_ids"])
        eligible = matches[~own_model].sort_values(["model_rank", "model_id"]).copy()
        eligible["cellosaurus_url"] = eligible["model_id"].map(
            lambda value: "https://www.cellosaurus.org/" + value.split(":")[-1]
        )
        recommended = eligible["evidence_status"].isin(RECOMMENDED_EVIDENCE)
        own_call = self.models[self.models["model_id"].isin(identity["model_ids"])]
        summary = {
            "catalogue_status": coverage_category(matches),
            "catalogue_matched_model_root_records": len(matches),
            "catalogue_recommended_model_root_records": int(
                matches["evidence_status"].isin(RECOMMENDED_EVIDENCE).sum()
            ),
            "excluded_self_model_root_records": int(own_model.sum()),
            "eligible_matched_model_root_records": len(eligible),
            "eligible_recommended_model_root_records": int(recommended.sum()),
            "status_after_self_exclusion": coverage_category(eligible),
            "own_catalogue_call": _records(own_call),
            "own_catalogue_call_note": (
                "Derived from existing profiles including the representative; "
                "not an independent validation."
            ),
            "sibling_exclusion": (
                "Unavailable: release has no Cellosaurus same-donor/derivative hierarchy."
            ),
            "ranking": (
                "Published model rank; Tier A/B only recommended. No new "
                "query-specific functional ranking."
            ),
            "scope": (
                "Assessed catalogue only; no conclusion about unassessed cell "
                "lines, organoids or PDX."
            ),
        }
        return summary, eligible


def _read_samples(samples_path: Path) -> pd.DataFrame:
    samples = pd.read_csv(samples_path, sep="\t", dtype=str, keep_default_na=False)
    if "sample_id" not in samples.columns:
        raise KeyError("samples.tsv requires sample_id")
    if samples.empty or samples["sample_id"].eq("").any():
        raise ValueError("samples.tsv requires nonempty sample IDs")
    if samples["sample_id"].duplicated().any():
        raise ValueError("samples.tsv sample IDs must be unique")
    if "root" not in samples.columns:
        samples = _canonical_samples(samples)
    return samples


def run_reference_query(
    *, release_dir, samples_path, segments_path, out_dir,
    model_signature_dir=None, mutation_dir=None, nearest=5, top_models=5,
) -> list[dict]:
    """Project samples and write private reports with aggregate molecular context."""
    if int(nearest) < 1 or int(top_models) < 1:
        raise ValueError("nearest and top_models must be at least 1")
    release_dir = Path(release_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError("Output directory must be new")
    model_signature_dir = Path(model_signature_dir or release_dir / "cell_line/signature_evidence").expanduser().resolve()
    mutation_dir = Path(mutation_dir or release_dir / "validation/mutation_context").expanduser().resolve()
    for source in (release_dir, Path(samples_path), Path(segments_path), model_signature_dir, mutation_dir):
        source = source.expanduser().resolve()
        if out_dir.is_relative_to(source) or source.is_relative_to(out_dir):
            raise ValueError("Output directory must not overlap an input")
    samples = _read_samples(Path(samples_path))
    release = ReferenceRelease(release_dir)
    signatures, signature_status = optional_context(
        model_signature_dir, ["subtype_signature_coverage.tsv", "model_signature_evidence.tsv"],
        ModelSignatureContext, release.signatures, release.models)
    mutations, mutation_status = optional_context(
        mutation_dir, MUTATION_FILES.values(), MutationContext, release.subtypes)
    identities = {row["sample_id"]: release.query_identity(row) for row in samples.to_dict("records")}
    out_dir.mkdir(parents=True, mode=0o700)
    try:
        return _project_and_report(
            samples, Path(samples_path), Path(segments_path), out_dir, release, identities,
            signatures, signature_status, mutations, mutation_status, int(nearest), int(top_models))
    finally:
        # Query reports include user inputs and reference-case identifiers.
        for path in out_dir.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)


def _project_and_report(samples, samples_path, segments_path, out_dir, release, identities,
                        signatures, signature_status, mutations, mutation_status, nearest, top_models):
    projection = project_samples(
        bundle_dir=release.bundle_dir, samples_path=samples_path,
        segments_path=segments_path, out_dir=out_dir / "projection", nearest_samples=nearest)
    assignments = pd.read_csv(projection["assignments"], sep="\t", dtype={"sample_id": str})
    candidates = pd.read_csv(projection["candidates"], sep="\t", dtype={"sample_id": str})
    coordinates = pd.read_parquet(projection["coordinates"])
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    coordinates = coordinates.set_index("sample_id")
    summaries = []
    for index, sample in enumerate(samples.to_dict("records"), start=1):
        sample_id = sample["sample_id"]
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", sample_id)[:60]
        sample_dir = out_dir / f"query_{index:03d}_{safe_id}"
        sample_dir.mkdir(mode=0o700)
        assignment = _clean(assignments[assignments.sample_id.eq(sample_id)].iloc[0].to_dict())
        identity = identities[sample_id]
        report = {
            "schema_version": "cnvsim_reference_query_v1",
            "release_version": release.version,
            "query": sample, "assignment": assignment,
            "support": support_metrics(assignment), "identity": identity,
            "descriptive_candidates": _records(candidates[candidates.sample_id.eq(sample_id)]),
            "reference_cohort": None, "nearest_reference_cases": [],
            "models": None, "recommended_models_preview": [], "signature_preview": [],
            "cell_line_signature_context": signature_status.copy(),
            "tumour_mutation_context": mutation_status.copy(),
            "tables": {},
        }
        if assignment["assignment_status"] == ACCEPTED_STATUS:
            key = assignment["assigned_reference_key"]
            cohort_summary, cohort = release.reference_cohort(assignment, identity, coordinates.loc[sample_id])
            model_summary, models = release.model_candidates(key, identity)
            recommended = models[models.evidence_status.isin(RECOMMENDED_EVIDENCE)]
            structural = release.signatures[release.signatures.reference_key.eq(key)].copy()
            preview = (structural.sort_values(["track", "effect_size", "feature"], ascending=[True, False, True])
                       .groupby("track", sort=True).head(3))
            tables = {
                "reference_cohort": cohort, "nearest_reference_cases": cohort.head(nearest),
                "model_candidates": models, "structural_signatures": structural,
            }
            if signatures is not None:
                context, event_tables = signatures.lookup(key, models.model_id)
                tables.update(event_tables)
                evidence = event_tables["model_signature_evidence"]
                context["coverage"] = _records(event_tables["subtype_signature_coverage"])
                context["recommended_model_evidence_preview"] = _records(
                    evidence[evidence.model_id.isin(recommended.head(top_models).model_id)])
                report["cell_line_signature_context"] = context
            if mutations is not None:
                context, mutation_tables = mutations.lookup(key, sample)
                tables.update(mutation_tables)
                context["genie"] = _records(mutation_tables["genie_mutation_context"])
                context["tcga"] = _records(mutation_tables["tcga_mutation_context"])
                report["tumour_mutation_context"] = context
            for name, frame in tables.items():
                frame.to_csv(sample_dir / f"{name}.tsv", sep="\t", index=False)
            report.update(
                reference_cohort=cohort_summary, nearest_reference_cases=_records(cohort.head(nearest)),
                models=model_summary, recommended_models_preview=_records(recommended.head(top_models)),
                signature_preview=_records(preview), tables={name: f"{name}.tsv" for name in tables})
        (sample_dir / "report.json").write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
        summaries.append({
            "sample_id": sample_id, "assignment_status": assignment["assignment_status"],
            "assigned_reference_key": assignment.get("assigned_reference_key"),
            "report": str((sample_dir / "report.json").relative_to(out_dir)),
        })
    (out_dir / "reports.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries
