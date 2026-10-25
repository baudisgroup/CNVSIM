from __future__ import annotations

"""Validate an optional second-stage projection into conditional CNV contexts."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cnvsim.analysis.config import ReferenceProjectionParams
from cnvsim.analysis.ir import dimension_columns, read_analysis_coordinates
from cnvsim.analysis.reference_projection import (
    calibrate_anchor_projection,
    classify_reference_queries,
    score_reference_knn,
)


@dataclass(frozen=True)
class ConditionalContextValidationParams:
    """Fixed-label validation and admission thresholds."""

    test_fraction: float = 0.20
    random_seed: int = 20260722
    minimum_test_samples: int = 30
    minimum_rescue_eligible_samples: int = 20
    minimum_assignment_coverage: float = 0.50
    minimum_conditional_accuracy: float = 0.90
    minimum_operational_precision: float = 0.90
    maximum_core_control_acceptance: float = 0.05
    core_controls_per_root: int = 2000
    minimum_platform_test_samples: int = 10


EXPECTED_OUTPUTS = (
    "candidate_catalog.tsv",
    "holdout_split.parquet",
    "predictions.parquet",
    "projection_candidates.parquet",
    "projection_calibration.tsv",
    "projection_calibration_samples.parquet",
    "context_metrics.tsv",
    "platform_metrics.tsv",
    "source_metrics.tsv",
    "root_summary.tsv",
    "conditional_contexts.tsv",
    "summary_global.tsv",
    "manifest.json",
)

REPEATED_EXPECTED_OUTPUTS = (
    "candidate_catalog.tsv",
    "repeat_context_metrics.tsv",
    "context_stability.tsv",
    "platform_stability.tsv",
    "source_stability.tsv",
    "root_summary.tsv",
    "conditional_contexts.tsv",
    "repeat_summary.tsv",
    "manifest.json",
)

FROZEN_OUTPUTS = (
    "context_stability.tsv",
    "platform_stability.tsv",
    "source_stability.tsv",
    "root_summary.tsv",
    "conditional_contexts.tsv",
    "repeat_summary.tsv",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_params(params: ConditionalContextValidationParams) -> None:
    fractions = {
        "test_fraction": params.test_fraction,
        "minimum_assignment_coverage": params.minimum_assignment_coverage,
        "minimum_conditional_accuracy": params.minimum_conditional_accuracy,
        "minimum_operational_precision": params.minimum_operational_precision,
        "maximum_core_control_acceptance": params.maximum_core_control_acceptance,
    }
    invalid = {
        name: value for name, value in fractions.items() if not 0 < value < 1
    }
    if invalid:
        raise ValueError(f"Conditional-validation fractions must be in (0, 1): {invalid}")
    for name in (
        "minimum_test_samples",
        "minimum_rescue_eligible_samples",
        "core_controls_per_root",
        "minimum_platform_test_samples",
    ):
        if int(getattr(params, name)) < 1:
            raise ValueError(f"{name} must be positive")


def _stable_seed(root: str, context: int, base_seed: int) -> int:
    digest = hashlib.sha1(f"{root}\x1f{context}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32 - 1)


def _conditional_key(root: str, context: int) -> str:
    return f"{root}-COND{int(context)}"


def build_candidate_catalog(contexts: pd.DataFrame) -> pd.DataFrame:
    """Classify intermittent communities without using diagnostic labels."""

    _require_columns(
        contexts,
        [
            "root",
            "final_context",
            "community_key",
            "non_reference_structure_class",
            "evidence_class",
            "n_samples",
        ],
        "non_reference_context_concordance",
    )
    catalog = contexts[
        contexts["non_reference_structure_class"].eq("intermittent_coherent")
    ].copy()
    if catalog.empty:
        raise ValueError("No intermittent_coherent communities were found")
    catalog["root"] = catalog["root"].astype(str)
    catalog["final_context"] = pd.to_numeric(
        catalog["final_context"], errors="raise"
    ).astype(int)
    if catalog.duplicated(["root", "final_context"]).any():
        raise ValueError("Conditional candidate catalog contains duplicate communities")
    catalog["conditional_reference_key"] = [
        _conditional_key(root, context)
        for root, context in zip(
            catalog["root"], catalog["final_context"], strict=True
        )
    ]
    catalog["validation_role"] = np.where(
        catalog["evidence_class"].eq("dual_supported"),
        "primary_candidate",
        "negative_control",
    )
    catalog["prevalidation_eligible"] = catalog["validation_role"].eq(
        "primary_candidate"
    )
    catalog["prevalidation_exclusion_reason"] = np.where(
        catalog["prevalidation_eligible"],
        "",
        "lacks_independent_dual_track_evidence",
    )
    leading = [
        "community_key",
        "conditional_reference_key",
        "root",
        "final_context",
        "validation_role",
        "prevalidation_eligible",
        "prevalidation_exclusion_reason",
        "n_samples",
        "evidence_class",
        "n_track1_events",
        "n_track2_markers",
        "n_architecture_linked_markers",
        "detection_rate",
        "conditional_recall_q10",
        "conditional_precision_q10",
        "conditional_jaccard_q10",
        "platform_distribution",
        "source_distribution",
    ]
    ordered = [column for column in leading if column in catalog]
    ordered.extend(column for column in catalog.columns if column not in ordered)
    return catalog[ordered].sort_values(["root", "final_context"]).reset_index(
        drop=True
    )


def _split_members(
    members: pd.DataFrame,
    *,
    test_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (root, context), group in members.groupby(
        ["root", "final_context"], sort=True
    ):
        group = group.sort_values("analysis_id").reset_index(drop=True)
        n_test = max(1, int(round(len(group) * float(test_fraction))))
        n_test = min(n_test, len(group) - 1)
        platform = group["platform_family_refined"].astype("string").fillna(
            "__missing__"
        )
        counts = platform.value_counts()
        stratify = None
        if (
            len(counts) > 1
            and counts.min() >= 2
            and n_test >= len(counts)
            and len(group) - n_test >= len(counts)
        ):
            stratify = platform
        train_index, test_index = train_test_split(
            np.arange(len(group)),
            test_size=n_test,
            random_state=_stable_seed(str(root), int(context), random_seed),
            stratify=stratify,
        )
        group["holdout_role"] = "train"
        group.loc[test_index, "holdout_role"] = "test"
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["root", "final_context", "analysis_id"]
    )


def _sample_controls(
    frame: pd.DataFrame, *, limit: int, random_seed: int, root: str
) -> pd.DataFrame:
    if len(frame) <= int(limit):
        return frame.copy()
    return frame.sample(
        n=int(limit),
        random_state=_stable_seed(str(root), -1, random_seed),
    ).copy()


def _summarize_stratum(
    predictions: pd.DataFrame,
    *,
    stratum: str,
    minimum_samples: int,
) -> pd.DataFrame:
    test = predictions[predictions["query_role"].eq("conditional_holdout")].copy()
    if stratum not in test or test.empty:
        return pd.DataFrame()
    test[stratum] = test[stratum].astype("string").fillna("__missing__")
    rows: list[dict[str, object]] = []
    for (root, key, value), group in test.groupby(
        ["root", "true_conditional_reference_key", stratum], sort=True
    ):
        eligible = group[group["rescue_eligible"]]
        accepted = eligible[eligible["conditional_projection_status"].eq(
            "projected_assignment"
        )]
        correct = accepted[accepted["conditional_assignment_correct"]]
        rows.append(
            {
                "root": root,
                "conditional_reference_key": key,
                stratum: value,
                "n_test": int(len(group)),
                "n_rescue_eligible": int(len(eligible)),
                "n_assigned": int(len(accepted)),
                "n_correct": int(len(correct)),
                "assignment_coverage": (
                    len(accepted) / len(eligible) if len(eligible) else np.nan
                ),
                "conditional_accuracy": (
                    len(correct) / len(accepted) if len(accepted) else np.nan
                ),
                "high_confidence_recall": (
                    len(correct) / len(eligible) if len(eligible) else np.nan
                ),
                "sufficient_test_samples": bool(len(group) >= minimum_samples),
            }
        )
    return pd.DataFrame(rows)


def _context_metrics(
    *,
    catalog: pd.DataFrame,
    split: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    params: ConditionalContextValidationParams,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    accepted = predictions[
        predictions["conditional_projection_status"].eq("projected_assignment")
    ].copy()
    for candidate in catalog.itertuples(index=False):
        root = str(candidate.root)
        context = int(candidate.final_context)
        key = str(candidate.conditional_reference_key)
        members = split[
            split["root"].eq(root) & split["final_context"].eq(context)
        ]
        test = predictions[
            predictions["query_role"].eq("conditional_holdout")
            & predictions["true_conditional_reference_key"].eq(key)
        ]
        eligible = test[test["rescue_eligible"]]
        assigned = eligible[
            eligible["conditional_projection_status"].eq("projected_assignment")
        ]
        correct = assigned[assigned["conditional_assignment_correct"]]
        predicted_to_context = accepted[
            accepted["predicted_conditional_reference_key"].eq(key)
            & ~accepted["query_role"].eq("core_anchor_stress_control")
        ]
        false_positive = predicted_to_context[
            ~predicted_to_context["conditional_assignment_correct"]
        ]
        core_controls = predictions[
            predictions["root"].eq(root)
            & predictions["query_role"].eq("core_anchor_stress_control")
        ]
        core_attracted = core_controls[
            core_controls["conditional_projection_status"].eq(
                "projected_assignment"
            )
            & core_controls["predicted_conditional_reference_key"].eq(key)
        ]
        threshold = thresholds[
            thresholds["root"].eq(root)
            & thresholds["conditional_reference_key"].eq(key)
        ]
        calibration_available = bool(
            len(threshold) == 1 and threshold.iloc[0]["calibration_available"]
        )
        assignment_coverage = (
            len(assigned) / len(eligible) if len(eligible) else np.nan
        )
        conditional_accuracy = (
            len(correct) / len(assigned) if len(assigned) else np.nan
        )
        operational_precision = (
            len(correct) / (len(correct) + len(false_positive))
            if len(correct) + len(false_positive)
            else np.nan
        )
        core_acceptance = (
            len(core_attracted) / len(core_controls) if len(core_controls) else np.nan
        )
        gates = {
            "passes_independent_evidence": bool(candidate.prevalidation_eligible),
            "passes_test_size": len(test) >= params.minimum_test_samples,
            "passes_rescue_eligible_size": (
                len(eligible) >= params.minimum_rescue_eligible_samples
            ),
            "passes_calibration": calibration_available,
            "passes_assignment_coverage": bool(
                np.isfinite(assignment_coverage)
                and assignment_coverage >= params.minimum_assignment_coverage
            ),
            "passes_conditional_accuracy": bool(
                np.isfinite(conditional_accuracy)
                and conditional_accuracy >= params.minimum_conditional_accuracy
            ),
            "passes_operational_precision": bool(
                np.isfinite(operational_precision)
                and operational_precision >= params.minimum_operational_precision
            ),
            "passes_core_control_specificity": bool(
                np.isfinite(core_acceptance)
                and core_acceptance <= params.maximum_core_control_acceptance
            ),
        }
        failed = [name.removeprefix("passes_") for name, passed in gates.items() if not passed]
        rows.append(
            {
                "community_key": candidate.community_key,
                "conditional_reference_key": key,
                "root": root,
                "final_context": context,
                "validation_role": candidate.validation_role,
                "evidence_class": candidate.evidence_class,
                "n_full_members": int(len(members)),
                "n_train": int(members["holdout_role"].eq("train").sum()),
                "n_test": int(len(test)),
                "n_test_retained_core": int(
                    test["core_assignment_status"].eq("projected_assignment").sum()
                ),
                "n_rescue_eligible": int(len(eligible)),
                "n_conditional_assigned": int(len(assigned)),
                "n_conditional_correct": int(len(correct)),
                "n_conditional_incorrect": int(len(assigned) - len(correct)),
                "n_operational_false_positive": int(len(false_positive)),
                "n_core_controls": int(len(core_controls)),
                "n_core_controls_attracted": int(len(core_attracted)),
                "rescue_eligible_fraction": len(eligible) / len(test) if len(test) else np.nan,
                "assignment_coverage": assignment_coverage,
                "conditional_accuracy": conditional_accuracy,
                "high_confidence_recall": (
                    len(correct) / len(eligible) if len(eligible) else np.nan
                ),
                "all_test_conditional_recall": (
                    len(correct) / len(test) if len(test) else np.nan
                ),
                "operational_precision": operational_precision,
                "core_control_acceptance": core_acceptance,
                "calibration_available": calibration_available,
                **gates,
                "conditional_layer_eligible": bool(all(gates.values())),
                "failed_admission_gates": "|".join(failed),
            }
        )
    return pd.DataFrame(rows).sort_values(["root", "final_context"]).reset_index(
        drop=True
    )


def validate_conditional_contexts(
    *,
    atlas_dir: Path,
    analysis_work_dir: Path,
    diagnostic_dir: Path,
    out_dir: Path,
    projection_params: ReferenceProjectionParams,
    validation_params: ConditionalContextValidationParams = ConditionalContextValidationParams(),
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Validate conditional contexts in a core-first rescue projection design."""

    _validate_params(validation_params)
    atlas_dir = Path(atlas_dir).expanduser().resolve()
    analysis_dir = Path(analysis_work_dir).expanduser().resolve()
    diagnostic_dir = Path(diagnostic_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    existing = [out_dir / name for name in EXPECTED_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Conditional validation output exists; pass --overwrite or use a new "
            f"directory: {existing}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "diagnostic_contexts": diagnostic_dir / "non_reference_context_concordance.tsv",
        "sample_status": analysis_dir / "reference_contexts" / "sample_reference_status.parquet",
        "core_projection": analysis_dir / "projection" / "non_anchor_recovery" / "projection_assignments.parquet",
        "selected_parameters": analysis_dir / "parameters" / "selected_parameters.tsv",
        "ir_dir": analysis_dir / "ir",
        "metadata": (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else atlas_dir
            / "source_file"
            / "metadata"
            / "tumour_analysis_metadata.parquet"
        ),
    }
    missing = [str(path) for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing conditional-validation inputs: {missing}")

    contexts = pd.read_csv(paths["diagnostic_contexts"], sep="\t")
    catalog = build_candidate_catalog(contexts)
    status = pd.read_parquet(paths["sample_status"])
    projection = pd.read_parquet(paths["core_projection"])
    metadata = pd.read_parquet(paths["metadata"])
    parameters = pd.read_csv(paths["selected_parameters"], sep="\t")
    _require_columns(
        status,
        ["analysis_id", "root", "source_final_context", "assignment_status"],
        "sample_reference_status",
    )
    _require_columns(
        projection,
        ["analysis_id", "assignment_status"],
        "core_projection",
    )
    _require_columns(
        metadata,
        ["analysis_id", "platform_family_refined", "source_group_refined"],
        "tumour metadata",
    )
    _require_columns(parameters, ["root", "metric", "n_pcs", "K"], "parameters")
    for frame in (status, projection, metadata, parameters):
        if "analysis_id" in frame:
            frame["analysis_id"] = frame["analysis_id"].astype(str)
    status["root"] = status["root"].astype(str)
    parameters["root"] = parameters["root"].astype(str)
    if parameters.duplicated("root").any():
        raise ValueError("Selected parameters contain duplicate roots")
    parameter_lookup = parameters.set_index("root")
    projection = projection.rename(
        columns={
            "assignment_status": "core_assignment_status",
            "assigned_reference_id": "core_assigned_reference_id",
        }
    )
    if projection["analysis_id"].duplicated().any():
        raise ValueError("Core projection contains duplicate analysis IDs")

    metadata_columns = [
        "analysis_id",
        "platform_family_refined",
        "source_group_refined",
    ]
    metadata = metadata[metadata_columns].drop_duplicates("analysis_id", keep="first")
    nonanchors = status[status["assignment_status"].eq("non_anchor_query")].copy()
    nonanchors["final_context"] = pd.to_numeric(
        nonanchors["source_final_context"], errors="raise"
    ).astype(int)
    nonanchors = nonanchors.merge(
        metadata, on="analysis_id", how="left", validate="one_to_one"
    ).merge(
        projection[
            [
                column
                for column in (
                    "analysis_id",
                    "core_assignment_status",
                    "core_assigned_reference_id",
                )
                if column in projection
            ]
        ],
        on="analysis_id",
        how="left",
        validate="one_to_one",
    )
    if nonanchors["core_assignment_status"].isna().any():
        raise ValueError("Some non-anchor samples lack core projection outcomes")

    catalog_keys = catalog[["root", "final_context", "conditional_reference_key", "validation_role"]]
    candidate_members = nonanchors.merge(
        catalog_keys,
        on=["root", "final_context"],
        how="inner",
        validate="many_to_one",
    )
    observed_sizes = candidate_members.groupby(["root", "final_context"]).size()
    expected_sizes = catalog.set_index(["root", "final_context"])["n_samples"].astype(int)
    if not observed_sizes.reindex(expected_sizes.index).fillna(0).astype(int).equals(
        expected_sizes
    ):
        raise ValueError("Conditional candidate membership disagrees with diagnostic output")

    primary_members = candidate_members[
        candidate_members["validation_role"].eq("primary_candidate")
    ].copy()
    split = _split_members(
        primary_members,
        test_fraction=validation_params.test_fraction,
        random_seed=validation_params.random_seed,
    )
    negative = candidate_members[
        candidate_members["validation_role"].eq("negative_control")
    ].copy()
    negative["holdout_role"] = "negative_control"
    split_output = pd.concat([split, negative], ignore_index=True, sort=False)

    prediction_parts: list[pd.DataFrame] = []
    candidate_score_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    roots = sorted(split["root"].unique())
    for root_index, root in enumerate(roots, start=1):
        print(
            f"[conditional validation] [{root_index}/{len(roots)}] root={root}",
            flush=True,
        )
        if root not in parameter_lookup.index:
            raise KeyError(f"[{root}] selected parameters are missing")
        parameter = parameter_lookup.loc[root]
        metric = str(parameter["metric"]).lower()
        n_pcs = int(parameter["n_pcs"])
        k = int(parameter["K"])
        coords_path = paths["ir_dir"] / "perroot" / root / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(coords_path)
        coords = read_analysis_coordinates(coords_path)
        dimensions = dimension_columns(coords)[:n_pcs]
        if len(dimensions) != n_pcs:
            raise ValueError(f"[{root}] requested {n_pcs} dimensions, found {len(dimensions)}")

        root_split = split[split["root"].eq(root)].copy()
        train = root_split[root_split["holdout_role"].eq("train")].copy()
        test = root_split[root_split["holdout_role"].eq("test")].copy()
        train_ids = train["analysis_id"].astype(str).tolist()
        missing_coords = set(train_ids) - set(coords.index)
        if missing_coords:
            raise ValueError(f"[{root}] {len(missing_coords)} train samples lack IR coordinates")
        train_contexts = train["final_context"].astype(int).to_numpy()
        train_values = coords.loc[train_ids, dimensions].to_numpy(np.float32)
        thresholds, calibration = calibrate_anchor_projection(
            analysis_ids=train_ids,
            values=train_values,
            contexts=train_contexts,
            metric=metric,
            n_neighbors=k,
            params=projection_params,
        )
        context_to_key = {
            int(row.final_context): str(row.conditional_reference_key)
            for row in catalog[catalog["root"].eq(root)].itertuples(index=False)
            if bool(row.prevalidation_eligible)
        }
        threshold_output = thresholds.copy()
        threshold_output.insert(0, "root", root)
        threshold_output["conditional_reference_key"] = threshold_output[
            "context"
        ].map(context_to_key)
        threshold_output["metric"] = metric
        threshold_output["n_pcs"] = n_pcs
        threshold_output["K"] = k
        threshold_parts.append(threshold_output)
        calibration.insert(1, "root", root)
        calibration["true_conditional_reference_key"] = calibration[
            "true_context"
        ].map(context_to_key)
        calibration["predicted_conditional_reference_key"] = calibration[
            "predicted_context"
        ].map(context_to_key)
        calibration_parts.append(calibration)

        test["query_role"] = "conditional_holdout"
        test["true_conditional_reference_key"] = test[
            "conditional_reference_key"
        ]
        test["rescue_eligible"] = test["core_assignment_status"].isin(
            ["ambiguous", "unresolved"]
        )
        retained = test[~test["rescue_eligible"]].copy()
        retained["conditional_projection_status"] = "not_routed"
        retained["conditional_route_status"] = "retained_core_assignment"
        retained["predicted_conditional_reference_key"] = pd.NA
        retained["conditional_assignment_correct"] = False
        prediction_parts.append(retained)

        candidate_contexts = set(train["final_context"].astype(int))
        other_controls = nonanchors[
            nonanchors["root"].eq(root)
            & ~nonanchors["final_context"].isin(candidate_contexts)
            & nonanchors["core_assignment_status"].isin(["ambiguous", "unresolved"])
        ].copy()
        other_controls["query_role"] = "noncandidate_nonanchor_control"
        other_controls["true_conditional_reference_key"] = pd.NA
        other_controls["rescue_eligible"] = True

        anchors = status[
            status["root"].eq(root)
            & status["assignment_status"].eq("anchor_member")
        ].copy()
        anchors = anchors.merge(
            metadata, on="analysis_id", how="left", validate="one_to_one"
        )
        anchors = _sample_controls(
            anchors,
            limit=validation_params.core_controls_per_root,
            random_seed=validation_params.random_seed,
            root=root,
        )
        anchors["query_role"] = "core_anchor_stress_control"
        anchors["true_conditional_reference_key"] = pd.NA
        anchors["rescue_eligible"] = False
        anchors["core_assignment_status"] = "anchor_member"

        routed_test = test[test["rescue_eligible"]].copy()
        scored_queries = pd.concat(
            [routed_test, other_controls, anchors], ignore_index=True, sort=False
        )
        score_ids = scored_queries["analysis_id"].astype(str).tolist()
        missing_coords = set(score_ids) - set(coords.index)
        if missing_coords:
            raise ValueError(f"[{root}] {len(missing_coords)} query/control samples lack IR coordinates")
        candidates, scores = score_reference_knn(
            query_ids=score_ids,
            query_values=coords.loc[score_ids, dimensions].to_numpy(np.float32),
            reference_ids=train_ids,
            reference_values=train_values,
            reference_labels=train_contexts,
            metric=metric,
            n_neighbors=k,
            top_n=projection_params.top_candidates,
            exclude_self=False,
        )
        classified = classify_reference_queries(scores, thresholds, root=root)
        classified = classified.rename(
            columns={"assignment_status": "conditional_projection_status"}
        )
        classified["predicted_conditional_reference_key"] = classified[
            "predicted_context"
        ].map(context_to_key)
        classified["conditional_route_status"] = np.select(
            [
                classified["conditional_projection_status"].eq("projected_assignment"),
                classified["conditional_projection_status"].eq("ambiguous"),
            ],
            ["conditional_assignment", "conditional_ambiguous"],
            default="conditional_unresolved",
        )
        scored_queries = scored_queries.merge(
            classified, on="analysis_id", how="left", validate="one_to_one"
        )
        scored_queries["conditional_assignment_correct"] = (
            scored_queries["query_role"].eq("conditional_holdout")
            & scored_queries["conditional_projection_status"].eq(
                "projected_assignment"
            )
            & scored_queries["predicted_conditional_reference_key"].eq(
                scored_queries["true_conditional_reference_key"]
            )
        )
        prediction_parts.append(scored_queries)
        candidates.insert(1, "root", root)
        candidates["candidate_conditional_reference_key"] = candidates[
            "context"
        ].map(context_to_key)
        candidate_score_parts.append(candidates)

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    if predictions.duplicated(["analysis_id", "query_role"]).any():
        raise ValueError("Conditional prediction output contains duplicate query rows")
    for column in (
        "conditional_projection_status",
        "conditional_route_status",
        "true_conditional_reference_key",
        "predicted_conditional_reference_key",
    ):
        if column not in predictions:
            predictions[column] = pd.NA
    predictions["conditional_assignment_correct"] = predictions[
        "conditional_assignment_correct"
    ].fillna(False).astype(bool)
    thresholds = pd.concat(threshold_parts, ignore_index=True, sort=False)
    calibration = pd.concat(calibration_parts, ignore_index=True, sort=False)
    candidate_scores = (
        pd.concat(candidate_score_parts, ignore_index=True, sort=False)
        if candidate_score_parts
        else pd.DataFrame()
    )
    metrics = _context_metrics(
        catalog=catalog,
        split=split_output,
        predictions=predictions,
        thresholds=thresholds,
        params=validation_params,
    )
    platform_metrics = _summarize_stratum(
        predictions,
        stratum="platform_family_refined",
        minimum_samples=validation_params.minimum_platform_test_samples,
    )
    source_metrics = _summarize_stratum(
        predictions,
        stratum="source_group_refined",
        minimum_samples=validation_params.minimum_platform_test_samples,
    )
    root_summary = (
        metrics.groupby("root", sort=True)
        .agg(
            n_candidates=("conditional_reference_key", "size"),
            n_prevalidation_eligible=("passes_independent_evidence", "sum"),
            n_admitted=("conditional_layer_eligible", "sum"),
            n_candidate_samples=("n_full_members", "sum"),
            n_test=("n_test", "sum"),
            n_rescue_eligible=("n_rescue_eligible", "sum"),
            n_conditional_correct=("n_conditional_correct", "sum"),
            n_operational_false_positive=("n_operational_false_positive", "sum"),
        )
        .reset_index()
    )
    admitted = metrics[metrics["conditional_layer_eligible"]].copy()
    admitted["reference_level"] = "conditional"
    admitted["assignment_method"] = "core_first_conditional_knn"
    summary = pd.DataFrame(
        [
            {
                "scope": "all",
                "n_structural_candidates": int(len(catalog)),
                "n_primary_candidates": int(catalog["prevalidation_eligible"].sum()),
                "n_negative_controls": int((~catalog["prevalidation_eligible"]).sum()),
                "n_admitted_conditional_contexts": int(len(admitted)),
                "n_structural_candidate_samples": int(catalog["n_samples"].sum()),
                "n_primary_candidate_samples": int(
                    catalog.loc[catalog["prevalidation_eligible"], "n_samples"].sum()
                ),
                "n_admitted_definition_samples": int(admitted["n_full_members"].sum()),
            }
        ]
    )

    outputs = {
        "candidate_catalog": out_dir / "candidate_catalog.tsv",
        "holdout_split": out_dir / "holdout_split.parquet",
        "predictions": out_dir / "predictions.parquet",
        "projection_candidates": out_dir / "projection_candidates.parquet",
        "projection_calibration": out_dir / "projection_calibration.tsv",
        "projection_calibration_samples": out_dir
        / "projection_calibration_samples.parquet",
        "context_metrics": out_dir / "context_metrics.tsv",
        "platform_metrics": out_dir / "platform_metrics.tsv",
        "source_metrics": out_dir / "source_metrics.tsv",
        "root_summary": out_dir / "root_summary.tsv",
        "conditional_contexts": out_dir / "conditional_contexts.tsv",
        "summary_global": out_dir / "summary_global.tsv",
    }
    catalog.to_csv(outputs["candidate_catalog"], sep="\t", index=False)
    split_output.to_parquet(outputs["holdout_split"], index=False)
    predictions.to_parquet(outputs["predictions"], index=False)
    candidate_scores.to_parquet(outputs["projection_candidates"], index=False)
    thresholds.to_csv(outputs["projection_calibration"], sep="\t", index=False)
    calibration.to_parquet(outputs["projection_calibration_samples"], index=False)
    metrics.to_csv(outputs["context_metrics"], sep="\t", index=False)
    platform_metrics.to_csv(outputs["platform_metrics"], sep="\t", index=False)
    source_metrics.to_csv(outputs["source_metrics"], sep="\t", index=False)
    root_summary.to_csv(outputs["root_summary"], sep="\t", index=False)
    admitted.to_csv(outputs["conditional_contexts"], sep="\t", index=False)
    summary.to_csv(outputs["summary_global"], sep="\t", index=False)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cnvsim_conditional_context_validation_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "validation_design": {
                    "representation": "frozen_full_atlas_IR",
                    "labels": "train_only_fixed_conditional_labels",
                    "routing": "core_first_then_conditional_rescue",
                    "candidate_selection_uses_diagnostic_labels": False,
                    "core_anchor_controls_are_operational_queries": False,
                    "negative_control_role": (
                        "intermittent coherent community lacking dual-track evidence"
                    ),
                },
                "projection": asdict(projection_params),
                "admission": asdict(validation_params),
                "counts": summary.iloc[0].to_dict(),
                "inputs": {name: str(path) for name, path in paths.items()},
                "outputs": {name: path.name for name, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "manifest": manifest_path}


def freeze_conditional_context_validation(
    *,
    atlas_dir: Path,
    source_dir: Path | None = None,
    conditional_reference_dir: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Freeze compact repeated-validation evidence for admitted contexts."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    source_dir = (
        Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else atlas_dir / "work" / "validation" / "conditional_contexts_repeated_v1"
    )
    conditional_reference_dir = (
        Path(conditional_reference_dir).expanduser().resolve()
        if conditional_reference_dir is not None
        else atlas_dir
        / "work"
        / "analysis_v1"
        / "analysis"
        / "conditional_reference"
    )
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else atlas_dir / "validation" / "conditional_contexts_v1"
    )
    source_manifest_path = source_dir / "manifest.json"
    reference_contexts_path = conditional_reference_dir / "conditional_contexts.tsv"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    if not reference_contexts_path.exists():
        raise FileNotFoundError(reference_contexts_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema_version") != (
        "cnvsim_conditional_context_repeated_validation_v1"
    ):
        raise ValueError("Freeze requires repeated conditional validation v1")

    source_paths = {name: source_dir / name for name in FROZEN_OUTPUTS}
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete repeated conditional validation: " + ", ".join(missing)
        )
    admitted = pd.read_csv(source_paths["conditional_contexts.tsv"], sep="\t")
    references = pd.read_csv(reference_contexts_path, sep="\t")
    _require_columns(
        admitted,
        {"conditional_reference_key"},
        "validated conditional contexts",
    )
    reference_key_column = (
        "reference_key"
        if "reference_key" in references
        else "conditional_reference_key"
    )
    _require_columns(
        references,
        {reference_key_column},
        "conditional projection reference",
    )
    admitted_keys = set(admitted["conditional_reference_key"].astype(str))
    reference_keys = set(references[reference_key_column].astype(str))
    if admitted_keys != reference_keys:
        raise ValueError(
            "Frozen conditional reference disagrees with repeated validation: "
            f"validation_only={sorted(admitted_keys - reference_keys)}, "
            f"reference_only={sorted(reference_keys - admitted_keys)}"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Frozen conditional validation exists: {output_dir}"
            )
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    outputs: dict[str, Path] = {}
    for name, source_path in source_paths.items():
        destination = output_dir / name
        destination.write_bytes(source_path.read_bytes())
        outputs[Path(name).stem] = destination

    manifest = {
        "schema_version": "cnvsim_conditional_context_validation_release_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "frozen",
        "validation_design": source_manifest["validation_design"],
        "projection": source_manifest["projection"],
        "admission": source_manifest["admission"],
        "counts": source_manifest["counts"],
        "inputs": {
            "repeated_validation_manifest_sha256": _sha256(source_manifest_path),
            "conditional_reference_contexts_sha256": _sha256(
                reference_contexts_path
            ),
        },
        "outputs": {
            name: {"path": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksums_path = output_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in [*outputs.values(), manifest_path]
        )
    )
    return {**outputs, "manifest": manifest_path, "checksums": checksums_path}


def _quantile(series: pd.Series, value: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(value)) if not numeric.empty else np.nan


def _aggregate_context_repeats(
    metrics: pd.DataFrame,
    *,
    repeats: int,
    minimum_repeat_pass_rate: float,
    params: ConditionalContextValidationParams,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby("conditional_reference_key", sort=True):
        first = group.iloc[0]
        n_repeats = int(group["repeat"].nunique())
        n_passed = int(group["conditional_layer_eligible"].fillna(False).sum())
        pass_rate = n_passed / n_repeats if n_repeats else np.nan
        calibration_rate = float(
            group["calibration_available"].fillna(False).mean()
        )
        coverage_q10 = _quantile(group["assignment_coverage"], 0.10)
        accuracy_q10 = _quantile(group["conditional_accuracy"], 0.10)
        precision_q10 = _quantile(group["operational_precision"], 0.10)
        core_acceptance_q90 = _quantile(group["core_control_acceptance"], 0.90)
        gates = {
            "passes_independent_evidence": bool(
                first["passes_independent_evidence"]
            ),
            "passes_repeat_accounting": n_repeats == int(repeats),
            "passes_repeat_pass_rate": bool(
                np.isfinite(pass_rate)
                and pass_rate >= float(minimum_repeat_pass_rate)
            ),
            "passes_calibration_rate": calibration_rate
            >= float(minimum_repeat_pass_rate),
            "passes_coverage_q10": bool(
                np.isfinite(coverage_q10)
                and coverage_q10 >= params.minimum_assignment_coverage
            ),
            "passes_accuracy_q10": bool(
                np.isfinite(accuracy_q10)
                and accuracy_q10 >= params.minimum_conditional_accuracy
            ),
            "passes_precision_q10": bool(
                np.isfinite(precision_q10)
                and precision_q10 >= params.minimum_operational_precision
            ),
            "passes_core_specificity_q90": bool(
                np.isfinite(core_acceptance_q90)
                and core_acceptance_q90
                <= params.maximum_core_control_acceptance
            ),
        }
        failed = [
            name.removeprefix("passes_")
            for name, passed in gates.items()
            if not passed
        ]
        rows.append(
            {
                "community_key": first["community_key"],
                "conditional_reference_key": key,
                "root": first["root"],
                "final_context": int(first["final_context"]),
                "validation_role": first["validation_role"],
                "evidence_class": first["evidence_class"],
                "n_full_members": int(first["n_full_members"]),
                "n_repeats": n_repeats,
                "n_repeats_passed": n_passed,
                "repeat_pass_rate": pass_rate,
                "calibration_rate": calibration_rate,
                "assignment_coverage_median": _quantile(
                    group["assignment_coverage"], 0.50
                ),
                "assignment_coverage_q10": coverage_q10,
                "conditional_accuracy_median": _quantile(
                    group["conditional_accuracy"], 0.50
                ),
                "conditional_accuracy_q10": accuracy_q10,
                "operational_precision_median": _quantile(
                    group["operational_precision"], 0.50
                ),
                "operational_precision_q10": precision_q10,
                "core_control_acceptance_median": _quantile(
                    group["core_control_acceptance"], 0.50
                ),
                "core_control_acceptance_q90": core_acceptance_q90,
                "high_confidence_recall_median": _quantile(
                    group["high_confidence_recall"], 0.50
                ),
                "high_confidence_recall_q10": _quantile(
                    group["high_confidence_recall"], 0.10
                ),
                "all_test_conditional_recall_median": _quantile(
                    group["all_test_conditional_recall"], 0.50
                ),
                "all_test_conditional_recall_q10": _quantile(
                    group["all_test_conditional_recall"], 0.10
                ),
                "n_rescue_eligible_median": _quantile(
                    group["n_rescue_eligible"], 0.50
                ),
                "n_operational_false_positive_median": _quantile(
                    group["n_operational_false_positive"], 0.50
                ),
                **gates,
                "conditional_layer_eligible": bool(all(gates.values())),
                "failed_repeated_admission_gates": "|".join(failed),
            }
        )
    return pd.DataFrame(rows).sort_values(["root", "final_context"]).reset_index(
        drop=True
    )


def _aggregate_stratum_repeats(
    metrics: pd.DataFrame,
    *,
    stratum: str,
    repeats: int,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = metrics.groupby(
        ["root", "conditional_reference_key", stratum], sort=True, dropna=False
    )
    for (root, key, value), group in grouped:
        sufficient = group[group["sufficient_test_samples"].fillna(False)]
        rows.append(
            {
                "root": root,
                "conditional_reference_key": key,
                stratum: value,
                "n_repeats": int(group["repeat"].nunique()),
                "n_sufficient_repeats": int(sufficient["repeat"].nunique()),
                "sufficient_repeat_fraction": (
                    sufficient["repeat"].nunique() / repeats if repeats else np.nan
                ),
                "n_test_median": _quantile(group["n_test"], 0.50),
                "assignment_coverage_median": _quantile(
                    sufficient["assignment_coverage"], 0.50
                ),
                "assignment_coverage_q10": _quantile(
                    sufficient["assignment_coverage"], 0.10
                ),
                "conditional_accuracy_median": _quantile(
                    sufficient["conditional_accuracy"], 0.50
                ),
                "conditional_accuracy_q10": _quantile(
                    sufficient["conditional_accuracy"], 0.10
                ),
                "high_confidence_recall_median": _quantile(
                    sufficient["high_confidence_recall"], 0.50
                ),
                "high_confidence_recall_q10": _quantile(
                    sufficient["high_confidence_recall"], 0.10
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_conditional_contexts_repeated(
    *,
    atlas_dir: Path,
    analysis_work_dir: Path,
    diagnostic_dir: Path,
    out_dir: Path,
    projection_params: ReferenceProjectionParams,
    validation_params: ConditionalContextValidationParams = ConditionalContextValidationParams(),
    repeats: int = 10,
    minimum_repeat_pass_rate: float = 0.80,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Repeat fixed-label validation and admit only reproducible contexts."""

    if int(repeats) < 2:
        raise ValueError("Repeated validation requires at least two repeats")
    if not 0 < float(minimum_repeat_pass_rate) <= 1:
        raise ValueError("minimum_repeat_pass_rate must be in (0, 1]")
    out_dir = Path(out_dir).expanduser().resolve()
    existing = [
        out_dir / name
        for name in REPEATED_EXPECTED_OUTPUTS
        if (out_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Repeated conditional validation output exists; pass --overwrite or "
            f"use a new directory: {existing}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_root = out_dir / "repeats"
    repeat_root.mkdir(parents=True, exist_ok=True)

    context_parts: list[pd.DataFrame] = []
    platform_parts: list[pd.DataFrame] = []
    source_parts: list[pd.DataFrame] = []
    repeat_rows: list[dict[str, object]] = []
    candidate_catalog = pd.DataFrame()
    for repeat_index in range(int(repeats)):
        repeat_number = repeat_index + 1
        repeat_dir = repeat_root / f"repeat_{repeat_number:02d}"
        repeat_params = ConditionalContextValidationParams(
            **{
                **asdict(validation_params),
                "random_seed": int(validation_params.random_seed) + repeat_index,
            }
        )
        outputs = validate_conditional_contexts(
            atlas_dir=atlas_dir,
            analysis_work_dir=analysis_work_dir,
            diagnostic_dir=diagnostic_dir,
            out_dir=repeat_dir,
            projection_params=projection_params,
            validation_params=repeat_params,
            metadata_path=metadata_path,
            overwrite=overwrite,
        )
        candidate_catalog = pd.read_csv(outputs["candidate_catalog"], sep="\t")
        context = pd.read_csv(outputs["context_metrics"], sep="\t")
        context.insert(0, "repeat", repeat_number)
        context["random_seed"] = repeat_params.random_seed
        context_parts.append(context)
        platform = pd.read_csv(outputs["platform_metrics"], sep="\t")
        if not platform.empty:
            platform.insert(0, "repeat", repeat_number)
            platform_parts.append(platform)
        source = pd.read_csv(outputs["source_metrics"], sep="\t")
        if not source.empty:
            source.insert(0, "repeat", repeat_number)
            source_parts.append(source)
        repeat_rows.append(
            {
                "repeat": repeat_number,
                "random_seed": repeat_params.random_seed,
                "n_candidates_passed": int(
                    context["conditional_layer_eligible"].fillna(False).sum()
                ),
                "repeat_dir": str(repeat_dir),
            }
        )

    repeat_context = pd.concat(context_parts, ignore_index=True, sort=False)
    repeat_platform = (
        pd.concat(platform_parts, ignore_index=True, sort=False)
        if platform_parts
        else pd.DataFrame()
    )
    repeat_source = (
        pd.concat(source_parts, ignore_index=True, sort=False)
        if source_parts
        else pd.DataFrame()
    )
    stability = _aggregate_context_repeats(
        repeat_context,
        repeats=int(repeats),
        minimum_repeat_pass_rate=float(minimum_repeat_pass_rate),
        params=validation_params,
    )
    platform_stability = _aggregate_stratum_repeats(
        repeat_platform,
        stratum="platform_family_refined",
        repeats=int(repeats),
    )
    source_stability = _aggregate_stratum_repeats(
        repeat_source,
        stratum="source_group_refined",
        repeats=int(repeats),
    )
    admitted = stability[stability["conditional_layer_eligible"]].copy()
    admitted = admitted.merge(
        candidate_catalog.drop(
            columns=[
                column
                for column in (
                    "root",
                    "final_context",
                    "community_key",
                    "validation_role",
                    "evidence_class",
                )
                if column in candidate_catalog
            ]
        ),
        on="conditional_reference_key",
        how="left",
        validate="one_to_one",
    )
    admitted["reference_level"] = "conditional"
    admitted["assignment_method"] = "core_first_conditional_knn"
    root_summary = (
        stability.groupby("root", sort=True)
        .agg(
            n_structural_candidates=("conditional_reference_key", "size"),
            n_primary_candidates=("passes_independent_evidence", "sum"),
            n_admitted=("conditional_layer_eligible", "sum"),
            n_candidate_samples=("n_full_members", "sum"),
        )
        .reset_index()
    )
    repeat_summary = pd.DataFrame(repeat_rows)
    summary_counts = {
        "n_repeats": int(repeats),
        "n_structural_candidates": int(len(stability)),
        "n_primary_candidates": int(stability["passes_independent_evidence"].sum()),
        "n_negative_controls": int((~stability["passes_independent_evidence"]).sum()),
        "n_admitted_conditional_contexts": int(len(admitted)),
        "n_structural_candidate_samples": int(stability["n_full_members"].sum()),
        "n_primary_candidate_samples": int(
            stability.loc[stability["passes_independent_evidence"], "n_full_members"].sum()
        ),
        "n_admitted_definition_samples": int(admitted["n_full_members"].sum()),
    }

    outputs = {
        "candidate_catalog": out_dir / "candidate_catalog.tsv",
        "repeat_context_metrics": out_dir / "repeat_context_metrics.tsv",
        "context_stability": out_dir / "context_stability.tsv",
        "platform_stability": out_dir / "platform_stability.tsv",
        "source_stability": out_dir / "source_stability.tsv",
        "root_summary": out_dir / "root_summary.tsv",
        "conditional_contexts": out_dir / "conditional_contexts.tsv",
        "repeat_summary": out_dir / "repeat_summary.tsv",
    }
    candidate_catalog.to_csv(outputs["candidate_catalog"], sep="\t", index=False)
    repeat_context.to_csv(outputs["repeat_context_metrics"], sep="\t", index=False)
    stability.to_csv(outputs["context_stability"], sep="\t", index=False)
    platform_stability.to_csv(outputs["platform_stability"], sep="\t", index=False)
    source_stability.to_csv(outputs["source_stability"], sep="\t", index=False)
    root_summary.to_csv(outputs["root_summary"], sep="\t", index=False)
    admitted.to_csv(outputs["conditional_contexts"], sep="\t", index=False)
    repeat_summary.to_csv(outputs["repeat_summary"], sep="\t", index=False)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cnvsim_conditional_context_repeated_validation_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "validation_design": {
                    "representation": "frozen_full_atlas_IR",
                    "labels": "repeated_train_only_fixed_conditional_labels",
                    "routing": "core_first_then_conditional_rescue",
                    "candidate_selection_uses_diagnostic_labels": False,
                    "repeats": int(repeats),
                    "minimum_repeat_pass_rate": float(minimum_repeat_pass_rate),
                    "repeat_seed_start": int(validation_params.random_seed),
                },
                "projection": asdict(projection_params),
                "admission": asdict(validation_params),
                "counts": summary_counts,
                "repeat_root": str(repeat_root),
                "outputs": {name: path.name for name, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "manifest": manifest_path}
