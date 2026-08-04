from __future__ import annotations

"""Calibrated projection of non-anchor samples into stable CNV contexts."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cnvsim.analysis.config import ReferenceProjectionParams
from cnvsim.analysis.ir import dimension_columns, read_analysis_coordinates


def _vote_candidates(
    labels: np.ndarray,
    distances: np.ndarray,
    *,
    top_n: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n_neighbors = int(len(labels))
    for context in sorted(set(int(value) for value in labels)):
        mask = labels.astype(int) == context
        rows.append(
            {
                "context": context,
                "n_neighbors": int(mask.sum()),
                "vote_fraction": float(mask.sum() / n_neighbors),
                "mean_neighbor_distance": float(np.mean(distances[mask])),
                "min_neighbor_distance": float(np.min(distances[mask])),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["vote_fraction"]),
            float(row["mean_neighbor_distance"]),
            int(row["context"]),
        )
    )
    return rows[: int(top_n)]


def score_reference_knn_with_neighbors(
    *,
    query_ids: list[str],
    query_values: np.ndarray,
    reference_ids: list[str],
    reference_values: np.ndarray,
    reference_labels: np.ndarray,
    metric: str,
    n_neighbors: int,
    top_n: int,
    exclude_self: bool,
    nearest_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score queries and retain the nearest labelled reference samples."""

    if len(reference_ids) < 2:
        raise ValueError("At least two anchor samples are required for projection")
    if int(nearest_n) < 0:
        raise ValueError("nearest_n must be non-negative")
    k = min(int(n_neighbors), len(reference_ids) - int(exclude_self))
    if k < 1:
        raise ValueError("No reference neighbours remain after excluding self")
    search_k = min(k + int(exclude_self), len(reference_ids))
    model = NearestNeighbors(
        n_neighbors=search_k, metric=str(metric), algorithm="auto"
    )
    model.fit(reference_values)
    distances, indices = model.kneighbors(query_values, return_distance=True)

    reference_ids_array = np.asarray(reference_ids, dtype=object)
    candidate_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    for row_index, analysis_id in enumerate(query_ids):
        neighbor_indices = indices[row_index]
        neighbor_distances = distances[row_index]
        if exclude_self:
            keep = reference_ids_array[neighbor_indices] != str(analysis_id)
            neighbor_indices = neighbor_indices[keep][:k]
            neighbor_distances = neighbor_distances[keep][:k]
        else:
            neighbor_indices = neighbor_indices[:k]
            neighbor_distances = neighbor_distances[:k]
        for rank, (reference_index, distance) in enumerate(
            zip(
                neighbor_indices[: int(nearest_n)],
                neighbor_distances[: int(nearest_n)],
                strict=True,
            ),
            start=1,
        ):
            neighbor_rows.append(
                {
                    "analysis_id": str(analysis_id),
                    "neighbor_rank": rank,
                    "reference_analysis_id": str(
                        reference_ids_array[reference_index]
                    ),
                    "reference_context": int(reference_labels[reference_index]),
                    "distance": float(distance),
                }
            )
        candidates = _vote_candidates(
            reference_labels[neighbor_indices],
            neighbor_distances,
            top_n=top_n,
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "analysis_id": str(analysis_id),
                    "candidate_rank": rank,
                    **candidate,
                }
            )
        top = candidates[0]
        second_vote = (
            float(candidates[1]["vote_fraction"]) if len(candidates) > 1 else 0.0
        )
        assignment_rows.append(
            {
                "analysis_id": str(analysis_id),
                "predicted_context": int(top["context"]),
                "top_vote_fraction": float(top["vote_fraction"]),
                "second_vote_fraction": second_vote,
                "vote_margin": float(top["vote_fraction"]) - second_vote,
                "mean_neighbor_distance": float(top["mean_neighbor_distance"]),
                "min_neighbor_distance": float(top["min_neighbor_distance"]),
                "n_neighbors_used": k,
            }
        )
    neighbors = pd.DataFrame(
        neighbor_rows,
        columns=[
            "analysis_id",
            "neighbor_rank",
            "reference_analysis_id",
            "reference_context",
            "distance",
        ],
    )
    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(assignment_rows),
        neighbors,
    )


def score_reference_knn(
    *,
    query_ids: list[str],
    query_values: np.ndarray,
    reference_ids: list[str],
    reference_values: np.ndarray,
    reference_labels: np.ndarray,
    metric: str,
    n_neighbors: int,
    top_n: int,
    exclude_self: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score query coordinates against labelled reference anchors."""

    candidates, assignments, _ = score_reference_knn_with_neighbors(
        query_ids=query_ids,
        query_values=query_values,
        reference_ids=reference_ids,
        reference_values=reference_values,
        reference_labels=reference_labels,
        metric=metric,
        n_neighbors=n_neighbors,
        top_n=top_n,
        exclude_self=exclude_self,
        nearest_n=0,
    )
    return candidates, assignments


def calibrate_anchor_projection(
    *,
    analysis_ids: list[str],
    values: np.ndarray,
    contexts: np.ndarray,
    metric: str,
    n_neighbors: int,
    params: ReferenceProjectionParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate context-specific limits from leave-one-out anchor recovery."""

    _, samples = score_reference_knn(
        query_ids=analysis_ids,
        query_values=values,
        reference_ids=analysis_ids,
        reference_values=values,
        reference_labels=contexts,
        metric=metric,
        n_neighbors=n_neighbors,
        top_n=params.top_candidates,
        exclude_self=True,
    )
    samples["true_context"] = contexts.astype(int)
    samples["correct"] = samples["predicted_context"].eq(samples["true_context"])
    rows: list[dict[str, object]] = []
    for context in sorted(set(int(value) for value in contexts)):
        members = samples[samples["true_context"].eq(context)]
        correct = members[members["correct"]]
        loo_accuracy = float(members["correct"].mean())
        calibration = correct
        available = (
            len(calibration) >= params.min_correct_calibration_samples
            and loo_accuracy >= params.min_context_loo_accuracy
        )
        rows.append(
            {
                "context": context,
                "n_anchor_samples": int(len(members)),
                "n_correct": int(len(correct)),
                "loo_accuracy": loo_accuracy,
                "calibration_scope": "context" if available else "unavailable",
                "n_threshold_samples": int(len(calibration)) if available else 0,
                "min_vote_fraction": (
                    float(
                        calibration["top_vote_fraction"].quantile(
                            params.lower_confidence_quantile
                        )
                    )
                    if available
                    else np.nan
                ),
                "min_vote_margin": (
                    float(
                        calibration["vote_margin"].quantile(
                            params.lower_confidence_quantile
                        )
                    )
                    if available
                    else np.nan
                ),
                "max_mean_neighbor_distance": (
                    float(
                        calibration["mean_neighbor_distance"].quantile(
                            params.upper_distance_quantile
                        )
                    )
                    if available
                    else np.nan
                ),
                "calibration_available": bool(available),
            }
        )
    return pd.DataFrame(rows), samples


def classify_reference_queries(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    root: str,
    reference_key_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Apply calibrated context thresholds to KNN projection scores."""

    result = scores.merge(
        thresholds,
        left_on="predicted_context",
        right_on="context",
        how="left",
        validate="many_to_one",
    ).drop(columns="context")
    result["passes_vote"] = result["top_vote_fraction"].ge(
        result["min_vote_fraction"]
    )
    result["passes_margin"] = result["vote_margin"].ge(
        result["min_vote_margin"]
    )
    result["passes_distance"] = result["mean_neighbor_distance"].le(
        result["max_mean_neighbor_distance"]
    )
    calibrated = result["calibration_available"].fillna(False).astype(bool)
    result["assignment_status"] = np.select(
        [
            ~calibrated | ~result["passes_distance"].fillna(False),
            ~result["passes_vote"].fillna(False)
            | ~result["passes_margin"].fillna(False),
        ],
        ["unresolved", "ambiguous"],
        default="projected_assignment",
    )
    result["assigned_context"] = pd.Series(
        np.where(
            result["assignment_status"].eq("projected_assignment"),
            result["predicted_context"],
            pd.NA,
        ),
        dtype="Int64",
    )
    result["assigned_reference_id"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    accepted = result["assigned_context"].notna()
    result.loc[accepted, "assigned_reference_id"] = [
        (
            reference_key_map[int(context)]
            if reference_key_map is not None
            else f"{root}-CTX{int(context)}"
        )
        for context in result.loc[accepted, "assigned_context"]
    ]
    return result


def project_reference_contexts(
    *,
    reference_contexts_dir: Path,
    ir_dir: Path,
    selected_parameters_path: Path,
    out_dir: Path,
    params: ReferenceProjectionParams,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Project internal non-anchor queries and route root-only references."""

    reference_contexts_dir = Path(reference_contexts_dir)
    ir_dir = Path(ir_dir)
    out_dir = Path(out_dir)
    expected_outputs = (
        "projection_assignments.parquet",
        "projection_candidates.parquet",
        "projection_calibration.tsv",
        "projection_calibration_samples.parquet",
        "root_projection_summary.tsv",
        "manifest.json",
    )
    existing_outputs = [
        out_dir / name for name in expected_outputs if (out_dir / name).exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Projection output already exists; use a new --out-dir or pass "
            f"--overwrite. Existing files: {existing_outputs}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = reference_contexts_dir / "sample_reference_status.parquet"
    for required in (status_path, selected_parameters_path, ir_dir):
        if not Path(required).exists():
            raise FileNotFoundError(required)

    sample_status = pd.read_parquet(status_path)
    sample_status["analysis_id"] = sample_status["analysis_id"].astype(str)
    sample_status["root"] = sample_status["root"].astype(str)
    parameters = pd.read_csv(selected_parameters_path, sep="\t")
    if "root" not in parameters.columns:
        raise KeyError("selected parameters table is missing root")
    parameters["root"] = parameters["root"].astype(str)
    if parameters.duplicated("root").any():
        raise ValueError("selected parameters must contain one row per root")
    parameters = parameters.set_index("root")

    assignment_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    calibration_sample_parts: list[pd.DataFrame] = []
    root_rows: list[dict[str, object]] = []

    root_only = sample_status[
        sample_status["assignment_status"].eq("root_only_member")
    ].copy()
    if not root_only.empty:
        routed = root_only[
            ["analysis_id", "root", "source_final_context", "reference_id"]
        ].rename(columns={"reference_id": "assigned_reference_id"})
        routed["assignment_status"] = "root_only_reference"
        routed["assigned_context"] = pd.Series(
            pd.NA, index=routed.index, dtype="Int64"
        )
        routed["projection_reason"] = "no_stable_within_root_subtype_partition"
        assignment_parts.append(routed)
        for root, frame in routed.groupby("root", sort=True):
            root_rows.append(
                {
                    "root": root,
                    "reference_route": "root_only",
                    "n_anchor_samples": 0,
                    "n_queries": 0,
                    "n_projected": 0,
                    "n_ambiguous": 0,
                    "n_unresolved": 0,
                    "n_root_only": int(len(frame)),
                }
            )

    queries = sample_status[
        sample_status["assignment_status"].eq("non_anchor_query")
    ].copy()
    anchor_roots = set(
        sample_status.loc[
            sample_status["assignment_status"].eq("anchor_member"), "root"
        ]
    )
    query_roots = set(queries["root"])
    for root in sorted(anchor_roots | query_roots):
        root_queries = queries[queries["root"].eq(root)].copy()
        root_anchors = sample_status[
            sample_status["root"].eq(root)
            & sample_status["assignment_status"].eq("anchor_member")
        ].copy()
        if root_anchors.empty:
            unresolved = root_queries[
                ["analysis_id", "root", "source_final_context"]
            ].copy()
            unresolved["assignment_status"] = "unresolved"
            unresolved["assigned_context"] = pd.Series(
                pd.NA, index=unresolved.index, dtype="Int64"
            )
            unresolved["assigned_reference_id"] = pd.Series(
                pd.NA, index=unresolved.index, dtype="string"
            )
            unresolved["projection_reason"] = "no_stable_reference_anchor"
            assignment_parts.append(unresolved)
            root_rows.append(
                {
                    "root": root,
                    "reference_route": "no_reference",
                    "n_anchor_samples": 0,
                    "n_queries": int(len(root_queries)),
                    "n_projected": 0,
                    "n_ambiguous": 0,
                    "n_unresolved": int(len(root_queries)),
                    "n_root_only": 0,
                }
            )
            continue
        if root not in parameters.index:
            raise KeyError(f"[{root}] selected parameters are missing")
        parameter = parameters.loc[root]
        metric = str(parameter["metric"]).lower()
        n_pcs = int(parameter["n_pcs"])
        k = int(parameter["K"])

        coords_path = ir_dir / "perroot" / root / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(coords_path)
        coords = read_analysis_coordinates(coords_path)
        dimensions = dimension_columns(coords)[:n_pcs]
        if len(dimensions) != n_pcs:
            raise ValueError(f"[{root}] requested {n_pcs} dimensions, found {len(dimensions)}")
        anchor_ids = root_anchors["analysis_id"].tolist()
        query_ids = root_queries["analysis_id"].tolist()
        missing = (set(anchor_ids) | set(query_ids)) - set(coords.index)
        if missing:
            raise ValueError(f"[{root}] {len(missing)} samples lack IR coordinates")
        anchor_contexts = root_anchors["context"].astype(int).to_numpy()
        anchor_values = coords.loc[anchor_ids, dimensions].to_numpy(np.float32)
        query_values = coords.loc[query_ids, dimensions].to_numpy(np.float32)

        thresholds, calibration_samples = calibrate_anchor_projection(
            analysis_ids=anchor_ids,
            values=anchor_values,
            contexts=anchor_contexts,
            metric=metric,
            n_neighbors=k,
            params=params,
        )
        threshold_output = thresholds.copy()
        threshold_output.insert(0, "root", root)
        threshold_output["metric"] = metric
        threshold_output["n_pcs"] = n_pcs
        threshold_output["K"] = k
        calibration_samples.insert(1, "root", root)
        calibration_parts.append(threshold_output)
        calibration_sample_parts.append(calibration_samples)

        if root_queries.empty:
            root_rows.append(
                {
                    "root": root,
                    "reference_route": "subtype_projection",
                    "n_anchor_samples": int(len(root_anchors)),
                    "n_queries": 0,
                    "n_projected": 0,
                    "n_ambiguous": 0,
                    "n_unresolved": 0,
                    "n_root_only": 0,
                    "loo_accuracy": float(calibration_samples["correct"].mean()),
                }
            )
            continue

        candidates, scores = score_reference_knn(
            query_ids=query_ids,
            query_values=query_values,
            reference_ids=anchor_ids,
            reference_values=anchor_values,
            reference_labels=anchor_contexts,
            metric=metric,
            n_neighbors=k,
            top_n=params.top_candidates,
            exclude_self=False,
        )
        assignments = classify_reference_queries(scores, thresholds, root=root)
        assignments = root_queries[
            ["analysis_id", "root", "source_final_context"]
        ].merge(assignments, on="analysis_id", how="left", validate="one_to_one")
        assignments["projection_reason"] = np.select(
            [
                assignments["assignment_status"].eq("projected_assignment"),
                assignments["assignment_status"].eq("ambiguous"),
            ],
            ["passes_calibrated_limits", "weak_vote_or_margin"],
            default="outside_calibrated_distance_or_no_calibration",
        )
        assignments["metric"] = metric
        assignments["n_pcs"] = n_pcs
        assignments["K"] = k
        candidates.insert(1, "root", root)
        assignment_parts.append(assignments)
        candidate_parts.append(candidates)
        root_rows.append(
            {
                "root": root,
                "reference_route": "subtype_projection",
                "n_anchor_samples": int(len(root_anchors)),
                "n_queries": int(len(root_queries)),
                "n_projected": int(
                    assignments["assignment_status"].eq("projected_assignment").sum()
                ),
                "n_ambiguous": int(
                    assignments["assignment_status"].eq("ambiguous").sum()
                ),
                "n_unresolved": int(
                    assignments["assignment_status"].eq("unresolved").sum()
                ),
                "n_root_only": 0,
                "loo_accuracy": float(calibration_samples["correct"].mean()),
            }
        )

    assignments = (
        pd.concat(assignment_parts, ignore_index=True, sort=False)
        if assignment_parts
        else pd.DataFrame(
            columns=[
                "analysis_id",
                "root",
                "source_final_context",
                "assignment_status",
                "assigned_context",
                "assigned_reference_id",
                "projection_reason",
            ]
        )
    )
    candidates = (
        pd.concat(candidate_parts, ignore_index=True, sort=False)
        if candidate_parts
        else pd.DataFrame()
    )
    calibration = (
        pd.concat(calibration_parts, ignore_index=True, sort=False)
        if calibration_parts
        else pd.DataFrame()
    )
    calibration_samples = (
        pd.concat(calibration_sample_parts, ignore_index=True, sort=False)
        if calibration_sample_parts
        else pd.DataFrame()
    )
    root_summary = (
        pd.DataFrame(root_rows).sort_values("root").reset_index(drop=True)
        if root_rows
        else pd.DataFrame(
            columns=[
                "root",
                "reference_route",
                "n_anchor_samples",
                "n_queries",
                "n_projected",
                "n_ambiguous",
                "n_unresolved",
                "n_root_only",
            ]
        )
    )

    outputs = {
        "assignments": out_dir / "projection_assignments.parquet",
        "candidates": out_dir / "projection_candidates.parquet",
        "calibration": out_dir / "projection_calibration.tsv",
        "calibration_samples": out_dir / "projection_calibration_samples.parquet",
        "root_summary": out_dir / "root_projection_summary.tsv",
    }
    assignments.to_parquet(outputs["assignments"], index=False)
    candidates.to_parquet(outputs["candidates"], index=False)
    calibration.to_csv(outputs["calibration"], sep="\t", index=False)
    calibration_samples.to_parquet(outputs["calibration_samples"], index=False)
    root_summary.to_csv(outputs["root_summary"], sep="\t", index=False)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "projection_scope": "internal_non_anchor_recovery",
                "independent_validation": False,
                "reference_contexts_dir": str(reference_contexts_dir),
                "ir_dir": str(ir_dir),
                "selected_parameters": str(selected_parameters_path),
                "projection": asdict(params),
                "counts": {
                    "query_or_root_only_samples": int(len(assignments)),
                    "projected_assignments": int(
                        assignments["assignment_status"]
                        .eq("projected_assignment")
                        .sum()
                    ),
                    "ambiguous": int(
                        assignments["assignment_status"].eq("ambiguous").sum()
                    ),
                    "unresolved": int(
                        assignments["assignment_status"].eq("unresolved").sum()
                    ),
                    "root_only": int(
                        assignments["assignment_status"]
                        .eq("root_only_reference")
                        .sum()
                    ),
                },
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "manifest": manifest_path}
