from __future__ import annotations

"""Cross-calibrated sensitivity analysis for projection distance thresholds."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from cnvsim.analysis.config import ReferenceProjectionParams


CALIBRATION_COLUMNS = {
    "analysis_id",
    "root",
    "predicted_context",
    "true_context",
    "top_vote_fraction",
    "vote_margin",
    "mean_neighbor_distance",
    "correct",
}

QUERY_SCORE_COLUMNS = [
    "analysis_id",
    "root",
    "source_final_context",
    "predicted_context",
    "top_vote_fraction",
    "second_vote_fraction",
    "vote_margin",
    "mean_neighbor_distance",
    "min_neighbor_distance",
    "n_neighbors_used",
]


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(sorted(set(float(value) for value in quantiles)))
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Distance quantiles must be unique values in (0, 1)")
    return values


def _assign_stratified_folds(
    samples: pd.DataFrame,
    *,
    n_folds: int,
    random_seed: int,
) -> pd.Series:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    folds = pd.Series(-1, index=samples.index, dtype=int)
    for (root, context), index in samples.groupby(
        ["root", "true_context"], sort=True
    ).groups.items():
        context_index = np.asarray(list(index), dtype=int)
        if len(context_index) < n_folds:
            raise ValueError(
                f"[{root}/CTX{context}] {len(context_index)} samples cannot "
                f"support {n_folds}-fold calibration"
            )
        token = zlib.crc32(f"{root}:{int(context)}".encode("utf-8"))
        rng = np.random.default_rng(int(random_seed) + int(token))
        shuffled = rng.permutation(context_index)
        folds.loc[shuffled] = np.arange(len(shuffled)) % int(n_folds)
    if folds.lt(0).any():
        raise ValueError("Some calibration samples were not assigned to a fold")
    return folds


def build_projection_thresholds(
    calibration_samples: pd.DataFrame,
    *,
    distance_quantile: float,
    params: ReferenceProjectionParams,
) -> pd.DataFrame:
    """Build context-specific limits from a calibration subset."""

    rows: list[dict[str, object]] = []
    for (root, context), members in calibration_samples.groupby(
        ["root", "true_context"], sort=True
    ):
        correct = members[members["correct"].astype(bool)]
        loo_accuracy = float(members["correct"].astype(bool).mean())
        available = (
            len(correct) >= params.min_correct_calibration_samples
            and loo_accuracy >= params.min_context_loo_accuracy
        )
        rows.append(
            {
                "root": str(root),
                "context": int(context),
                "distance_quantile": float(distance_quantile),
                "n_calibration_samples": int(len(members)),
                "n_correct_calibration_samples": int(len(correct)),
                "loo_accuracy": loo_accuracy,
                "min_vote_fraction": (
                    float(
                        correct["top_vote_fraction"].quantile(
                            params.lower_confidence_quantile
                        )
                    )
                    if available
                    else np.nan
                ),
                "min_vote_margin": (
                    float(
                        correct["vote_margin"].quantile(
                            params.lower_confidence_quantile
                        )
                    )
                    if available
                    else np.nan
                ),
                "max_mean_neighbor_distance": (
                    float(
                        correct["mean_neighbor_distance"].quantile(
                            float(distance_quantile)
                        )
                    )
                    if available
                    else np.nan
                ),
                "calibration_available": bool(available),
            }
        )
    return pd.DataFrame(rows)


def classify_projection_scores(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply calibrated vote, margin and distance limits to scored samples."""

    result = scores.merge(
        thresholds,
        left_on=["root", "predicted_context"],
        right_on=["root", "context"],
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
    available = result["calibration_available"].fillna(False).astype(bool)
    result["assignment_status"] = np.select(
        [
            ~available | ~result["passes_distance"].fillna(False),
            ~result["passes_vote"].fillna(False)
            | ~result["passes_margin"].fillna(False),
        ],
        ["unresolved", "ambiguous"],
        default="projected_assignment",
    )
    return result


def _context_performance(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = frame[frame["assignment_status"].eq("projected_assignment")]
    true_counts = (
        frame.groupby(["root", "true_context"])
        .size()
        .rename("n_true")
        .reset_index()
        .rename(columns={"true_context": "context"})
    )
    correct_counts = (
        assigned[assigned["correct"]]
        .groupby(["root", "true_context"])
        .size()
        .rename("n_correct_assigned")
        .reset_index()
        .rename(columns={"true_context": "context"})
    )
    predicted_counts = (
        assigned.groupby(["root", "predicted_context"])
        .size()
        .rename("n_predicted_assigned")
        .reset_index()
        .rename(columns={"predicted_context": "context"})
    )
    correct_predicted = (
        assigned[assigned["correct"]]
        .groupby(["root", "predicted_context"])
        .size()
        .rename("n_correct_predicted")
        .reset_index()
        .rename(columns={"predicted_context": "context"})
    )
    result = true_counts.merge(correct_counts, on=["root", "context"], how="left")
    result = result.merge(predicted_counts, on=["root", "context"], how="left")
    result = result.merge(correct_predicted, on=["root", "context"], how="left")
    count_columns = [
        "n_correct_assigned",
        "n_predicted_assigned",
        "n_correct_predicted",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["high_confidence_recall"] = (
        result["n_correct_assigned"] / result["n_true"]
    )
    result["high_confidence_precision"] = np.where(
        result["n_predicted_assigned"].gt(0),
        result["n_correct_predicted"] / result["n_predicted_assigned"],
        np.nan,
    )
    return result


def _anchor_summary(
    frame: pd.DataFrame,
    context_performance: pd.DataFrame,
) -> dict[str, object]:
    assigned = frame["assignment_status"].eq("projected_assignment")
    correct_assigned = assigned & frame["correct"].astype(bool)
    context_precision = context_performance["high_confidence_precision"].dropna()
    conditional_precision = (
        float(frame.loc[assigned, "correct"].astype(bool).mean())
        if assigned.any()
        else np.nan
    )
    return {
        "n_anchor_evaluations": int(len(frame)),
        "n_high_confidence": int(assigned.sum()),
        "high_confidence_coverage": float(assigned.mean()),
        "high_confidence_precision": conditional_precision,
        "high_confidence_error_rate": (
            float(1.0 - conditional_precision)
            if pd.notna(conditional_precision)
            else np.nan
        ),
        "incorrect_high_confidence_fraction": float(
            (assigned & ~frame["correct"].astype(bool)).mean()
        ),
        "correct_high_confidence_recall": float(correct_assigned.mean()),
        "ambiguous_fraction": float(
            frame["assignment_status"].eq("ambiguous").mean()
        ),
        "unresolved_fraction": float(
            frame["assignment_status"].eq("unresolved").mean()
        ),
        "minimum_context_precision": (
            float(context_precision.min()) if not context_precision.empty else np.nan
        ),
        "q10_context_precision": (
            float(context_precision.quantile(0.10))
            if not context_precision.empty
            else np.nan
        ),
        "q10_context_recall": float(
            context_performance["high_confidence_recall"].quantile(0.10)
        ),
        "n_contexts_without_high_confidence_prediction": int(
            context_performance["n_predicted_assigned"].eq(0).sum()
        ),
    }


def evaluate_projection_thresholds(
    *,
    projection_dir: Path,
    reference_contexts_dir: Path,
    out_dir: Path,
    params: ReferenceProjectionParams,
    distance_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    n_folds: int = 5,
    random_seed: int = 2701,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Evaluate distance quantiles with cross-calibrated anchors and queries."""

    projection_dir = Path(projection_dir)
    reference_contexts_dir = Path(reference_contexts_dir)
    out_dir = Path(out_dir)
    quantiles = _validate_quantiles(distance_quantiles)
    baseline_quantile = float(params.upper_distance_quantile)
    if not any(np.isclose(value, baseline_quantile) for value in quantiles):
        raise ValueError(
            "Distance quantiles must include the configured baseline "
            f"upper_distance_quantile={baseline_quantile:g}"
        )
    expected_outputs = (
        "anchor_cv_summary.tsv",
        "anchor_cv_by_root.tsv",
        "anchor_cv_by_context.tsv",
        "query_sensitivity_summary.tsv",
        "query_sensitivity_by_root.tsv",
        "query_sensitivity_by_recovery_class.tsv",
        "anchor_cv_assignments.parquet",
        "query_sensitivity_assignments.parquet",
        "manifest.json",
    )
    existing = [out_dir / name for name in expected_outputs if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Threshold-sensitivity output already exists; use a new --out-dir "
            f"or pass --overwrite. Existing files: {existing}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = projection_dir / "projection_calibration_samples.parquet"
    assignments_path = projection_dir / "projection_assignments.parquet"
    lineage_path = reference_contexts_dir / "context_lineage.tsv"
    for required in (calibration_path, assignments_path, lineage_path):
        if not required.exists():
            raise FileNotFoundError(required)

    calibration = pd.read_parquet(calibration_path)
    missing = sorted(CALIBRATION_COLUMNS - set(calibration.columns))
    if missing:
        raise KeyError(f"Calibration samples are missing columns: {missing}")
    calibration["analysis_id"] = calibration["analysis_id"].astype(str)
    calibration["root"] = calibration["root"].astype(str)
    calibration["predicted_context"] = calibration["predicted_context"].astype(int)
    calibration["true_context"] = calibration["true_context"].astype(int)
    calibration["correct"] = calibration["predicted_context"].eq(
        calibration["true_context"]
    )
    calibration["cv_fold"] = _assign_stratified_folds(
        calibration, n_folds=n_folds, random_seed=random_seed
    )

    cv_parts: list[pd.DataFrame] = []
    for distance_quantile in quantiles:
        for fold in range(n_folds):
            train = calibration[calibration["cv_fold"].ne(fold)].copy()
            evaluation = calibration[calibration["cv_fold"].eq(fold)].copy()
            thresholds = build_projection_thresholds(
                train,
                distance_quantile=distance_quantile,
                params=params,
            )
            classified = classify_projection_scores(evaluation, thresholds)
            classified["distance_quantile"] = float(distance_quantile)
            classified["cv_fold"] = int(fold)
            cv_parts.append(classified)
    anchor_cv = pd.concat(cv_parts, ignore_index=True, sort=False)

    context_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    for distance_quantile, frame in anchor_cv.groupby("distance_quantile", sort=True):
        context_performance = _context_performance(frame)
        context_performance.insert(0, "distance_quantile", float(distance_quantile))
        context_parts.append(context_performance)
        summary_rows.append(
            {
                "distance_quantile": float(distance_quantile),
                **_anchor_summary(frame, context_performance),
            }
        )
        for root, root_frame in frame.groupby("root", sort=True):
            root_contexts = context_performance[
                context_performance["root"].eq(root)
            ]
            root_rows.append(
                {
                    "distance_quantile": float(distance_quantile),
                    "root": str(root),
                    **_anchor_summary(root_frame, root_contexts),
                }
            )
    anchor_context = pd.concat(context_parts, ignore_index=True, sort=False)
    anchor_summary = pd.DataFrame(summary_rows)
    anchor_by_root = pd.DataFrame(root_rows)

    query_scores = pd.read_parquet(assignments_path)
    missing_query_columns = sorted(
        set(QUERY_SCORE_COLUMNS + ["assignment_status"])
        - set(query_scores.columns)
    )
    if missing_query_columns:
        raise KeyError(
            f"Projection assignments are missing columns: {missing_query_columns}"
        )
    query_scores = query_scores[
        query_scores["predicted_context"].notna()
        & query_scores["root"].notna()
    ].copy()
    query_scores["analysis_id"] = query_scores["analysis_id"].astype(str)
    query_scores["root"] = query_scores["root"].astype(str)
    query_scores["predicted_context"] = query_scores["predicted_context"].astype(int)
    canonical_status = query_scores[
        ["analysis_id", "assignment_status"]
    ].rename(columns={"assignment_status": "canonical_assignment_status"})
    # Canonical assignments contain their q90 threshold columns. Retain only
    # score geometry so each candidate quantile is applied independently.
    query_scores = query_scores[QUERY_SCORE_COLUMNS].copy()

    query_parts: list[pd.DataFrame] = []
    for distance_quantile in quantiles:
        thresholds = build_projection_thresholds(
            calibration,
            distance_quantile=distance_quantile,
            params=params,
        )
        classified = classify_projection_scores(query_scores, thresholds)
        classified["distance_quantile"] = float(distance_quantile)
        query_parts.append(classified)
    query_sensitivity = pd.concat(query_parts, ignore_index=True, sort=False)

    lineage = pd.read_csv(lineage_path, sep="\t")
    lineage["root"] = lineage["root"].astype(str)
    lineage["final_context"] = pd.to_numeric(
        lineage["final_context"], errors="raise"
    ).astype(int)
    query_sensitivity = query_sensitivity.merge(
        lineage[["root", "final_context", "recovery_class"]].rename(
            columns={"final_context": "source_final_context"}
        ),
        on=["root", "source_final_context"],
        how="left",
        validate="many_to_one",
    )

    query_summary_rows: list[dict[str, object]] = []
    query_root_rows: list[dict[str, object]] = []
    query_recovery_rows: list[dict[str, object]] = []
    for distance_quantile, frame in query_sensitivity.groupby(
        "distance_quantile", sort=True
    ):
        counts = frame["assignment_status"].value_counts()
        query_summary_rows.append(
            {
                "distance_quantile": float(distance_quantile),
                "n_queries": int(len(frame)),
                "n_projected": int(counts.get("projected_assignment", 0)),
                "n_ambiguous": int(counts.get("ambiguous", 0)),
                "n_unresolved": int(counts.get("unresolved", 0)),
                "projected_fraction": float(
                    frame["assignment_status"].eq("projected_assignment").mean()
                ),
                "ambiguous_fraction": float(
                    frame["assignment_status"].eq("ambiguous").mean()
                ),
                "unresolved_fraction": float(
                    frame["assignment_status"].eq("unresolved").mean()
                ),
            }
        )
        for root, root_frame in frame.groupby("root", sort=True):
            root_counts = root_frame["assignment_status"].value_counts()
            query_root_rows.append(
                {
                    "distance_quantile": float(distance_quantile),
                    "root": str(root),
                    "n_queries": int(len(root_frame)),
                    "n_projected": int(root_counts.get("projected_assignment", 0)),
                    "n_ambiguous": int(root_counts.get("ambiguous", 0)),
                    "n_unresolved": int(root_counts.get("unresolved", 0)),
                    "projected_fraction": float(
                        root_frame["assignment_status"]
                        .eq("projected_assignment")
                        .mean()
                    ),
                }
            )
        for recovery_class, recovery_frame in frame.groupby(
            "recovery_class", dropna=False, sort=True
        ):
            recovery_counts = recovery_frame["assignment_status"].value_counts()
            query_recovery_rows.append(
                {
                    "distance_quantile": float(distance_quantile),
                    "recovery_class": (
                        str(recovery_class)
                        if pd.notna(recovery_class)
                        else "missing"
                    ),
                    "n_queries": int(len(recovery_frame)),
                    "n_projected": int(
                        recovery_counts.get("projected_assignment", 0)
                    ),
                    "n_ambiguous": int(recovery_counts.get("ambiguous", 0)),
                    "n_unresolved": int(recovery_counts.get("unresolved", 0)),
                    "projected_fraction": float(
                        recovery_frame["assignment_status"]
                        .eq("projected_assignment")
                        .mean()
                    ),
                }
            )
    query_summary = pd.DataFrame(query_summary_rows)
    query_by_root = pd.DataFrame(query_root_rows)
    query_by_recovery = pd.DataFrame(query_recovery_rows)

    baseline = query_sensitivity[
        np.isclose(query_sensitivity["distance_quantile"], baseline_quantile)
    ][["analysis_id", "assignment_status"]]
    baseline = baseline.merge(
        canonical_status, on="analysis_id", how="left", validate="one_to_one"
    )
    baseline_mismatches = int(
        baseline["assignment_status"].ne(
            baseline["canonical_assignment_status"]
        ).sum()
    )
    query_summary["delta_projected_from_baseline"] = (
        query_summary["n_projected"]
        - int(
            query_summary.loc[
                np.isclose(query_summary["distance_quantile"], baseline_quantile),
                "n_projected",
            ].iloc[0]
        )
    )

    outputs = {
        "anchor_summary": out_dir / "anchor_cv_summary.tsv",
        "anchor_by_root": out_dir / "anchor_cv_by_root.tsv",
        "anchor_by_context": out_dir / "anchor_cv_by_context.tsv",
        "query_summary": out_dir / "query_sensitivity_summary.tsv",
        "query_by_root": out_dir / "query_sensitivity_by_root.tsv",
        "query_by_recovery": out_dir / "query_sensitivity_by_recovery_class.tsv",
        "anchor_assignments": out_dir / "anchor_cv_assignments.parquet",
        "query_assignments": out_dir / "query_sensitivity_assignments.parquet",
    }
    anchor_summary.to_csv(outputs["anchor_summary"], sep="\t", index=False)
    anchor_by_root.to_csv(outputs["anchor_by_root"], sep="\t", index=False)
    anchor_context.to_csv(outputs["anchor_by_context"], sep="\t", index=False)
    query_summary.to_csv(outputs["query_summary"], sep="\t", index=False)
    query_by_root.to_csv(outputs["query_by_root"], sep="\t", index=False)
    query_by_recovery.to_csv(outputs["query_by_recovery"], sep="\t", index=False)
    anchor_cv.to_parquet(outputs["anchor_assignments"], index=False)
    query_sensitivity.to_parquet(outputs["query_assignments"], index=False)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "projection_dir": str(projection_dir),
                "reference_contexts_dir": str(reference_contexts_dir),
                "projection": asdict(params),
                "distance_quantiles": list(quantiles),
                "cross_calibration_folds": int(n_folds),
                "random_seed": int(random_seed),
                "baseline_status_mismatches": baseline_mismatches,
                "counts": {
                    "anchor_samples": int(calibration["analysis_id"].nunique()),
                    "anchor_contexts": int(
                        calibration[["root", "true_context"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "subtype_queries": int(query_scores["analysis_id"].nunique()),
                },
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "manifest": manifest_path}
