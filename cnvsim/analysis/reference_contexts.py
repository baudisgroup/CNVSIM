from __future__ import annotations

"""Classify refined communities as reference anchors or non-anchor queries."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.analysis.config import ReferenceContextParams


REQUIRED_STABILITY_COLUMNS = {
    "root",
    "final_context",
    "repeats",
    "detection_rate",
    "conditional_recall_q10",
    "conditional_precision_q10",
    "conditional_jaccard_q10",
    "n_samples",
}


def classify_context_stability(
    stability: pd.DataFrame,
    *,
    params: ReferenceContextParams,
    expected_repeats: int,
) -> pd.DataFrame:
    """Apply explicit recovery gates without using signature evidence."""

    missing = sorted(REQUIRED_STABILITY_COLUMNS - set(stability.columns))
    if missing:
        raise KeyError(f"Context stability table is missing columns: {missing}")
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be positive")

    result = stability.copy()
    result["root"] = result["root"].astype(str)
    result["final_context"] = pd.to_numeric(
        result["final_context"], errors="raise"
    ).astype(int)
    numeric_columns = [
        "repeats",
        "detection_rate",
        "conditional_recall_q10",
        "conditional_precision_q10",
        "conditional_jaccard_q10",
        "n_samples",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["passes_repeat_coverage"] = result["repeats"].ge(expected_repeats)
    result["passes_context_size"] = result["n_samples"].ge(
        params.min_context_size
    )
    result["passes_detection"] = result["detection_rate"].ge(
        params.min_detection_rate
    )
    result["passes_conditional_recall"] = result[
        "conditional_recall_q10"
    ].ge(params.min_conditional_recall_q10)
    result["passes_conditional_precision"] = result[
        "conditional_precision_q10"
    ].ge(params.min_conditional_precision_q10)
    result["passes_conditional_jaccard"] = result[
        "conditional_jaccard_q10"
    ].ge(params.min_conditional_jaccard_q10)
    gate_columns = [
        "passes_repeat_coverage",
        "passes_context_size",
        "passes_detection",
        "passes_conditional_recall",
        "passes_conditional_precision",
        "passes_conditional_jaccard",
    ]
    result["is_reference_anchor"] = result[gate_columns].all(axis=1)

    failure_names = {
        "passes_repeat_coverage": "insufficient_repeats",
        "passes_context_size": "undersized_context",
        "passes_detection": "low_detection_rate",
        "passes_conditional_recall": "low_conditional_recall",
        "passes_conditional_precision": "low_conditional_precision",
        "passes_conditional_jaccard": "low_conditional_jaccard",
    }
    result["stability_failure_reason"] = [
        "|".join(
            failure_names[column]
            for column in gate_columns
            if not bool(row[column])
        )
        for _, row in result.iterrows()
    ]
    conditional_pass = result[
        [
            "passes_conditional_recall",
            "passes_conditional_precision",
            "passes_conditional_jaccard",
        ]
    ].all(axis=1)
    result["recovery_class"] = np.select(
        [
            result["is_reference_anchor"],
            result["detection_rate"].fillna(0).eq(0),
            ~result["passes_detection"] & conditional_pass,
        ],
        ["stable_anchor", "not_detected", "intermittent"],
        default="boundary_unstable",
    )
    result["reference_status"] = np.where(
        result["is_reference_anchor"], "stable_anchor", "non_reference"
    )
    return result.sort_values(["root", "final_context"]).reset_index(drop=True)


def build_reference_contexts(
    *,
    final_partitions_dir: Path,
    stability: pd.DataFrame,
    out_dir: Path,
    params: ReferenceContextParams,
    expected_repeats: int,
    roots: list[str] | None = None,
) -> dict[str, Path]:
    """Create anchor-only partitions and preserve non-anchor sample lineage."""

    final_partitions_dir = Path(final_partitions_dir)
    out_dir = Path(out_dir)
    anchor_partitions_dir = out_dir / "anchor_partitions"
    anchor_partitions_dir.mkdir(parents=True, exist_ok=True)

    classified = classify_context_stability(
        stability, params=params, expected_repeats=expected_repeats
    )
    selected_roots = (
        sorted({str(root) for root in roots})
        if roots
        else sorted(classified["root"].unique())
    )
    classified = classified[classified["root"].isin(selected_roots)].copy()
    if classified.empty:
        raise ValueError("No refined contexts remain for reference classification")
    if classified.duplicated(["root", "final_context"]).any():
        raise ValueError("Context stability rows must be unique by root/final_context")

    sample_parts: list[pd.DataFrame] = []
    root_rows: list[dict[str, object]] = []
    for root in selected_roots:
        root_stability = classified[classified["root"].eq(root)].copy()
        if root_stability.empty:
            raise ValueError(f"[{root}] no context stability rows found")
        partition_path = final_partitions_dir / f"{root}_final_partition.parquet"
        if not partition_path.exists():
            raise FileNotFoundError(partition_path)
        partition = pd.read_parquet(partition_path)
        required = {"analysis_id", "provisional_context", "final_context"}
        missing = sorted(required - set(partition.columns))
        if missing:
            raise KeyError(f"[{root}] final partition is missing columns: {missing}")
        partition = partition[
            ["analysis_id", "provisional_context", "final_context"]
        ].copy()
        partition["analysis_id"] = partition["analysis_id"].astype(str)
        partition["provisional_context"] = pd.to_numeric(
            partition["provisional_context"], errors="raise"
        ).astype(int)
        partition["final_context"] = pd.to_numeric(
            partition["final_context"], errors="raise"
        ).astype(int)

        observed_contexts = set(
            partition.loc[partition["final_context"].ge(0), "final_context"]
        )
        classified_contexts = set(root_stability["final_context"])
        if observed_contexts != classified_contexts:
            raise ValueError(
                f"[{root}] partition/stability context mismatch: "
                f"partition_only={sorted(observed_contexts - classified_contexts)}, "
                f"stability_only={sorted(classified_contexts - observed_contexts)}"
            )

        anchor_contexts = set(
            root_stability.loc[
                root_stability["is_reference_anchor"], "final_context"
            ].astype(int)
        )
        partition["root"] = root
        partition["source_final_context"] = partition["final_context"]
        partition["context"] = pd.Series(pd.NA, index=partition.index, dtype="Int64")
        anchor_mask = partition["final_context"].isin(anchor_contexts)
        outlier_mask = partition["final_context"].lt(0)
        is_root_only_reference = (
            not anchor_contexts and root in set(params.root_only_fallback_roots)
        )
        root_only_mask = (
            ~outlier_mask
            if is_root_only_reference
            else pd.Series(False, index=partition.index)
        )
        query_mask = ~anchor_mask & ~root_only_mask & ~outlier_mask
        partition.loc[anchor_mask, "context"] = partition.loc[
            anchor_mask, "final_context"
        ].astype("Int64")
        partition["assignment_status"] = np.select(
            [anchor_mask, root_only_mask, query_mask, outlier_mask],
            [
                "anchor_member",
                "root_only_member",
                "non_anchor_query",
                "outlier",
            ],
            default="unclassified",
        )
        partition["reference_id"] = pd.Series(
            pd.NA, index=partition.index, dtype="string"
        )
        partition.loc[anchor_mask, "reference_id"] = [
            f"{root}-CTX{int(context)}"
            for context in partition.loc[anchor_mask, "context"]
        ]
        partition.loc[root_only_mask, "reference_id"] = f"{root}-ROOT"
        sample_parts.append(
            partition[
                [
                    "analysis_id",
                    "root",
                    "provisional_context",
                    "source_final_context",
                    "context",
                    "assignment_status",
                    "reference_id",
                ]
            ]
        )

        anchor_partition_path = anchor_partitions_dir / f"{root}_contexts.parquet"
        if anchor_partition_path.exists():
            anchor_partition_path.unlink()
        anchor_partition = partition.loc[anchor_mask, ["analysis_id", "context"]]
        anchor_partition["context"] = anchor_partition["context"].astype(int)
        if not anchor_partition.empty:
            anchor_partition.to_parquet(anchor_partition_path, index=False)

        n_inliers = int((~outlier_mask).sum())
        n_anchor_samples = int(anchor_mask.sum())
        n_anchor_contexts = int(len(anchor_contexts))
        anchor_fraction = n_anchor_samples / n_inliers if n_inliers else np.nan
        if is_root_only_reference:
            disposition = "root_only_reference"
        elif n_anchor_contexts >= 2:
            disposition = "multi_context_reference"
        elif n_anchor_contexts == 1:
            disposition = "single_context_reference"
        else:
            disposition = "no_stable_reference"
        root_rows.append(
            {
                "root": root,
                "n_refined_contexts": int(len(root_stability)),
                "n_reference_anchors": n_anchor_contexts,
                "n_non_reference_communities": int(
                    len(root_stability) - n_anchor_contexts
                ),
                "n_inlier_samples": n_inliers,
                "n_anchor_samples": n_anchor_samples,
                "n_non_anchor_samples": int(query_mask.sum()),
                "n_root_only_samples": int(root_only_mask.sum()),
                "n_outlier_samples": int(outlier_mask.sum()),
                "anchor_sample_fraction": anchor_fraction,
                "reference_disposition": disposition,
                "anchor_coverage_status": (
                    "pass"
                    if np.isfinite(anchor_fraction)
                    and anchor_fraction >= params.anchor_coverage_warning
                    else "warning"
                ),
            }
        )

    sample_status = pd.concat(sample_parts, ignore_index=True).sort_values(
        ["root", "analysis_id"]
    )
    root_summary = pd.DataFrame(root_rows).sort_values("root")
    non_reference = classified[~classified["is_reference_anchor"]].copy()
    context_lineage = classified[
        [
            "root",
            "final_context",
            "reference_status",
            "recovery_class",
            "stability_failure_reason",
        ]
    ].copy()
    context_lineage["context"] = pd.Series(
        np.where(
            classified["is_reference_anchor"],
            classified["final_context"],
            pd.NA,
        ),
        dtype="Int64",
    )

    outputs = {
        "context_stability": out_dir / "context_stability.tsv",
        "root_summary": out_dir / "root_summary.tsv",
        "context_lineage": out_dir / "context_lineage.tsv",
        "non_reference_communities": out_dir / "non_reference_communities.tsv",
        "sample_status": out_dir / "sample_reference_status.parquet",
        "non_anchor_samples": out_dir / "non_anchor_samples.parquet",
        "roots_without_stable_reference": (
            out_dir / "roots_without_stable_reference.tsv"
        ),
        "root_only_references": out_dir / "root_only_references.tsv",
        "low_anchor_coverage_roots": out_dir / "low_anchor_coverage_roots.tsv",
    }
    classified.to_csv(outputs["context_stability"], sep="\t", index=False)
    root_summary.to_csv(outputs["root_summary"], sep="\t", index=False)
    context_lineage.to_csv(outputs["context_lineage"], sep="\t", index=False)
    non_reference.to_csv(
        outputs["non_reference_communities"], sep="\t", index=False
    )
    sample_status.to_parquet(outputs["sample_status"], index=False)
    sample_status[sample_status["assignment_status"].eq("non_anchor_query")].to_parquet(
        outputs["non_anchor_samples"], index=False
    )
    root_summary[root_summary["n_reference_anchors"].eq(0)].to_csv(
        outputs["roots_without_stable_reference"], sep="\t", index=False
    )
    root_only = root_summary[
        root_summary["reference_disposition"].eq("root_only_reference")
    ].copy()
    root_only["reference_id"] = root_only["root"].astype(str) + "-ROOT"
    root_only.to_csv(outputs["root_only_references"], sep="\t", index=False)
    root_summary[root_summary["anchor_coverage_status"].eq("warning")].to_csv(
        outputs["low_anchor_coverage_roots"], sep="\t", index=False
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reference_contexts": asdict(params),
                "expected_subsampling_repeats": int(expected_repeats),
                "partitions": str(anchor_partitions_dir),
                "outputs": {key: str(path) for key, path in outputs.items()},
                "counts": {
                    "roots": int(len(root_summary)),
                    "refined_contexts": int(len(classified)),
                    "reference_anchors": int(classified["is_reference_anchor"].sum()),
                    "non_reference_communities": int(
                        (~classified["is_reference_anchor"]).sum()
                    ),
                    "inlier_samples": int(root_summary["n_inlier_samples"].sum()),
                    "anchor_samples": int(root_summary["n_anchor_samples"].sum()),
                    "non_anchor_samples": int(
                        root_summary["n_non_anchor_samples"].sum()
                    ),
                    "root_only_references": int(len(root_only)),
                    "root_only_samples": int(
                        root_summary["n_root_only_samples"].sum()
                    ),
                    "outlier_samples": int(root_summary["n_outlier_samples"].sum()),
                    "roots_without_stable_reference": int(
                        root_summary["n_reference_anchors"].eq(0).sum()
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {
        **outputs,
        "anchor_partitions": anchor_partitions_dir,
        "manifest": manifest_path,
    }
