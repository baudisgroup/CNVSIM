from __future__ import annotations

"""Conservative evidence-aware refinement of initial Leiden communities."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from cnvsim.analysis.config import ContextRefinementParams, SubsamplingSelectionParams
from cnvsim.analysis.ir import dimension_columns


class _UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _pairwise_centroid_table(
    coords: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    n_pcs: int,
    distance_quantile: float,
) -> pd.DataFrame:
    dimensions = dimension_columns(coords)[: int(n_pcs)]
    if not dimensions:
        raise ValueError("No D* dimensions found in analysis IR")
    indexed = coords.copy()
    if "analysis_id" in indexed.columns:
        indexed = indexed.set_index("analysis_id")
    indexed.index = indexed.index.astype(str)
    labels = assignments.set_index("analysis_id")["provisional_context"]
    labels.index = labels.index.astype(str)
    aligned = indexed.loc[labels.index, dimensions].copy()
    aligned["context"] = labels.to_numpy(int)
    centroids = aligned.groupby("context")[dimensions].mean().sort_index()

    rows: list[dict[str, object]] = []
    contexts = list(centroids.index.astype(int))
    for index, left in enumerate(contexts):
        for right in contexts[index + 1 :]:
            distance = float(
                np.linalg.norm(
                    centroids.loc[left].to_numpy(float)
                    - centroids.loc[right].to_numpy(float)
                )
            )
            rows.append(
                {
                    "context_a": int(left),
                    "context_b": int(right),
                    "centroid_distance": distance,
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    threshold = float(pairs["centroid_distance"].quantile(distance_quantile))
    nearest: set[tuple[int, int]] = set()
    for context in contexts:
        touching = pairs[
            pairs["context_a"].eq(context) | pairs["context_b"].eq(context)
        ]
        if touching.empty:
            continue
        row = touching.sort_values("centroid_distance").iloc[0]
        nearest.add(tuple(sorted((int(row["context_a"]), int(row["context_b"])))))
    pairs["centroid_distance_threshold"] = threshold
    pairs["is_local_neighbor"] = [
        distance <= threshold or (int(left), int(right)) in nearest
        for left, right, distance in pairs[
            ["context_a", "context_b", "centroid_distance"]
        ].itertuples(index=False, name=None)
    ]
    return pairs


def _comerge_table(context_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contexts = sorted(context_runs["full_cluster"].astype(int).unique())
    for index, left in enumerate(contexts):
        left_runs = context_runs[context_runs["full_cluster"].astype(int).eq(left)][
            ["repeat", "matched_subsample_cluster"]
        ].rename(columns={"matched_subsample_cluster": "left_match"})
        for right in contexts[index + 1 :]:
            right_runs = context_runs[
                context_runs["full_cluster"].astype(int).eq(right)
            ][["repeat", "matched_subsample_cluster"]].rename(
                columns={"matched_subsample_cluster": "right_match"}
            )
            paired = left_runs.merge(right_runs, on="repeat", how="inner")
            rows.append(
                {
                    "context_a": int(left),
                    "context_b": int(right),
                    "n_repeats_compared": int(len(paired)),
                    "comerge_rate": (
                        float(paired["left_match"].eq(paired["right_match"]).mean())
                        if len(paired)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _context_recovery(context_runs: pd.DataFrame) -> pd.DataFrame:
    return (
        context_runs.groupby("full_cluster", as_index=False)
        .agg(
            recovery_repeats=("repeat", "nunique"),
            recall_q10=("recall", lambda values: float(values.quantile(0.10))),
            precision_q10=("precision", lambda values: float(values.quantile(0.10))),
            jaccard_q10=("jaccard", lambda values: float(values.quantile(0.10))),
        )
        .rename(columns={"full_cluster": "context"})
    )


def _technical_context(row: pd.Series, params: ContextRefinementParams) -> bool:
    for annotation in ("platform", "source"):
        top_fraction = pd.to_numeric(
            pd.Series([row.get(f"top_{annotation}_fraction", np.nan)]),
            errors="coerce",
        ).iloc[0]
        enrichment = pd.to_numeric(
            pd.Series([row.get(f"top_{annotation}_enrichment_ratio", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if (
            np.isfinite(top_fraction)
            and np.isfinite(enrichment)
            and top_fraction >= params.technical_top_fraction
            and enrichment >= params.technical_enrichment_ratio
        ):
            return True
    return False


PAIR_EVIDENCE_COLUMNS = [
    "context_a",
    "context_b",
    "evidence_type",
    "feature",
    "event",
    "gene_symbol",
    "supported_context",
    "evidence_direction",
    "context_a_value",
    "context_b_value",
    "pair_delta_a_minus_b",
    "pair_prevalence_delta_a_minus_b",
    "architecture_linked",
    "supports_boundary_veto",
]


def _direction_sign(value: object) -> int:
    text = str(value).strip().lower()
    if text == "enriched":
        return 1
    if text == "depleted":
        return -1
    return 0


def _track1_pair_evidence(
    *,
    left: int,
    right: int,
    profiles: pd.DataFrame,
    supported: pd.DataFrame,
    min_abs_delta: float,
) -> list[dict[str, object]]:
    """Return supported chromosome events that directly separate one pair."""

    if profiles.empty or supported.empty:
        return []
    profile_index = (
        profiles.drop_duplicates("context", keep="first").set_index("context")
        if "context" in profiles.columns
        else profiles
    )
    if left not in profile_index.index or right not in profile_index.index:
        return []
    pair_support = supported[
        supported["context"].astype(int).isin({int(left), int(right)})
    ].copy()
    rows: list[dict[str, object]] = []
    for support in pair_support.itertuples(index=False):
        feature = str(support.feature)
        if feature not in profile_index.columns:
            continue
        value_a = pd.to_numeric(
            pd.Series([profile_index.loc[left, feature]]), errors="coerce"
        ).iloc[0]
        value_b = pd.to_numeric(
            pd.Series([profile_index.loc[right, feature]]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(value_a) or not np.isfinite(value_b):
            continue
        pair_delta = float(value_a - value_b)
        supported_context = int(support.context)
        source_delta = pair_delta if supported_context == left else -pair_delta
        direction = str(support.evidence_direction)
        if (
            _direction_sign(direction) == 0
            or np.sign(source_delta) != _direction_sign(direction)
            or abs(pair_delta) < float(min_abs_delta)
        ):
            continue
        rows.append(
            {
                "context_a": int(left),
                "context_b": int(right),
                "evidence_type": "track1",
                "feature": feature,
                "event": str(support.event),
                "gene_symbol": pd.NA,
                "supported_context": supported_context,
                "evidence_direction": direction,
                "context_a_value": float(value_a),
                "context_b_value": float(value_b),
                "pair_delta_a_minus_b": pair_delta,
                "pair_prevalence_delta_a_minus_b": np.nan,
                "architecture_linked": False,
                "supports_boundary_veto": True,
            }
        )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["_abs_effect"] = frame["pair_delta_a_minus_b"].abs()
    frame = frame.sort_values("_abs_effect", ascending=False).drop_duplicates(
        "feature", keep="first"
    )
    return frame.drop(columns="_abs_effect").to_dict("records")


def _track2_pair_evidence(
    *,
    left: int,
    right: int,
    marker_stats: pd.DataFrame,
    strict_evidence: pd.DataFrame,
    min_abs_adjusted_delta: float,
    min_abs_prevalence_delta: float,
    require_architecture_or_high_level: bool,
) -> list[dict[str, object]]:
    """Return strict gene markers whose effects directly separate one pair."""

    if marker_stats.empty or strict_evidence.empty:
        return []
    key_columns = ["context", "event", "gene_symbol"]
    stats = (
        marker_stats.drop_duplicates(key_columns, keep="first").set_index(key_columns)
        if all(column in marker_stats.columns for column in key_columns)
        else marker_stats
    )
    pair_strict = strict_evidence[
        strict_evidence["context"].astype(int).isin({int(left), int(right)})
    ].copy()
    rows: list[dict[str, object]] = []
    for evidence in pair_strict.itertuples(index=False):
        event = str(evidence.event)
        gene = str(evidence.gene_symbol)
        key_a = (int(left), event, gene)
        key_b = (int(right), event, gene)
        if key_a not in stats.index or key_b not in stats.index:
            continue
        row_a = stats.loc[key_a]
        row_b = stats.loc[key_b]
        adjusted_a = pd.to_numeric(
            pd.Series([row_a.get("adj_cluster_mean", np.nan)]), errors="coerce"
        ).iloc[0]
        adjusted_b = pd.to_numeric(
            pd.Series([row_b.get("adj_cluster_mean", np.nan)]), errors="coerce"
        ).iloc[0]
        prevalence_a = pd.to_numeric(
            pd.Series([row_a.get("raw_prev_cluster", np.nan)]), errors="coerce"
        ).iloc[0]
        prevalence_b = pd.to_numeric(
            pd.Series([row_b.get("raw_prev_cluster", np.nan)]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(adjusted_a) or not np.isfinite(adjusted_b):
            continue
        adjusted_delta = float(adjusted_a - adjusted_b)
        prevalence_delta = (
            float(prevalence_a - prevalence_b)
            if np.isfinite(prevalence_a) and np.isfinite(prevalence_b)
            else np.nan
        )
        supported_context = int(evidence.context)
        source_delta = (
            adjusted_delta if supported_context == left else -adjusted_delta
        )
        direction = str(evidence.evidence_direction)
        effect_ok = abs(adjusted_delta) >= float(min_abs_adjusted_delta) or (
            np.isfinite(prevalence_delta)
            and abs(prevalence_delta) >= float(min_abs_prevalence_delta)
        )
        if (
            _direction_sign(direction) == 0
            or np.sign(source_delta) != _direction_sign(direction)
            or not effect_ok
        ):
            continue
        architecture_linked = getattr(evidence, "architecture_linked", False)
        architecture_linked = (
            bool(architecture_linked) if pd.notna(architecture_linked) else False
        )
        high_level_event = event in {"hldup_frac", "hldel_frac"}
        rows.append(
            {
                "context_a": int(left),
                "context_b": int(right),
                "evidence_type": "track2",
                "feature": f"{gene}|{event}",
                "event": event,
                "gene_symbol": gene,
                "supported_context": supported_context,
                "evidence_direction": direction,
                "context_a_value": float(adjusted_a),
                "context_b_value": float(adjusted_b),
                "pair_delta_a_minus_b": adjusted_delta,
                "pair_prevalence_delta_a_minus_b": prevalence_delta,
                "architecture_linked": architecture_linked,
                "supports_boundary_veto": bool(
                    not require_architecture_or_high_level
                    or architecture_linked
                    or high_level_event
                ),
            }
        )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["_abs_effect"] = np.maximum(
        frame["pair_delta_a_minus_b"].abs(),
        frame["pair_prevalence_delta_a_minus_b"].abs().fillna(0.0),
    )
    frame = frame.sort_values("_abs_effect", ascending=False).drop_duplicates(
        ["event", "gene_symbol"], keep="first"
    )
    return frame.drop(columns="_abs_effect").to_dict("records")


def _annotate_pair_decisions(
    pairs: pd.DataFrame,
    *,
    context_runs: pd.DataFrame,
    diagnostics: pd.DataFrame,
    track1_profiles: pd.DataFrame,
    track1_supported: pd.DataFrame,
    track2_marker_stats: pd.DataFrame,
    track2_strict_evidence: pd.DataFrame,
    params: ContextRefinementParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairs.empty:
        return pairs, pd.DataFrame(columns=PAIR_EVIDENCE_COLUMNS)
    recovery = _context_recovery(context_runs).set_index("context")
    diagnostic_index = diagnostics.set_index("context") if not diagnostics.empty else None
    indexed_track1_profiles = (
        track1_profiles.drop_duplicates("context", keep="first").set_index("context")
        if not track1_profiles.empty
        else track1_profiles
    )
    track2_key_columns = ["context", "event", "gene_symbol"]
    indexed_track2_stats = (
        track2_marker_stats.drop_duplicates(track2_key_columns, keep="first").set_index(
            track2_key_columns
        )
        if not track2_marker_stats.empty
        else track2_marker_stats
    )
    rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        left = int(pair.context_a)
        right = int(pair.context_b)
        left_recovery = recovery.loc[left] if left in recovery.index else pd.Series(dtype=float)
        right_recovery = recovery.loc[right] if right in recovery.index else pd.Series(dtype=float)
        unstable = any(
            (
                pd.to_numeric(pd.Series([recovery_row.get("precision_q10", np.nan)]), errors="coerce").iloc[0]
                < params.min_context_precision_q10
            )
            or (
                pd.to_numeric(pd.Series([recovery_row.get("jaccard_q10", np.nan)]), errors="coerce").iloc[0]
                < params.min_context_jaccard_q10
            )
            for recovery_row in (left_recovery, right_recovery)
        )

        left_diag = (
            diagnostic_index.loc[left]
            if diagnostic_index is not None and left in diagnostic_index.index
            else pd.Series(dtype=float)
        )
        right_diag = (
            diagnostic_index.loc[right]
            if diagnostic_index is not None and right in diagnostic_index.index
            else pd.Series(dtype=float)
        )
        left_burden = pd.to_numeric(
            pd.Series([left_diag.get("median_cnv_fraction", np.nan)]), errors="coerce"
        ).iloc[0]
        right_burden = pd.to_numeric(
            pd.Series([right_diag.get("median_cnv_fraction", np.nan)]), errors="coerce"
        ).iloc[0]
        burden_delta = (
            float(abs(left_burden - right_burden))
            if np.isfinite(left_burden) and np.isfinite(right_burden)
            else np.nan
        )
        burden_only = bool(
            np.isfinite(burden_delta) and burden_delta >= params.burden_median_delta
        )
        technical = _technical_context(left_diag, params) or _technical_context(
            right_diag, params
        )
        structural_merge_candidate = bool(
            pair.is_local_neighbor
            and np.isfinite(pair.comerge_rate)
            and pair.comerge_rate >= params.min_comerge_rate
            and (unstable or burden_only or technical)
        )
        # Evaluate every pair so later component merges cannot cross a supported
        # boundary transitively through a third community.
        track1_boundary = _track1_pair_evidence(
            left=left,
            right=right,
            profiles=indexed_track1_profiles,
            supported=track1_supported,
            min_abs_delta=params.track1_pair_min_abs_median_delta,
        )
        track2_boundary = _track2_pair_evidence(
            left=left,
            right=right,
            marker_stats=indexed_track2_stats,
            strict_evidence=track2_strict_evidence,
            min_abs_adjusted_delta=params.track2_pair_min_abs_adjusted_delta,
            min_abs_prevalence_delta=(
                params.track2_pair_min_abs_prevalence_delta
            ),
            require_architecture_or_high_level=(
                params.track2_pair_veto_requires_architecture_or_high_level
            ),
        )
        evidence_rows.extend(track1_boundary)
        evidence_rows.extend(track2_boundary)
        track1_veto = any(
            bool(row["supports_boundary_veto"]) for row in track1_boundary
        )
        track2_veto_boundary = [
            row for row in track2_boundary if bool(row["supports_boundary_veto"])
        ]
        track2_veto = bool(track2_veto_boundary)
        evidence_veto = track1_veto or track2_veto
        eligible = bool(structural_merge_candidate and not evidence_veto)
        reasons = [
            name
            for name, value in (
                ("unstable_boundary", unstable),
                ("burden_only", burden_only),
                ("technical_composition", technical),
            )
            if value
        ]
        rows.append(
            {
                **pair._asdict(),
                "context_a_precision_q10": left_recovery.get("precision_q10", np.nan),
                "context_b_precision_q10": right_recovery.get("precision_q10", np.nan),
                "context_a_jaccard_q10": left_recovery.get("jaccard_q10", np.nan),
                "context_b_jaccard_q10": right_recovery.get("jaccard_q10", np.nan),
                "cnv_fraction_median_delta": burden_delta,
                "unstable_boundary": unstable,
                "burden_only_candidate": burden_only,
                "technical_composition_candidate": technical,
                "structural_merge_candidate": structural_merge_candidate,
                "n_track1_pair_events": len(track1_boundary),
                "n_track2_pair_markers": len(track2_boundary),
                "n_track2_pair_veto_markers": len(track2_veto_boundary),
                "n_track2_architecture_linked_pair_markers": sum(
                    bool(row["architecture_linked"]) for row in track2_boundary
                ),
                "track1_evidence_veto": track1_veto,
                "track2_evidence_veto": track2_veto,
                "evidence_veto": evidence_veto,
                "merge_eligible": eligible,
                "merge_reason": "|".join(reasons),
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(evidence_rows, columns=PAIR_EVIDENCE_COLUMNS),
    )


def _final_mapping(
    contexts: list[int], pair_decisions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    union = _UnionFind(contexts)
    decisions = pair_decisions.copy()
    decisions["merge_applied"] = False
    decisions["merge_blocked_by_component_evidence_veto"] = False
    if not decisions.empty:
        evidence_veto = (
            decisions["evidence_veto"].fillna(False).astype(bool)
            if "evidence_veto" in decisions.columns
            else pd.Series(False, index=decisions.index)
        )
        order = decisions.sort_values(
            ["merge_eligible", "comerge_rate", "centroid_distance"],
            ascending=[False, False, True],
        ).index
        for index in order:
            row = decisions.loc[index]
            if not bool(row["merge_eligible"]):
                continue
            left = int(row["context_a"])
            right = int(row["context_b"])
            left_root = union.find(left)
            right_root = union.find(right)
            if left_root == right_root:
                continue
            left_members = {
                context for context in contexts if union.find(context) == left_root
            }
            right_members = {
                context for context in contexts if union.find(context) == right_root
            }
            crosses_supported_boundary = evidence_veto & (
                (
                    decisions["context_a"].astype(int).isin(left_members)
                    & decisions["context_b"].astype(int).isin(right_members)
                )
                | (
                    decisions["context_a"].astype(int).isin(right_members)
                    & decisions["context_b"].astype(int).isin(left_members)
                )
            )
            if crosses_supported_boundary.any():
                decisions.loc[
                    index, "merge_blocked_by_component_evidence_veto"
                ] = True
                continue
            union.union(left, right)
            decisions.loc[index, "merge_applied"] = True

    components: dict[int, list[int]] = {}
    for context in contexts:
        components.setdefault(union.find(context), []).append(int(context))
    ordered_components = sorted(components.values(), key=lambda values: min(values))
    final_by_initial = {
        initial: final
        for final, component in enumerate(ordered_components)
        for initial in component
    }
    mapping = pd.DataFrame(
        [
            {
                "initial_context": int(initial),
                "final_context": int(final_by_initial[initial]),
            }
            for initial in sorted(final_by_initial)
        ]
    )
    return mapping, decisions


def _apply_final_mapping(
    partition: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Map inlier communities while retaining outliers as lineage-only rows."""

    inliers = partition[partition["provisional_context"].ge(0)].copy()
    outliers = partition[partition["provisional_context"].lt(0)].copy()
    final_inliers = inliers.merge(
        mapping[["initial_context", "final_context"]],
        left_on="provisional_context",
        right_on="initial_context",
        how="left",
        validate="many_to_one",
    ).drop(columns="initial_context")
    if final_inliers["final_context"].isna().any():
        raise ValueError("Some inlier communities lack a final-context mapping")
    final_inliers["final_context"] = final_inliers["final_context"].astype(int)
    outliers["final_context"] = -1
    final_partition = pd.concat(
        [final_inliers, outliers], ignore_index=True
    ).sort_values("analysis_id")
    final_sizes = final_inliers.groupby("final_context").size()
    return final_partition, final_sizes


def _merged_stability(
    overlap_cells: pd.DataFrame,
    mapping: pd.DataFrame,
    original_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = overlap_cells.merge(
        mapping,
        left_on="full_cluster",
        right_on="initial_context",
        how="left",
        validate="many_to_one",
    )
    if mapped["final_context"].isna().any():
        raise ValueError("Subsampling overlap cells contain unmapped initial contexts")

    run_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    for repeat, repeat_cells in mapped.groupby("repeat", sort=True):
        collapsed = (
            repeat_cells.groupby(
                ["final_context", "subsample_cluster"], as_index=False
            )["n_overlap"]
            .sum()
        )
        predicted_map = (
            collapsed.sort_values(
                ["subsample_cluster", "n_overlap", "final_context"],
                ascending=[True, False, True],
            )
            .drop_duplicates("subsample_cluster", keep="first")
            .set_index("subsample_cluster")["final_context"]
        )
        collapsed["predicted_final_context"] = collapsed["subsample_cluster"].map(
            predicted_map
        )
        final_contingency = (
            collapsed.groupby(
                ["final_context", "predicted_final_context"], as_index=False
            )["n_overlap"]
            .sum()
        )
        truth = np.concatenate(
            [
                np.repeat(int(row.final_context), int(row.n_overlap))
                for row in final_contingency.itertuples(index=False)
            ]
        )
        observed = np.concatenate(
            [
                np.repeat(int(row.predicted_final_context), int(row.n_overlap))
                for row in final_contingency.itertuples(index=False)
            ]
        )
        table = pd.crosstab(
            pd.Series(truth, name="truth"), pd.Series(observed, name="observed")
        )
        rows, columns = linear_sum_assignment(-table.to_numpy(int))
        matched = int(table.to_numpy(int)[rows, columns].sum())
        run_rows.append(
            {
                "repeat": int(repeat),
                "n_evaluated": int(len(truth)),
                "n_final_contexts": int(len(np.unique(truth))),
                "n_predicted_contexts": int(len(np.unique(observed))),
                "cluster_count_ratio": float(
                    len(np.unique(observed)) / len(np.unique(truth))
                ),
                "ari": float(adjusted_rand_score(truth, observed)),
                "nmi": float(normalized_mutual_info_score(truth, observed)),
                "matched_fraction": float(matched / len(truth)),
            }
        )
        for context in sorted(np.unique(truth)):
            truth_mask = truth == int(context)
            observed_mask = observed == int(context)
            overlap = int(np.sum(truth_mask & observed_mask))
            n_truth = int(truth_mask.sum())
            n_observed = int(observed_mask.sum())
            denominator = n_truth + n_observed - overlap
            context_rows.append(
                {
                    "repeat": int(repeat),
                    "final_context": int(context),
                    "n_true": n_truth,
                    "n_predicted": n_observed,
                    "n_overlap": overlap,
                    "recall": overlap / n_truth if n_truth else np.nan,
                    "precision": overlap / n_observed if n_observed else np.nan,
                    "jaccard": overlap / denominator if denominator else np.nan,
                    "detected": bool(n_observed > 0),
                }
            )
    run_frame = pd.DataFrame(run_rows)
    unchanged_columns = [
        column
        for column in ("repeat", "eval_coverage", "success")
        if column in original_runs.columns
    ]
    if "repeat" in unchanged_columns:
        run_frame = run_frame.merge(
            original_runs[unchanged_columns].drop_duplicates("repeat", keep="first"),
            on="repeat",
            how="left",
            validate="one_to_one",
        )
    context_frame = pd.DataFrame(context_rows)
    summary_rows: list[dict[str, object]] = []
    for context, context_runs in context_frame.groupby("final_context", sort=True):
        detected = context_runs[context_runs["detected"]].copy()

        def quantile(frame: pd.DataFrame, column: str, q: float) -> float:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            return float(values.quantile(q)) if not values.empty else np.nan

        recall = pd.to_numeric(context_runs["recall"], errors="coerce").dropna()
        n_repeats = int(context_runs["repeat"].nunique())
        n_detected = int(detected["repeat"].nunique())
        summary_rows.append(
            {
                "final_context": int(context),
                "repeats": n_repeats,
                "n_detected_repeats": n_detected,
                "detection_rate": (
                    float(n_detected / n_repeats) if n_repeats else np.nan
                ),
                "recall_median": float(recall.median()) if not recall.empty else np.nan,
                # Retain the original unconditional metrics for continuity with
                # previous refinement audits.
                "recall_q10": quantile(context_runs, "recall", 0.10),
                "precision_q10": quantile(context_runs, "precision", 0.10),
                "jaccard_q10": quantile(context_runs, "jaccard", 0.10),
                # Conditional metrics separate intermittent disappearance from
                # an imprecise boundary when the context is recovered.
                "conditional_recall_q10": quantile(detected, "recall", 0.10),
                "conditional_precision_q10": quantile(
                    detected, "precision", 0.10
                ),
                "conditional_jaccard_q10": quantile(detected, "jaccard", 0.10),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return run_frame, context_frame, summary


def refine_contexts(
    *,
    selected_parameters: pd.DataFrame,
    subsampling_runs: pd.DataFrame,
    subsampling_contexts: pd.DataFrame,
    overlap_cells: pd.DataFrame,
    provisional_partitions_dir: Path,
    context_diagnostics: pd.DataFrame,
    track1_context_profiles: pd.DataFrame,
    track1_supported_events: pd.DataFrame,
    track2_marker_stats: pd.DataFrame,
    track2_strict_evidence: pd.DataFrame,
    ir_dir: Path,
    out_dir: Path,
    refinement: ContextRefinementParams,
    subsampling: SubsamplingSelectionParams,
    roots: list[str] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    partitions_out = out_dir / "final_partitions"
    partitions_out.mkdir(parents=True, exist_ok=True)
    selected = selected_parameters.copy()
    if "median_cnv_fraction" not in context_diagnostics.columns:
        raise KeyError(
            "Context diagnostics lack median_cnv_fraction; rerun partition "
            "assessment with --shape-proxies"
        )
    burden_coverage = pd.to_numeric(
        context_diagnostics["median_cnv_fraction"], errors="coerce"
    ).notna().mean()
    if burden_coverage < 0.95:
        raise ValueError(
            "Context-level CNV burden coverage is incomplete "
            f"({burden_coverage:.1%}); rerun assessment with shape proxies"
        )
    if roots:
        selected = selected[selected["root"].astype(str).isin(set(roots))].copy()

    mapping_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    boundary_evidence_parts: list[pd.DataFrame] = []
    run_parts: list[pd.DataFrame] = []
    context_run_parts: list[pd.DataFrame] = []
    stability_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []

    for _, parameter in selected.sort_values("root").iterrows():
        root = str(parameter["root"])
        candidate_id = str(parameter["candidate_id"])
        partition_path = provisional_partitions_dir / f"{root}_provisional_partition.parquet"
        if not partition_path.exists():
            raise FileNotFoundError(partition_path)
        partition = pd.read_parquet(partition_path)[
            ["analysis_id", "provisional_context"]
        ].copy()
        partition["analysis_id"] = partition["analysis_id"].astype(str)
        partition["provisional_context"] = partition["provisional_context"].astype(int)
        inlier_partition = partition[partition["provisional_context"].ge(0)].copy()
        outlier_partition = partition[partition["provisional_context"].lt(0)].copy()
        contexts = sorted(inlier_partition["provisional_context"].unique())
        if len(contexts) < int(refinement.min_final_contexts):
            raise ValueError(
                f"[{root}] provisional partition contains fewer than "
                f"{refinement.min_final_contexts} inlier contexts"
            )

        root_runs = subsampling_contexts[
            subsampling_contexts["candidate_id"].astype(str).eq(candidate_id)
        ].copy()
        root_overlap = overlap_cells[
            overlap_cells["candidate_id"].astype(str).eq(candidate_id)
        ].copy()
        root_run_summary = subsampling_runs[
            subsampling_runs["candidate_id"].astype(str).eq(candidate_id)
        ].copy()
        if root_runs.empty or root_overlap.empty or root_run_summary.empty:
            raise ValueError(f"[{root}] selected candidate lacks subsampling overlap data")

        coords_path = Path(ir_dir) / "perroot" / root / "coords.parquet"
        coords = pd.read_parquet(coords_path)
        centroid_pairs = _pairwise_centroid_table(
            coords,
            inlier_partition,
            n_pcs=int(parameter["n_pcs"]),
            distance_quantile=refinement.centroid_distance_quantile,
        )
        pair_table = centroid_pairs.merge(
            _comerge_table(root_runs),
            on=["context_a", "context_b"],
            how="left",
            validate="one_to_one",
        )
        root_diagnostics = context_diagnostics[
            context_diagnostics["root"].astype(str).eq(root)
        ].copy()
        root_t1_profiles = track1_context_profiles[
            track1_context_profiles["root"].astype(str).eq(root)
        ].copy()
        root_t1_supported = track1_supported_events[
            track1_supported_events["root"].astype(str).eq(root)
        ].copy()
        root_t2_stats = track2_marker_stats[
            track2_marker_stats["root"].astype(str).eq(root)
        ].copy()
        root_t2_strict = track2_strict_evidence[
            track2_strict_evidence["root"].astype(str).eq(root)
        ].copy()
        pair_table, pair_boundary_evidence = _annotate_pair_decisions(
            pair_table,
            context_runs=root_runs,
            diagnostics=root_diagnostics,
            track1_profiles=root_t1_profiles,
            track1_supported=root_t1_supported,
            track2_marker_stats=root_t2_stats,
            track2_strict_evidence=root_t2_strict,
            params=refinement,
        )
        mapping, pair_table = _final_mapping(contexts, pair_table)
        mapping.insert(0, "root", root)
        pair_table.insert(0, "root", root)
        pair_boundary_evidence.insert(0, "root", root)
        mapping_parts.append(mapping)
        decision_parts.append(pair_table)
        boundary_evidence_parts.append(pair_boundary_evidence)

        final_partition, final_sizes = _apply_final_mapping(partition, mapping)
        final_partition.to_parquet(
            partitions_out / f"{root}_final_partition.parquet", index=False
        )
        merged_runs, merged_context_runs, merged_stability = _merged_stability(
            root_overlap, mapping, root_run_summary
        )
        merged_runs.insert(0, "root", root)
        merged_context_runs.insert(0, "root", root)
        merged_stability.insert(0, "root", root)
        merged_stability = merged_stability.join(
            final_sizes.rename("n_samples"), on="final_context"
        )
        run_parts.append(merged_runs)
        context_run_parts.append(merged_context_runs)
        stability_parts.append(merged_stability)

        failed_stability = merged_stability[
            merged_stability["precision_q10"].lt(
                subsampling.min_worst_context_precision_q10
            )
            | merged_stability["jaccard_q10"].lt(
                subsampling.min_worst_context_jaccard_q10
            )
        ]
        undersized = final_sizes[final_sizes.lt(refinement.min_final_context_size)]
        too_few_contexts = int(len(final_sizes)) < int(refinement.min_final_contexts)
        run_gate_failures = {
            "insufficient_successful_repeats": int(merged_runs["repeat"].nunique())
            < int(subsampling.repeats),
            "merged_ari": float(merged_runs["ari"].mean())
            < subsampling.min_subsample_ari
            or float(merged_runs["ari"].quantile(0.10))
            < subsampling.min_subsample_ari,
            "merged_matched_fraction": float(merged_runs["matched_fraction"].mean())
            < subsampling.min_matched_fraction,
            "merged_cluster_count_ratio": not (
                subsampling.min_cluster_count_ratio
                <= float(merged_runs["cluster_count_ratio"].median())
                <= subsampling.max_cluster_count_ratio
            ),
            "eval_coverage": (
                "eval_coverage" in merged_runs.columns
                and float(merged_runs["eval_coverage"].mean())
                < subsampling.min_eval_coverage
            ),
        }
        requires_review = bool(
            not failed_stability.empty
            or not undersized.empty
            or too_few_contexts
            or any(run_gate_failures.values())
        )
        if requires_review:
            review_rows.append(
                {
                    "root": root,
                    "reason": "|".join(
                        reason
                        for reason, condition in (
                            ("merged_stability_gate", not failed_stability.empty),
                            ("undersized_final_context", not undersized.empty),
                            ("too_few_final_contexts", too_few_contexts),
                            *tuple(run_gate_failures.items()),
                        )
                        if condition
                    ),
                }
            )
        structural_pairs = pair_table["structural_merge_candidate"].astype(bool)
        summary_rows.append(
            {
                "root": root,
                "selected_candidate_id": candidate_id,
                "n_initial_contexts": len(contexts),
                "n_final_contexts": int(len(final_sizes)),
                "n_outlier_samples": int(len(outlier_partition)),
                "n_merges_applied": int(pair_table["merge_applied"].sum()),
                "n_structural_merge_candidates": int(
                    pair_table["structural_merge_candidate"].sum()
                ),
                "n_pairs_with_track1_veto": int(
                    (structural_pairs & pair_table["track1_evidence_veto"]).sum()
                ),
                "n_pairs_with_track2_veto": int(
                    (structural_pairs & pair_table["track2_evidence_veto"]).sum()
                ),
                "n_pairs_with_any_evidence_veto": int(
                    (structural_pairs & pair_table["evidence_veto"]).sum()
                ),
                "n_merges_blocked_by_component_evidence_veto": int(
                    pair_table["merge_blocked_by_component_evidence_veto"].sum()
                ),
                "n_pair_boundary_evidence_rows": int(len(pair_boundary_evidence)),
                "merged_ari_mean": float(merged_runs["ari"].mean()),
                "merged_matched_fraction_mean": float(
                    merged_runs["matched_fraction"].mean()
                ),
                "requires_review": bool(
                    requires_review
                ),
            }
        )

    outputs = {
        "mapping": out_dir / "initial_to_final_context.tsv",
        "pair_decisions": out_dir / "pair_merge_decisions.tsv",
        "pair_boundary_evidence": out_dir / "pair_boundary_evidence.tsv",
        "merged_runs": out_dir / "merged_subsampling_runs.tsv",
        "merged_context_runs": out_dir / "merged_context_runs.tsv",
        "merged_context_stability": out_dir / "merged_context_stability.tsv",
        "summary": out_dir / "refinement_summary.tsv",
        "reviews": out_dir / "roots_requiring_refinement_review.tsv",
    }
    pd.concat(mapping_parts, ignore_index=True).to_csv(
        outputs["mapping"], sep="\t", index=False
    )
    pd.concat(decision_parts, ignore_index=True).to_csv(
        outputs["pair_decisions"], sep="\t", index=False
    )
    pd.concat(boundary_evidence_parts, ignore_index=True).to_csv(
        outputs["pair_boundary_evidence"], sep="\t", index=False
    )
    pd.concat(run_parts, ignore_index=True).to_csv(
        outputs["merged_runs"], sep="\t", index=False
    )
    pd.concat(context_run_parts, ignore_index=True).to_csv(
        outputs["merged_context_runs"], sep="\t", index=False
    )
    pd.concat(stability_parts, ignore_index=True).to_csv(
        outputs["merged_context_stability"], sep="\t", index=False
    )
    pd.DataFrame(summary_rows).to_csv(outputs["summary"], sep="\t", index=False)
    pd.DataFrame(review_rows, columns=["root", "reason"]).to_csv(
        outputs["reviews"], sep="\t", index=False
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "refinement": asdict(refinement),
                "subsampling_gates": asdict(subsampling),
                "boundary_veto_definition": (
                    "context-vs-rest evidence plus a direction-consistent "
                    "pair-specific effect contrast"
                ),
                "partitions": str(partitions_out),
                "outputs": {key: str(path) for key, path in outputs.items()},
                "counts": {
                    "roots": len(summary_rows),
                    "initial_contexts": int(
                        sum(row["n_initial_contexts"] for row in summary_rows)
                    ),
                    "final_contexts": int(
                        sum(row["n_final_contexts"] for row in summary_rows)
                    ),
                    "review_roots": len(review_rows),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "partitions": partitions_out, "manifest": manifest_path}
