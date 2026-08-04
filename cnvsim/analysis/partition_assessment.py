from __future__ import annotations

"""Pre-materialization diagnostics for selected CNV partitions."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_samples

from cnvsim.analysis.config import AnalysisConfig
from cnvsim.analysis.ir import dimension_columns
from cnvsim.analysis.leiden import run_leiden_cpm
from cnvsim.analysis.subsampling_selection import (
    build_subsample_graph,
    mark_small_clusters,
)


def _stratified_sample_indices(
    labels: np.ndarray,
    *,
    maximum: int,
    random_seed: int,
) -> np.ndarray:
    valid = np.flatnonzero(labels >= 0)
    if len(valid) <= int(maximum):
        return valid

    rng = np.random.default_rng(int(random_seed))
    chosen: list[int] = []
    for cluster in sorted(np.unique(labels[valid])):
        members = np.flatnonzero(labels == cluster)
        take = min(2, len(members))
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())

    chosen_set = set(chosen)
    pool = np.asarray([index for index in valid if index not in chosen_set], dtype=int)
    remaining = max(0, int(maximum) - len(chosen))
    if remaining and len(pool):
        chosen.extend(
            rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist()
        )
    return np.asarray(sorted(set(chosen)), dtype=int)


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected Cramer's V for categorical association."""

    from scipy.stats import chi2_contingency

    x_series = pd.Series(x).fillna("Unknown").astype(str)
    y_series = pd.Series(y).fillna("Unknown").astype(str)
    table = pd.crosstab(x_series, y_series)
    if table.empty or min(table.shape) < 2:
        return 0.0
    chi2, _p_value, _dof, _expected = chi2_contingency(table.to_numpy())
    n = float(table.to_numpy().sum())
    if n <= 1:
        return np.nan
    rows, columns = table.shape
    phi2 = float(chi2 / n)
    phi2_corrected = max(
        0.0,
        phi2 - ((columns - 1) * (rows - 1)) / max(1.0, n - 1.0),
    )
    rows_corrected = rows - ((rows - 1) ** 2) / max(1.0, n - 1.0)
    columns_corrected = columns - ((columns - 1) ** 2) / max(1.0, n - 1.0)
    denominator = min(rows_corrected - 1.0, columns_corrected - 1.0)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def _eta_squared(values: pd.Series, labels: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    labels = np.asarray(labels, dtype=np.int32)
    valid = np.isfinite(numeric) & (labels >= 0)
    if valid.sum() < 2:
        return np.nan
    numeric = numeric[valid]
    labels = labels[valid]
    grand_mean = float(numeric.mean())
    total = float(np.square(numeric - grand_mean).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for cluster in np.unique(labels):
        group = numeric[labels == cluster]
        between += float(len(group) * (group.mean() - grand_mean) ** 2)
    return float(between / total)


def _rank_biserial(context_values: pd.Series, other_values: pd.Series) -> float:
    from scipy.stats import mannwhitneyu

    context = pd.to_numeric(context_values, errors="coerce").dropna().to_numpy(float)
    other = pd.to_numeric(other_values, errors="coerce").dropna().to_numpy(float)
    if not len(context) or not len(other):
        return np.nan
    statistic = float(mannwhitneyu(context, other, alternative="two-sided").statistic)
    return float((2.0 * statistic) / (len(context) * len(other)) - 1.0)


def _align_metadata(meta: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    aligned = meta.copy()
    if "analysis_id" in aligned.columns:
        aligned["analysis_id"] = aligned["analysis_id"].astype(str)
        aligned = aligned.drop_duplicates("analysis_id").set_index("analysis_id")
        return aligned.reindex(coords.index.astype(str))
    return aligned.reindex(coords.index)


def _composition_metrics(
    values: pd.Series,
    members: np.ndarray,
    *,
    prefix: str,
    valid: np.ndarray,
) -> dict:
    clean = values.fillna("Unknown").astype(str)
    root_counts = clean.loc[valid].value_counts(dropna=False)
    context_counts = clean.loc[members].value_counts(dropna=False)
    if root_counts.empty or context_counts.empty:
        return {}

    root_fractions = root_counts / root_counts.sum()
    context_fractions = context_counts / context_counts.sum()
    top = str(context_counts.index[0])
    differences = context_fractions.subtract(root_fractions, fill_value=0.0)
    enriched = str(differences.idxmax())
    context_enriched_fraction = float(context_fractions.get(enriched, 0.0))
    root_enriched_fraction = float(root_fractions.get(enriched, 0.0))
    return {
        f"top_{prefix}": top,
        f"top_{prefix}_fraction": float(context_fractions.loc[top]),
        f"top_{prefix}_root_fraction": float(root_fractions.loc[top]),
        f"top_{prefix}_enrichment_ratio": float(
            context_fractions.loc[top] / root_fractions.loc[top]
        ),
        f"enriched_{prefix}": enriched,
        f"enriched_{prefix}_context_fraction": context_enriched_fraction,
        f"enriched_{prefix}_root_fraction": root_enriched_fraction,
        f"enriched_{prefix}_fraction_difference": float(
            context_enriched_fraction - root_enriched_fraction
        ),
        f"enriched_{prefix}_ratio": float(
            context_enriched_fraction / root_enriched_fraction
        ),
        f"{prefix}_one_vs_rest_cramers_v": _cramers_v(
            members[valid].astype(str), clean.loc[valid].to_numpy(dtype=object)
        ),
    }


def _numeric_context_metrics(
    values: pd.Series,
    members: np.ndarray,
    *,
    prefix: str,
    valid: np.ndarray,
    effect_transform=None,
) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    context = numeric.loc[members].dropna()
    other = numeric.loc[valid & ~members].dropna()
    transformed_context = effect_transform(context) if effect_transform else context
    transformed_other = effect_transform(other) if effect_transform else other
    return {
        f"n_{prefix}_profiles": int(len(context)),
        f"{prefix}_coverage": float(len(context) / max(1, int(members.sum()))),
        f"median_{prefix}": float(context.median()) if len(context) else np.nan,
        f"other_median_{prefix}": float(other.median()) if len(other) else np.nan,
        f"{prefix}_median_difference_vs_other": (
            float(context.median() - other.median())
            if len(context) and len(other)
            else np.nan
        ),
        f"{prefix}_rank_biserial_vs_other": _rank_biserial(
            transformed_context, transformed_other
        ),
    }


def _composition_enrichment_rows(
    *,
    root: str,
    labels: np.ndarray,
    aligned_meta: pd.DataFrame,
) -> list[dict]:
    annotation_columns = {
        "platform": "platform_family_refined",
        "source": "source_group_refined",
        "morphology": "icdo_morphology_id",
    }
    rows: list[dict] = []
    valid = labels >= 0
    for annotation, column in annotation_columns.items():
        if column not in aligned_meta.columns:
            continue
        clean = aligned_meta[column].fillna("Unknown").astype(str)
        root_counts = clean.loc[valid].value_counts(dropna=False)
        root_total = int(root_counts.sum())
        if root_total == 0:
            continue
        for cluster in sorted(int(value) for value in np.unique(labels[valid])):
            members = labels == cluster
            context_counts = clean.loc[members].value_counts(dropna=False)
            context_total = int(context_counts.sum())
            for category, root_n in root_counts.items():
                context_n = int(context_counts.get(category, 0))
                root_fraction = float(root_n / root_total)
                context_fraction = float(context_n / context_total)
                rows.append(
                    {
                        "root": str(root),
                        "context": int(cluster),
                        "annotation": annotation,
                        "category": str(category),
                        "context_n": context_n,
                        "context_fraction": context_fraction,
                        "root_n": int(root_n),
                        "root_fraction": root_fraction,
                        "fraction_difference": float(context_fraction - root_fraction),
                        "enrichment_ratio": float(context_fraction / root_fraction),
                    }
                )
    return rows


def _weighted_graph_coherence(graph, labels: np.ndarray) -> tuple[dict, dict[int, dict]]:
    edges = np.asarray(graph.get_edgelist(), dtype=int)
    weights = np.asarray(graph.es["weight"], dtype=float)
    if edges.size == 0 or weights.size == 0:
        return {}, {}

    valid_edge = (labels[edges[:, 0]] >= 0) & (labels[edges[:, 1]] >= 0)
    edges = edges[valid_edge]
    weights = weights[valid_edge]
    if edges.size == 0 or float(weights.sum()) <= 0:
        return {}, {}
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    within_weight_fraction = float(weights[same].sum() / weights.sum())

    total_weight = np.zeros(len(labels), dtype=float)
    same_weight = np.zeros(len(labels), dtype=float)
    np.add.at(total_weight, edges[:, 0], weights)
    np.add.at(total_weight, edges[:, 1], weights)
    if same.any():
        np.add.at(same_weight, edges[same, 0], weights[same])
        np.add.at(same_weight, edges[same, 1], weights[same])
    vertex_purity = np.divide(
        same_weight,
        total_weight,
        out=np.full(len(labels), np.nan, dtype=float),
        where=total_weight > 0,
    )

    valid_labels = labels[labels >= 0]
    fractions = pd.Series(valid_labels).value_counts(normalize=True)
    graph_volume = float(total_weight.sum())
    volume_fractions = {
        int(cluster): float(total_weight[labels == cluster].sum() / graph_volume)
        for cluster in fractions.index
    }
    expected_same = float(sum(fraction**2 for fraction in volume_fractions.values()))
    denominator = max(1e-12, 1.0 - expected_same)
    normalized_enrichment = float((within_weight_fraction - expected_same) / denominator)

    context_metrics: dict[int, dict] = {}
    for cluster in fractions.index:
        cluster = int(cluster)
        baseline_fraction = volume_fractions[cluster]
        members = labels == cluster
        cluster_edges = members[edges[:, 0]] ^ members[edges[:, 1]]
        cut_weight = float(weights[cluster_edges].sum())
        cluster_volume = float(total_weight[members].sum())
        conductance_denominator = min(cluster_volume, graph_volume - cluster_volume)
        conductance = (
            float(cut_weight / conductance_denominator)
            if conductance_denominator > 0
            else np.nan
        )
        purity = vertex_purity[members]
        purity_enrichment = (purity - baseline_fraction) / max(
            1e-12, 1.0 - baseline_fraction
        )
        context_metrics[cluster] = {
            "knn_purity_median": float(np.nanmedian(purity)),
            "knn_purity_q10": float(np.nanquantile(purity, 0.10)),
            "knn_purity_enrichment_median": float(np.nanmedian(purity_enrichment)),
            "graph_conductance": conductance,
        }

    return {
        "weighted_within_edge_fraction": within_weight_fraction,
        "expected_within_edge_fraction": expected_same,
        "weighted_edge_enrichment": normalized_enrichment,
    }, context_metrics


def compute_partition_diagnostics(
    *,
    root: str,
    coords: pd.DataFrame,
    meta: pd.DataFrame,
    graph,
    labels: np.ndarray,
    metric: str,
    n_pcs: int,
    silhouette_sample_size: int,
    random_seed: int,
) -> tuple[dict, list[dict]]:
    """Compare a provisional partition with its one-context centroid baseline."""

    labels = np.asarray(labels, dtype=np.int32)
    dimensions = dimension_columns(coords)[: int(n_pcs)]
    if not dimensions:
        raise ValueError(f"[{root}] no D-prefixed analysis dimensions were found")
    values = coords[dimensions].to_numpy(dtype=np.float64)
    valid = labels >= 0
    context_ids = sorted(int(cluster) for cluster in np.unique(labels[valid]))
    if len(context_ids) < 1:
        raise ValueError(f"[{root}] provisional partition contains no assessable contexts")

    centroids = np.vstack([values[labels == cluster].mean(axis=0) for cluster in context_ids])
    distances = pairwise_distances(values, centroids, metric=str(metric))
    context_to_column = {cluster: index for index, cluster in enumerate(context_ids)}
    own_columns = np.asarray(
        [context_to_column.get(int(label), 0) for label in labels], dtype=int
    )
    own_distance = distances[np.arange(len(values)), own_columns]
    if len(context_ids) > 1:
        masked = distances.copy()
        masked[np.arange(len(values)), own_columns] = np.inf
        nearest_other = masked.min(axis=1)
        assignment_margin = nearest_other - own_distance
    else:
        assignment_margin = np.full(len(values), np.nan)

    root_centroid = values[valid].mean(axis=0, keepdims=True)
    one_context_distance = pairwise_distances(
        values[valid], root_centroid, metric=str(metric)
    ).ravel()
    if str(metric) == "euclidean":
        one_context_dispersion = float(np.square(one_context_distance).sum())
        partition_dispersion = float(np.square(own_distance[valid]).sum())
    else:
        one_context_dispersion = float(one_context_distance.sum())
        partition_dispersion = float(own_distance[valid].sum())
    dispersion_reduction_vs_one_context = (
        float(1.0 - partition_dispersion / one_context_dispersion)
        if one_context_dispersion > 0
        else np.nan
    )

    silhouette = np.full(len(values), np.nan, dtype=float)
    sample_indices = _stratified_sample_indices(
        labels,
        maximum=int(silhouette_sample_size),
        random_seed=int(random_seed),
    )
    if len(context_ids) > 1 and len(sample_indices) > len(context_ids):
        silhouette[sample_indices] = silhouette_samples(
            values[sample_indices],
            labels[sample_indices],
            metric=str(metric),
        )

    graph_root, graph_context = _weighted_graph_coherence(graph, labels)
    valid_count = int(valid.sum())
    counts = pd.Series(labels[valid]).value_counts().sort_index()
    fractions = counts / max(1, valid_count)

    aligned_meta = _align_metadata(meta, coords)
    platform_values = (
        aligned_meta["platform_family_refined"]
        if "platform_family_refined" in aligned_meta.columns
        else pd.Series("Unknown", index=aligned_meta.index)
    )
    source_values = (
        aligned_meta["source_group_refined"]
        if "source_group_refined" in aligned_meta.columns
        else pd.Series("Unknown", index=aligned_meta.index)
    )
    morphology_values = (
        aligned_meta["icdo_morphology_id"]
        if "icdo_morphology_id" in aligned_meta.columns
        else pd.Series("Unknown", index=aligned_meta.index)
    )
    cnv_fraction_values = (
        aligned_meta["cnv_fraction"]
        if "cnv_fraction" in aligned_meta.columns
        else pd.Series(np.nan, index=aligned_meta.index, dtype=float)
    )
    variant_count_values = (
        aligned_meta["variant_count"]
        if "variant_count" in aligned_meta.columns
        else pd.Series(np.nan, index=aligned_meta.index, dtype=float)
    )

    context_rows: list[dict] = []
    for cluster in context_ids:
        members = labels == cluster
        sampled_silhouette = silhouette[members & np.isfinite(silhouette)]
        margins = assignment_margin[members & np.isfinite(assignment_margin)]
        row = {
            "root": str(root),
            "context": int(cluster),
            "n_samples": int(counts.loc[cluster]),
            "sample_fraction": float(fractions.loc[cluster]),
            "centroid_distance_median": float(np.median(own_distance[members])),
            "centroid_distance_q90": float(np.quantile(own_distance[members], 0.90)),
            "assignment_margin_median": (
                float(np.median(margins)) if len(margins) else np.nan
            ),
            "assignment_margin_q10": (
                float(np.quantile(margins, 0.10)) if len(margins) else np.nan
            ),
            "assignment_margin_positive_fraction": (
                float(np.mean(margins > 0)) if len(margins) else np.nan
            ),
            "silhouette_sample_n": int(len(sampled_silhouette)),
            "silhouette_median": (
                float(np.median(sampled_silhouette)) if len(sampled_silhouette) else np.nan
            ),
            "silhouette_q10": (
                float(np.quantile(sampled_silhouette, 0.10))
                if len(sampled_silhouette)
                else np.nan
            ),
            "silhouette_positive_fraction": (
                float(np.mean(sampled_silhouette > 0))
                if len(sampled_silhouette)
                else np.nan
            ),
            **_composition_metrics(
                platform_values, members, prefix="platform", valid=valid
            ),
            **_composition_metrics(source_values, members, prefix="source", valid=valid),
            **_composition_metrics(
                morphology_values, members, prefix="morphology", valid=valid
            ),
            **graph_context.get(cluster, {}),
        }
        row.update(
            _numeric_context_metrics(
                cnv_fraction_values,
                members,
                prefix="cnv_fraction",
                valid=valid,
            )
        )
        row.update(
            _numeric_context_metrics(
                variant_count_values,
                members,
                prefix="variant_count",
                valid=valid,
                effect_transform=np.log1p,
            )
        )
        context_rows.append(row)

    sampled_root_silhouette = silhouette[np.isfinite(silhouette)]
    root_row = {
        "root": str(root),
        "metric": str(metric),
        "n_pcs": int(n_pcs),
        "n_samples": int(len(labels)),
        "n_assessed_samples": valid_count,
        "n_contexts": int(len(context_ids)),
        "outlier_fraction": float(np.mean(labels < 0)),
        "largest_context_fraction": float(fractions.max()),
        "smallest_context_size": int(counts.min()),
        "one_context_dispersion": one_context_dispersion,
        "partition_dispersion": partition_dispersion,
        "dispersion_reduction_vs_one_context": dispersion_reduction_vs_one_context,
        "silhouette_sample_n": int(len(sampled_root_silhouette)),
        "silhouette_mean": (
            float(np.mean(sampled_root_silhouette))
            if len(sampled_root_silhouette)
            else np.nan
        ),
        "silhouette_q10": (
            float(np.quantile(sampled_root_silhouette, 0.10))
            if len(sampled_root_silhouette)
            else np.nan
        ),
        "silhouette_positive_fraction": (
            float(np.mean(sampled_root_silhouette > 0))
            if len(sampled_root_silhouette)
            else np.nan
        ),
        "platform_context_cramers_v": _cramers_v(
            labels[valid], platform_values.loc[valid].to_numpy(dtype=object)
        ),
        "source_context_cramers_v": _cramers_v(
            labels[valid], source_values.loc[valid].to_numpy(dtype=object)
        ),
        "morphology_context_cramers_v": _cramers_v(
            labels[valid], morphology_values.loc[valid].to_numpy(dtype=object)
        ),
        "burden_profile_coverage": float(
            pd.to_numeric(cnv_fraction_values, errors="coerce").loc[valid].notna().mean()
        ),
        "cnv_fraction_context_eta2": _eta_squared(cnv_fraction_values, labels),
        "log1p_variant_count_context_eta2": _eta_squared(
            np.log1p(pd.to_numeric(variant_count_values, errors="coerce")),
            labels,
        ),
        **graph_root,
    }
    return root_row, context_rows


def _load_shape_proxies(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis shape proxies: {path}")
    proxies = pd.read_parquet(path)
    if "analysis_id" not in proxies.columns and proxies.index.name == "analysis_id":
        proxies = proxies.reset_index()
    required = {"analysis_id", "cnv_fraction", "variant_count"}
    missing = sorted(required - set(proxies.columns))
    if missing:
        raise KeyError(f"Shape proxy table is missing columns: {missing}")
    proxies = proxies[["analysis_id", "cnv_fraction", "variant_count"]].copy()
    proxies["analysis_id"] = proxies["analysis_id"].astype(str)
    if proxies["analysis_id"].duplicated().any():
        duplicated = proxies.loc[
            proxies["analysis_id"].duplicated(keep=False), "analysis_id"
        ].head(10)
        raise ValueError(
            "Shape proxy table contains duplicate analysis IDs. Examples: "
            + ", ".join(duplicated.astype(str))
        )
    for column in ("cnv_fraction", "variant_count"):
        proxies[column] = pd.to_numeric(proxies[column], errors="coerce")
    invalid_fraction = proxies["cnv_fraction"].dropna().lt(0) | proxies[
        "cnv_fraction"
    ].dropna().gt(1)
    if invalid_fraction.any():
        raise ValueError("cnv_fraction must be between 0 and 1")
    if proxies["variant_count"].dropna().lt(0).any():
        raise ValueError("variant_count must be non-negative")
    return proxies.set_index("analysis_id")


def assess_selected_partitions(
    *,
    ir_dir: Path,
    selected_parameters_path: Path,
    out_dir: Path,
    config: AnalysisConfig,
    roots: list[str] | None = None,
    silhouette_sample_size: int = 3000,
    shape_proxies_path: Path | None = None,
) -> dict[str, Path]:
    """Build provisional selected partitions and write diagnostic-only outputs."""

    ir_dir = Path(ir_dir)
    out_dir = Path(out_dir)
    parameters = pd.read_csv(selected_parameters_path, sep="\t", low_memory=False)
    parameters["root"] = parameters["root"].astype(str)
    if roots:
        root_set = {str(root) for root in roots}
        parameters = parameters[parameters["root"].isin(root_set)].copy()
    if parameters.empty:
        raise ValueError("No selected parameter rows remain for assessment")
    shape_proxies = _load_shape_proxies(shape_proxies_path)

    root_rows: list[dict] = []
    context_rows: list[dict] = []
    enrichment_rows: list[dict] = []
    review_rows: list[dict] = []
    provisional_dir = out_dir / "assessment_partitions"
    provisional_dir.mkdir(parents=True, exist_ok=True)

    for _, parameter in parameters.sort_values("root").iterrows():
        root = str(parameter["root"])
        status = str(parameter.get("selection_status", ""))
        if status != "subsampling_selected":
            review_rows.append(
                {
                    "root": root,
                    "reason": "parameter_selection_review",
                    "detail": status,
                }
            )

        root_dir = ir_dir / "perroot" / root
        coords = pd.read_parquet(root_dir / "coords.parquet")
        coords.index = coords.index.astype(str)
        meta = pd.read_parquet(root_dir / "meta.parquet")
        if shape_proxies is not None:
            if "analysis_id" not in meta.columns:
                raise KeyError(f"[{root}] analysis IR metadata is missing analysis_id")
            meta = meta.copy()
            meta["analysis_id"] = meta["analysis_id"].astype(str)
            meta = meta.drop(columns=["cnv_fraction", "variant_count"], errors="ignore")
            meta = meta.join(shape_proxies, on="analysis_id", validate="many_to_one")
        metric = str(parameter["metric"]).lower()
        n_pcs = int(parameter["n_pcs"])
        k = int(parameter["K"])
        resolution = float(parameter["stage2_selected_resolution"])

        graph = build_subsample_graph(
            coords,
            sample_indices=np.arange(len(coords), dtype=int),
            metric=metric,
            n_pcs=n_pcs,
            k=k,
            sigma_quantile=float(config.knn_selection.sigma_quantile),
            make_undirected=bool(config.knn_selection.make_undirected),
            combine_mode=str(config.knn_selection.combine_mode),
        )
        labels, _ = run_leiden_cpm(
            graph,
            resolution=resolution,
            seed=int(config.subsampling.partition_seed),
        )
        labels = mark_small_clusters(labels, int(config.leiden.min_cluster_size))
        provisional = pd.DataFrame(
            {
                "analysis_id": coords.index.astype(str),
                "provisional_context": labels,
            }
        )
        provisional.to_parquet(
            provisional_dir / f"{root}_provisional_partition.parquet",
            index=False,
        )

        root_row, root_context_rows = compute_partition_diagnostics(
            root=root,
            coords=coords,
            meta=meta,
            graph=graph,
            labels=labels,
            metric=metric,
            n_pcs=n_pcs,
            silhouette_sample_size=int(silhouette_sample_size),
            random_seed=int(config.subsampling.random_seed),
        )
        root_row.update(
            {
                "K": k,
                "resolution": resolution,
                "parameter_selection_status": status,
                "selected_n_contexts": int(parameter["n_clusters_full"]),
                "partition_count_matches_selection": (
                    int(root_row["n_contexts"]) == int(parameter["n_clusters_full"])
                ),
            }
        )
        for row in root_context_rows:
            row["parameter_selection_status"] = status
        root_rows.append(root_row)
        context_rows.extend(root_context_rows)
        enrichment_rows.extend(
            _composition_enrichment_rows(
                root=root,
                labels=labels,
                aligned_meta=_align_metadata(meta, coords),
            )
        )

        reasons: list[str] = []
        if not root_row["partition_count_matches_selection"]:
            reasons.append("partition_count_mismatch")
        if float(root_row["largest_context_fraction"]) >= 0.80:
            reasons.append("dominant_context_fraction_ge_0.80")
        if np.isfinite(root_row["silhouette_mean"]) and float(root_row["silhouette_mean"]) <= 0:
            reasons.append("nonpositive_root_silhouette")
        diffuse = [
            row
            for row in root_context_rows
            if (
                np.isfinite(row["assignment_margin_median"])
                and float(row["assignment_margin_median"]) <= 0
            )
            or (
                np.isfinite(row.get("knn_purity_enrichment_median", np.nan))
                and float(row["knn_purity_enrichment_median"]) <= 0
            )
        ]
        if diffuse:
            reasons.append("noncoherent_context_detected")
        if reasons:
            review_rows.append(
                {
                    "root": root,
                    "reason": ";".join(reasons),
                    "detail": "diagnostic_only",
                }
            )

    root_path = out_dir / "selected_partition_diagnostics.tsv"
    context_path = out_dir / "selected_context_diagnostics.tsv"
    enrichment_path = out_dir / "selected_context_composition_enrichment.tsv"
    review_path = out_dir / "roots_requiring_structure_review.tsv"
    pd.DataFrame(root_rows).to_csv(root_path, sep="\t", index=False)
    pd.DataFrame(context_rows).to_csv(context_path, sep="\t", index=False)
    pd.DataFrame(enrichment_rows).to_csv(enrichment_path, sep="\t", index=False)
    pd.DataFrame(
        review_rows,
        columns=["root", "reason", "detail"],
    ).to_csv(review_path, sep="\t", index=False)

    manifest_path = out_dir / "partition_assessment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "diagnostic_only": True,
                "ir_dir": str(ir_dir.resolve()),
                "selected_parameters": str(Path(selected_parameters_path).resolve()),
                "shape_proxies": (
                    str(Path(shape_proxies_path).resolve())
                    if shape_proxies_path is not None
                    else None
                ),
                "silhouette_sample_size": int(silhouette_sample_size),
                "screening_flags": {
                    "dominant_context_fraction": 0.80,
                    "root_silhouette": "<= 0",
                    "context_assignment_margin_median": "<= 0",
                    "context_knn_purity_enrichment_median": "<= 0",
                },
                "counts": {
                    "assessed_roots": len(root_rows),
                    "assessment_contexts": len(context_rows),
                    "composition_enrichment_rows": len(enrichment_rows),
                    "review_rows": len(review_rows),
                },
                "outputs": {
                    "root_diagnostics": str(root_path),
                    "context_diagnostics": str(context_path),
                    "composition_enrichment": str(enrichment_path),
                    "structure_reviews": str(review_path),
                    "provisional_partitions": str(provisional_dir),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "root_diagnostics": root_path,
        "context_diagnostics": context_path,
        "composition_enrichment": enrichment_path,
        "reviews": review_path,
        "manifest": manifest_path,
        "provisional_partitions": provisional_dir,
    }
