from __future__ import annotations

"""Subsampling-based model selection across stable Leiden plateaus."""

import json
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from cnvsim.analysis.config import LeidenParams, SubsamplingSelectionParams
from cnvsim.analysis.ir import dimension_columns
from cnvsim.analysis.knn_parameter import (
    build_igraph_from_knn,
    knn_indices_and_distances,
)
from cnvsim.analysis.leiden import run_leiden_cpm


def mark_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Relabel clusters smaller than the configured minimum as outliers."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    counts = pd.Series(labels).value_counts()
    small = counts[counts < int(min_cluster_size)].index.to_numpy(dtype=np.int32)
    if small.size:
        labels[np.isin(labels, small)] = -1
    return labels


def scaled_subsample_min_cluster_size(
    full_min_cluster_size: int,
    sample_fraction: float,
) -> int:
    """Scale the reference-context size threshold to a subsampled cohort."""

    return max(1, int(np.ceil(float(full_min_cluster_size) * float(sample_fraction))))


def _candidate_feasibility_mask(rows: pd.DataFrame, leiden: LeidenParams) -> pd.Series:
    numeric = {
        column: pd.to_numeric(rows.get(column), errors="coerce")
        for column in [
            "n_clusters_median",
            "ari_mean_pairwise",
            "platform_assoc_cramersV",
            "frac_nodes_in_small_clusters_median",
            "giant_cluster_frac_median",
        ]
    }
    mask = (
        numeric["n_clusters_median"].ge(2)
        & numeric["ari_mean_pairwise"].ge(float(leiden.min_ari))
        & numeric["frac_nodes_in_small_clusters_median"].le(float(leiden.max_frac_in_small))
        & numeric["giant_cluster_frac_median"].le(float(leiden.max_giant_frac))
    )
    if leiden.platform_assoc_max is not None:
        mask &= numeric["platform_assoc_cramersV"].notna()
        mask &= numeric["platform_assoc_cramersV"].le(float(leiden.platform_assoc_max))
    return mask.fillna(False)


def _plateau_centres_for_group(
    rows: pd.DataFrame,
    *,
    leiden: LeidenParams,
) -> list[dict]:
    ordered = rows.sort_values("resolution").reset_index(drop=True).copy()
    ordered["_scan_rank"] = np.arange(len(ordered), dtype=int)
    feasible = ordered[_candidate_feasibility_mask(ordered, leiden)].copy()
    if feasible.empty:
        return []

    previous_rank = feasible["_scan_rank"].shift(1)
    adjacent = feasible["_scan_rank"].eq(previous_rank + 1)
    stable_edge = adjacent & pd.to_numeric(
        feasible["resolution_previous_ari"], errors="coerce"
    ).ge(float(leiden.min_resolution_neighbor_ari))
    feasible["_plateau_id"] = (~stable_edge).cumsum()

    candidates: list[dict] = []
    for plateau_id, plateau in feasible.groupby("_plateau_id", sort=False):
        if len(plateau) < int(leiden.min_plateau_points):
            continue
        resolutions = pd.to_numeric(plateau["resolution"], errors="raise").to_numpy(float)
        log_centre = float(np.median(np.log(resolutions)))
        ranked = plateau.copy()
        ranked["_centre_distance"] = np.abs(np.log(resolutions) - log_centre)
        ranked = ranked.sort_values(
            ["_centre_distance", "ari_mean_pairwise", "resolution"],
            ascending=[True, False, True],
        )
        centre = ranked.iloc[0]
        internal_edges = pd.to_numeric(
            plateau.sort_values("resolution")["resolution_previous_ari"].iloc[1:],
            errors="coerce",
        )
        row = centre.to_dict()
        row.update(
            {
                "plateau_id": int(plateau_id),
                "plateau_qualified": True,
                "plateau_n_points": int(len(plateau)),
                "plateau_log_width": float(np.log(resolutions.max()) - np.log(resolutions.min())),
                "plateau_internal_ari_min": float(internal_edges.min()),
                "plateau_internal_ari_mean": float(internal_edges.mean()),
                "candidate_source": "stable_plateau",
            }
        )
        candidates.append(row)
    return candidates


def _limit_root_candidates(rows: pd.DataFrame, maximum: int) -> pd.DataFrame:
    rows = rows.sort_values(
        ["n_clusters_median", "candidate_source", "ari_mean_pairwise"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    if len(rows) <= int(maximum):
        return rows

    required_mask = rows["is_preliminary_selected"].astype(bool) | rows["is_complexity_anchor"].astype(bool)
    required = set(rows.index[required_mask].tolist())
    target = np.linspace(0, len(rows) - 1, int(maximum), dtype=int).tolist()
    selected = list(dict.fromkeys([*sorted(required), *target]))
    if len(selected) > int(maximum):
        non_required = [index for index in selected if index not in required]
        selected = [*sorted(required), *non_required[: max(0, int(maximum) - len(required))]]
    if len(selected) < int(maximum):
        selected.extend(index for index in rows.index if index not in selected)
    return rows.loc[selected[: int(maximum)]].sort_values("n_clusters_median").reset_index(drop=True)


def extract_plateau_candidates(
    parameter_scan: pd.DataFrame,
    preliminary_parameters: pd.DataFrame,
    *,
    leiden: LeidenParams,
    subsampling: SubsamplingSelectionParams,
) -> pd.DataFrame:
    """Extract one centre from each feasible stable plateau in the parameter scan."""

    scan = parameter_scan.copy()
    required = {
        "scan_stage",
        "root",
        "metric",
        "n_pcs",
        "K",
        "grid",
        "resolution",
        "resolution_previous_ari",
    }
    missing = sorted(required - set(scan.columns))
    if missing:
        raise ValueError(f"parameter_scan is missing columns: {missing}")
    scan = scan[scan["scan_stage"].astype(str).eq("stage2_resolution")].copy()
    if scan.empty:
        raise ValueError("parameter_scan contains no stage2_resolution rows")

    candidate_rows: list[dict] = []
    group_columns = ["root", "metric", "n_pcs", "K", "grid"]
    for _, group in scan.groupby(group_columns, sort=True, dropna=False):
        candidate_rows.extend(_plateau_centres_for_group(group, leiden=leiden))

    candidates = pd.DataFrame(candidate_rows)
    best = preliminary_parameters.copy()
    best["root"] = best["root"].astype(str)

    # Preserve the preliminary choice even if its fine plateau was not
    # recovered after applying the stricter subsampling feasibility checks.
    additions: list[dict] = []
    for _, selected in best.iterrows():
        root = str(selected["root"])
        metric = str(selected["metric"])
        n_pcs = int(selected["n_pcs"])
        k = int(selected["K"])
        resolution = float(selected["stage2_selected_resolution"])
        if not candidates.empty:
            same = candidates[
                candidates["root"].astype(str).eq(root)
                & candidates["metric"].astype(str).eq(metric)
                & pd.to_numeric(candidates["n_pcs"], errors="coerce").eq(n_pcs)
                & pd.to_numeric(candidates["K"], errors="coerce").eq(k)
            ]
            if not same.empty and np.isclose(
                pd.to_numeric(same["resolution"], errors="coerce"), resolution, rtol=1e-8, atol=1e-12
            ).any():
                continue
        pool = scan[
            scan["root"].astype(str).eq(root)
            & scan["metric"].astype(str).eq(metric)
            & pd.to_numeric(scan["n_pcs"], errors="coerce").eq(n_pcs)
            & pd.to_numeric(scan["K"], errors="coerce").eq(k)
        ].copy()
        if pool.empty:
            continue
        pool["_distance"] = np.abs(pd.to_numeric(pool["resolution"], errors="coerce") - resolution)
        row = pool.sort_values("_distance").iloc[0].to_dict()
        row.update(
            {
                "resolution": resolution,
                "plateau_id": -1,
                "plateau_qualified": bool(selected.get("stage2_plateau_qualified", False)),
                "plateau_n_points": np.nan,
                "plateau_log_width": np.nan,
                "plateau_internal_ari_min": np.nan,
                "plateau_internal_ari_mean": np.nan,
                "candidate_source": "preliminary_selected",
            }
        )
        additions.append(row)

    # Plateau stability can exclude an isolated but otherwise valid coarse
    # solution. Preserve the lowest-complexity feasible resolution so repeated
    # test whether subsampling supports it instead of only comparing fine
    # partitions selected by the preliminary scan.
    anchor_rows: list[dict] = []
    anchor_keys: set[tuple[str, str, int, int, float]] = set()
    for root, root_scan in scan.groupby("root", sort=True):
        feasible = root_scan[_candidate_feasibility_mask(root_scan, leiden)].copy()
        if feasible.empty:
            continue
        feasible["_grid_preference"] = feasible["grid"].astype(str).map({"broad": 0, "fine": 1}).fillna(2)
        feasible = feasible.sort_values(
            [
                "n_clusters_median",
                "ari_mean_pairwise",
                "platform_assoc_cramersV",
                "_grid_preference",
                "resolution",
            ],
            ascending=[True, False, True, True, True],
        )
        anchor = feasible.iloc[0].to_dict()
        key = (
            str(anchor["root"]),
            str(anchor["metric"]),
            int(anchor["n_pcs"]),
            int(anchor["K"]),
            float(anchor["resolution"]),
        )
        anchor_keys.add(key)
        anchor.update(
            {
                "plateau_id": -2,
                "plateau_qualified": False,
                "plateau_n_points": 1,
                "plateau_log_width": 0.0,
                "plateau_internal_ari_min": np.nan,
                "plateau_internal_ari_mean": np.nan,
                "candidate_source": "complexity_anchor",
            }
        )
        anchor_rows.append(anchor)

    supplemental = [*additions, *anchor_rows]
    if supplemental:
        candidates = pd.concat([candidates, pd.DataFrame(supplemental)], ignore_index=True, sort=False)
    if candidates.empty:
        raise ValueError("No subsampling candidates could be extracted")

    candidates["root"] = candidates["root"].astype(str)
    candidates["metric"] = candidates["metric"].astype(str)
    candidates["n_pcs"] = pd.to_numeric(candidates["n_pcs"], errors="raise").astype(int)
    candidates["K"] = pd.to_numeric(candidates["K"], errors="raise").astype(int)
    candidates["resolution"] = pd.to_numeric(candidates["resolution"], errors="raise")
    candidates["n_clusters_median"] = pd.to_numeric(candidates["n_clusters_median"], errors="coerce")
    candidates = candidates.drop_duplicates(
        ["root", "metric", "n_pcs", "K", "resolution"], keep="first"
    ).reset_index(drop=True)

    best_keys = {
        (
            str(row["root"]),
            str(row["metric"]),
            int(row["n_pcs"]),
            int(row["K"]),
            float(row["stage2_selected_resolution"]),
        )
        for _, row in best.iterrows()
    }
    candidates["is_preliminary_selected"] = [
        any(
            root == key[0]
            and metric == key[1]
            and n_pcs == key[2]
            and k == key[3]
            and np.isclose(resolution, key[4], rtol=1e-8, atol=1e-12)
            for key in best_keys
        )
        for root, metric, n_pcs, k, resolution in candidates[
            ["root", "metric", "n_pcs", "K", "resolution"]
        ].itertuples(index=False, name=None)
    ]
    candidates["is_complexity_anchor"] = [
        any(
            root == key[0]
            and metric == key[1]
            and n_pcs == key[2]
            and k == key[3]
            and np.isclose(resolution, key[4], rtol=1e-8, atol=1e-12)
            for key in anchor_keys
        )
        for root, metric, n_pcs, k, resolution in candidates[
            ["root", "metric", "n_pcs", "K", "resolution"]
        ].itertuples(index=False, name=None)
    ]
    candidates["scan_feasible"] = _candidate_feasibility_mask(candidates, leiden)

    limited = []
    for root, root_rows in candidates.groupby("root", sort=True):
        selected = _limit_root_candidates(root_rows, int(subsampling.max_candidates_per_root))
        selected = selected.copy()
        selected["candidate_rank"] = np.arange(1, len(selected) + 1)
        selected["candidate_id"] = [f"{root}__P{rank:02d}" for rank in selected["candidate_rank"]]
        limited.append(selected)
    return pd.concat(limited, ignore_index=True).sort_values(["root", "candidate_rank"]).reset_index(drop=True)


def build_subsample_graph(
    coords: pd.DataFrame,
    *,
    sample_indices: np.ndarray,
    metric: str,
    n_pcs: int,
    k: int,
    sigma_quantile: float,
    make_undirected: bool,
    combine_mode: str,
):
    """Build a parameterized KNN graph for a coordinate subsample."""
    selected_dimensions = dimension_columns(coords)[: int(n_pcs)]
    if not selected_dimensions:
        raise ValueError("No D* dimensions found in analysis IR")
    subset = coords.iloc[np.asarray(sample_indices, dtype=int)]
    ids = subset.index.astype(str).to_numpy(dtype=object)
    values = subset[selected_dimensions].to_numpy(dtype=np.float32)
    neighbour_indices, neighbour_distances = knn_indices_and_distances(
        Z=values, k=int(k), metric=str(metric)
    )
    if neighbour_indices.size == 0:
        raise ValueError(f"n={len(subset)} is too small for K={k}")
    graph, _ = build_igraph_from_knn(
        ids=ids,
        neigh_idx=neighbour_indices,
        neigh_dists=neighbour_distances,
        sigma_quantile=float(sigma_quantile),
        make_undirected=bool(make_undirected),
        combine_mode=str(combine_mode),
    )
    return graph


def _partition_metrics(
    full_labels: np.ndarray,
    subset_labels: np.ndarray,
    *,
    candidate_id: str,
    repeat: int,
) -> tuple[dict, list[dict], list[dict]]:
    full_labels = np.asarray(full_labels, dtype=np.int32)
    subset_labels = np.asarray(subset_labels, dtype=np.int32)
    valid = (full_labels >= 0) & (subset_labels >= 0)
    coverage = float(valid.mean()) if len(valid) else 0.0
    if int(valid.sum()) < 2:
        return {
            "ari": np.nan,
            "nmi": np.nan,
            "matched_fraction": np.nan,
            "eval_coverage": coverage,
            "context_recall_q10": np.nan,
        }, [], []

    truth = full_labels[valid]
    observed = subset_labels[valid]
    contingency = pd.crosstab(pd.Series(truth, name="full"), pd.Series(observed, name="subsample"))
    overlap_rows = [
        {
            "candidate_id": candidate_id,
            "repeat": int(repeat),
            "full_cluster": int(full_cluster),
            "subsample_cluster": int(subsample_cluster),
            "n_overlap": int(contingency.loc[full_cluster, subsample_cluster]),
        }
        for full_cluster in contingency.index
        for subsample_cluster in contingency.columns
        if int(contingency.loc[full_cluster, subsample_cluster]) > 0
    ]
    row_indices, column_indices = linear_sum_assignment(-contingency.to_numpy(dtype=int))
    matched = int(contingency.to_numpy(dtype=int)[row_indices, column_indices].sum())

    context_rows: list[dict] = []
    recalls: list[float] = []
    for full_cluster in contingency.index:
        overlaps = contingency.loc[full_cluster]
        matched_cluster = int(overlaps.idxmax())
        overlap = int(overlaps.max())
        full_count = int(overlaps.sum())
        subset_count = int(contingency[matched_cluster].sum())
        recall = float(overlap / full_count) if full_count else np.nan
        precision = float(overlap / subset_count) if subset_count else np.nan
        denominator = full_count + subset_count - overlap
        jaccard = float(overlap / denominator) if denominator else np.nan
        recalls.append(recall)
        context_rows.append(
            {
                "candidate_id": candidate_id,
                "repeat": int(repeat),
                "full_cluster": int(full_cluster),
                "matched_subsample_cluster": matched_cluster,
                "n_full_subset": full_count,
                "n_matched_overlap": overlap,
                "recall": recall,
                "precision": precision,
                "jaccard": jaccard,
            }
        )

    metrics = {
        "ari": float(adjusted_rand_score(truth, observed)),
        "nmi": float(normalized_mutual_info_score(truth, observed)),
        "matched_fraction": float(matched / len(truth)),
        "eval_coverage": coverage,
        "context_recall_q10": float(np.quantile(recalls, 0.10)) if recalls else np.nan,
    }
    return metrics, context_rows, overlap_rows


def evaluate_candidates_by_subsampling(
    candidates: pd.DataFrame,
    *,
    ir_dir: Path,
    leiden: LeidenParams,
    subsampling: SubsamplingSelectionParams,
    sigma_quantile: float,
    make_undirected: bool,
    combine_mode: str,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild candidate partitions after repeated sample perturbation."""

    candidate_updates: list[dict] = []
    run_rows: list[dict] = []
    context_rows: list[dict] = []
    overlap_rows: list[dict] = []
    subsample_min_cluster_size = scaled_subsample_min_cluster_size(
        leiden.min_cluster_size,
        subsampling.sample_fraction,
    )

    for root, root_candidates in candidates.groupby("root", sort=True):
        checkpoint_root = Path(checkpoint_dir) / str(root) if checkpoint_dir is not None else None
        checkpoint_signature = {
            "schema_version": 2,
            "candidates": [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "metric": str(row["metric"]),
                    "n_pcs": int(row["n_pcs"]),
                    "K": int(row["K"]),
                    "resolution": float(row["resolution"]),
                }
                for _, row in root_candidates.sort_values("candidate_id").iterrows()
            ],
            "leiden_min_cluster_size": int(leiden.min_cluster_size),
            "subsampling": asdict(subsampling),
            "graph": {
                "sigma_quantile": float(sigma_quantile),
                "make_undirected": bool(make_undirected),
                "combine_mode": str(combine_mode),
            },
        }
        if checkpoint_root is not None:
            update_path = checkpoint_root / "candidate_updates.tsv"
            run_path = checkpoint_root / "subsampling_runs.tsv"
            context_path = checkpoint_root / "subsampling_by_subtype.tsv"
            overlap_path = checkpoint_root / "subsampling_overlap_cells.tsv"
            signature_path = checkpoint_root / "checkpoint_manifest.json"
            if (
                update_path.exists()
                and run_path.exists()
                and context_path.exists()
                and overlap_path.exists()
                and signature_path.exists()
                and json.loads(signature_path.read_text()) == checkpoint_signature
            ):
                saved_updates = pd.read_csv(update_path, sep="\t")
                expected_ids = set(root_candidates["candidate_id"].astype(str))
                saved_ids = set(saved_updates["candidate_id"].astype(str))
                if expected_ids == saved_ids:
                    print(f"[subsampling] {root}: resume from checkpoint", flush=True)
                    candidate_updates.extend(saved_updates.to_dict(orient="records"))
                    run_rows.extend(pd.read_csv(run_path, sep="\t").to_dict(orient="records"))
                    try:
                        saved_contexts = pd.read_csv(context_path, sep="\t")
                    except pd.errors.EmptyDataError:
                        saved_contexts = pd.DataFrame()
                    context_rows.extend(saved_contexts.to_dict(orient="records"))
                    try:
                        saved_overlaps = pd.read_csv(overlap_path, sep="\t")
                    except pd.errors.EmptyDataError:
                        saved_overlaps = pd.DataFrame()
                    overlap_rows.extend(saved_overlaps.to_dict(orient="records"))
                    continue

        print(f"[subsampling] {root}: {len(root_candidates)} candidate(s)", flush=True)
        candidate_start = len(candidate_updates)
        run_start = len(run_rows)
        context_start = len(context_rows)
        overlap_start = len(overlap_rows)
        coords_path = Path(ir_dir) / "perroot" / str(root) / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(coords_path)
        coords = pd.read_parquet(coords_path)
        coords.index = coords.index.astype(str)
        all_indices = np.arange(len(coords), dtype=int)
        root_offset = int(zlib.crc32(str(root).encode("utf-8")))
        subsets: dict[int, np.ndarray] = {}
        for repeat in range(int(subsampling.repeats)):
            rng = np.random.default_rng(int(subsampling.random_seed) + root_offset + repeat)
            n_subset = max(2, int(np.floor(float(subsampling.sample_fraction) * len(coords))))
            subsets[repeat] = np.sort(rng.choice(all_indices, size=n_subset, replace=False))

        graph_columns = ["metric", "n_pcs", "K"]
        for graph_values, graph_candidates in root_candidates.groupby(graph_columns, sort=True):
            metric, n_pcs, k = str(graph_values[0]), int(graph_values[1]), int(graph_values[2])
            full_graph = build_subsample_graph(
                coords,
                sample_indices=all_indices,
                metric=metric,
                n_pcs=n_pcs,
                k=k,
                sigma_quantile=float(sigma_quantile),
                make_undirected=bool(make_undirected),
                combine_mode=str(combine_mode),
            )
            full_partitions: dict[str, np.ndarray] = {}
            full_cluster_counts: dict[str, int] = {}
            for _, candidate in graph_candidates.iterrows():
                candidate_id = str(candidate["candidate_id"])
                membership, _ = run_leiden_cpm(
                    full_graph,
                    resolution=float(candidate["resolution"]),
                    seed=int(subsampling.partition_seed),
                )
                membership = mark_small_clusters(membership, int(leiden.min_cluster_size))
                full_partitions[candidate_id] = membership
                full_cluster_counts[candidate_id] = int(
                    pd.Series(membership[membership >= 0]).nunique()
                )
                candidate_updates.append(
                    {
                        "candidate_id": candidate_id,
                        "n_samples_full": int(len(coords)),
                        "n_clusters_full": full_cluster_counts[candidate_id],
                        "full_outlier_fraction": float(np.mean(membership < 0)),
                        "full_min_cluster_size": int(leiden.min_cluster_size),
                    }
                )
            del full_graph

            for repeat, subset_indices in subsets.items():
                try:
                    subset_graph = build_subsample_graph(
                        coords,
                        sample_indices=subset_indices,
                        metric=metric,
                        n_pcs=n_pcs,
                        k=k,
                        sigma_quantile=float(sigma_quantile),
                        make_undirected=bool(make_undirected),
                        combine_mode=str(combine_mode),
                    )
                except Exception as error:
                    for candidate_id in full_partitions:
                        run_rows.append(
                            {
                                "root": str(root),
                                "candidate_id": candidate_id,
                                "repeat": int(repeat),
                                "n_subset": int(len(subset_indices)),
                                "n_clusters_full": full_cluster_counts[candidate_id],
                                "n_clusters_subsample": np.nan,
                                "subsample_min_cluster_size": int(subsample_min_cluster_size),
                                "cluster_count_ratio": np.nan,
                                "ari": np.nan,
                                "nmi": np.nan,
                                "matched_fraction": np.nan,
                                "eval_coverage": np.nan,
                                "context_recall_q10": np.nan,
                                "success": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    continue

                for _, candidate in graph_candidates.iterrows():
                    candidate_id = str(candidate["candidate_id"])
                    full_cluster_count = full_cluster_counts[candidate_id]
                    try:
                        subset_membership, _ = run_leiden_cpm(
                            subset_graph,
                            resolution=float(candidate["resolution"]),
                            seed=int(subsampling.partition_seed),
                        )
                        subset_membership = mark_small_clusters(
                            subset_membership, int(subsample_min_cluster_size)
                        )
                        full_on_subset = full_partitions[candidate_id][subset_indices]
                        metrics, per_context, overlap_cells = _partition_metrics(
                            full_on_subset,
                            subset_membership,
                            candidate_id=candidate_id,
                            repeat=int(repeat),
                        )
                        subset_cluster_count = int(
                            pd.Series(subset_membership[subset_membership >= 0]).nunique()
                        )
                        run_rows.append(
                            {
                                "root": str(root),
                                "candidate_id": candidate_id,
                                "repeat": int(repeat),
                                "n_subset": int(len(subset_indices)),
                                "n_clusters_full": full_cluster_count,
                                "n_clusters_subsample": subset_cluster_count,
                                "subsample_min_cluster_size": int(subsample_min_cluster_size),
                                "cluster_count_ratio": (
                                    float(subset_cluster_count / full_cluster_count)
                                    if full_cluster_count
                                    else np.nan
                                ),
                                "success": True,
                                "error": "",
                                **metrics,
                            }
                        )
                        for row in per_context:
                            row["root"] = str(root)
                        context_rows.extend(per_context)
                        for row in overlap_cells:
                            row["root"] = str(root)
                        overlap_rows.extend(overlap_cells)
                    except Exception as error:
                        run_rows.append(
                            {
                                "root": str(root),
                                "candidate_id": candidate_id,
                                "repeat": int(repeat),
                                "n_subset": int(len(subset_indices)),
                                "n_clusters_full": full_cluster_count,
                                "n_clusters_subsample": np.nan,
                                "subsample_min_cluster_size": int(subsample_min_cluster_size),
                                "cluster_count_ratio": np.nan,
                                "ari": np.nan,
                                "nmi": np.nan,
                                "matched_fraction": np.nan,
                                "eval_coverage": np.nan,
                                "context_recall_q10": np.nan,
                                "success": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                del subset_graph

        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(candidate_updates[candidate_start:]).to_csv(
                checkpoint_root / "candidate_updates.tsv", sep="\t", index=False
            )
            pd.DataFrame(run_rows[run_start:]).to_csv(
                checkpoint_root / "subsampling_runs.tsv", sep="\t", index=False
            )
            context_checkpoint = pd.DataFrame(context_rows[context_start:])
            if context_checkpoint.empty:
                context_checkpoint = pd.DataFrame(
                    columns=[
                        "candidate_id",
                        "repeat",
                        "full_cluster",
                        "matched_subsample_cluster",
                        "n_full_subset",
                        "n_matched_overlap",
                        "recall",
                        "precision",
                        "jaccard",
                        "root",
                    ]
                )
            context_checkpoint.to_csv(
                checkpoint_root / "subsampling_by_subtype.tsv", sep="\t", index=False
            )
            overlap_checkpoint = pd.DataFrame(overlap_rows[overlap_start:])
            if overlap_checkpoint.empty:
                overlap_checkpoint = pd.DataFrame(
                    columns=[
                        "candidate_id",
                        "repeat",
                        "full_cluster",
                        "subsample_cluster",
                        "n_overlap",
                        "root",
                    ]
                )
            overlap_checkpoint.to_csv(
                checkpoint_root / "subsampling_overlap_cells.tsv",
                sep="\t",
                index=False,
            )
            (checkpoint_root / "checkpoint_manifest.json").write_text(
                json.dumps(checkpoint_signature, indent=2, sort_keys=True) + "\n"
            )

    updated_candidates = candidates.merge(pd.DataFrame(candidate_updates), on="candidate_id", how="left")
    return (
        updated_candidates,
        pd.DataFrame(run_rows),
        pd.DataFrame(context_rows),
        pd.DataFrame(overlap_rows),
    )


def summarize_context_stability(
    contexts: pd.DataFrame,
    *,
    sample_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize repeat-level recovery for every full-data context."""

    required = {
        "candidate_id",
        "full_cluster",
        "n_full_subset",
        "recall",
        "precision",
        "jaccard",
    }
    missing = sorted(required - set(contexts.columns))
    if missing:
        raise ValueError(f"subsampling_by_subtype is missing columns: {missing}")
    if contexts.empty:
        raise ValueError("subsampling_by_subtype contains no context-level results")

    per_context = (
        contexts.groupby(["candidate_id", "full_cluster"], as_index=False)
        .agg(
            context_repeats=("repeat", "nunique"),
            median_subset_size=("n_full_subset", "median"),
            context_recall_q10=("recall", lambda values: float(values.quantile(0.10))),
            context_precision_q10=(
                "precision", lambda values: float(values.quantile(0.10))
            ),
            context_jaccard_q10=("jaccard", lambda values: float(values.quantile(0.10))),
        )
    )
    per_context["estimated_full_size"] = (
        per_context["median_subset_size"] / float(sample_fraction)
    )
    per_candidate = (
        per_context.groupby("candidate_id", as_index=False)
        .agg(
            n_contexts_evaluated=("full_cluster", "nunique"),
            min_context_repeats=("context_repeats", "min"),
            estimated_min_full_size=("estimated_full_size", "min"),
            worst_context_recall_q10=("context_recall_q10", "min"),
            context_recall_q10_across_contexts=(
                "context_recall_q10", lambda values: float(values.quantile(0.10))
            ),
            worst_context_precision_q10=("context_precision_q10", "min"),
            context_precision_q10_across_contexts=(
                "context_precision_q10", lambda values: float(values.quantile(0.10))
            ),
            worst_context_jaccard_q10=("context_jaccard_q10", "min"),
            context_jaccard_q10_across_contexts=(
                "context_jaccard_q10", lambda values: float(values.quantile(0.10))
            ),
        )
    )
    return per_context, per_candidate


def summarize_and_select_candidates(
    candidates: pd.DataFrame,
    runs: pd.DataFrame,
    contexts: pd.DataFrame,
    *,
    leiden: LeidenParams,
    subsampling: SubsamplingSelectionParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply stability gates and select a robust resolution plateau per root."""

    successful_mask = runs["success"].astype(str).str.lower().eq("true")
    successful = runs[successful_mask].copy()
    if successful.empty:
        raise ValueError("No successful subsampling runs")
    summary = successful.groupby("candidate_id", as_index=False).agg(
        n_repeats_success=("repeat", "nunique"),
        subsample_ari_mean=("ari", "mean"),
        subsample_ari_std=("ari", "std"),
        subsample_ari_median=("ari", "median"),
        subsample_ari_q10=("ari", lambda values: float(values.quantile(0.10))),
        matched_fraction_mean=("matched_fraction", "mean"),
        eval_coverage_mean=("eval_coverage", "mean"),
        context_recall_q10_median=("context_recall_q10", "median"),
        cluster_count_ratio_median=("cluster_count_ratio", "median"),
    )
    summary["subsample_ari_se"] = summary["subsample_ari_std"].fillna(0.0) / np.sqrt(
        summary["n_repeats_success"].clip(lower=1)
    )
    summary = candidates.merge(summary, on="candidate_id", how="left")
    _, context_summary = summarize_context_stability(
        contexts,
        sample_fraction=float(subsampling.sample_fraction),
    )
    summary = summary.merge(context_summary, on="candidate_id", how="left")
    scan_feasible = (
        summary.get("scan_feasible", pd.Series(False, index=summary.index))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    candidate_source = summary.get(
        "candidate_source", pd.Series("", index=summary.index)
    ).fillna("").astype(str)
    plateau_qualified = (
        summary.get("plateau_qualified", pd.Series(False, index=summary.index))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    summary["automatic_selection_eligible"] = (
        candidate_source.eq("stable_plateau") & plateau_qualified
    )
    summary["subsampling_pass"] = (
        scan_feasible
        & summary["n_clusters_full"].ge(2)
        & summary["full_outlier_fraction"].le(float(leiden.max_frac_in_small))
        & summary["n_repeats_success"].ge(int(subsampling.repeats))
        & summary["subsample_ari_mean"].ge(float(subsampling.min_subsample_ari))
        & summary["subsample_ari_q10"].ge(float(subsampling.min_subsample_ari))
        & summary["matched_fraction_mean"].ge(float(subsampling.min_matched_fraction))
        & summary["eval_coverage_mean"].ge(float(subsampling.min_eval_coverage))
        & summary["context_recall_q10_median"].ge(float(subsampling.min_context_recall_q10))
        & summary["n_contexts_evaluated"].eq(summary["n_clusters_full"])
        & summary["min_context_repeats"].ge(int(subsampling.repeats))
        & summary["worst_context_precision_q10"].ge(
            float(subsampling.min_worst_context_precision_q10)
        )
        & summary["worst_context_jaccard_q10"].ge(
            float(subsampling.min_worst_context_jaccard_q10)
        )
        & summary["cluster_count_ratio_median"].between(
            float(subsampling.min_cluster_count_ratio),
            float(subsampling.max_cluster_count_ratio),
            inclusive="both",
        )
    ).fillna(False)

    selected_rows: list[dict] = []
    review_rows: list[dict] = []
    for root, root_summary in summary.groupby("root", sort=True):
        all_passing = root_summary[root_summary["subsampling_pass"]].copy()
        passing = all_passing[all_passing["automatic_selection_eligible"]].copy()
        if passing.empty:
            available = (
                all_passing.copy()
                if not all_passing.empty
                else root_summary[root_summary["subsample_ari_mean"].notna()].copy()
            )
            if available.empty:
                review_rows.append({"root": str(root), "reason": "no_successful_candidate"})
                continue
            chosen = available.sort_values(
                [
                    "subsample_ari_q10",
                    "subsample_ari_mean",
                    "matched_fraction_mean",
                    "n_clusters_full",
                ],
                ascending=[False, False, False, True],
            ).iloc[0].copy()
            chosen["selection_status"] = "review_best_available"
            reason = (
                "no_stable_plateau_candidate_passed"
                if not all_passing.empty
                else "no_candidate_passed_subsampling_gates"
            )
            chosen["selection_reason"] = reason
            chosen["one_se_threshold"] = np.nan
            review_rows.append(
                {
                    "root": str(root),
                    "reason": reason,
                    "best_candidate_id": str(chosen["candidate_id"]),
                }
            )
        else:
            best = passing.sort_values("subsample_ari_mean", ascending=False).iloc[0]
            threshold = float(best["subsample_ari_mean"]) - float(
                subsampling.one_se_multiplier
            ) * float(best["subsample_ari_se"])
            equivalent = passing[passing["subsample_ari_mean"].ge(threshold)].copy()
            # Stability metrics systematically favour coarse partitions. Within
            # the one-SE equivalence band, prefer candidates that remain stable
            # across the broadest resolution interval. Cluster count is only a
            # final tie-breaker and cannot drive roots toward two-cluster anchors.
            chosen = equivalent.sort_values(
                [
                    "plateau_log_width",
                    "plateau_internal_ari_min",
                    "plateau_internal_ari_mean",
                    "subsample_ari_q10",
                    "context_recall_q10_median",
                    "subsample_ari_mean",
                    "matched_fraction_mean",
                    "platform_assoc_cramersV",
                    "n_clusters_full",
                    "resolution",
                ],
                ascending=[False, False, False, False, False, False, False, True, True, True],
            ).iloc[0].copy()
            chosen["selection_status"] = "subsampling_selected"
            chosen["selection_reason"] = (
                "most_robust_plateau_within_one_se_of_best_subsampling_stability"
            )
            chosen["one_se_threshold"] = threshold

        chosen["stage2_selected_resolution"] = float(chosen["resolution"])
        selected_rows.append(chosen.to_dict())

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected = selected.sort_values("root").reset_index(drop=True)
    reviews = pd.DataFrame(
        review_rows,
        columns=["root", "reason", "best_candidate_id"],
    )
    return summary.sort_values(["root", "n_clusters_full"]), selected, reviews


def finalize_existing_subsampling_selection(
    *,
    out_dir: Path,
    leiden: LeidenParams,
    subsampling: SubsamplingSelectionParams,
    roots: list[str] | None = None,
) -> dict[str, Path]:
    """Reapply final selection policy without rebuilding subsampled graphs."""

    out_dir = Path(out_dir)
    candidate_path = out_dir / "candidate_plateaus.tsv"
    runs_path = out_dir / "subsampling_runs.tsv"
    contexts_path = out_dir / "subsampling_by_subtype.tsv"
    overlap_path = out_dir / "subsampling_overlap_cells.tsv"
    if (
        not candidate_path.exists()
        or not runs_path.exists()
        or not contexts_path.exists()
        or not overlap_path.exists()
    ):
        raise FileNotFoundError(
            "Finalization requires candidate_plateaus.tsv, subsampling_runs.tsv, "
            "subsampling_by_subtype.tsv, and subsampling_overlap_cells.tsv"
        )

    candidates = pd.read_csv(candidate_path, sep="\t", low_memory=False)
    runs = pd.read_csv(runs_path, sep="\t", low_memory=False)
    contexts = pd.read_csv(contexts_path, sep="\t", low_memory=False)
    if roots:
        root_set = {str(root) for root in roots}
        candidates = candidates[candidates["root"].astype(str).isin(root_set)].copy()
        candidate_ids = set(candidates["candidate_id"].astype(str))
        runs = runs[runs["candidate_id"].astype(str).isin(candidate_ids)].copy()
        contexts = contexts[contexts["candidate_id"].astype(str).isin(candidate_ids)].copy()

    summary, selected, reviews = summarize_and_select_candidates(
        candidates,
        runs,
        contexts,
        leiden=leiden,
        subsampling=subsampling,
    )
    summary_path = out_dir / "subsampling_by_candidate.tsv"
    selected_path = out_dir / "selected_parameters.tsv"
    review_path = out_dir / "roots_requiring_review.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    selected.to_csv(selected_path, sep="\t", index=False)
    reviews.to_csv(review_path, sep="\t", index=False)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["finalized_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["final_selection_policy"] = (
        "stable_plateau_robustness_within_one_se"
    )
    manifest["counts"] = {
        **dict(manifest.get("counts") or {}),
        "selected_roots": int(len(selected)),
        "review_roots": int(len(reviews)),
        "selected_clusters_total": int(
            pd.to_numeric(
                selected.loc[
                    selected["selection_status"].eq("subsampling_selected"),
                    "n_clusters_full",
                ],
                errors="coerce",
            ).sum()
        )
        if not selected.empty
        else 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {
        "summary": summary_path,
        "selected": selected_path,
        "reviews": review_path,
        "manifest": manifest_path,
    }


def run_subsampling_model_selection(
    *,
    parameter_scan_path: Path,
    preliminary_parameters_path: Path,
    ir_dir: Path,
    out_dir: Path,
    leiden: LeidenParams,
    subsampling: SubsamplingSelectionParams,
    sigma_quantile: float,
    make_undirected: bool,
    combine_mode: str,
    roots: list[str] | None = None,
) -> dict[str, Path]:
    parameter_scan = pd.read_csv(parameter_scan_path, sep="\t", low_memory=False)
    preliminary_parameters = pd.read_csv(
        preliminary_parameters_path,
        sep="\t",
        low_memory=False,
    )
    if roots:
        root_set = {str(root) for root in roots}
        parameter_scan = parameter_scan[parameter_scan["root"].astype(str).isin(root_set)].copy()
        preliminary_parameters = preliminary_parameters[
            preliminary_parameters["root"].astype(str).isin(root_set)
        ].copy()
        missing = sorted(root_set - set(preliminary_parameters["root"].astype(str)))
        if missing:
            raise ValueError(f"Requested roots are missing from preliminary_parameters: {missing}")
    candidates = extract_plateau_candidates(
        parameter_scan,
        preliminary_parameters,
        leiden=leiden,
        subsampling=subsampling,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "candidate_plateaus.tsv"
    candidates.to_csv(candidate_path, sep="\t", index=False)

    candidates, runs, contexts, overlap_cells = evaluate_candidates_by_subsampling(
        candidates,
        ir_dir=Path(ir_dir),
        leiden=leiden,
        subsampling=subsampling,
        sigma_quantile=float(sigma_quantile),
        make_undirected=bool(make_undirected),
        combine_mode=str(combine_mode),
        checkpoint_dir=out_dir / "checkpoints",
    )
    candidates.to_csv(candidate_path, sep="\t", index=False)
    runs_path = out_dir / "subsampling_runs.tsv"
    contexts_path = out_dir / "subsampling_by_subtype.tsv"
    overlap_path = out_dir / "subsampling_overlap_cells.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    contexts.to_csv(contexts_path, sep="\t", index=False)
    overlap_cells.to_csv(overlap_path, sep="\t", index=False)

    summary, selected, reviews = summarize_and_select_candidates(
        candidates,
        runs,
        contexts,
        leiden=leiden,
        subsampling=subsampling,
    )
    summary_path = out_dir / "subsampling_by_candidate.tsv"
    selected_path = out_dir / "selected_parameters.tsv"
    review_path = out_dir / "roots_requiring_review.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    selected.to_csv(selected_path, sep="\t", index=False)
    reviews.to_csv(review_path, sep="\t", index=False)

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "parameter_scan": str(Path(parameter_scan_path).resolve()),
            "preliminary_parameters": str(Path(preliminary_parameters_path).resolve()),
            "ir_dir": str(Path(ir_dir).resolve()),
        },
        "leiden": asdict(leiden),
        "subsampling": asdict(subsampling),
        "final_selection_policy": "stable_plateau_robustness_within_one_se",
        "counts": {
            "roots": int(candidates["root"].nunique()),
            "candidates": int(len(candidates)),
            "selected_roots": int(len(selected)),
            "review_roots": int(len(reviews)),
            "selected_clusters_total": int(
                pd.to_numeric(
                    selected.loc[selected["selection_status"].eq("subsampling_selected"), "n_clusters_full"],
                    errors="coerce",
                ).sum()
            )
            if not selected.empty
            else 0,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {
        "candidates": candidate_path,
        "runs": runs_path,
        "contexts": contexts_path,
        "overlap_cells": overlap_path,
        "summary": summary_path,
        "selected": selected_path,
        "reviews": review_path,
        "manifest": manifest_path,
    }
