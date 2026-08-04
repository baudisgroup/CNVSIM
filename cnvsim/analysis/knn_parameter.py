from __future__ import annotations

"""Runtime selection of root-specific kNN and preliminary Leiden parameters.

Goal (mainline):
- Primary: graph health + conservative graph preference
- Platform mixing is a guardrail/diagnostic, not the first sorting key.
- Morphology is only a weak guardrail (optional).

The selected table is an output consumed by the kNN and Leiden stages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cnvsim.analysis.ir import dimension_columns


@dataclass(frozen=True)
class KNNGrid:
    metrics: tuple[str, ...] = ("euclidean", "cosine")
    n_pcs_list: tuple[int, ...] = (10, 20, 30, 50)
    k_list: tuple[int, ...] = (30, 50, 80)


@dataclass(frozen=True)
class Stage1Thresholds:
    # graph health
    min_giant_frac: float = 0.99
    max_components: int = 2
    # platform mixing (enrichment over random baseline)
    max_platform_enrichment: float | None = None  # if None, do not hard-cap; use for ranking only
    platform_enrichment_tolerance: float | None = 0.05
    preferred_metric_order: tuple[str, ...] = ("cosine", "euclidean")
    preferred_k_order: tuple[int, ...] = (80, 50, 30)
    preferred_n_pcs: tuple[int, ...] = (30, 50)
    # morphology weak constraint
    max_morph_purity_drop: float | None = None


@dataclass(frozen=True)
class Stage2Leiden:
    enabled: bool = True
    top_n: int = 2
    sigma_quantile: float = 0.90
    make_undirected: bool = True
    combine_mode: str = "max"  # max|mean


def _load_ir_coords_meta(*, ir_root_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords_p = Path(ir_root_dir) / "coords.parquet"
    meta_p = Path(ir_root_dir) / "meta.parquet"
    if not coords_p.exists():
        raise FileNotFoundError(coords_p)
    if not meta_p.exists():
        raise FileNotFoundError(meta_p)
    coords = pd.read_parquet(coords_p)
    meta = pd.read_parquet(meta_p)
    return coords, meta


def knn_indices_and_distances(
    *, Z: np.ndarray, k: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-self nearest-neighbour indices and distances."""
    from sklearn.neighbors import NearestNeighbors

    n = int(Z.shape[0])
    if n <= k + 1:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric=str(metric), algorithm="auto")
    nn.fit(Z)
    d, idx = nn.kneighbors(Z, return_distance=True)
    # drop self neighbor
    return idx[:, 1:].astype(np.int32, copy=False), d[:, 1:].astype(np.float32, copy=False)


def _neighbor_purity(*, idx: np.ndarray, labels: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    labs = labels.astype(object)
    pur = [(labs[idx[i]] == labs[i]).mean() for i in range(idx.shape[0])]
    return float(np.mean(pur))


def _random_match_rate(labels: np.ndarray) -> float:
    # sum_p freq(p)^2
    s = pd.Series(labels.astype(object)).value_counts(normalize=True)
    return float(np.sum(np.square(s.to_numpy(dtype=float))))


def build_igraph_from_knn(
    *,
    ids: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_dists: np.ndarray,
    sigma_quantile: float,
    make_undirected: bool,
    combine_mode: str,
):
    """Build the weighted analysis graph shared by parameter selection stages."""
    import igraph as ig
    from cnvsim.analysis.graph import rbf_weights_from_dists

    n = int(len(ids))
    k = int(neigh_idx.shape[1])

    flat_d = neigh_dists.reshape(-1)
    sigma = float(np.quantile(flat_d, float(sigma_quantile))) if flat_d.size else 1.0
    if (not np.isfinite(sigma)) or sigma <= 0:
        pos = flat_d[flat_d > 0]
        sigma = float(np.median(pos)) if len(pos) else 1.0

    W = rbf_weights_from_dists(neigh_dists, sigma=sigma)

    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = neigh_idx.reshape(-1).astype(np.int32)
    w = W.reshape(-1)

    g = ig.Graph(n=n, directed=True)
    g.vs["analysis_id"] = ids.astype(str).tolist()
    g.add_edges(list(zip(src.tolist(), dst.tolist())))
    g.es["weight"] = w.tolist()

    if make_undirected:
        if str(combine_mode) == "max":
            comb = {"weight": "max"}
        elif str(combine_mode) == "mean":
            comb = {"weight": "mean"}
        else:
            raise ValueError("combine_mode must be 'max' or 'mean'")
        g = g.as_undirected(combine_edges=comb)
        g = g.simplify(combine_edges=comb)

    info = {"sigma": sigma, "sigma_quantile": float(sigma_quantile), "edges": int(g.ecount())}
    return g, info


def _graph_health_from_knn(*, n: int, idx: np.ndarray) -> tuple[int, float]:
    """Return (n_components, giant_frac) for an undirected unweighted kNN graph."""
    import igraph as ig

    if idx.size == 0:
        return 0, float("nan")

    k = int(idx.shape[1])
    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = idx.reshape(-1).astype(np.int32)
    g = ig.Graph(n=n, directed=True)
    g.add_edges(list(zip(src.tolist(), dst.tolist())))
    g = g.as_undirected(combine_edges=None)
    g = g.simplify()

    comps = g.components()
    n_comps = int(len(comps))
    sizes = np.asarray(comps.sizes(), dtype=int) if n_comps else np.asarray([], dtype=int)
    giant = int(sizes.max()) if sizes.size else 0
    giant_frac = float(giant / max(1, n))
    return n_comps, giant_frac


def scan_root_parameters_from_ir(
    *,
    ir_dir: Path,
    roots: Iterable[str],
    platform_col: str = "platform_family_refined",
    morph_col: str | None = "icdo_morphology_id",
    grid: KNNGrid = KNNGrid(),
    stage1: Stage1Thresholds = Stage1Thresholds(),
    stage2: Stage2Leiden = Stage2Leiden(),
    leiden_params=None,
    resolution_grid: np.ndarray | None = None,
    fine_grid_points: int = 12,
    seeds: tuple[int, ...] | None = None,
    stage1_by_root: dict[str, Stage1Thresholds] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select per-root best (metric,n_pcs,K) from IR coords.

    Returns:
      best_df: one row per root with chosen setting (+ optional stability fields)
      grid_df: all scanned candidates (stage1 metrics; stage2 stability for evaluated candidates)

    Notes:
      - Stage2 requires `leiden_params` + `resolution_grid` and uses scan_root_leiden_cpm.
    """

    from cnvsim.analysis.leiden import pick_resolution_from_summary, scan_root_leiden_cpm  # noqa
    from cnvsim.analysis.leiden import LeidenParams

    if stage2.enabled:
        if leiden_params is None or resolution_grid is None:
            raise ValueError("Stage2 enabled but leiden_params or resolution_grid missing")
        if not isinstance(leiden_params, LeidenParams):
            raise TypeError("leiden_params must be a cnvsim.analysis.leiden.LeidenParams")

    grid_rows: list[dict] = []
    best_rows: list[dict] = []

    roots = [str(r) for r in roots]

    for i_root, root in enumerate(roots, start=1):
        stage1_root = (stage1_by_root or {}).get(str(root), stage1)
        print(f"[knn_parameter] [{i_root}/{len(roots)}] root={root}", flush=True)
        ir_root = Path(ir_dir) / "perroot" / str(root)
        coords, meta = _load_ir_coords_meta(ir_root_dir=ir_root)

        if platform_col not in meta.columns:
            # cannot score platform mixing; skip
            continue

        ids = meta["analysis_id"].astype(str).to_numpy(dtype=object)
        coords = coords.reset_index(drop=True)
        meta = meta.reset_index(drop=True)

        cols = dimension_columns(coords)
        if not cols:
            continue

        plat = meta[platform_col].fillna("NA").astype(str).to_numpy(dtype=object)
        plat_base = _random_match_rate(plat)

        morph = None
        morph_base = None
        if morph_col and morph_col in meta.columns:
            morph = meta[morph_col].fillna("NA").astype(str).to_numpy(dtype=object)

        # baseline morph purity for weak constraint: use a reference setting (grid median-ish)
        ref_metric = grid.metrics[0]
        ref_n_pcs = int(np.median(np.array(grid.n_pcs_list, dtype=int)))
        ref_k = int(np.median(np.array(grid.k_list, dtype=int)))
        used_ref = cols[: min(ref_n_pcs, len(cols))]
        idx_ref, _d_ref = knn_indices_and_distances(Z=coords[used_ref].to_numpy(np.float32), k=ref_k, metric=ref_metric)
        if morph is not None:
            morph_base = _neighbor_purity(idx=idx_ref, labels=morph)

        # Stage1 scan
        stage1_rows = []
        for metric in grid.metrics:
            for n_pcs in grid.n_pcs_list:
                used_cols = cols[: min(int(n_pcs), len(cols))]
                Z = coords[used_cols].to_numpy(dtype=np.float32)
                for K in grid.k_list:
                    idx, dists = knn_indices_and_distances(Z=Z, k=int(K), metric=str(metric))
                    if idx.size == 0:
                        continue

                    n_comps, giant_frac = _graph_health_from_knn(n=Z.shape[0], idx=idx)
                    plat_pur = _neighbor_purity(idx=idx, labels=plat)
                    plat_enr = float(plat_pur - plat_base) if np.isfinite(plat_pur) else np.nan

                    morph_pur = np.nan
                    morph_drop = np.nan
                    if morph is not None and morph_base is not None and np.isfinite(morph_base):
                        morph_pur = _neighbor_purity(idx=idx, labels=morph)
                        morph_drop = float(morph_pur - morph_base) if np.isfinite(morph_pur) else np.nan

                    row = {
                        "scan_stage": "stage1_graph",
                        "root": root,
                        "metric": str(metric),
                        "n_pcs": int(len(used_cols)),
                        "K": int(K),
                        "n_samples": int(Z.shape[0]),
                        "n_components": int(n_comps),
                        "giant_frac": float(giant_frac),
                        "platform_neighbor_purity": float(plat_pur),
                        "platform_match_rate": float(plat_base),
                        "platform_enrichment": float(plat_enr),
                        "morph_neighbor_purity": float(morph_pur) if morph_pur == morph_pur else np.nan,
                        "morph_delta_vs_ref": float(morph_drop) if morph_drop == morph_drop else np.nan,
                    }
                    grid_rows.append(row)
                    stage1_rows.append(row)

        df1 = pd.DataFrame(stage1_rows)
        if df1.empty:
            continue

        # Stage1 constraints
        ok = (df1["giant_frac"] >= float(stage1_root.min_giant_frac)) & (df1["n_components"] <= int(stage1_root.max_components))
        if (
            stage1_root.max_morph_purity_drop is not None
            and morph is not None
            and morph_base is not None
            and np.isfinite(morph_base)
        ):
            ok = ok & (df1["morph_delta_vs_ref"].isna() | (df1["morph_delta_vs_ref"] >= -float(stage1_root.max_morph_purity_drop)))
        if stage1_root.max_platform_enrichment is not None:
            ok = ok & (df1["platform_enrichment"] <= float(stage1_root.max_platform_enrichment))

        cand = df1[ok].copy()
        if cand.empty:
            continue

        # Rank within an acceptable platform-enrichment band rather than
        # always choosing the absolute minimum. This avoids over-correcting
        # platform mixing at the cost of an unstable or biologically implausible
        # graph.
        cand = cand.copy()
        platform_min = float(cand["platform_enrichment"].min()) if not cand.empty else np.nan
        platform_tol = stage1_root.platform_enrichment_tolerance
        if platform_tol is not None and np.isfinite(platform_min):
            band = cand[cand["platform_enrichment"] <= platform_min + float(platform_tol)].copy()
            if not band.empty:
                cand = band

        metric_order = {str(v).lower(): i for i, v in enumerate(stage1_root.preferred_metric_order)}
        k_order = {int(v): i for i, v in enumerate(stage1_root.preferred_k_order)}
        preferred_n_pcs = {int(v) for v in stage1_root.preferred_n_pcs}
        cand["_metric_rank"] = cand["metric"].astype(str).str.lower().map(metric_order).fillna(len(metric_order)).astype(int)
        cand["_k_rank"] = cand["K"].astype(int).map(k_order).fillna(len(k_order)).astype(int)
        cand["_n_pcs_not_preferred"] = (~cand["n_pcs"].astype(int).isin(preferred_n_pcs)).astype(int)
        cand["stage1_platform_enrichment_min"] = platform_min
        cand["stage1_platform_enrichment_tolerance"] = np.nan if platform_tol is None else float(platform_tol)
        cand["stage1_rank_policy"] = "platform_band_conservative_preference"

        cand = cand.sort_values(
            [
                "_metric_rank",
                "_k_rank",
                "_n_pcs_not_preferred",
                "platform_enrichment",
                "morph_delta_vs_ref",
                "giant_frac",
                "n_components",
                "n_pcs",
            ],
            ascending=[True, True, True, True, False, False, True, False],
        )
        top = cand.head(int(stage2.top_n) if stage2.enabled else 1).copy()

        # Stage2: Leiden stability on top candidates
        chosen = top.iloc[0].to_dict()
        if stage2.enabled:
            best_score = -1.0
            best_key = None
            best_row = None

            for _i, r in top.iterrows():
                metric = str(r["metric"])
                n_pcs = int(r["n_pcs"])
                K = int(r["K"])

                used_cols = cols[: min(n_pcs, len(cols))]
                Z = coords[used_cols].to_numpy(dtype=np.float32)
                idx, dists = knn_indices_and_distances(Z=Z, k=K, metric=metric)
                if idx.size == 0:
                    continue

                g, _info = build_igraph_from_knn(
                    ids=ids,
                    neigh_idx=idx,
                    neigh_dists=dists,
                    sigma_quantile=float(stage2.sigma_quantile),
                    make_undirected=bool(stage2.make_undirected),
                    combine_mode=str(stage2.combine_mode),
                )

                runs_df, summary_df = scan_root_leiden_cpm(
                    root=str(root),
                    g=g,
                    resolution_grid=np.asarray(resolution_grid, dtype=float),
                    params=leiden_params,
                    fine_grid_points=int(fine_grid_points),
                    platform_labels=plat,
                )

                for scan_row in summary_df.to_dict(orient="records"):
                    grid_rows.append(
                        {
                            **scan_row,
                            "scan_stage": "stage2_resolution",
                            "metric": metric,
                            "n_pcs": n_pcs,
                            "K": K,
                        }
                    )

                try:
                    selected_resolution = float(
                        pick_resolution_from_summary(
                            summary_df,
                            prefer_grid="fine",
                            params=leiden_params,
                        )
                    )
                except Exception:
                    continue

                summ = summary_df.copy()
                summ["_resolution_dist"] = (summ["resolution"].astype(float) - selected_resolution).abs()
                selected_summary = summ.sort_values("_resolution_dist").iloc[0]
                ari_best = float(selected_summary["ari_mean_pairwise"])
                plat_v = (
                    float(selected_summary["platform_assoc_cramersV"])
                    if "platform_assoc_cramersV" in selected_summary.index
                    else float("nan")
                )
                platform_weight = float(getattr(leiden_params, "platform_assoc_weight", 0.0) or 0.0)
                selection_policy = str(
                    getattr(leiden_params, "selection_policy", "score_band_center")
                ).strip().lower()
                neighbor_min = float(selected_summary.get("resolution_neighbor_ari_min", np.nan))
                neighbor_mean = float(selected_summary.get("resolution_neighbor_ari_mean", np.nan))
                neighbor_max = float(selected_summary.get("resolution_neighbor_ari_max", np.nan))
                plateau_qualified = bool(
                    np.isfinite(neighbor_max)
                    and neighbor_max >= float(leiden_params.min_resolution_neighbor_ari)
                )
                score = (
                    ari_best
                    if selection_policy == "stable_plateau"
                    else ari_best - (platform_weight * plat_v if np.isfinite(plat_v) else 0.0)
                )

                # Instead of updating grid_rows in place (slow), append stage2 record row.
                grid_rows.append({
                    "scan_stage": "stage2_selected",
                    "root": root,
                    "metric": metric,
                    "n_pcs": n_pcs,
                    "K": K,
                    "stage2_ari_best": ari_best,
                    "stage2_platform_assoc": plat_v,
                    "stage2_score": score,
                    "stage2_selection_policy": selection_policy,
                    "stage2_resolution_neighbor_ari_min": neighbor_min,
                    "stage2_resolution_neighbor_ari_mean": neighbor_mean,
                    "stage2_resolution_neighbor_ari_max": neighbor_max,
                    "stage2_plateau_qualified": plateau_qualified,
                    "stage2_selected_resolution": selected_resolution,
                    "stage2_n_clusters_median": float(selected_summary.get("n_clusters_median", np.nan)),
                    "stage2_giant_cluster_frac_median": float(selected_summary.get("giant_cluster_frac_median", np.nan)),
                    "stage2_frac_small_median": float(selected_summary.get("frac_nodes_in_small_clusters_median", np.nan)),
                })

                if selection_policy == "stable_plateau":
                    candidate_key = (
                        int(plateau_qualified),
                        neighbor_mean if np.isfinite(neighbor_mean) else -np.inf,
                        ari_best if np.isfinite(ari_best) else -np.inf,
                        -plat_v if np.isfinite(plat_v) else -np.inf,
                    )
                    choose_candidate = best_key is None or candidate_key > best_key
                else:
                    candidate_key = None
                    choose_candidate = score > best_score

                if choose_candidate:
                    best_score = score
                    best_key = candidate_key
                    best_row = {
                        **r.to_dict(),
                        "stage2_ari_best": ari_best,
                        "stage2_platform_assoc": plat_v,
                        "stage2_score": score,
                        "stage2_selection_policy": selection_policy,
                        "stage2_resolution_neighbor_ari_min": neighbor_min,
                        "stage2_resolution_neighbor_ari_mean": neighbor_mean,
                        "stage2_resolution_neighbor_ari_max": neighbor_max,
                        "stage2_plateau_qualified": plateau_qualified,
                        "stage2_selected_resolution": selected_resolution,
                        "stage2_n_clusters_median": float(selected_summary.get("n_clusters_median", np.nan)),
                        "stage2_giant_cluster_frac_median": float(selected_summary.get("giant_cluster_frac_median", np.nan)),
                        "stage2_frac_small_median": float(selected_summary.get("frac_nodes_in_small_clusters_median", np.nan)),
                    }

            if best_row is not None:
                chosen = best_row

        best_rows.append({
            "root": root,
            "metric": chosen["metric"],
            "n_pcs": int(chosen["n_pcs"]),
            "K": int(chosen["K"]),
            "platform_enrichment": float(chosen.get("platform_enrichment", np.nan)),
            "stage1_platform_enrichment_min": float(chosen.get("stage1_platform_enrichment_min", np.nan)),
            "stage1_platform_enrichment_tolerance": float(chosen.get("stage1_platform_enrichment_tolerance", np.nan)),
            "stage1_rank_policy": str(chosen.get("stage1_rank_policy", "")),
            "n_components": int(chosen.get("n_components", -1)) if chosen.get("n_components") == chosen.get("n_components") else -1,
            "giant_frac": float(chosen.get("giant_frac", np.nan)),
            "morph_delta_vs_ref": float(chosen.get("morph_delta_vs_ref", np.nan)),
            "stage2_ari_best": float(chosen.get("stage2_ari_best", np.nan)),
            "stage2_platform_assoc": float(chosen.get("stage2_platform_assoc", np.nan)),
            "stage2_score": float(chosen.get("stage2_score", np.nan)),
            "stage2_selection_policy": str(chosen.get("stage2_selection_policy", "")),
            "stage2_resolution_neighbor_ari_min": float(chosen.get("stage2_resolution_neighbor_ari_min", np.nan)),
            "stage2_resolution_neighbor_ari_mean": float(chosen.get("stage2_resolution_neighbor_ari_mean", np.nan)),
            "stage2_resolution_neighbor_ari_max": float(chosen.get("stage2_resolution_neighbor_ari_max", np.nan)),
            "stage2_plateau_qualified": bool(chosen.get("stage2_plateau_qualified", False)),
            "stage2_selected_resolution": float(chosen.get("stage2_selected_resolution", np.nan)),
            "stage2_n_clusters_median": float(chosen.get("stage2_n_clusters_median", np.nan)),
            "stage2_giant_cluster_frac_median": float(chosen.get("stage2_giant_cluster_frac_median", np.nan)),
            "stage2_frac_small_median": float(chosen.get("stage2_frac_small_median", np.nan)),
        })

    best_df = pd.DataFrame(best_rows).sort_values("root")
    grid_df = pd.DataFrame(grid_rows)
    return best_df, grid_df
