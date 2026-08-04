from __future__ import annotations

import numpy as np
import pandas as pd

from cnvsim.analysis.config import LeidenParams


def run_leiden_cpm(g, *, resolution: float, seed: int, weight_attr: str = "weight") -> tuple[np.ndarray, dict]:
    """Leiden CPM wrapper."""
    import leidenalg

    part = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights=g.es[weight_attr] if (weight_attr in g.es.attributes()) else None,
        resolution_parameter=float(resolution),
        seed=int(seed),
    )

    membership = np.asarray(part.membership, dtype=np.int32)
    stats = {
        "n_clusters": int(len(part)),
        "quality": float(part.quality()),
    }
    return membership, stats


def summarize_cluster_sizes(membership: np.ndarray, *, min_cluster_size: int) -> dict:
    vc = pd.Series(membership).value_counts()
    sizes = vc.values.astype(int)

    n = int(sizes.sum())
    min_size = int(sizes.min()) if len(sizes) else 0
    med_size = float(np.median(sizes)) if len(sizes) else np.nan
    max_size = int(sizes.max()) if len(sizes) else 0

    p10 = float(np.quantile(sizes, 0.10)) if len(sizes) else np.nan
    p25 = float(np.quantile(sizes, 0.25)) if len(sizes) else np.nan

    small_mask = sizes < int(min_cluster_size)
    frac_nodes_in_small = float(sizes[small_mask].sum() / n) if n > 0 else np.nan
    n_small_clusters = int(small_mask.sum())

    max_frac = float(max_size / n) if n > 0 else np.nan

    return {
        "min_size": min_size,
        "median_size": med_size,
        "max_size": max_size,
        "p10_size": p10,
        "p25_size": p25,
        "n_small_clusters": n_small_clusters,
        "frac_nodes_in_small_clusters": frac_nodes_in_small,
        "max_cluster_frac": max_frac,
    }


def pairwise_ari(memberships: list[np.ndarray]) -> float:
    """mean ARI across all pairs at a fixed resolution."""
    from sklearn.metrics import adjusted_rand_score

    if len(memberships) < 2:
        return float("nan")
    aris: list[float] = []
    for i in range(len(memberships)):
        for j in range(i + 1, len(memberships)):
            aris.append(float(adjusted_rand_score(memberships[i], memberships[j])))
    return float(np.mean(aris)) if aris else float("nan")


def _representative_membership(
    runs_at_resolution: pd.DataFrame,
    memberships: list[np.ndarray],
    seeds: list[int],
) -> np.ndarray:
    """Choose one seed partition for cross-resolution comparisons."""

    ranked = runs_at_resolution.sort_values(
        ["frac_nodes_in_small_clusters", "quality"],
        ascending=[True, False],
    )
    strict = ranked[ranked["frac_nodes_in_small_clusters"] == 0.0]
    if not strict.empty:
        ranked = strict.sort_values("quality", ascending=False)
    best_seed = int(ranked.iloc[0]["seed"])
    return memberships[seeds.index(best_seed)]


def _annotate_resolution_neighbour_stability(
    summary_df: pd.DataFrame,
    representative_by_resolution: dict[float, np.ndarray],
) -> pd.DataFrame:
    """Add ARI to adjacent resolution partitions on the same scan grid."""

    from sklearn.metrics import adjusted_rand_score

    out = summary_df.sort_values("resolution").reset_index(drop=True).copy()
    resolutions = out["resolution"].astype(float).tolist()
    previous_ari: list[float] = []
    next_ari: list[float] = []
    for index, resolution in enumerate(resolutions):
        current = representative_by_resolution[float(resolution)]
        previous_ari.append(
            float(adjusted_rand_score(current, representative_by_resolution[float(resolutions[index - 1])]))
            if index > 0
            else float("nan")
        )
        next_ari.append(
            float(adjusted_rand_score(current, representative_by_resolution[float(resolutions[index + 1])]))
            if index + 1 < len(resolutions)
            else float("nan")
        )

    out["resolution_previous_ari"] = previous_ari
    out["resolution_next_ari"] = next_ari
    neighbour_values = out[["resolution_previous_ari", "resolution_next_ari"]]
    out["resolution_neighbor_ari_min"] = neighbour_values.min(axis=1, skipna=True)
    out["resolution_neighbor_ari_mean"] = neighbour_values.mean(axis=1, skipna=True)
    out["resolution_neighbor_ari_max"] = neighbour_values.max(axis=1, skipna=True)
    out["resolution_neighbor_n"] = neighbour_values.notna().sum(axis=1).astype(int)
    return out


def _pick_stable_plateau_resolution(candidates: pd.DataFrame, params: LeidenParams) -> float | None:
    """Return the centre of the strongest contiguous stable-resolution plateau."""

    required = {"_grid_rank", "resolution", "resolution_previous_ari"}
    if not required.issubset(candidates.columns):
        return None

    threshold = float(params.min_resolution_neighbor_ari)
    ordered = candidates.sort_values("resolution").copy()
    previous_rank = ordered["_grid_rank"].shift(1)
    adjacent_rank = ordered["_grid_rank"] == previous_rank + 1
    stable_with_previous = adjacent_rank & (
        pd.to_numeric(ordered["resolution_previous_ari"], errors="coerce") >= threshold
    )
    ordered["_plateau_id"] = (~stable_with_previous).cumsum()

    plateau_rows: list[dict[str, float | int]] = []
    for plateau_id, plateau in ordered.groupby("_plateau_id", sort=False):
        n_points = int(len(plateau))
        if n_points < int(params.min_plateau_points):
            continue
        resolutions = pd.to_numeric(plateau["resolution"], errors="raise").to_numpy(dtype=float)
        platform = pd.to_numeric(plateau.get("platform_assoc_cramersV"), errors="coerce")
        plateau_rows.append(
            {
                "plateau_id": int(plateau_id),
                "n_points": n_points,
                "log_width": float(np.log(resolutions.max()) - np.log(resolutions.min())),
                "neighbor_ari_min": float(
                    pd.to_numeric(plateau["resolution_neighbor_ari_min"], errors="coerce").min()
                ),
                "seed_ari_median": float(
                    pd.to_numeric(plateau["ari_mean_pairwise"], errors="coerce").median()
                ),
                "platform_assoc_median": float(platform.median()) if platform.notna().any() else float("inf"),
                "giant_fraction_median": float(
                    pd.to_numeric(plateau["giant_cluster_frac_median"], errors="coerce").median()
                ),
            }
        )

    if not plateau_rows:
        return None

    plateau_summary = pd.DataFrame(plateau_rows).sort_values(
        [
            "n_points",
            "log_width",
            "neighbor_ari_min",
            "seed_ari_median",
            "platform_assoc_median",
            "giant_fraction_median",
        ],
        ascending=[False, False, False, False, True, True],
    )
    selected_id = int(plateau_summary.iloc[0]["plateau_id"])
    selected = ordered[ordered["_plateau_id"] == selected_id].copy()
    log_resolution = np.log(pd.to_numeric(selected["resolution"], errors="raise").to_numpy(dtype=float))
    log_centre = float(np.median(log_resolution))
    selected["_distance_to_log_centre"] = np.abs(log_resolution - log_centre)
    selected = selected.sort_values(
        ["_distance_to_log_centre", "resolution_neighbor_ari_min", "ari_mean_pairwise"],
        ascending=[True, False, False],
    )
    return float(selected.iloc[0]["resolution"])


def make_resolution_grids(*, r_min: float = 1e-4, r_max: float = 5e-2, n_broad: int = 15, n_fine: int = 12) -> tuple[np.ndarray, int]:
    """coarse grid (logspace) + fine grid point count."""
    coarse = np.logspace(np.log10(float(r_min)), np.log10(float(r_max)), int(n_broad))
    return coarse, int(n_fine)


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Cramér's V for two categorical vectors. Returns 0..1 (lower = less associated)."""

    try:
        from scipy.stats import chi2_contingency

        x = pd.Series(x).astype(str)
        y = pd.Series(y).astype(str)
        tab = pd.crosstab(x, y)
        if tab.size == 0:
            return float("nan")
        chi2, _p, _dof, _exp = chi2_contingency(tab.values)
        n = tab.values.sum()
        if n <= 0:
            return float("nan")
        r, k = tab.shape
        phi2 = chi2 / n
        # bias correction (Bergsma 2013 style)
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1.0, (n - 1)))
        rcorr = r - ((r - 1) ** 2) / max(1.0, (n - 1))
        kcorr = k - ((k - 1) ** 2) / max(1.0, (n - 1))
        denom = max(1e-12, min((kcorr - 1), (rcorr - 1)))
        return float(np.sqrt(phi2corr / denom))
    except Exception:
        return float("nan")


def scan_root_leiden_cpm(
    *,
    root: str,
    g,
    resolution_grid: np.ndarray,
    params: LeidenParams,
    fine_grid_points: int = 12,
    platform_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """broad scan + fine scan around r_star."""

    MAX_FRAC_IN_SMALL = float(params.max_frac_in_small)
    MAX_GIANT_FRAC = float(params.max_giant_frac)

    n = int(g.vcount())
    if n == 0:
        raise ValueError(f"[{root}] Graph has 0 vertices.")

    resolution_grid = np.asarray([float(x) for x in resolution_grid], dtype=float)
    if np.any(resolution_grid <= 0):
        raise ValueError(f"[{root}] resolution_grid must be strictly > 0 for log scanning.")
    resolution_grid = np.asarray(sorted(resolution_grid), dtype=float)

    seed_list = list(params.seeds)

    runs = []
    mem_by_r: dict[float, list[np.ndarray]] = {}

    # broad scan
    for r in resolution_grid:
        r = float(r)
        mem_by_r[r] = []
        for seed in seed_list:
            mem, st = run_leiden_cpm(g, resolution=r, seed=int(seed))
            mem_by_r[r].append(mem)

            size_stats = summarize_cluster_sizes(mem, min_cluster_size=int(params.min_cluster_size))
            runs.append({
                "root": root,
                "resolution": r,
                "seed": int(seed),
                "n_clusters": int(st["n_clusters"]),
                "quality": float(st["quality"]),
                **size_stats,
                "grid": "broad",
            })

    runs_df = pd.DataFrame(runs)

    # summarize broad per resolution
    summary = []
    representative_by_r: dict[float, np.ndarray] = {}
    for r, df_r in runs_df.groupby("resolution", sort=True):
        mems = mem_by_r[float(r)]
        ari = float(pairwise_ari(mems))

        representative = _representative_membership(df_r, mems, seed_list)
        representative_by_r[float(r)] = representative

        # platform association (A1 confounding proxy): compute on the chosen seed at this resolution
        platform_assoc = float("nan")
        if platform_labels is not None:
            try:
                platform_assoc = _cramers_v(representative, platform_labels)
            except Exception:
                platform_assoc = float("nan")

        n_clusters_med = float(df_r["n_clusters"].median())
        quality_med = float(df_r["quality"].median())

        frac_small_med = float(df_r["frac_nodes_in_small_clusters"].median())
        giant_frac_med = float(df_r["max_cluster_frac"].median())

        too_tiny_mass = frac_small_med > MAX_FRAC_IN_SMALL
        too_giant = giant_frac_med > MAX_GIANT_FRAC
        summary.append({
            "root": root,
            "resolution": float(r),
            "ari_mean_pairwise": ari,
            "platform_assoc_cramersV": platform_assoc,
            "n_clusters_median": n_clusters_med,
            "quality_median": quality_med,
            "frac_nodes_in_small_clusters_median": frac_small_med,
            "giant_cluster_frac_median": giant_frac_med,
            "too_tiny": bool(too_tiny_mass),
            "too_giant": bool(too_giant),
            "grid": "broad",
        })

    summary_df = _annotate_resolution_neighbour_stability(
        pd.DataFrame(summary),
        representative_by_r,
    )

    # choose r_star from broad
    cand = summary_df[
        (summary_df["n_clusters_median"] >= 2)
        & np.isfinite(summary_df["ari_mean_pairwise"])
        & (~summary_df["too_tiny"])
        & (~summary_df["too_giant"])
    ].copy()

    if len(cand) == 0:
        summary_df["grid"] = pd.Categorical(summary_df["grid"], categories=["broad", "fine"], ordered=True)
        return runs_df, summary_df

    # Use the same robust selection rule for locating the broad scan region and
    # for selecting the final fine-grid resolution.
    r_star = pick_resolution_from_summary(summary_df, prefer_grid="broad", params=params)

    # fine scan around r_star
    log_grid = np.log(resolution_grid)
    idx0 = int(np.argmin(np.abs(log_grid - np.log(r_star))))
    lo = max(0, idx0 - 1)
    hi = min(len(resolution_grid) - 1, idx0 + 1)

    r_lo = float(resolution_grid[lo])
    r_hi = float(resolution_grid[hi])
    fine_grid = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), int(fine_grid_points)))

    fine_runs = []
    fine_mem_by_r: dict[float, list[np.ndarray]] = {}

    for r in fine_grid:
        r = float(r)
        fine_mem_by_r[r] = []
        for seed in seed_list:
            mem, st = run_leiden_cpm(g, resolution=r, seed=int(seed))
            fine_mem_by_r[r].append(mem)

            size_stats = summarize_cluster_sizes(mem, min_cluster_size=int(params.min_cluster_size))
            fine_runs.append({
                "root": root,
                "resolution": r,
                "seed": int(seed),
                "n_clusters": int(st["n_clusters"]),
                "quality": float(st["quality"]),
                **size_stats,
                "grid": "fine",
            })

    fine_runs_df = pd.DataFrame(fine_runs)
    runs_df = pd.concat([runs_df, fine_runs_df], ignore_index=True)

    fine_summary = []
    fine_representative_by_r: dict[float, np.ndarray] = {}
    for r, df_r in fine_runs_df.groupby("resolution", sort=True):
        mems = fine_mem_by_r[float(r)]
        ari = float(pairwise_ari(mems))

        representative = _representative_membership(df_r, mems, seed_list)
        fine_representative_by_r[float(r)] = representative

        platform_assoc = float("nan")
        if platform_labels is not None:
            try:
                platform_assoc = _cramers_v(representative, platform_labels)
            except Exception:
                platform_assoc = float("nan")

        n_clusters_med = float(df_r["n_clusters"].median())
        quality_med = float(df_r["quality"].median())

        frac_small_med = float(df_r["frac_nodes_in_small_clusters"].median())
        giant_frac_med = float(df_r["max_cluster_frac"].median())

        too_tiny_mass = frac_small_med > MAX_FRAC_IN_SMALL
        too_giant = giant_frac_med > MAX_GIANT_FRAC
        fine_summary.append({
            "root": root,
            "resolution": float(r),
            "ari_mean_pairwise": ari,
            "platform_assoc_cramersV": platform_assoc,
            "n_clusters_median": n_clusters_med,
            "quality_median": quality_med,
            "frac_nodes_in_small_clusters_median": frac_small_med,
            "giant_cluster_frac_median": giant_frac_med,
            "too_tiny": bool(too_tiny_mass),
            "too_giant": bool(too_giant),
            "grid": "fine",
        })

    fine_summary_df = _annotate_resolution_neighbour_stability(
        pd.DataFrame(fine_summary),
        fine_representative_by_r,
    )
    summary_df = pd.concat([summary_df, fine_summary_df], ignore_index=True)

    summary_df["grid"] = pd.Categorical(summary_df["grid"], categories=["broad", "fine"], ordered=True)
    summary_df = summary_df.sort_values(["grid", "resolution"]).reset_index(drop=True)

    return runs_df, summary_df


def pick_resolution_from_summary(
    summary_df: pd.DataFrame,
    *,
    prefer_grid: str = "fine",
    params: LeidenParams,
) -> float:
    """Select a robust resolution from feasible, near-optimal candidates."""

    df = summary_df.copy()
    if "grid" in df.columns and prefer_grid is not None:
        df_pref = df[df["grid"] == prefer_grid].copy()
        if len(df_pref) > 0:
            df = df_pref
    df = df.sort_values("resolution").reset_index(drop=True)
    df["_grid_rank"] = np.arange(len(df), dtype=int)

    df2 = df[df["n_clusters_median"] >= 2].copy()
    if len(df2) == 0:
        df_any = summary_df[summary_df["n_clusters_median"] >= 2].copy()
        if len(df_any) == 0:
            return float(summary_df.sort_values("resolution").iloc[0]["resolution"])
        df_any = df_any.sort_values(["ari_mean_pairwise", "quality_median"], ascending=[False, False])
        return float(df_any.iloc[0]["resolution"])

    cand = df2[
        (df2["frac_nodes_in_small_clusters_median"] <= float(params.max_frac_in_small))
        & (df2["giant_cluster_frac_median"] <= float(params.max_giant_frac))
        & (~df2.get("too_tiny", pd.Series(False, index=df2.index)))
        & (~df2.get("too_giant", pd.Series(False, index=df2.index)))
    ].copy()

    if len(cand) == 0:
        cand = df2[(df2["giant_cluster_frac_median"] <= float(params.max_giant_frac))].copy()
    if len(cand) == 0:
        cand = df2.copy()

    strong = cand[cand["ari_mean_pairwise"] >= float(params.min_ari)].copy()
    if len(strong) > 0:
        cand = strong

    # Prefer low platform association if requested
    w = float(getattr(params, "platform_assoc_weight", 0.0) or 0.0)
    vmax = getattr(params, "platform_assoc_max", None)
    if vmax is not None and "platform_assoc_cramersV" in cand.columns:
        okv = cand[np.isfinite(cand["platform_assoc_cramersV"]) & (cand["platform_assoc_cramersV"] <= float(vmax))].copy()
        if len(okv) > 0:
            cand = okv

    policy = str(getattr(params, "selection_policy", "score_band_center")).strip().lower()
    if policy == "stable_plateau":
        plateau_resolution = _pick_stable_plateau_resolution(cand, params)
        if plateau_resolution is not None:
            return float(plateau_resolution)
    elif policy != "score_band_center":
        raise ValueError(f"Unknown Leiden resolution selection policy: {params.selection_policy}")

    # Score combines seed stability and platform association. Small-cluster
    # mass is already constrained above and is retained as a final tie-breaker.
    cand = cand.copy()
    if "platform_assoc_cramersV" in cand.columns and w > 0:
        platform_assoc = pd.to_numeric(cand["platform_assoc_cramersV"], errors="coerce")
        finite_platform = platform_assoc[np.isfinite(platform_assoc)]
        fill_value = float(finite_platform.median()) if not finite_platform.empty else 0.0
        cand["_score"] = cand["ari_mean_pairwise"] - w * platform_assoc.fillna(fill_value)
    else:
        cand["_score"] = cand["ari_mean_pairwise"]

    finite_scores = pd.to_numeric(cand["_score"], errors="coerce")
    if finite_scores.notna().any():
        best_score = float(finite_scores.max())
        tolerance = max(0.0, float(params.selection_score_tolerance))
        near_optimal = cand[finite_scores >= best_score - tolerance].copy()
        if not near_optimal.empty:
            cand = near_optimal

    # Near-equivalent candidates may occupy different resolution regimes.
    # Selecting the log-scale centre is less sensitive to a single noisy score
    # maximum than selecting the numerically highest-scoring point.
    log_resolution = np.log(pd.to_numeric(cand["resolution"], errors="raise").to_numpy(dtype=float))
    log_centre = float(np.median(log_resolution))
    cand["_distance_to_log_centre"] = np.abs(log_resolution - log_centre)
    sort_cols = ["_distance_to_log_centre", "_score"]
    ascending = [True, False]
    if "frac_nodes_in_small_clusters_median" in cand.columns:
        sort_cols.append("frac_nodes_in_small_clusters_median")
        ascending.append(True)
    if "quality_median" in cand.columns:
        sort_cols.append("quality_median")
        ascending.append(False)
    cand = cand.sort_values(sort_cols, ascending=ascending)

    return float(cand.iloc[0]["resolution"])


def pick_best_seed_at_resolution(*, g, resolution: float, params: LeidenParams) -> dict:
    """Notebook-exact: prefer seeds with 0 tiny mass, else minimize tiny mass then max quality."""

    runs = []
    mems = {}
    for seed in params.seeds:
        mem, st = run_leiden_cpm(g, resolution=float(resolution), seed=int(seed))
        mems[int(seed)] = mem
        size_stats = summarize_cluster_sizes(mem, min_cluster_size=int(params.min_cluster_size))
        runs.append({
            "seed": int(seed),
            "n_clusters": int(st["n_clusters"]),
            "quality": float(st["quality"]),
            **size_stats,
        })

    runs_df = pd.DataFrame(runs)

    strict = runs_df[runs_df["frac_nodes_in_small_clusters"] == 0.0].copy()
    if len(strict) > 0:
        cand = strict.sort_values(["quality"], ascending=False)
        reason = "strict_no_small_clusters"
    else:
        cand = runs_df.sort_values(["frac_nodes_in_small_clusters", "quality"], ascending=[True, False])
        reason = "min_small_cluster_mass_then_max_quality"

    best = cand.iloc[0].to_dict()
    best_seed = int(best["seed"])

    return {
        "best_seed": best_seed,
        "membership": mems[best_seed],
        "stats": {"n_clusters": int(best["n_clusters"]), "quality": float(best["quality"])},
        "size_stats": {k: best[k] for k in [
            "min_size",
            "median_size",
            "max_size",
            "p10_size",
            "p25_size",
            "n_small_clusters",
            "frac_nodes_in_small_clusters",
            "max_cluster_frac",
        ] if k in best},
        "seed_choice_reason": reason,
        "runs_df": runs_df,
    }


def run_final_partition_best_seed(*, g, resolution: float, params: LeidenParams, id_attr: str = "analysis_id") -> tuple[pd.DataFrame, dict]:
    pick = pick_best_seed_at_resolution(g=g, resolution=float(resolution), params=params)
    ids = np.asarray(g.vs[id_attr], dtype=str)
    out = pd.DataFrame({"analysis_id": ids, "cluster": pick["membership"].astype(int)})
    stats = {
        "resolution": float(resolution),
        "best_seed": int(pick["best_seed"]),
        "seed_choice_reason": str(pick["seed_choice_reason"]),
        **pick["stats"],
        **pick["size_stats"],
    }
    return out, stats


def mark_tiny_clusters_as_outliers(
    clusters_df: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    min_cluster_size: int = 50,
    outlier_label: int = -1,
) -> pd.DataFrame:
    df = clusters_df.copy()
    vc = df[cluster_col].value_counts()
    tiny = set(vc[vc < int(min_cluster_size)].index.tolist())
    df["cluster_raw"] = df[cluster_col]
    df[cluster_col] = df[cluster_col].where(~df["cluster_raw"].isin(tiny), int(outlier_label))
    return df
