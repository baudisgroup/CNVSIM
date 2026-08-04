from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from scipy.stats import mannwhitneyu



def safe_logscore(q: np.ndarray, p_floor: float = 1e-300) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q2 = np.clip(q, float(p_floor), 1.0)
    return -np.log10(q2)


def bh_fdr(pvals: np.ndarray, p_floor: float = 1e-300) -> np.ndarray:
    """
    BH-FDR wrapper using scipy.stats.false_discovery_control if available.
    Returns q-values aligned to input order. Non-finite pvals -> NaN.
    """

    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    msk = np.isfinite(p)
    if not bool(msk.any()):
        return q

    p2 = np.clip(p[msk], float(p_floor), 1.0)

    try:
        q_tmp = false_discovery_control(p2, method="bh")
        q[msk] = np.clip(q_tmp, 0.0, 1.0)
        return q
    except Exception:
        # fallback: manual BH
        m = int(p2.size)
        order = np.argsort(p2)
        ranked = p2[order]
        q_ranked = ranked * (m / (np.arange(m) + 1.0))
        # monotone
        q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
        q_ranked = np.clip(q_ranked, 0.0, 1.0)
        q_out = np.empty_like(q_ranked)
        q_out[order] = q_ranked
        q[msk] = q_out
        return q


def mwu_with_auc(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided"):
    """Returns (p_value, auc, U) where auc = U/(n_x*n_y)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan

    try:
        res = mannwhitneyu(x, y, alternative=str(alternative))
        p = float(res.pvalue)
        U = float(res.statistic)
        auc = U / (x.size * y.size)
        return p, auc, U
    except Exception:
        return np.nan, np.nan, np.nan


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a proportion."""

    if n <= 0:
        return np.nan, np.nan, np.nan

    p = k / n
    z = 1.959963984540054  # ~norm.ppf(1-alpha/2)

    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)

    return float(p), float(lo), float(hi)


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05):
    """Newcombe CI for difference of proportions p1-p2 using Wilson intervals."""

    p1, l1, u1 = wilson_ci(k1, n1, alpha=alpha)
    p2, l2, u2 = wilson_ci(k2, n2, alpha=alpha)

    diff = p1 - p2
    lo = l1 - u2
    hi = u1 - l2

    return float(diff), float(lo), float(hi)


def bootstrap_ci_delta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    func=np.mean,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Bootstrap CI for delta = func(x) - func(y)."""

    rng = np.random.default_rng(int(seed))

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan

    delta = float(func(x) - func(y))

    boots = []
    for _ in range(int(n_boot)):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        boots.append(func(xb) - func(yb))

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(delta), float(lo), float(hi)


def compute_marker_effect_stats(
    x_adj: np.ndarray,
    y_adj: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
):
    """Compute the core Track2 effect statistics for one marker comparison.

    Returns a dict aligned with the existing pooled marker_stats schema.
    If inputs are too small after finite filtering, returns None.
    """

    x_adj = np.asarray(x_adj, dtype=float)
    y_adj = np.asarray(y_adj, dtype=float)
    x_adj = x_adj[np.isfinite(x_adj)]
    y_adj = y_adj[np.isfinite(y_adj)]

    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    x_raw = x_raw[np.isfinite(x_raw)]
    y_raw = y_raw[np.isfinite(y_raw)]

    if x_adj.size < 2 or y_adj.size < 2 or x_raw.size < 2 or y_raw.size < 2:
        return None

    p, auc, _ = mwu_with_auc(x_adj, y_adj, alternative=alternative)
    mean_x = float(np.mean(x_adj))
    mean_y = float(np.mean(y_adj))
    delta = mean_x - mean_y

    k1 = int(np.sum(x_raw > 0))
    n1g = int(x_raw.size)
    k2 = int(np.sum(y_raw > 0))
    n2g = int(y_raw.size)

    prev1, prev1_lo, prev1_hi = wilson_ci(k1, n1g, alpha=alpha)
    prev2, prev2_lo, prev2_hi = wilson_ci(k2, n2g, alpha=alpha)
    dprev, dprev_lo, dprev_hi = newcombe_diff_ci(k1, n1g, k2, n2g, alpha=alpha)

    return {
        "adj_cluster_mean": mean_x,
        "adj_rest_mean": mean_y,
        "adj_delta": float(delta),
        "auc": float(auc) if auc == auc else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "raw_prev_cluster": float(prev1),
        "raw_prev_cluster_lo": float(prev1_lo),
        "raw_prev_cluster_hi": float(prev1_hi),
        "raw_prev_rest": float(prev2),
        "raw_prev_rest_lo": float(prev2_lo),
        "raw_prev_rest_hi": float(prev2_hi),
        "raw_prev_delta": float(dprev),
        "raw_prev_delta_lo": float(dprev_lo),
        "raw_prev_delta_hi": float(dprev_hi),
        "n_gene_cluster": n1g,
        "n_gene_rest": n2g,
    }


def summarize_platform_marker_stability(
    per_platform_df,
    pooled_row,
    *,
    min_abs_prevalence_delta: float = 0.05,
    min_abs_auc_shift: float = 0.10,
    min_platforms_supporting: int = 2,
    min_support_fraction: float = 2.0 / 3.0,
    max_auc_gap_allowed: float = 0.15,
):
    """Classify cross-platform support for one pooled Track 2 marker."""

    if per_platform_df is None or len(per_platform_df) == 0:
        return {
            "n_platforms_tested": 0,
            "n_platforms_supporting": 0,
            "support_fraction": np.nan,
            "direction_consistent": np.nan,
            "max_auc_gap": np.nan,
            "max_prev_delta_gap": np.nan,
            "single_platform_driven": np.nan,
            "platform_support_class": "not_evaluable",
            "passes_strict_support": False,
        }

    df = per_platform_df.copy()
    pooled_delta = float(pooled_row.get("adj_delta", np.nan))
    pooled_auc = float(pooled_row.get("auc", np.nan))
    pooled_sign = 0
    if np.isfinite(pooled_delta) and pooled_delta != 0:
        pooled_sign = 1 if pooled_delta > 0 else -1
    elif np.isfinite(pooled_auc) and pooled_auc != 0.5:
        pooled_sign = 1 if pooled_auc > 0.5 else -1

    def _supports(r) -> bool:
        delta = float(r.get("adj_delta", np.nan))
        auc = float(r.get("auc", np.nan))
        raw_delta = float(r.get("raw_prev_delta", np.nan))
        if not np.isfinite(delta) or not np.isfinite(auc) or pooled_sign == 0:
            return False
        sign_ok = (delta > 0 and pooled_sign > 0) or (delta < 0 and pooled_sign < 0)
        auc_ok = (auc >= 0.5 and pooled_sign > 0) or (auc <= 0.5 and pooled_sign < 0)
        effect_ok = (
            (
                np.isfinite(raw_delta)
                and abs(raw_delta) >= float(min_abs_prevalence_delta)
            )
            or abs(auc - 0.5) >= float(min_abs_auc_shift)
        )
        return bool(sign_ok and auc_ok and effect_ok)

    df["supports_pooled_direction"] = df.apply(_supports, axis=1)

    n_tested = int(len(df))
    n_support = int(df["supports_pooled_direction"].sum())
    support_fraction = float(n_support / n_tested) if n_tested > 0 else np.nan

    delta_signs = np.sign(pd.to_numeric(df["adj_delta"], errors="coerce").to_numpy(dtype=float))
    delta_signs = delta_signs[np.isfinite(delta_signs) & (delta_signs != 0)]
    direction_consistent = bool(len(np.unique(delta_signs)) == 1) if len(delta_signs) else np.nan

    auc_vals = pd.to_numeric(df["auc"], errors="coerce")
    prev_vals = pd.to_numeric(df["raw_prev_delta"], errors="coerce")
    max_auc_gap = float(auc_vals.max() - auc_vals.min()) if auc_vals.notna().any() else np.nan
    max_prev_gap = float(prev_vals.max() - prev_vals.min()) if prev_vals.notna().any() else np.nan

    single_platform_driven = bool(n_support <= 1) if n_tested > 0 else np.nan

    pooled_q = float(pooled_row.get("q_value", np.nan))
    pooled_raw_delta = float(pooled_row.get("raw_prev_delta", np.nan))
    pooled_effect_ok = (
        (
            np.isfinite(pooled_raw_delta)
            and abs(pooled_raw_delta) >= float(min_abs_prevalence_delta)
        )
        or (
            np.isfinite(pooled_auc)
            and abs(pooled_auc - 0.5) >= float(min_abs_auc_shift)
        )
    )
    strict = bool(
        np.isfinite(pooled_q)
        and pooled_q <= 0.05
        and pooled_effect_ok
        and n_support >= int(min_platforms_supporting)
        and bool(direction_consistent)
        and np.isfinite(support_fraction)
        and support_fraction >= float(min_support_fraction)
        and np.isfinite(max_auc_gap)
        and max_auc_gap <= float(max_auc_gap_allowed)
    )

    if strict:
        support_class = "cross_platform_supported"
    elif n_support >= int(min_platforms_supporting) and bool(direction_consistent):
        support_class = "cross_platform_partial"
    elif n_support >= 1:
        support_class = "limited_platform_support"
    else:
        support_class = "unsupported"

    return {
        "n_platforms_tested": n_tested,
        "n_platforms_supporting": n_support,
        "support_fraction": support_fraction,
        "direction_consistent": direction_consistent,
        "max_auc_gap": max_auc_gap,
        "max_prev_delta_gap": max_prev_gap,
        "single_platform_driven": single_platform_driven,
        "platform_support_class": support_class,
        "passes_strict_support": strict,
    }
