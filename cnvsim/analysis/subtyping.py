from __future__ import annotations

import numpy as np
import pandas as pd

from cnvsim.analysis.subtyping_stats import (
    bh_fdr,
    bootstrap_ci_delta,
    compute_marker_effect_stats,
    mwu_with_auc,
    newcombe_diff_ci,
    safe_logscore,
    summarize_platform_marker_stability,
    wilson_ci,
)

# Track 1 is computed from whole-genome segments in chromosome_architecture.py.
# This module computes the complementary four-channel gene-level evidence.

BACKGROUND_LABEL = -2


def _evidence_labels(
    clusters_df: pd.DataFrame,
    *,
    id_col: str,
    cluster_col: str,
    outlier_label: int,
    min_cluster_size: int,
    background_analysis_ids: set[str] | list[str] | tuple[str, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Return anchor labels plus optional background-only sample labels."""

    anchors = clusters_df[[id_col, cluster_col]].copy()
    anchors[id_col] = anchors[id_col].astype(str)
    anchors[cluster_col] = pd.to_numeric(anchors[cluster_col], errors="coerce")
    anchors = anchors[
        anchors[cluster_col].notna()
        & anchors[cluster_col].ne(int(outlier_label))
        & anchors[cluster_col].ge(0)
    ].copy()
    anchors[cluster_col] = anchors[cluster_col].astype(int)
    counts = anchors[cluster_col].value_counts()
    contexts = sorted(
        int(value) for value in counts[counts.ge(int(min_cluster_size))].index
    )
    anchors = anchors[anchors[cluster_col].isin(contexts)].copy()
    cluster_sizes = (
        counts.loc[contexts].rename_axis("cluster").reset_index(name="n")
        if contexts
        else pd.DataFrame({"cluster": [], "n": []})
    )

    if background_analysis_ids:
        anchor_ids = set(anchors[id_col])
        background_ids = sorted(
            {str(value) for value in background_analysis_ids} - anchor_ids
        )
        anchors = pd.concat(
            [
                anchors,
                pd.DataFrame(
                    {
                        id_col: background_ids,
                        cluster_col: [BACKGROUND_LABEL] * len(background_ids),
                    }
                ),
            ],
            ignore_index=True,
        )
    return anchors.drop_duplicates(id_col, keep="first"), cluster_sizes, contexts

def track2_gene_markers(
    *,
    gene_panel: pd.DataFrame,
    clusters_df: pd.DataFrame,
    event_cols: list[str],
    id_col: str = "analysis_id",
    cluster_col: str = "cluster",
    outlier_label: int = -1,
    min_cluster_size: int = 50,
    top_n: int = 30,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    min_nonzero: int | str = "auto",
    q_cut: float = 0.05,
    min_abs_delta_prev: float = 0.05,
    min_auc_enrich: float = 0.60,
    max_auc_deplete: float = 0.40,
    autosomes_only: bool = False,
    background_analysis_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Track 2: gene-level markers with sample-level burden-adjusted evidence.

    The adjustment subtracts each sample's mean value for the same event across
    the configured gene panel. It reduces global CNV-burden effects, but does
    not remove chromosome-level Track1 signal.
    """

    C, cluster_sizes, contexts = _evidence_labels(
        clusters_df,
        id_col=id_col,
        cluster_col=cluster_col,
        outlier_label=outlier_label,
        min_cluster_size=min_cluster_size,
        background_analysis_ids=background_analysis_ids,
    )

    if C.empty or not contexts:
        return {"cluster_sizes": cluster_sizes, "marker_stats": pd.DataFrame(), "signatures": pd.DataFrame()}

    G = gene_panel.merge(C, on=id_col, how="inner")
    if G.empty:
        return {"cluster_sizes": cluster_sizes, "marker_stats": pd.DataFrame(), "signatures": pd.DataFrame()}

    if bool(autosomes_only):
        if "chrom" not in G.columns:
            raise KeyError("track2_gene_markers autosomes_only=True requires gene_panel to contain 'chrom'")
        autosomes = {str(i) for i in range(1, 23)}
        G["chrom"] = G["chrom"].astype(str).str.replace("chr", "", regex=False)
        G = G[G["chrom"].isin(autosomes)].copy()
        if G.empty:
            return {"cluster_sizes": cluster_sizes, "marker_stats": pd.DataFrame(), "signatures": pd.DataFrame()}

    # ensure numeric
    for ev in event_cols:
        G[ev] = pd.to_numeric(G[ev], errors="coerce")

    if "gene_symbol" not in G.columns:
        raise KeyError("track2_gene_markers requires gene_panel to contain 'gene_symbol'")

    clusters = contexts

    # burdens per event (mean over genes)
    burdens = {}
    for ev in event_cols:
        b = G.groupby(id_col)[ev].mean()
        burdens[ev] = b.rename(f"burden_{ev}")

    B = pd.concat(burdens.values(), axis=1).reset_index()
    G2 = G.merge(B, on=id_col, how="left")

    for ev in event_cols:
        G2[f"{ev}_adj"] = G2[ev] - G2[f"burden_{ev}"]

    rows = []

    by_cluster_gene = {}
    for cl, dfc in G2.groupby(cluster_col, sort=False):
        by_cluster_gene[int(cl)] = {g: dfg for g, dfg in dfc.groupby("gene_symbol", sort=False)}

    for cl in clusters:
        cl = int(cl)
        dfc_map = by_cluster_gene.get(cl, {})
        df_rest = G2[G2[cluster_col] != cl]

        n1 = int(G2.loc[G2[cluster_col] == cl, id_col].nunique())
        n2 = int(df_rest[id_col].nunique())

        if min_nonzero == "auto":
            mnz = max(3, int(np.ceil(0.01 * min(n1, n2))))
        else:
            mnz = int(min_nonzero)

        rest_gene_map = {g: dfg for g, dfg in df_rest.groupby("gene_symbol", sort=False)}
        genes = sorted(set(dfc_map.keys()) & set(rest_gene_map.keys()))

        for gene in genes:
            df1 = dfc_map[gene]
            df2 = rest_gene_map[gene]

            for ev in event_cols:
                x = df1[f"{ev}_adj"].to_numpy(dtype=float)
                y = df2[f"{ev}_adj"].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                y = y[np.isfinite(y)]

                x_raw = df1[ev].to_numpy(dtype=float)
                y_raw = df2[ev].to_numpy(dtype=float)
                x_raw = x_raw[np.isfinite(x_raw)]
                y_raw = y_raw[np.isfinite(y_raw)]

                if (np.count_nonzero(x_raw) < mnz) and (np.count_nonzero(y_raw) < mnz):
                    continue
                if x.size < 2 or y.size < 2:
                    continue

                stats = compute_marker_effect_stats(
                    x,
                    y,
                    x_raw,
                    y_raw,
                    alpha=alpha,
                    alternative=alternative,
                )
                if stats is None:
                    continue

                rows.append({
                    "cluster": cl,
                    "event": ev,
                    "gene_symbol": gene,
                    **stats,
                    "n_cluster_samples": n1,
                    "n_rest_samples": n2,
                })

    marker_stats = pd.DataFrame(rows)
    if marker_stats.empty:
        return {"cluster_sizes": cluster_sizes, "marker_stats": marker_stats, "signatures": pd.DataFrame()}

    marker_stats["q_value"] = np.nan
    for (cl, ev), idx in marker_stats.groupby(["cluster", "event"]).groups.items():
        marker_stats.loc[idx, "q_value"] = bh_fdr(marker_stats.loc[idx, "p_value"].to_numpy(dtype=float))

    marker_stats["score"] = np.sign(marker_stats["adj_delta"]) * safe_logscore(marker_stats["q_value"].to_numpy())

    sig_rows = []
    for cl in sorted(marker_stats["cluster"].unique()):
        for ev in event_cols:
            sub = marker_stats[(marker_stats["cluster"] == cl) & (marker_stats["event"] == ev)].copy()
            if sub.empty:
                continue

            sub = sub[sub["q_value"] <= float(q_cut)].copy()
            if sub.empty:
                continue

            sub = sub[
                (np.abs(sub["raw_prev_delta"]) >= float(min_abs_delta_prev))
                | (sub["auc"] >= float(min_auc_enrich))
                | (sub["auc"] <= float(max_auc_deplete))
            ].copy()
            if sub.empty:
                continue

            top_pos = (
                sub[(sub["adj_delta"] > 0) & (sub["auc"] >= float(min_auc_enrich))]
                .sort_values(["q_value", "raw_prev_delta", "auc"], ascending=[True, False, False])
                .head(int(top_n))
            )

            top_neg = (
                sub[(sub["adj_delta"] < 0) & (sub["auc"] <= float(max_auc_deplete))]
                .sort_values(["q_value", "raw_prev_delta", "auc"], ascending=[True, True, True])
                .head(int(top_n))
            )

            pick = pd.concat([top_pos, top_neg], ignore_index=True)

            for _, r in pick.iterrows():
                sig_rows.append({
                    "cluster": int(cl),
                    "event": ev,
                    "gene_symbol": r["gene_symbol"],
                    "adj_cluster_mean": float(r["adj_cluster_mean"]),
                    "adj_rest_mean": float(r["adj_rest_mean"]),
                    "adj_delta": float(r["adj_delta"]),
                    "auc": float(r["auc"]),
                    "p_value": float(r["p_value"]),
                    "q_value": float(r["q_value"]),
                    "score": float(r["score"]),
                    "raw_prev_cluster": float(r["raw_prev_cluster"]),
                    "raw_prev_cluster_lo": float(r["raw_prev_cluster_lo"]),
                    "raw_prev_cluster_hi": float(r["raw_prev_cluster_hi"]),
                    "raw_prev_rest": float(r["raw_prev_rest"]),
                    "raw_prev_rest_lo": float(r["raw_prev_rest_lo"]),
                    "raw_prev_rest_hi": float(r["raw_prev_rest_hi"]),
                    "raw_prev_delta": float(r["raw_prev_delta"]),
                    "raw_prev_delta_lo": float(r["raw_prev_delta_lo"]),
                    "raw_prev_delta_hi": float(r["raw_prev_delta_hi"]),
                })

    signatures = pd.DataFrame(sig_rows)
    if not signatures.empty:
        signatures = signatures.sort_values(["cluster", "event", "q_value"], ascending=[True, True, True])

    return {"cluster_sizes": cluster_sizes, "marker_stats": marker_stats, "signatures": signatures}


def track2_gene_markers_by_platform(
    *,
    gene_panel: pd.DataFrame,
    clusters_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    signatures_df: pd.DataFrame,
    event_cols: list[str],
    platform_col: str = "platform_family_refined",
    id_col: str = "analysis_id",
    cluster_col: str = "cluster",
    outlier_label: int = -1,
    min_cluster_size: int = 50,
    min_platform_samples: int = 20,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    min_nonzero: int | str = "auto",
    autosomes_only: bool = False,
    background_analysis_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compute per-platform Track2 stats for pooled top markers only."""

    out_cols = [
        "cluster",
        "event",
        "gene_symbol",
        "platform",
        "n_cluster_samples_platform",
        "n_rest_samples_platform",
        "adj_cluster_mean",
        "adj_rest_mean",
        "adj_delta",
        "auc",
        "p_value",
        "raw_prev_cluster",
        "raw_prev_cluster_lo",
        "raw_prev_cluster_hi",
        "raw_prev_rest",
        "raw_prev_rest_lo",
        "raw_prev_rest_hi",
        "raw_prev_delta",
        "raw_prev_delta_lo",
        "raw_prev_delta_hi",
        "n_gene_cluster",
        "n_gene_rest",
    ]

    if signatures_df is None or signatures_df.empty:
        return pd.DataFrame(columns=out_cols)

    if platform_col not in meta_df.columns:
        raise KeyError(f"track2_gene_markers_by_platform requires meta_df to contain {platform_col}")

    C, _, contexts = _evidence_labels(
        clusters_df,
        id_col=id_col,
        cluster_col=cluster_col,
        outlier_label=outlier_label,
        min_cluster_size=min_cluster_size,
        background_analysis_ids=background_analysis_ids,
    )
    if C.empty or not contexts:
        return pd.DataFrame(columns=out_cols)

    M = meta_df[[id_col, platform_col]].copy()
    M[id_col] = M[id_col].astype(str)
    M[platform_col] = M[platform_col].astype(str)
    M = M.drop_duplicates(id_col, keep="first")

    G = gene_panel.merge(C, on=id_col, how="inner").merge(M, on=id_col, how="inner")
    if G.empty:
        return pd.DataFrame(columns=out_cols)

    if bool(autosomes_only):
        if "chrom" not in G.columns:
            raise KeyError("track2_gene_markers_by_platform autosomes_only=True requires gene_panel to contain 'chrom'")
        autosomes = {str(i) for i in range(1, 23)}
        G["chrom"] = G["chrom"].astype(str).str.replace("chr", "", regex=False)
        G = G[G["chrom"].isin(autosomes)].copy()
        if G.empty:
            return pd.DataFrame(columns=out_cols)

    for ev in event_cols:
        G[ev] = pd.to_numeric(G[ev], errors="coerce")

    burdens = {}
    for ev in event_cols:
        b = G.groupby(id_col)[ev].mean()
        burdens[ev] = b.rename(f"burden_{ev}")
    B = pd.concat(burdens.values(), axis=1).reset_index()
    G2 = G.merge(B, on=id_col, how="left")
    for ev in event_cols:
        G2[f"{ev}_adj"] = G2[ev] - G2[f"burden_{ev}"]

    pooled_markers = signatures_df[["cluster", "event", "gene_symbol"]].drop_duplicates().copy()
    pooled_markers["cluster"] = pooled_markers["cluster"].astype(int)
    pooled_set = {
        (int(r.cluster), str(r.event), str(r.gene_symbol))
        for r in pooled_markers.itertuples(index=False)
    }

    rows = []
    for plat, dfp in G2.groupby(platform_col, sort=False):
        n_by_cluster = dfp.groupby(cluster_col)[id_col].nunique()
        total_n = int(dfp[id_col].nunique())
        if total_n < 2 * int(min_platform_samples):
            continue

        for (cl, ev, gene) in sorted(pooled_set):
            n1 = int(n_by_cluster.get(cl, 0))
            n2 = int(total_n - n1)
            if n1 < int(min_platform_samples) or n2 < int(min_platform_samples):
                continue

            df_gene = dfp[dfp["gene_symbol"].astype(str) == str(gene)].copy()
            if df_gene.empty:
                continue
            df1 = df_gene[df_gene[cluster_col].astype(int) == int(cl)]
            df2 = df_gene[df_gene[cluster_col].astype(int) != int(cl)]
            if df1.empty or df2.empty:
                continue

            x = df1[f"{ev}_adj"].to_numpy(dtype=float)
            y = df2[f"{ev}_adj"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]

            x_raw = df1[ev].to_numpy(dtype=float)
            y_raw = df2[ev].to_numpy(dtype=float)
            x_raw = x_raw[np.isfinite(x_raw)]
            y_raw = y_raw[np.isfinite(y_raw)]

            if min_nonzero == "auto":
                mnz = max(3, int(np.ceil(0.01 * min(len(x_raw), len(y_raw)))))
            else:
                mnz = int(min_nonzero)

            if (np.count_nonzero(x_raw) < mnz) and (np.count_nonzero(y_raw) < mnz):
                continue

            stats = compute_marker_effect_stats(
                x,
                y,
                x_raw,
                y_raw,
                alpha=alpha,
                alternative=alternative,
            )
            if stats is None:
                continue

            rows.append({
                "cluster": int(cl),
                "event": str(ev),
                "gene_symbol": str(gene),
                "platform": str(plat),
                "n_cluster_samples_platform": int(n1),
                "n_rest_samples_platform": int(n2),
                **stats,
            })

    out = pd.DataFrame(rows, columns=out_cols)
    if not out.empty:
        out = out.sort_values(["cluster", "event", "gene_symbol", "platform"]).reset_index(drop=True)
    return out


def track2_platform_stability(
    *,
    signatures_df: pd.DataFrame,
    per_platform_df: pd.DataFrame,
    min_abs_prevalence_delta: float = 0.05,
    min_abs_auc_shift: float = 0.10,
    min_platforms_supporting: int = 2,
    min_support_fraction: float = 2.0 / 3.0,
    max_auc_gap_allowed: float = 0.15,
) -> pd.DataFrame:
    """Summarize cross-platform stability for pooled Track2 signatures."""

    out_cols = [
        "cluster",
        "event",
        "gene_symbol",
        "pooled_adj_delta",
        "pooled_auc",
        "pooled_q_value",
        "pooled_score",
        "pooled_raw_prev_delta",
        "n_platforms_tested",
        "n_platforms_supporting",
        "support_fraction",
        "direction_consistent",
        "max_auc_gap",
        "max_prev_delta_gap",
        "single_platform_driven",
        "platform_support_class",
        "passes_strict_support",
    ]

    if signatures_df is None or signatures_df.empty:
        return pd.DataFrame(columns=out_cols)

    pooled = signatures_df.copy()
    pooled["cluster"] = pooled["cluster"].astype(int)
    keep_cols = [
        "cluster",
        "event",
        "gene_symbol",
        "adj_delta",
        "auc",
        "q_value",
        "score",
        "raw_prev_delta",
    ]
    pooled = pooled[[c for c in keep_cols if c in pooled.columns]].drop_duplicates(["cluster", "event", "gene_symbol"])

    rows = []
    for _, prow in pooled.iterrows():
        cl = int(prow["cluster"])
        ev = str(prow["event"])
        gene = str(prow["gene_symbol"])
        sub = per_platform_df[
            (per_platform_df["cluster"].astype(int) == cl)
            & (per_platform_df["event"].astype(str) == ev)
            & (per_platform_df["gene_symbol"].astype(str) == gene)
        ].copy() if (per_platform_df is not None and not per_platform_df.empty) else pd.DataFrame()

        stab = summarize_platform_marker_stability(
            sub,
            prow,
            min_abs_prevalence_delta=min_abs_prevalence_delta,
            min_abs_auc_shift=min_abs_auc_shift,
            min_platforms_supporting=min_platforms_supporting,
            min_support_fraction=min_support_fraction,
            max_auc_gap_allowed=max_auc_gap_allowed,
        )
        rows.append({
            "cluster": cl,
            "event": ev,
            "gene_symbol": gene,
            "pooled_adj_delta": float(prow.get("adj_delta", np.nan)),
            "pooled_auc": float(prow.get("auc", np.nan)),
            "pooled_q_value": float(prow.get("q_value", np.nan)),
            "pooled_score": float(prow.get("score", np.nan)),
            "pooled_raw_prev_delta": float(prow.get("raw_prev_delta", np.nan)),
            **stab,
        })

    out = pd.DataFrame(rows, columns=out_cols)
    if not out.empty:
        out = out.sort_values(["cluster", "event", "pooled_q_value", "gene_symbol"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out


def compute_track2_signatures(
    *,
    root: str,
    gene_panel: pd.DataFrame,
    clusters_df: pd.DataFrame,
    event_cols: list[str],
    id_col: str = "analysis_id",
    cluster_col: str = "cluster",
    meta_df: pd.DataFrame | None = None,
    platform_col: str = "platform_family_refined",
    outlier_label: int = -1,
    min_cluster_size: int = 50,
    autosomes_only_track2: bool = False,
    top_n: int = 30,
    q_cut: float = 0.05,
    min_abs_prevalence_delta: float = 0.05,
    min_abs_auc_shift: float = 0.10,
    min_platform_samples: int = 20,
    min_platforms_supporting: int = 2,
    min_platform_support_fraction: float = 2.0 / 3.0,
    max_auc_gap_allowed: float = 0.15,
    background_analysis_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Compute pooled and platform-aware Track 2 gene evidence for one root."""

    t2 = track2_gene_markers(
        gene_panel=gene_panel,
        clusters_df=clusters_df,
        event_cols=event_cols,
        id_col=id_col,
        cluster_col=cluster_col,
        outlier_label=outlier_label,
        min_cluster_size=min_cluster_size,
        top_n=int(top_n),
        alternative="two-sided",
        q_cut=float(q_cut),
        min_nonzero="auto",
        min_abs_delta_prev=float(min_abs_prevalence_delta),
        min_auc_enrich=float(0.5 + min_abs_auc_shift),
        max_auc_deplete=float(0.5 - min_abs_auc_shift),
        autosomes_only=bool(autosomes_only_track2),
        background_analysis_ids=background_analysis_ids,
    )
    t2_platform_stats = pd.DataFrame()
    t2_platform_stability = pd.DataFrame()
    if meta_df is not None and not t2["signatures"].empty:
        t2_platform_stats = track2_gene_markers_by_platform(
            gene_panel=gene_panel,
            clusters_df=clusters_df,
            meta_df=meta_df,
            signatures_df=t2["signatures"],
            event_cols=event_cols,
            platform_col=platform_col,
            id_col=id_col,
            cluster_col=cluster_col,
            outlier_label=outlier_label,
            min_cluster_size=min_cluster_size,
            min_platform_samples=int(min_platform_samples),
            alternative="two-sided",
            alpha=0.05,
            min_nonzero="auto",
            autosomes_only=bool(autosomes_only_track2),
            background_analysis_ids=background_analysis_ids,
        )
        t2_platform_stability = track2_platform_stability(
            signatures_df=t2["signatures"],
            per_platform_df=t2_platform_stats,
            min_abs_prevalence_delta=float(min_abs_prevalence_delta),
            min_abs_auc_shift=float(min_abs_auc_shift),
            min_platforms_supporting=int(min_platforms_supporting),
            min_support_fraction=float(min_platform_support_fraction),
            max_auc_gap_allowed=float(max_auc_gap_allowed),
        )

    return {
        "track2_markers": t2,
        "track2_platform_marker_stats": t2_platform_stats,
        "track2_platform_stability": t2_platform_stability,
    }
