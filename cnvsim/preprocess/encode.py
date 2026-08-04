from __future__ import annotations

"""Encode sample-gene CNV summaries as root-specific feature matrices.

For each grouped ICD-O-T root, this module converts gene-level overlap
fractions into separate ``GENE__gain`` and ``GENE__loss`` columns. Low-level
and high-level event indicators contribute 1 and 2 to the corresponding score,
respectively. A sample-gene feature can therefore range from 0 to 3 when both
indicators are present.

Output per root
---------------
- ``X``: samples by gain/loss features, stored as ``float32``.
- ``meta``: one metadata row per sample, ordered identically to ``X``, with
  burden counts calculated from the final encoded feature set.
- ``info``: sample, gene, threshold, and filtering counts.

Required columns
----------------
meta_df (bios_df):
- analysis_id (str)
- icdot_root_grouped (root key)
- platform_family_refined
- icdo_morphology_id (used downstream)

gene_df (gene_panel):
- analysis_id (str)
- gene_symbol
- dup_frac, del_frac, hldup_frac, hldel_frac
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_COLS = ["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]
FEATURE_BURDEN_COLS = [
    "n_genes_altered",
    "n_genes_gain",
    "n_genes_loss",
    "n_genes_hl_gain",
    "n_genes_hl_loss",
    "architecture_informative",
]


@dataclass(frozen=True)
class EncodeParams:
    """Thresholds controlling platform, sample, gene, and event filtering."""

    min_samples_per_family: int = 50
    min_altered_genes_per_sample: int = 1
    stable_gene_min_coverage: float | None = 0.95
    min_frac_low: float = 0.10
    min_frac_high: float = 0.20


def add_cnv_event_flags(
    df: pd.DataFrame,
    min_frac_low: float = 0.05,
    min_frac_high: float = 0.10,
) -> pd.DataFrame:
    """Add Boolean low- and high-level gain/loss indicators.

    The four input columns are fractions of each gene interval covered by the
    corresponding CNV event class. No rows are aggregated in this function.
    """

    out = df.copy()
    required = EVENT_COLS
    for col in required:
        if col not in out.columns:
            raise KeyError(f"add_cnv_event_flags: missing column '{col}' in input df")
        out[col] = out[col].astype(float)

    out["low_gain"] = out["dup_frac"] >= float(min_frac_low)
    out["low_loss"] = out["del_frac"] >= float(min_frac_low)
    out["hl_gain"] = out["hldup_frac"] >= float(min_frac_high)
    out["hl_loss"] = out["hldel_frac"] >= float(min_frac_high)
    out["any_cnv"] = out["low_gain"] | out["low_loss"] | out["hl_gain"] | out["hl_loss"]
    return out


def compute_sample_burden(panel_flags: pd.DataFrame) -> pd.DataFrame:
    """Count altered panel genes of each event class for every analysis."""

    tmp = (
        panel_flags.groupby(["analysis_id", "gene_symbol"], as_index=False)
        .agg(
            any_cnv=("any_cnv", "max"),
            any_gain=("low_gain", "max"),
            any_loss=("low_loss", "max"),
            any_hl_gain=("hl_gain", "max"),
            any_hl_loss=("hl_loss", "max"),
        )
    )
    burden = (
        tmp.groupby("analysis_id", as_index=False)
        .agg(
            n_genes_altered=("any_cnv", "sum"),
            n_genes_gain=("any_gain", "sum"),
            n_genes_loss=("any_loss", "sum"),
            n_genes_hl_gain=("any_hl_gain", "sum"),
            n_genes_hl_loss=("any_hl_loss", "sum"),
        )
    )
    return burden


def compute_feature_burden(
    features: pd.DataFrame,
    *,
    gene_chunk_size: int = 256,
) -> pd.DataFrame:
    """Count altered genes from the final gain/loss feature matrix.

    Counts are derived after stable-gene filtering, so they describe the
    representation that is actually passed to PCA, graph construction, and
    projection. A gene with both gain and loss scores is counted once in
    ``n_genes_altered``. Scores of two or more indicate a high-level event.
    """

    gain_columns = {
        column[: -len("__gain")]: column
        for column in features.columns
        if str(column).endswith("__gain")
    }
    loss_columns = {
        column[: -len("__loss")]: column
        for column in features.columns
        if str(column).endswith("__loss")
    }
    genes = sorted(set(gain_columns) | set(loss_columns))
    if not genes:
        raise ValueError("Feature matrix contains no __gain or __loss columns")

    n_samples = int(features.shape[0])
    counts = {
        "n_genes_altered": np.zeros(n_samples, dtype=np.int32),
        "n_genes_gain": np.zeros(n_samples, dtype=np.int32),
        "n_genes_loss": np.zeros(n_samples, dtype=np.int32),
        "n_genes_hl_gain": np.zeros(n_samples, dtype=np.int32),
        "n_genes_hl_loss": np.zeros(n_samples, dtype=np.int32),
    }

    for start in range(0, len(genes), int(gene_chunk_size)):
        chunk = genes[start : start + int(gene_chunk_size)]
        gain = features.reindex(
            columns=[gain_columns.get(gene, "") for gene in chunk],
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)
        loss = features.reindex(
            columns=[loss_columns.get(gene, "") for gene in chunk],
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)
        gain_present = gain > 0
        loss_present = loss > 0
        counts["n_genes_altered"] += np.logical_or(
            gain_present, loss_present
        ).sum(axis=1, dtype=np.int32)
        counts["n_genes_gain"] += gain_present.sum(axis=1, dtype=np.int32)
        counts["n_genes_loss"] += loss_present.sum(axis=1, dtype=np.int32)
        counts["n_genes_hl_gain"] += (gain >= 2).sum(axis=1, dtype=np.int32)
        counts["n_genes_hl_loss"] += (loss >= 2).sum(axis=1, dtype=np.int32)

    burden = pd.DataFrame(counts, index=features.index.astype(str))
    burden.index.name = "analysis_id"
    burden["architecture_informative"] = burden["n_genes_altered"] >= 1
    return burden


def gene_coverage_by_platform(panel_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the fraction of samples with each gene evaluable per platform.

    Presence of a sample-gene row indicates that the gene was covered by the
    source assay; it does not require that the gene was gained or lost.
    """

    tmp = panel_df[["analysis_id", "gene_symbol"]].drop_duplicates().copy()
    tmp["analysis_id"] = tmp["analysis_id"].astype(str)

    meta2 = meta_df.copy()
    if "analysis_id" not in meta2.columns:
        if meta2.index.name == "analysis_id":
            meta2 = meta2.reset_index()
        else:
            meta2 = meta2.reset_index().rename(columns={"index": "analysis_id"})

    meta2 = meta2[["analysis_id", "platform_family_refined"]].copy()
    meta2["analysis_id"] = meta2["analysis_id"].astype(str)
    meta2["platform_family_refined"] = meta2["platform_family_refined"].fillna("Unknown")

    tmp = tmp.merge(meta2, on="analysis_id", how="left")

    cov = (
        tmp.groupby(["platform_family_refined", "gene_symbol"])["analysis_id"]
        .nunique()
        .reset_index(name="n_samples_with_gene")
    )

    totals = (
        meta2.groupby("platform_family_refined")["analysis_id"]
        .nunique()
        .reset_index(name="n_samples_total")
    )

    cov = cov.merge(totals, on="platform_family_refined", how="left")
    cov["coverage_frac"] = cov["n_samples_with_gene"] / cov["n_samples_total"]
    return cov


def make_icdot_panel(
    icdot_root: str,
    bios_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    min_samples_per_family: int | None = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Select one root and remove platform families below the sample threshold.

    Returns the selected sample-gene rows, the pre-filter platform counts, and
    the selected analysis identifiers.
    """

    fam_bios = bios_df[bios_df["icdot_root_grouped"] == icdot_root].copy()

    family_counts = (
        fam_bios.groupby("platform_family_refined")["analysis_id"]
        .nunique()
        .reset_index(name="n_analyses")
        .sort_values("n_analyses", ascending=False)
    )

    if (min_samples_per_family is not None) and (min_samples_per_family > 0):
        big_families = family_counts.loc[
            family_counts["n_analyses"] >= int(min_samples_per_family),
            "platform_family_refined",
        ]
        fam_bios = fam_bios[fam_bios["platform_family_refined"].isin(big_families)].copy()

    expected_ids = fam_bios["analysis_id"].astype(str).unique()

    fam_panel = gene_df.merge(
        fam_bios[["analysis_id", "platform_family_refined", "source_group_refined", "icdot_root_grouped"]],
        on="analysis_id",
        how="inner",
    )

    return fam_panel, family_counts, expected_ids


def build_feature_matrix_for_root(
    icdot_root: str,
    bios_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    *,
    min_samples_per_family: int = 50,
    min_altered_genes_per_sample: int = 1,
    stable_gene_min_coverage: float | None = 0.95,
    min_frac_low: float = 0.10,
    min_frac_high: float = 0.20,
    autosomes_only: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """Build the gain/loss feature matrix used for one root's PCA space.

    Processing order is: optional autosome restriction, root and platform-size
    filtering, event thresholding, duplicate sample-gene aggregation, matrix
    pivoting, metadata alignment, stable-gene filtering, and optional sample-
    burden filtering. Calculating burden after stable-gene filtering ensures
    that it describes the representation actually used downstream. A gene is
    retained only when its assay coverage reaches
    ``stable_gene_min_coverage`` in every retained platform family for that
    root.
    """

    if bool(autosomes_only):
        if "chrom" not in gene_df.columns:
            raise KeyError("autosomes_only=True requires gene_df to contain 'chrom'")
        autosomes = {str(i) for i in range(1, 23)}
        gene_df = gene_df.copy()
        gene_df["chrom"] = gene_df["chrom"].astype(str).str.replace("chr", "", regex=False)
        gene_df = gene_df[gene_df["chrom"].isin(autosomes)].copy()

    panel, fam_info, expected_ids = make_icdot_panel(
        icdot_root=icdot_root,
        bios_df=bios_df,
        gene_df=gene_df,
        min_samples_per_family=min_samples_per_family,
    )

    if panel.empty:
        print(f"[build_feature_matrix_for_root] {icdot_root}: no panel data.")
        return None, None, None

    panel_flags = add_cnv_event_flags(panel, min_frac_low=min_frac_low, min_frac_high=min_frac_high)

    enc = panel_flags.copy()
    enc["gain_score"] = enc["low_gain"].astype(float) + 2.0 * enc["hl_gain"].astype(float)
    enc["loss_score"] = enc["low_loss"].astype(float) + 2.0 * enc["hl_loss"].astype(float)

    enc_agg = enc.groupby(["analysis_id", "gene_symbol"], as_index=False)[["gain_score", "loss_score"]].max()

    mat_gain = enc_agg.pivot(index="analysis_id", columns="gene_symbol", values="gain_score")
    mat_loss = enc_agg.pivot(index="analysis_id", columns="gene_symbol", values="loss_score")

    all_genes = sorted(set(mat_gain.columns) | set(mat_loss.columns))

    mat_gain = mat_gain.reindex(columns=all_genes).fillna(0.0)
    mat_loss = mat_loss.reindex(columns=all_genes).fillna(0.0)

    X = pd.concat([mat_gain.add_suffix("__gain"), mat_loss.add_suffix("__loss")], axis=1).astype("float32")

    # meta alignment
    X.index = X.index.astype(str)
    bios2 = bios_df.copy()
    bios2["analysis_id"] = bios2["analysis_id"].astype(str)

    meta = (
        bios2[bios2["analysis_id"].isin(X.index)]
        .drop_duplicates(subset=["analysis_id"], keep="first")
        .set_index("analysis_id")
        .reindex(X.index)
        .reset_index()
    )

    if meta.isna().all(axis=1).any():
        raise ValueError(f"[{icdot_root}] some analysis_id present in X but missing from bios_df/meta.")

    if stable_gene_min_coverage is not None:
        cov = gene_coverage_by_platform(panel_flags, meta)
        stable_genes = cov.groupby("gene_symbol")["coverage_frac"].min()
        stable_genes = stable_genes[stable_genes >= float(stable_gene_min_coverage)].index.tolist()

        keep_cols = [f"{g}__gain" for g in stable_genes] + [f"{g}__loss" for g in stable_genes]
        keep_cols = [c for c in keep_cols if c in X.columns]
        X = X[keep_cols].astype("float32")

    feature_burden = compute_feature_burden(X)
    n_before_burden_filter = int(X.shape[0])
    n_zero_before_filter = int((~feature_burden["architecture_informative"]).sum())

    if int(min_altered_genes_per_sample) > 0:
        keep_ids = feature_burden.index[
            feature_burden["n_genes_altered"]
            >= int(min_altered_genes_per_sample)
        ]
        X = X.reindex(keep_ids).copy()
        feature_burden = feature_burden.reindex(keep_ids).copy()

    if X.empty:
        print(f"[{icdot_root}] All samples removed by burden filter.")
        return None, None, None

    meta = (
        meta.drop_duplicates(subset=["analysis_id"], keep="first")
        .set_index("analysis_id")
        .reindex(X.index)
        .reset_index()
    )
    feature_burden_out = feature_burden.reset_index()
    meta = meta.merge(
        feature_burden_out,
        on="analysis_id",
        how="left",
        validate="one_to_one",
    )

    info = {
        "n_samples_before_qc": int(panel["analysis_id"].nunique()),
        "n_samples_before_burden_filter": n_before_burden_filter,
        "n_samples_zero_altered_genes_before_filter": n_zero_before_filter,
        "n_samples_removed_by_burden_filter": int(
            n_before_burden_filter - X.shape[0]
        ),
        "n_samples_after_qc": int(X.shape[0]),
        "n_genes_union": int(len(all_genes)),
        "n_features_after_stable_filter": int(X.shape[1]),
        "n_genes_after_stable_filter": int(X.shape[1] // 2),
        "min_altered_genes_per_sample": int(min_altered_genes_per_sample),
        "min_frac_low": float(min_frac_low),
        "min_frac_high": float(min_frac_high),
        "autosomes_only": bool(autosomes_only),
    }
    if stable_gene_min_coverage is not None:
        info["stable_gene_min_coverage"] = float(stable_gene_min_coverage)

    return X, meta, info
