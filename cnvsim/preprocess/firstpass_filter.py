from __future__ import annotations

"""Detect and remove small sample groups dominated by one assay platform.

For each anatomical root, standardized gene-event features are reduced by PCA
and UMAP, then clustered with HDBSCAN. A non-noise cluster is flagged as likely
technical only when it is small, strongly dominated by one platform family,
and not dominated by one morphology label. Samples in flagged clusters are
removed before the final root-specific PCA space is fitted.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import umap
import hdbscan

from cnvsim.preprocess.matrix_utils import platform_predictability
from cnvsim.preprocess.scaling import make_feature_scaler


PLATFORM_COL = "platform_family_refined"
LABEL_COL = "icdo_morphology_id"
ID_COL = "analysis_id"


@dataclass(frozen=True)
class FirstPassParams:
    """PCA/UMAP/HDBSCAN settings and technical-cluster thresholds."""

    random_state: int = 0
    n_pcs: int = 50

    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    hdb_min_cluster_size: int = 200
    hdb_min_samples: int = 50

    max_rel_cluster_size: float = 0.10
    min_platform_dominance: float = 0.90
    max_top_morph_frac: float = 0.50


def ensure_aligned(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Reorder one-row-per-analysis metadata to exactly match ``X.index``."""

    X = X.copy()
    X.index = X.index.astype(str)

    meta2 = meta.copy()
    if ID_COL not in meta2.columns:
        if meta2.index.name == ID_COL:
            meta2 = meta2.reset_index()
        else:
            meta2 = meta2.reset_index().rename(columns={"index": ID_COL})

    meta2[ID_COL] = meta2[ID_COL].astype(str)

    meta2 = (
        meta2.drop_duplicates(ID_COL, keep="first")
        .set_index(ID_COL)
        .reindex(X.index)
        .reset_index()
    )

    if meta2.isna().all(axis=1).any():
        n = int(meta2.isna().all(axis=1).sum())
        bad = meta2.loc[meta2.isna().all(axis=1), ID_COL].head(10).tolist()
        raise ValueError(f"Meta missing for {n} samples. Examples: {bad}")

    return meta2


def fit_pca_umap(
    X: pd.DataFrame,
    params: FirstPassParams,
    *,
    feature_scaling: str = "standard",
):
    """Fit feature scaling, PCA, and UMAP and return their coordinates."""

    scaler = make_feature_scaler(feature_scaling)
    Xs = scaler.fit_transform(X.values)

    pca = PCA(
        n_components=min(int(params.n_pcs), X.shape[0] - 1, X.shape[1]),
        random_state=int(params.random_state),
    )
    pcs = pca.fit_transform(Xs)

    reducer = umap.UMAP(
        n_neighbors=int(params.umap_n_neighbors),
        min_dist=float(params.umap_min_dist),
        metric=str(params.umap_metric),
        random_state=int(params.random_state),
        n_jobs=1,
    )
    emb = reducer.fit_transform(pcs)

    pcs_df = pd.DataFrame(pcs, index=X.index.astype(str), columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    emb_df = pd.DataFrame(emb, index=X.index.astype(str), columns=["UMAP1", "UMAP2"])

    return scaler, pca, pcs_df, reducer, emb_df


def hdbscan_on_umap(emb_df: pd.DataFrame, params: FirstPassParams) -> np.ndarray:
    """Assign HDBSCAN labels from the two-dimensional UMAP coordinates."""

    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(params.hdb_min_cluster_size),
        min_samples=int(params.hdb_min_samples),
    )
    labels = cl.fit_predict(emb_df[["UMAP1", "UMAP2"]].values)
    return labels


def summarize_clusters(meta_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize cluster size and dominant platform and morphology labels."""

    df = meta_df.copy()
    df["cluster_id"] = labels

    sizes = df.groupby("cluster_id", as_index=False).size().rename(columns={"size": "n_samples"})
    n_total = df.shape[0]
    sizes["rel_size"] = sizes["n_samples"] / n_total

    plat = (
        df.groupby(["cluster_id", PLATFORM_COL])[ID_COL]
        .nunique()
        .reset_index(name="n")
    )
    plat["frac"] = plat.groupby("cluster_id")["n"].transform(lambda x: x / x.sum())
    plat_dom = (
        plat.sort_values(["cluster_id", "frac"], ascending=[True, False])
        .groupby("cluster_id", as_index=False)
        .first()
        .rename(columns={PLATFORM_COL: "top_platform", "frac": "top_platform_frac"})
    )

    morph_available = (LABEL_COL in df.columns) and (not df[LABEL_COL].isna().all())

    if morph_available:
        morph = (
            df.groupby(["cluster_id", LABEL_COL])[ID_COL]
            .nunique()
            .reset_index(name="n")
        )
        morph["frac"] = morph.groupby("cluster_id")["n"].transform(lambda x: x / x.sum())
        morph_dom = (
            morph.sort_values(["cluster_id", "frac"], ascending=[True, False])
            .groupby("cluster_id", as_index=False)
            .first()
            .rename(columns={LABEL_COL: "top_morph", "frac": "top_morph_frac"})
        )
    else:
        morph_dom = pd.DataFrame({"cluster_id": sizes["cluster_id"], "top_morph": np.nan, "top_morph_frac": np.nan})

    out = sizes.merge(plat_dom, on="cluster_id", how="left").merge(morph_dom, on="cluster_id", how="left")
    return out


def flag_technical_clusters(cluster_df: pd.DataFrame, params: FirstPassParams) -> pd.DataFrame:
    """Apply the size, platform-dominance, and morphology-coherence rule.

    HDBSCAN noise is never removed by this rule. If morphology is unavailable,
    the morphology condition is treated as passed and the decision relies on
    cluster size and platform dominance.
    """

    df = cluster_df.copy()
    df["is_noise"] = df["cluster_id"] == -1
    df["is_small"] = df["rel_size"] <= float(params.max_rel_cluster_size)
    df["is_platform_pure"] = df["top_platform_frac"] >= float(params.min_platform_dominance)

    # Morphology is optional for some user cohorts; don't penalize when missing.
    if ("top_morph_frac" not in df.columns) or (df["top_morph_frac"].isna().all()):
        df["is_not_morph_pure"] = True
    else:
        df["is_not_morph_pure"] = df["top_morph_frac"].fillna(0.0) < float(params.max_top_morph_frac)

    df["is_technical"] = (~df["is_noise"]) & df["is_small"] & df["is_platform_pure"] & df["is_not_morph_pure"]
    return df

def run_firstpass_for_root(root: str, X: pd.DataFrame, meta: pd.DataFrame, params: FirstPassParams) -> dict:
    """Run technical-group detection for one root.

    Returns fitted transformation objects, PCA/UMAP coordinates, aligned
    metadata, the cluster-level decision table, and IDs selected for removal.
    """

    X = X.copy()
    X.index = X.index.astype(str)

    meta_aligned = ensure_aligned(X, meta)
    scaler, pca, pcs_df, reducer, emb_df = fit_pca_umap(X, params=params)
    labels = hdbscan_on_umap(emb_df, params=params)

    clus = summarize_clusters(meta_aligned, labels)
    clus = flag_technical_clusters(clus, params=params)

    bad_clusters = set(clus.loc[clus["is_technical"], "cluster_id"].tolist())
    bad_mask = pd.Series(labels, index=X.index).isin(bad_clusters)
    bad_ids = bad_mask[bad_mask].index.astype(str).tolist()

    return {
        "scaler": scaler,
        "pca": pca,
        "pcs": pcs_df,
        "umap": reducer,
        "emb": emb_df,
        "labels": labels,
        "meta": meta_aligned,
        "cluster_report": clus,
        "bad_ids": bad_ids,
    }
