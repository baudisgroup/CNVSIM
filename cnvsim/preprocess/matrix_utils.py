from __future__ import annotations

"""Metrics used to quantify platform signal and neighbourhood preservation."""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors


def platform_predictability(
    pcs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    platform_col: str = "platform_family_refined",
    n_pcs: int = 10,
    min_per_class: int = 50,
    cv: int = 5,
) -> float:
    """Cross-validated balanced accuracy for predicting platform from PCs.

    Returns ``NaN`` when fewer than two platform classes are present or any
    class has fewer than ``min_per_class`` samples.
    """

    vc = meta_df[platform_col].astype(str).value_counts()
    if vc.shape[0] < 2:
        return float("nan")
    if (vc < int(min_per_class)).any():
        return float("nan")

    Z = pcs_df.iloc[:, : min(int(n_pcs), pcs_df.shape[1])].values
    y = meta_df[platform_col].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(np.unique(y)) < 2:
        return float("nan")

    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    return float(cross_val_score(clf, Z, y, cv=int(cv), scoring="balanced_accuracy").mean())


def eta2_by_group(x: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of one variable's variance explained by group labels."""

    x = np.asarray(x, float)
    g = np.asarray(groups)
    mu = np.mean(x)
    ss_tot = np.sum((x - mu) ** 2)
    if ss_tot <= 0:
        return 0.0

    ss_between = 0.0
    for lvl in np.unique(g):
        idx = (g == lvl)
        mu_g = np.mean(x[idx])
        ss_between += idx.sum() * (mu_g - mu) ** 2

    return float(ss_between / ss_tot)


def _knn_graph(pcs_df: pd.DataFrame, k: int = 15, n_pcs: int = 20):
    Z = pcs_df.iloc[:, : min(int(n_pcs), pcs_df.shape[1])].values
    if Z.shape[0] <= int(k) + 1:
        return None

    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean").fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return idx


def _connected_components_from_idx(idx: np.ndarray) -> tuple[int, float]:
    idx = np.asarray(idx)
    if idx.size == 0:
        return 0, float("nan")

    n = int(idx.shape[0])
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in idx[i]:
            union(i, int(j))

    roots = np.fromiter((find(i) for i in range(n)), count=n, dtype=np.int32)
    comp_sizes = pd.Series(roots).value_counts().to_numpy(dtype=int)
    n_components = int(comp_sizes.size)
    giant_frac = float(comp_sizes.max() / max(1, n)) if comp_sizes.size else float("nan")
    return n_components, giant_frac


def knn_platform_mixing(
    pcs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    platform_col: str,
    *,
    id_col: str = "analysis_id",
    k: int = 15,
    n_pcs: int = 20,
) -> float:
    """Mean fraction of each sample's neighbours from another platform."""

    idx = _knn_graph(pcs_df, k=k, n_pcs=n_pcs)
    if idx is None:
        return float("nan")

    m = meta_df.set_index(id_col).reindex(pcs_df.index).reset_index()
    plat = m[platform_col].astype(str).values
    mix = [(plat[idx[i]] != plat[i]).mean() for i in range(len(plat))]
    return float(np.mean(mix))


def knn_graph_connectivity(
    pcs_df: pd.DataFrame,
    *,
    k: int = 15,
    n_pcs: int = 20,
) -> dict:
    """Return component count and largest-component fraction of a kNN graph."""

    idx = _knn_graph(pcs_df, k=k, n_pcs=n_pcs)
    if idx is None:
        return {"n_components": float("nan"), "giant_frac": float("nan")}

    n_components, giant_frac = _connected_components_from_idx(idx)
    return {"n_components": int(n_components), "giant_frac": float(giant_frac)}


def knn_label_purity_fixed(
    pcs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    label_col: str,
    *,
    id_col: str = "analysis_id",
    k: int = 15,
    n_pcs: int = 20,
) -> float:
    """Mean fraction of neighbours sharing each sample's biological label."""

    idx = _knn_graph(pcs_df, k=k, n_pcs=n_pcs)
    if idx is None or label_col not in meta_df.columns:
        return float("nan")

    m = meta_df.set_index(id_col).reindex(pcs_df.index).reset_index()
    lab = m[label_col].astype(str).values
    pur = [(lab[idx[i]] == lab[i]).mean() for i in range(len(lab))]
    return float(np.mean(pur))


def knn_label_purity_within_groups(
    pcs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    label_col: str,
    group_col: str,
    *,
    id_col: str = "analysis_id",
    k: int = 15,
    n_pcs: int = 20,
    min_group_n: int = 200,
) -> float:
    """Compute label purity using only neighbours from the same group.

    This prevents separation between platform groups from artificially raising
    morphology purity. The result is a sample-size-weighted mean over groups
    with at least ``min_group_n`` samples.
    """

    if label_col not in meta_df.columns or group_col not in meta_df.columns:
        return float("nan")

    m = meta_df.set_index(id_col).reindex(pcs_df.index).reset_index()
    # overall kNN graph (in full space)
    idx = _knn_graph(pcs_df, k=k, n_pcs=n_pcs)
    if idx is None:
        return float("nan")

    lab = m[label_col].astype(str).values
    grp = m[group_col].astype(str).values

    pur_all = []
    w_all = []
    for g in np.unique(grp):
        mask = (grp == g)
        n = int(mask.sum())
        if n < int(min_group_n):
            continue
        # indices of samples in this group
        idx_g = np.where(mask)[0]
        # for each sample i in group, only count neighbors also in group
        pur_g = []
        idx_set = set(idx_g.tolist())
        for ii in idx_g:
            neigh = [j for j in idx[ii] if j in idx_set]
            if not neigh:
                continue
            pur_g.append(float(np.mean(lab[neigh] == lab[ii])))
        if pur_g:
            pur_all.append(float(np.mean(pur_g)))
            w_all.append(n)

    if not pur_all:
        return float("nan")
    w = np.asarray(w_all, float)
    return float(np.average(np.asarray(pur_all, float), weights=w))
