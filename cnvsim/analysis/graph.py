from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def load_knn_tables(
    *,
    root: str,
    knn_dir: Path,
    K: int,
    n_pcs: int,
    metric: str,
    id_col: str = "analysis_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    metric = str(metric).lower()
    knn_dir = Path(knn_dir)

    idx_path = knn_dir / f"{root}_knn_idx_k{int(K)}_pcs{int(n_pcs)}_{metric}.parquet"
    dist_path = knn_dir / f"{root}_knn_dist_k{int(K)}_pcs{int(n_pcs)}_{metric}.parquet"

    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not dist_path.exists():
        raise FileNotFoundError(dist_path)

    idx_df = pd.read_parquet(idx_path)
    dist_df = pd.read_parquet(dist_path)

    if id_col not in idx_df.columns or id_col not in dist_df.columns:
        raise ValueError(f"[{root}] both idx/dist must include {id_col}")

    ids = idx_df[id_col].astype(str).to_numpy(dtype=object)

    nn_cols = [c for c in idx_df.columns if c != id_col]
    d_cols = [c for c in dist_df.columns if c != id_col]

    neigh_ids = idx_df[nn_cols].astype(str).to_numpy(dtype=object)
    neigh_dists = dist_df[d_cols].to_numpy(dtype=np.float32)

    if neigh_ids.shape[1] != int(K) or neigh_dists.shape[1] != int(K):
        raise ValueError(f"[{root}] K mismatch in saved files (expected {K})")

    return ids, neigh_ids, neigh_dists


def rbf_weights_from_dists(d: np.ndarray, sigma: float, eps: float = 1e-12) -> np.ndarray:
    sigma = float(max(sigma, eps))
    d = np.asarray(d, dtype=np.float32)
    return np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)


def build_weighted_knn_igraph(
    *,
    root: str,
    knn_dir: Path,
    K: int,
    n_pcs: int,
    metric: str,
    id_col: str = "analysis_id",
    sigma_quantile: float = 0.90,
    make_undirected: bool = True,
    combine_mode: str = "max",  # max/mean
):
    import igraph as ig

    ids, neigh_ids, neigh_dists = load_knn_tables(root=root, knn_dir=knn_dir, K=K, n_pcs=n_pcs, metric=metric, id_col=id_col)

    flat_d = neigh_dists.reshape(-1)
    sigma = float(np.quantile(flat_d, float(sigma_quantile)))
    if (not np.isfinite(sigma)) or sigma <= 0:
        pos = flat_d[flat_d > 0]
        sigma = float(np.median(pos)) if len(pos) else 1.0

    W = rbf_weights_from_dists(neigh_dists, sigma=sigma)

    ids = ids.astype(str)
    id_to_idx = {a: i for i, a in enumerate(ids)}

    src_idx = np.repeat(np.arange(len(ids), dtype=np.int32), int(K))
    dst_ids = neigh_ids.reshape(-1).astype(str)
    dst_idx = np.fromiter((id_to_idx.get(x, -1) for x in dst_ids), count=len(dst_ids), dtype=np.int32)
    w_flat = W.reshape(-1)

    keep = dst_idx >= 0
    src_idx = src_idx[keep]
    dst_idx = dst_idx[keep]
    w_flat = w_flat[keep]

    g = ig.Graph(n=len(ids), directed=True)
    g.vs[id_col] = ids.tolist()
    g.add_edges(list(zip(src_idx.tolist(), dst_idx.tolist())))
    g.es["weight"] = w_flat.tolist()

    if make_undirected:
        if combine_mode == "max":
            comb = {"weight": "max"}
        elif combine_mode == "mean":
            comb = {"weight": "mean"}
        else:
            raise ValueError("combine_mode must be 'max' or 'mean'")

        g = g.as_undirected(combine_edges=comb)
        g = g.simplify(combine_edges=comb)

    info = {
        "root": root,
        "n": int(len(ids)),
        "K": int(K),
        "metric": str(metric),
        "n_pcs": int(n_pcs),
        "sigma_q": float(sigma_quantile),
        "sigma": float(sigma),
        "w_min": float(np.min(w_flat)) if len(w_flat) else np.nan,
        "w_med": float(np.median(w_flat)) if len(w_flat) else np.nan,
        "w_max": float(np.max(w_flat)) if len(w_flat) else np.nan,
        "edges": int(g.ecount()),
        "directed": bool(g.is_directed()),
    }

    return g, info


def sanity_check_graph(
    g,
    *,
    id_attr: str = "analysis_id",
    weight_attr: str = "weight",
    giant_min_frac: float = 0.90,
    max_components_warn: int = 50,
) -> dict:

    n = g.vcount()
    m = g.ecount()

    ids = g.vs[id_attr] if id_attr in g.vs.attributes() else None
    if ids is None:
        id_ok = False
        n_unique_ids = None
        dup_ids = None
    else:
        ids = np.asarray(ids, dtype=object)
        n_unique_ids = int(pd.Index(ids).nunique())
        id_ok = (n_unique_ids == n)
        dup_ids = int(n - n_unique_ids)

    loop_mask = np.array(g.is_loop(), dtype=bool) if m else np.array([], dtype=bool)
    n_loops = int(loop_mask.sum())

    if weight_attr in g.es.attributes():
        w = np.array(g.es[weight_attr], dtype=float)
        finite = np.isfinite(w)
        n_nonfinite = int((~finite).sum())
        n_nonpos = int((w <= 0).sum())
        w_min = float(np.nanmin(w)) if m else np.nan
        w_med = float(np.nanmedian(w)) if m else np.nan
        w_max = float(np.nanmax(w)) if m else np.nan
    else:
        n_nonfinite = None
        n_nonpos = None
        w_min = w_med = w_max = None

    if g.is_directed():
        comps = g.components(mode="WEAK")
    else:
        comps = g.components()

    n_comps = int(len(comps))
    comp_sizes = np.array(comps.sizes(), dtype=int) if n_comps else np.array([], dtype=int)
    giant_size = int(comp_sizes.max()) if len(comp_sizes) else 0
    giant_frac = float(giant_size / n) if n else np.nan

    issues = []
    if not id_ok:
        issues.append(f"vertex_id_mismatch(unique={n_unique_ids}, vcount={n}, dup={dup_ids})")
    if n_loops > 0:
        issues.append(f"self_loops={n_loops}")
    if n_nonfinite not in [None, 0]:
        issues.append(f"nonfinite_weights={n_nonfinite}")
    if n_nonpos not in [None, 0]:
        issues.append(f"nonpositive_weights={n_nonpos}")
    if n_comps > max_components_warn:
        issues.append(f"many_components={n_comps}")
    if np.isfinite(giant_frac) and giant_frac < giant_min_frac:
        issues.append(f"giant_component_small(frac={giant_frac:.3f})")

    return {
        "v": int(n),
        "e": int(m),
        "directed": bool(g.is_directed()),
        "id_ok": bool(id_ok),
        "n_unique_ids": n_unique_ids,
        "self_loops": int(n_loops),
        "w_min": w_min,
        "w_med": w_med,
        "w_max": w_max,
        "n_nonfinite_w": n_nonfinite,
        "n_nonpos_w": n_nonpos,
        "n_components": int(n_comps),
        "giant_size": int(giant_size),
        "giant_frac": float(giant_frac) if giant_frac == giant_frac else np.nan,
        "issues": "; ".join(issues) if issues else "",
        "ok": (len(issues) == 0),
    }
