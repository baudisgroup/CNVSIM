from __future__ import annotations

"""
- per-root parameter selection is performed before kNN table construction
- build_and_save_knn_for_root writes two parquet tables:
    {root}_knn_idx_k{K}_pcs{n_pcs}_{metric}.parquet
    {root}_knn_dist_k{K}_pcs{n_pcs}_{metric}.parquet

"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cnvsim.analysis.config import KNNBuildParams


def build_and_save_knn_for_root(
    *,
    root: str,
    coords_df: pd.DataFrame,
    out_dir: Path,
    params: KNNBuildParams,
    id_col: str = "analysis_id",
    file_tag: str = "",
) -> dict:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = coords_df.copy()
    coords.index = coords.index.astype(str)

    used_cols = [c for c in coords.columns if str(c).startswith("D")][: int(params.n_pcs)]
    if not used_cols:
        raise ValueError(f"[{root}] no D* columns available")

    ids = coords.index.astype(str).to_numpy(dtype=object)
    X = coords[used_cols].to_numpy(dtype=np.float32)

    n = X.shape[0]
    K = int(params.K)
    if n <= K:
        raise ValueError(f"[{root}] n={n} must be > K={K}")

    nn = NearestNeighbors(n_neighbors=K + 1, metric=str(params.metric), algorithm="auto")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)

    idx = idx[:, 1:]
    dists = dists[:, 1:]

    neigh_ids = ids[idx]

    self_mask = (neigh_ids == ids[:, None])
    if bool(self_mask.any()):
        neigh_ids = neigh_ids.astype(object)
        neigh_ids[self_mask] = None
        dists = dists.astype(np.float32, copy=True)
        dists[self_mask] = np.inf

    metric = str(params.metric).lower()
    n_pcs_used = int(len(used_cols))

    out_idx = out_dir / f"{root}_knn_idx_k{K}_pcs{n_pcs_used}_{metric}{file_tag}.parquet"
    out_dist = out_dir / f"{root}_knn_dist_k{K}_pcs{n_pcs_used}_{metric}{file_tag}.parquet"

    df_idx = pd.DataFrame(neigh_ids, columns=[f"nn{i+1}" for i in range(K)])
    df_idx.insert(0, id_col, ids)

    df_dist = pd.DataFrame(dists.astype(np.float32), columns=[f"d{i+1}" for i in range(K)])
    df_dist.insert(0, id_col, ids)

    df_idx.to_parquet(out_idx, index=False)
    df_dist.to_parquet(out_dist, index=False)

    return {
        "icdot_root": root,
        "n_samples": int(n),
        "k": int(K),
        "metric": metric,
        "n_pcs_used": int(n_pcs_used),
        "pcs_used": ",".join([str(c) for c in used_cols]),
        "knn_idx_file": out_idx.name,
        "knn_dist_file": out_dist.name,
    }


def build_knn_tables_from_release_ir(
    *,
    release_dir: Path,
    ir_subdir: str = "analysis/ir",
    out_subdir: str = "analysis/knn/perroot_selected",
    params_default: KNNBuildParams = KNNBuildParams(),
    settings_tsv: Path | None = None,
    roots: list[str] | None = None,
    overwrite: bool = False,
    ir_base_dir: Path | None = None,
    out_base_dir: Path | None = None,
) -> pd.DataFrame:
    
    release_dir = Path(release_dir)
    ir_base = Path(ir_base_dir) if ir_base_dir is not None else release_dir

    ir_dir = ir_base / str(ir_subdir) / "perroot"
    if not ir_dir.exists():
        raise FileNotFoundError(f"Missing analysis IR: {ir_dir}")

    out_base = Path(out_base_dir) if out_base_dir is not None else release_dir
    out_dir = out_base / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # load settings
    settings_map: dict[str, dict] = {}
    settings_roots: list[str] = []
    if settings_tsv is not None:
        st = pd.read_csv(Path(settings_tsv), sep="\t", low_memory=False)
        root_col = "icdot_root" if "icdot_root" in st.columns else ("root" if "root" in st.columns else None)
        if root_col is None:
            raise ValueError("settings_tsv must have 'root' or 'icdot_root'")
        required = {"metric", "n_pcs", "K"}
        missing = sorted(required - set(st.columns))
        if missing:
            raise ValueError(f"settings_tsv is missing columns: {missing}")
        st[root_col] = st[root_col].astype(str)
        if st[root_col].duplicated().any():
            duplicated = sorted(st.loc[st[root_col].duplicated(False), root_col].unique())
            raise ValueError(f"settings_tsv contains duplicate roots: {duplicated}")
        if "selection_status" in st.columns:
            invalid = st[~st["selection_status"].astype(str).eq("subsampling_selected")]
            if not invalid.empty:
                raise ValueError(
                    "settings_tsv contains parameters that did not pass subsampling: "
                    + ", ".join(sorted(invalid[root_col].astype(str)))
                )
        settings_roots = sorted(st[root_col].tolist())
        settings_map = st.set_index(root_col).to_dict(orient="index")

    if roots is None:
        roots = settings_roots or sorted([p.name for p in ir_dir.iterdir() if p.is_dir()])
    elif settings_map:
        missing = sorted(set(roots) - set(settings_roots))
        if missing:
            raise ValueError(f"Requested roots are missing from settings_tsv: {missing}")

    rows = []

    for root in roots:
        coords_path = ir_dir / root / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(f"[{root}] missing analysis coordinates: {coords_path}")

        # expected output files 
        p = params_default
        s = settings_map.get(str(root), None) if settings_map else None
        if s is not None:
            p = KNNBuildParams(
                K=int(s.get("K", p.K)),
                metric=str(s.get("metric", p.metric)).lower(),
                n_pcs=int(s.get("n_pcs", p.n_pcs)),
            )

        metric = str(p.metric).lower()
        out_idx = out_dir / f"{root}_knn_idx_k{int(p.K)}_pcs{int(p.n_pcs)}_{metric}.parquet"
        out_dist = out_dir / f"{root}_knn_dist_k{int(p.K)}_pcs{int(p.n_pcs)}_{metric}.parquet"
        if (not overwrite) and out_idx.exists() and out_dist.exists():
            raise FileExistsError(
                f"[{root}] KNN outputs already exist; pass --overwrite: {out_idx}, {out_dist}"
            )

        coords = pd.read_parquet(coords_path)
        row = build_and_save_knn_for_root(root=root, coords_df=coords, out_dir=out_dir, params=p)
        if s is not None:
            row["selection_source"] = "runtime_per_root_selection"
            if "accepted" in s:
                row["accepted"] = bool(s.get("accepted"))
            if "reason" in s:
                row["reason"] = str(s.get("reason"))
        else:
            row["selection_source"] = "config_fallback"
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "icdot_root" if "icdot_root" in df.columns else ("root" if "root" in df.columns else None)
    if sort_col is not None:
        df = df.sort_values(sort_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    man_path = out_dir / "knn_manifest.tsv"
    df.to_csv(man_path, sep="\t", index=False)

    (out_dir / "knn_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "release_dir": str(release_dir),
        "ir_base_dir": str(ir_base),
        "out_base_dir": str(out_base),
        "ir_subdir": str(ir_subdir),
        "out_dir": str(out_dir),
        "default_params": params_default.__dict__,
        "settings_tsv": str(settings_tsv) if settings_tsv else None,
        "n_roots": int(df.shape[0]),
        "outputs": {"manifest": str(man_path)},
    }, indent=2))

    return df
