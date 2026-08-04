from __future__ import annotations

"""Construct the root-specific PCA coordinates used for subtype discovery."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.preprocess.firstpass_filter import (
    FirstPassParams,
    fit_pca_umap,
)
from cnvsim.preprocess.matrix_utils import eta2_by_group, platform_predictability
from cnvsim.preprocess.scaling import l2_normalize_rows

ID_COL = "analysis_id"


@dataclass(frozen=True)
class Stage2AParams:
    """PCA transformation and platform-QC evaluation settings."""

    n_pcs: int = 50
    plat_eval_pcs: int = 10
    min_per_class: int = 50
    random_state: int = 0
    feature_scaling: str = "standard"


def compute_per_pc_platform_eta2(
    pcs_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    platform_col: str = "platform_family_refined",
) -> pd.DataFrame:
    """Compute per-PC eta2 against platform labels, aligned by analysis_id."""

    pcs = pcs_df.copy()
    pcs.index = pcs.index.astype(str)

    meta = meta_df.copy()
    if ID_COL not in meta.columns:
        raise ValueError(f"meta_df must contain {ID_COL}")
    if platform_col not in meta.columns:
        raise ValueError(f"meta_df must contain {platform_col}")

    meta[ID_COL] = meta[ID_COL].astype(str)
    meta_al = (
        meta.drop_duplicates(ID_COL, keep="first")
        .set_index(ID_COL)
        .reindex(pcs.index)
        .reset_index()
    )

    groups = meta_al[platform_col].astype(str).values
    rows = []
    for j, col in enumerate(pcs.columns):
        vals = pcs.iloc[:, j].to_numpy(dtype=float)
        eta2 = eta2_by_group(vals, groups)
        rows.append(
            {
                "pc_index": int(j + 1),
                "pc": str(col),
                "platform_eta2": float(eta2),
            }
        )
    return pd.DataFrame(rows)


def run_stage2a_for_root(
    root: str,
    X: pd.DataFrame,
    meta_stage1: pd.DataFrame,
    out_dir: Path,
    params: Stage2AParams,
    fp_params: FirstPassParams,
) -> dict:
    """Fit the post-filter PCA space for one anatomical root.

    Samples are ordered from ``meta_stage1`` and L2-normalized. Features are
    then variance-standardized before PCA.
    The function writes PC coordinates and per-PC platform eta-squared values,
    then reports cross-validated platform predictability as a technical QC
    metric.

    ``params`` supplies Stage 2A dimensions, seed, and QC settings;
    ``fp_params`` supplies the shared scaler/PCA/UMAP implementation settings.
    Stage 2A dimensions and seed override their first-pass counterparts.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # enforce analysis_id order from meta_stage1
    ids = meta_stage1[ID_COL].astype(str).tolist()
    X = X.copy()
    X.index = X.index.astype(str)
    X_f = X.reindex(ids)

    if X_f.isna().any(axis=None):
        # should not happen if meta_stage1 was derived from X
        bad = X_f.index[X_f.isna().all(axis=1)].tolist()[:10]
        raise ValueError(f"[{root}] Stage2A: X_f has missing rows. Examples: {bad}")

    Xn = l2_normalize_rows(X_f)

    fit_params = replace(
        fp_params,
        n_pcs=int(params.n_pcs),
        random_state=int(params.random_state),
    )
    # UMAP is fitted by the shared transformation function but only PCs are
    # retained as the Stage 2A comparison space.
    _sc, _pca, pcs_df, _u, _emb = fit_pca_umap(
        Xn,
        params=fit_params,
        feature_scaling=str(params.feature_scaling),
    )

    pcs_path = out_dir / f"{root}_PCs.parquet"
    pcs_df.to_parquet(pcs_path)

    eta2_df = compute_per_pc_platform_eta2(pcs_df, meta_stage1, platform_col="platform_family_refined")
    eta2_path = out_dir / f"{root}_platform_eta2.tsv"
    eta2_df.to_csv(eta2_path, sep="\t", index=False, na_rep="NaN")

    balacc = platform_predictability(pcs_df, meta_stage1, n_pcs=int(params.plat_eval_pcs), min_per_class=int(params.min_per_class))

    top_eta2 = eta2_df.sort_values(["platform_eta2", "pc_index"], ascending=[False, True]).iloc[0] if not eta2_df.empty else None

    return {
        "icdot_root": root,
        "n_samples": int(X_f.shape[0]),
        "n_features": int(X_f.shape[1]),
        "n_components": int(pcs_df.shape[1]),
        "feature_scaling": str(params.feature_scaling),
        "platform_balacc_after_stage2A_L2": float(balacc) if balacc == balacc else float("nan"),
        "platform_eta2_max": float(eta2_df["platform_eta2"].max()) if not eta2_df.empty else float("nan"),
        "platform_eta2_median": float(eta2_df["platform_eta2"].median()) if not eta2_df.empty else float("nan"),
        "platform_eta2_pc1": float(eta2_df.iloc[0]["platform_eta2"]) if not eta2_df.empty else float("nan"),
        "top_platform_pc": str(top_eta2["pc"]) if top_eta2 is not None else "",
        "top_platform_pc_eta2": float(top_eta2["platform_eta2"]) if top_eta2 is not None else float("nan"),
        "pcs_file": pcs_path.name,
        "platform_eta2_file": eta2_path.name,
    }
