from __future__ import annotations

"""Export fitted transformations and final coordinates for atlas reuse.

The exported scaler, PCA model, ordered feature list, optional Stage 2B rule,
final sample coordinates, and aligned metadata are the preprocessing artifacts
needed by downstream subtype discovery and fixed-atlas projection.
"""

import json
from datetime import datetime
from pathlib import Path
from shutil import copy2

import joblib
import pandas as pd

from cnvsim.config import PreprocessConfig
from cnvsim.preprocess.firstpass_filter import FirstPassParams, fit_pca_umap
from cnvsim.preprocess.scaling import l2_normalize_rows


def export_stage2a_models(*, run_dir: Path, cfg: PreprocessConfig, out_models_dir: Path, roots: list[str] | None = None) -> Path:
    """Refit and serialize each root's scaler and PCA transformation.

    Models are fitted to the same retained samples, feature order, L2
    normalization, PCA dimensions, and random seed used for Stage 2A. The
    ordered feature list is stored beside each model so external profiles can
    be encoded and aligned before transformation.

    Returns the model manifest path.
    """

    feat_dir = Path(run_dir) / "features_unfiltered"
    stage1_dir = Path(run_dir) / "final_space" / "stage1"

    out_models_dir.mkdir(parents=True, exist_ok=True)

    man1 = stage1_dir / "firstpass_stage1_manifest.tsv"
    if roots is None:
        if man1.exists():
            roots = pd.read_csv(man1, sep="\t")["root"].astype(str).tolist()
        else:
            roots = sorted({p.name.replace("_meta_stage1.tsv", "") for p in stage1_dir.glob("*_meta_stage1.tsv")})

    roots = [r for r in roots if r not in set(cfg.filters.excluded_roots)]

    n_pcs_save = int(cfg.stage2a.n_pcs)

    fp_params = FirstPassParams(
        random_state=int(cfg.stage2a.random_state),
        n_pcs=int(n_pcs_save),
        umap_n_neighbors=int(cfg.firstpass.umap.n_neighbors),
        umap_min_dist=float(cfg.firstpass.umap.min_dist),
        umap_metric=str(cfg.firstpass.umap.metric),
        hdb_min_cluster_size=int(cfg.firstpass.hdbscan.min_cluster_size),
        hdb_min_samples=int(cfg.firstpass.hdbscan.min_samples),
        max_rel_cluster_size=float(cfg.firstpass.technical.max_rel_cluster_size),
        min_platform_dominance=float(cfg.firstpass.technical.min_platform_dominance),
        max_top_morph_frac=float(cfg.firstpass.technical.max_top_morph_frac),
    )

    rows = []
    for root in roots:
        x_path = feat_dir / f"{root}_features.parquet"
        meta1_path = stage1_dir / f"{root}_meta_stage1.tsv"
        if not x_path.exists() or not meta1_path.exists():
            continue

        X = pd.read_parquet(x_path)
        meta1 = pd.read_csv(meta1_path, sep="\t", dtype=str, low_memory=False)

        X.index = X.index.astype(str)
        ids = meta1["analysis_id"].astype(str).tolist()
        X_f = X.reindex(ids)
        Xn = l2_normalize_rows(X_f)

        scaler, pca, _pcs_df, _reducer, _emb_df = fit_pca_umap(
            Xn,
            params=fp_params,
            feature_scaling=str(cfg.stage2a.feature_scaling),
        )

        scaler_path = out_models_dir / f"{root}_scaler.joblib"
        pca_path = out_models_dir / f"{root}_pca.joblib"
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)

        feat_cols = list(Xn.columns)
        feat_path = out_models_dir / f"{root}_feature_columns.json"
        feat_path.write_text(json.dumps(feat_cols))

        meta_obj = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "icdot_root": root,
            "n_samples_fit": int(Xn.shape[0]),
            "n_features_fit": int(Xn.shape[1]),
            "n_pcs_fit": int(pca.n_components_),
            "feature_scaling": str(cfg.stage2a.feature_scaling),
            "feature_columns_file": feat_path.name,
            "scaler_file": scaler_path.name,
            "pca_file": pca_path.name,
            "note": (
                "Stage2A: L2-normalized features -> "
                f"{cfg.stage2a.feature_scaling} feature scaling -> PCA. "
                "Stage2B is applied later."
            ),
        }
        model_meta_path = out_models_dir / f"{root}_model_meta.json"
        model_meta_path.write_text(json.dumps(meta_obj, indent=2))

        rows.append(
            {
                "icdot_root": root,
                "scaler_path": str(scaler_path),
                "pca_path": str(pca_path),
                "feature_columns_path": str(feat_path),
                "model_meta_path": str(model_meta_path),
                "n_samples_fit": int(Xn.shape[0]),
                "n_features_fit": int(Xn.shape[1]),
                "n_pcs_fit": int(pca.n_components_),
                "feature_scaling": str(cfg.stage2a.feature_scaling),
            }
        )

    manifest = pd.DataFrame(rows).sort_values("icdot_root").reset_index(drop=True)
    manifest_path = out_models_dir / "models_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    (out_models_dir / "models_stage2A_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(Path(run_dir).resolve()),
                "n_pcs_save": int(n_pcs_save),
                "params": fp_params.__dict__,
                "outputs": {"manifest": str(manifest_path)},
            },
            indent=2,
        )
    )

    return manifest_path


def export_reference_space(
    *,
    cfg: PreprocessConfig,
    run_dir: Path,
    release_dir: Path,
    roots: list[str] | None = None,
    include_final_space: bool = True,
) -> Path:
    """Assemble projection models and the canonical comparison space.

    ``include_final_space`` copies final PC coordinates and aligned metadata.
    Returns the created ``reference_space`` directory.
    """

    run_dir = Path(run_dir)
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = release_dir / "reference_space"
    ref_dir.mkdir(parents=True, exist_ok=True)

    models_dir = ref_dir / "models_stage2A"

    man_path = export_stage2a_models(run_dir=run_dir, cfg=cfg, out_models_dir=models_dir, roots=roots)

    copied = {}
    if include_final_space:
        src_final = run_dir / "final_space"
        src_final_dir_candidates = [src_final / "final", src_final]
        src_final_dir = next((p for p in src_final_dir_candidates if p.exists()), None)
        src_meta_dir = src_final / "meta_final"

        if src_final_dir is None:
            raise FileNotFoundError(f"Missing final space dir (checked: {src_final_dir_candidates})")
        if not src_meta_dir.exists():
            raise FileNotFoundError(f"Missing meta_final dir: {src_meta_dir}")

        dst_final = ref_dir / "final_space"
        dst_final.mkdir(parents=True, exist_ok=True)
        dst_meta = dst_final / "meta_final"
        dst_meta.mkdir(parents=True, exist_ok=True)

        for src in [src_final_dir / "manifest_final.tsv"]:
            if src.exists():
                copy2(src, dst_final / src.name)

        for p in src_final_dir.glob("*_PCs_final.parquet"):
            copy2(p, dst_final / p.name)

        for p in src_meta_dir.glob("*_meta_final.tsv"):
            copy2(p, dst_meta / p.name)

        copied = {
            "final_space_dir": str(dst_final),
            "final_space_files": [p.name for p in dst_final.glob("*")],
            "meta_final_files": [p.name for p in dst_meta.glob("*")],
        }

    (ref_dir / "reference_space_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir.resolve()),
                "release_dir": str(release_dir.resolve()),
                "reference_space_dir": str(ref_dir),
                "options": {
                    "include_final_space": bool(include_final_space),
                },
                "outputs": {
                    "models_manifest": str(man_path),
                    "models_dir": str(models_dir),
                    **({"final_space": copied} if copied else {}),
                },
            },
            indent=2,
        )
    )

    return ref_dir
