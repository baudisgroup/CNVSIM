from __future__ import annotations

"""Run the root-specific preprocessing stages and write their artifacts.

The Progenetix adapter first writes ``run_dir/meta_cleaned.tsv``. This module
then performs the following ordered steps:

1. ``run_encode`` writes root-specific sample-by-gene-event matrices.
2. ``run_firstpass_stage1`` detects and removes likely technical sample groups.
3. ``run_stage2a_all`` fits L2-normalized PCA comparison spaces.
4. ``materialize_final_space_from_stage2a`` publishes those coordinates as
   the canonical comparison space.

Numerical settings are read from ``PreprocessConfig``. Optional ``roots``
arguments restrict a run without changing the configured root exclusions.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.config import PreprocessConfig


def write_pcs_with_analysis_id(pcs: pd.DataFrame, path: Path) -> Path:
    """Write PC coordinates with analysis_id as an explicit parquet column."""

    out = pcs.copy()
    if "analysis_id" in out.columns:
        out["analysis_id"] = out["analysis_id"].astype(str)
    else:
        out.index = out.index.astype(str)
        out.index.name = "analysis_id"
        out = out.reset_index()

    if out["analysis_id"].duplicated().any():
        raise ValueError(f"Duplicate analysis_id values while writing {path}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def run_encode(
    *,
    cfg: PreprocessConfig,
    run_dir: Path,
    gene_fraction_store: Path,
    roots: list[str] | None = None,
) -> Path:
    """Write one unfiltered gene-event feature matrix per anatomical root.

    Sample IDs and root labels come from ``meta_cleaned.tsv``. Gene-level CNV
    rows are read from the gene-CNV fraction store in the source package.
    Encoding and filtering thresholds come from ``cfg.encoding``. Outputs and
    their dimensions are recorded under ``features_unfiltered/``.

    Returns the path to ``encode_manifest.tsv``.
    """

    from cnvsim.preprocess.encode import (
        FEATURE_BURDEN_COLS,
        EncodeParams,
        build_feature_matrix_for_root,
    )
    from cnvsim.preprocess.gene_panel_store import (
        load_gene_panel_subset_from_store,
    )

    run_dir = Path(run_dir)
    meta_path = run_dir / "meta_cleaned.tsv"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_cleaned.tsv not found: {meta_path}")

    bios = pd.read_csv(meta_path, sep="\t", dtype={"analysis_id": str}, low_memory=False)

    store_dir = Path(gene_fraction_store).expanduser().resolve()
    if not store_dir.is_dir():
        raise FileNotFoundError(f"Gene-CNV fraction store not found: {store_dir}")

    params = EncodeParams(
        min_samples_per_family=int(cfg.encoding.min_samples_per_family),
        min_altered_genes_per_sample=int(
            cfg.encoding.min_altered_genes_per_sample
        ),
        stable_gene_min_coverage=cfg.encoding.stable_gene_coverage,
        min_frac_low=cfg.encoding.low_frac_threshold,
        min_frac_high=cfg.encoding.high_frac_threshold,
    )

    if roots is None:
        roots = sorted(set(bios["icdot_root_grouped"].dropna().astype(str).unique().tolist()))

    roots = [r for r in roots if r not in set(cfg.filters.excluded_roots)]

    out_dir = run_dir / "features_unfiltered"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for root in roots:
        ids = bios.loc[bios["icdot_root_grouped"].astype(str) == str(root), "analysis_id"].astype(str).unique().tolist()
        if not ids:
            continue

        gene_panel = load_gene_panel_subset_from_store(
            store_dir=store_dir,
            analysis_ids=ids,
            gene_symbols=None,
            columns=["analysis_id", "gene_symbol", "chrom", "dup_frac", "del_frac", "hldup_frac", "hldel_frac"],
        )

        if gene_panel is None or gene_panel.empty:
            continue

        X_root, meta_root, info = build_feature_matrix_for_root(
            icdot_root=root,
            bios_df=bios,
            gene_df=gene_panel,
            min_samples_per_family=int(params.min_samples_per_family),
            min_altered_genes_per_sample=int(params.min_altered_genes_per_sample),
            stable_gene_min_coverage=params.stable_gene_min_coverage,
            min_frac_low=float(params.min_frac_low),
            min_frac_high=float(params.min_frac_high),
            autosomes_only=bool(getattr(cfg.encoding, "autosomes_only", True)),
        )

        if X_root is None:
            continue

        x_path = out_dir / f"{root}_features.parquet"
        meta_path_out = out_dir / f"{root}_meta.tsv"
        burden_path = out_dir / f"{root}_sample_burden.tsv"
        info_path = out_dir / f"{root}_info.json"

        X_root.to_parquet(x_path)
        meta_root.to_csv(meta_path_out, sep="\t", index=False)
        meta_root[["analysis_id", *FEATURE_BURDEN_COLS]].to_csv(
            burden_path,
            sep="\t",
            index=False,
        )
        info_path.write_text(json.dumps(info, indent=2))

        manifest_rows.append(
            {
                "icdot_root": root,
                "n_samples": int(X_root.shape[0]),
                "n_features": int(X_root.shape[1]),
                "n_zero_altered_genes": int(
                    (~meta_root["architecture_informative"].astype(bool)).sum()
                ),
                "features_file": x_path.name,
                "meta_file": meta_path_out.name,
                "sample_burden_file": burden_path.name,
                "info_file": info_path.name,
            }
        )

    if not manifest_rows:
        selected = "all configured roots" if roots is None else ", ".join(roots)
        raise ValueError(
            "No root feature matrices were produced for "
            f"{selected}; verify metadata/cache compatibility"
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["n_samples"], ascending=False
    )
    out_path = out_dir / "encode_manifest.tsv"
    manifest.to_csv(out_path, sep="\t", index=False)
    return out_path


def run_firstpass_stage1(*, cfg: PreprocessConfig, run_dir: Path, roots: list[str] | None = None) -> Path:
    """Remove likely platform-driven sample groups and refit each root space.

    The function consumes ``features_unfiltered`` and applies the thresholds in
    ``cfg.firstpass``. It stores pre-filter diagnostics, removed analysis IDs,
    post-filter metadata, and post-filter PCA/UMAP coordinates under
    ``final_space/stage1``. The retained metadata are also copied to
    ``final_space/meta_final`` for downstream clustering.

    Returns the Stage 1 manifest path.
    """

    from cnvsim.preprocess.firstpass_filter import FirstPassParams, ensure_aligned, run_firstpass_for_root, fit_pca_umap, LABEL_COL
    from cnvsim.preprocess.matrix_utils import platform_predictability, knn_label_purity_fixed

    run_dir = Path(run_dir)
    feat_dir = run_dir / "features_unfiltered"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing features dir: {feat_dir}")

    if roots is None:
        roots = sorted({p.name.replace("_features.parquet", "") for p in feat_dir.glob("*_features.parquet")})

    roots = [r for r in roots if r not in set(cfg.filters.excluded_roots)]

    params = FirstPassParams(
        random_state=int(cfg.firstpass.random_state),
        n_pcs=int(cfg.firstpass.n_pcs),
        umap_n_neighbors=int(cfg.firstpass.umap.n_neighbors),
        umap_min_dist=float(cfg.firstpass.umap.min_dist),
        umap_metric=str(cfg.firstpass.umap.metric),
        hdb_min_cluster_size=int(cfg.firstpass.hdbscan.min_cluster_size),
        hdb_min_samples=int(cfg.firstpass.hdbscan.min_samples),
        max_rel_cluster_size=float(cfg.firstpass.technical.max_rel_cluster_size),
        min_platform_dominance=float(cfg.firstpass.technical.min_platform_dominance),
        max_top_morph_frac=float(cfg.firstpass.technical.max_top_morph_frac),
    )

    out_dir = run_dir / "final_space" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_meta_dir = run_dir / "final_space" / "meta_final"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    qc_rows = []
    meta_manifest_rows = []

    for root in roots:
        x_path = feat_dir / f"{root}_features.parquet"
        m_path = feat_dir / f"{root}_meta.tsv"
        if not x_path.exists() or not m_path.exists():
            continue

        X = pd.read_parquet(x_path)
        X.index = X.index.astype(str)
        meta = pd.read_csv(m_path, sep="\t", dtype=str, low_memory=False)

        first = run_firstpass_for_root(root, X, meta, params=params)
        clus = first["cluster_report"]
        bad_ids = first["bad_ids"]
        labels = first["labels"]

        pcs_pre_path = out_dir / f"{root}_PCs_firstpass.parquet"
        first["pcs"].to_parquet(pcs_pre_path)

        umap_pre_path = out_dir / f"{root}_UMAP_firstpass.parquet"
        first["emb"].assign(cluster_id=labels).to_parquet(umap_pre_path)

        rep_path = out_dir / f"{root}_cluster_report.tsv"
        clus.to_csv(rep_path, sep="\t", index=False)

        removed_path = out_dir / f"{root}_removed_samples.tsv"
        pd.DataFrame({"analysis_id": bad_ids}).to_csv(removed_path, sep="\t", index=False)

        keep_mask = ~X.index.astype(str).isin(set(bad_ids))
        X_f = X.loc[keep_mask].copy()
        meta_f = ensure_aligned(X_f, meta)

        meta_stage1_path = out_dir / f"{root}_meta_stage1.tsv"
        meta_f.to_csv(meta_stage1_path, sep="\t", index=False)

        _sc2, _p2, pcs_post, _u2, _e2 = fit_pca_umap(X_f, params=params)

        pcs_post_path = out_dir / f"{root}_PCs_finalpass.parquet"
        pcs_post.to_parquet(pcs_post_path)

        umap_post_path = out_dir / f"{root}_UMAP_finalpass.parquet"
        _e2.to_parquet(umap_post_path)

        acc_pre = platform_predictability(first["pcs"], first["meta"], platform_col="platform_family_refined", n_pcs=10, min_per_class=50)
        pur_pre = knn_label_purity_fixed(first["pcs"], first["meta"], LABEL_COL, k=15, n_pcs=10)
        acc_post = platform_predictability(pcs_post, meta_f, platform_col="platform_family_refined", n_pcs=10, min_per_class=50)
        pur_post = knn_label_purity_fixed(pcs_post, meta_f, LABEL_COL, k=15, n_pcs=10)

        qc_rows.append(
            {
                "icdot_root": root,
                "n_pre": int(X.shape[0]),
                "n_post": int(X_f.shape[0]),
                "drop_frac": float(1.0 - (X_f.shape[0] / max(1, X.shape[0]))),
                "platform_balacc_pre": acc_pre,
                "platform_balacc_post": acc_post,
                "delta_platform_balacc": (acc_post - acc_pre) if np.isfinite(acc_pre) and np.isfinite(acc_post) else np.nan,
                "knn_purity_pre": pur_pre,
                "knn_purity_post": pur_post,
                "delta_knn_purity": (pur_post - pur_pre) if np.isfinite(pur_pre) and np.isfinite(pur_post) else np.nan,
                "n_technical_clusters": int(clus["is_technical"].sum()),
            }
        )

        manifest_rows.append(
            {
                "root": root,
                "n_pre": int(X.shape[0]),
                "n_post": int(X_f.shape[0]),
                "drop_frac": float(1.0 - (X_f.shape[0] / max(1, X.shape[0]))),
                "n_technical_clusters": int(clus["is_technical"].sum()),
                "pcs_firstpass": str(pcs_pre_path),
                "umap_firstpass": str(umap_pre_path),
                "cluster_report": str(rep_path),
                "removed_samples": str(removed_path),
                "meta_stage1": str(meta_stage1_path),
                "pcs_finalpass": str(pcs_post_path),
                "umap_finalpass": str(umap_post_path),
            }
        )

        m = meta_f.copy()
        if "analysis_id" not in m.columns:
            if m.index.name == "analysis_id":
                m = m.reset_index()
            else:
                m = m.reset_index().rename(columns={"index": "analysis_id"})
        m["analysis_id"] = m["analysis_id"].astype(str)
        m = m.drop_duplicates(subset=["analysis_id"], keep="first")

        meta_final_path = out_meta_dir / f"{root}_meta_final.tsv"
        m.to_csv(meta_final_path, sep="\t", index=False)

        meta_manifest_rows.append(
            {
                "icdot_root": root,
                "meta_file": meta_final_path.name,
                "n_samples_meta": int(m.shape[0]),
                "columns": ",".join(m.columns),
            }
        )

    qc_df = pd.DataFrame(qc_rows).sort_values(["delta_platform_balacc", "drop_frac"], ascending=[True, False]).reset_index(drop=True)
    qc_path = out_dir / "qc_stage1.tsv"
    qc_df.to_csv(qc_path, sep="\t", index=False, na_rep="NaN")

    man_df = pd.DataFrame(manifest_rows).sort_values(["drop_frac"], ascending=False)
    man_path = out_dir / "firstpass_stage1_manifest.tsv"
    man_df.to_csv(man_path, sep="\t", index=False)

    meta_manifest = pd.DataFrame(meta_manifest_rows).sort_values("icdot_root").reset_index(drop=True)
    meta_manifest_path = out_meta_dir / "meta_manifest.tsv"
    meta_manifest.to_csv(meta_manifest_path, sep="\t", index=False)

    (out_dir / "firstpass_stage1_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "params": params.__dict__,
                "outputs": {"qc": str(qc_path), "manifest": str(man_path), "meta_manifest": str(meta_manifest_path)},
            },
            indent=2,
        )
    )

    return man_path


def run_stage2a_all(*, cfg: PreprocessConfig, run_dir: Path, roots: list[str] | None = None) -> Path:
    """Fit and evaluate the post-filter PCA comparison space for every root.

    Feature matrices come from ``features_unfiltered`` and retained sample order
    comes from ``final_space/stage1``. Stage 2A dimensions and QC thresholds are
    read from ``cfg.stage2a``; shared UMAP settings come from ``cfg.firstpass``.
    Returns ``final_space/stage2A/manifest.tsv``.
    """

    from cnvsim.preprocess.firstpass_filter import FirstPassParams
    from cnvsim.preprocess.stage2a import Stage2AParams, run_stage2a_for_root

    run_dir = Path(run_dir)
    feat_dir = run_dir / "features_unfiltered"
    stage1_dir = run_dir / "final_space" / "stage1"

    if roots is None:
        man1 = stage1_dir / "firstpass_stage1_manifest.tsv"
        if man1.exists():
            roots = pd.read_csv(man1, sep="\t")["root"].astype(str).tolist()
        else:
            roots = sorted({p.name.replace("_meta_stage1.tsv", "") for p in stage1_dir.glob("*_meta_stage1.tsv")})

    roots = [r for r in roots if r not in set(cfg.filters.excluded_roots)]

    fp_params = FirstPassParams(
        random_state=int(cfg.stage2a.random_state),
        n_pcs=int(cfg.stage2a.n_pcs),
        umap_n_neighbors=int(cfg.firstpass.umap.n_neighbors),
        umap_min_dist=float(cfg.firstpass.umap.min_dist),
        umap_metric=str(cfg.firstpass.umap.metric),
        hdb_min_cluster_size=int(cfg.firstpass.hdbscan.min_cluster_size),
        hdb_min_samples=int(cfg.firstpass.hdbscan.min_samples),
        max_rel_cluster_size=float(cfg.firstpass.technical.max_rel_cluster_size),
        min_platform_dominance=float(cfg.firstpass.technical.min_platform_dominance),
        max_top_morph_frac=float(cfg.firstpass.technical.max_top_morph_frac),
    )

    params = Stage2AParams(
        n_pcs=int(cfg.stage2a.n_pcs),
        plat_eval_pcs=int(cfg.stage2a.plat_eval_pcs),
        min_per_class=int(cfg.stage2a.min_per_class),
        random_state=int(cfg.stage2a.random_state),
        feature_scaling=str(cfg.stage2a.feature_scaling),
    )

    out_dir = run_dir / "final_space" / "stage2A"
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []
    manifest_rows = []

    for root in roots:
        x_path = feat_dir / f"{root}_features.parquet"
        meta1_path = stage1_dir / f"{root}_meta_stage1.tsv"
        if not x_path.exists() or not meta1_path.exists():
            continue

        X = pd.read_parquet(x_path)
        meta1 = pd.read_csv(meta1_path, sep="\t", dtype=str, low_memory=False)

        stats = run_stage2a_for_root(root, X, meta1, out_dir, params=params, fp_params=fp_params)
        qc_rows.append(
            {
                "icdot_root": root,
                "feature_scaling": str(stats["feature_scaling"]),
                "platform_balacc_after_stage2A_L2": stats["platform_balacc_after_stage2A_L2"],
                "platform_eta2_max": stats.get("platform_eta2_max", np.nan),
                "platform_eta2_median": stats.get("platform_eta2_median", np.nan),
                "platform_eta2_pc1": stats.get("platform_eta2_pc1", np.nan),
                "top_platform_pc": stats.get("top_platform_pc", ""),
                "top_platform_pc_eta2": stats.get("top_platform_pc_eta2", np.nan),
                "platform_eta2_file": stats.get("platform_eta2_file", ""),
            }
        )
        manifest_rows.append(stats)

    qc_df = pd.DataFrame(qc_rows).sort_values(
        ["platform_balacc_after_stage2A_L2", "platform_eta2_max"],
        ascending=[False, False],
    )
    qc_path = out_dir / "qc_stage2A.tsv"
    qc_df.to_csv(qc_path, sep="\t", index=False, na_rep="NaN")

    man_df = pd.DataFrame(manifest_rows).sort_values("n_samples", ascending=False)
    man_path = out_dir / "manifest.tsv"
    man_df.to_csv(man_path, sep="\t", index=False)

    return man_path


def materialize_final_space_from_stage2a(*, cfg: PreprocessConfig, run_dir: Path, roots: list[str] | None = None) -> Path:
    """Publish uncorrected Stage 2A PCs as the final baseline coordinates."""

    run_dir = Path(run_dir)
    stage2a_dir = run_dir / "final_space" / "stage2A"
    stage1_dir = run_dir / "final_space" / "stage1"

    if roots is None:
        man1 = stage1_dir / "firstpass_stage1_manifest.tsv"
        if man1.exists():
            roots = pd.read_csv(man1, sep="\t")["root"].astype(str).tolist()
        else:
            roots = sorted({p.name.replace("_PCs.parquet", "") for p in stage2a_dir.glob("*_PCs.parquet")})

    roots = [r for r in roots if r not in set(cfg.filters.excluded_roots)]

    final_dir = run_dir / "final_space" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for r in roots:
        pcs_path = stage2a_dir / f"{r}_PCs.parquet"
        if not pcs_path.exists():
            continue

        pcs = pd.read_parquet(pcs_path)
        if "analysis_id" in pcs.columns:
            pcs = pcs.set_index("analysis_id")
        pcs.index = pcs.index.astype(str)

        final_path = final_dir / f"{r}_PCs_final.parquet"
        write_pcs_with_analysis_id(pcs, final_path)

        manifest_rows.append({"icdot_root": r, "final_pcs_file": final_path.name, "final_choice": "stage2A"})

    manifest_df = pd.DataFrame(manifest_rows).sort_values("icdot_root")
    manifest_path = final_dir / "manifest_final.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    return manifest_path
