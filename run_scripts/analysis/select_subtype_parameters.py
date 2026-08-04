from __future__ import annotations

"""Select root-specific graph and Leiden parameters for the CNVSIM atlas."""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from cnvsim.analysis.config import AnalysisConfig, load_analysis_config
from cnvsim.analysis.knn_parameter import (
    KNNGrid,
    Stage1Thresholds,
    Stage2Leiden,
    scan_root_parameters_from_ir,
)
from cnvsim.analysis.leiden import LeidenParams, make_resolution_grids
from cnvsim.analysis.partition_assessment import assess_selected_partitions
from cnvsim.analysis.subsampling_selection import (
    extract_plateau_candidates,
    finalize_existing_subsampling_selection,
    run_subsampling_model_selection,
)


def _parse_roots(value: str) -> list[str]:
    return [root.strip() for root in str(value).split(",") if root.strip()]


def _analysis_roots(ir_dir: Path, requested: list[str], excluded: tuple[str, ...]) -> list[str]:
    perroot_dir = ir_dir / "perroot"
    if not perroot_dir.exists():
        raise FileNotFoundError(f"Missing analysis IR: {perroot_dir}")
    roots = requested or sorted(path.name for path in perroot_dir.iterdir() if path.is_dir())
    roots = [root for root in roots if root not in set(excluded)]
    if not roots:
        raise ValueError("No analysis roots remain after applying excluded_roots")
    missing = sorted(root for root in roots if not (perroot_dir / root).is_dir())
    if missing:
        raise ValueError(f"Requested roots are missing from the analysis IR: {missing}")
    return roots


def _scan_parameters(
    *,
    ir_dir: Path,
    out_dir: Path,
    roots: list[str],
    config: AnalysisConfig,
) -> dict[str, Path]:
    selection = config.knn_selection
    grid = KNNGrid(
        metrics=tuple(selection.metrics),
        n_pcs_list=tuple(selection.n_pcs),
        k_list=tuple(selection.k_values),
    )
    stage1 = Stage1Thresholds(
        min_giant_frac=float(selection.min_giant_frac),
        max_components=int(selection.max_components),
        max_platform_enrichment=selection.max_platform_enrichment,
        platform_enrichment_tolerance=selection.platform_enrichment_tolerance,
        preferred_metric_order=tuple(selection.preferred_metrics),
        preferred_k_order=tuple(selection.preferred_k),
        preferred_n_pcs=tuple(selection.preferred_n_pcs),
        max_morph_purity_drop=selection.max_morph_purity_drop,
    )
    stage2 = Stage2Leiden(
        enabled=bool(selection.leiden_scan_enabled),
        top_n=int(selection.leiden_scan_top_n),
        sigma_quantile=float(selection.sigma_quantile),
        make_undirected=bool(selection.make_undirected),
        combine_mode=str(selection.combine_mode),
    )
    lcfg = config.leiden
    leiden = LeidenParams(
        seeds=tuple(lcfg.seeds),
        min_cluster_size=int(lcfg.min_cluster_size),
        max_frac_in_small=float(lcfg.max_frac_in_small),
        max_giant_frac=float(lcfg.max_giant_frac),
        min_ari=float(lcfg.min_ari),
        selection_score_tolerance=float(lcfg.selection_score_tolerance),
        selection_policy=str(lcfg.selection_policy),
        min_resolution_neighbor_ari=float(lcfg.min_resolution_neighbor_ari),
        min_plateau_points=int(lcfg.min_plateau_points),
        platform_assoc_weight=float(lcfg.platform_assoc_weight),
        platform_assoc_max=lcfg.platform_assoc_max,
    )
    resolution_grid, n_fine = make_resolution_grids(
        r_min=float(config.resolution_grid.r_min),
        r_max=float(config.resolution_grid.r_max),
        n_broad=int(config.resolution_grid.n_broad),
        n_fine=int(config.resolution_grid.n_fine),
    )

    preliminary, scan = scan_root_parameters_from_ir(
        ir_dir=ir_dir,
        roots=roots,
        platform_col="platform_family_refined",
        morph_col="icdo_morphology_id",
        grid=grid,
        stage1=stage1,
        stage2=stage2,
        leiden_params=leiden,
        resolution_grid=np.asarray(resolution_grid, dtype=float),
        fine_grid_points=int(n_fine),
    )
    selected_roots = set(preliminary["root"].astype(str)) if not preliminary.empty else set()
    missing = sorted(set(roots) - selected_roots)
    if missing:
        raise RuntimeError(
            "Parameter scanning produced no feasible preliminary selection for roots: "
            + ", ".join(missing)
        )
    preliminary = preliminary.sort_values("root").reset_index(drop=True)
    preliminary["selection_status"] = "preliminary_selected"

    out_dir.mkdir(parents=True, exist_ok=True)
    preliminary_path = out_dir / "preliminary_parameters.tsv"
    scan_path = out_dir / "parameter_scan.tsv"
    preliminary.to_csv(preliminary_path, sep="\t", index=False)
    scan.to_csv(scan_path, sep="\t", index=False)
    manifest_path = out_dir / "scan_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "ir_dir": str(ir_dir.resolve()),
                "roots": roots,
                "counts": {
                    "roots": len(roots),
                    "graph_settings_scanned_per_root": (
                        len(grid.metrics) * len(grid.n_pcs_list) * len(grid.k_list)
                    ),
                    "graph_settings_with_leiden_scan_per_root": int(stage2.top_n),
                },
                "outputs": {
                    "preliminary_parameters": str(preliminary_path),
                    "parameter_scan": str(scan_path),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "preliminary": preliminary_path,
        "scan": scan_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan root-specific graph/resolution settings and select the final "
            "parameters by repeated subsampling, then assess selected partitions."
        )
    )
    parser.add_argument("--ir-base-dir", required=True)
    parser.add_argument("--ir-subdir", default="analysis/ir")
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stage",
        choices=["scan", "select", "finalize", "assess", "all"],
        default="all",
    )
    parser.add_argument("--roots", default="", help="Optional comma-separated roots.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--max-candidates-per-root", type=int, default=None)
    parser.add_argument(
        "--assessment-silhouette-sample-size",
        type=int,
        default=3000,
        help="Maximum samples per root used for exact silhouette diagnostics.",
    )
    parser.add_argument(
        "--shape-proxies",
        default="",
        help=(
            "Optional analysis_shape_proxies.parquet used to annotate assessment "
            "partitions with CNV fraction and event-count burden."
        ),
    )
    parser.add_argument(
        "--extract-candidates-only",
        action="store_true",
        help="Stop after writing candidate_plateaus.tsv from an existing or new scan.",
    )
    args = parser.parse_args()

    config = load_analysis_config(args.analysis_config)
    ir_dir = Path(args.ir_base_dir) / str(args.ir_subdir)
    out_dir = Path(args.out_dir)
    requested_roots = _parse_roots(args.roots)
    roots = _analysis_roots(ir_dir, requested_roots, config.excluded_roots)

    if args.stage in {"scan", "all"}:
        outputs = _scan_parameters(
            ir_dir=ir_dir,
            out_dir=out_dir,
            roots=roots,
            config=config,
        )
        print(f"[CNVSIM] preliminary parameters: {outputs['preliminary']}")
        print(f"[CNVSIM] parameter scan: {outputs['scan']}")

    if args.stage == "scan":
        return

    subsampling = config.subsampling
    overrides = {}
    if args.repeats is not None:
        overrides["repeats"] = int(args.repeats)
    if args.sample_fraction is not None:
        overrides["sample_fraction"] = float(args.sample_fraction)
    if args.max_candidates_per_root is not None:
        overrides["max_candidates_per_root"] = int(args.max_candidates_per_root)
    if overrides:
        subsampling = replace(subsampling, **overrides)
    if not 0.0 < float(subsampling.sample_fraction) < 1.0:
        raise ValueError("sample_fraction must be between 0 and 1")
    if int(subsampling.repeats) < 2:
        raise ValueError("repeats must be at least 2")

    if args.stage == "assess":
        selected_path = out_dir / "selected_parameters.tsv"
        if not selected_path.exists():
            raise FileNotFoundError(
                "Partition assessment requires selected_parameters.tsv; run "
                "--stage select or --stage finalize first."
            )
        outputs = assess_selected_partitions(
            ir_dir=ir_dir,
            selected_parameters_path=selected_path,
            out_dir=out_dir,
            config=config,
            roots=roots if requested_roots else None,
            silhouette_sample_size=int(args.assessment_silhouette_sample_size),
            shape_proxies_path=(Path(args.shape_proxies) if args.shape_proxies else None),
        )
        print("[CNVSIM] pre-materialization partition assessment complete")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        print(
            "[CNVSIM] Assessment flags are diagnostic only; review them before "
            "materializing the released atlas."
        )
        return

    if args.stage == "finalize":
        outputs = finalize_existing_subsampling_selection(
            out_dir=out_dir,
            leiden=config.leiden,
            subsampling=subsampling,
            roots=roots if requested_roots else None,
        )
        print("[CNVSIM] final selection policy applied to existing subsampling results")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        reviews = pd.read_csv(outputs["reviews"], sep="\t")
        if not reviews.empty:
            print(
                "[CNVSIM] WARNING: reviewed roots must be resolved before "
                "selected_parameters.tsv is used for the released atlas."
            )
        return

    preliminary_path = out_dir / "preliminary_parameters.tsv"
    scan_path = out_dir / "parameter_scan.tsv"
    if not preliminary_path.exists() or not scan_path.exists():
        raise FileNotFoundError(
            "Selection requires preliminary_parameters.tsv and parameter_scan.tsv; "
            "run --stage scan first or use --stage all."
        )

    if args.extract_candidates_only:
        scan = pd.read_csv(scan_path, sep="\t", low_memory=False)
        preliminary = pd.read_csv(preliminary_path, sep="\t", low_memory=False)
        if requested_roots:
            root_set = set(roots)
            scan = scan[scan["root"].astype(str).isin(root_set)].copy()
            preliminary = preliminary[preliminary["root"].astype(str).isin(root_set)].copy()
        candidates = extract_plateau_candidates(
            scan,
            preliminary,
            leiden=config.leiden,
            subsampling=subsampling,
        )
        candidate_path = out_dir / "candidate_plateaus.tsv"
        candidates.to_csv(candidate_path, sep="\t", index=False)
        print(f"[CNVSIM] candidates: {candidate_path}")
        print(candidates.groupby("root")["candidate_id"].count().to_string())
        return

    outputs = run_subsampling_model_selection(
        parameter_scan_path=scan_path,
        preliminary_parameters_path=preliminary_path,
        ir_dir=ir_dir,
        out_dir=out_dir,
        leiden=config.leiden,
        subsampling=subsampling,
        sigma_quantile=float(config.knn_selection.sigma_quantile),
        make_undirected=bool(config.knn_selection.make_undirected),
        combine_mode=str(config.knn_selection.combine_mode),
        roots=roots if requested_roots else None,
    )
    print("[CNVSIM] parameter selection complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    reviews = pd.read_csv(outputs["reviews"], sep="\t")
    if not reviews.empty:
        print(
            "[CNVSIM] WARNING: reviewed roots must be resolved before "
            "selected_parameters.tsv is used for the released atlas."
        )


if __name__ == "__main__":
    main()
