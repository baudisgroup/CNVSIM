from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return bool(default)


def _as_int_tuple(x: Any, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    if x is None:
        return default
    if isinstance(x, (tuple, list)):
        return tuple(int(v) for v in x)
    # allow strings like "0,1,2" or "(0, 1, 2)"
    s = str(x).strip().strip("()[]")
    if not s:
        return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _as_str_tuple(x: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if x is None:
        return default
    if isinstance(x, (tuple, list)):
        return tuple(str(v).strip() for v in x if str(v).strip())
    parts = [p.strip() for p in str(x).strip().strip("()[]").split(",") if p.strip()]
    return tuple(parts) if parts else default


@dataclass(frozen=True)
class IRBuildParams:
    n_dims: int = 50


@dataclass(frozen=True)
class KNNBuildParams:
    K: int = 50
    metric: str = "euclidean"
    n_pcs: int = 10


@dataclass(frozen=True)
class KNNSelectionParams:
    metrics: tuple[str, ...] = ("euclidean", "cosine")
    n_pcs: tuple[int, ...] = (10, 20, 30, 50)
    k_values: tuple[int, ...] = (30, 50, 80)

    min_giant_frac: float = 0.99
    max_components: int = 2
    max_platform_enrichment: float | None = None
    platform_enrichment_tolerance: float | None = 0.05
    preferred_metrics: tuple[str, ...] = ("cosine", "euclidean")
    preferred_k: tuple[int, ...] = (80, 50, 30)
    preferred_n_pcs: tuple[int, ...] = (30, 50)
    max_morph_purity_drop: float | None = None

    leiden_scan_enabled: bool = True
    leiden_scan_top_n: int = 6
    sigma_quantile: float = 0.90
    make_undirected: bool = True
    combine_mode: str = "max"


@dataclass(frozen=True)
class LeidenParams:
    # seeds
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    # constraints
    min_cluster_size: int = 200
    # selection constraints (notebook defaults)
    max_frac_in_small: float = 0.02
    max_giant_frac: float = 0.99
    min_ari: float = 0.80

    # Treat near-optimal stability/platform scores as equivalent, then select
    # the central candidate on the log-resolution scale. This prevents small
    # score fluctuations from moving the scan to a distant resolution regime.
    selection_score_tolerance: float = 0.02

    # Resolution-selection policy. score_band_center preserves the V3 policy;
    # stable_plateau selects the centre of a contiguous resolution interval
    # whose representative partitions remain similar.
    selection_policy: str = "score_band_center"
    min_resolution_neighbor_ari: float = 0.85
    min_plateau_points: int = 2

    # A1 confounding control: penalize association between cluster labels and platform labels
    # when selecting the resolution. Association is measured via Cramér's V (0..1; lower is better).
    platform_assoc_weight: float = 0.0
    platform_assoc_max: float | None = None


@dataclass(frozen=True)
class SubsamplingSelectionParams:
    """Repeated-subsampling model selection for stable Leiden candidates."""

    sample_fraction: float = 0.80
    repeats: int = 20
    random_seed: int = 1701
    partition_seed: int = 0
    max_candidates_per_root: int = 6

    min_subsample_ari: float = 0.80
    min_matched_fraction: float = 0.80
    min_eval_coverage: float = 0.95
    min_context_recall_q10: float = 0.70
    min_worst_context_precision_q10: float = 0.50
    min_worst_context_jaccard_q10: float = 0.50
    min_cluster_count_ratio: float = 0.75
    max_cluster_count_ratio: float = 1.25
    one_se_multiplier: float = 1.0


@dataclass(frozen=True)
class ContextRefinementParams:
    """Conservative post-Leiden merging of unsupported community splits."""

    min_comerge_rate: float = 0.50
    centroid_distance_quantile: float = 0.25
    min_context_precision_q10: float = 0.50
    min_context_jaccard_q10: float = 0.50
    burden_median_delta: float = 0.10
    technical_top_fraction: float = 0.80
    technical_enrichment_ratio: float = 1.50
    track1_pair_min_abs_median_delta: float = 0.10
    track2_pair_min_abs_adjusted_delta: float = 0.10
    track2_pair_min_abs_prevalence_delta: float = 0.10
    track2_pair_veto_requires_architecture_or_high_level: bool = True
    min_final_context_size: int = 200
    min_final_contexts: int = 2


@dataclass(frozen=True)
class ReferenceContextParams:
    """Recovery requirements for using a refined context as an atlas anchor."""

    min_detection_rate: float = 0.90
    min_conditional_recall_q10: float = 0.70
    min_conditional_precision_q10: float = 0.50
    min_conditional_jaccard_q10: float = 0.50
    min_context_size: int = 200
    anchor_coverage_warning: float = 0.80
    root_only_fallback_roots: tuple[str, ...] = ("C47",)


@dataclass(frozen=True)
class ReferenceProjectionParams:
    """Calibration settings for anchor-based reference projection."""

    lower_confidence_quantile: float = 0.10
    upper_distance_quantile: float = 0.90
    min_correct_calibration_samples: int = 20
    min_context_loo_accuracy: float = 0.80
    top_candidates: int = 5


@dataclass(frozen=True)
class SignatureEvidenceParams:
    """Thresholds for Track 1 and strict Track 2 evidence."""

    min_platform_samples: int = 20
    track1_q_cut: float = 0.05
    track1_min_abs_median_delta: float = 0.10
    track1_min_abs_auc_shift: float = 0.10
    track1_min_platform_delta: float = 0.05
    track1_min_platforms_supporting: int = 2
    track2_top_n: int = 30
    track2_q_cut: float = 0.05
    track2_min_abs_prevalence_delta: float = 0.05
    track2_min_abs_auc_shift: float = 0.10
    track2_min_platforms_supporting: int = 2
    track2_min_platform_support_fraction: float = 2.0 / 3.0
    track2_max_auc_gap: float = 0.15


@dataclass(frozen=True)
class ValidationParams:
    outlier_label: int = -1
    min_samples: int = 200
    test_size: float = 0.25
    random_state: int = 0

    # marker selection
    top_k_per_cluster_event: int = 30
    q_cut: float = 0.05

    # platform checks
    do_within_platform: bool = True
    do_lopo: bool = False

    # model
    lr_C: float = 0.5
    max_iter: int = 10000
    n_jobs: int = 1


@dataclass(frozen=True)
class ResolutionGridParams:
    r_min: float = 1e-4
    r_max: float = 5e-2
    n_broad: int = 15
    n_fine: int = 12


@dataclass(frozen=True)
class AnalysisConfig:
    run_name: str
    excluded_roots: tuple[str, ...]
    ir_build: IRBuildParams
    knn_selection: KNNSelectionParams
    leiden: LeidenParams
    subsampling: SubsamplingSelectionParams
    refinement: ContextRefinementParams
    reference_contexts: ReferenceContextParams
    projection: ReferenceProjectionParams
    signatures: SignatureEvidenceParams
    validation: ValidationParams
    resolution_grid: ResolutionGridParams


def _validate_analysis_config(config: AnalysisConfig) -> None:
    """Reject internally inconsistent parameter-selection settings early."""

    errors: list[str] = []
    if config.ir_build.n_dims < max(config.knn_selection.n_pcs, default=1):
        errors.append("ir.n_dims must cover every knn.selection.n_pcs value")
    if not config.knn_selection.metrics:
        errors.append("knn.selection.metrics cannot be empty")
    if any(metric not in {"euclidean", "cosine"} for metric in config.knn_selection.metrics):
        errors.append("knn.selection.metrics must contain only euclidean or cosine")
    if any(value < 1 for value in config.knn_selection.n_pcs):
        errors.append("knn.selection.n_pcs values must be positive")
    if any(value < 2 for value in config.knn_selection.k_values):
        errors.append("knn.selection.k values must be at least 2")
    if not 0.0 < config.knn_selection.min_giant_frac <= 1.0:
        errors.append("knn.selection.min_giant_frac must be in (0, 1]")
    if config.knn_selection.max_components < 1:
        errors.append("knn.selection.max_components must be positive")
    if config.knn_selection.leiden_scan_top_n < 1:
        errors.append("knn.selection.leiden_scan_top_n must be positive")
    if config.knn_selection.combine_mode not in {"max", "mean"}:
        errors.append("knn.selection.combine_mode must be max or mean")

    if len(config.leiden.seeds) < 2:
        errors.append("leiden.seeds must contain at least two seeds")
    if config.leiden.min_cluster_size < 2:
        errors.append("leiden.min_cluster_size must be at least 2")
    for name, value in {
        "leiden.max_frac_in_small": config.leiden.max_frac_in_small,
        "leiden.max_giant_frac": config.leiden.max_giant_frac,
        "leiden.min_ari": config.leiden.min_ari,
    }.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]")
    if config.leiden.selection_policy != "stable_plateau":
        errors.append("leiden.selection_policy must be stable_plateau")
    if config.leiden.min_plateau_points < 2:
        errors.append("leiden.min_plateau_points must be at least 2")
    if config.leiden.platform_assoc_max is not None and not 0.0 <= config.leiden.platform_assoc_max <= 1.0:
        errors.append("leiden.platform_assoc_max must be in [0, 1]")

    if not 0.0 < config.subsampling.sample_fraction < 1.0:
        errors.append("subsampling.sample_fraction must be in (0, 1)")
    if config.subsampling.repeats < 2:
        errors.append("subsampling.repeats must be at least 2")
    if config.subsampling.max_candidates_per_root < 1:
        errors.append("subsampling.max_candidates_per_root must be positive")
    for name, value in {
        "subsampling.min_subsample_ari": config.subsampling.min_subsample_ari,
        "subsampling.min_matched_fraction": config.subsampling.min_matched_fraction,
        "subsampling.min_eval_coverage": config.subsampling.min_eval_coverage,
        "subsampling.min_context_recall_q10": config.subsampling.min_context_recall_q10,
        "subsampling.min_worst_context_precision_q10": (
            config.subsampling.min_worst_context_precision_q10
        ),
        "subsampling.min_worst_context_jaccard_q10": (
            config.subsampling.min_worst_context_jaccard_q10
        ),
        "subsampling.min_cluster_count_ratio": config.subsampling.min_cluster_count_ratio,
        "subsampling.max_cluster_count_ratio": config.subsampling.max_cluster_count_ratio,
    }.items():
        if value < 0.0:
            errors.append(f"{name} cannot be negative")
    if config.subsampling.min_cluster_count_ratio > config.subsampling.max_cluster_count_ratio:
        errors.append("subsampling cluster-count ratio bounds are reversed")
    if config.validation.min_samples != config.leiden.min_cluster_size:
        errors.append("validation.min_samples must equal leiden.min_cluster_size")
    for name, value in {
        "refinement.min_comerge_rate": config.refinement.min_comerge_rate,
        "refinement.centroid_distance_quantile": (
            config.refinement.centroid_distance_quantile
        ),
        "refinement.min_context_precision_q10": (
            config.refinement.min_context_precision_q10
        ),
        "refinement.min_context_jaccard_q10": (
            config.refinement.min_context_jaccard_q10
        ),
        "refinement.technical_top_fraction": (
            config.refinement.technical_top_fraction
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]")
    if config.refinement.burden_median_delta < 0:
        errors.append("refinement.burden_median_delta cannot be negative")
    if config.refinement.technical_enrichment_ratio < 1.0:
        errors.append("refinement.technical_enrichment_ratio must be at least 1")
    for name, value in {
        "refinement.track1_pair_min_abs_median_delta": (
            config.refinement.track1_pair_min_abs_median_delta
        ),
        "refinement.track2_pair_min_abs_adjusted_delta": (
            config.refinement.track2_pair_min_abs_adjusted_delta
        ),
        "refinement.track2_pair_min_abs_prevalence_delta": (
            config.refinement.track2_pair_min_abs_prevalence_delta
        ),
    }.items():
        if value < 0:
            errors.append(f"{name} cannot be negative")
    if config.refinement.min_final_context_size != config.leiden.min_cluster_size:
        errors.append(
            "refinement.min_final_context_size must equal leiden.min_cluster_size"
        )
    if config.refinement.min_final_contexts < 2:
        errors.append("refinement.min_final_contexts must be at least 2")
    for name, value in {
        "reference_contexts.min_detection_rate": (
            config.reference_contexts.min_detection_rate
        ),
        "reference_contexts.min_conditional_recall_q10": (
            config.reference_contexts.min_conditional_recall_q10
        ),
        "reference_contexts.min_conditional_precision_q10": (
            config.reference_contexts.min_conditional_precision_q10
        ),
        "reference_contexts.min_conditional_jaccard_q10": (
            config.reference_contexts.min_conditional_jaccard_q10
        ),
        "reference_contexts.anchor_coverage_warning": (
            config.reference_contexts.anchor_coverage_warning
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]")
    if config.reference_contexts.min_context_size != config.leiden.min_cluster_size:
        errors.append(
            "reference_contexts.min_context_size must equal "
            "leiden.min_cluster_size"
        )
    for name, value in {
        "projection.lower_confidence_quantile": (
            config.projection.lower_confidence_quantile
        ),
        "projection.upper_distance_quantile": (
            config.projection.upper_distance_quantile
        ),
        "projection.min_context_loo_accuracy": (
            config.projection.min_context_loo_accuracy
        ),
    }.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]")
    if config.projection.min_correct_calibration_samples < 2:
        errors.append(
            "projection.min_correct_calibration_samples must be at least 2"
        )
    if config.projection.top_candidates < 1:
        errors.append("projection.top_candidates must be positive")
    if config.signatures.min_platform_samples < 2:
        errors.append("signatures.min_platform_samples must be at least 2")
    for name, value in {
        "signatures.track1_q_cut": config.signatures.track1_q_cut,
        "signatures.track1_min_abs_auc_shift": (
            config.signatures.track1_min_abs_auc_shift
        ),
        "signatures.track2_q_cut": config.signatures.track2_q_cut,
        "signatures.track2_min_abs_auc_shift": (
            config.signatures.track2_min_abs_auc_shift
        ),
        "signatures.track2_min_platform_support_fraction": (
            config.signatures.track2_min_platform_support_fraction
        ),
        "signatures.track2_max_auc_gap": config.signatures.track2_max_auc_gap,
    }.items():
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be in [0, 1]")
    if not 0.0 < config.resolution_grid.r_min < config.resolution_grid.r_max:
        errors.append("leiden.resolution_grid requires 0 < r_min < r_max")
    if config.resolution_grid.n_broad < 3 or config.resolution_grid.n_fine < 3:
        errors.append("resolution grids require at least three points")

    if errors:
        raise ValueError("Invalid analysis configuration:\n- " + "\n- ".join(errors))


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    """Load the canonical analysis configuration."""

    path = Path(path)
    data = yaml.safe_load(path.read_text())

    run_name = str(data.get("run_name", "analysis"))
    excluded_roots = _as_str_tuple(data.get("excluded_roots"), default=())
    ir_d = data.get("ir") or {}
    knn_d = data.get("knn") or {}
    knn_selection_d = knn_d.get("selection", {}) if isinstance(knn_d, dict) else {}
    leiden_d = data.get("leiden") or {}
    subsampling_d = data.get("subsampling") or {}
    refinement_d = data.get("refinement") or {}
    reference_contexts_d = data.get("reference_contexts") or {}
    projection_d = data.get("projection") or {}
    signatures_d = data.get("signatures") or {}
    validation_d = data.get("validation") or {}

    # resolution grid: v2 path is leiden.resolution_grid
    rg_d = {}
    if isinstance(leiden_d, dict) and "resolution_grid" in leiden_d and isinstance(leiden_d["resolution_grid"], dict):
        rg_d = dict(leiden_d.get("resolution_grid") or {})

    cfg = AnalysisConfig(
        run_name=run_name,
        excluded_roots=excluded_roots,
        ir_build=IRBuildParams(n_dims=int(ir_d.get("n_dims", 50))),
        knn_selection=KNNSelectionParams(
            metrics=tuple(
                v.lower()
                for v in _as_str_tuple(
                    knn_selection_d.get("metrics"), default=("euclidean", "cosine")
                )
            ),
            n_pcs=_as_int_tuple(knn_selection_d.get("n_pcs", (10, 20, 30, 50)), default=(10, 20, 30, 50)),
            k_values=_as_int_tuple(knn_selection_d.get("k", (30, 50, 80)), default=(30, 50, 80)),
            min_giant_frac=float(knn_selection_d.get("min_giant_frac", 0.99)),
            max_components=int(knn_selection_d.get("max_components", 2)),
            max_platform_enrichment=(
                None
                if knn_selection_d.get("max_platform_enrichment", None) is None
                else float(knn_selection_d.get("max_platform_enrichment"))
            ),
            platform_enrichment_tolerance=(
                None
                if knn_selection_d.get("platform_enrichment_tolerance", 0.05) is None
                else float(knn_selection_d.get("platform_enrichment_tolerance", 0.05))
            ),
            preferred_metrics=tuple(
                v.lower()
                for v in _as_str_tuple(
                    knn_selection_d.get("preferred_metrics"), default=("cosine", "euclidean")
                )
            ),
            preferred_k=_as_int_tuple(knn_selection_d.get("preferred_k", (80, 50, 30)), default=(80, 50, 30)),
            preferred_n_pcs=_as_int_tuple(
                knn_selection_d.get("preferred_n_pcs", (30, 50)), default=(30, 50)
            ),
            max_morph_purity_drop=(
                None
                if knn_selection_d.get("max_morph_purity_drop", None) is None
                else float(knn_selection_d.get("max_morph_purity_drop"))
            ),
            leiden_scan_enabled=_as_bool(knn_selection_d.get("leiden_scan_enabled", True), default=True),
            leiden_scan_top_n=int(knn_selection_d.get("leiden_scan_top_n", 6)),
            sigma_quantile=float(knn_selection_d.get("sigma_quantile", 0.90)),
            make_undirected=_as_bool(knn_selection_d.get("make_undirected", True), default=True),
            combine_mode=str(knn_selection_d.get("combine_mode", "max")),
        ),
        leiden=LeidenParams(
            seeds=_as_int_tuple(leiden_d.get("seeds", (0, 1, 2, 3, 4)), default=(0, 1, 2, 3, 4)),
            min_cluster_size=int(leiden_d.get("min_cluster_size", 200)),
            max_frac_in_small=float(leiden_d.get("max_frac_in_small", 0.02)),
            max_giant_frac=float(leiden_d.get("max_giant_frac", 0.99)),
            min_ari=float(leiden_d.get("min_ari", 0.80)),
            selection_score_tolerance=float(leiden_d.get("selection_score_tolerance", 0.02)),
            selection_policy=str(leiden_d.get("selection_policy", "score_band_center")),
            min_resolution_neighbor_ari=float(leiden_d.get("min_resolution_neighbor_ari", 0.85)),
            min_plateau_points=int(leiden_d.get("min_plateau_points", 2)),
            platform_assoc_weight=float(leiden_d.get("platform_assoc_weight", 0.0)),
            platform_assoc_max=(None if leiden_d.get("platform_assoc_max", None) is None else float(leiden_d.get("platform_assoc_max"))),
        ),
        subsampling=SubsamplingSelectionParams(
            sample_fraction=float(subsampling_d.get("sample_fraction", 0.80)),
            repeats=int(subsampling_d.get("repeats", 20)),
            random_seed=int(subsampling_d.get("random_seed", 1701)),
            partition_seed=int(subsampling_d.get("partition_seed", 0)),
            max_candidates_per_root=int(subsampling_d.get("max_candidates_per_root", 6)),
            min_subsample_ari=float(subsampling_d.get("min_subsample_ari", 0.80)),
            min_matched_fraction=float(subsampling_d.get("min_matched_fraction", 0.80)),
            min_eval_coverage=float(subsampling_d.get("min_eval_coverage", 0.95)),
            min_context_recall_q10=float(subsampling_d.get("min_context_recall_q10", 0.70)),
            min_worst_context_precision_q10=float(
                subsampling_d.get("min_worst_context_precision_q10", 0.50)
            ),
            min_worst_context_jaccard_q10=float(
                subsampling_d.get("min_worst_context_jaccard_q10", 0.50)
            ),
            min_cluster_count_ratio=float(subsampling_d.get("min_cluster_count_ratio", 0.75)),
            max_cluster_count_ratio=float(subsampling_d.get("max_cluster_count_ratio", 1.25)),
            one_se_multiplier=float(subsampling_d.get("one_se_multiplier", 1.0)),
        ),
        refinement=ContextRefinementParams(
            min_comerge_rate=float(refinement_d.get("min_comerge_rate", 0.50)),
            centroid_distance_quantile=float(
                refinement_d.get("centroid_distance_quantile", 0.25)
            ),
            min_context_precision_q10=float(
                refinement_d.get("min_context_precision_q10", 0.50)
            ),
            min_context_jaccard_q10=float(
                refinement_d.get("min_context_jaccard_q10", 0.50)
            ),
            burden_median_delta=float(
                refinement_d.get("burden_median_delta", 0.10)
            ),
            technical_top_fraction=float(
                refinement_d.get("technical_top_fraction", 0.80)
            ),
            technical_enrichment_ratio=float(
                refinement_d.get("technical_enrichment_ratio", 1.50)
            ),
            track1_pair_min_abs_median_delta=float(
                refinement_d.get("track1_pair_min_abs_median_delta", 0.10)
            ),
            track2_pair_min_abs_adjusted_delta=float(
                refinement_d.get("track2_pair_min_abs_adjusted_delta", 0.10)
            ),
            track2_pair_min_abs_prevalence_delta=float(
                refinement_d.get("track2_pair_min_abs_prevalence_delta", 0.10)
            ),
            track2_pair_veto_requires_architecture_or_high_level=_as_bool(
                refinement_d.get(
                    "track2_pair_veto_requires_architecture_or_high_level", True
                ),
                True,
            ),
            min_final_context_size=int(
                refinement_d.get("min_final_context_size", 200)
            ),
            min_final_contexts=int(refinement_d.get("min_final_contexts", 2)),
        ),
        reference_contexts=ReferenceContextParams(
            min_detection_rate=float(
                reference_contexts_d.get("min_detection_rate", 0.90)
            ),
            min_conditional_recall_q10=float(
                reference_contexts_d.get("min_conditional_recall_q10", 0.70)
            ),
            min_conditional_precision_q10=float(
                reference_contexts_d.get("min_conditional_precision_q10", 0.50)
            ),
            min_conditional_jaccard_q10=float(
                reference_contexts_d.get("min_conditional_jaccard_q10", 0.50)
            ),
            min_context_size=int(
                reference_contexts_d.get("min_context_size", 200)
            ),
            anchor_coverage_warning=float(
                reference_contexts_d.get("anchor_coverage_warning", 0.80)
            ),
            root_only_fallback_roots=_as_str_tuple(
                reference_contexts_d.get("root_only_fallback_roots"),
                default=("C47",),
            ),
        ),
        projection=ReferenceProjectionParams(
            lower_confidence_quantile=float(
                projection_d.get("lower_confidence_quantile", 0.10)
            ),
            upper_distance_quantile=float(
                projection_d.get("upper_distance_quantile", 0.90)
            ),
            min_correct_calibration_samples=int(
                projection_d.get("min_correct_calibration_samples", 20)
            ),
            min_context_loo_accuracy=float(
                projection_d.get("min_context_loo_accuracy", 0.80)
            ),
            top_candidates=int(projection_d.get("top_candidates", 5)),
        ),
        signatures=SignatureEvidenceParams(
            min_platform_samples=int(
                signatures_d.get("min_platform_samples", 20)
            ),
            track1_q_cut=float(signatures_d.get("track1_q_cut", 0.05)),
            track1_min_abs_median_delta=float(
                signatures_d.get("track1_min_abs_median_delta", 0.10)
            ),
            track1_min_abs_auc_shift=float(
                signatures_d.get("track1_min_abs_auc_shift", 0.10)
            ),
            track1_min_platform_delta=float(
                signatures_d.get("track1_min_platform_delta", 0.05)
            ),
            track1_min_platforms_supporting=int(
                signatures_d.get("track1_min_platforms_supporting", 2)
            ),
            track2_top_n=int(signatures_d.get("track2_top_n", 30)),
            track2_q_cut=float(signatures_d.get("track2_q_cut", 0.05)),
            track2_min_abs_prevalence_delta=float(
                signatures_d.get("track2_min_abs_prevalence_delta", 0.05)
            ),
            track2_min_abs_auc_shift=float(
                signatures_d.get("track2_min_abs_auc_shift", 0.10)
            ),
            track2_min_platforms_supporting=int(
                signatures_d.get("track2_min_platforms_supporting", 2)
            ),
            track2_min_platform_support_fraction=float(
                signatures_d.get("track2_min_platform_support_fraction", 2.0 / 3.0)
            ),
            track2_max_auc_gap=float(
                signatures_d.get("track2_max_auc_gap", 0.15)
            ),
        ),
        validation=ValidationParams(
            outlier_label=int(validation_d.get("outlier_label", -1)),
            min_samples=int(validation_d.get("min_samples", 200)),
            test_size=float(validation_d.get("test_size", 0.25)),
            random_state=int(validation_d.get("random_state", 0)),
            top_k_per_cluster_event=int(validation_d.get("top_k_per_cluster_event", 30)),
            q_cut=float(validation_d.get("q_cut", 0.05)),
            do_within_platform=_as_bool(validation_d.get("do_within_platform", True), default=True),
            do_lopo=_as_bool(validation_d.get("do_lopo", False), default=False),
            lr_C=float(validation_d.get("lr_C", 0.5)),
            max_iter=int(validation_d.get("max_iter", 10000)),
            n_jobs=int(validation_d.get("n_jobs", 1)),
        ),
        resolution_grid=ResolutionGridParams(
            r_min=float(rg_d.get("r_min", 1e-4)),
            r_max=float(rg_d.get("r_max", 5e-2)),
            n_broad=int(rg_d.get("n_broad", 15)),
            n_fine=int(rg_d.get("n_fine", 12)),
        ),
    )

    _validate_analysis_config(cfg)
    return cfg
