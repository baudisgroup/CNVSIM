from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Pipeline:
    mode: str


@dataclass(frozen=True)
class Filters:
    excluded_roots: list[str]


@dataclass(frozen=True)
class Encoding:
    min_samples_per_family: int
    min_altered_genes_per_sample: int
    low_frac_threshold: float
    high_frac_threshold: float
    stable_gene_coverage: float
    autosomes_only: bool = True


@dataclass(frozen=True)
class FirstPassUMAP:
    n_neighbors: int
    min_dist: float
    metric: str


@dataclass(frozen=True)
class FirstPassHDBSCAN:
    min_cluster_size: int
    min_samples: int


@dataclass(frozen=True)
class FirstPassTechnical:
    max_rel_cluster_size: float
    min_platform_dominance: float
    max_top_morph_frac: float


@dataclass(frozen=True)
class FirstPass:
    random_state: int
    n_pcs: int
    umap: FirstPassUMAP
    hdbscan: FirstPassHDBSCAN
    technical: FirstPassTechnical


@dataclass(frozen=True)
class Stage2A:
    n_pcs: int
    plat_eval_pcs: int
    min_per_class: int
    random_state: int
    feature_scaling: str = "standard"


@dataclass(frozen=True)
class PreprocessConfig:
    run_name: str
    pipeline: Pipeline
    filters: Filters
    encoding: Encoding
    firstpass: FirstPass
    stage2a: Stage2A


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    """Load scientific preprocessing settings from YAML.

    Input and output locations are runtime concerns and must be passed to the
    public command-line entrypoint. Keeping paths out of this configuration
    makes the same analysis settings portable across machines.
    """

    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    misplaced_path_keys = sorted(set(data) & {"inputs", "out_dir", "paths"})
    if misplaced_path_keys:
        raise ValueError(
            "Runtime paths do not belong in preprocess YAML; pass "
            "--source-dir, --work-dir, and --release-dir instead. "
            f"Remove: {misplaced_path_keys}"
        )

    run_name = str(data.get("run_name", "preprocess"))
    pipeline_d = data.get("pipeline", {})
    if not isinstance(pipeline_d, dict):
        raise ValueError("pipeline must be a YAML mapping")
    if "release_dir" in pipeline_d:
        raise ValueError(
            "pipeline.release_dir is a runtime path; pass --release-dir instead"
        )
    filters_d = data.get("filters", {})
    encoding_d = data.get("encoding", {})
    firstpass_d = data.get("firstpass", {})
    stage2a_d = data.get("stage2a", {})
    pipeline_mode = str(pipeline_d.get("mode", "")).strip()
    valid_modes = {"baseline"}
    if pipeline_mode not in valid_modes:
        raise ValueError(
            "pipeline.mode must be one of "
            f"{sorted(valid_modes)}; got {pipeline_mode or '<missing>'!r}"
        )

    stage2a_feature_scaling = str(
        stage2a_d.get("feature_scaling", "standard")
    ).strip().lower()
    valid_stage2a_feature_scalings = {"standard"}
    if stage2a_feature_scaling not in valid_stage2a_feature_scalings:
        raise ValueError(
            "stage2a.feature_scaling must be one of "
            f"{sorted(valid_stage2a_feature_scalings)}; got "
            f"{stage2a_feature_scaling!r}"
        )

    cfg = PreprocessConfig(
        run_name=run_name,
        pipeline=Pipeline(mode=pipeline_mode),
        filters=Filters(
            excluded_roots=list(filters_d.get("excluded_roots", [])),
        ),
        encoding=Encoding(
            min_samples_per_family=int(
                encoding_d.get("min_samples_per_family", 50)
            ),
            min_altered_genes_per_sample=int(
                encoding_d.get("min_altered_genes_per_sample", 1)
            ),
            low_frac_threshold=float(encoding_d.get("low_frac_threshold", 0.10)),
            high_frac_threshold=float(encoding_d.get("high_frac_threshold", 0.20)),
            stable_gene_coverage=float(encoding_d.get("stable_gene_coverage", 0.95)),
            autosomes_only=bool(encoding_d.get("autosomes_only", True)),
        ),
        firstpass=FirstPass(
            random_state=int(firstpass_d.get("random_state", 0)),
            n_pcs=int(firstpass_d.get("n_pcs", 50)),
            umap=FirstPassUMAP(
                n_neighbors=int((firstpass_d.get("umap", {}) or {}).get("n_neighbors", 30)),
                min_dist=float((firstpass_d.get("umap", {}) or {}).get("min_dist", 0.1)),
                metric=str((firstpass_d.get("umap", {}) or {}).get("metric", "cosine")),
            ),
            hdbscan=FirstPassHDBSCAN(
                min_cluster_size=int((firstpass_d.get("hdbscan", {}) or {}).get("min_cluster_size", 200)),
                min_samples=int((firstpass_d.get("hdbscan", {}) or {}).get("min_samples", 50)),
            ),
            technical=FirstPassTechnical(
                max_rel_cluster_size=float((firstpass_d.get("technical", {}) or {}).get("max_rel_cluster_size", 0.10)),
                min_platform_dominance=float((firstpass_d.get("technical", {}) or {}).get("min_platform_dominance", 0.90)),
                max_top_morph_frac=float((firstpass_d.get("technical", {}) or {}).get("max_top_morph_frac", 0.50)),
            ),
        ),
        stage2a=Stage2A(
            n_pcs=int(stage2a_d.get("n_pcs", 50)),
            plat_eval_pcs=int(stage2a_d.get("plat_eval_pcs", 10)),
            min_per_class=int(stage2a_d.get("min_per_class", 50)),
            random_state=int(stage2a_d.get("random_state", 0)),
            feature_scaling=stage2a_feature_scaling,
        ),
    )

    if not cfg.encoding.autosomes_only:
        raise ValueError("encoding.autosomes_only must be true for the canonical pipeline")
    if cfg.encoding.min_altered_genes_per_sample < 1:
        raise ValueError(
            "encoding.min_altered_genes_per_sample must be at least 1 for the "
            "canonical reference space"
        )

    return cfg
