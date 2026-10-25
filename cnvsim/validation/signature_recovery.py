from __future__ import annotations

"""Five-fold fixed-parameter recovery of released CNV signatures."""

import hashlib
import json
import re
import shutil
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cnvsim.analysis.config import AnalysisConfig
from cnvsim.analysis.leiden import run_leiden_cpm
from cnvsim.analysis.signatures import extract_context_signatures
from cnvsim.analysis.subsampling_selection import (
    build_subsample_graph,
    mark_small_clusters,
    scaled_subsample_min_cluster_size,
)
from cnvsim.preprocess.encode import EVENT_COLS
from cnvsim.preprocess.gene_panel_store import (
    load_gene_panel_subset_from_store,
)
from cnvsim.validation.config import SignatureRecoveryParams


CORE_REFERENCE_PATTERN = re.compile(r"^(?P<root>.+)-CTX(?P<context>\d+)$")
TRACK2_RELEASE_CHANNELS = {
    ("gain", "low"): "dup_frac",
    ("gain", "high"): "hldup_frac",
    ("loss", "low"): "del_frac",
    ("loss", "high"): "hldel_frac",
}


@dataclass(frozen=True)
class SignatureRecoveryInputs:
    """Frozen atlas artifacts required by the validation."""

    atlas_dir: Path
    analysis_dir: Path
    ir_dir: Path
    selected_parameters: Path
    provisional_partitions: Path
    refinement_dir: Path
    reference_contexts_dir: Path
    reference_signatures_dir: Path
    release_candidate_dir: Path
    source_dir: Path
    preprocess_release_dir: Path
    chr44_profiles: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_existing(label: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve {label}; checked: {[str(path) for path in candidates]}"
    )


def resolve_signature_recovery_inputs(
    *,
    atlas_dir: Path,
    analysis_dir: Path | None = None,
) -> SignatureRecoveryInputs:
    """Resolve the frozen public-atlas layout without accepting stale outputs."""

    atlas = Path(atlas_dir).expanduser().resolve()
    analysis = (
        Path(analysis_dir).expanduser().resolve()
        if analysis_dir is not None
        else atlas / "work" / "analysis_v1" / "analysis"
    )
    parameters = analysis / "parameters"
    refinement = _first_existing(
        "refinement directory",
        [
            analysis / "refinement",
            analysis / "contexts" / "refinement",
            analysis / "refined_contexts",
        ],
    )
    provisional = _first_existing(
        "provisional partitions",
        [
            parameters / "assessment_partitions",
            analysis / "partition_assessment" / "assessment_partitions",
            analysis / "assessment_partitions",
        ],
    )
    reference_contexts = _first_existing(
        "reference contexts",
        [
            analysis / "reference_contexts",
            analysis / "contexts" / "reference_contexts",
        ],
    )
    reference_signatures = _first_existing(
        "reference signatures",
        [
            analysis / "signatures" / "refined_all",
            analysis / "signatures" / "reference",
        ],
    )
    inputs = SignatureRecoveryInputs(
        atlas_dir=atlas,
        analysis_dir=analysis.resolve(),
        ir_dir=_first_existing("analysis IR", [analysis / "ir"]),
        selected_parameters=_first_existing(
            "selected parameters", [parameters / "selected_parameters.tsv"]
        ),
        provisional_partitions=provisional,
        refinement_dir=refinement,
        reference_contexts_dir=reference_contexts,
        reference_signatures_dir=reference_signatures,
        release_candidate_dir=_first_existing(
            "release candidate", [atlas / "release_candidate"]
        ),
        source_dir=_first_existing("source files", [atlas / "source_file"]),
        preprocess_release_dir=_first_existing(
            "preprocessing release",
            [
                atlas / "preprocessing" / "baseline" / "reference_space",
                atlas / "preprocessing" / "baseline",
            ],
        ),
        chr44_profiles=_first_existing(
            "chromosome profiles",
            [atlas / "source_derived" / "sample_chr44_profiles.parquet"],
        ),
    )
    required = [
        inputs.refinement_dir / "initial_to_final_context.tsv",
        inputs.refinement_dir / "final_partitions",
        inputs.reference_contexts_dir / "context_stability.tsv",
        inputs.reference_signatures_dir / "track1_event_stats.parquet",
        inputs.reference_signatures_dir / "track2_marker_stats.parquet",
        inputs.release_candidate_dir / "tables" / "reference_contexts.tsv",
        inputs.release_candidate_dir / "tables" / "signature_events.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete fixed-parameter signature-recovery inputs: "
            + ", ".join(missing)
        )
    return inputs


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _as_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def _core_reference_catalog(inputs: SignatureRecoveryInputs) -> pd.DataFrame:
    references = pd.read_csv(
        inputs.release_candidate_dir / "tables" / "reference_contexts.tsv",
        sep="\t",
        low_memory=False,
    )
    _require_columns(
        references,
        {
            "reference_key",
            "root",
            "reference_level",
            "n_definition_members",
        },
        "released reference contexts",
    )
    core = references[references["reference_level"].eq("subtype")].copy()
    parsed = core["reference_key"].astype(str).str.extract(
        CORE_REFERENCE_PATTERN
    )
    if parsed.isna().any().any():
        invalid = core.loc[
            parsed.isna().any(axis=1), "reference_key"
        ].astype(str)
        raise ValueError(
            f"Cannot parse released core reference keys: {invalid.tolist()}"
        )
    if not parsed["root"].eq(core["root"].astype(str)).all():
        raise ValueError("Released reference keys disagree with root columns")
    core["final_context"] = parsed["context"].astype(int)
    core["root"] = core["root"].astype(str)
    if core.duplicated(["root", "final_context"]).any():
        raise ValueError("Released core references are not unique by context")
    return core.sort_values(["root", "final_context"]).reset_index(drop=True)


def _parameter_resolution(row: pd.Series) -> float:
    for column in ("stage2_selected_resolution", "resolution"):
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    raise ValueError(f"[{row.get('root', '?')}] selected resolution is absent")


def _load_frozen_parameters(
    inputs: SignatureRecoveryInputs,
    core: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = pd.read_csv(
        inputs.selected_parameters, sep="\t", low_memory=False
    )
    _require_columns(
        parameters,
        {"root", "metric", "n_pcs", "K"},
        "selected parameters",
    )
    roots = set(core["root"].astype(str))
    parameters["root"] = parameters["root"].astype(str)
    parameters = parameters[parameters["root"].isin(roots)].copy()
    if parameters.duplicated("root").any() or set(parameters["root"]) != roots:
        raise ValueError(
            "Selected parameters must contain exactly one row per core root"
        )
    parameters["resolution"] = parameters.apply(
        _parameter_resolution, axis=1
    )
    parameters["metric"] = parameters["metric"].astype(str).str.lower()
    parameters["n_pcs"] = pd.to_numeric(
        parameters["n_pcs"], errors="raise"
    ).astype(int)
    parameters["K"] = pd.to_numeric(
        parameters["K"], errors="raise"
    ).astype(int)
    parameters["partition_seed"] = int(
        analysis_config.subsampling.partition_seed
    )
    parameters["full_min_cluster_size"] = int(
        analysis_config.leiden.min_cluster_size
    )
    parameters["n_core_contexts"] = parameters["root"].map(
        core.groupby("root").size()
    )

    mapping = pd.read_csv(
        inputs.refinement_dir / "initial_to_final_context.tsv",
        sep="\t",
        low_memory=False,
    )
    _require_columns(
        mapping,
        {"root", "initial_context", "final_context"},
        "frozen refinement mapping",
    )
    mapping["root"] = mapping["root"].astype(str)
    mapping = mapping[mapping["root"].isin(roots)].copy()
    for column in ("initial_context", "final_context"):
        mapping[column] = pd.to_numeric(
            mapping[column], errors="raise"
        ).astype(int)
    if mapping.duplicated(["root", "initial_context"]).any():
        raise ValueError("Frozen refinement mapping contains duplicate inputs")
    parameters["n_initial_contexts"] = parameters["root"].map(
        mapping.groupby("root")["initial_context"].nunique()
    )
    parameters["n_refined_contexts"] = parameters["root"].map(
        mapping.groupby("root")["final_context"].nunique()
    )
    frozen_columns = [
        "root",
        "metric",
        "n_pcs",
        "K",
        "resolution",
        "partition_seed",
        "full_min_cluster_size",
        "n_initial_contexts",
        "n_refined_contexts",
        "n_core_contexts",
    ]
    return (
        parameters[frozen_columns].sort_values("root").reset_index(drop=True),
        mapping.sort_values(["root", "initial_context"]).reset_index(drop=True),
    )


def align_fold_partition(
    *,
    analysis_ids: np.ndarray,
    full_final_context: np.ndarray,
    fold_cluster: np.ndarray,
    core_contexts: set[int],
    root: str,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align fold clusters using membership overlap only.

    Multiple fold clusters may align to the same full refined context. This
    preserves split recovery without using Track 1 or Track 2 evidence.
    """

    ids = np.asarray(analysis_ids, dtype=object)
    truth = np.asarray(full_final_context, dtype=np.int32)
    observed = np.asarray(fold_cluster, dtype=np.int32)
    if not (len(ids) == len(truth) == len(observed)):
        raise ValueError("Fold alignment arrays have inconsistent lengths")

    aligned = np.full(len(observed), -1, dtype=np.int32)
    mapping_rows: list[dict[str, object]] = []
    for cluster in sorted(np.unique(observed[observed >= 0])):
        cluster_mask = observed == int(cluster)
        inlier_truth = truth[cluster_mask & (truth >= 0)]
        if len(inlier_truth) == 0:
            mapping_rows.append(
                {
                    "root": str(root),
                    "fold": int(fold),
                    "fold_cluster": int(cluster),
                    "mapped_final_context": -1,
                    "n_cluster_samples": int(cluster_mask.sum()),
                    "n_refined_overlap": 0,
                    "n_best_overlap": 0,
                    "mapping_purity": 0.0,
                    "mapping_status": "no_refined_overlap",
                }
            )
            continue
        counts = pd.Series(inlier_truth).value_counts().sort_index()
        best_count = int(counts.max())
        best_context = int(counts[counts.eq(best_count)].index.min())
        aligned[cluster_mask] = best_context
        mapping_rows.append(
            {
                "root": str(root),
                "fold": int(fold),
                "fold_cluster": int(cluster),
                "mapped_final_context": best_context,
                "n_cluster_samples": int(cluster_mask.sum()),
                "n_refined_overlap": int(len(inlier_truth)),
                "n_best_overlap": best_count,
                "mapping_purity": float(
                    best_count / int(cluster_mask.sum())
                ),
                "mapping_status": (
                    "mapped_core"
                    if best_context in core_contexts
                    else "mapped_non_core"
                ),
            }
        )

    sample_frame = pd.DataFrame(
        {
            "analysis_id": ids.astype(str),
            "root": str(root),
            "fold": int(fold),
            "full_final_context": truth,
            "fold_cluster": observed,
            "aligned_final_context": aligned,
        }
    )
    sample_frame["assignment_status"] = np.where(
        sample_frame["aligned_final_context"].isin(core_contexts),
        "anchor_member",
        "non_anchor_query",
    )

    context_rows: list[dict[str, object]] = []
    for context in sorted(core_contexts):
        truth_mask = truth == int(context)
        predicted_mask = aligned == int(context)
        overlap = int(np.sum(truth_mask & predicted_mask))
        n_truth = int(truth_mask.sum())
        n_predicted = int(predicted_mask.sum())
        denominator = n_truth + n_predicted - overlap
        context_rows.append(
            {
                "root": str(root),
                "fold": int(fold),
                "final_context": int(context),
                "n_truth": n_truth,
                "n_predicted": n_predicted,
                "n_overlap": overlap,
                "detected": bool(n_predicted > 0),
                "membership_recall": (
                    float(overlap / n_truth) if n_truth else np.nan
                ),
                "membership_precision": (
                    float(overlap / n_predicted)
                    if n_predicted
                    else np.nan
                ),
                "membership_jaccard": (
                    float(overlap / denominator)
                    if denominator
                    else np.nan
                ),
            }
        )
    return (
        sample_frame,
        pd.DataFrame(mapping_rows),
        pd.DataFrame(context_rows),
    )


def make_complementary_fold_assignments(
    *,
    analysis_ids: np.ndarray,
    full_final_context: np.ndarray,
    root: str,
    n_folds: int,
    random_seed: int,
) -> pd.DataFrame:
    """Assign each sample to exactly one context-balanced held-out fold."""

    ids = np.asarray(analysis_ids, dtype=object)
    contexts = np.asarray(full_final_context, dtype=np.int32)
    if len(ids) != len(contexts):
        raise ValueError("Fold-assignment arrays have inconsistent lengths")
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2")
    heldout_fold = np.full(len(ids), -1, dtype=np.int16)
    for context in sorted(np.unique(contexts)):
        positions = np.flatnonzero(contexts == int(context))
        stratum_offset = int(
            zlib.crc32(f"{root}|{int(context)}".encode("utf-8"))
        )
        rng = np.random.default_rng(int(random_seed) + stratum_offset)
        shuffled = rng.permutation(positions)
        start = int(rng.integers(0, int(n_folds)))
        heldout_fold[shuffled] = (
            np.arange(len(shuffled), dtype=np.int32) + start
        ) % int(n_folds)
    if np.any(heldout_fold < 0):
        raise ValueError(f"[{root}] some samples lack a held-out fold")
    result = pd.DataFrame(
        {
            "analysis_id": ids.astype(str),
            "root": str(root),
            "full_final_context": contexts,
            "heldout_fold": heldout_fold.astype(int),
        }
    )
    if result.duplicated("analysis_id").any():
        raise ValueError(f"[{root}] fold assignments contain duplicate samples")
    return result


def _load_root_partitions(
    *,
    inputs: SignatureRecoveryInputs,
    root: str,
    coords: pd.DataFrame,
    root_mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    provisional_path = (
        inputs.provisional_partitions
        / f"{root}_provisional_partition.parquet"
    )
    final_path = (
        inputs.refinement_dir
        / "final_partitions"
        / f"{root}_final_partition.parquet"
    )
    for path in (provisional_path, final_path):
        if not path.exists():
            raise FileNotFoundError(path)
    provisional = pd.read_parquet(provisional_path)
    final = pd.read_parquet(final_path)
    _require_columns(
        provisional,
        {"analysis_id", "provisional_context"},
        f"[{root}] provisional partition",
    )
    _require_columns(
        final,
        {"analysis_id", "final_context"},
        f"[{root}] final partition",
    )
    index = pd.Index(coords.index.astype(str), name="analysis_id")
    provisional = provisional.set_index(
        provisional["analysis_id"].astype(str)
    ).reindex(index)
    final = final.set_index(final["analysis_id"].astype(str)).reindex(index)
    if provisional["provisional_context"].isna().any():
        raise ValueError(f"[{root}] provisional partition misses IR samples")
    if final["final_context"].isna().any():
        raise ValueError(f"[{root}] final partition misses IR samples")
    provisional_labels = pd.to_numeric(
        provisional["provisional_context"], errors="raise"
    ).astype(int)
    final_labels = pd.to_numeric(
        final["final_context"], errors="raise"
    ).astype(int)
    mapping = root_mapping.set_index("initial_context")["final_context"]
    derived = provisional_labels.map(mapping).fillna(-1).astype(int)
    mismatch = derived.to_numpy() != final_labels.to_numpy()
    if mismatch.any():
        raise ValueError(
            f"[{root}] frozen mapping disagrees with final partition for "
            f"{int(mismatch.sum())} samples"
        )
    return provisional_labels.to_numpy(np.int32), derived.to_numpy(np.int32)


def rebuild_fixed_parameter_partitions(
    *,
    inputs: SignatureRecoveryInputs,
    out_dir: Path,
    analysis_config: AnalysisConfig,
    params: SignatureRecoveryParams,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Rebuild complementary 80% folds with frozen atlas parameters."""

    out = Path(out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if overwrite:
            shutil.rmtree(out)
        elif (out / "rebuild_manifest.json").exists():
            raise FileExistsError(
                f"Signature-recovery output exists: {out}; use --overwrite"
            )
    out.mkdir(parents=True, exist_ok=True)
    folds_dir = out / "fold_reference_contexts"
    folds_dir.mkdir(exist_ok=True)
    checkpoints_dir = out / "rebuild_checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    core = _core_reference_catalog(inputs)
    frozen, mapping = _load_frozen_parameters(
        inputs, core, analysis_config
    )
    frozen_path = out / "frozen_parameters.tsv"
    mapping_path = out / "frozen_refinement_mapping.tsv"
    frozen.to_csv(frozen_path, sep="\t", index=False)
    mapping.to_csv(mapping_path, sep="\t", index=False)

    source_stability = pd.read_csv(
        inputs.reference_contexts_dir / "context_stability.tsv",
        sep="\t",
        low_memory=False,
    )
    run_rows: list[dict[str, object]] = []
    cluster_mapping_parts: list[pd.DataFrame] = []
    context_metric_parts: list[pd.DataFrame] = []
    fold_assignment_parts: list[pd.DataFrame] = []
    fold_status: dict[int, list[pd.DataFrame]] = {
        fold: [] for fold in range(int(params.n_folds))
    }
    fold_partitions: dict[int, list[tuple[str, pd.DataFrame]]] = {
        fold: [] for fold in range(int(params.n_folds))
    }
    train_fraction = (int(params.n_folds) - 1) / int(params.n_folds)
    fold_min_size = scaled_subsample_min_cluster_size(
        int(analysis_config.leiden.min_cluster_size),
        train_fraction,
    )

    for root_index, parameter in frozen.sort_values("root").iterrows():
        root = str(parameter["root"])
        checkpoint = checkpoints_dir / root
        checkpoint_signature = {
            "schema_version": 1,
            "root": root,
            "parameters": {
                column: (
                    int(parameter[column])
                    if column
                    in {
                        "n_pcs",
                        "K",
                        "partition_seed",
                        "full_min_cluster_size",
                    }
                    else float(parameter[column])
                    if column == "resolution"
                    else str(parameter[column])
                )
                for column in (
                    "metric",
                    "n_pcs",
                    "K",
                    "resolution",
                    "partition_seed",
                    "full_min_cluster_size",
                )
            },
            "validation": asdict(params),
        }
        checkpoint_manifest = checkpoint / "manifest.json"
        if checkpoint_manifest.exists():
            observed_signature = json.loads(
                checkpoint_manifest.read_text()
            )
            if observed_signature != checkpoint_signature:
                raise ValueError(
                    f"[{root}] rebuild checkpoint does not match the "
                    "current frozen configuration"
                )
            print(
                f"[signature recovery] reuse root checkpoint={root}",
                flush=True,
            )
            run_rows.extend(
                pd.read_csv(
                    checkpoint / "run_metrics.tsv", sep="\t"
                ).to_dict(orient="records")
            )
            cluster_mapping_parts.append(
                pd.read_parquet(checkpoint / "cluster_mapping.parquet")
            )
            context_metric_parts.append(
                pd.read_parquet(checkpoint / "context_metrics.parquet")
            )
            fold_assignment_parts.append(
                pd.read_parquet(checkpoint / "fold_assignments.parquet")
            )
            for fold in range(int(params.n_folds)):
                fold_status[fold].append(
                    pd.read_parquet(
                        checkpoint
                        / "status"
                        / f"fold_{fold:02d}.parquet"
                    )
                )
                fold_partitions[fold].append(
                    (
                        root,
                        pd.read_parquet(
                            checkpoint
                            / "partitions"
                            / f"fold_{fold:02d}.parquet"
                        ),
                    )
                )
            continue
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        print(
            f"[signature recovery] root={root} "
            f"({root_index + 1}/{len(frozen)})",
            flush=True,
        )
        coords_path = inputs.ir_dir / "perroot" / root / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(coords_path)
        coords = pd.read_parquet(coords_path)
        coords.index = coords.index.astype(str)
        root_mapping = mapping[mapping["root"].eq(root)].copy()
        _, full_final = _load_root_partitions(
            inputs=inputs,
            root=root,
            coords=coords,
            root_mapping=root_mapping,
        )
        core_contexts = set(
            core.loc[core["root"].eq(root), "final_context"].astype(int)
        )
        root_folds = make_complementary_fold_assignments(
            analysis_ids=coords.index.to_numpy(),
            full_final_context=full_final,
            root=root,
            n_folds=int(params.n_folds),
            random_seed=int(params.random_seed),
        )
        fold_assignment_parts.append(root_folds)
        root_run_rows: list[dict[str, object]] = []
        root_cluster_mapping: list[pd.DataFrame] = []
        root_context_metrics: list[pd.DataFrame] = []
        root_status: dict[int, pd.DataFrame] = {}
        root_partitions: dict[int, pd.DataFrame] = {}

        for fold in range(int(params.n_folds)):
            subset_indices = np.flatnonzero(
                root_folds["heldout_fold"].to_numpy(int) != int(fold)
            )
            heldout_indices = np.flatnonzero(
                root_folds["heldout_fold"].to_numpy(int) == int(fold)
            )
            graph = build_subsample_graph(
                coords,
                sample_indices=subset_indices,
                metric=str(parameter["metric"]),
                n_pcs=int(parameter["n_pcs"]),
                k=int(parameter["K"]),
                sigma_quantile=float(
                    analysis_config.knn_selection.sigma_quantile
                ),
                make_undirected=bool(
                    analysis_config.knn_selection.make_undirected
                ),
                combine_mode=str(
                    analysis_config.knn_selection.combine_mode
                ),
            )
            labels, _ = run_leiden_cpm(
                graph,
                resolution=float(parameter["resolution"]),
                seed=int(parameter["partition_seed"]),
            )
            labels = mark_small_clusters(labels, fold_min_size)
            sample_frame, cluster_mapping, context_metrics = (
                align_fold_partition(
                    analysis_ids=coords.index.to_numpy()[subset_indices],
                    full_final_context=full_final[subset_indices],
                    fold_cluster=labels,
                    core_contexts=core_contexts,
                    root=root,
                    fold=fold,
                )
            )
            root_cluster_mapping.append(cluster_mapping)
            context_metrics = context_metrics.merge(
                core[
                    core["root"].eq(root)
                ][
                    [
                        "root",
                        "final_context",
                        "reference_key",
                        "n_definition_members",
                    ]
                ],
                on=["root", "final_context"],
                how="left",
                validate="many_to_one",
            )
            root_context_metrics.append(context_metrics)
            status_frame = sample_frame[
                ["analysis_id", "root", "assignment_status"]
            ].copy()
            root_status[fold] = status_frame
            fold_status[fold].append(status_frame)
            anchor_partition = (
                sample_frame[
                    sample_frame["assignment_status"].eq(
                        "anchor_member"
                    )
                ][["analysis_id", "aligned_final_context"]]
                .rename(
                    columns={
                        "aligned_final_context": "final_context"
                    }
                )
                .reset_index(drop=True)
            )
            root_partitions[fold] = anchor_partition
            fold_partitions[fold].append((root, anchor_partition))

            truth_core = sample_frame["full_final_context"].isin(
                core_contexts
            ).to_numpy()
            predicted_core = sample_frame["aligned_final_context"].isin(
                core_contexts
            ).to_numpy()
            evaluated = truth_core | predicted_core
            if int(evaluated.sum()) >= 2:
                truth_labels = np.where(
                    truth_core[evaluated],
                    sample_frame.loc[
                        evaluated, "full_final_context"
                    ].to_numpy(int),
                    -1,
                )
                predicted_labels = np.where(
                    predicted_core[evaluated],
                    sample_frame.loc[
                        evaluated, "aligned_final_context"
                    ].to_numpy(int),
                    -1,
                )
                ari = float(
                    adjusted_rand_score(truth_labels, predicted_labels)
                )
                matched = float(np.mean(truth_labels == predicted_labels))
            else:
                ari = np.nan
                matched = np.nan
            root_run_rows.append(
                {
                    "root": root,
                    "fold": int(fold),
                    "n_root_samples": int(len(coords)),
                    "n_train": int(len(subset_indices)),
                    "n_heldout": int(len(heldout_indices)),
                    "train_fraction": float(
                        len(subset_indices) / len(coords)
                    ),
                    "n_fold_clusters": int(
                        pd.Series(labels[labels >= 0]).nunique()
                    ),
                    "n_fold_outliers": int(np.sum(labels < 0)),
                    "n_core_truth": int(truth_core.sum()),
                    "n_core_predicted": int(predicted_core.sum()),
                    "core_union_ari": ari,
                    "core_union_matched_fraction": matched,
                }
            )

        root_cluster_frame = pd.concat(
            root_cluster_mapping, ignore_index=True
        )
        root_context_frame = pd.concat(
            root_context_metrics, ignore_index=True
        )
        run_rows.extend(root_run_rows)
        cluster_mapping_parts.append(root_cluster_frame)
        context_metric_parts.append(root_context_frame)
        (checkpoint / "status").mkdir(parents=True)
        (checkpoint / "partitions").mkdir()
        pd.DataFrame(root_run_rows).to_csv(
            checkpoint / "run_metrics.tsv", sep="\t", index=False
        )
        root_cluster_frame.to_parquet(
            checkpoint / "cluster_mapping.parquet", index=False
        )
        root_context_frame.to_parquet(
            checkpoint / "context_metrics.parquet", index=False
        )
        root_folds.to_parquet(
            checkpoint / "fold_assignments.parquet", index=False
        )
        for fold in range(int(params.n_folds)):
            root_status[fold].to_parquet(
                checkpoint
                / "status"
                / f"fold_{fold:02d}.parquet",
                index=False,
            )
            root_partitions[fold].to_parquet(
                checkpoint
                / "partitions"
                / f"fold_{fold:02d}.parquet",
                index=False,
            )
        checkpoint_manifest.write_text(
            json.dumps(checkpoint_signature, indent=2, sort_keys=True)
            + "\n"
        )

    for fold in range(int(params.n_folds)):
        fold_dir = folds_dir / f"fold_{fold:02d}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        partitions_dir = fold_dir / "anchor_partitions"
        partitions_dir.mkdir(parents=True)
        for root, partition in fold_partitions[fold]:
            partition.to_parquet(
                partitions_dir / f"{root}_final_partition.parquet",
                index=False,
            )
        status = pd.concat(fold_status[fold], ignore_index=True)
        if status.duplicated("analysis_id").any():
            raise ValueError(
                f"Fold {fold} assigns an analysis to multiple roots"
            )
        status.to_parquet(
            fold_dir / "sample_reference_status.parquet", index=False
        )
        source_stability.to_csv(
            fold_dir / "context_stability.tsv",
            sep="\t",
            index=False,
        )

    run_metrics_path = out / "fold_partition_metrics.tsv"
    cluster_mapping_path = out / "fold_cluster_mapping.parquet"
    context_metrics_path = out / "fold_membership_metrics.parquet"
    fold_assignments_path = out / "fold_assignments.parquet"
    pd.DataFrame(run_rows).to_csv(
        run_metrics_path, sep="\t", index=False
    )
    pd.concat(cluster_mapping_parts, ignore_index=True).to_parquet(
        cluster_mapping_path, index=False
    )
    pd.concat(context_metric_parts, ignore_index=True).to_parquet(
        context_metrics_path, index=False
    )
    fold_assignments = pd.concat(
        fold_assignment_parts, ignore_index=True
    )
    if fold_assignments.duplicated("analysis_id").any():
        raise ValueError("An analysis was assigned in multiple roots")
    fold_assignments.to_parquet(fold_assignments_path, index=False)
    manifest_path = out / "rebuild_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cnvsim_fixed_parameter_rebuild_v2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "validation_role": "fixed_parameter_context_rebuild",
                "representation_policy": (
                    "frozen full-atlas root coordinates; no PCA refit"
                ),
                "parameter_policy": (
                    "metric, n_pcs, K, resolution, partition seed, and "
                    "initial-to-final refinement mapping are frozen"
                ),
                "fold_policy": (
                    "complementary folds balanced within full-atlas final "
                    "contexts; every sample is held out exactly once"
                ),
                "alignment_policy": (
                    "fold clusters map to frozen final contexts by maximum "
                    "membership overlap; signatures are not used"
                ),
                "params": asdict(params),
                "train_fraction": float(train_fraction),
                "fold_min_cluster_size": int(fold_min_size),
                "counts": {
                    "roots": int(core["root"].nunique()),
                    "core_contexts": int(len(core)),
                    "folds": int(params.n_folds),
                },
                "inputs": {
                    "selected_parameters": str(inputs.selected_parameters),
                    "selected_parameters_sha256": _sha256(
                        inputs.selected_parameters
                    ),
                    "refinement_mapping": str(
                        inputs.refinement_dir
                        / "initial_to_final_context.tsv"
                    ),
                    "refinement_mapping_sha256": _sha256(
                        inputs.refinement_dir
                        / "initial_to_final_context.tsv"
                    ),
                    "release_manifest": str(
                        inputs.release_candidate_dir
                        / "provenance"
                        / "release_manifest.json"
                    ),
                },
                "outputs": {
                    "frozen_parameters": str(frozen_path),
                    "frozen_refinement_mapping": str(mapping_path),
                    "fold_reference_contexts": str(folds_dir),
                    "fold_partition_metrics": str(run_metrics_path),
                    "fold_cluster_mapping": str(cluster_mapping_path),
                    "fold_membership_metrics": str(
                        context_metrics_path
                    ),
                    "fold_assignments": str(fold_assignments_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "frozen_parameters": frozen_path,
        "frozen_refinement_mapping": mapping_path,
        "fold_reference_contexts": folds_dir,
        "fold_partition_metrics": run_metrics_path,
        "fold_cluster_mapping": cluster_mapping_path,
        "fold_membership_metrics": context_metrics_path,
        "fold_assignments": fold_assignments_path,
        "manifest": manifest_path,
    }


def extract_fold_signatures(
    *,
    inputs: SignatureRecoveryInputs,
    out_dir: Path,
    analysis_config: AnalysisConfig,
    params: SignatureRecoveryParams,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Recompute Track 1 and Track 2 for every rebuilt fold partition."""

    out = Path(out_dir).expanduser().resolve()
    folds_dir = out / "fold_reference_contexts"
    if not folds_dir.exists():
        raise FileNotFoundError(
            f"Run the rebuild stage first: {folds_dir}"
        )
    frozen = pd.read_csv(out / "frozen_parameters.tsv", sep="\t")
    roots = sorted(frozen["root"].astype(str).unique())
    signature_root = out / "fold_signatures"
    signature_root.mkdir(exist_ok=True)
    train_fraction = (int(params.n_folds) - 1) / int(params.n_folds)
    min_context_size = scaled_subsample_min_cluster_size(
        int(analysis_config.leiden.min_cluster_size),
        train_fraction,
    )
    manifest_rows: list[dict[str, object]] = []
    for fold in range(int(params.n_folds)):
        fold_reference = folds_dir / f"fold_{fold:02d}"
        fold_out = signature_root / f"fold_{fold:02d}"
        existing_manifest = fold_out / "manifest.json"
        if existing_manifest.exists() and not overwrite:
            print(
                f"[signature recovery] reuse signatures fold={fold:02d}",
                flush=True,
            )
        else:
            if fold_out.exists():
                shutil.rmtree(fold_out)
            print(
                f"[signature recovery] signatures fold={fold:02d}",
                flush=True,
            )
            extract_context_signatures(
                source_dir=inputs.source_dir,
                release_dir=inputs.preprocess_release_dir,
                chr44_profiles_path=inputs.chr44_profiles,
                out_dir=fold_out,
                config=analysis_config,
                reference_contexts_dir=fold_reference,
                roots=roots,
                min_context_size=min_context_size,
            )
        manifest = json.loads(existing_manifest.read_text())
        manifest_rows.append(
            {
                "fold": int(fold),
                "signature_dir": str(fold_out),
                "min_context_size": int(min_context_size),
                "roots": len(manifest.get("roots", [])),
            }
        )
    fold_manifest_path = out / "fold_signature_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(
        fold_manifest_path, sep="\t", index=False
    )
    return {
        "fold_signatures": signature_root,
        "fold_signature_manifest": fold_manifest_path,
    }


def _cosine(
    left: np.ndarray,
    right: np.ndarray,
    *,
    minimum_features: int,
) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(valid.sum()) < int(minimum_features):
        return np.nan
    first = first[valid]
    second = second[valid]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return np.nan
    return float(np.dot(first, second) / denominator)


def _normalized_chromosome(value: object) -> str:
    return str(value).strip().lower().removeprefix("chr")


def _track1_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["chromosome"].map(_normalized_chromosome)
        + "|"
        + frame["event"].astype(str)
    )


def _track2_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["gene_symbol"].astype(str)
        + "|"
        + frame["event"].astype(str)
    )


def _released_targets(
    inputs: SignatureRecoveryInputs,
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signatures = pd.read_csv(
        inputs.release_candidate_dir / "tables" / "signature_events.tsv",
        sep="\t",
        low_memory=False,
    )
    signatures = signatures.merge(
        core[["reference_key", "root", "final_context"]],
        on="reference_key",
        how="inner",
        validate="many_to_one",
    )
    track1 = signatures[signatures["track"].eq("track1")].copy()
    track1["target_key"] = (
        track1["feature"].map(_normalized_chromosome)
        + "|"
        + track1["event"].astype(str)
    )
    track2 = signatures[signatures["track"].eq("track2")].copy()
    track2["raw_event"] = [
        TRACK2_RELEASE_CHANNELS.get(
            (str(event).lower(), str(amplitude).lower())
        )
        for event, amplitude in track2[
            ["event", "amplitude"]
        ].itertuples(index=False, name=None)
    ]
    if track2["raw_event"].isna().any():
        invalid = track2.loc[
            track2["raw_event"].isna(),
            ["event", "amplitude"],
        ].drop_duplicates()
        raise ValueError(
            "Released Track 2 events have unknown channel mappings: "
            f"{invalid.to_dict(orient='records')}"
        )
    track2["target_key"] = (
        track2["feature"].astype(str)
        + "|"
        + track2["raw_event"].astype(str)
    )
    return track1, track2


def _group_frames(
    frame: pd.DataFrame,
) -> dict[tuple[str, int], pd.DataFrame]:
    if frame.empty:
        return {}
    result = frame.copy()
    result["root"] = result["root"].astype(str)
    result["context"] = pd.to_numeric(
        result["context"], errors="raise"
    ).astype(int)
    return {
        (str(root), int(context)): group.copy()
        for (root, context), group in result.groupby(
            ["root", "context"], sort=False
        )
    }


def _target_recovery(
    *,
    target: pd.DataFrame,
    fold_stats: pd.DataFrame,
    fold_supported: pd.DataFrame,
    key_builder,
    effect_column: str,
    minimum_features: int,
) -> dict[str, object]:
    n_target = int(len(target))
    if n_target == 0:
        return {
            "n_target_events": 0,
            "n_effects_observed": 0,
            "n_events_recovered": 0,
            "event_recall": np.nan,
            "effect_coverage": np.nan,
            "effect_cosine": np.nan,
            "direction_concordance": np.nan,
        }
    stats = fold_stats.copy()
    supported = fold_supported.copy()
    stats["target_key"] = (
        key_builder(stats) if not stats.empty else pd.Series(dtype=str)
    )
    supported["target_key"] = (
        key_builder(supported)
        if not supported.empty
        else pd.Series(dtype=str)
    )
    stats = stats.drop_duplicates("target_key", keep="first").set_index(
        "target_key"
    )
    supported_keys = set(supported["target_key"].astype(str))
    fold_effect = np.array(
        [
            (
                float(stats.loc[key, effect_column])
                if key in stats.index
                and pd.notna(stats.loc[key, effect_column])
                else 0.0
            )
            for key in target["target_key"].astype(str)
        ],
        dtype=float,
    )
    reference_effect = pd.to_numeric(
        target["reference_effect"], errors="coerce"
    ).to_numpy(float)
    target_directions = target["direction"].astype(str).str.lower().to_numpy()
    fold_directions = np.where(
        fold_effect > 0,
        "enriched",
        np.where(fold_effect < 0, "depleted", "missing"),
    )
    recovered = [
        str(key) in supported_keys
        and fold_direction == target_direction
        for key, fold_direction, target_direction in zip(
            target["target_key"].astype(str),
            fold_directions,
            target_directions,
            strict=True,
        )
    ]
    observed = np.array(
        [str(key) in stats.index for key in target["target_key"]],
        dtype=bool,
    )
    return {
        "n_target_events": n_target,
        "n_effects_observed": int(observed.sum()),
        "n_events_recovered": int(np.sum(recovered)),
        "event_recall": float(np.mean(recovered)),
        "effect_coverage": float(observed.mean()),
        "effect_cosine": _cosine(
            reference_effect,
            fold_effect,
            minimum_features=minimum_features,
        ),
        "direction_concordance": float(
            np.mean(fold_directions == target_directions)
        ),
    }


def _reference_target_effects(
    *,
    targets: pd.DataFrame,
    stats: pd.DataFrame,
    key_builder,
    effect_column: str,
) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    stats_by_context = _group_frames(stats)
    for (root, context), target in targets.groupby(
        ["root", "final_context"], sort=True
    ):
        reference = stats_by_context.get((str(root), int(context)))
        if reference is None:
            raise ValueError(
                f"[{root}-CTX{context}] reference signature stats are absent"
            )
        reference = reference.copy()
        reference["target_key"] = key_builder(reference)
        reference = reference.drop_duplicates(
            "target_key", keep="first"
        ).set_index("target_key")
        scoped = target.copy()
        scoped["reference_effect"] = [
            (
                float(reference.loc[key, effect_column])
                if key in reference.index
                and pd.notna(reference.loc[key, effect_column])
                else np.nan
            )
            for key in scoped["target_key"].astype(str)
        ]
        if scoped["reference_effect"].isna().any():
            missing = scoped.loc[
                scoped["reference_effect"].isna(), "target_key"
            ].astype(str)
            raise ValueError(
                f"[{root}-CTX{context}] released targets lack raw reference "
                f"effects: {missing.tolist()}"
            )
        result_parts.append(scoped)
    return (
        pd.concat(result_parts, ignore_index=True)
        if result_parts
        else targets.assign(reference_effect=pd.Series(dtype=float))
    )


def _effect_direction(value: float) -> str:
    if not np.isfinite(value) or value == 0.0:
        return "missing"
    return "enriched" if value > 0.0 else "depleted"


def score_heldout_track1_targets(
    *,
    profiles: pd.DataFrame,
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    root: str,
    fold: int,
) -> pd.DataFrame:
    """Estimate released Track 1 target effects in one held-out fold."""

    profile = profiles.copy()
    profile["analysis_id"] = profile["analysis_id"].astype(str)
    label = labels[["analysis_id", "full_final_context"]].copy()
    label["analysis_id"] = label["analysis_id"].astype(str)
    frame = label.merge(
        profile, on="analysis_id", how="left", validate="one_to_one"
    )
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        chromosome, event = str(target.target_key).split("|", 1)
        feature = f"CHR|chr{chromosome}_{event}"
        member = frame["full_final_context"].eq(int(target.final_context))
        rest = ~member
        effect = np.nan
        if (
            feature in frame.columns
            and int(member.sum()) >= 2
            and int(rest.sum()) >= 2
        ):
            left = pd.to_numeric(
                frame.loc[member, feature], errors="coerce"
            ).dropna()
            right = pd.to_numeric(
                frame.loc[rest, feature], errors="coerce"
            ).dropna()
            if len(left) >= 2 and len(right) >= 2:
                effect = float(left.median() - right.median())
        observed = bool(np.isfinite(effect))
        rows.append(
            {
                "root": str(root),
                "fold": int(fold),
                "reference_key": str(target.reference_key),
                "final_context": int(target.final_context),
                "target_key": str(target.target_key),
                "direction": str(target.direction).lower(),
                "reference_effect": float(target.reference_effect),
                "heldout_effect": effect,
                "effect_observed": observed,
                "direction_concordant": (
                    bool(
                        _effect_direction(effect)
                        == str(target.direction).lower()
                    )
                    if observed
                    else False
                ),
                "n_context": int(member.sum()),
                "n_rest": int(rest.sum()),
            }
        )
    return pd.DataFrame(rows)


def score_heldout_track2_targets(
    *,
    gene_panel: pd.DataFrame,
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    root: str,
    fold: int,
) -> pd.DataFrame:
    """Estimate burden-adjusted Track 2 target effects in held-out samples."""

    panel = gene_panel.copy()
    panel["analysis_id"] = panel["analysis_id"].astype(str)
    if "chrom" not in panel.columns:
        raise KeyError("Held-out Track 2 scoring requires gene chromosome")
    panel["chrom"] = (
        panel["chrom"].astype(str).str.replace("chr", "", regex=False)
    )
    panel = panel[
        panel["chrom"].isin({str(index) for index in range(1, 23)})
    ].copy()
    for event in EVENT_COLS:
        panel[event] = pd.to_numeric(panel[event], errors="coerce")
        burden = panel.groupby("analysis_id")[event].transform("mean")
        panel[f"{event}_adj"] = panel[event] - burden

    label = labels[["analysis_id", "full_final_context"]].copy()
    label["analysis_id"] = label["analysis_id"].astype(str)
    panel = panel.merge(
        label, on="analysis_id", how="inner", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        gene = str(target.feature)
        event = str(target.raw_event)
        scoped = panel[
            panel["gene_symbol"].astype(str).eq(gene)
        ].copy()
        member = scoped["full_final_context"].eq(
            int(target.final_context)
        )
        rest = ~member
        effect = np.nan
        raw_delta = np.nan
        n_context = int(scoped.loc[member, "analysis_id"].nunique())
        n_rest = int(scoped.loc[rest, "analysis_id"].nunique())
        if n_context >= 2 and n_rest >= 2:
            left = pd.to_numeric(
                scoped.loc[member, f"{event}_adj"], errors="coerce"
            ).dropna()
            right = pd.to_numeric(
                scoped.loc[rest, f"{event}_adj"], errors="coerce"
            ).dropna()
            raw_left = pd.to_numeric(
                scoped.loc[member, event], errors="coerce"
            ).dropna()
            raw_right = pd.to_numeric(
                scoped.loc[rest, event], errors="coerce"
            ).dropna()
            if len(left) >= 2 and len(right) >= 2:
                effect = float(left.mean() - right.mean())
            if len(raw_left) >= 2 and len(raw_right) >= 2:
                raw_delta = float(raw_left.mean() - raw_right.mean())
        observed = bool(np.isfinite(effect))
        rows.append(
            {
                "root": str(root),
                "fold": int(fold),
                "reference_key": str(target.reference_key),
                "final_context": int(target.final_context),
                "target_key": str(target.target_key),
                "direction": str(target.direction).lower(),
                "reference_effect": float(target.reference_effect),
                "heldout_effect": effect,
                "heldout_raw_prevalence_delta": raw_delta,
                "effect_observed": observed,
                "direction_concordant": (
                    bool(
                        _effect_direction(effect)
                        == str(target.direction).lower()
                    )
                    if observed
                    else False
                ),
                "n_context": n_context,
                "n_rest": n_rest,
            }
        )
    return pd.DataFrame(rows)


def compute_heldout_target_effects(
    *,
    inputs: SignatureRecoveryInputs,
    fold_assignments: pd.DataFrame,
    track1_targets: pd.DataFrame,
    track2_targets: pd.DataFrame,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score released target effects in samples omitted from each fold."""

    profiles = pd.read_parquet(inputs.chr44_profiles)
    profiles["analysis_id"] = profiles["analysis_id"].astype(str)
    cache_dir = inputs.source_dir / "gene_panel" / "cache"
    track1_parts: list[pd.DataFrame] = []
    track2_parts: list[pd.DataFrame] = []
    roots = sorted(fold_assignments["root"].astype(str).unique())
    for root_index, root in enumerate(roots, start=1):
        print(
            f"[signature recovery] held-out confirmation root={root} "
            f"({root_index}/{len(roots)})",
            flush=True,
        )
        root_assignments = fold_assignments[
            fold_assignments["root"].astype(str).eq(root)
        ].copy()
        root_t1 = track1_targets[
            track1_targets["root"].astype(str).eq(root)
        ]
        root_t2 = track2_targets[
            track2_targets["root"].astype(str).eq(root)
        ]
        root_profiles = profiles[
            profiles["analysis_id"].isin(
                set(root_assignments["analysis_id"].astype(str))
            )
        ]
        root_gene_panel = (
            load_gene_panel_subset_from_store(
                store_dir=cache_dir,
                analysis_ids=set(root_assignments["analysis_id"].astype(str)),
                gene_symbols=None,
                columns=None,
            )
            if not root_t2.empty
            else pd.DataFrame()
        )
        if not root_gene_panel.empty:
            root_gene_panel["analysis_id"] = root_gene_panel[
                "analysis_id"
            ].astype(str)
        for fold in range(int(n_folds)):
            labels = root_assignments[
                root_assignments["heldout_fold"].eq(int(fold))
            ].copy()
            if not root_t1.empty:
                track1_parts.append(
                    score_heldout_track1_targets(
                        profiles=root_profiles,
                        labels=labels,
                        targets=root_t1,
                        root=root,
                        fold=fold,
                    )
                )
            if not root_t2.empty:
                analysis_ids = set(labels["analysis_id"].astype(str))
                track2_parts.append(
                    score_heldout_track2_targets(
                        gene_panel=root_gene_panel[
                            root_gene_panel["analysis_id"].isin(analysis_ids)
                        ],
                        labels=labels,
                        targets=root_t2,
                        root=root,
                        fold=fold,
                    )
                )
    return (
        pd.concat(track1_parts, ignore_index=True)
        if track1_parts
        else pd.DataFrame(),
        pd.concat(track2_parts, ignore_index=True)
        if track2_parts
        else pd.DataFrame(),
    )


def _heldout_confirmation_by_context(
    *,
    core: pd.DataFrame,
    targets: pd.DataFrame,
    effects: pd.DataFrame,
    track: str,
    n_folds: int,
    minimum_features: int,
) -> pd.DataFrame:
    """Summarize held-out target confirmation for each context and fold."""

    target_groups = {
        (str(root), int(context)): group.copy()
        for (root, context), group in targets.groupby(
            ["root", "final_context"], sort=False
        )
    }
    effect_groups = {
        (str(root), int(fold), int(context)): group.copy()
        for (root, fold, context), group in effects.groupby(
            ["root", "fold", "final_context"], sort=False
        )
    }
    rows: list[dict[str, object]] = []
    prefix = f"heldout_{track}"
    for fold in range(int(n_folds)):
        for context in core.itertuples(index=False):
            key = (str(context.root), int(context.final_context))
            target = target_groups.get(key, pd.DataFrame())
            effect = effect_groups.get(
                (str(context.root), int(fold), int(context.final_context)),
                pd.DataFrame(),
            )
            n_target = int(len(target))
            if n_target:
                effect = effect.set_index("target_key").reindex(
                    target["target_key"].astype(str)
                )
                observed = _as_boolean(effect["effect_observed"])
                concordant = _as_boolean(effect["direction_concordant"])
                heldout_values = pd.to_numeric(
                    effect["heldout_effect"], errors="coerce"
                ).to_numpy(float)
                reference_values = pd.to_numeric(
                    target["reference_effect"], errors="coerce"
                ).to_numpy(float)
                n_observed = int(observed.sum())
                n_concordant = int((observed & concordant).sum())
                cosine = _cosine(
                    reference_values,
                    heldout_values,
                    minimum_features=minimum_features,
                )
            else:
                n_observed = 0
                n_concordant = 0
                cosine = np.nan
            rows.append(
                {
                    "root": str(context.root),
                    "fold": int(fold),
                    "reference_key": str(context.reference_key),
                    "final_context": int(context.final_context),
                    f"{prefix}_target_events": n_target,
                    f"{prefix}_effects_observed": n_observed,
                    f"{prefix}_direction_concordant_events": n_concordant,
                    f"{prefix}_effect_coverage": (
                        float(n_observed / n_target)
                        if n_target
                        else np.nan
                    ),
                    f"{prefix}_direction_concordance": (
                        float(n_concordant / n_target)
                        if n_target
                        else np.nan
                    ),
                    f"{prefix}_effect_cosine": cosine,
                }
            )
    return pd.DataFrame(rows)


def _track1_profile_cosine(
    reference: pd.DataFrame,
    folded: pd.DataFrame,
    *,
    minimum_features: int,
) -> float:
    if reference.empty:
        return np.nan
    left = reference.copy()
    left["profile_key"] = _track1_key(left)
    right = folded.copy()
    right["profile_key"] = (
        _track1_key(right) if not right.empty else pd.Series(dtype=str)
    )
    left = left.drop_duplicates("profile_key").set_index("profile_key")
    right = right.drop_duplicates("profile_key").set_index("profile_key")
    keys = left.index.astype(str)
    reference_values = pd.to_numeric(
        left.loc[keys, "median_delta"], errors="coerce"
    ).to_numpy(float)
    fold_values = np.array(
        [
            (
                float(right.loc[key, "median_delta"])
                if key in right.index
                and pd.notna(right.loc[key, "median_delta"])
                else 0.0
            )
            for key in keys
        ]
    )
    return _cosine(
        reference_values,
        fold_values,
        minimum_features=minimum_features,
    )


def compare_fold_signatures(
    *,
    membership: pd.DataFrame,
    core: pd.DataFrame,
    track1_targets: pd.DataFrame,
    track2_targets: pd.DataFrame,
    reference_track1_stats: pd.DataFrame,
    fold_track1_stats: pd.DataFrame,
    fold_track1_supported: pd.DataFrame,
    fold_track2_stats: pd.DataFrame,
    fold_track2_supported: pd.DataFrame,
    fold: int,
    minimum_features: int,
) -> pd.DataFrame:
    """Compare one fold against published core events and full effects."""

    fold_membership = membership[
        membership["fold"].eq(int(fold))
    ].copy()
    t1_reference_groups = _group_frames(reference_track1_stats)
    t1_stats_groups = _group_frames(fold_track1_stats)
    t1_supported_groups = _group_frames(fold_track1_supported)
    t2_stats_groups = _group_frames(fold_track2_stats)
    t2_supported_groups = _group_frames(fold_track2_supported)
    t1_target_groups = {
        (str(root), int(context)): group.copy()
        for (root, context), group in track1_targets.groupby(
            ["root", "final_context"], sort=False
        )
    }
    t2_target_groups = {
        (str(root), int(context)): group.copy()
        for (root, context), group in track2_targets.groupby(
            ["root", "final_context"], sort=False
        )
    }

    rows: list[dict[str, object]] = []
    for catalog_row in core.itertuples(index=False):
        root = str(catalog_row.root)
        context = int(catalog_row.final_context)
        key = (root, context)
        membership_row = fold_membership[
            fold_membership["root"].eq(root)
            & fold_membership["final_context"].eq(context)
        ]
        if len(membership_row) != 1:
            raise ValueError(
                f"[fold {fold}, {root}-CTX{context}] membership row absent"
            )
        membership_values = membership_row.iloc[0].to_dict()
        t1_target = t1_target_groups.get(
            key,
            pd.DataFrame(columns=["target_key", "direction", "reference_effect"]),
        )
        t2_target = t2_target_groups.get(
            key,
            pd.DataFrame(columns=["target_key", "direction", "reference_effect"]),
        )
        t1 = _target_recovery(
            target=t1_target,
            fold_stats=t1_stats_groups.get(key, pd.DataFrame()),
            fold_supported=t1_supported_groups.get(key, pd.DataFrame()),
            key_builder=_track1_key,
            effect_column="median_delta",
            minimum_features=minimum_features,
        )
        t2 = _target_recovery(
            target=t2_target,
            fold_stats=t2_stats_groups.get(key, pd.DataFrame()),
            fold_supported=t2_supported_groups.get(key, pd.DataFrame()),
            key_builder=_track2_key,
            effect_column="adj_delta",
            minimum_features=minimum_features,
        )
        rows.append(
            {
                **membership_values,
                "track1_target_events": t1["n_target_events"],
                "track1_effects_observed": t1["n_effects_observed"],
                "track1_events_recovered": t1["n_events_recovered"],
                "track1_event_recall": t1["event_recall"],
                "track1_effect_coverage": t1["effect_coverage"],
                "track1_effect_cosine": t1["effect_cosine"],
                "track1_direction_concordance": t1[
                    "direction_concordance"
                ],
                "track1_profile_cosine": _track1_profile_cosine(
                    t1_reference_groups.get(key, pd.DataFrame()),
                    t1_stats_groups.get(key, pd.DataFrame()),
                    minimum_features=minimum_features,
                ),
                "track2_target_events": t2["n_target_events"],
                "track2_effects_observed": t2["n_effects_observed"],
                "track2_events_recovered": t2["n_events_recovered"],
                "track2_event_recall": t2["event_recall"],
                "track2_effect_coverage": t2["effect_coverage"],
                "track2_effect_cosine": t2["effect_cosine"],
                "track2_direction_concordance": t2[
                    "direction_concordance"
                ],
            }
        )
    return pd.DataFrame(rows)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    total = float(pd.to_numeric(denominator, errors="coerce").sum())
    if total == 0:
        return np.nan
    return float(pd.to_numeric(numerator, errors="coerce").sum() / total)


def _aggregate_fold_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        detected_mask = _as_boolean(group["detected"])
        detected = group[detected_mask]
        row.update(
            {
                "n_contexts": int(len(group)),
                "n_contexts_detected": int(detected_mask.sum()),
                "n_contexts_track1": int(
                    group["track1_target_events"].gt(0).sum()
                ),
                "n_contexts_track2": int(
                    group["track2_target_events"].gt(0).sum()
                ),
                "membership_recall_sample_weighted": _ratio(
                    group["n_overlap"], group["n_truth"]
                ),
                "membership_precision_sample_weighted": _ratio(
                    group["n_overlap"], group["n_predicted"]
                ),
                "context_detection_fraction": float(detected_mask.mean()),
                "track1_event_recall_event_weighted": _ratio(
                    group["track1_events_recovered"],
                    group["track1_target_events"],
                ),
                "track2_event_recall_event_weighted": _ratio(
                    group["track2_events_recovered"],
                    group["track2_target_events"],
                ),
                "track1_event_recall_detected_event_weighted": _ratio(
                    detected["track1_events_recovered"],
                    detected["track1_target_events"],
                ),
                "track2_event_recall_detected_event_weighted": _ratio(
                    detected["track2_events_recovered"],
                    detected["track2_target_events"],
                ),
                "track1_profile_cosine_context_median": float(
                    pd.to_numeric(
                        group["track1_profile_cosine"], errors="coerce"
                    ).median()
                ),
                "track1_effect_cosine_context_median": float(
                    pd.to_numeric(
                        group["track1_effect_cosine"], errors="coerce"
                    ).median()
                ),
                "track2_effect_cosine_context_median": float(
                    pd.to_numeric(
                        group["track2_effect_cosine"], errors="coerce"
                    ).median()
                ),
                "heldout_track1_effect_coverage_event_weighted": _ratio(
                    group["heldout_track1_effects_observed"],
                    group["heldout_track1_target_events"],
                ),
                "heldout_track1_direction_concordance_event_weighted": _ratio(
                    group["heldout_track1_direction_concordant_events"],
                    group["heldout_track1_target_events"],
                ),
                "heldout_track2_effect_coverage_event_weighted": _ratio(
                    group["heldout_track2_effects_observed"],
                    group["heldout_track2_target_events"],
                ),
                "heldout_track2_direction_concordance_event_weighted": _ratio(
                    group["heldout_track2_direction_concordant_events"],
                    group["heldout_track2_target_events"],
                ),
                "heldout_track1_effect_cosine_context_median": float(
                    pd.to_numeric(
                        group["heldout_track1_effect_cosine"],
                        errors="coerce",
                    ).median()
                ),
                "heldout_track2_effect_cosine_context_median": float(
                    pd.to_numeric(
                        group["heldout_track2_effect_cosine"],
                        errors="coerce",
                    ).median()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_folds(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        row["n_folds"] = int(group["fold"].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = (
                float(values.median()) if not values.empty else np.nan
            )
            row[f"{metric}_minimum"] = (
                float(values.min()) if not values.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_signature_recovery(
    *,
    inputs: SignatureRecoveryInputs,
    out_dir: Path,
    params: SignatureRecoveryParams,
) -> dict[str, Path]:
    """Create five-fold recovery endpoints without post-hoc pass thresholds."""

    out = Path(out_dir).expanduser().resolve()
    membership_path = out / "fold_membership_metrics.parquet"
    fold_assignments_path = out / "fold_assignments.parquet"
    signature_root = out / "fold_signatures"
    for required in (
        membership_path,
        fold_assignments_path,
        signature_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    core = _core_reference_catalog(inputs)
    membership = pd.read_parquet(membership_path)
    fold_assignments = pd.read_parquet(fold_assignments_path)
    observed_folds = set(
        pd.to_numeric(
            fold_assignments["heldout_fold"], errors="raise"
        ).astype(int)
    )
    expected_folds = set(range(int(params.n_folds)))
    if observed_folds != expected_folds:
        raise ValueError(
            "Fold assignments disagree with validation configuration: "
            f"observed={sorted(observed_folds)}, "
            f"expected={sorted(expected_folds)}"
        )

    t1_targets, t2_targets = _released_targets(inputs, core)
    reference_t1 = pd.read_parquet(
        inputs.reference_signatures_dir / "track1_event_stats.parquet"
    )
    reference_t2 = pd.read_parquet(
        inputs.reference_signatures_dir / "track2_marker_stats.parquet"
    )
    t1_targets = _reference_target_effects(
        targets=t1_targets,
        stats=reference_t1,
        key_builder=_track1_key,
        effect_column="median_delta",
    )
    t2_targets = _reference_target_effects(
        targets=t2_targets,
        stats=reference_t2,
        key_builder=_track2_key,
        effect_column="adj_delta",
    )

    heldout_t1, heldout_t2 = compute_heldout_target_effects(
        inputs=inputs,
        fold_assignments=fold_assignments,
        track1_targets=t1_targets,
        track2_targets=t2_targets,
        n_folds=int(params.n_folds),
    )
    heldout_t1_path = out / "heldout_track1_target_effects.parquet"
    heldout_t2_path = out / "heldout_track2_target_effects.parquet"
    heldout_t1.to_parquet(heldout_t1_path, index=False)
    heldout_t2.to_parquet(heldout_t2_path, index=False)
    heldout_t1_context = _heldout_confirmation_by_context(
        core=core,
        targets=t1_targets,
        effects=heldout_t1,
        track="track1",
        n_folds=int(params.n_folds),
        minimum_features=int(params.min_effect_features),
    )
    heldout_t2_context = _heldout_confirmation_by_context(
        core=core,
        targets=t2_targets,
        effects=heldout_t2,
        track="track2",
        n_folds=int(params.n_folds),
        minimum_features=int(params.min_effect_features),
    )

    fold_parts: list[pd.DataFrame] = []
    for fold in range(int(params.n_folds)):
        fold_dir = signature_root / f"fold_{fold:02d}"
        required = {
            "track1_stats": fold_dir / "track1_event_stats.parquet",
            "track1_supported": (
                fold_dir / "track1_supported_events.parquet"
            ),
            "track2_stats": fold_dir / "track2_marker_stats.parquet",
            "track2_supported": (
                fold_dir / "track2_strict_evidence.parquet"
            ),
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Fold {fold} signature outputs are incomplete: {missing}"
            )
        fold_parts.append(
            compare_fold_signatures(
                membership=membership,
                core=core,
                track1_targets=t1_targets,
                track2_targets=t2_targets,
                reference_track1_stats=reference_t1,
                fold_track1_stats=pd.read_parquet(
                    required["track1_stats"]
                ),
                fold_track1_supported=pd.read_parquet(
                    required["track1_supported"]
                ),
                fold_track2_stats=pd.read_parquet(
                    required["track2_stats"]
                ),
                fold_track2_supported=pd.read_parquet(
                    required["track2_supported"]
                ),
                fold=fold,
                minimum_features=int(params.min_effect_features),
            )
        )
    fold_context = pd.concat(fold_parts, ignore_index=True)
    merge_columns = ["root", "fold", "reference_key", "final_context"]
    fold_context = fold_context.merge(
        heldout_t1_context,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    ).merge(
        heldout_t2_context,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )
    fold_context["context_detected"] = _as_boolean(
        fold_context["detected"]
    ).astype(float)
    fold_context_path = out / "fold_context_signature_metrics.parquet"
    fold_context.to_parquet(fold_context_path, index=False)

    fold_root = _aggregate_fold_metrics(fold_context, ["root", "fold"])
    fold_global = _aggregate_fold_metrics(
        fold_context.assign(scope="all"), ["scope", "fold"]
    )
    fold_root_path = out / "fold_root_signature_metrics.tsv"
    fold_global_path = out / "fold_global_signature_metrics.tsv"
    fold_root.to_csv(fold_root_path, sep="\t", index=False)
    fold_global.to_csv(fold_global_path, sep="\t", index=False)

    context_metrics = [
        "context_detected",
        "membership_recall",
        "membership_precision",
        "membership_jaccard",
        "track1_event_recall",
        "track1_effect_coverage",
        "track1_profile_cosine",
        "track1_effect_cosine",
        "track1_direction_concordance",
        "track2_event_recall",
        "track2_effect_coverage",
        "track2_effect_cosine",
        "track2_direction_concordance",
        "heldout_track1_effect_coverage",
        "heldout_track1_direction_concordance",
        "heldout_track1_effect_cosine",
        "heldout_track2_effect_coverage",
        "heldout_track2_direction_concordance",
        "heldout_track2_effect_cosine",
    ]
    context_summary = _summarize_folds(
        fold_context,
        ["root", "reference_key", "final_context"],
        context_metrics,
    )
    target_counts = fold_context.groupby(
        ["root", "reference_key", "final_context"], as_index=False
    ).agg(
        n_definition_members=("n_definition_members", "first"),
        track1_target_events=("track1_target_events", "first"),
        track2_target_events=("track2_target_events", "first"),
    )
    context_summary = context_summary.merge(
        target_counts,
        on=["root", "reference_key", "final_context"],
        how="left",
        validate="one_to_one",
    )
    context_summary_path = out / "context_signature_recovery.tsv"
    context_summary.to_csv(context_summary_path, sep="\t", index=False)

    aggregate_metrics = [
        "context_detection_fraction",
        "membership_recall_sample_weighted",
        "membership_precision_sample_weighted",
        "track1_event_recall_event_weighted",
        "track1_event_recall_detected_event_weighted",
        "track2_event_recall_event_weighted",
        "track2_event_recall_detected_event_weighted",
        "track1_profile_cosine_context_median",
        "track1_effect_cosine_context_median",
        "track2_effect_cosine_context_median",
        "heldout_track1_effect_coverage_event_weighted",
        "heldout_track1_direction_concordance_event_weighted",
        "heldout_track2_effect_coverage_event_weighted",
        "heldout_track2_direction_concordance_event_weighted",
        "heldout_track1_effect_cosine_context_median",
        "heldout_track2_effect_cosine_context_median",
    ]
    root_summary = _summarize_folds(
        fold_root, ["root"], aggregate_metrics
    )
    root_counts = fold_root.groupby("root", as_index=False).agg(
        n_contexts=("n_contexts", "first"),
        n_contexts_track1=("n_contexts_track1", "first"),
        n_contexts_track2=("n_contexts_track2", "first"),
    )
    root_summary = root_summary.merge(
        root_counts, on="root", how="left", validate="one_to_one"
    )
    root_summary_path = out / "root_signature_recovery.tsv"
    root_summary.to_csv(root_summary_path, sep="\t", index=False)

    global_summary = _summarize_folds(
        fold_global, ["scope"], aggregate_metrics
    )
    global_summary.insert(1, "n_roots", int(core["root"].nunique()))
    global_summary.insert(2, "n_contexts", int(len(core)))
    global_summary.insert(
        3,
        "n_contexts_track1",
        int(t1_targets["reference_key"].nunique()),
    )
    global_summary.insert(
        4,
        "n_contexts_track2",
        int(t2_targets["reference_key"].nunique()),
    )
    global_summary.insert(
        5,
        "train_fraction",
        float((int(params.n_folds) - 1) / int(params.n_folds)),
    )
    global_summary_path = out / "signature_recovery_summary.tsv"
    global_summary.to_csv(global_summary_path, sep="\t", index=False)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cnvsim_signature_recovery_v2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "validation_role": "post_selection_signature_robustness",
                "primary_question": (
                    "Are released core-context CNV signatures recovered in "
                    "five complementary 80% rebuilds with the atlas "
                    "parameters fixed, and are their effect directions "
                    "confirmed in the omitted samples?"
                ),
                "not_an_independent_projection_endpoint": True,
                "mapping_uses_signature_evidence": False,
                "fold_labels_used_for": (
                    "context-balanced fold assignment and evaluation only"
                ),
                "params": asdict(params),
                "target_policy": (
                    "Only released core Track 1 and Track 2 events in the "
                    "current release candidate are recovery targets"
                ),
                "heldout_effect_policy": (
                    "Estimate only released target effects in each omitted "
                    "fold; do not repeat marker discovery, significance "
                    "testing, or platform/source selection"
                ),
                "threshold_policy": (
                    "Continuous fold median and minimum endpoints; no "
                    "post-hoc binary reproduction threshold"
                ),
                "counts": {
                    "roots": int(core["root"].nunique()),
                    "core_contexts": int(len(core)),
                    "track1_target_events": int(len(t1_targets)),
                    "track2_target_events": int(len(t2_targets)),
                    "folds": int(params.n_folds),
                },
                "outputs": {
                    "fold_context_signature_metrics": str(
                        fold_context_path
                    ),
                    "fold_root_signature_metrics": str(fold_root_path),
                    "fold_global_signature_metrics": str(
                        fold_global_path
                    ),
                    "heldout_track1_target_effects": str(heldout_t1_path),
                    "heldout_track2_target_effects": str(heldout_t2_path),
                    "context_signature_recovery": str(
                        context_summary_path
                    ),
                    "root_signature_recovery": str(root_summary_path),
                    "signature_recovery_summary": str(global_summary_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "fold_context_signature_metrics": fold_context_path,
        "fold_root_signature_metrics": fold_root_path,
        "fold_global_signature_metrics": fold_global_path,
        "heldout_track1_target_effects": heldout_t1_path,
        "heldout_track2_target_effects": heldout_t2_path,
        "context_signature_recovery": context_summary_path,
        "root_signature_recovery": root_summary_path,
        "signature_recovery_summary": global_summary_path,
        "manifest": manifest_path,
    }
