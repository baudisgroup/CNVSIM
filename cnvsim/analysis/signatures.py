from __future__ import annotations

"""Extract chromosome-scale and gene-level evidence for CNV contexts."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.analysis.chromosome_architecture import compute_track1_evidence
from cnvsim.analysis.config import AnalysisConfig
from cnvsim.analysis.subtyping import compute_track2_signatures
from cnvsim.preprocess.encode import EVENT_COLS
from cnvsim.preprocess.gene_panel_store import (
    load_gene_panel_subset_from_store,
    resolve_gene_fraction_store,
)


ASSIGNMENT_SUFFIXES = (
    "_final_partition.parquet",
    "_provisional_partition.parquet",
    "_contexts.parquet",
    "_clusters_leiden_cpm.parquet",
)
LABEL_COLUMNS = ("final_context", "provisional_context", "context", "cluster")


def _load_root_meta(release_dir: Path, root: str) -> pd.DataFrame:
    candidates = [
        release_dir / "final_space" / "meta_final" / f"{root}_meta_final.tsv",
        release_dir
        / "reference_space"
        / "final_space"
        / "meta_final"
        / f"{root}_meta_final.tsv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Missing meta_final for {root}. Checked: {candidates}")
    frame = pd.read_csv(path, sep="\t", dtype={"analysis_id": str}, low_memory=False)
    required = ["analysis_id", "platform_family_refined"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"[{root}] meta_final missing columns: {missing}")
    return frame[required].drop_duplicates("analysis_id", keep="first").copy()


def _discover_roots(assignments_dir: Path) -> list[str]:
    roots: set[str] = set()
    for suffix in ASSIGNMENT_SUFFIXES:
        for path in assignments_dir.rglob(f"*{suffix}"):
            roots.add(path.name.removesuffix(suffix))
    return sorted(roots)


def _load_assignments(assignments_dir: Path, root: str) -> pd.DataFrame:
    candidates = [assignments_dir / f"{root}{suffix}" for suffix in ASSIGNMENT_SUFFIXES]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        matches = [
            candidate
            for suffix in ASSIGNMENT_SUFFIXES
            for candidate in assignments_dir.rglob(f"{root}{suffix}")
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one assignment table for {root} below {assignments_dir}; "
                f"found {matches}"
            )
        path = matches[0]
    frame = pd.read_parquet(path)
    label_column = next((column for column in LABEL_COLUMNS if column in frame.columns), None)
    if "analysis_id" not in frame.columns or label_column is None:
        raise KeyError(f"Assignment table lacks analysis_id/context labels: {path}")
    out = frame[["analysis_id", label_column]].rename(
        columns={label_column: "context"}
    )
    out["analysis_id"] = out["analysis_id"].astype(str)
    out["context"] = pd.to_numeric(out["context"], errors="coerce")
    out = out[out["context"].notna() & out["context"].ge(0)].copy()
    out["context"] = out["context"].astype(int)
    return out.drop_duplicates("analysis_id", keep="first")


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy() if frame is not None else pd.DataFrame()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns]


def _gene_chromosome_map(gene_panel: pd.DataFrame) -> pd.DataFrame:
    genes = gene_panel[["gene_symbol", "chrom"]].dropna().copy()
    genes["chromosome"] = (
        genes["chrom"].astype(str).str.replace("chr", "", regex=False)
    )
    genes = genes[genes["chromosome"].isin({str(index) for index in range(1, 23)})]
    counts = genes.groupby("gene_symbol")["chromosome"].nunique()
    unambiguous = set(counts[counts.eq(1)].index.astype(str))
    return (
        genes[genes["gene_symbol"].astype(str).isin(unambiguous)]
        .drop_duplicates("gene_symbol", keep="first")[["gene_symbol", "chromosome"]]
    )


def _link_track2_to_architecture(
    strict_track2: pd.DataFrame,
    track1_supported: pd.DataFrame,
) -> pd.DataFrame:
    result = strict_track2.copy()
    if result.empty:
        result["event_group"] = pd.Series(dtype="object")
        result["evidence_direction"] = pd.Series(dtype="object")
        result["architecture_linked"] = pd.Series(dtype=bool)
        return result

    result["event_group"] = result["event"].map(
        {
            "dup_frac": "gain",
            "hldup_frac": "gain",
            "del_frac": "loss",
            "hldel_frac": "loss",
        }
    )
    result["evidence_direction"] = np.where(
        pd.to_numeric(result["adj_delta"], errors="coerce").ge(0),
        "enriched",
        "depleted",
    )
    support_keys = {
        (
            int(row.context),
            str(row.chromosome).replace("chr", ""),
            str(row.event),
            str(row.evidence_direction),
        )
        for row in track1_supported.itertuples(index=False)
    }
    result["architecture_linked"] = [
        (
            int(row.context),
            str(row.chromosome).replace("chr", ""),
            str(row.event_group),
            str(row.evidence_direction),
        )
        in support_keys
        for row in result.itertuples(index=False)
    ]
    return result


def extract_context_signatures(
    *,
    source_dir: Path,
    release_dir: Path,
    chr44_profiles_path: Path,
    out_dir: Path,
    config: AnalysisConfig,
    assignments_dir: Path | None = None,
    reference_contexts_dir: Path | None = None,
    roots: list[str] | None = None,
    min_context_size: int | None = None,
) -> dict[str, Path]:
    """Extract Track 1 and Track 2 evidence for supplied CNV contexts."""

    if (assignments_dir is None) == (reference_contexts_dir is None):
        raise ValueError(
            "Provide exactly one of assignments_dir or reference_contexts_dir"
        )

    source_dir = Path(source_dir).expanduser().resolve()
    release_dir = Path(release_dir).expanduser().resolve()
    reference_contexts_dir = (
        Path(reference_contexts_dir).expanduser().resolve()
        if reference_contexts_dir is not None
        else None
    )
    assignments_dir = (
        reference_contexts_dir / "anchor_partitions"
        if reference_contexts_dir is not None
        else Path(assignments_dir).expanduser().resolve()
    )
    chr44_path = Path(chr44_profiles_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_reference_status = pd.DataFrame()
    context_stability = pd.DataFrame()
    if reference_contexts_dir is not None:
        status_path = reference_contexts_dir / "sample_reference_status.parquet"
        stability_path = reference_contexts_dir / "context_stability.tsv"
        for required in (assignments_dir, status_path, stability_path):
            if not required.exists():
                raise FileNotFoundError(required)
        sample_reference_status = pd.read_parquet(status_path)
        sample_reference_status["analysis_id"] = sample_reference_status[
            "analysis_id"
        ].astype(str)
        sample_reference_status["root"] = sample_reference_status["root"].astype(str)
        context_stability = pd.read_csv(stability_path, sep="\t")
        context_stability["root"] = context_stability["root"].astype(str)

    gene_panel_store = resolve_gene_fraction_store(source_dir)
    for required in (
        gene_panel_store / "manifest.json",
        gene_panel_store / "analysis_index.parquet",
        chr44_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing frozen input: {required}")

    excluded_roots = set(config.excluded_roots)
    if roots is None:
        roots = _discover_roots(assignments_dir)
    else:
        roots = [str(root).strip() for root in roots if str(root).strip()]
    roots = [root for root in roots if root not in excluded_roots]
    if not roots:
        raise ValueError("No roots remain after applying excluded_roots")

    chr44_profiles = pd.read_parquet(chr44_path)
    if "analysis_id" not in chr44_profiles.columns:
        raise KeyError("chr44 profile table is missing analysis_id")
    chr44_profiles["analysis_id"] = chr44_profiles["analysis_id"].astype(str)

    track1_profiles_parts: list[pd.DataFrame] = []
    track1_stats_parts: list[pd.DataFrame] = []
    track1_supported_parts: list[pd.DataFrame] = []
    track2_stats_parts: list[pd.DataFrame] = []
    track2_candidates_parts: list[pd.DataFrame] = []
    track2_strict_parts: list[pd.DataFrame] = []
    evidence_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    min_cluster_size = int(
        min_context_size
        if min_context_size is not None
        else config.leiden.min_cluster_size
    )

    for index, root in enumerate(roots, start=1):
        print(f"[{index}/{len(roots)}] context evidence: {root}", flush=True)
        assignments = _load_assignments(assignments_dir, root)
        metadata = _load_root_meta(release_dir, root)
        keep_ids = set(assignments["analysis_id"])
        background_ids: set[str] = set()
        if reference_contexts_dir is not None:
            root_status = sample_reference_status[
                sample_reference_status["root"].eq(root)
            ].copy()
            expected_anchor_ids = set(
                root_status.loc[
                    root_status["assignment_status"].eq("anchor_member"),
                    "analysis_id",
                ]
            )
            if keep_ids != expected_anchor_ids:
                raise ValueError(
                    f"[{root}] anchor partition/status mismatch: "
                    f"partition_only={len(keep_ids - expected_anchor_ids)}, "
                    f"status_only={len(expected_anchor_ids - keep_ids)}"
                )
            background_ids = set(
                root_status.loc[
                    root_status["assignment_status"].eq("non_anchor_query"),
                    "analysis_id",
                ]
            )
        evidence_ids = keep_ids | background_ids
        root_profiles = chr44_profiles[
            chr44_profiles["analysis_id"].isin(evidence_ids)
        ].copy()
        if len(root_profiles) != len(evidence_ids):
            missing = evidence_ids - set(root_profiles["analysis_id"])
            raise ValueError(f"[{root}] {len(missing)} evidence samples lack chr44 profiles")

        t1_profiles, t1_stats, t1_supported = compute_track1_evidence(
            profiles=root_profiles,
            assignments=assignments,
            metadata=metadata,
            root=root,
            background_analysis_ids=background_ids,
            min_platform_samples=config.signatures.min_platform_samples,
            q_cut=config.signatures.track1_q_cut,
            min_abs_median_delta=(
                config.signatures.track1_min_abs_median_delta
            ),
            min_abs_auc_shift=config.signatures.track1_min_abs_auc_shift,
            min_platform_delta=config.signatures.track1_min_platform_delta,
            min_platforms_supporting=(
                config.signatures.track1_min_platforms_supporting
            ),
        )
        track1_profiles_parts.append(t1_profiles)
        track1_stats_parts.append(t1_stats)
        track1_supported_parts.append(t1_supported)

        gene_panel = load_gene_panel_subset_from_store(
            store_dir=gene_panel_store,
            analysis_ids=evidence_ids,
            gene_symbols=None,
            columns=None,
        )
        if gene_panel.empty:
            raise ValueError(f"[{root}] no gene-panel rows found for context members")
        track2 = compute_track2_signatures(
            root=root,
            gene_panel=gene_panel,
            clusters_df=assignments.rename(columns={"context": "cluster"}),
            meta_df=metadata,
            event_cols=list(EVENT_COLS),
            min_cluster_size=min_cluster_size,
            autosomes_only_track2=True,
            top_n=config.signatures.track2_top_n,
            q_cut=config.signatures.track2_q_cut,
            min_abs_prevalence_delta=(
                config.signatures.track2_min_abs_prevalence_delta
            ),
            min_abs_auc_shift=config.signatures.track2_min_abs_auc_shift,
            min_platform_samples=config.signatures.min_platform_samples,
            min_platforms_supporting=(
                config.signatures.track2_min_platforms_supporting
            ),
            min_platform_support_fraction=(
                config.signatures.track2_min_platform_support_fraction
            ),
            max_auc_gap_allowed=config.signatures.track2_max_auc_gap,
            background_analysis_ids=background_ids,
        )
        marker_stats = track2["track2_markers"]["marker_stats"].copy()
        candidates = track2["track2_markers"]["signatures"].copy()
        platform_stats = track2["track2_platform_marker_stats"].copy()
        stability = track2["track2_platform_stability"].copy()
        if marker_stats.empty and not {
            "cluster",
            "event",
            "gene_symbol",
        }.issubset(marker_stats.columns):
            marker_stats = pd.DataFrame(
                columns=[
                    "cluster",
                    "event",
                    "gene_symbol",
                    "adj_cluster_mean",
                    "adj_rest_mean",
                    "adj_delta",
                    "auc",
                    "p_value",
                    "q_value",
                    "score",
                    "raw_prev_cluster",
                    "raw_prev_rest",
                    "raw_prev_delta",
                    "n_cluster_samples",
                    "n_rest_samples",
                ]
            )
        if candidates.empty and not {
            "cluster",
            "event",
            "gene_symbol",
        }.issubset(candidates.columns):
            candidates = pd.DataFrame(
                columns=[
                    "cluster",
                    "event",
                    "gene_symbol",
                    "adj_cluster_mean",
                    "adj_rest_mean",
                    "adj_delta",
                    "auc",
                    "p_value",
                    "q_value",
                    "score",
                    "raw_prev_cluster",
                    "raw_prev_rest",
                    "raw_prev_delta",
                ]
            )
        for frame in (marker_stats, candidates, platform_stats, stability):
            # Empty tables keep the same keys so the strict-support merge works.
            frame.rename(columns={"cluster": "context"}, inplace=True)
            if "root" not in frame.columns:
                frame.insert(0, "root", str(root))

        chromosome_map = _gene_chromosome_map(gene_panel)
        if not marker_stats.empty:
            marker_stats = marker_stats.merge(chromosome_map, on="gene_symbol", how="left")
        elif "chromosome" not in marker_stats.columns:
            marker_stats["chromosome"] = pd.Series(dtype="object")
        if not candidates.empty:
            candidates = candidates.merge(chromosome_map, on="gene_symbol", how="left")
        elif "chromosome" not in candidates.columns:
            candidates["chromosome"] = pd.Series(dtype="object")

        support_columns = [
            "root",
            "context",
            "event",
            "gene_symbol",
            "n_platforms_tested",
            "n_platforms_supporting",
            "support_fraction",
            "direction_consistent",
            "max_auc_gap",
            "max_prev_delta_gap",
            "single_platform_driven",
            "platform_support_class",
            "passes_strict_support",
        ]
        stability = _ensure_columns(stability, support_columns)
        strict = candidates.merge(
            stability,
            on=["root", "context", "event", "gene_symbol"],
            how="left",
            validate="one_to_one",
        )
        strict["passes_strict_support"] = strict["passes_strict_support"].fillna(False)
        strict = strict[strict["passes_strict_support"].astype(bool)].copy()
        strict = _link_track2_to_architecture(strict, t1_supported)

        track2_stats_parts.append(marker_stats)
        track2_candidates_parts.append(candidates)
        track2_strict_parts.append(strict)

        context_sizes = assignments.groupby("context").size().rename("n_samples")
        evidence = context_sizes.reset_index()
        t1_counts = (
            t1_supported.groupby("context").size().rename("n_track1_events")
            if "context" in t1_supported.columns
            else pd.Series(dtype=int, name="n_track1_events")
        )
        t2_counts = strict.groupby("context").size().rename("n_track2_markers")
        linked_counts = (
            strict[strict.get("architecture_linked", False).astype(bool)]
            .groupby("context")
            .size()
            .rename("n_architecture_linked_markers")
            if not strict.empty
            else pd.Series(dtype=int, name="n_architecture_linked_markers")
        )
        evidence = evidence.join(t1_counts, on="context").join(t2_counts, on="context")
        evidence = evidence.join(linked_counts, on="context").fillna(0)
        for column in (
            "n_track1_events",
            "n_track2_markers",
            "n_architecture_linked_markers",
        ):
            evidence[column] = evidence[column].astype(int)
        evidence["evidence_class"] = np.select(
            [
                evidence["n_track1_events"].gt(0) & evidence["n_track2_markers"].gt(0),
                evidence["n_track1_events"].gt(0),
                evidence["n_track2_markers"].gt(0),
            ],
            [
                "dual_supported",
                "chromosome_supported",
                "gene_supported",
            ],
            default="reference_context_only",
        )
        evidence.insert(0, "root", str(root))
        if reference_contexts_dir is not None:
            stability_columns = [
                "root",
                "final_context",
                "recovery_class",
                "detection_rate",
                "conditional_recall_q10",
                "conditional_precision_q10",
                "conditional_jaccard_q10",
                "is_reference_anchor",
            ]
            root_stability = context_stability[
                context_stability["root"].eq(root)
            ][stability_columns].rename(columns={"final_context": "context"})
            evidence = evidence.merge(
                root_stability,
                on=["root", "context"],
                how="left",
                validate="one_to_one",
            )
            if not evidence["is_reference_anchor"].fillna(False).all():
                raise ValueError(f"[{root}] non-anchor context reached signature output")
        evidence_parts.append(evidence)

        marker_stats.to_parquet(out_dir / f"{root}_track2_marker_stats.parquet", index=False)
        candidates.to_parquet(out_dir / f"{root}_track2_candidates.parquet", index=False)
        platform_stats.to_parquet(
            out_dir / f"{root}_track2_platform_stats.parquet", index=False
        )
        stability.to_parquet(
            out_dir / f"{root}_track2_platform_support.parquet", index=False
        )
        strict.to_parquet(out_dir / f"{root}_track2_strict_evidence.parquet", index=False)

        manifest_rows.append(
            {
                "root": root,
                "n_contexts": int(assignments["context"].nunique()),
                "n_samples": int(len(assignments)),
                "n_background_samples": int(len(background_ids)),
                "n_evidence_samples": int(len(evidence_ids)),
                "n_track1_supported_events": int(len(t1_supported)),
                "n_track2_candidates": int(len(candidates)),
                "n_track2_strict_markers": int(len(strict)),
                "n_architecture_linked_markers": int(
                    strict.get("architecture_linked", pd.Series(dtype=bool)).sum()
                ),
            }
        )

    outputs = {
        "track1_context_profiles": pd.concat(track1_profiles_parts, ignore_index=True),
        "track1_event_stats": pd.concat(track1_stats_parts, ignore_index=True),
        "track1_supported_events": pd.concat(track1_supported_parts, ignore_index=True),
        "track2_marker_stats": pd.concat(track2_stats_parts, ignore_index=True),
        "track2_candidates": pd.concat(track2_candidates_parts, ignore_index=True),
        "track2_strict_evidence": pd.concat(track2_strict_parts, ignore_index=True),
        "context_evidence_classification": pd.concat(evidence_parts, ignore_index=True),
    }
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[name] = str(path)

    manifest_table = pd.DataFrame(manifest_rows).sort_values("root")
    manifest_table_path = out_dir / "signature_manifest.tsv"
    manifest_table.to_csv(manifest_table_path, sep="\t", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "release_dir": str(release_dir),
        "assignments_dir": str(assignments_dir),
        "reference_contexts_dir": (
            str(reference_contexts_dir) if reference_contexts_dir else None
        ),
        "signature_scope": (
            "stable_reference_anchors"
            if reference_contexts_dir is not None
            else "provided_assignments"
        ),
        "rest_definition": (
            "other anchors plus non-anchor queries in the same root"
            if reference_contexts_dir is not None
            else "other assigned contexts in the same root"
        ),
        "chr44_profiles": str(chr44_path),
        "roots": roots,
        "min_context_size": min_cluster_size,
        "track1_definition": "whole-genome segment union coverage on 22 autosomes x gain/loss",
        "track2_channels": list(EVENT_COLS),
        "evidence_thresholds": asdict(config.signatures),
        "outputs": output_paths,
        "manifest_table": str(manifest_table_path),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        **{name: Path(path) for name, path in output_paths.items()},
        "manifest_table": manifest_table_path,
        "manifest": manifest_path,
    }
