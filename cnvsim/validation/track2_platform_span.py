from __future__ import annotations

"""Assess whether mixed Track 2 segment spans are associated with platform."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from cnvsim.analysis.chromosome_architecture import normalize_chromosome
from cnvsim.validation.signature_architecture import (
    _load_definition_members,
    _load_gene_intervals,
    _marker_segment_channels,
    _read_segments,
    _union_length,
)


EVENT_KEY = ["reference_key", "feature", "event", "amplitude"]
PLATFORM_FAMILY_COLUMN = "platform_family_refined"
PLATFORM_ID_COLUMN = "platform_id"
PLATFORM_LABEL_COLUMN = "platform_label"
PLATFORM_DETAIL_COLUMN = "platform_detail"
SOURCE_COLUMN = "source_group_refined"


@dataclass(frozen=True)
class Track2PlatformSpanParams:
    """Thresholds for the mixed-event platform sensitivity analysis."""

    minimum_carriers_per_platform: int = 10
    minimum_gene_coverage: float = 0.50
    focal_fraction_chromosome: float = 0.02
    broad_fraction_chromosome: float = 0.50
    dominance_fraction: float = 0.60
    mixed_fraction: float = 0.25
    association_fdr: float = 0.05
    minimum_epsilon_squared: float = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Track 2 platform-span inputs: " + ", ".join(missing)
        )


def _validate_params(params: Track2PlatformSpanParams) -> None:
    if params.minimum_carriers_per_platform < 2:
        raise ValueError("minimum_carriers_per_platform must be at least 2")
    fractions = {
        "minimum_gene_coverage": params.minimum_gene_coverage,
        "focal_fraction_chromosome": params.focal_fraction_chromosome,
        "broad_fraction_chromosome": params.broad_fraction_chromosome,
        "dominance_fraction": params.dominance_fraction,
        "mixed_fraction": params.mixed_fraction,
        "association_fdr": params.association_fdr,
        "minimum_epsilon_squared": params.minimum_epsilon_squared,
    }
    invalid = {
        name: value
        for name, value in fractions.items()
        if not 0.0 <= float(value) <= 1.0
    }
    if invalid:
        raise ValueError(f"Platform-span fractions must be within [0, 1]: {invalid}")
    if params.focal_fraction_chromosome >= params.broad_fraction_chromosome:
        raise ValueError(
            "focal_fraction_chromosome must be below broad_fraction_chromosome"
        )
    if params.mixed_fraction >= params.dominance_fraction:
        raise ValueError("mixed_fraction must be below dominance_fraction")


def _span_class(values: pd.Series, params: Track2PlatformSpanParams) -> str:
    span = pd.to_numeric(values, errors="coerce").dropna()
    if span.empty:
        return "no_segment_support"
    fractions = {
        "focal": float(span.le(params.focal_fraction_chromosome).mean()),
        "regional": float(
            (
                span.gt(params.focal_fraction_chromosome)
                & span.lt(params.broad_fraction_chromosome)
            ).mean()
        ),
        "broad": float(span.ge(params.broad_fraction_chromosome).mean()),
    }
    if sum(value >= params.mixed_fraction for value in fractions.values()) >= 2:
        return "mixed"
    for label in ("focal", "regional", "broad"):
        if fractions[label] >= params.dominance_fraction:
            return label
    return "unresolved"


def _span_fractions(
    values: pd.Series,
    params: Track2PlatformSpanParams,
) -> dict[str, float]:
    span = pd.to_numeric(values, errors="coerce").dropna()
    if span.empty:
        return {"focal": np.nan, "regional": np.nan, "broad": np.nan}
    return {
        "focal": float(span.le(params.focal_fraction_chromosome).mean()),
        "regional": float(
            (
                span.gt(params.focal_fraction_chromosome)
                & span.lt(params.broad_fraction_chromosome)
            ).mean()
        ),
        "broad": float(span.ge(params.broad_fraction_chromosome).mean()),
    }


def _bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = numeric.loc[order].to_numpy(dtype=float)
    n_values = len(ranked)
    adjusted = ranked * n_values / np.arange(1, n_values + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def _format_counts(frame: pd.DataFrame, label: str) -> str:
    if frame.empty:
        return ""
    counts = frame[label].astype(str).value_counts().sort_index()
    return ";".join(f"{key}={int(value)}" for key, value in counts.items())


def _format_platform_classes(
    frame: pd.DataFrame,
    group_column: str,
) -> str:
    if frame.empty:
        return ""
    ordered = frame.sort_values(group_column)
    return ";".join(
        f"{row[group_column]}={row['platform_span_class']}"
        for _, row in ordered.iterrows()
    )


def _categorical_audit_status(platforms: pd.DataFrame) -> str:
    if len(platforms) < 2:
        return "insufficient_platform_coverage"
    classes = platforms["platform_span_class"].astype(str)
    if classes.eq("mixed").any():
        return "mixed_persists_within_platform"
    if classes.eq("unresolved").any():
        return "within_platform_unresolved"
    if classes.nunique() > 1:
        return "platform_separated_span_classes"
    return "consistent_span_class_across_platforms"


def _interpretation(status: str) -> str:
    return {
        "insufficient_platform_coverage": (
            "Too few segment carriers for comparison across platform groups."
        ),
        "mixed_persists_within_platform": (
            "Mixed segment spans persist within at least one platform group."
        ),
        "within_platform_unresolved": (
            "At least one platform group has no dominant segment-span class."
        ),
        "platform_separated_span_classes": (
            "Eligible platform groups have different predominant span classes."
        ),
        "consistent_span_class_across_platforms": (
            "Eligible platform groups share the same predominant span class."
        ),
    }[status]


def _load_metadata(path: Path) -> pd.DataFrame:
    columns = [
        "analysis_id",
        PLATFORM_FAMILY_COLUMN,
        PLATFORM_ID_COLUMN,
        PLATFORM_LABEL_COLUMN,
        SOURCE_COLUMN,
    ]
    metadata = pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    _require_columns(metadata, set(columns), "metadata")
    metadata["analysis_id"] = metadata["analysis_id"].astype(str)
    if metadata["analysis_id"].duplicated().any():
        raise ValueError("Metadata analysis IDs must be unique")
    for column in (
        PLATFORM_FAMILY_COLUMN,
        PLATFORM_ID_COLUMN,
        PLATFORM_LABEL_COLUMN,
    ):
        metadata[column] = metadata[column].astype("string").fillna("Unknown")
    metadata[PLATFORM_DETAIL_COLUMN] = (
        metadata[PLATFORM_ID_COLUMN].astype(str)
        + " | "
        + metadata[PLATFORM_LABEL_COLUMN].astype(str)
    )
    metadata[SOURCE_COLUMN] = (
        metadata[SOURCE_COLUMN].astype("string").fillna("Unknown")
    )
    return metadata


def _mixed_events(
    signatures: pd.DataFrame,
    references: pd.DataFrame,
    genes: pd.DataFrame,
) -> pd.DataFrame:
    required = set(EVENT_KEY) | {
        "track",
        "direction",
        "segment_span_class",
        "segment_support_level",
        "track1_relation",
    }
    _require_columns(signatures, required, "signature_events")
    events = signatures[
        signatures["track"].eq("track2")
        & signatures["direction"].eq("enriched")
        & signatures["segment_span_class"].eq("mixed")
    ].copy()
    if events.empty:
        raise ValueError("No released mixed Track 2 events were found")
    if events.duplicated(EVENT_KEY).any():
        raise ValueError("Mixed Track 2 event keys must be unique")
    events = events.merge(
        references[["reference_key", "root", "reference_level"]],
        on="reference_key",
        how="left",
        validate="many_to_one",
    ).merge(
        genes,
        left_on="feature",
        right_on="gene_symbol",
        how="left",
        validate="many_to_one",
    )
    if events[["root", "reference_level", "gene_chromosome"]].isna().any().any():
        raise ValueError("Mixed Track 2 events have incomplete reference or gene metadata")
    return events


def _carrier_spans(
    *,
    events: pd.DataFrame,
    members: pd.DataFrame,
    metadata: pd.DataFrame,
    segments: pd.DataFrame,
    params: Track2PlatformSpanParams,
) -> pd.DataFrame:
    member_metadata = members[["analysis_id", "reference_key"]].merge(
        metadata,
        on="analysis_id",
        how="left",
        validate="one_to_one",
    )
    if member_metadata[
        [PLATFORM_FAMILY_COLUMN, PLATFORM_ID_COLUMN, PLATFORM_LABEL_COLUMN]
    ].isna().any().any():
        raise ValueError("Definition members are missing platform metadata")
    labelled_segments = segments.merge(
        member_metadata,
        on="analysis_id",
        how="inner",
        validate="many_to_one",
    )
    segment_groups = {
        (str(reference_key), str(channel), str(chromosome)): group
        for (reference_key, channel, chromosome), group in labelled_segments.groupby(
            ["reference_key", "segment_event_channel", "chromosome"],
            sort=False,
        )
    }

    rows: list[pd.DataFrame] = []
    for index, marker in enumerate(events.itertuples(index=False), start=1):
        if index == 1 or index % 250 == 0 or index == len(events):
            print(f"[Track 2 platform span] event {index}/{len(events)}")
        chromosome = normalize_chromosome(marker.gene_chromosome)
        channels = _marker_segment_channels(str(marker.event), str(marker.amplitude))
        groups = [
            segment_groups[(str(marker.reference_key), channel, chromosome)]
            for channel in channels
            if (str(marker.reference_key), channel, chromosome) in segment_groups
        ]
        if not groups:
            continue
        group = pd.concat(groups, ignore_index=True)
        start = float(marker.gene_start)
        end = float(marker.gene_end)
        overlap = group[group["start"].lt(end) & group["end"].gt(start)].copy()
        if overlap.empty:
            continue
        gene_length = max(end - start, 1.0)
        overlap["gene_overlap_start"] = np.maximum(overlap["start"], start)
        overlap["gene_overlap_end"] = np.minimum(overlap["end"], end)
        coverage = pd.Series(
            {
                str(analysis_id): _union_length(
                    frame["gene_overlap_start"],
                    frame["gene_overlap_end"],
                )
                / gene_length
                for analysis_id, frame in overlap.groupby("analysis_id", sort=False)
            },
            dtype=float,
        )
        carriers = set(
            coverage[coverage.ge(params.minimum_gene_coverage)].index.astype(str)
        )
        if not carriers:
            continue
        # Match the frozen architecture audit by retaining the shortest
        # qualifying same-direction segment for each carrier analysis.
        effective = (
            overlap[overlap["analysis_id"].astype(str).isin(carriers)]
            .sort_values(["analysis_id", "segment_length_bp"])
            .drop_duplicates("analysis_id", keep="first")
            .copy()
        )
        effective = effective[
            [
                "analysis_id",
                PLATFORM_FAMILY_COLUMN,
                PLATFORM_ID_COLUMN,
                PLATFORM_LABEL_COLUMN,
                PLATFORM_DETAIL_COLUMN,
                SOURCE_COLUMN,
                "segment_length_bp",
                "segment_fraction_chromosome",
            ]
        ]
        effective.insert(0, "amplitude", str(marker.amplitude))
        effective.insert(0, "event", str(marker.event))
        effective.insert(0, "feature", str(marker.feature))
        effective.insert(0, "reference_key", str(marker.reference_key))
        rows.append(effective)
    if not rows:
        raise ValueError("No mixed Track 2 segment carriers were reconstructed")
    carriers = pd.concat(rows, ignore_index=True)
    if carriers.duplicated(EVENT_KEY + ["analysis_id"]).any():
        raise ValueError("Track 2 carrier rows must be unique per event and analysis")
    return carriers


def _platform_details(
    carriers: pd.DataFrame,
    events: pd.DataFrame,
    params: Track2PlatformSpanParams,
    *,
    platform_level: str,
    group_column: str,
) -> pd.DataFrame:
    event_meta = events[
        EVENT_KEY
        + ["root", "reference_level", "segment_support_level", "track1_relation"]
    ]
    rows: list[dict[str, object]] = []
    for key, group in carriers.groupby(EVENT_KEY + [group_column], sort=False):
        span = pd.to_numeric(group["segment_fraction_chromosome"], errors="coerce")
        fractions = _span_fractions(span, params)
        source_counts = group[SOURCE_COLUMN].astype(str).value_counts()
        detail_id = (
            str(group[PLATFORM_ID_COLUMN].iloc[0])
            if platform_level == "platform_label"
            else ""
        )
        detail_label = (
            str(group[PLATFORM_LABEL_COLUMN].iloc[0])
            if platform_level == "platform_label"
            else str(key[-1])
        )
        rows.append(
            {
                **dict(zip(EVENT_KEY, key[:-1], strict=True)),
                "platform_level": platform_level,
                "platform_group": str(key[-1]),
                "platform_id": detail_id,
                "platform_label": detail_label,
                "n_segment_carriers": int(len(group)),
                "evaluable_platform": bool(
                    len(group) >= params.minimum_carriers_per_platform
                ),
                "platform_span_class": _span_class(span, params),
                "segment_span_fraction_q25": float(span.quantile(0.25)),
                "segment_span_fraction_median": float(span.median()),
                "segment_span_fraction_q75": float(span.quantile(0.75)),
                "focal_fraction_among_carriers": fractions["focal"],
                "regional_fraction_among_carriers": fractions["regional"],
                "broad_fraction_among_carriers": fractions["broad"],
                "n_sources": int(len(source_counts)),
                "dominant_source": str(source_counts.index[0]),
                "dominant_source_fraction": float(source_counts.iloc[0] / len(group)),
            }
        )
    details = pd.DataFrame(rows).merge(
        event_meta,
        on=EVENT_KEY,
        how="left",
        validate="many_to_one",
    )
    leading = EVENT_KEY + [
        "root",
        "reference_level",
        "platform_level",
        "platform_group",
        "platform_id",
        "platform_label",
        "n_segment_carriers",
        "evaluable_platform",
        "platform_span_class",
    ]
    return details[leading + [column for column in details if column not in leading]]


def _event_summary(
    carriers: pd.DataFrame,
    details: pd.DataFrame,
    events: pd.DataFrame,
    params: Track2PlatformSpanParams,
    *,
    platform_level: str,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    detail_groups = {
        tuple(key): group
        for key, group in details.groupby(EVENT_KEY, sort=False)
    }
    carrier_groups = {
        tuple(key): group
        for key, group in carriers.groupby(EVENT_KEY, sort=False)
    }
    for marker in events.itertuples(index=False):
        key = tuple(str(getattr(marker, column)) for column in EVENT_KEY)
        event_carriers = carrier_groups.get(key, pd.DataFrame())
        event_platforms = detail_groups.get(key, pd.DataFrame())
        eligible = (
            event_platforms[event_platforms["evaluable_platform"]].copy()
            if not event_platforms.empty
            else pd.DataFrame()
        )
        status = _categorical_audit_status(eligible)
        n_mixed_platforms = (
            int(eligible["platform_span_class"].eq("mixed").sum())
            if not eligible.empty
            else 0
        )
        any_platform_mixed = bool(n_mixed_platforms > 0) if len(eligible) >= 2 else pd.NA
        all_platforms_mixed = (
            bool(n_mixed_platforms == len(eligible)) if len(eligible) >= 2 else pd.NA
        )
        h_statistic = np.nan
        p_value = np.nan
        epsilon_squared = np.nan
        n_tested_carriers = 0
        if len(eligible) >= 2:
            eligible_names = set(eligible["platform_group"].astype(str))
            tested = event_carriers[
                event_carriers[group_column].astype(str).isin(eligible_names)
            ]
            samples = [
                pd.to_numeric(group["segment_fraction_chromosome"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
                for _, group in tested.groupby(group_column, sort=False)
            ]
            samples = [sample for sample in samples if len(sample) > 0]
            n_tested_carriers = int(sum(len(sample) for sample in samples))
            if len(samples) >= 2:
                try:
                    result = kruskal(*samples)
                    h_statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except ValueError:
                    h_statistic = 0.0
                    p_value = 1.0
                denominator = n_tested_carriers - len(samples)
                if denominator > 0:
                    epsilon_squared = float(
                        np.clip(
                            (h_statistic - len(samples) + 1) / denominator,
                            0.0,
                            1.0,
                        )
                    )
        rows.append(
            {
                **dict(zip(EVENT_KEY, key, strict=True)),
                "root": str(marker.root),
                "reference_level": str(marker.reference_level),
                "segment_support_level": str(marker.segment_support_level),
                "track1_relation": str(marker.track1_relation),
                "platform_level": platform_level,
                "n_segment_carriers": int(len(event_carriers)),
                "n_platforms_observed": int(len(event_platforms)),
                "n_platforms_evaluable": int(len(eligible)),
                "n_evaluable_platforms_mixed": n_mixed_platforms,
                "any_evaluable_platform_mixed": any_platform_mixed,
                "all_evaluable_platforms_mixed": all_platforms_mixed,
                "n_platform_tested_carriers": n_tested_carriers,
                "platform_carrier_counts": _format_counts(
                    event_carriers, group_column
                ),
                "evaluable_platform_span_classes": _format_platform_classes(
                    eligible, "platform_group"
                ),
                "categorical_audit_status": status,
                "kruskal_h": h_statistic,
                "platform_span_p_value": p_value,
                "platform_span_epsilon_squared": epsilon_squared,
                "interpretation": _interpretation(status),
            }
        )
    summary = pd.DataFrame(rows)
    for column in (
        "any_evaluable_platform_mixed",
        "all_evaluable_platforms_mixed",
    ):
        summary[column] = summary[column].astype("boolean")
    summary["platform_span_q_value"] = _bh_adjust(
        summary["platform_span_p_value"]
    )
    tested = summary["platform_span_q_value"].notna()
    summary["platform_span_association"] = pd.Series(
        pd.NA,
        index=summary.index,
        dtype="boolean",
    )
    summary.loc[tested, "platform_span_association"] = (
        summary.loc[tested, "platform_span_q_value"].le(params.association_fdr)
        & summary.loc[tested, "platform_span_epsilon_squared"].ge(
            params.minimum_epsilon_squared
        )
    )
    order = EVENT_KEY + [
        "root",
        "reference_level",
        "segment_support_level",
        "track1_relation",
        "platform_level",
        "n_segment_carriers",
        "n_platforms_observed",
        "n_platforms_evaluable",
        "n_evaluable_platforms_mixed",
        "any_evaluable_platform_mixed",
        "all_evaluable_platforms_mixed",
        "n_platform_tested_carriers",
        "platform_carrier_counts",
        "evaluable_platform_span_classes",
        "categorical_audit_status",
        "platform_span_association",
        "platform_span_p_value",
        "platform_span_q_value",
        "platform_span_epsilon_squared",
        "kruskal_h",
        "interpretation",
    ]
    return summary[order].sort_values(
        ["platform_level", "root", "reference_key", "feature"]
    )


def _scope_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, str, pd.DataFrame]] = []
    for platform_level, level_events in events.groupby("platform_level", sort=True):
        scopes.append((str(platform_level), "all", "all", level_events))
        scopes.extend(
            (
                str(platform_level),
                "root",
                str(root),
                group,
            )
            for root, group in level_events.groupby("root", sort=True)
        )
    for platform_level, scope_type, scope_value, group in scopes:
        tested = group[group["platform_span_association"].notna()]
        associated = (
            tested["platform_span_association"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        any_mixed = (
            tested["any_evaluable_platform_mixed"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        epsilon = pd.to_numeric(
            tested["platform_span_epsilon_squared"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "platform_level": platform_level,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "n_mixed_events": int(len(group)),
                "n_evaluable_events": int(len(tested)),
                "n_mixed_persists_within_platform": int(any_mixed.sum()),
                "mixed_persistence_fraction": (
                    float(any_mixed.mean()) if len(any_mixed) else np.nan
                ),
                "n_span_difference_supported": int(associated.sum()),
                "span_difference_supported_fraction": (
                    float(associated.mean()) if len(associated) else np.nan
                ),
                "median_platform_span_epsilon_squared": (
                    float(epsilon.median()) if not epsilon.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _supplementary_table(
    event_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one manuscript-facing row per released mixed Track 2 event."""

    identity = (
        event_summary[event_summary["platform_level"].eq("platform_family")][
            EVENT_KEY + ["root", "reference_level", "n_segment_carriers"]
        ]
        .drop_duplicates(EVENT_KEY)
        .copy()
    )
    result = identity
    level_columns = [
        "n_platforms_evaluable",
        "n_evaluable_platforms_mixed",
        "any_evaluable_platform_mixed",
        "platform_span_association",
    ]
    for level, prefix in (
        ("platform_family", "platform_family"),
        ("platform_label", "assay_platform"),
    ):
        plural = (
            "platform_families"
            if prefix == "platform_family"
            else "assay_platforms"
        )
        level_summary = event_summary[event_summary["platform_level"].eq(level)][
            EVENT_KEY + level_columns
        ].rename(
            columns={
                "n_platforms_evaluable": f"n_evaluable_{plural}",
                "n_evaluable_platforms_mixed": f"n_mixed_{plural}",
                "any_evaluable_platform_mixed": (
                    f"mixed_persists_within_{prefix}"
                ),
                "platform_span_association": (
                    f"{prefix}_span_difference_supported"
                ),
            }
        )
        result = result.merge(
            level_summary,
            on=EVENT_KEY,
            how="left",
            validate="one_to_one",
        )
        evaluable_column = f"n_evaluable_{plural}"
        mixed_column = f"n_mixed_{plural}"
        percent_column = f"mixed_{plural}_percent"
        result[percent_column] = np.where(
            result[evaluable_column].ge(2),
            100.0 * result[mixed_column] / result[evaluable_column],
            np.nan,
        )
        result[percent_column] = result[percent_column].round(1)

    result = result.rename(
        columns={
            "reference_level": "reference_layer",
            "feature": "gene_symbol",
            "event": "cnv_event",
            "amplitude": "cnv_level",
        }
    )
    result["reference_layer"] = result["reference_layer"].replace(
        {"subtype": "core"}
    )
    status_columns = [
        "mixed_persists_within_platform_family",
        "platform_family_span_difference_supported",
        "mixed_persists_within_assay_platform",
        "assay_platform_span_difference_supported",
    ]
    for column in status_columns:
        result[column] = (
            result[column]
            .astype("boolean")
            .map({True: "yes", False: "no"})
            .astype("string")
            .fillna("not_evaluable")
        )
    order = [
        "reference_key",
        "root",
        "reference_layer",
        "gene_symbol",
        "cnv_event",
        "cnv_level",
        "n_segment_carriers",
        "n_evaluable_platform_families",
        "n_mixed_platform_families",
        "mixed_platform_families_percent",
        "mixed_persists_within_platform_family",
        "platform_family_span_difference_supported",
        "n_evaluable_assay_platforms",
        "n_mixed_assay_platforms",
        "mixed_assay_platforms_percent",
        "mixed_persists_within_assay_platform",
        "assay_platform_span_difference_supported",
    ]
    result = result[order]
    return result.sort_values(["root", "reference_key", "gene_symbol"])


def validate_track2_platform_span(
    *,
    atlas_dir: Path,
    out_dir: Path,
    signatures_path: Path | None = None,
    references_path: Path | None = None,
    assignments_path: Path | None = None,
    segments_path: Path | None = None,
    gene_intervals_path: Path | None = None,
    metadata_path: Path | None = None,
    params: Track2PlatformSpanParams | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run the mixed Track 2 platform-span sensitivity analysis."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    signatures_path = (
        Path(signatures_path).expanduser().resolve()
        if signatures_path is not None
        else atlas_dir / "release_candidate" / "tables" / "signature_events.tsv"
    )
    references_path = (
        Path(references_path).expanduser().resolve()
        if references_path is not None
        else atlas_dir / "release_candidate" / "tables" / "reference_contexts.tsv"
    )
    assignments_path = (
        Path(assignments_path).expanduser().resolve()
        if assignments_path is not None
        else atlas_dir / "release_candidate" / "tables" / "analysis_assignments.parquet"
    )
    segments_path = (
        Path(segments_path).expanduser().resolve()
        if segments_path is not None
        else atlas_dir / "source_derived" / "tumour_autosomal_segments.parquet"
    )
    gene_intervals_path = (
        Path(gene_intervals_path).expanduser().resolve()
        if gene_intervals_path is not None
        else atlas_dir / "projection_bundle_candidate" / "gene_intervals.tsv"
    )
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else atlas_dir / "work" / "preprocess_v1_baseline" / "meta_cleaned.tsv"
    )
    params = params or Track2PlatformSpanParams()
    _validate_params(params)
    input_paths = [
        signatures_path,
        references_path,
        assignments_path,
        segments_path,
        gene_intervals_path,
        metadata_path,
    ]
    _require_paths(input_paths)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Track 2 platform-span output exists; pass overwrite=True: {out_dir}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    signatures = pd.read_csv(signatures_path, sep="\t", low_memory=False)
    members, references = _load_definition_members(assignments_path, references_path)
    genes = _load_gene_intervals(gene_intervals_path)
    metadata = _load_metadata(metadata_path)
    events = _mixed_events(signatures, references, genes)
    segments, segment_qc = _read_segments(
        segments_path,
        set(members["analysis_id"]),
    )
    carriers = _carrier_spans(
        events=events,
        members=members,
        metadata=metadata,
        segments=segments,
        params=params,
    )
    detail_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for platform_level, group_column in (
        ("platform_family", PLATFORM_FAMILY_COLUMN),
        ("platform_label", PLATFORM_DETAIL_COLUMN),
    ):
        level_details = _platform_details(
            carriers,
            events,
            params,
            platform_level=platform_level,
            group_column=group_column,
        )
        detail_frames.append(level_details)
        event_frames.append(
            _event_summary(
                carriers,
                level_details,
                events,
                params,
                platform_level=platform_level,
                group_column=group_column,
            )
        )
    details = pd.concat(detail_frames, ignore_index=True)
    event_summary = pd.concat(event_frames, ignore_index=True)
    scope_summary = _scope_summary(event_summary)
    supplementary = _supplementary_table(event_summary)

    outputs = {
        "summary": out_dir / "track2_platform_span_summary.tsv",
        "supplementary_table": out_dir / "mixed_track2_platform_sensitivity.tsv",
        "manifest": out_dir / "manifest.json",
    }
    scope_summary.to_csv(outputs["summary"], sep="\t", index=False)
    supplementary.to_csv(outputs["supplementary_table"], sep="\t", index=False)
    manifest = {
        "schema_version": "cnvsim_track2_platform_span_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "released_mixed_track2_events",
        "interpretation": (
            "Platform association is descriptive and does not establish causation."
        ),
        "platform_levels": {
            "platform_family": PLATFORM_FAMILY_COLUMN,
            "platform_label": (
                f"{PLATFORM_ID_COLUMN} combined with {PLATFORM_LABEL_COLUMN}"
            ),
        },
        "parameters": asdict(params),
        "inputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths
        },
        "counts": {
            "mixed_track2_events": int(len(events)),
            "events_with_reconstructed_carriers": int(
                event_summary["n_segment_carriers"].gt(0).sum()
            ),
            "segment_carrier_rows": int(len(carriers)),
            "supplementary_event_rows": int(len(supplementary)),
            "categorical_audit_status": {
                str(level): {
                    str(key): int(value)
                    for key, value in group[
                        "categorical_audit_status"
                    ].value_counts().items()
                }
                for level, group in event_summary.groupby(
                    "platform_level", sort=True
                )
            },
            "platform_span_association": {
                str(level): {
                    "tested": int(group["platform_span_association"].notna().sum()),
                    "associated": int(
                        group["platform_span_association"]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                }
                for level, group in event_summary.groupby(
                    "platform_level", sort=True
                )
            },
        },
        "segment_qc": segment_qc,
        "outputs": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs
