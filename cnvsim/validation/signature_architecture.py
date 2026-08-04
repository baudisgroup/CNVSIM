from __future__ import annotations

"""Validate Track 2 marker geometry against raw CNV segments."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cnvsim.analysis.chromosome_architecture import (
    CHR_LENGTH_BP_GRCH38,
    normalize_chromosome,
    normalize_segments,
)


EVENT_KEY = ["reference_key", "track", "feature", "event", "amplitude"]
SEGMENT_STATE_COLUMNS = (
    "event_group",
    "copy_change",
    "state_label",
    "variant_state_label",
    "variant_type",
    "cnv_state",
)
FROZEN_FILES = {
    "event_architecture": "track2_event_segment_architecture.tsv",
    "context_summary": "context_signature_architecture.tsv",
    "summary": "signature_architecture_summary.tsv",
}
NO_SEGMENT_SUPPORT_EXCLUSION = "no_observed_same_direction_segment_carrier"


@dataclass(frozen=True)
class SignatureArchitectureParams:
    """Thresholds for descriptive Track 2 segment architecture."""

    focal_fraction_chromosome: float = 0.02
    broad_fraction_chromosome: float = 0.50
    minimum_segment_carrier_fraction: float = 0.05
    low_event_gene_coverage: float = 0.10
    high_event_gene_coverage: float = 0.20
    minimum_gene_coverage: float = 0.50
    dominance_fraction: float = 0.60
    mixed_fraction: float = 0.25
    max_negligible_raw_prevalence_delta: float = 0.02
    max_segment_encoding_abs_delta: float = 0.02


def _validate_params(params: SignatureArchitectureParams) -> None:
    bounded = {
        "focal_fraction_chromosome": params.focal_fraction_chromosome,
        "broad_fraction_chromosome": params.broad_fraction_chromosome,
        "minimum_segment_carrier_fraction": (
            params.minimum_segment_carrier_fraction
        ),
        "low_event_gene_coverage": params.low_event_gene_coverage,
        "high_event_gene_coverage": params.high_event_gene_coverage,
        "minimum_gene_coverage": params.minimum_gene_coverage,
        "dominance_fraction": params.dominance_fraction,
        "mixed_fraction": params.mixed_fraction,
        "max_negligible_raw_prevalence_delta": (
            params.max_negligible_raw_prevalence_delta
        ),
        "max_segment_encoding_abs_delta": (
            params.max_segment_encoding_abs_delta
        ),
    }
    invalid = [
        name for name, value in bounded.items() if not 0.0 <= float(value) <= 1.0
    ]
    if invalid:
        raise ValueError(
            "Signature-architecture fractions must be within [0, 1]: "
            f"{invalid}"
        )
    if params.focal_fraction_chromosome >= params.broad_fraction_chromosome:
        raise ValueError(
            "focal_fraction_chromosome must be below "
            "broad_fraction_chromosome"
        )
    if params.mixed_fraction >= params.dominance_fraction:
        raise ValueError("mixed_fraction must be below dominance_fraction")


def _union_length(starts: pd.Series, ends: pd.Series) -> float:
    """Return the union length of overlapping half-open intervals."""

    if starts.empty:
        return 0.0
    order = np.argsort(starts.to_numpy(dtype=float))
    ordered_starts = starts.to_numpy(dtype=float)[order]
    ordered_ends = ends.to_numpy(dtype=float)[order]
    current_start = float(ordered_starts[0])
    current_end = float(ordered_ends[0])
    total = 0.0
    for start, end in zip(ordered_starts[1:], ordered_ends[1:]):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0.0, current_end - current_start)
            current_start, current_end = start, end
    return float(total + max(0.0, current_end - current_start))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing signature-architecture inputs: " + ", ".join(missing))


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _quantile(values: pd.Series, probability: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(float(probability)))


def _signed_direction(event: object, direction: object) -> int:
    event_sign = {"gain": 1, "loss": -1}.get(str(event), 0)
    direction_sign = {"enriched": 1, "depleted": -1}.get(str(direction), 0)
    return event_sign * direction_sign


def _segment_event_channel(frame: pd.DataFrame) -> pd.Series:
    """Recover low/high gain/loss channels without collapsing amplitude."""

    text = pd.Series("", index=frame.index, dtype="object")
    for column in SEGMENT_STATE_COLUMNS:
        if column in frame.columns:
            text = text + " " + frame[column].fillna("").astype(str).str.lower()
    high_gain = text.str.contains(
        r"high[ _-]?(?:level[ _-]?)?(?:gain|dup)"
        r"|amplification|hldup|focal[ _-]?genome[ _-]?amplification",
        regex=True,
    )
    high_loss = text.str.contains(
        r"high[ _-]?(?:level[ _-]?)?(?:loss|del)"
        r"|deep[ _-]?loss|homozygous[ _-]?deletion"
        r"|complete[ _-]?(?:genomic[ _-]?)?deletion|hldel",
        regex=True,
    )
    gain = text.str.contains(r"gain|dup", regex=True)
    loss = text.str.contains(r"loss|del", regex=True)
    return pd.Series(
        np.select(
            [high_gain, high_loss, gain, loss],
            ["high_gain", "high_loss", "gain", "loss"],
            default="other",
        ),
        index=frame.index,
        dtype="object",
    )


def _marker_segment_channels(event: str, amplitude: str) -> tuple[str, ...]:
    """Return raw channels contributing to one encoded marker.

    Low-level features are inclusive: a high-level gain/loss also satisfies
    the corresponding low-level event flag. High-level features remain
    specific to their high-amplitude segment channel.
    """

    if amplitude == "high":
        return (f"high_{event}",)
    return (event, f"high_{event}")


def _raw_direction_concordant(
    direction: str,
    raw_prevalence_delta: object,
) -> bool:
    delta = pd.to_numeric(
        pd.Series([raw_prevalence_delta]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(delta) or float(delta) == 0.0:
        return False
    return bool(
        (direction == "enriched" and delta > 0)
        or (direction == "depleted" and delta < 0)
    )


def _raw_direction_status(
    *,
    direction: str,
    raw_prevalence_delta: object,
    negligible_delta: float,
) -> str:
    delta = pd.to_numeric(
        pd.Series([raw_prevalence_delta]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(delta):
        return "missing"
    if abs(float(delta)) <= negligible_delta:
        return "negligible"
    return (
        "concordant"
        if _raw_direction_concordant(direction, delta)
        else "discordant"
    )


def _track1_relation(
    *,
    reference_key: str,
    chromosome: str,
    event: str,
    direction: str,
    track1_lookup: dict[tuple[str, str], list[tuple[str, str]]],
) -> str:
    if not chromosome:
        return "missing_gene_coordinates"
    observed = track1_lookup.get((reference_key, chromosome), [])
    if not observed:
        return "no_same_chromosome_track1"
    if (event, direction) in observed:
        return "exact_channel_aligned"

    signed = _signed_direction(event, direction)
    if signed and any(_signed_direction(ev, direct) == signed for ev, direct in observed):
        return "signed_architecture_aligned"
    if signed and any(_signed_direction(ev, direct) == -signed for ev, direct in observed):
        return "opposing_architecture"
    return "other_same_chromosome_track1"


def _segment_architecture(
    *,
    n_effective_samples: int,
    carrier_fraction: float,
    focal_fraction: float,
    regional_fraction: float,
    broad_fraction: float,
    params: SignatureArchitectureParams,
) -> str:
    if int(n_effective_samples) == 0:
        return "no_segment_support"
    if (
        not np.isfinite(carrier_fraction)
        or carrier_fraction < params.minimum_segment_carrier_fraction
    ):
        return "low_segment_support"

    fractions = [focal_fraction, regional_fraction, broad_fraction]
    if sum(value >= params.mixed_fraction for value in fractions if np.isfinite(value)) >= 2:
        return "mixed"
    if np.isfinite(focal_fraction) and focal_fraction >= params.dominance_fraction:
        return "focal"
    if np.isfinite(regional_fraction) and regional_fraction >= params.dominance_fraction:
        return "regional"
    if np.isfinite(broad_fraction) and broad_fraction >= params.dominance_fraction:
        return "broad"
    return "unresolved"


def _event_interpretation(
    *,
    direction: str,
    architecture: str,
    track1_relation: str,
    raw_direction_status: str,
    encoded_event_prevalence: float,
    segment_encoding_abs_delta: float,
    params: SignatureArchitectureParams,
) -> str:
    if (
        direction == "enriched"
        and raw_direction_status == "negligible"
        and np.isfinite(encoded_event_prevalence)
        and encoded_event_prevalence
        < params.minimum_segment_carrier_fraction
    ):
        return "adjusted_effect_low_event_prevalence"
    if raw_direction_status == "discordant":
        return "adjusted_raw_direction_discordant"
    if raw_direction_status in {"negligible", "missing"}:
        return "adjusted_effect_negligible_raw_delta"
    if direction != "enriched":
        return "depleted_relative_marker"
    if (
        np.isfinite(segment_encoding_abs_delta)
        and segment_encoding_abs_delta
        > params.max_segment_encoding_abs_delta
    ):
        return "segment_encoding_discordant"
    if architecture in {
        "missing_gene_coordinates",
        "no_segment_support",
        "low_segment_support",
        "unresolved",
    }:
        return "insufficient_segment_support"

    aligned = track1_relation in {
        "exact_channel_aligned",
        "signed_architecture_aligned",
    }
    if track1_relation == "opposing_architecture":
        return "gene_event_opposes_track1_context"
    if aligned and architecture == "broad":
        return "broad_track1_gene_readout"
    if aligned and architecture in {"focal", "regional"}:
        return "local_gene_event_with_track1_context"
    if aligned and architecture == "mixed":
        return "mixed_gene_event_with_track1_context"
    if architecture == "broad":
        return "track2_only_broad_event"
    if architecture in {"focal", "regional"}:
        return "track2_only_local_event"
    if architecture == "mixed":
        return "track2_only_mixed_event"
    return "unresolved"


def _load_gene_intervals(path: Path) -> pd.DataFrame:
    genes = pd.read_csv(path, sep="\t", low_memory=False)
    symbol = next(
        (column for column in ("gene_symbol", "symbol", "gene") if column in genes),
        None,
    )
    chrom = next(
        (column for column in ("chrom", "chromosome") if column in genes),
        None,
    )
    start = next(
        (column for column in ("start", "gene_start") if column in genes),
        None,
    )
    end = next(
        (column for column in ("end", "gene_end") if column in genes),
        None,
    )
    if None in (symbol, chrom, start, end):
        raise KeyError(
            f"{path} must provide gene symbol, chromosome, start, and end columns"
        )

    result = genes[[symbol, chrom, start, end]].rename(
        columns={
            symbol: "gene_symbol",
            chrom: "gene_chromosome",
            start: "gene_start",
            end: "gene_end",
        }
    )
    result["gene_symbol"] = result["gene_symbol"].astype(str)
    result["gene_chromosome"] = result["gene_chromosome"].map(normalize_chromosome)
    result["gene_start"] = pd.to_numeric(result["gene_start"], errors="coerce")
    result["gene_end"] = pd.to_numeric(result["gene_end"], errors="coerce")
    result = result[
        result["gene_chromosome"].isin(CHR_LENGTH_BP_GRCH38)
        & result["gene_start"].notna()
        & result["gene_end"].notna()
        & result["gene_end"].gt(result["gene_start"])
    ].copy()
    inconsistent = (
        result.groupby("gene_symbol")[
            ["gene_chromosome", "gene_start", "gene_end"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        examples = inconsistent[inconsistent].index.astype(str).tolist()[:10]
        raise ValueError(f"Gene intervals are ambiguous for: {examples}")
    return result.drop_duplicates("gene_symbol", keep="first")


def _load_definition_members(
    assignments_path: Path,
    references_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_parquet(assignments_path)
    references = pd.read_csv(references_path, sep="\t", low_memory=False)
    _require_columns(
        assignments,
        {
            "analysis_id",
            "root",
            "pipeline_status",
            "assigned_reference_key",
        },
        "analysis_assignments",
    )
    _require_columns(
        references,
        {
            "reference_key",
            "root",
            "reference_level",
            "n_definition_members",
        },
        "reference_contexts",
    )

    members = assignments[
        assignments["pipeline_status"].isin(
            ["anchor_member", "conditional_definition_member"]
        )
        & assignments["assigned_reference_key"].notna()
    ][["analysis_id", "root", "assigned_reference_key"]].rename(
        columns={"assigned_reference_key": "reference_key"}
    )
    members["analysis_id"] = members["analysis_id"].astype(str)
    members["root"] = members["root"].astype(str)
    members["reference_key"] = members["reference_key"].astype(str)
    if members["analysis_id"].duplicated().any():
        raise ValueError("Definition-member analysis IDs must be unique")

    reference_meta = references[
        ["reference_key", "root", "reference_level", "n_definition_members"]
    ].copy()
    reference_meta["reference_key"] = reference_meta["reference_key"].astype(str)
    reference_meta["root"] = reference_meta["root"].astype(str)
    members = members.merge(
        reference_meta[
            ["reference_key", "root", "reference_level"]
        ].rename(columns={"root": "reference_root"}),
        on="reference_key",
        how="left",
        validate="many_to_one",
    )
    if members["reference_level"].isna().any():
        raise ValueError("Definition members reference unknown contexts")
    if members["root"].ne(members["reference_root"]).any():
        raise ValueError("Definition-member roots disagree with reference roots")
    members = members.drop(columns="reference_root")

    observed = members.groupby("reference_key").size()
    expected = reference_meta.set_index("reference_key")["n_definition_members"]
    expected_definition_keys = set(
        reference_meta.loc[
            reference_meta["reference_level"].isin(["subtype", "conditional"]),
            "reference_key",
        ]
    )
    if set(observed.index) != expected_definition_keys:
        raise ValueError(
            "Core/conditional reference membership is incomplete: "
            f"missing={sorted(expected_definition_keys - set(observed.index))}, "
            f"extra={sorted(set(observed.index) - expected_definition_keys)}"
        )
    comparable = expected.index.intersection(observed.index)
    mismatch = observed.loc[comparable].astype(int).ne(expected.loc[comparable].astype(int))
    if mismatch.any():
        examples = {
            key: {
                "observed": int(observed.loc[key]),
                "expected": int(expected.loc[key]),
            }
            for key in mismatch[mismatch].index[:10]
        }
        raise ValueError(f"Definition-member counts disagree with release: {examples}")
    return members, reference_meta


def _read_segments(path: Path, analysis_ids: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - required by parquet workflows
        raise RuntimeError("pyarrow is required for segment architecture validation") from exc

    available = set(pq.ParquetFile(path).schema.names)
    required = {"analysis_id", "chromosome", "start", "end"}
    missing = sorted(required - available)
    if missing:
        raise KeyError(f"{path} is missing segment columns: {missing}")
    optional = set(SEGMENT_STATE_COLUMNS) | {"variant_internal_id"}
    raw = pd.read_parquet(path, columns=sorted(required | (optional & available)))
    raw["analysis_id"] = raw["analysis_id"].astype(str)
    raw = raw[raw["analysis_id"].isin(analysis_ids)].copy()
    raw["segment_event_channel"] = _segment_event_channel(raw)
    normalized, qc = normalize_segments(raw)
    qc["segment_event_channel"] = {
        str(key): int(value)
        for key, value in normalized["segment_event_channel"]
        .value_counts()
        .items()
    }
    normalized["segment_length_bp"] = normalized["end"] - normalized["start"]
    normalized["segment_fraction_chromosome"] = (
        normalized["segment_length_bp"] / normalized["chromosome_length_bp"]
    )
    return normalized, qc


def build_track2_segment_architecture(
    *,
    signatures: pd.DataFrame,
    references: pd.DataFrame,
    members: pd.DataFrame,
    segments: pd.DataFrame,
    gene_intervals: pd.DataFrame,
    params: SignatureArchitectureParams,
) -> pd.DataFrame:
    """Annotate released Track 2 events with raw-segment geometry."""

    _require_columns(
        signatures,
        set(EVENT_KEY)
        | {
            "direction",
            "architecture_linked",
            "n_context",
            "context_value",
            "background_value",
            "effect_size",
            "adjusted_effect_size",
            "q_value",
            "n_platforms_tested",
            "n_platforms_supporting",
        },
        "signature_events",
    )
    track1 = signatures[signatures["track"].eq("track1")].copy()
    track2 = signatures[signatures["track"].eq("track2")].copy()
    if track2.empty:
        raise ValueError("The release contains no Track 2 events")
    segments = segments.copy()
    if "segment_event_channel" not in segments.columns:
        segments["segment_event_channel"] = _segment_event_channel(segments)
    required_segment_channels = {
        channel
        for row in track2.loc[
            track2["direction"].eq("enriched")
            & track2["amplitude"].eq("high"),
            ["event", "amplitude"],
        ].itertuples(index=False)
        for channel in _marker_segment_channels(
            str(row.event),
            str(row.amplitude),
        )
    }
    available_segment_channels = set(
        segments["segment_event_channel"].dropna().astype(str)
    )
    missing_segment_channels = sorted(
        required_segment_channels - available_segment_channels
    )
    if missing_segment_channels:
        raise ValueError(
            "Raw segment states cannot reconstruct released Track 2 amplitude "
            f"channels: {missing_segment_channels}. Preserve low/high CNV state "
            "labels in the segment source."
        )

    track1["chromosome"] = track1["feature"].map(normalize_chromosome)
    track1_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, group in track1.groupby(["reference_key", "chromosome"], sort=False):
        track1_lookup[(str(key[0]), str(key[1]))] = list(
            group[["event", "direction"]].astype(str).itertuples(index=False, name=None)
        )

    reference_meta = references[
        ["reference_key", "root", "reference_level", "n_definition_members"]
    ].copy()
    track2 = track2.merge(
        reference_meta,
        on="reference_key",
        how="left",
        validate="many_to_one",
    ).merge(
        gene_intervals,
        left_on="feature",
        right_on="gene_symbol",
        how="left",
        validate="many_to_one",
    )
    if track2["reference_level"].isna().any():
        raise ValueError("Track 2 events reference unknown contexts")

    member_labels = members[["analysis_id", "reference_key"]]
    labelled_segments = segments.merge(
        member_labels,
        on="analysis_id",
        how="inner",
        validate="many_to_one",
    )
    segment_groups = {
        (str(reference_key), str(event), str(chromosome)): group
        for (reference_key, event, chromosome), group in labelled_segments.groupby(
            ["reference_key", "segment_event_channel", "chromosome"],
            sort=False,
        )
    }

    rows: list[dict[str, object]] = []
    for marker in track2.itertuples(index=False):
        reference_key = str(marker.reference_key)
        event = str(marker.event)
        direction = str(marker.direction)
        chromosome = (
            normalize_chromosome(marker.gene_chromosome)
            if pd.notna(marker.gene_chromosome)
            else ""
        )
        relation = _track1_relation(
            reference_key=reference_key,
            chromosome=chromosome,
            event=event,
            direction=direction,
            track1_lookup=track1_lookup,
        )
        existing_link = _as_bool(marker.architecture_linked)
        raw_direction_status = _raw_direction_status(
            direction=direction,
            raw_prevalence_delta=marker.effect_size,
            negligible_delta=params.max_negligible_raw_prevalence_delta,
        )
        raw_direction_concordant = raw_direction_status == "concordant"
        segment_event_channels = _marker_segment_channels(
            event,
            str(marker.amplitude),
        )
        encoding_gene_coverage_threshold = (
            params.high_event_gene_coverage
            if str(marker.amplitude) == "high"
            else params.low_event_gene_coverage
        )
        base = {
            "reference_key": reference_key,
            "root": str(marker.root),
            "reference_level": str(marker.reference_level),
            "track": "track2",
            "feature": str(marker.feature),
            "event": event,
            "amplitude": str(marker.amplitude),
            "direction": direction,
            "segment_event_channel": ";".join(segment_event_channels),
            "encoding_gene_coverage_threshold": (
                encoding_gene_coverage_threshold
            ),
            "gene_chromosome": chromosome or pd.NA,
            "gene_start": marker.gene_start,
            "gene_end": marker.gene_end,
            "n_definition_members": int(marker.n_definition_members),
            "n_context_evaluable": int(marker.n_context),
            "encoded_event_prevalence": float(marker.context_value),
            "background_event_prevalence": float(marker.background_value),
            "raw_prevalence_delta": float(marker.effect_size),
            "adjusted_effect_size": float(marker.adjusted_effect_size),
            "q_value": float(marker.q_value),
            "n_platforms_tested": int(marker.n_platforms_tested),
            "n_platforms_supporting": int(marker.n_platforms_supporting),
            "raw_direction_concordant": raw_direction_concordant,
            "raw_direction_status": raw_direction_status,
            "existing_architecture_linked": existing_link,
            "track1_relation": relation,
            "linkage_consistent_with_release": (
                existing_link == (relation == "exact_channel_aligned")
            ),
        }

        if direction != "enriched":
            architecture = "not_applicable_depleted"
            rows.append(
                {
                    **base,
                    "n_any_overlap_samples": pd.NA,
                    "any_overlap_fraction": np.nan,
                    "n_encoding_carrier_samples": pd.NA,
                    "encoding_carrier_fraction": np.nan,
                    "n_segment_carrier_samples": pd.NA,
                    "segment_carrier_fraction": np.nan,
                    "segment_length_bp_q25": np.nan,
                    "segment_length_bp_median": np.nan,
                    "segment_length_bp_q75": np.nan,
                    "segment_span_fraction_q25": np.nan,
                    "segment_span_fraction_median": np.nan,
                    "segment_span_fraction_q75": np.nan,
                    "focal_fraction_among_carriers": np.nan,
                    "regional_fraction_among_carriers": np.nan,
                    "broad_fraction_among_carriers": np.nan,
                    "segment_encoding_abs_delta": np.nan,
                    "segment_encoding_within_tolerance": pd.NA,
                    "segment_architecture": architecture,
                    "event_interpretation": _event_interpretation(
                        direction=direction,
                        architecture=architecture,
                        track1_relation=relation,
                        raw_direction_status=raw_direction_status,
                        encoded_event_prevalence=float(marker.context_value),
                        segment_encoding_abs_delta=np.nan,
                        params=params,
                    ),
                }
            )
            continue

        if (
            not chromosome
            or pd.isna(marker.gene_start)
            or pd.isna(marker.gene_end)
        ):
            architecture = "missing_gene_coordinates"
            rows.append(
                {
                    **base,
                    "n_any_overlap_samples": 0,
                    "any_overlap_fraction": 0.0,
                    "n_encoding_carrier_samples": 0,
                    "encoding_carrier_fraction": 0.0,
                    "n_segment_carrier_samples": 0,
                    "segment_carrier_fraction": 0.0,
                    "segment_length_bp_q25": np.nan,
                    "segment_length_bp_median": np.nan,
                    "segment_length_bp_q75": np.nan,
                    "segment_span_fraction_q25": np.nan,
                    "segment_span_fraction_median": np.nan,
                    "segment_span_fraction_q75": np.nan,
                    "focal_fraction_among_carriers": np.nan,
                    "regional_fraction_among_carriers": np.nan,
                    "broad_fraction_among_carriers": np.nan,
                    "segment_encoding_abs_delta": np.nan,
                    "segment_encoding_within_tolerance": pd.NA,
                    "segment_architecture": architecture,
                    "event_interpretation": _event_interpretation(
                        direction=direction,
                        architecture=architecture,
                        track1_relation=relation,
                        raw_direction_status=raw_direction_status,
                        encoded_event_prevalence=float(marker.context_value),
                        segment_encoding_abs_delta=np.nan,
                        params=params,
                    ),
                }
            )
            continue

        channel_groups = [
            segment_groups[(reference_key, channel, chromosome)]
            for channel in segment_event_channels
            if (reference_key, channel, chromosome) in segment_groups
        ]
        group = (
            pd.concat(channel_groups, ignore_index=True)
            if channel_groups
            else pd.DataFrame()
        )
        start = float(marker.gene_start)
        end = float(marker.gene_end)
        overlap = (
            group[
                group["start"].lt(end)
                & group["end"].gt(start)
            ].copy()
            if not group.empty
            else pd.DataFrame()
        )
        # Track 2 prevalence is calculated over marker-evaluable context members.
        # Use the same denominator for raw-segment support.
        n_total = int(marker.n_context)
        if n_total <= 0:
            raise ValueError(
                f"[{reference_key} {marker.feature}] invalid n_context={n_total}"
            )
        if overlap.empty:
            any_samples = 0
            coverage_by_sample = pd.Series(dtype=float)
            effective = pd.DataFrame()
        else:
            gene_length = max(end - start, 1.0)
            overlap["gene_overlap_start"] = np.maximum(overlap["start"], start)
            overlap["gene_overlap_end"] = np.minimum(overlap["end"], end)
            coverage_by_sample = pd.Series(
                {
                    str(analysis_id): _union_length(
                        frame["gene_overlap_start"],
                        frame["gene_overlap_end"],
                    )
                    / gene_length
                    for analysis_id, frame in overlap.groupby(
                        "analysis_id",
                        sort=False,
                    )
                },
                dtype=float,
            )
            any_samples = int(overlap["analysis_id"].nunique())
            architecture_carriers = set(
                coverage_by_sample[
                    coverage_by_sample.ge(params.minimum_gene_coverage)
                ].index.astype(str)
            )
            effective = (
                overlap[
                    overlap["analysis_id"].astype(str).isin(
                        architecture_carriers
                    )
                ]
                .sort_values(["analysis_id", "segment_length_bp"])
                .drop_duplicates("analysis_id", keep="first")
                .copy()
            )

        n_encoding_carriers = int(
            coverage_by_sample.ge(encoding_gene_coverage_threshold).sum()
        )
        encoding_carrier_fraction = float(n_encoding_carriers / n_total)
        n_effective = int(effective["analysis_id"].nunique()) if not effective.empty else 0
        carrier_fraction = float(n_effective / n_total) if n_total else np.nan
        span = (
            pd.to_numeric(effective["segment_fraction_chromosome"], errors="coerce")
            if not effective.empty
            else pd.Series(dtype=float)
        )
        focal_fraction = (
            float(span.le(params.focal_fraction_chromosome).mean())
            if not span.empty
            else np.nan
        )
        regional_fraction = (
            float(
                (
                    span.gt(params.focal_fraction_chromosome)
                    & span.lt(params.broad_fraction_chromosome)
                ).mean()
            )
            if not span.empty
            else np.nan
        )
        broad_fraction = (
            float(span.ge(params.broad_fraction_chromosome).mean())
            if not span.empty
            else np.nan
        )
        architecture = _segment_architecture(
            n_effective_samples=n_effective,
            carrier_fraction=carrier_fraction,
            focal_fraction=focal_fraction,
            regional_fraction=regional_fraction,
            broad_fraction=broad_fraction,
            params=params,
        )
        encoding_abs_delta = abs(
            encoding_carrier_fraction - float(marker.context_value)
        )
        rows.append(
            {
                **base,
                "n_any_overlap_samples": any_samples,
                "any_overlap_fraction": float(any_samples / n_total) if n_total else np.nan,
                "n_encoding_carrier_samples": n_encoding_carriers,
                "encoding_carrier_fraction": encoding_carrier_fraction,
                "n_segment_carrier_samples": n_effective,
                "segment_carrier_fraction": carrier_fraction,
                "segment_length_bp_q25": (
                    _quantile(effective["segment_length_bp"], 0.25)
                    if not effective.empty
                    else np.nan
                ),
                "segment_length_bp_median": (
                    _quantile(effective["segment_length_bp"], 0.50)
                    if not effective.empty
                    else np.nan
                ),
                "segment_length_bp_q75": (
                    _quantile(effective["segment_length_bp"], 0.75)
                    if not effective.empty
                    else np.nan
                ),
                "segment_span_fraction_q25": _quantile(span, 0.25),
                "segment_span_fraction_median": _quantile(span, 0.50),
                "segment_span_fraction_q75": _quantile(span, 0.75),
                "focal_fraction_among_carriers": focal_fraction,
                "regional_fraction_among_carriers": regional_fraction,
                "broad_fraction_among_carriers": broad_fraction,
                "segment_encoding_abs_delta": encoding_abs_delta,
                "segment_encoding_within_tolerance": bool(
                    encoding_abs_delta
                    <= params.max_segment_encoding_abs_delta
                ),
                "segment_architecture": architecture,
                "event_interpretation": _event_interpretation(
                    direction=direction,
                    architecture=architecture,
                    track1_relation=relation,
                    raw_direction_status=raw_direction_status,
                    encoded_event_prevalence=float(marker.context_value),
                    segment_encoding_abs_delta=encoding_abs_delta,
                    params=params,
                ),
            }
        )

    result = pd.DataFrame(rows)
    excluded = (
        result["direction"].eq("enriched")
        & result["segment_architecture"].eq("no_segment_support")
    )
    result["release_eligible"] = ~excluded
    result["release_exclusion_reason"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result.loc[excluded, "release_exclusion_reason"] = (
        NO_SEGMENT_SUPPORT_EXCLUSION
    )
    if result.duplicated(EVENT_KEY).any():
        raise ValueError("Track 2 architecture event keys are not unique")
    if not result["linkage_consistent_with_release"].all():
        examples = result.loc[
            ~result["linkage_consistent_with_release"],
            EVENT_KEY + ["existing_architecture_linked", "track1_relation"],
        ].head()
        raise ValueError(
            "Recomputed exact Track 1 linkage disagrees with released linkage:\n"
            + examples.to_string(index=False)
        )
    return result.sort_values(
        ["root", "reference_level", "reference_key", "direction", "event", "feature"]
    ).reset_index(drop=True)


def summarize_context_architecture(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in events.groupby(
        ["root", "reference_level", "reference_key"],
        sort=True,
    ):
        enriched = group[group["direction"].eq("enriched")]
        encoding_delta = pd.to_numeric(
            enriched["segment_encoding_abs_delta"],
            errors="coerce",
        ).dropna()
        encoding_status = enriched[
            "segment_encoding_within_tolerance"
        ].dropna()
        n_encoding_discordant = int(
            (~encoding_status.astype(bool)).sum()
        )
        architecture_counts = enriched["segment_architecture"].value_counts()
        relation_counts = group["track1_relation"].value_counts()
        interpretation_counts = group["event_interpretation"].value_counts()
        rows.append(
            {
                "root": key[0],
                "reference_level": key[1],
                "reference_key": key[2],
                "n_track2_events": int(len(group)),
                "n_release_eligible_events": int(
                    group["release_eligible"].sum()
                ),
                "n_release_excluded_events": int(
                    (~group["release_eligible"]).sum()
                ),
                "n_enriched_events": int(len(enriched)),
                "n_depleted_events": int(group["direction"].eq("depleted").sum()),
                "n_focal_events": int(architecture_counts.get("focal", 0)),
                "n_regional_events": int(architecture_counts.get("regional", 0)),
                "n_broad_events": int(architecture_counts.get("broad", 0)),
                "n_mixed_events": int(architecture_counts.get("mixed", 0)),
                "n_low_or_no_support_events": int(
                    architecture_counts.get("low_segment_support", 0)
                    + architecture_counts.get("no_segment_support", 0)
                    + architecture_counts.get("missing_gene_coordinates", 0)
                    + architecture_counts.get("unresolved", 0)
                ),
                "n_exact_track1_linked": int(
                    relation_counts.get("exact_channel_aligned", 0)
                ),
                "n_raw_direction_concordant": int(
                    group["raw_direction_concordant"].fillna(False).sum()
                ),
                "n_raw_direction_negligible": int(
                    group["raw_direction_status"].eq("negligible").sum()
                ),
                "n_raw_direction_discordant": int(
                    group["raw_direction_status"].eq("discordant").sum()
                ),
                "raw_direction_concordant_fraction": float(
                    group["raw_direction_concordant"].fillna(False).mean()
                ),
                "n_adjusted_raw_direction_discordant": int(
                    interpretation_counts.get(
                        "adjusted_raw_direction_discordant",
                        0,
                    )
                ),
                "n_adjusted_only_low_prevalence": int(
                    interpretation_counts.get(
                        "adjusted_effect_low_event_prevalence",
                        0,
                    )
                ),
                "n_adjusted_negligible_raw_delta": int(
                    interpretation_counts.get(
                        "adjusted_effect_negligible_raw_delta",
                        0,
                    )
                ),
                "n_segment_encoding_discordant": n_encoding_discordant,
                "median_segment_carrier_fraction": (
                    float(enriched["segment_carrier_fraction"].median())
                    if not enriched.empty
                    else np.nan
                ),
                "median_segment_span_fraction": (
                    float(enriched["segment_span_fraction_median"].median())
                    if not enriched.empty
                    else np.nan
                ),
                "median_segment_encoding_abs_delta": (
                    float(encoding_delta.median())
                    if not encoding_delta.empty
                    else np.nan
                ),
                "segment_encoding_within_tolerance_fraction": (
                    float(encoding_status.astype(bool).mean())
                    if not encoding_status.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_signature_architecture(events: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, str, pd.DataFrame]] = [("all", "all", events)]
    scopes.extend(
        ("reference_level", str(level), group)
        for level, group in events.groupby("reference_level", sort=True)
    )
    rows: list[dict[str, object]] = []
    for scope_type, scope_value, group in scopes:
        enriched = group[group["direction"].eq("enriched")]
        encoding_delta = pd.to_numeric(
            enriched["segment_encoding_abs_delta"],
            errors="coerce",
        ).dropna()
        encoding_status = enriched[
            "segment_encoding_within_tolerance"
        ].dropna()
        n_encoding_discordant = int(
            (~encoding_status.astype(bool)).sum()
        )
        architecture_counts = enriched["segment_architecture"].value_counts()
        relation_counts = group["track1_relation"].value_counts()
        interpretation_counts = group["event_interpretation"].value_counts()
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "n_track2_events": int(len(group)),
                "n_release_eligible_events": int(
                    group["release_eligible"].sum()
                ),
                "n_release_excluded_events": int(
                    (~group["release_eligible"]).sum()
                ),
                "n_enriched_events": int(len(enriched)),
                "n_depleted_events": int(group["direction"].eq("depleted").sum()),
                "n_focal_events": int(architecture_counts.get("focal", 0)),
                "n_regional_events": int(architecture_counts.get("regional", 0)),
                "n_broad_events": int(architecture_counts.get("broad", 0)),
                "n_mixed_events": int(architecture_counts.get("mixed", 0)),
                "n_low_segment_support_events": int(
                    architecture_counts.get("low_segment_support", 0)
                ),
                "n_no_segment_support_events": int(
                    architecture_counts.get("no_segment_support", 0)
                ),
                "n_exact_track1_linked": int(
                    relation_counts.get("exact_channel_aligned", 0)
                ),
                "n_signed_track1_linked": int(
                    relation_counts.get("signed_architecture_aligned", 0)
                ),
                "n_opposing_track1_events": int(
                    relation_counts.get("opposing_architecture", 0)
                ),
                "n_raw_direction_concordant": int(
                    group["raw_direction_concordant"].fillna(False).sum()
                ),
                "n_raw_direction_negligible": int(
                    group["raw_direction_status"].eq("negligible").sum()
                ),
                "n_raw_direction_discordant": int(
                    group["raw_direction_status"].eq("discordant").sum()
                ),
                "raw_direction_concordant_fraction": float(
                    group["raw_direction_concordant"].fillna(False).mean()
                ),
                "n_adjusted_raw_direction_discordant": int(
                    interpretation_counts.get(
                        "adjusted_raw_direction_discordant",
                        0,
                    )
                ),
                "n_adjusted_only_low_prevalence": int(
                    interpretation_counts.get(
                        "adjusted_effect_low_event_prevalence",
                        0,
                    )
                ),
                "n_adjusted_negligible_raw_delta": int(
                    interpretation_counts.get(
                        "adjusted_effect_negligible_raw_delta",
                        0,
                    )
                ),
                "n_segment_encoding_discordant": n_encoding_discordant,
                "n_broad_track1_gene_readouts": int(
                    interpretation_counts.get("broad_track1_gene_readout", 0)
                ),
                "n_local_events_with_track1": int(
                    interpretation_counts.get(
                        "local_gene_event_with_track1_context", 0
                    )
                ),
                "n_track2_only_local_events": int(
                    interpretation_counts.get("track2_only_local_event", 0)
                ),
                "median_segment_carrier_fraction": (
                    float(enriched["segment_carrier_fraction"].median())
                    if not enriched.empty
                    else np.nan
                ),
                "median_segment_span_fraction": (
                    float(enriched["segment_span_fraction_median"].median())
                    if not enriched.empty
                    else np.nan
                ),
                "median_segment_encoding_abs_delta": (
                    float(encoding_delta.median())
                    if not encoding_delta.empty
                    else np.nan
                ),
                "q90_segment_encoding_abs_delta": (
                    float(encoding_delta.quantile(0.90))
                    if not encoding_delta.empty
                    else np.nan
                ),
                "segment_encoding_within_tolerance_fraction": (
                    float(encoding_status.astype(bool).mean())
                    if not encoding_status.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_signature_architecture(
    *,
    atlas_dir: Path,
    out_dir: Path,
    signatures_path: Path | None = None,
    references_path: Path | None = None,
    assignments_path: Path | None = None,
    segments_path: Path | None = None,
    gene_intervals_path: Path | None = None,
    params: SignatureArchitectureParams | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run the released-signature segment architecture audit."""

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
    params = params or SignatureArchitectureParams()
    _validate_params(params)
    _require_paths(
        [
            signatures_path,
            references_path,
            assignments_path,
            segments_path,
            gene_intervals_path,
        ]
    )
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Signature-architecture output exists; pass overwrite=True: {out_dir}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    signatures = pd.read_csv(signatures_path, sep="\t", low_memory=False)
    members, references = _load_definition_members(assignments_path, references_path)
    genes = _load_gene_intervals(gene_intervals_path)
    segments, segment_qc = _read_segments(
        segments_path,
        set(members["analysis_id"]),
    )
    events = build_track2_segment_architecture(
        signatures=signatures,
        references=references,
        members=members,
        segments=segments,
        gene_intervals=genes,
        params=params,
    )
    contexts = summarize_context_architecture(events)
    summary = summarize_signature_architecture(events)

    outputs = {
        "event_architecture": out_dir / FROZEN_FILES["event_architecture"],
        "context_summary": out_dir / FROZEN_FILES["context_summary"],
        "summary": out_dir / FROZEN_FILES["summary"],
        "manifest": out_dir / "manifest.json",
    }
    events.to_csv(outputs["event_architecture"], sep="\t", index=False)
    contexts.to_csv(outputs["context_summary"], sep="\t", index=False)
    summary.to_csv(outputs["summary"], sep="\t", index=False)
    manifest = {
        "schema_version": "cnvsim_signature_architecture_v6",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "work",
        "analysis_scope": "released_core_and_conditional_track2_events",
        "depleted_event_policy": "linkage_only_segment_geometry_not_applicable",
        "release_policy": {
            "enriched_track2_requires_observed_segment_carrier": True,
            "excluded_segment_architecture": "no_segment_support",
            "exclusion_reason": NO_SEGMENT_SUPPORT_EXCLUSION,
            "retained_low_support": True,
            "retained_unresolved_architecture": True,
        },
        "parameters": asdict(params),
        "inputs": {
            "signature_events": str(signatures_path),
            "signature_events_sha256": _sha256(signatures_path),
            "reference_contexts": str(references_path),
            "reference_contexts_sha256": _sha256(references_path),
            "analysis_assignments": str(assignments_path),
            "analysis_assignments_sha256": _sha256(assignments_path),
            "segments": str(segments_path),
            "segments_sha256": _sha256(segments_path),
            "gene_intervals": str(gene_intervals_path),
            "gene_intervals_sha256": _sha256(gene_intervals_path),
        },
        "counts": {
            "definition_members": int(len(members)),
            "definition_references": int(members["reference_key"].nunique()),
            "track2_events": int(len(events)),
            "release_eligible_track2_events": int(
                events["release_eligible"].sum()
            ),
            "release_excluded_track2_events": int(
                (~events["release_eligible"]).sum()
            ),
            "enriched_track2_events": int(events["direction"].eq("enriched").sum()),
            "depleted_track2_events": int(events["direction"].eq("depleted").sum()),
            "segment_architecture": {
                str(key): int(value)
                for key, value in events["segment_architecture"].value_counts().items()
            },
            "track1_relation": {
                str(key): int(value)
                for key, value in events["track1_relation"].value_counts().items()
            },
            "event_interpretation": {
                str(key): int(value)
                for key, value in events["event_interpretation"].value_counts().items()
            },
        },
        "segment_qc": segment_qc,
        "outputs": {
            key: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def freeze_signature_architecture(
    *,
    atlas_dir: Path,
    source_dir: Path,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Freeze checked signature architecture for release inclusion."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    source_dir = Path(source_dir).expanduser().resolve()
    out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else atlas_dir / "validation" / "signature_architecture_v1"
    )
    source_manifest_path = source_dir / "manifest.json"
    required = [source_dir / filename for filename in FROZEN_FILES.values()]
    _require_paths([source_manifest_path, *required])
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema_version") != "cnvsim_signature_architecture_v6":
        raise ValueError("Unexpected signature-architecture schema")
    if source_manifest.get("release_status") != "work":
        raise ValueError("Only a completed work bundle can be frozen")
    manifest_outputs = source_manifest.get("outputs", {})
    for key, filename in FROZEN_FILES.items():
        path = source_dir / filename
        recorded = manifest_outputs.get(key, {})
        if recorded.get("path") != path.name:
            raise ValueError(
                f"Work manifest output path disagrees for {key}: {recorded}"
            )
        if recorded.get("sha256") != _sha256(path):
            raise ValueError(f"Work output changed after validation: {path}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Frozen signature architecture exists; pass overwrite=True: {out_dir}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    outputs = {
        key: out_dir / filename for key, filename in FROZEN_FILES.items()
    }
    for key, destination in outputs.items():
        shutil.copy2(source_dir / FROZEN_FILES[key], destination)

    frozen_manifest = {
        **source_manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "frozen",
        "source_work_manifest_sha256": _sha256(source_manifest_path),
        "outputs": {
            key: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for key, path in outputs.items()
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(frozen_manifest, indent=2) + "\n")
    checksums_path = out_dir / "checksums.sha256"
    checksum_files = [*outputs.values(), manifest_path]
    checksums_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_files)
    )
    return {
        **outputs,
        "manifest": manifest_path,
        "checksums": checksums_path,
    }
