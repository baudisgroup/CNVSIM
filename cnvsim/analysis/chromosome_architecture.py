from __future__ import annotations

"""Chromosome-scale CNV profiles and context-level Track 1 evidence."""

import numpy as np
import pandas as pd

from cnvsim.analysis.subtyping_stats import bh_fdr, mwu_with_auc


AUTOSOMES = tuple(str(i) for i in range(1, 23))
EVENTS = ("gain", "loss")
TRACK1_STAT_COLUMNS = [
    "root",
    "context",
    "feature",
    "chromosome",
    "event",
    "n_context",
    "n_rest",
    "context_median",
    "rest_median",
    "median_delta",
    "auc",
    "p_value",
    "n_platforms_tested",
    "n_platforms_direction_supporting",
    "q_value",
    "evidence_direction",
    "passes_track1_support",
]

# GRCh38 primary-assembly lengths used by the frozen Progenetix release.
CHR_LENGTH_BP_GRCH38 = {
    "1": 248_956_422,
    "2": 242_193_529,
    "3": 198_295_559,
    "4": 190_214_555,
    "5": 181_538_259,
    "6": 170_805_979,
    "7": 159_345_973,
    "8": 145_138_636,
    "9": 138_394_717,
    "10": 133_797_422,
    "11": 135_086_622,
    "12": 133_275_309,
    "13": 114_364_328,
    "14": 107_043_718,
    "15": 101_991_189,
    "16": 90_338_345,
    "17": 83_257_441,
    "18": 80_373_285,
    "19": 58_617_616,
    "20": 64_444_167,
    "21": 46_709_983,
    "22": 50_818_468,
}


def chr44_feature_columns(prefix: str = "CHR") -> list[str]:
    return [
        f"{prefix}|chr{chromosome}_{event}"
        for chromosome in AUTOSOMES
        for event in EVENTS
    ]


def normalize_chromosome(value: object) -> str:
    if pd.isna(value):
        return ""
    chromosome = str(value).strip().replace("chr", "").replace("CHR", "")
    return chromosome[:-2] if chromosome.endswith(".0") else chromosome


def infer_event_group(frame: pd.DataFrame) -> pd.Series:
    text = pd.Series("", index=frame.index, dtype="object")
    for column in (
        "segment_event_channel",
        "event_group",
        "copy_change",
        "state_label",
        "variant_state_label",
        "variant_type",
        "cnv_state",
    ):
        if column in frame.columns:
            text = text + " " + frame[column].fillna("").astype(str).str.lower()
    return pd.Series(
        np.select(
            [
                text.str.contains("gain|dup", regex=True),
                text.str.contains("loss|del", regex=True),
            ],
            ["gain", "loss"],
            default="other",
        ),
        index=frame.index,
        dtype="object",
    )


def normalize_segments(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"analysis_id", "chromosome", "start", "end"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Segment table is missing required columns: {missing}")

    original_rows = int(len(frame))
    segments = frame.copy()
    segments["analysis_id"] = segments["analysis_id"].astype(str)
    segments["chromosome"] = segments["chromosome"].map(normalize_chromosome)
    segments["start"] = pd.to_numeric(segments["start"], errors="coerce")
    segments["end"] = pd.to_numeric(segments["end"], errors="coerce")
    segments["event_group"] = infer_event_group(segments)
    segments = segments[
        segments["chromosome"].isin(AUTOSOMES)
        & segments["event_group"].isin(EVENTS)
        & segments["start"].notna()
        & segments["end"].notna()
    ].copy()

    segments["chromosome_length_bp"] = segments["chromosome"].map(
        CHR_LENGTH_BP_GRCH38
    ).astype(float)
    segments["start"] = np.maximum(0.0, segments["start"].astype(float))
    segments["end"] = np.minimum(
        segments["chromosome_length_bp"], segments["end"].astype(float)
    )
    segments = segments[segments["end"].gt(segments["start"])].copy()

    dedup_event_column = (
        "segment_event_channel"
        if "segment_event_channel" in segments.columns
        else "event_group"
    )
    coordinate_key = (
        segments["analysis_id"]
        + "|"
        + segments["chromosome"]
        + "|"
        + segments["start"].astype(str)
        + "|"
        + segments["end"].astype(str)
        + "|"
        + segments[dedup_event_column].astype(str)
    )
    if "variant_internal_id" in segments.columns:
        internal = segments["variant_internal_id"].astype("string")
        has_internal = internal.notna() & internal.str.strip().ne("")
        segments["_dedup_key"] = np.where(
            has_internal,
            segments["analysis_id"] + "|" + internal.fillna("").astype(str),
            coordinate_key,
        )
    else:
        segments["_dedup_key"] = coordinate_key

    before_dedup = int(len(segments))
    segments = segments.drop_duplicates("_dedup_key", keep="first").drop(
        columns="_dedup_key"
    )
    qc = {
        "input_rows": original_rows,
        "autosomal_gain_loss_rows_before_dedup": before_dedup,
        "duplicate_rows_removed": before_dedup - int(len(segments)),
        "normalized_rows": int(len(segments)),
        "analyses_with_segments": int(segments["analysis_id"].nunique()),
    }
    return segments.reset_index(drop=True), qc


def _union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts)
    starts = starts[order].astype(float)
    ends = ends[order].astype(float)
    total = 0.0
    current_start = float(starts[0])
    current_end = float(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0.0, current_end - current_start)
            current_start, current_end = start, end
    return float(total + max(0.0, current_end - current_start))


def build_chr44_profiles(
    segments: pd.DataFrame,
    *,
    analysis_ids: list[str],
    prefix: str = "CHR",
) -> pd.DataFrame:
    """Convert normalized segments into 44 union-coverage fractions per sample."""

    selected_ids = pd.Index(pd.Series(analysis_ids, dtype="string").drop_duplicates())
    selected = segments[segments["analysis_id"].isin(set(selected_ids))].copy()
    rows: list[dict[str, object]] = []
    for (analysis_id, chromosome, event), group in selected.groupby(
        ["analysis_id", "chromosome", "event_group"], sort=False
    ):
        covered_bp = _union_length(
            group["start"].to_numpy(float), group["end"].to_numpy(float)
        )
        rows.append(
            {
                "analysis_id": str(analysis_id),
                "feature": f"{prefix}|chr{chromosome}_{event}",
                "coverage_fraction": min(
                    1.0, covered_bp / float(CHR_LENGTH_BP_GRCH38[str(chromosome)])
                ),
            }
        )

    base = pd.DataFrame({"analysis_id": selected_ids.astype(str)})
    if rows:
        wide = (
            pd.DataFrame(rows)
            .pivot_table(
                index="analysis_id",
                columns="feature",
                values="coverage_fraction",
                aggfunc="max",
                fill_value=0.0,
            )
            .reset_index()
        )
        base = base.merge(wide, on="analysis_id", how="left")
    features = chr44_feature_columns(prefix)
    for feature in features:
        if feature not in base.columns:
            base[feature] = 0.0
        base[feature] = pd.to_numeric(base[feature], errors="coerce").fillna(0.0)
    return base[["analysis_id", *features]]


def compute_track1_evidence(
    *,
    profiles: pd.DataFrame,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    root: str,
    background_analysis_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    context_col: str = "context",
    platform_col: str = "platform_family_refined",
    min_platform_samples: int = 20,
    q_cut: float = 0.05,
    min_abs_median_delta: float = 0.10,
    min_abs_auc_shift: float = 0.10,
    min_platform_delta: float = 0.05,
    min_platforms_supporting: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute chr44 context-vs-rest evidence from anchor-defined contexts.

    Optional background samples contribute only to the same-root rest group.
    They never define a context profile or context membership.
    """

    required_assignments = {"analysis_id", context_col}
    missing = sorted(required_assignments - set(assignments.columns))
    if missing:
        raise KeyError(f"Assignments are missing required columns: {missing}")
    if platform_col not in metadata.columns:
        raise KeyError(f"Metadata is missing platform column: {platform_col}")

    labels = assignments[["analysis_id", context_col]].copy()
    labels["analysis_id"] = labels["analysis_id"].astype(str)
    labels[context_col] = pd.to_numeric(labels[context_col], errors="coerce")
    labels = labels[labels[context_col].notna() & labels[context_col].ge(0)].copy()
    labels[context_col] = labels[context_col].astype(int)
    if background_analysis_ids:
        anchor_ids = set(labels["analysis_id"])
        background_ids = sorted(
            {str(value) for value in background_analysis_ids} - anchor_ids
        )
        labels = pd.concat(
            [
                labels,
                pd.DataFrame(
                    {
                        "analysis_id": background_ids,
                        context_col: [-2] * len(background_ids),
                    }
                ),
            ],
            ignore_index=True,
        )

    meta = metadata[["analysis_id", platform_col]].drop_duplicates(
        "analysis_id", keep="first"
    ).copy()
    meta["analysis_id"] = meta["analysis_id"].astype(str)
    frame = labels.merge(profiles, on="analysis_id", how="left", validate="one_to_one")
    frame = frame.merge(meta, on="analysis_id", how="left", validate="one_to_one")
    features = chr44_feature_columns()
    if frame[features].isna().any().any():
        missing_profiles = int(frame[features].isna().all(axis=1).sum())
        raise ValueError(f"{missing_profiles} assigned samples lack chr44 profiles")
    if frame[platform_col].isna().any():
        raise ValueError("Some assigned samples lack platform annotations")

    profile_rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    contexts = sorted(
        int(value) for value in frame.loc[frame[context_col].ge(0), context_col].unique()
    )
    for context in contexts:
        members = frame[context_col].eq(int(context))
        others = ~members
        if int(members.sum()) < 2 or int(others.sum()) < 2:
            continue
        profile_row: dict[str, object] = {
            "root": str(root),
            "context": int(context),
            "n_samples": int(members.sum()),
        }
        for feature in features:
            in_context = frame.loc[members, feature].to_numpy(float)
            outside = frame.loc[others, feature].to_numpy(float)
            context_median = float(np.median(in_context))
            rest_median = float(np.median(outside))
            median_delta = context_median - rest_median
            p_value, auc, _ = mwu_with_auc(in_context, outside)
            direction = int(np.sign(median_delta))
            n_tested = 0
            n_supporting = 0
            for platform in sorted(frame[platform_col].astype(str).unique()):
                platform_context = members & frame[platform_col].astype(str).eq(platform)
                platform_rest = others & frame[platform_col].astype(str).eq(platform)
                if (
                    int(platform_context.sum()) < int(min_platform_samples)
                    or int(platform_rest.sum()) < int(min_platform_samples)
                ):
                    continue
                n_tested += 1
                platform_delta = float(
                    np.median(frame.loc[platform_context, feature])
                    - np.median(frame.loc[platform_rest, feature])
                )
                if (
                    direction != 0
                    and np.sign(platform_delta) == direction
                    and abs(platform_delta) >= float(min_platform_delta)
                ):
                    n_supporting += 1
            chromosome_event = feature.split("|", 1)[1]
            chromosome, event = chromosome_event.rsplit("_", 1)
            profile_row[feature] = context_median
            stat_rows.append(
                {
                    "root": str(root),
                    "context": int(context),
                    "feature": feature,
                    "chromosome": chromosome,
                    "event": event,
                    "n_context": int(members.sum()),
                    "n_rest": int(others.sum()),
                    "context_median": context_median,
                    "rest_median": rest_median,
                    "median_delta": median_delta,
                    "auc": auc,
                    "p_value": p_value,
                    "n_platforms_tested": n_tested,
                    "n_platforms_direction_supporting": n_supporting,
                }
            )
        profile_rows.append(profile_row)

    profiles_by_context = pd.DataFrame(profile_rows)
    stats = pd.DataFrame(stat_rows)
    if stats.empty:
        stats = pd.DataFrame(columns=TRACK1_STAT_COLUMNS)
        return profiles_by_context, stats, stats.copy()

    stats["q_value"] = np.nan
    for context, index in stats.groupby("context").groups.items():
        stats.loc[index, "q_value"] = bh_fdr(
            stats.loc[index, "p_value"].to_numpy(float)
        )
    stats["evidence_direction"] = np.where(
        stats["median_delta"].ge(0), "enriched", "depleted"
    )
    stats["passes_track1_support"] = (
        stats["q_value"].le(float(q_cut))
        & stats["median_delta"].abs().ge(float(min_abs_median_delta))
        & (stats["auc"] - 0.5).abs().ge(float(min_abs_auc_shift))
        & stats["n_platforms_direction_supporting"].ge(
            int(min_platforms_supporting)
        )
    )
    supported = stats[stats["passes_track1_support"]].copy()
    sort_columns = ["root", "context", "q_value", "feature"]
    return (
        profiles_by_context.sort_values(["root", "context"]).reset_index(drop=True),
        stats.sort_values(sort_columns).reset_index(drop=True),
        supported.sort_values(sort_columns).reset_index(drop=True),
    )
