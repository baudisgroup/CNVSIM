#!/usr/bin/env python3
from __future__ import annotations

"""Render the chromosome and gene-event architecture panels for Figure 4."""

import argparse
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


AUTOSOMES = list(range(1, 23))
CHROMOSOME_LENGTHS = {
    1: 248956422,
    2: 242193529,
    3: 198295559,
    4: 190214555,
    5: 181538259,
    6: 170805979,
    7: 159345973,
    8: 145138636,
    9: 138394717,
    10: 133797422,
    11: 135086622,
    12: 133275309,
    13: 114364328,
    14: 107043718,
    15: 101991189,
    16: 90338345,
    17: 83257441,
    18: 80373285,
    19: 58617616,
    20: 64444167,
    21: 46709983,
    22: 50818468,
}
TEXT = "#111111"
MUTED = "#6D777C"
GRID = "#DDE3E6"
WHITE = "#FFFFFF"
CONDITIONAL = "#D6872C"
GAIN = "#E1B83B"
LOSS = "#2E5EAA"
MIXED = "#8651A3"
TRACK1_DISPLAY_MAX = 0.25
TRACK1_COLOR_POWER = 0.35
ROOT_HEADER_HEIGHT = 1.85

# Source sizes are calibrated for the final A/C top row and full-width B panel.
A_SMALL_TEXT = 12.5
A_TEXT = 14.0
B_MICRO_TEXT = 5.3
B_SMALL_TEXT = 5.8
B_TEXT = 7.0
C_SMALL_TEXT = 11.5
C_TEXT = 13.5

ARCHITECTURE_ORDER = [
    "focal",
    "regional",
    "broad",
    "mixed",
]
ARCHITECTURE_LABELS = {
    "focal": "Focal\n≤2%",
    "regional": "Regional\n2–50%",
    "broad": "Broad\n≥50%",
    "mixed": "Mixed",
}
ROOT_LABELS = {
    "C00_C14": "C00-C14  Oral/pharynx",
    "C15_C17_C21_C26": "C15/C17/C21/C26  Other digestive",
    "C16": "C16  Stomach",
    "C18_C20": "C18-C20  Colon/rectum",
    "C22_C24": "C22-C24  Liver/biliary",
    "C25": "C25  Pancreas",
    "C30_C39": "C30-C39  Respiratory",
    "C40_C41_C48": "C40/C41/C48  Bone/connective",
    "C42": "C42  Haematopoietic",
    "C44": "C44  Skin",
    "C49": "C49  Soft tissue",
    "C50": "C50  Breast",
    "C51_C58": "C51-C58  Female genital",
    "C60_C63": "C60-C63  Male genital",
    "C64_C68": "C64-C68  Urinary tract",
    "C69_C70_C72": "C69/C70/C72  Eye/CNS",
    "C71": "C71  Brain",
    "C73_C75": "C73-C75  Endocrine",
    "C77": "C77  Lymph nodes",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


def _read_tsv(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")
    return frame


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _chromosome(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.replace(r"^chr", "", regex=True, case=False)
    )
    return pd.to_numeric(normalized, errors="coerce").astype("Int64")


def _context_number(reference_key: str) -> int:
    match = re.search(r"(?:CTX|COND)(\d+)$", str(reference_key))
    return int(match.group(1)) if match else 10**9


def _input_path(
    explicit: Path | None,
    atlas_dir: Path | None,
    relative: str,
    label: str,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if atlas_dir is None:
        raise ValueError(
            f"{label} is required when --atlas-dir is not supplied"
        )
    return atlas_dir / relative


def _ordered_references(frame: pd.DataFrame) -> pd.DataFrame:
    references = frame[
        frame["reference_level"].astype(str).isin({"subtype", "conditional"})
    ].copy()
    references["reference_key"] = references["reference_key"].astype(str)
    references["root"] = references["root"].astype(str)
    references["n_definition_members"] = pd.to_numeric(
        references["n_definition_members"], errors="raise"
    ).astype(int)
    references["layer"] = references["reference_level"].map(
        {"subtype": "core", "conditional": "conditional"}
    )
    if references["reference_key"].duplicated().any():
        duplicates = sorted(
            references.loc[
                references["reference_key"].duplicated(False), "reference_key"
            ].unique()
        )
        raise ValueError(f"Duplicate reference contexts: {duplicates}")

    root_totals = (
        references.groupby("root")["n_definition_members"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    references["_root_order"] = references["root"].map(
        {root: index for index, root in enumerate(root_totals.index)}
    )
    references["_layer_order"] = references["layer"].map(
        {"core": 0, "conditional": 1}
    )
    references["_context_order"] = references["reference_key"].map(
        _context_number
    )
    return references.sort_values(
        ["_root_order", "_layer_order", "_context_order", "reference_key"],
        kind="stable",
    ).reset_index(drop=True)


def _load_inputs(
    *,
    atlas_dir: Path | None,
    reference_contexts: Path | None,
    signature_events: Path | None,
    signature_architecture: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = atlas_dir.expanduser().resolve() if atlas_dir is not None else None
    reference_path = _input_path(
        reference_contexts,
        atlas,
        "release_candidate/tables/reference_contexts.tsv",
        "--reference-contexts",
    )
    signature_path = _input_path(
        signature_events,
        atlas,
        "release_candidate/tables/signature_events.tsv",
        "--signature-events",
    )
    architecture_path = _input_path(
        signature_architecture,
        atlas,
        (
            "validation/signature_architecture_v1/"
            "track2_event_segment_architecture.tsv"
        ),
        "--signature-architecture",
    )

    references = _ordered_references(
        _read_tsv(
            reference_path,
            {
                "reference_key",
                "root",
                "reference_level",
                "n_definition_members",
            },
            "reference contexts",
        )
    )
    signatures = _read_tsv(
        signature_path,
        {
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "direction",
            "context_value",
            "adjusted_effect_size",
            "segment_span_class",
            "segment_support_level",
            "track1_relation",
        },
        "signature events",
    )
    architecture = _read_tsv(
        architecture_path,
        {
            "reference_key",
            "root",
            "reference_level",
            "direction",
            "feature",
            "event",
            "amplitude",
            "gene_chromosome",
            "gene_start",
            "adjusted_effect_size",
            "segment_span_fraction_median",
        },
        "signature architecture",
    )

    known = set(references["reference_key"])
    signatures = signatures[
        signatures["reference_key"].astype(str).isin(known)
    ].copy()
    signatures["reference_key"] = signatures["reference_key"].astype(str)
    architecture = architecture[
        architecture["reference_key"].astype(str).isin(known)
    ].copy()
    architecture["reference_key"] = architecture["reference_key"].astype(str)

    release_track2 = (
        signatures.loc[
            signatures["track"].astype(str).eq("track2"),
            [
                "reference_key",
                "feature",
                "event",
                "amplitude",
                "direction",
                "segment_span_class",
                "segment_support_level",
                "track1_relation",
            ],
        ]
        .drop_duplicates()
    )
    release_key_columns = [
        "reference_key",
        "feature",
        "event",
        "amplitude",
        "direction",
    ]
    if release_track2.duplicated(release_key_columns).any():
        raise ValueError("Released Track 2 event keys are not unique")
    if len(release_track2) != int(
        signatures["track"].astype(str).eq("track2").sum()
    ):
        raise ValueError("Released Track 2 event keys are not unique")

    if architecture.duplicated(release_key_columns).any():
        raise ValueError("Signature architecture event keys are not unique")

    source_architecture_count = len(architecture)
    architecture = architecture.merge(
        release_track2,
        on=release_key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_validation", ""),
    )
    if len(architecture) != len(release_track2):
        raise ValueError(
            "Released Track 2 signatures do not match signature architecture "
            f"({len(release_track2)} release events; "
            f"{len(architecture)} matched events)"
        )
    if "adjusted_effect_size_validation" in architecture:
        architecture = architecture.drop(
            columns="adjusted_effect_size_validation"
        )
        architecture = architecture.merge(
            signatures.loc[
                signatures["track"].astype(str).eq("track2"),
                release_key_columns + ["adjusted_effect_size"],
            ],
            on=release_key_columns,
            how="left",
            validate="one_to_one",
        )
    architecture.attrs["source_track2_events"] = source_architecture_count
    architecture.attrs["released_track2_events"] = len(architecture)
    architecture.attrs["excluded_track2_events"] = (
        source_architecture_count - len(architecture)
    )

    architecture.attrs["references_without_track2"] = sorted(
        known - set(architecture["reference_key"])
    )
    return references, signatures, architecture


def _track1_color(color: str, prevalence: float) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=float)
    if prevalence <= 0:
        return matplotlib.colors.to_rgb(WHITE)
    scaled = np.clip(prevalence / TRACK1_DISPLAY_MAX, 0.0, 1.0)
    # A power transform preserves rank while making low-prevalence events visible.
    strength = float(scaled**TRACK1_COLOR_POWER)
    return tuple(1.0 - (1.0 - rgb) * strength)


def _track1_support(
    references: pd.DataFrame,
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    track1 = signatures[
        signatures["track"].astype(str).eq("track1")
        & signatures["direction"].astype(str).eq("enriched")
    ].copy()
    track1["chromosome"] = _chromosome(track1["feature"])
    track1["event"] = track1["event"].astype(str).str.lower()
    track1["context_value"] = pd.to_numeric(
        track1["context_value"], errors="coerce"
    )
    track1 = track1[
        track1["chromosome"].isin(AUTOSOMES)
        & track1["event"].isin({"gain", "loss"})
        & track1["context_value"].notna()
    ].copy()
    track1["context_value"] = track1["context_value"].clip(0.0, 1.0)
    track1 = (
        track1.sort_values("context_value", ascending=False, kind="stable")
        .drop_duplicates(
            ["reference_key", "chromosome", "event"], keep="first"
        )
        .merge(
            references[
                ["reference_key", "root", "n_definition_members"]
            ],
            on="reference_key",
            how="left",
            validate="many_to_one",
        )
    )
    track1["weighted_carriers"] = (
        track1["context_value"] * track1["n_definition_members"]
    )
    root_totals = references.groupby("root")["n_definition_members"].sum()
    support = (
        track1.groupby(["root", "chromosome", "event"])[
            "weighted_carriers"
        ]
        .sum()
        .rename("weighted_carriers")
        .reset_index()
    )
    support["support_fraction"] = support.apply(
        lambda row: row["weighted_carriers"] / root_totals.loc[row["root"]],
        axis=1,
    )
    return support


def _draw_track1(
    references: pd.DataFrame,
    signatures: pd.DataFrame,
    out_path: Path,
) -> None:
    support = _track1_support(references, signatures)
    root_order = (
        references.groupby("root")["n_definition_members"]
        .sum()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    support_lookup = {
        (str(row.root), int(row.chromosome), str(row.event)): float(
            row.support_fraction
        )
        for row in support.itertuples(index=False)
    }

    rgb = np.empty((len(root_order), len(AUTOSOMES), 3), dtype=float)
    neutral = matplotlib.colors.to_rgb(WHITE)
    rgb[:] = neutral
    for root_index, root in enumerate(root_order):
        for chromosome_index, chromosome in enumerate(AUTOSOMES):
            gain = support_lookup.get((root, chromosome, "gain"), 0.0)
            loss = support_lookup.get((root, chromosome, "loss"), 0.0)
            if gain > 0 and loss > 0:
                color = MIXED
                prevalence = gain + loss
            elif gain > 0:
                color = GAIN
                prevalence = gain
            elif loss > 0:
                color = LOSS
                prevalence = loss
            else:
                continue
            rgb[root_index, chromosome_index] = _track1_color(
                color, prevalence
            )

    fig, axis = plt.subplots(figsize=(7.2, 4.50))
    axis.imshow(rgb, aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(AUTOSOMES)))
    axis.set_xticklabels(AUTOSOMES, fontsize=A_SMALL_TEXT)
    axis.set_yticks(np.arange(len(root_order)))
    axis.set_yticklabels(
        [ROOT_LABELS.get(root, root) for root in root_order],
        fontsize=A_SMALL_TEXT,
    )
    axis.set_xlabel("Chromosome", fontsize=A_TEXT)
    axis.set_xticks(np.arange(-0.5, len(AUTOSOMES), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(root_order), 1), minor=True)
    axis.grid(which="minor", color=GRID, linewidth=0.32)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", colors=TEXT, length=2.5)
    for spine in axis.spines.values():
        spine.set_color("#9AA8AE")
        spine.set_linewidth(0.8)

    axis.legend(
        handles=[
            Patch(facecolor=GAIN, label="Gain"),
            Patch(facecolor=LOSS, label="Loss"),
            Patch(facecolor=MIXED, label="Both"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=A_SMALL_TEXT,
        handlelength=1.3,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.29, right=0.99, top=0.88, bottom=0.11)
    _save(fig, out_path)


def _track2_genomic_events(
    references: pd.DataFrame,
    architecture: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    valid_directions = {"enriched", "depleted"}
    valid_events = {"gain", "loss"}
    invalid_directions = sorted(
        set(architecture["direction"].dropna().astype(str))
        - valid_directions
    )
    invalid_events = sorted(
        set(architecture["event"].dropna().astype(str)) - valid_events
    )
    if invalid_directions:
        raise ValueError(
            f"Unsupported Track 2 directions: {invalid_directions}"
        )
    if invalid_events:
        raise ValueError(f"Unsupported Track 2 events: {invalid_events}")

    events = architecture.copy()
    events["gene_chromosome"] = pd.to_numeric(
        events["gene_chromosome"], errors="coerce"
    ).astype("Int64")
    events["gene_start"] = pd.to_numeric(
        events["gene_start"], errors="coerce"
    )
    events["adjusted_effect_size"] = pd.to_numeric(
        events["adjusted_effect_size"], errors="coerce"
    )
    events = events[
        events["gene_chromosome"].isin(AUTOSOMES)
        & events["gene_start"].notna()
    ].copy()
    events = events.drop_duplicates(
        ["reference_key", "feature", "event", "direction"],
        keep="first",
    )
    events = events.merge(
        references[["reference_key", "root", "layer"]],
        on="reference_key",
        how="left",
        suffixes=("_architecture", ""),
        validate="many_to_one",
    )
    if "root_architecture" in events:
        events = events.drop(columns="root_architecture")

    root_order = (
        references.groupby("root", sort=False)["n_definition_members"]
        .sum()
        .sort_values(ascending=False, kind="stable")
        .index.astype(str)
        .tolist()
    )
    events["chromosome_position"] = events.apply(
        lambda row: (
            int(row["gene_chromosome"])
            - 1
            + min(
                max(
                    float(row["gene_start"])
                    / CHROMOSOME_LENGTHS[int(row["gene_chromosome"])],
                    0.0,
                ),
                1.0,
            )
        ),
        axis=1,
    )
    return events, root_order


def _split_root_columns(
    references: pd.DataFrame,
    root_order: list[str],
) -> list[list[str]]:
    lane_counts = {
        root: (
            int((references["root"].astype(str) == root).sum())
            + ROOT_HEADER_HEIGHT
        )
        for root in root_order
    }
    target = sum(lane_counts.values()) / 2
    cumulative = 0
    split_index = 1
    best_delta = float("inf")
    for index, root in enumerate(root_order[:-1], start=1):
        cumulative += lane_counts[root]
        delta = abs(cumulative - target)
        if delta < best_delta:
            best_delta = delta
            split_index = index
    return [root_order[:split_index], root_order[split_index:]]


def _context_lane_label(reference_key: str) -> str:
    match = re.search(r"((?:CTX|COND)\d+)$", str(reference_key))
    return match.group(1) if match else str(reference_key)


def _draw_track2_subtype_landscape(
    references: pd.DataFrame,
    architecture: pd.DataFrame,
    out_path: Path,
) -> None:
    events, root_order = _track2_genomic_events(references, architecture)
    display_events = events[
        events["direction"].astype(str).eq("enriched")
    ].copy()
    root_columns = _split_root_columns(references, root_order)
    color_by_event = {"gain": GAIN, "loss": LOSS}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 7.40),
        gridspec_kw={"wspace": 0.20},
    )

    for column_index, (axis, column_roots) in enumerate(
        zip(axes, root_columns, strict=True)
    ):
        lane_by_reference: dict[str, float] = {}
        lane_labels: list[str] = []
        lane_positions: list[float] = []
        lane_layers: list[str] = []
        cursor = 0.0

        for root in column_roots:
            root_references = references[
                references["root"].astype(str).eq(root)
            ]
            header_start = cursor - 0.5
            header_y = header_start + ROOT_HEADER_HEIGHT / 2
            cursor += ROOT_HEADER_HEIGHT
            axis.axhspan(
                header_start,
                cursor - 0.5,
                facecolor=WHITE,
                edgecolor="none",
                zorder=1.5,
            )
            axis.axhline(
                cursor - 0.5,
                color="#B5C1C6",
                linewidth=0.52,
                zorder=2,
            )

            for row in root_references.itertuples(index=False):
                reference_key = str(row.reference_key)
                lane_by_reference[reference_key] = cursor
                lane_positions.append(cursor)
                lane_labels.append(_context_lane_label(reference_key))
                lane_layers.append(str(row.layer))
                if str(row.layer) == "conditional":
                    axis.axhspan(
                        cursor - 0.48,
                        cursor + 0.48,
                        facecolor=CONDITIONAL,
                        edgecolor="none",
                        alpha=0.085,
                        zorder=0,
                    )
                axis.axhline(
                    cursor + 0.5,
                    color="#D9E0E3",
                    linewidth=0.35,
                    zorder=1,
                )
                cursor += 1

            root_end = cursor - 0.5
            axis.axhline(
                root_end,
                color="#9FAEB5",
                linewidth=0.65,
                zorder=2,
            )
            axis.text(
                0.12,
                header_y,
                ROOT_LABELS.get(root, root),
                ha="left",
                va="center",
                fontsize=B_TEXT,
                color=TEXT,
                fontweight="bold",
                zorder=5,
            )

        column_events = display_events[
            display_events["root"].astype(str).isin(column_roots)
        ].copy()
        column_events["plot_y"] = column_events["reference_key"].map(
            lane_by_reference
        )
        for event in ("gain", "loss"):
            subset = column_events[
                column_events["event"].astype(str).eq(event)
            ]
            axis.scatter(
                subset["chromosome_position"],
                subset["plot_y"],
                s=3.7,
                color=color_by_event[event],
                alpha=0.76,
                linewidths=0,
                zorder=3,
            )

        contexts_with_events = set(
            column_events["reference_key"].astype(str)
        )
        for reference_key, lane_y in lane_by_reference.items():
            if reference_key not in contexts_with_events:
                axis.text(
                    21.65,
                    lane_y,
                    "no Track 2 signature",
                    ha="right",
                    va="center",
                    fontsize=B_MICRO_TEXT,
                    color=MUTED,
                    fontstyle="italic",
                    zorder=4,
                )

        for chromosome in AUTOSOMES:
            left = chromosome - 1
            if chromosome < AUTOSOMES[-1]:
                axis.axvline(
                    left + 1,
                    color="#CDD6DA",
                    linewidth=0.48,
                    zorder=1,
                )

        axis.set_xlim(0, 22)
        axis.set_ylim(cursor - 0.5, -0.5)
        axis.set_xticks(np.arange(0.5, 22.5, 1))
        axis.set_xticklabels(AUTOSOMES, fontsize=B_SMALL_TEXT)
        axis.set_xlabel("Chromosome", fontsize=B_TEXT)
        axis.set_yticks(lane_positions)
        if column_index == 1:
            axis.yaxis.tick_right()
            axis.tick_params(axis="y", labelright=True, labelleft=False)
        axis.set_yticklabels(lane_labels, fontsize=B_SMALL_TEXT)
        for tick, layer in zip(
            axis.get_yticklabels(), lane_layers, strict=True
        ):
            tick.set_color(CONDITIONAL if layer == "conditional" else TEXT)
            tick.set_fontweight("normal")
        axis.tick_params(axis="x", length=1.8, pad=2)
        axis.tick_params(axis="y", length=0, pad=2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9AA8AE")
        axis.spines["bottom"].set_color("#9AA8AE")
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=GAIN,
                markeredgecolor=GAIN,
                markersize=4.5,
                label="Gain",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=LOSS,
                markeredgecolor=LOSS,
                markersize=4.5,
                label="Loss",
            ),
            Patch(
                facecolor=WHITE,
                edgecolor="#9FAEB5",
                linewidth=0.7,
                label="Core subtype",
            ),
            Patch(
                facecolor=WHITE,
                edgecolor=CONDITIONAL,
                linewidth=0.7,
                label="Conditional subtype",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=4,
        frameon=False,
        fontsize=B_TEXT,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    legend = fig.legends[-1]
    if len(legend.get_texts()) >= 4:
        legend.get_texts()[3].set_color(CONDITIONAL)
    fig.subplots_adjust(left=0.085, right=0.915, top=0.955, bottom=0.065)
    _save(fig, out_path)


def _draw_cross_scale_relationship(
    enriched: pd.DataFrame,
    out_path: Path,
) -> None:
    enriched = enriched.copy()
    enriched["architecture_class"] = enriched[
        "segment_span_class"
    ].astype("string")
    invalid = sorted(
        set(enriched["architecture_class"].dropna())
        - set(ARCHITECTURE_ORDER)
    )
    if invalid:
        raise ValueError(f"Unsupported segment architecture classes: {invalid}")
    class_colors = {
        "focal": "#0E8C7F",
        "regional": "#3F6FA8",
        "broad": "#5B4B8A",
        "mixed": "#7A8F35",
    }
    displayed_architectures = ARCHITECTURE_ORDER
    enriched["span_percent"] = (
        pd.to_numeric(
            enriched["segment_span_fraction_median"],
            errors="coerce",
        )
        * 100.0
    )
    enriched = enriched[
        enriched["span_percent"].notna() & enriched["span_percent"].gt(0)
    ].copy()

    positions = np.arange(1, len(displayed_architectures) + 1)
    plot_data: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(20260728)
    for position, architecture in zip(
        positions, displayed_architectures, strict=True
    ):
        subset = enriched[
            enriched["architecture_class"].astype(str).eq(architecture)
        ].copy()
        points = subset
        if len(points) > 320:
            points = points.sample(
                n=320,
                random_state=1000 + position,
            )
        plot_data[architecture] = {
            "values": subset["span_percent"].to_numpy(dtype=float),
            "point_values": points["span_percent"].to_numpy(dtype=float),
            "jitter": rng.normal(
                0.0,
                0.115 if architecture == "focal" else 0.075,
                size=len(points),
            ),
        }

    fig, axis = plt.subplots(figsize=(5.6, 4.20))
    for position, architecture in zip(
        positions, displayed_architectures, strict=True
    ):
        values = plot_data[architecture]["values"]
        color = class_colors[architecture]
        boxplot = axis.boxplot(
            values,
            positions=[position],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": TEXT, "linewidth": 1.2, "zorder": 5},
            boxprops={
                "facecolor": color,
                "alpha": 0.43,
                "edgecolor": TEXT,
                "linewidth": 0.8,
                "zorder": 4,
            },
            whiskerprops={"color": TEXT, "linewidth": 0.8, "zorder": 4},
            capprops={"color": TEXT, "linewidth": 0.8, "zorder": 4},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.43)
        axis.scatter(
            np.full(
                len(plot_data[architecture]["point_values"]),
                position,
            )
            + plot_data[architecture]["jitter"],
            plot_data[architecture]["point_values"],
            s=8 if architecture == "focal" else 10,
            marker="o",
            color=color,
            alpha=0.30 if architecture == "focal" else 0.25,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    axis.set_yscale("log")
    axis.set_ylim(5e-5, 120.0)
    axis.set_xlim(0.48, len(displayed_architectures) + 0.52)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [ARCHITECTURE_LABELS[value] for value in displayed_architectures],
        fontsize=C_SMALL_TEXT,
    )
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(
        axis="y",
        labelsize=C_SMALL_TEXT,
        length=2.5,
        pad=2,
    )
    axis.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10.0, numticks=8)
    )
    axis.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    axis.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100,
        )
    )
    axis.grid(axis="y", which="major", color=GRID, linewidth=0.65)
    axis.grid(axis="y", which="minor", color="#EDF1F3", linewidth=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA8AE")
    axis.spines["bottom"].set_color("#9AA8AE")

    for threshold, label in (
        (2.0, "2% chromosome"),
        (50.0, "50% chromosome"),
    ):
        axis.axhline(
            threshold,
            color="#9C968B",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        axis.text(
            0.02,
            threshold,
            label,
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=C_SMALL_TEXT,
            color=MUTED,
        )

    for position, architecture in zip(
        positions, displayed_architectures, strict=True
    ):
        axis.text(
            position,
            1.035,
            f"n={len(plot_data[architecture]['values']):,}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=C_SMALL_TEXT,
            color=TEXT,
        )

    axis.set_ylabel(
        "Median carrier-segment span\n(% chromosome, log scale)",
        fontsize=C_TEXT,
        color=TEXT,
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.15)
    _save(fig, out_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the final Figure 4 signature architecture panels."
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        help="Atlas root used to resolve inputs not supplied explicitly.",
    )
    parser.add_argument("--reference-contexts", type=Path)
    parser.add_argument("--signature-events", type=Path)
    parser.add_argument("--signature-architecture", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=("A", "B", "C"),
        default=("A", "B", "C"),
        help="Panels to render. Use '--panels B C' to preserve an existing A.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "track1": out_dir / "A_track1_chromosome_landscape.png",
        "track2_subtype_landscape": out_dir
        / "B_track2_subtype_landscape.png",
        "track2_cross_scale": out_dir
        / "C_track2_segment_architecture.png",
    }
    selected = {
        "A": "track1",
        "B": "track2_subtype_landscape",
        "C": "track2_cross_scale",
    }
    selected_outputs = {
        name: outputs[selected[name]] for name in args.panels
    }
    existing = [path for path in selected_outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Figure 4 outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    references, signatures, architecture = _load_inputs(
        atlas_dir=args.atlas_dir,
        reference_contexts=args.reference_contexts,
        signature_events=args.signature_events,
        signature_architecture=args.signature_architecture,
    )
    if "A" in args.panels:
        _draw_track1(references, signatures, outputs["track1"])
    enriched = architecture[
        architecture["direction"].astype(str).eq("enriched")
    ].copy()
    if "B" in args.panels:
        _draw_track2_subtype_landscape(
            references,
            architecture,
            outputs["track2_subtype_landscape"],
        )
    if "C" in args.panels:
        _draw_cross_scale_relationship(
            enriched,
            outputs["track2_cross_scale"],
        )

    counts = references["layer"].value_counts()
    print("[CNVSIM] Figure 4 signature architecture complete")
    print(
        "  references: "
        f"{len(references)} "
        f"(core={int(counts.get('core', 0))}, "
        f"conditional={int(counts.get('conditional', 0))})"
    )
    print(f"  roots: {references['root'].nunique()}")
    print(
        "  Track 2 release filter: "
        f"{architecture.attrs.get('source_track2_events', len(architecture))} "
        "source -> "
        f"{architecture.attrs.get('released_track2_events', len(architecture))} "
        "released"
    )
    print(f"  enriched Track 2 events: {len(enriched)}")
    print("  Track 2 segment span classes:")
    print(
        enriched["segment_span_class"]
        .value_counts()
        .reindex(ARCHITECTURE_ORDER, fill_value=0)
        .to_string()
    )
    print("  Track 2 support levels:")
    print(enriched["segment_support_level"].value_counts().to_string())
    for name, path in selected_outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
