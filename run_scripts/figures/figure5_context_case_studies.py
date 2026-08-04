#!/usr/bin/env python3
from __future__ import annotations

"""Render three root-specific CNV context case studies for Figure 5."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


AUTOSOMES = list(range(1, 23))
WHITE = "#FFFFFF"
TEXT = "#111111"
MUTED = "#687379"
GRID = "#E5EAEC"
NEUTRAL = "#F1EEE8"
CONDITIONAL = "#D6872C"
CONDITIONAL_BACKGROUND = "#FFF5EC"
TRACK1_GAIN = "#E1B83B"
TRACK1_LOSS = "#2E5EAA"
TRACK1_DUAL = "#8651A3"
TRACK2_GAIN = "#E8783D"
TRACK2_LOSS = "#2E69A5"
CONTENT_WIDTH_FRACTION = 0.80
TOPOGRAPHY_COLORS = ("#4F8C70", "#A7CAB3", "#ECEAE5")
MORPHOLOGY_COLORS = ("#C9785B", "#E3B2A0", "#ECEAE5")

CASE_STUDIES = (
    {
        "panel": "A",
        "root": "C18_C20",
        "title": "C18-C20  Colon/rectum",
        "contexts": (
            "C18_C20-CTX1",
            "C18_C20-CTX5",
            "C18_C20-CTX6",
            "C18_C20-CTX8",
        ),
    },
    {
        "panel": "B",
        "root": "C71",
        "title": "C71  Brain",
        "contexts": (
            "C71-CTX0",
            "C71-CTX2",
            "C71-CTX5",
            "C71-CTX13",
        ),
    },
    {
        "panel": "C",
        "root": "C16",
        "title": "C16  Stomach",
        "contexts": (
            "C16-CTX0",
            "C16-CTX1",
            "C16-COND3",
            "C16-COND4",
        ),
    },
)

GENE_LABELS = {
    "C16-COND3": ("CDH17", "RSPO2", "RAD21", "MYC", "NDRG1"),
    "C16-COND4": ("MAX", "RAD51B", "DICER1", "HIF1A", "NFKBIA"),
    "C18_C20-CTX1": ("CASP8", "PMS1", "SF3B1", "BCL2L11", "IDH1"),
    "C18_C20-CTX5": ("MAP2K2", "BRCA1", "STAT3", "KEAP1", "SMARCA4"),
    "C18_C20-CTX6": ("JAK1", "MUTYH", "AURKA", "CDX2", "IRS2"),
    "C18_C20-CTX8": ("ERCC1", "ERCC2", "SMARCA4", "KEAP1", "NOTCH3"),
    "C71-CTX0": ("EGFR", "PTEN", "SUFU", "BMPR1A", "TET1"),
    "C71-CTX2": ("EGFR", "PTEN", "SUFU", "SMARCA4", "NOTCH3"),
    "C71-CTX5": ("ARID1A", "MYCL", "MUTYH", "PIK3R3", "JAK1"),
    "C71-CTX13": ("PPM1D", "RPS6KB1", "SOX9", "BIRC5", "H3-3B"),
}

LABEL_Y_POSITIONS = {
    "C18_C20-CTX5": {
        "MAP2K2−": 0.86,
        "KEAP1−": 0.58,
        "SMARCA4−": 0.40,
        "BRCA1−": 0.86,
        "STAT3−": 0.06,
    },
    "C18_C20-CTX6": {
        "AURKA+": 0.90,
        "JAK1−": 0.64,
        "MUTYH−": 0.38,
        "CDX2+": 0.98,
        "IRS2+": 0.12,
    },
    "C71-CTX2": {
        "SMARCA4+": 0.90,
        "NOTCH3+": 0.68,
        "PTEN−": 0.45,
        "SUFU−": 0.22,
        "EGFR+": 0.50,
    },
}

PANEL_FILENAMES = {
    "A": "A_C18_C20_colon_rectum.png",
    "B": "B_C71_brain.png",
    "C": "C_C16_stomach.png",
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


def _load_inputs(
    release_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    references = _read_tsv(
        release_dir / "tables" / "reference_contexts.tsv",
        {
            "reference_key",
            "reference_level",
            "n_definition_members",
        },
        "reference contexts",
    )
    annotations = _read_tsv(
        release_dir / "tables" / "context_annotations.tsv",
        {
            "reference_key",
            "annotation_type",
            "annotation_label",
            "fraction",
            "rank",
        },
        "context annotations",
    )
    signatures = _read_tsv(
        release_dir / "tables" / "signature_events.tsv",
        {
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "direction",
        },
        "signature events",
    )
    architecture = _read_tsv(
        release_dir
        / "validation"
        / "signature_architecture_v1"
        / "track2_event_segment_architecture.tsv",
        {
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "direction",
            "encoded_event_prevalence",
            "adjusted_effect_size",
            "segment_span_fraction_median",
        },
        "Track 2 segment architecture",
    )

    selected = {
        context
        for case in CASE_STUDIES
        for context in case["contexts"]
    }
    missing = sorted(selected - set(references["reference_key"].astype(str)))
    if missing:
        raise ValueError(f"Selected Figure 5 contexts are absent: {missing}")

    references = references[
        references["reference_key"].astype(str).isin(selected)
    ].copy()
    references["n_definition_members"] = pd.to_numeric(
        references["n_definition_members"], errors="raise"
    ).astype(int)
    annotations = annotations[
        annotations["reference_key"].astype(str).isin(selected)
    ].copy()
    annotations["fraction"] = pd.to_numeric(
        annotations["fraction"], errors="coerce"
    )
    annotations["rank"] = pd.to_numeric(
        annotations["rank"], errors="coerce"
    )
    signatures = signatures[
        signatures["reference_key"].astype(str).isin(selected)
    ].copy()

    released_track2 = signatures[
        signatures["track"].astype(str).eq("track2")
    ][
        [
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "direction",
        ]
    ].drop_duplicates()
    architecture = architecture[
        architecture["reference_key"].astype(str).isin(selected)
        & architecture["track"].astype(str).eq("track2")
        & architecture["direction"].astype(str).eq("enriched")
    ].copy()
    architecture = architecture.merge(
        released_track2,
        on=[
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "direction",
        ],
        how="inner",
        validate="one_to_one",
    )
    for column in (
        "encoded_event_prevalence",
        "adjusted_effect_size",
        "segment_span_fraction_median",
    ):
        architecture[column] = pd.to_numeric(
            architecture[column], errors="coerce"
        )
    return references, annotations, signatures, architecture


def _short_label(value: object) -> str:
    label = str(value)
    replacements = {
        "Adenocarcinoma, NOS": "Adenocarcinoma",
        "Adenocarcinoma, intestinal type": "Intestinal adenocarcinoma",
        "Gastrointestinal stromal sarcoma": "GIST sarcoma",
        "Gastrointestinal stromal tumor, NOS": "GIST",
        "Gastrointestinal stromal tumor, benign": "GIST, benign",
        "Oligodendroglioma, NOS": "Oligodendroglioma",
        "Oligodendroglioma, anaplastic": "Anaplastic oligodendroglioma",
        "Medulloblastoma, NOS": "Medulloblastoma",
        "Glioblastoma, NOS": "Glioblastoma",
        "Brain, NOS": "Brain",
        "Cerebellum, NOS": "Cerebellum",
        "Colon, NOS": "Colon",
        "Rectum, NOS": "Rectum",
        "Stomach, NOS": "Stomach",
    }
    return replacements.get(label, label.replace(", NOS", ""))


def _annotation_entries(
    annotations: pd.DataFrame,
    reference_key: str,
    annotation_type: str,
    *,
    maximum: int = 2,
) -> list[tuple[str, float]]:
    rows = annotations[
        annotations["reference_key"].astype(str).eq(reference_key)
        & annotations["annotation_type"].astype(str).eq(annotation_type)
    ].sort_values(["rank", "fraction"], ascending=[True, False])
    entries: list[tuple[str, float]] = []
    for row in rows.head(maximum).itertuples(index=False):
        if pd.isna(row.fraction):
            continue
        entries.append((_short_label(row.annotation_label), float(row.fraction)))
    return entries


def _ellipsize(value: str, maximum: int = 25) -> str:
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


def _draw_annotation_bar(
    axis: plt.Axes,
    entries: list[tuple[str, float]],
    y: float,
    colors: tuple[str, str, str],
) -> None:
    values = [fraction for _, fraction in entries[:2]]
    values.extend([0.0] * (2 - len(values)))
    other = max(0.0, 1.0 - sum(values))
    left = 0.0
    for value, color in zip(values + [other], colors, strict=True):
        axis.barh(
            y,
            value,
            left=left,
            height=0.13,
            color=color,
            edgecolor=WHITE,
            linewidth=0.35,
        )
        left += value


def _draw_diagnostic_composition(
    axis: plt.Axes,
    annotations: pd.DataFrame,
    reference_key: str,
) -> None:
    topography = _annotation_entries(
        annotations, reference_key, "icdo_topography"
    )
    morphology = _annotation_entries(
        annotations, reference_key, "icdo_morphology"
    )
    _draw_annotation_bar(axis, topography, 0.62, TOPOGRAPHY_COLORS)
    _draw_annotation_bar(axis, morphology, 0.18, MORPHOLOGY_COLORS)

    topography_text = (
        f"T: {_ellipsize(topography[0][0])} {topography[0][1]:.0%}"
        if topography
        else "T: unavailable"
    )
    morphology_text = (
        f"M: {_ellipsize(morphology[0][0])} {morphology[0][1]:.0%}"
        if morphology
        else "M: unavailable"
    )
    axis.text(
        0.0,
        0.88,
        topography_text,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=5.7,
        color=TEXT,
    )
    axis.text(
        0.0,
        0.43,
        morphology_text,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=5.7,
        color=TEXT,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_track1(
    axis: plt.Axes,
    signatures: pd.DataFrame,
    reference_key: str,
    *,
    conditional: bool,
) -> None:
    rows = signatures[
        signatures["reference_key"].astype(str).eq(reference_key)
        & signatures["track"].astype(str).eq("track1")
    ].copy()
    calls: dict[int, set[str]] = {chromosome: set() for chromosome in AUTOSOMES}
    for row in rows.itertuples(index=False):
        chromosome = str(row.feature).lower().replace("chr", "")
        if not chromosome.isdigit():
            continue
        number = int(chromosome)
        if number in calls:
            calls[number].add(str(row.event))

    colors = []
    for chromosome in AUTOSOMES:
        events = calls[chromosome]
        if {"gain", "loss"}.issubset(events):
            colors.append(TRACK1_DUAL)
        elif "gain" in events:
            colors.append(TRACK1_GAIN)
        elif "loss" in events:
            colors.append(TRACK1_LOSS)
        else:
            colors.append(
                CONDITIONAL_BACKGROUND if conditional else NEUTRAL
            )
    rgb = np.asarray(
        [matplotlib.colors.to_rgb(color) for color in colors],
        dtype=float,
    )[None, :, :]
    axis.imshow(rgb, aspect="auto", interpolation="none")
    axis.set_xlim(-0.5, len(AUTOSOMES) - 0.5)
    axis.set_xticks([0, 4, 8, 12, 16, 20])
    axis.set_xticklabels(["1", "5", "9", "13", "17", "21"], fontsize=5.2)
    axis.set_yticks([])
    axis.tick_params(axis="x", length=1.5, width=0.45, pad=1)
    axis.set_xticks(np.arange(-0.5, len(AUTOSOMES), 1), minor=True)
    axis.grid(which="minor", color=WHITE, linewidth=0.5)
    axis.tick_params(which="minor", bottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_track2(
    axis: plt.Axes,
    architecture: pd.DataFrame,
    reference_key: str,
    *,
    show_y_ticks: bool,
) -> None:
    rows = architecture[
        architecture["reference_key"].astype(str).eq(reference_key)
    ].dropna(
        subset=[
            "segment_span_fraction_median",
            "encoded_event_prevalence",
        ]
    ).copy()
    rows["span_pct"] = rows["segment_span_fraction_median"] * 100.0
    rows["carrier_pct"] = rows["encoded_event_prevalence"] * 100.0
    rows = rows[rows["span_pct"].gt(0)].copy()

    if rows.empty:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.52,
            "No Track 2 signature",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=6.2,
            color=MUTED,
        )
        return

    for event, color in (("gain", TRACK2_GAIN), ("loss", TRACK2_LOSS)):
        group = rows[rows["event"].astype(str).eq(event)]
        axis.scatter(
            group["span_pct"],
            group["carrier_pct"],
            s=11,
            c=color,
            edgecolors=WHITE,
            linewidths=0.2,
            alpha=0.5,
            zorder=2,
            rasterized=True,
        )

    def limits(values: pd.Series, minimum_span: float) -> tuple[float, float]:
        lower = float(values.min())
        upper = float(values.max())
        span = max(upper - lower, minimum_span)
        center = (lower + upper) / 2.0
        padding = span * 0.14
        return (
            max(-3.0, center - span / 2.0 - padding),
            min(103.0, center + span / 2.0 + padding),
        )

    x_limits = limits(rows["span_pct"], 18.0)
    y_limits = limits(rows["carrier_pct"], 20.0)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)

    selected_markers: list[dict[str, object]] = []
    for gene in GENE_LABELS.get(reference_key, ()):
        marker = rows[rows["feature"].astype(str).eq(gene)].copy()
        if marker.empty:
            continue
        marker = marker.sort_values(
            ["adjusted_effect_size", "carrier_pct"],
            ascending=False,
        ).iloc[0]
        color = (
            TRACK2_GAIN if str(marker["event"]) == "gain" else TRACK2_LOSS
        )
        sign = "+" if str(marker["event"]) == "gain" else "−"
        selected_markers.append(
            {
                "label": f"{gene}{sign}",
                "x": float(marker["span_pct"]),
                "y": float(marker["carrier_pct"]),
                "event": str(marker["event"]),
                "color": color,
            }
        )

    labelled_markers: list[dict[str, object]] = []
    clusters: list[list[dict[str, object]]] = []
    for event in ("gain", "loss"):
        event_markers = sorted(
            (
                marker
                for marker in selected_markers
                if marker["event"] == event
            ),
            key=lambda marker: float(marker["x"]),
        )
        for marker in event_markers:
            if (
                clusters
                and clusters[-1][0]["event"] == event
                and abs(
                    float(marker["x"])
                    - float(
                        np.median(
                            [
                                float(value["x"])
                                for value in clusters[-1]
                            ]
                        )
                    )
                )
                <= 0.3
            ):
                clusters[-1].append(marker)
            else:
                clusters.append([marker])

    for cluster in clusters:
        anchor_x = float(np.median([float(marker["x"]) for marker in cluster]))
        anchor_y = float(np.median([float(marker["y"]) for marker in cluster]))
        axis.scatter(
            [anchor_x],
            [anchor_y],
            s=23,
            c=str(cluster[0]["color"]),
            edgecolors=TEXT,
            linewidths=0.55,
            zorder=4,
        )
        side = "right" if anchor_x <= sum(x_limits) / 2.0 else "left"
        for marker in cluster:
            labelled_markers.append(
                {
                    "label": marker["label"],
                    "x": anchor_x,
                    "y": anchor_y,
                    "side": side,
                }
            )

    for side in ("left", "right"):
        markers = [
            marker for marker in labelled_markers if marker["side"] == side
        ]
        markers.sort(key=lambda marker: float(marker["y"]), reverse=True)
        y_positions = (
            [0.5]
            if len(markers) == 1
            else np.linspace(0.86, 0.14, len(markers))
        )
        x_position = 0.04 if side == "left" else 0.96
        alignment = "left" if side == "left" else "right"
        for marker, y_position in zip(markers, y_positions, strict=True):
            y_position = LABEL_Y_POSITIONS.get(reference_key, {}).get(
                str(marker["label"]),
                float(y_position),
            )
            axis.annotate(
                str(marker["label"]),
                xy=(float(marker["x"]), float(marker["y"])),
                xytext=(x_position, float(y_position)),
                textcoords=axis.transAxes,
                ha=alignment,
                va="center",
                fontsize=5.4,
                color=TEXT,
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.4,
                    "color": "#788489",
                    "shrinkA": 1,
                    "shrinkB": 2,
                },
                annotation_clip=False,
                zorder=5,
            )

    x_tick_limits = (max(0.0, x_limits[0]), min(100.0, x_limits[1]))
    y_tick_limits = (max(0.0, y_limits[0]), min(100.0, y_limits[1]))
    axis.set_xticks(np.linspace(*x_tick_limits, 3))
    axis.set_yticks(np.linspace(*y_tick_limits, 3))
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    )
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    )
    axis.tick_params(axis="both", labelsize=5.2, length=1.6, width=0.45)
    axis.grid(axis="both", color=GRID, linewidth=0.45)
    axis.set_xlabel("Segment span (% chromosome)", fontsize=5.8, labelpad=1)
    if not show_y_ticks:
        axis.set_yticklabels([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA8AE")
    axis.spines["bottom"].set_color("#9AA8AE")


def _context_title(reference: pd.Series) -> tuple[str, str]:
    reference_key = str(reference["reference_key"])
    short_key = reference_key.split("-", 1)[1]
    layer = (
        "Conditional"
        if str(reference["reference_level"]) == "conditional"
        else "Core"
    )
    title = f"{short_key}  |  {layer}"
    subtitle = f"n={int(reference['n_definition_members']):,}"
    return title, subtitle


def _draw_case_panel(
    case: dict[str, object],
    references: pd.DataFrame,
    annotations: pd.DataFrame,
    signatures: pd.DataFrame,
    architecture: pd.DataFrame,
    out_path: Path,
) -> None:
    contexts = tuple(str(value) for value in case["contexts"])
    fig = plt.figure(figsize=(7.2, 2.65), facecolor=WHITE)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=len(contexts),
        height_ratios=(0.72, 0.20, 1.32),
        left=0.11,
        right=0.98,
        top=0.84,
        bottom=0.10,
        hspace=0.18,
        wspace=0.02,
    )

    fig.text(
        0.012,
        0.98,
        str(case["panel"]),
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
        color=TEXT,
    )
    fig.text(
        0.055,
        0.98,
        str(case["title"]),
        ha="left",
        va="top",
        fontsize=10.5,
        weight="bold",
        color=TEXT,
    )
    track2_label_column = next(
        (
            column
            for column, reference_key in enumerate(contexts)
            if not architecture[
                architecture["reference_key"].astype(str).eq(reference_key)
            ].empty
        ),
        0,
    )
    for row, label in enumerate(
        (
            "Diagnostic\ncomposition",
            "Track 1",
            "Track 2\ncarrier\nfraction (%)",
        )
    ):
        position = grid[row, 0].get_position(fig)
        fig.text(
            0.095,
            (position.y0 + position.y1) / 2.0,
            label,
            ha="right",
            va="center",
            fontsize=6.5,
            color=TEXT,
        )
    for column, reference_key in enumerate(contexts):
        reference = references[
            references["reference_key"].astype(str).eq(reference_key)
        ].iloc[0]
        conditional = str(reference["reference_level"]) == "conditional"
        axes = (
            fig.add_subplot(grid[0, column]),
            fig.add_subplot(grid[1, column]),
            fig.add_subplot(grid[2, column]),
        )
        cell_position = axes[0].get_position()
        content_width = cell_position.width * CONTENT_WIDTH_FRACTION
        content_x0 = (
            cell_position.x0 + (cell_position.width - content_width) / 2.0
        )
        for axis in axes:
            position = axis.get_position()
            axis.set_position(
                [
                    content_x0,
                    position.y0,
                    content_width,
                    position.height,
                ]
            )
        if conditional:
            positions = [axis.get_position() for axis in axes]
            x0 = cell_position.x0 - 0.002
            x1 = cell_position.x1 + 0.002
            y0 = min(position.y0 for position in positions) - 0.070
            y1 = max(position.y1 for position in positions) + 0.072
            fig.add_artist(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    transform=fig.transFigure,
                    facecolor=CONDITIONAL_BACKGROUND,
                    edgecolor="none",
                    zorder=-1,
                )
            )
        for axis in axes:
            axis.set_facecolor(
                CONDITIONAL_BACKGROUND if conditional else WHITE
            )

        title, subtitle = _context_title(reference)
        axes[0].set_title(
            f"{title}\n{subtitle}",
            fontsize=6.8,
            weight="normal" if conditional else "bold",
            color=CONDITIONAL if conditional else TEXT,
            pad=3,
        )
        _draw_diagnostic_composition(
            axes[0],
            annotations,
            reference_key,
        )
        _draw_track1(
            axes[1],
            signatures,
            reference_key,
            conditional=conditional,
        )
        _draw_track2(
            axes[2],
            architecture,
            reference_key,
            show_y_ticks=column == track2_label_column,
        )

    fig.savefig(out_path, dpi=300, facecolor=WHITE)
    plt.close(fig)


def _legend_handles() -> list[Patch | Line2D]:
    return [
        Patch(facecolor=TRACK1_GAIN, label="Track 1 gain"),
        Patch(facecolor=TRACK1_LOSS, label="Track 1 loss"),
        Patch(facecolor=TRACK1_DUAL, label="Track 1 gain/loss"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TRACK2_GAIN,
            markeredgecolor="none",
            markersize=4.5,
            label="Track 2 gain",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TRACK2_LOSS,
            markeredgecolor="none",
            markersize=4.5,
            label="Track 2 loss",
        ),
    ]


def _draw_shared_legend(out_path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 0.26), facecolor=WHITE)
    fig.legend(
        handles=_legend_handles(),
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=5,
        frameon=False,
        fontsize=6.2,
        handlelength=1.0,
        columnspacing=1.4,
    )
    fig.savefig(out_path, dpi=300, facecolor=WHITE)
    plt.close(fig)


def _assemble_panels(
    panel_paths: list[Path],
    legend_path: Path,
    out_path: Path,
) -> tuple[int, int]:
    panels: list[Image.Image] = []
    for path in panel_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            panels.append(source.convert("RGB"))
    width = max(panel.width for panel in panels)
    if any(panel.width != width for panel in panels):
        resized = []
        for panel in panels:
            height = round(panel.height * width / panel.width)
            resized.append(
                panel.resize((width, height), Image.Resampling.LANCZOS)
            )
        panels = resized
    if not legend_path.exists():
        raise FileNotFoundError(legend_path)
    with Image.open(legend_path) as source:
        legend = source.convert("RGB")
    if legend.width != width:
        legend_height = round(legend.height * width / legend.width)
        legend = legend.resize(
            (width, legend_height),
            Image.Resampling.LANCZOS,
        )
    panel_gap = 30
    legend_gap = 24
    canvas = Image.new(
        "RGB",
        (
            width,
            sum(panel.height for panel in panels)
            + panel_gap * (len(panels) - 1)
            + legend_gap
            + legend.height,
        ),
        "white",
    )
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + panel_gap
    y += legend_gap - panel_gap
    canvas.paste(legend, (0, y))
    canvas.save(out_path, format="PNG", dpi=(300, 300), optimize=True)
    return canvas.size


def _write_selected_contexts(
    references: pd.DataFrame,
    annotations: pd.DataFrame,
    signatures: pd.DataFrame,
    architecture: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    for case in CASE_STUDIES:
        for reference_key in case["contexts"]:
            reference = references[
                references["reference_key"].astype(str).eq(reference_key)
            ].iloc[0]
            morphology = _annotation_entries(
                annotations,
                str(reference_key),
                "icdo_morphology",
                maximum=1,
            )
            topography = _annotation_entries(
                annotations,
                str(reference_key),
                "icdo_topography",
                maximum=1,
            )
            track1 = signatures[
                signatures["reference_key"].astype(str).eq(reference_key)
                & signatures["track"].astype(str).eq("track1")
            ]
            track2 = architecture[
                architecture["reference_key"].astype(str).eq(reference_key)
            ]
            rows.append(
                {
                    "panel": case["panel"],
                    "root": case["root"],
                    "reference_key": reference_key,
                    "reference_level": reference["reference_level"],
                    "n_definition_members": int(
                        reference["n_definition_members"]
                    ),
                    "top_morphology": morphology[0][0] if morphology else "",
                    "top_morphology_fraction": (
                        morphology[0][1] if morphology else np.nan
                    ),
                    "top_topography": topography[0][0] if topography else "",
                    "top_topography_fraction": (
                        topography[0][1] if topography else np.nan
                    ),
                    "n_track1_events": len(track1),
                    "n_released_track2_events": len(track2),
                    "labelled_track2_genes": ";".join(
                        GENE_LABELS.get(str(reference_key), ())
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render Figure 5 case studies from a frozen CNVSIM release."
        )
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="Frozen release_candidate directory.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        case["panel"]: out_dir / PANEL_FILENAMES[str(case["panel"])]
        for case in CASE_STUDIES
    }
    composite = out_dir / "Figure5.png"
    shared_legend = out_dir / "figure5_shared_legend.png"
    selected_contexts = out_dir / "figure5_selected_contexts.tsv"
    expected = [
        *outputs.values(),
        composite,
        shared_legend,
        selected_contexts,
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Figure 5 outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    references, annotations, signatures, architecture = _load_inputs(
        release_dir
    )
    for case in CASE_STUDIES:
        _draw_case_panel(
            case,
            references,
            annotations,
            signatures,
            architecture,
            outputs[str(case["panel"])],
        )
    _draw_shared_legend(shared_legend)
    width, height = _assemble_panels(
        [outputs[str(case["panel"])] for case in CASE_STUDIES],
        shared_legend,
        composite,
    )
    _write_selected_contexts(
        references,
        annotations,
        signatures,
        architecture,
        selected_contexts,
    )

    print("[CNVSIM] Figure 5 context case studies complete")
    for panel, path in outputs.items():
        print(f"  {panel}: {path}")
    print(f"  shared_legend: {shared_legend}")
    print(f"  composite: {composite}")
    print(f"  selected_contexts: {selected_contexts}")
    print(f"  composite_pixels: {width} x {height}")


if __name__ == "__main__":
    main()
