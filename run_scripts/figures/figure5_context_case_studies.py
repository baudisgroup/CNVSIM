#!/usr/bin/env python3
from __future__ import annotations

"""Render Figure 5: breast, colorectal and brain CNV subtype examples.

Read CNV tables and segment architecture from ATLAS_ROOT/release_candidate
and mutation annotations from the public --release-dir. Default outputs are
ATLAS_ROOT/figures/figure5/Figure5.pdf and Figure5.png. Use --out-dir to
override the destination, and --overwrite to replace these two files.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


AUTOSOMES = list(range(1, 23))
WHITE = "#FFFFFF"
TEXT = "#111111"
MUTED = "#687379"
GRID = "#E5EAEC"
NEUTRAL = "#F1EEE8"
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
        "root": "C50",
        "title": "C50  Breast",
        "contexts": (
            "C50-CTX0",
            "C50-CTX8",
            "C50-CTX3",
            "C50-CTX4",
        ),
    },
    {
        "panel": "B",
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
        "panel": "C",
        "root": "C71",
        "title": "C71  Brain",
        "contexts": (
            "C71-CTX0",
            "C71-CTX2",
            "C71-CTX5",
            "C71-CTX13",
        ),
    },
)

GENE_LABELS = {
    "C50-CTX0": ("PIK3R1", "MAP3K1", "MSH3", "MYC", "RAD21"),
    "C50-CTX8": ("BARD1", "ERBB4", "IDH1", "CASP8", "LRP1B"),
    "C50-CTX3": ("ALK", "MSH2", "MSH6", "PIK3R1", "MSH3"),
    "C50-CTX4": ("MDM2", "CDK4", "FRS2", "HMGA2", "ERBB3"),
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
    "C50-CTX0": {
        "PIK3R1−": 0.90,
        "MSH3−": 0.92,
        "MAP3K1−": 0.14,
    },
    "C18_C20-CTX5": {
        "MAP2K2−": 0.86,
        "KEAP1−": 0.58,
        "SMARCA4−": 0.40,
        "BRCA1−": 0.86,
        "STAT3−": 0.18,
    },
    "C18_C20-CTX6": {
        "AURKA+": 0.90,
        "JAK1−": 0.64,
        "MUTYH−": 0.38,
        "CDX2+": 0.92,
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

# Keep the selected breast example leaders from crossing each other.
LABEL_SIDES = {"C50-CTX0": {"PIK3R1−": "right", "MYC+": "above", "RAD21+": "below"}}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "font.size": 6.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
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
        "Infiltrating duct carcinoma, NOS": "Ductal carcinoma",
        "Breast, NOS": "Breast",
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
        fontsize=6.0,
        color=TEXT,
    )
    axis.text(
        0.0,
        0.43,
        morphology_text,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=6.0,
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
    axis.set_xticklabels(["1", "5", "9", "13", "17", "21"], fontsize=6.0)
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
    ].copy()
    numeric = rows[["segment_span_fraction_median", "encoded_event_prevalence"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite Track 2 values for {reference_key}; no events may be dropped")
    rows["span_pct"] = rows["segment_span_fraction_median"] * 100.0
    rows["carrier_pct"] = rows["encoded_event_prevalence"] * 100.0
    if not rows["span_pct"].gt(0).all():
        raise ValueError(f"Non-positive Track 2 spans for {reference_key}; no events may be dropped")

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
            rasterized=False,
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
                    "side": LABEL_SIDES.get(reference_key, {}).get(marker["label"], side),
                }
            )

    axis._cnv_leader_links = []
    for side in ("left", "right", "above", "below"):
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
            if side in {"above", "below"}:
                # Short vertical leaders distinguish nearby MYC/RAD21 events.
                # Anchor to actual marker coordinates, keeping data unchanged.
                position_px = axis.transData.transform((float(marker["x"]), float(marker["y"])))
                position_px[1] += (1 if side == "above" else -1)*9*axis.figure.dpi/72
                x_position, y_position = axis.transAxes.inverted().transform(position_px)
                alignment = "center"
            label_artist = axis.annotate(
                str(marker["label"]),
                xy=(float(marker["x"]), float(marker["y"])),
                xytext=(x_position, float(y_position)),
                textcoords=axis.transAxes,
                ha=alignment,
                va="center",
                fontsize=6.0,
                color=TEXT,
                annotation_clip=False,
                zorder=5,
            )
            axis._cnv_leader_links.append((label_artist, float(marker["x"]), float(marker["y"]), side))

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
    axis.tick_params(axis="both", labelsize=6.0, length=1.6, width=0.45)
    axis.grid(False)
    axis.set_xlabel("Segment span (% chromosome)", fontsize=6.1, labelpad=1)
    if not show_y_ticks:
        axis.set_yticklabels([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA8AE")
    axis.spines["bottom"].set_color("#9AA8AE")


WIDTH_MM = 183.0
HEIGHT_MM = 233.2
# Add 0.6 mm above mutation tables and 0.7 mm between case groups,
# preserving every plot's physical size and the existing top/bottom margins.
MUTATION_TOP_OFFSET_MM = 65.6
CASE_TOPS_MM = (231.2, 148.9, 74.6)
MUTATION_KEYS = ["root", "diagnosis", "target_subtype", "gene"]
MUTATION_GENES = {
    "C18_C20-CTX1": ("COAD", ("TP53", "BRAF")),
    "C18_C20-CTX5": ("COAD", ("TP53", "APC")),
    "C71-CTX2": ("GBM", ("TP53",)),
    "C50-CTX0": ("IDC", ("TP53", "GATA3", "ESR1", "PIK3CA")),
    "C50-CTX8": ("IDC", ("TP53",)),
    "C50-CTX3": ("IDC", ("TP53",)),
    "C50-CTX4": ("IDC", ("TP53", "GATA3", "ESR1")),
}


def text_mm(fig, x, y, value, **kwargs):
    return fig.text(x/WIDTH_MM, y/HEIGHT_MM, value, **kwargs)


def axes_mm(fig, x, y, width, height, label):
    return fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, width/WIDTH_MM, height/HEIGHT_MM], label=label)


def validate_against_public(references, annotations, signatures, architecture, release_dir):
    """Check the inherited CNV inputs against the published subtype/event set."""
    public = pd.read_csv(release_dir / "atlas/subtypes.tsv", sep="\t").set_index("reference_key")
    for row in references.itertuples(index=False):
        assert int(row.n_definition_members) == int(public.loc[row.reference_key, "n_definition_samples"])
        layer = {"subtype": "core", "conditional": "conditional"}[str(row.reference_level)]
        assert layer == str(public.loc[row.reference_key, "reference_layer"])
    published = pd.read_csv(release_dir / "atlas/signature_events.tsv", sep="\t")
    columns = ["reference_key", "track", "feature", "event", "amplitude"]
    selected = set(references.reference_key)
    # The public release renamed the inclusive legacy "low" channel to "all".
    normalized = signatures.assign(amplitude=signatures.amplitude.replace({"low": "all"}))
    a = set(normalized[columns].fillna("").itertuples(index=False, name=None))
    b = set(published.loc[published.reference_key.isin(selected), columns].fillna("").itertuples(index=False, name=None))
    assert a == b, "The CNV event set differs from the public release"
    normalized_architecture = architecture.assign(amplitude=architecture.amplitude.replace({"low": "all"}))
    check = normalized_architecture.merge(published, on=columns, validate="one_to_one", suffixes=("", "_public"))
    assert len(check) == len(architecture)
    assert np.allclose(check.encoded_event_prevalence, check.context_value, rtol=0, atol=1e-10)
    public_annotations = pd.read_csv(release_dir / "atlas/subtype_annotations.tsv", sep="\t")
    ann_keys = ["reference_key", "annotation_type", "annotation_label", "rank"]
    for ref in selected:
        left = annotations.loc[annotations.reference_key.eq(ref), ann_keys+["fraction"]]
        right = public_annotations.loc[public_annotations.reference_key.eq(ref), ann_keys+["fraction"]]
        pairs = left.merge(right, on=ann_keys, how="outer", suffixes=("_old", "_public"), validate="one_to_one")
        assert np.allclose(pairs.fraction_old, pairs.fraction_public, rtol=0, atol=1e-10)
    assert architecture.segment_span_fraction_median.gt(0).all()
    assert np.isfinite(architecture.encoded_event_prevalence).all()
    return {"subtypes": len(references), "cnv_events": len(signatures),
            "track2_events": len(architecture), "unplottable_track2_events": 0,
            "public_event_and_annotation_check": "PASS"}


def mutation_records(mutations, reference_key):
    if reference_key not in MUTATION_GENES:
        return []
    diagnosis, genes = MUTATION_GENES[reference_key]
    rows = []
    for gene in genes:
        match = mutations.loc[mutations.target_subtype.eq(reference_key)
                              & mutations.diagnosis.eq(diagnosis) & mutations.gene.eq(gene)]
        if len(match) != 1:
            raise ValueError(f"Expected one mutation annotation: {reference_key}/{diagnosis}/{gene}")
        rows.append(match.iloc[0])
    return rows


def draw_mutation_annotation(fig, reference_key, mutations, *, x, width, top):
    rows = mutation_records(mutations, reference_key)
    if not rows:
        return []
    diagnosis = rows[0].diagnosis
    y = top-MUTATION_TOP_OFFSET_MM
    rate_fraction = 0.72
    heading = f"{diagnosis} n={int(rows[0].target_n)}"
    text_mm(fig, x, y, heading, fontsize=6, color=MUTED, va="center")
    text_mm(fig, x+width*rate_fraction, y, "S/C %",
            fontsize=6, color=MUTED, ha="right", va="center")
    text_mm(fig, x+width, y, "Adj. OR", fontsize=6, color=MUTED, ha="right", va="center")
    records = []
    for index, row in enumerate(rows):
        yy = y-3.2*(index+1)
        color = TRACK2_GAIN if row.primary_or > 1 else TRACK2_LOSS
        text_mm(fig, x, yy, row.gene, fontsize=6.1, fontstyle="italic", va="center")
        text_mm(fig, x+width, yy, f"{row.primary_or:.2f}", fontsize=6.1,
                color=color, ha="right", va="center")
        target = 100*row.target_mutated/row.target_n
        control = 100*row.control_mutated/row.control_n
        rate_text = f"{target:.1f}/{control:.1f}"
        text_mm(fig, x+width*rate_fraction, yy, rate_text,
                fontsize=6.0, ha="right", va="center")
        records.append({"diagnosis": diagnosis, "subtype": reference_key, "gene": row.gene,
                        "primary_or": float(row.primary_or), "ci": [float(row.primary_ci_low), float(row.primary_ci_high)],
                        "target_n": int(row.target_n), "control_n": int(row.control_n),
                        "target_pct": target, "control_pct": control})
    return records


def draw_case(fig, case, references, annotations, signatures, architecture, mutations, *, top):
    contexts = case["contexts"]
    panel = case["panel"]
    text_mm(fig, 3, top-1.5, panel, fontsize=10, weight="bold", va="top")
    text_mm(fig, 12, top-1.5, case["title"], fontsize=9, weight="bold", va="top")
    start, stop, gap = 22.0, 180.0, 1.4
    cell_width = (stop-start-gap*(len(contexts)-1))/len(contexts)
    content_width = cell_width*CONTENT_WIDTH_FRACTION
    geometry = [(top-24, 12), (top-30.9, 3.8), (top-57, 23)]
    labels = ["Diagnostic\ncomposition", "Track 1", "Track 2\ncarrier\nfraction (%)"]
    axes, row_groups, column_groups, reports = [], [[], [], []], [], []
    for row, ((y, height), label) in enumerate(zip(geometry, labels)):
        text_mm(fig, 18, y+height/2, label, fontsize=6.3, ha="right", va="center")
    for column, ref in enumerate(contexts):
        x = start+column*(cell_width+gap)+(cell_width-content_width)/2
        reference = references.loc[references.reference_key.eq(ref)].iloc[0]
        short = ref.split("-", 1)[1]
        text_mm(fig, x+content_width/2, top-7.2, short, fontsize=6.8,
                weight="bold", ha="center", va="center")
        text_mm(fig, x+content_width/2, top-10.2,
                f"CNV n = {int(reference.n_definition_members):,}", fontsize=6.1,
                ha="center", va="center")
        cells = []
        for row, (y, height) in enumerate(geometry):
            name = f"{panel}_{short}_row{row}"
            ax = axes_mm(fig, x, y, content_width, height, name)
            cells.append(ax)
            row_groups[row].append(name)
        _draw_diagnostic_composition(cells[0], annotations, ref)
        _draw_track1(cells[1], signatures, ref, conditional=False)
        # Each subplot has its own range, as in the original case-study figure.
        # Retain its tick values so that annotations cannot suggest a common scale.
        _draw_track2(cells[2], architecture, ref, show_y_ticks=True)
        for ax in cells:
            ax.set_facecolor("white")
        reports.extend(draw_mutation_annotation(fig, ref, mutations, x=x, width=content_width,
                                                top=top))
        column_groups.append([a.get_label() for a in cells])
        axes.extend(cells)
    has_mutations = any(ref in MUTATION_GENES for ref in contexts)
    if has_mutations:
        text_mm(fig, 18, top-MUTATION_TOP_OFFSET_MM, "Selected\nassociations",
                fontsize=6.3, ha="right", va="center")
    return axes, row_groups, column_groups, reports


def place_gene_labels(fig, axes):
    """Keep gene labels clear of all measured event markers and other labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = fig.dpi/72*0.9

    def overlaps(a, b):
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    for ax in axes:
        links = getattr(ax, "_cnv_leader_links", [])
        if not links:
            continue
        bounds = ax.get_window_extent(renderer)
        point_boxes = []
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                continue
            points = collection.get_offset_transform().transform(offsets)
            sizes = collection.get_sizes()
            if not len(sizes):
                continue
            radius = (np.sqrt(float(max(sizes)))/2+0.5)*fig.dpi/72
            point_boxes.extend((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius) for p in points)
        options = {}
        for index, (artist, _, _, side) in enumerate(links):
            preferred = artist.get_position()
            candidates = []
            if side in {"above", "below"}:
                xx = [preferred[0]+offset for offset in (0, 0.02, -0.02, 0.04, -0.04)]
                direction = 1 if side == "above" else -1
                yy = [preferred[1]+direction*offset for offset in (0, 0.03, 0.06)]
            else:
                xx = [preferred[0], 0.025 if side == "left" else 0.975,
                      0.09 if side == "left" else 0.91]
                yy = sorted(set([preferred[1], *np.linspace(0.07, 0.93, 36)]))
            for x in xx:
                for y in yy:
                    artist.set_position((x, y))
                    artist.update_positions(renderer)
                    b = artist.get_window_extent(renderer)
                    box = (b.x0-pad, b.y0-pad, b.x1+pad, b.y1+pad)
                    if box[0] < bounds.x0+pad or box[2] > bounds.x1-pad or box[1] < bounds.y0+pad or box[3] > bounds.y1-pad:
                        continue
                    if not any(overlaps(box, p) for p in point_boxes):
                        cost = abs(x-preferred[0])*bounds.width + abs(y-preferred[1])*bounds.height
                        candidates.append((cost, (x, y), box))
            options[index] = sorted(candidates, key=lambda item: item[0])
            if not candidates:
                raise ValueError(f"No clear label position: {ax.get_label()} {artist.get_text()}")
        order = sorted(options, key=lambda index: len(options[index]))
        chosen = {}

        def assign(position):
            if position == len(order):
                return True
            index = order[position]
            for candidate in options[index]:
                if not any(overlaps(candidate[2], old[2]) for old in chosen.values()):
                    chosen[index] = candidate
                    if assign(position+1):
                        return True
                    del chosen[index]
            return False

        if not assign(0):
            raise ValueError(f"Cannot place gene labels without covering data: {ax.get_label()}")
        for index, (_, position, _) in chosen.items():
            links[index][0].set_position(position)


def route_gene_labels(fig, axes):
    """Route leaders around label boxes without moving data or hiding marks."""
    import heapq
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clearance = fig.dpi/72*0.9

    def enters_rectangle(a, b, rect):
        lo, hi = 0.0, 1.0
        delta = (b[0]-a[0], b[1]-a[1])
        for coordinate in (0, 1):
            lower, upper = rect[coordinate]+1e-5, rect[coordinate+2]-1e-5
            if abs(delta[coordinate]) < 1e-10:
                if not lower < a[coordinate] < upper:
                    return False
            else:
                ta, tb = (lower-a[coordinate])/delta[coordinate], (upper-a[coordinate])/delta[coordinate]
                lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
                if lo > hi:
                    return False
        return lo < hi and hi > 1e-7 and lo < 1-1e-7

    count = 0
    for ax in axes:
        links = getattr(ax, "_cnv_leader_links", [])
        if not links:
            continue
        boxes = [artist.get_window_extent(renderer).expanded(1, 1) for artist, *_ in links]
        obstacles = [(b.x0-clearance, b.y0-clearance, b.x1+clearance, b.y1+clearance) for b in boxes]
        bounds = ax.get_window_extent(renderer)
        corners = [corner for r in obstacles for corner in [(r[0], r[1]), (r[0], r[3]), (r[2], r[1]), (r[2], r[3])]
                   if bounds.x0 <= corner[0] <= bounds.x1 and bounds.y0 <= corner[1] <= bounds.y1]
        for index, (artist, x, y, side) in enumerate(links):
            box = obstacles[index]
            if side == "above":
                start = ((box[0]+box[2])/2, box[1]-0.05)
            elif side == "below":
                start = ((box[0]+box[2])/2, box[3]+0.05)
            else:
                start = (box[2]+0.05, (box[1]+box[3])/2) if side == "left" else (box[0]-0.05, (box[1]+box[3])/2)
            end = tuple(ax.transData.transform((x, y)))
            if any(r[0] < end[0] < r[2] and r[1] < end[1] < r[3] for r in obstacles):
                raise ValueError(f"Gene label covers a selected data point: {artist.get_text()}")
            nodes = [start, end, *corners]
            graph = [[] for _ in nodes]
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if not any(enters_rectangle(nodes[i], nodes[j], r) for r in obstacles):
                        distance = float(np.linalg.norm(np.subtract(nodes[i], nodes[j])))
                        graph[i].append((j, distance))
                        graph[j].append((i, distance))
            queue, distances, previous = [(0.0, 0)], {0: 0.0}, {}
            while queue:
                distance, current = heapq.heappop(queue)
                if current == 1:
                    break
                if distance > distances[current]+1e-9:
                    continue
                for neighbour, cost in graph[current]:
                    value = distance+cost
                    if value < distances.get(neighbour, float("inf"))-1e-9:
                        distances[neighbour], previous[neighbour] = value, current
                        heapq.heappush(queue, (value, neighbour))
            if 1 not in distances:
                raise ValueError(f"No collision-free leader for {artist.get_text()}")
            order, current = [1], 1
            while current != 0:
                current = previous[current]
                order.append(current)
            points = [nodes[i] for i in order[::-1]]
            last_delta = np.subtract(points[-1], points[-2])
            length = float(np.linalg.norm(last_delta))
            if length > clearance*3:
                points[-1] = tuple(np.subtract(points[-1], last_delta/length*clearance*2.5))
            coords = fig.transFigure.inverted().transform(points)
            fig.add_artist(Line2D(coords[:, 0], coords[:, 1], transform=fig.transFigure,
                                  color="#859198", lw=0.4, zorder=3))
            count += 1
    return count


def require_matplotlib_panel_alignment(fig, axes, row_groups, column_groups):
    fig.canvas.draw()
    by_id = {axis.get_label(): axis for axis in axes}
    width, height = fig.get_size_inches()*72
    for groups, indices in [(row_groups, [1, 3]), (column_groups, [0, 2])]:
        for group in groups:
            boxes = [by_id[name].get_position() for name in group]
            values = [[b.x0*width, b.y0*height, b.x1*width, b.y1*height] for b in boxes]
            if any(max(v[k] for v in values)-min(v[k] for v in values) > 1.5 for k in indices):
                raise ValueError(f"Misaligned case-study panels: {group}")
def main():
    parser = argparse.ArgumentParser(description="Render original-structure CNV case studies with mutation annotations.")
    parser.add_argument("--atlas-root", type=Path, required=True,
                        help="Atlas workspace containing release_candidate CNV tables and segment architecture")
    parser.add_argument("--release-dir", type=Path, required=True,
                        help="Public CNVSIM release containing atlas tables and mutation-context validation")
    parser.add_argument("--out-dir", type=Path,
                        help="Default: ATLAS_ROOT/figures/figure5")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing Figure5.pdf and Figure5.png")
    args = parser.parse_args()
    atlas = args.atlas_root.expanduser().resolve()
    out = (args.out_dir.expanduser().resolve() if args.out_dir
           else atlas / "figures/figure5")
    existing = [out/name for name in ["Figure5.pdf", "Figure5.png"] if (out/name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Figure 5 outputs already exist; pass --overwrite: "
                                 + ", ".join(str(path) for path in existing))
    cnv = atlas / "release_candidate"
    public = args.release_dir.expanduser().resolve()
    references, annotations, signatures, architecture = _load_inputs(cnv)
    validate_against_public(references, annotations, signatures, architecture, public)
    mutation_path = public / "validation/mutation_context/reader/focus_45_overview.tsv"
    mutations = pd.read_csv(mutation_path, sep="\t")
    if mutations.duplicated(MUTATION_KEYS).any():
        raise ValueError("Duplicate mutation comparison keys")
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), facecolor=WHITE, dpi=300)
    axes, rows, columns, mutation_annotations = [], [], [], []
    for case, top in zip(CASE_STUDIES, CASE_TOPS_MM, strict=True):
        aa, rr, cc, mm = draw_case(fig, case, references, annotations, signatures, architecture, mutations, top=top)
        axes.extend(aa)
        rows.extend(rr)
        columns.extend(cc)
        mutation_annotations.extend(mm)
    handles = [Patch(facecolor=TRACK1_GAIN, label="Track 1 gain"),
               Patch(facecolor=TRACK1_LOSS, label="Track 1 loss"),
               Patch(facecolor=TRACK1_DUAL, label="Track 1 gain/loss"),
               Line2D([], [], marker="o", linestyle="", color=TRACK2_GAIN, ms=3.5, label="Track 2 gain"),
               Line2D([], [], marker="o", linestyle="", color=TRACK2_LOSS, ms=3.5, label="Track 2 loss")]
    fig.legend(handles=handles, ncol=5, loc="center", bbox_to_anchor=(0.54, 1.8/HEIGHT_MM),
               fontsize=6, handlelength=1, handletextpad=0.4, columnspacing=1.1, frameon=False)
    place_gene_labels(fig, axes)
    route_gene_labels(fig, axes)
    require_matplotlib_panel_alignment(fig, axes, rows, columns)
    assert len(mutation_annotations) == 14
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"Figure5.pdf", metadata={"Title": "Representative CNV architectures and accompanying mutation contexts"})
    fig.savefig(out/"Figure5.png", dpi=400)
    plt.close(fig)
    print("Figure 5: 12 CNV subtypes; 14 mutation annotations; PDF and PNG written.")


if __name__ == "__main__":
    main()
