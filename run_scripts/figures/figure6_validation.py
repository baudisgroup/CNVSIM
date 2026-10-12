#!/usr/bin/env python3
from __future__ import annotations

"""Render the publication-facing validation panels for Figure 6."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TEXT = "#111111"
MUTED = "#68747A"
GRID = "#E3E8EA"
BLUE = "#2E5EAA"
ORANGE = "#D9822B"
WHITE = "#FFFFFF"
PAPER_TEXT_SIZE = 7.0
PAPER_SMALL_TEXT_SIZE = 6.2

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

GENERATED_OUTPUT_NAMES = {
    "A_technical_composition.png",
    "B_signature_recovery.png",
}
STALE_OUTPUT_NAMES = {
    "C_diagnostic_review.png",
    "C_fixed_reference_projection.png",
    "D_diagnostic_review.png",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": PAPER_TEXT_SIZE,
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


def _read_parquet(
    path: Path,
    required: set[str],
    name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")
    return frame


def _read_table(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return _read_parquet(path, required, name)
    return _read_tsv(path, required, name)


def _style_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#91A0A6")
    axis.spines["bottom"].set_color("#91A0A6")
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.75)
    axis.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _reference_number(reference_key: str) -> int:
    key = str(reference_key)
    for marker in ("CTX", "COND"):
        try:
            return int(key.rsplit(marker, 1)[1])
        except (IndexError, ValueError):
            continue
    return 10**6


def _ordered_references(references: pd.DataFrame) -> pd.DataFrame:
    released = references.loc[
        references["reference_level"].astype(str).isin(
            ["subtype", "conditional"]
        )
    ].copy()
    released["root"] = released["root"].astype(str)
    released["reference_number"] = released["reference_key"].map(
        _reference_number
    )
    released["level_order"] = released["reference_level"].map(
        {"subtype": 0, "conditional": 1}
    )
    root_order = list(ROOT_LABELS)
    observed = released["root"].drop_duplicates().tolist()
    root_order.extend(root for root in observed if root not in root_order)
    released["root_order"] = pd.Categorical(
        released["root"],
        categories=root_order,
        ordered=True,
    )
    return released.sort_values(
        ["root_order", "level_order", "reference_number", "reference_key"]
    ).reset_index(drop=True)


def _build_definition_technical_composition(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    definitions = assignments.loc[
        assignments["pipeline_status"].astype(str).isin(
            ["anchor_member", "conditional_definition_member"]
        ),
        [
            "analysis_id",
            "root",
            "pipeline_status",
            "assigned_reference_key",
        ],
    ].copy()
    if definitions.empty:
        raise ValueError("Release assignments contain no definition samples")
    if definitions["analysis_id"].duplicated().any():
        raise ValueError("Definition assignments contain duplicate analysis IDs")
    if definitions["assigned_reference_key"].isna().any():
        raise ValueError("A definition sample is missing its reference subtype")

    metadata_columns = [
        "analysis_id",
        "platform_id",
        "platform_label",
        "source_group_refined",
    ]
    if metadata["analysis_id"].duplicated().any():
        raise ValueError("Analysis metadata contain duplicate analysis IDs")
    composition = definitions.merge(
        metadata[metadata_columns],
        on="analysis_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_metadata = composition["_merge"].ne("both")
    if missing_metadata.any():
        example = composition.loc[missing_metadata, "analysis_id"].iloc[0]
        raise ValueError(
            "Definition samples are missing analysis metadata; "
            f"first missing analysis: {example}"
        )
    composition = composition.drop(columns="_merge").rename(
        columns={"assigned_reference_key": "full_reference_key"}
    )
    return composition


def draw_technical_composition(
    references: pd.DataFrame,
    composition: pd.DataFrame,
    out_path: Path,
) -> None:
    released = _ordered_references(references)
    keys = released["reference_key"].astype(str).tolist()
    composition = composition.loc[
        composition["full_reference_key"].astype(str).isin(keys)
    ].copy()
    observed_keys = set(composition["full_reference_key"].astype(str))
    missing_keys = sorted(set(keys) - observed_keys)
    if missing_keys:
        raise ValueError(
            "Released subtypes are missing definition samples; "
            f"first missing subtype: {missing_keys[0]}"
        )
    platform_counts = composition["platform_id"].astype(str).value_counts()
    major_platforms = platform_counts.head(9).index.tolist()
    composition["platform_plot"] = composition["platform_id"].astype(str).where(
        composition["platform_id"].astype(str).isin(major_platforms),
        "Other platforms",
    )
    platform_order = major_platforms + ["Other platforms"]

    platform_labels = (
        composition[["platform_id", "platform_label"]]
        .drop_duplicates("platform_id")
        .set_index("platform_id")["platform_label"]
        .astype(str)
        .to_dict()
    )
    platform_short_labels = {
        "NCIT:C158253": "Targeted NGS",
        "pgx:gpl-MSK-IMPACT": "MSK-IMPACT",
        "geo:GPL6801": "SNP 6.0",
        "EFO:0010937": "CGH",
        "OBI:0000626": "Unspecified DNA-seq",
        "geo:GPL3718": "250K Nsp",
        "geo:GPL3720": "250K Sty",
        "geo:GPL2005": "50K Xba",
        "geo:GPL2004": "50K Hind",
        "Other platforms": "Other platforms",
    }
    platform_display = [
        platform_short_labels.get(
            platform,
            platform_labels.get(platform, platform),
        )
        for platform in platform_order
    ]

    composition["source_plot"] = (
        composition["source_group_refined"]
        .astype(str)
        .replace({"ArrayExpress": "Other"})
    )
    source_order = ["GENIE", "GEO", "TCGA", "cBioPortal", "Other"]

    def fraction_matrix(column: str, categories: list[str]) -> np.ndarray:
        counts = pd.crosstab(
            composition["full_reference_key"].astype(str),
            composition[column].astype(str),
        )
        counts = counts.reindex(index=keys, columns=categories, fill_value=0)
        denominators = counts.sum(axis=1).to_numpy(dtype=float)
        values = counts.to_numpy(dtype=float)
        return np.divide(
            values,
            denominators[:, None],
            out=np.zeros_like(values, dtype=float),
            where=denominators[:, None] > 0,
        )

    platform_matrix = fraction_matrix("platform_plot", platform_order)
    source_matrix = fraction_matrix("source_plot", source_order)

    cmap = LinearSegmentedColormap.from_list(
        "technical_fraction",
        [
            "#FFFFFF",
            "#EFF3FF",
            "#BDD7E7",
            "#6BAED6",
            "#3182BD",
            "#08519C",
            "#08306B",
        ],
    )
    norm = PowerNorm(gamma=0.50, vmin=0.0, vmax=0.8)
    fig = plt.figure(figsize=(7.1, 5.15))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[10, 5, 0.42],
        wspace=0.08,
    )
    platform_axis = fig.add_subplot(grid[0, 0])
    source_axis = fig.add_subplot(grid[0, 1], sharey=platform_axis)
    color_axis = fig.add_subplot(grid[0, 2])

    image = platform_axis.imshow(
        platform_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    source_axis.imshow(
        source_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    platform_axis.set_title(
        "CNV assay platform",
        fontsize=PAPER_TEXT_SIZE,
        fontweight="bold",
        pad=8,
    )
    source_axis.set_title(
        "Data source",
        fontsize=PAPER_TEXT_SIZE,
        fontweight="bold",
        pad=8,
    )
    platform_axis.set_xticks(np.arange(len(platform_order)))
    platform_axis.set_xticklabels(
        platform_display,
        rotation=53,
        ha="right",
        rotation_mode="anchor",
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    source_axis.set_xticks(np.arange(len(source_order)))
    source_axis.set_xticklabels(
        source_order,
        rotation=53,
        ha="right",
        rotation_mode="anchor",
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    for axis in (platform_axis, source_axis):
        axis.tick_params(axis="x", length=0, pad=4)
        axis.tick_params(axis="y", length=0)
        for spine in axis.spines.values():
            spine.set_color("#91A0A6")
            spine.set_linewidth(0.8)

    roots = released["root"].astype(str).to_numpy()
    starts = np.r_[0, np.flatnonzero(roots[1:] != roots[:-1]) + 1]
    ends = np.r_[starts[1:], len(released)]
    platform_axis.set_yticks(
        [(start + end - 1) / 2 for start, end in zip(starts, ends)]
    )
    platform_axis.set_yticklabels(
        [ROOT_LABELS.get(roots[start], roots[start]) for start in starts],
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    platform_axis.tick_params(axis="y", pad=4)
    source_axis.tick_params(axis="y", labelleft=False)
    for start, end in zip(starts, ends):
        if start > 0:
            for axis in (platform_axis, source_axis):
                axis.axhline(
                    start - 0.5,
                    color=WHITE,
                    linewidth=1.35,
                    zorder=3,
                )
    colorbar = fig.colorbar(
        image,
        cax=color_axis,
        orientation="vertical",
    )
    colorbar.set_label("Within-subtype analysis fraction", labelpad=5)
    colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    colorbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "≥0.8"])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2)
    _save(fig, out_path)


def draw_signature_recovery(
    root_recovery: pd.DataFrame,
    context_recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    root_columns = {
        "root",
        "membership_recall_sample_weighted_median",
        "track1_event_recall_event_weighted_median",
        "track1_effect_cosine_context_median_median",
        "track2_event_recall_event_weighted_median",
        "track2_effect_cosine_context_median_median",
    }
    missing = sorted(root_columns - set(root_recovery.columns))
    if missing:
        raise KeyError(f"root recovery is missing columns: {missing}")
    context_columns = {
        "root",
        "reference_key",
        "context_detected_minimum",
    }
    missing = sorted(context_columns - set(context_recovery.columns))
    if missing:
        raise KeyError(f"context recovery is missing columns: {missing}")

    all_fold_detection = (
        context_recovery.assign(
            recovered_all_folds=np.isclose(
                pd.to_numeric(
                    context_recovery["context_detected_minimum"],
                    errors="raise",
                ),
                1.0,
            )
        )
        .groupby("root", observed=True)
        .agg(
            n_contexts_evaluated=("reference_key", "size"),
            n_contexts_recovered_all_folds=("recovered_all_folds", "sum"),
        )
    )
    all_fold_detection["context_all_fold_fraction"] = (
        all_fold_detection["n_contexts_recovered_all_folds"]
        / all_fold_detection["n_contexts_evaluated"]
    )

    roots = [root for root in ROOT_LABELS if root in set(root_recovery["root"])]
    frame = (
        root_recovery.set_index("root")
        .join(all_fold_detection, how="left")
        .reindex(roots)
    )
    panels = [
        (
            "Subtype",
            "context_all_fold_fraction",
            "membership_recall_sample_weighted_median",
        ),
        (
            "Track 1",
            "track1_event_recall_event_weighted_median",
            "track1_effect_cosine_context_median_median",
        ),
        (
            "Track 2",
            "track2_event_recall_event_weighted_median",
            "track2_effect_cosine_context_median_median",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 4.15),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    y = np.arange(len(roots))[::-1]
    for panel_index, (
        axis,
        (title, discrete_column, continuous_column),
    ) in enumerate(zip(axes, panels)):
        discrete = pd.to_numeric(frame[discrete_column], errors="coerce").to_numpy()
        continuous = pd.to_numeric(
            frame[continuous_column],
            errors="coerce",
        ).to_numpy()
        available = np.isfinite(discrete) & np.isfinite(continuous)

        for index in np.flatnonzero(available):
            axis.plot(
                [discrete[index], continuous[index]],
                [y[index], y[index]],
                color="#BBC5C9",
                linewidth=1.15,
                zorder=1,
            )
        axis.scatter(
            discrete[available],
            y[available],
            s=18,
            marker="o",
            facecolor=BLUE,
            edgecolor=WHITE,
            linewidth=0.45,
            zorder=3,
        )
        axis.scatter(
            continuous[available],
            y[available],
            s=36,
            marker="o",
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=1.25,
            zorder=4,
        )

        unavailable = ~available
        for index in np.flatnonzero(unavailable):
            axis.text(
                0.02,
                y[index],
                "—",
                va="center",
                ha="left",
                fontsize=PAPER_TEXT_SIZE,
                color=MUTED,
            )
        if panel_index == 0:
            incomplete = (
                available
                & np.isfinite(
                    frame["n_contexts_evaluated"].to_numpy(dtype=float)
                )
                & (discrete < 1.0 - 1e-12)
            )
            for index in np.flatnonzero(incomplete):
                axis.text(
                    max(0.03, discrete[index] - 0.04),
                    y[index],
                    (
                        f"{int(frame.iloc[index]['n_contexts_recovered_all_folds'])}/"
                        f"{int(frame.iloc[index]['n_contexts_evaluated'])}"
                    ),
                    ha="right",
                    va="center",
                    fontsize=5.4,
                    color=BLUE,
                )

        axis.set_title(
            title,
            fontsize=8.0,
            fontweight="bold",
            pad=8,
        )
        axis.set_xlim(0.0, 1.03)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xticklabels(["0", "50", "100"])
        axis.tick_params(axis="y", length=0)
        _style_axis(axis)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [ROOT_LABELS[root] for root in roots],
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    axes[0].tick_params(axis="y", pad=4)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=BLUE,
                markeredgecolor=WHITE,
                markeredgewidth=0.45,
                markersize=5,
                label="Subtype / signature recovery",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=ORANGE,
                markeredgewidth=1.25,
                markersize=6,
                label="Analysis recovery / signature similarity",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        ncol=2,
        fontsize=PAPER_SMALL_TEXT_SIZE,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.supxlabel(
        "Root-level recovery or similarity (%)",
        fontsize=PAPER_TEXT_SIZE,
        y=0.01,
    )
    fig.subplots_adjust(top=0.86, bottom=0.11)
    _save(fig, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final validation panels for Figure 6."
    )
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--signature-recovery-dir", type=Path)
    parser.add_argument("--analysis-metadata", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    recovery_dir = (
        args.signature_recovery_dir.expanduser().resolve()
        if args.signature_recovery_dir is not None
        else release_dir / "validation" / "signature_recovery_v1"
    )
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and not args.overwrite:
        occupied = [
            out_dir / name
            for name in GENERATED_OUTPUT_NAMES
            if (out_dir / name).exists()
        ]
        if occupied:
            raise FileExistsError(
                "Figure 6 output exists; pass --overwrite: "
                f"{occupied[0]}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in GENERATED_OUTPUT_NAMES | STALE_OUTPUT_NAMES:
            path = out_dir / name
            if path.exists():
                path.unlink()

    references = _read_tsv(
        release_dir / "tables" / "reference_contexts.tsv",
        {"reference_key", "root", "reference_level"},
        "reference contexts",
    )
    assignments = _read_parquet(
        release_dir / "tables" / "analysis_assignments.parquet",
        {
            "analysis_id",
            "root",
            "pipeline_status",
            "assigned_reference_key",
        },
        "release assignments",
    )
    metadata = _read_table(
        args.analysis_metadata.expanduser().resolve(),
        {
            "analysis_id",
            "platform_id",
            "platform_label",
            "source_group_refined",
        },
        "analysis metadata",
    )
    composition = _build_definition_technical_composition(assignments, metadata)
    root_recovery = _read_tsv(
        recovery_dir / "root_signature_recovery.tsv",
        {
            "root",
            "membership_recall_sample_weighted_median",
            "track1_event_recall_event_weighted_median",
            "track1_effect_cosine_context_median_median",
            "track2_event_recall_event_weighted_median",
            "track2_effect_cosine_context_median_median",
        },
        "root-level signature recovery",
    )
    context_recovery = _read_tsv(
        recovery_dir / "context_signature_recovery.tsv",
        {"root", "reference_key", "context_detected_minimum"},
        "context-level signature recovery",
    )
    outputs = {
        "A": out_dir / "A_technical_composition.png",
        "B": out_dir / "B_signature_recovery.png",
    }
    draw_technical_composition(references, composition, outputs["A"])
    draw_signature_recovery(
        root_recovery,
        context_recovery,
        outputs["B"],
    )
    print("[CNVSIM] Figure 6 validation panels complete")
    print(
        "  released definition composition: "
        f"{len(composition):,} analyses across "
        f"{composition['full_reference_key'].nunique()} subtypes"
    )
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
