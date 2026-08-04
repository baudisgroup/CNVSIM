#!/usr/bin/env python3
from __future__ import annotations

"""Render the real-data panels used to assemble Figure 2B."""

import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import figure2_workflow_panels as figure2a


TEXT = "#253238"
MUTED = "#6C777C"
GRID = "#DDE4E6"
TEAL = "#087F83"
GREEN = "#4C9A68"
BLUE = "#477DB3"
ORANGE = "#E8753D"
ORANGE_DARK = "#B9532F"
GREY = "#8FA0A5"
LIGHT_GREY = "#D8DFE1"
WHITE = "#FFFFFF"

CORE_COLORS = [TEAL, GREEN, BLUE, "#8064A2", "#C29B32"]
CONDITIONAL_COLORS = [ORANGE, ORANGE_DARK, "#F2A377", "#9F6A49"]


def _read_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {
        "analysis_id",
        "root",
        "pipeline_status",
        "assigned_reference_key",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Release assignments are missing columns: {missing}")
    frame = frame.copy()
    frame["analysis_id"] = frame["analysis_id"].astype(str)
    frame["root"] = frame["root"].astype(str)
    frame["pipeline_status"] = frame["pipeline_status"].astype(str)
    frame["assigned_reference_key"] = frame["assigned_reference_key"].astype(
        "string"
    )
    if frame["analysis_id"].duplicated().any():
        raise ValueError("Release assignments contain duplicate analysis IDs")
    return frame


def _reference_number(reference_key: str) -> int:
    match = re.search(r"(?:CTX|COND)(\d+)$", str(reference_key))
    return int(match.group(1)) if match else 10**9


def _reference_label(reference_key: str) -> str:
    match = re.search(r"((?:CTX|COND)\d+)$", str(reference_key))
    return match.group(1) if match else str(reference_key)


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed)


def _bounds(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    low, high = np.quantile(numeric, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(numeric.median())
        return center - 1.0, center + 1.0
    pad = 0.05 * (high - low)
    return float(low - pad), float(high + pad)


def _style_embedding_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GREY)
    axis.spines["bottom"].set_color(GREY)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axis.set_xlabel("UMAP1", color=MUTED, labelpad=3)
    axis.set_ylabel("UMAP2", color=MUTED, labelpad=3)


def _plot_layer(
    axis: plt.Axes,
    *,
    embedding: pd.DataFrame,
    assignments: pd.DataFrame,
    member_status: str,
    colors: list[str],
    maximum_background_points: int,
    random_state: int,
) -> list[tuple[str, str]]:
    background = _sample(
        embedding,
        maximum_background_points,
        random_state,
    )
    axis.scatter(
        background["x"],
        background["y"],
        s=2.1,
        c=LIGHT_GREY,
        alpha=0.26,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    members = assignments[assignments["pipeline_status"].eq(member_status)].copy()
    members = members[members["analysis_id"].isin(embedding.index)]
    if members.empty:
        raise ValueError(f"No {member_status} samples overlap the display embedding")
    references = sorted(
        members["assigned_reference_key"].dropna().astype(str).unique(),
        key=_reference_number,
    )
    legend_rows: list[tuple[str, str]] = []
    for index, reference_key in enumerate(references):
        ids = members.loc[
            members["assigned_reference_key"].astype(str).eq(reference_key),
            "analysis_id",
        ]
        points = embedding.loc[embedding.index.intersection(ids)]
        if points.empty:
            continue
        color = colors[index % len(colors)]
        marker = "D" if member_status == "conditional_definition_member" else "o"
        axis.scatter(
            points["x"],
            points["y"],
            s=4.0 if marker == "D" else 3.2,
            c=color,
            marker=marker,
            alpha=0.68,
            edgecolors=WHITE if marker == "D" else "none",
            linewidths=0.18 if marker == "D" else 0,
            rasterized=True,
            zorder=3,
        )
        center_x = float(points["x"].median())
        center_y = float(points["y"].median())
        axis.text(
            center_x,
            center_y,
            _reference_label(reference_key),
            ha="center",
            va="center",
            color=color,
            fontsize=8.2,
            fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground=WHITE)],
            zorder=5,
        )
        legend_rows.append((_reference_label(reference_key), color))
    return legend_rows


def _draw_reference_layer(
    embedding: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    root: str,
    member_status: str,
    colors: list[str],
    maximum_background_points: int,
    random_state: int,
) -> plt.Figure:
    root_assignments = assignments[assignments["root"].eq(root)].copy()
    members = root_assignments[
        root_assignments["pipeline_status"].eq(member_status)
    ]
    if members.empty:
        raise ValueError(f"[{root}] release has no {member_status} samples")

    figure, axis = plt.subplots(figsize=(3.25, 3.0))
    legend_rows = _plot_layer(
        axis,
        embedding=embedding,
        assignments=root_assignments,
        member_status=member_status,
        colors=colors,
        maximum_background_points=maximum_background_points,
        random_state=random_state,
    )

    axis.set_xlim(*_bounds(embedding["x"]))
    axis.set_ylim(*_bounds(embedding["y"]))
    _style_embedding_axis(axis)
    marker = "D" if member_status == "conditional_definition_member" else "o"
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor=WHITE if marker == "D" else "none",
            markeredgewidth=0.3 if marker == "D" else 0,
            label=label,
        )
        for label, color in legend_rows
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(handles), 3),
        frameon=False,
        fontsize=7.2,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.20)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render separate core and conditional reference UMAPs for Figure 2B."
        )
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument("--analysis-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--root", default="C16")
    parser.add_argument("--display-space", choices=("umap", "pc"), default="umap")
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=41)
    parser.add_argument("--maximum-points", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    figure2a._configure()
    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    analysis_dir = (
        Path(args.analysis_dir).expanduser().resolve()
        if args.analysis_dir
        else atlas_dir / "work" / "analysis_v1" / "analysis"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = str(args.root)

    selected_parameters_path = (
        analysis_dir / "parameters" / "selected_parameters.tsv"
    )
    assignments_path = (
        atlas_dir
        / "release_candidate"
        / "tables"
        / "analysis_assignments.parquet"
    )
    if not selected_parameters_path.exists():
        raise FileNotFoundError(selected_parameters_path)
    selected_parameters = pd.read_csv(selected_parameters_path, sep="\t")
    assignments = _read_assignments(assignments_path)

    embedding, display_metadata = figure2a._display_embedding(
        analysis_dir,
        selected_parameters,
        root,
        display_space=str(args.display_space),
        umap_min_dist=float(args.umap_min_dist),
        random_state=int(args.random_state),
    )

    outputs = {
        "core_reference": out_dir / "II_core_reference_umap.png",
        "conditional_reference": (
            out_dir / "II_conditional_reference_umap.png"
        ),
        "manifest": out_dir / "figure2B_panel_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Figure 2B outputs already exist; use --overwrite:\n  "
            + "\n  ".join(map(str, existing))
        )

    core_reference = _draw_reference_layer(
        embedding,
        assignments,
        root=root,
        member_status="anchor_member",
        colors=CORE_COLORS,
        maximum_background_points=int(args.maximum_points),
        random_state=int(args.random_state),
    )
    core_reference.savefig(
        outputs["core_reference"],
        dpi=int(args.dpi),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor=WHITE,
    )
    plt.close(core_reference)

    conditional_reference = _draw_reference_layer(
        embedding,
        assignments,
        root=root,
        member_status="conditional_definition_member",
        colors=CONDITIONAL_COLORS,
        maximum_background_points=int(args.maximum_points),
        random_state=int(args.random_state),
    )
    conditional_reference.savefig(
        outputs["conditional_reference"],
        dpi=int(args.dpi),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor=WHITE,
    )
    plt.close(conditional_reference)

    manifest = {
        "schema_version": "cnvsim_figure2B_reference_layers_v1",
        "atlas_dir": str(atlas_dir),
        "analysis_dir": str(analysis_dir),
        "root": root,
        "display": display_metadata,
        "inputs": {
            "selected_parameters": str(selected_parameters_path),
            "analysis_assignments": str(assignments_path),
        },
        "outputs": {
            key: path.name for key, path in outputs.items() if key != "manifest"
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

    print("[CNVSIM] Figure 2B projection panels complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
