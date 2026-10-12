#!/usr/bin/env python3
from __future__ import annotations

"""Render the data panels used to assemble Figure 2A in a graphics editor."""

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
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


ID = "analysis_id"
PLATFORM = "platform_family_refined"
TEXT = "#253238"
MUTED = "#6C777C"
GRID = "#DDE4E6"
TEAL = "#087F83"
ORANGE = "#E8753D"
BLUE = "#3A6FB0"
RED = "#C94C4C"
GREEN = "#3E8E65"
GREY = "#A9B2B6"
LIGHT = "#F2F5F5"

PLATFORM_ORDER = ["Arrays_HighRes", "Arrays_LowRes", "NGS_Based", "Other"]
PLATFORM_COLORS = {
    "Arrays_HighRes": "#007C83",
    "Arrays_LowRes": "#E8753D",
    "NGS_Based": "#3B6FB6",
    "Other": GREY,
}
PLATFORM_LABELS = {
    "Arrays_HighRes": "High-resolution arrays",
    "Arrays_LowRes": "Low-resolution arrays/CGH",
    "NGS_Based": "NGS-based assays",
    "Other": "Other/unknown",
}
COMMUNITY_COLORS = [
    "#007C83",
    "#E8753D",
    "#5B9C62",
    "#8064A2",
    "#D45478",
    "#C29B32",
    "#4C89B6",
    "#8B8178",
    "#3FA7A1",
    "#B66D3B",
    "#697BA7",
    "#A15D86",
    "#7A9E45",
    "#D08539",
    "#537F8E",
    "#B66A6A",
    "#6A8FBB",
    "#9A7A45",
    "#4C9C83",
    "#8C6CB0",
]
DISCOVERY_PANEL_SIZE = (3.45, 3.10)
COMMUNITY_SIZE_PANEL_SIZE = (3.45, 1.45)
REFINEMENT_PANEL_SIZE = (3.10, 3.10)


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "savefig.facecolor": "white",
        }
    )


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def _find(analysis_dir: Path, filename: str, label: str) -> Path:
    matches = sorted(analysis_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot locate {label}: {filename}")
    preferred = [
        path
        for path in matches
        if not {"validation", "train_analysis"}.intersection(path.parts)
    ]
    candidates = preferred or matches
    if len(candidates) > 1:
        exact = [
            path
            for path in candidates
            if "refined_all" in path.parts or "context_refinement" in path.parts
        ]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(
            f"Ambiguous {label}; pass a narrower --analysis-dir:\n  "
            + "\n  ".join(map(str, candidates))
        )
    return candidates[0]


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})
    )


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if ID in result.columns:
        result[ID] = result[ID].astype(str)
        result = result.set_index(ID)
    result.index = result.index.astype(str)
    result.index.name = ID
    if result.index.duplicated().any():
        raise ValueError("Coordinate table contains duplicate analysis IDs")
    return result


def _dimensions(frame: pd.DataFrame) -> list[str]:
    columns = [
        str(column)
        for column in frame.columns
        if (str(column).startswith("D") and str(column)[1:].isdigit())
        or (str(column).startswith("PC") and str(column)[2:].isdigit())
    ]
    return sorted(
        columns,
        key=lambda value: int(
            "".join(character for character in value if character.isdigit())
        ),
    )


def _platform(value: object) -> str:
    aliases = {
        "Panels_NGS": "NGS_Based",
        "NGS": "NGS_Based",
        "NGS_panel": "NGS_Based",
        "NGS_panels": "NGS_Based",
    }
    normalized = aliases.get(str(value).strip(), str(value).strip())
    return normalized if normalized in PLATFORM_ORDER[:-1] else "Other"


def _colour(context: int) -> str:
    return COMMUNITY_COLORS[int(context) % len(COMMUNITY_COLORS)]


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed)


def _balanced_sample(
    frame: pd.DataFrame, column: str, maximum: int, seed: int
) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    groups = list(frame.groupby(column, dropna=False))
    per_group = max(50, maximum // max(len(groups), 1))
    sampled = [
        _sample(group, per_group, seed + index)
        for index, (_value, group) in enumerate(groups)
    ]
    result = pd.concat(sampled)
    return _sample(result, maximum, seed) if len(result) > maximum else result


def _clean(axis: plt.Axes, *, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#879196")
    axis.spines["bottom"].set_color("#879196")
    axis.tick_params(length=2.5, width=0.55, colors=MUTED, pad=2)
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color=GRID,
            linewidth=0.55,
            zorder=0,
        )


def _robust_limits(values: pd.Series, lower: float = 0.005) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return (-1.0, 1.0)
    low, high = clean.quantile([lower, 1.0 - lower]).to_numpy(float)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        center = float(clean.median())
        return (center - 1.0, center + 1.0)
    pad = 0.06 * (high - low)
    return (float(low - pad), float(high + pad))


def _save(
    figure: plt.Figure,
    out_dir: Path,
    stem: str,
    dpi: int,
    *,
    fixed_canvas: bool = False,
) -> Path:
    path = out_dir / f"{stem}.png"
    save_options: dict[str, object] = {"dpi": dpi}
    if not fixed_canvas:
        save_options.update({"bbox_inches": "tight", "pad_inches": 0.03})
    figure.savefig(path, **save_options)
    plt.close(figure)
    return path


def _root_paths(analysis_dir: Path, root: str) -> dict[str, Path]:
    root_dir = analysis_dir / "ir" / "perroot" / root
    return {
        "coords": root_dir / "coords.parquet",
        "metadata": root_dir / "meta.parquet",
        "partition": _find(
            analysis_dir,
            f"{root}_provisional_partition.parquet",
            f"{root} provisional partition",
        ),
    }


def _comparison_coordinates(analysis_dir: Path, root: str) -> pd.DataFrame:
    paths = _root_paths(analysis_dir, root)
    coords = _indexed(_read(paths["coords"]))
    dimensions = _dimensions(coords)
    if len(dimensions) < 2:
        raise ValueError(f"[{root}] fewer than two comparison-space coordinates")
    return coords[dimensions].apply(pd.to_numeric, errors="coerce")


def _display_embedding(
    analysis_dir: Path,
    selected_parameters: pd.DataFrame,
    root: str,
    *,
    display_space: str,
    umap_min_dist: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    coords = _comparison_coordinates(analysis_dir, root)
    selected_rows = selected_parameters[
        selected_parameters["root"].astype(str).eq(root)
    ]
    if selected_rows.empty:
        raise ValueError(f"[{root}] selected parameters are absent")
    selected = selected_rows.iloc[0]
    requested_pcs = pd.to_numeric(
        pd.Series([selected.get("n_pcs", len(coords.columns))]),
        errors="coerce",
    ).iloc[0]
    n_pcs = (
        min(max(int(requested_pcs), 2), len(coords.columns))
        if np.isfinite(requested_pcs)
        else len(coords.columns)
    )
    used = coords.iloc[:, :n_pcs].copy()
    if used.isna().any().any():
        raise ValueError(f"[{root}] comparison-space coordinates contain NaNs")
    if display_space == "pc":
        frame = used.iloc[:, :2].copy()
        frame.columns = ["x", "y"]
        return frame, {
            "method": "pc",
            "dimensions_used": 2,
            "metric": "euclidean",
        }
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP display space requires umap-learn; use --display-space pc "
            "or install the preprocessing dependencies."
        ) from exc
    requested_neighbors = pd.to_numeric(
        pd.Series([selected.get("K", 30)]), errors="coerce"
    ).iloc[0]
    neighbors = int(requested_neighbors) if np.isfinite(requested_neighbors) else 30
    neighbors = min(max(neighbors, 2), max(len(used) - 1, 2))
    metric = str(selected.get("metric", "cosine")).strip().lower()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=float(umap_min_dist),
        metric=metric,
        random_state=int(random_state),
        low_memory=True,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(used.to_numpy(dtype=np.float32))
    centered = embedding - np.median(embedding, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, eigenvalues.argsort()[::-1]]
    embedding = centered @ basis
    if np.quantile(embedding[:, 0], 0.9) < abs(np.quantile(embedding[:, 0], 0.1)):
        embedding[:, 0] *= -1
    if np.quantile(embedding[:, 1], 0.9) < abs(np.quantile(embedding[:, 1], 0.1)):
        embedding[:, 1] *= -1
    frame = pd.DataFrame(
        embedding,
        index=used.index,
        columns=["x", "y"],
    )
    return frame, {
        "method": "umap",
        "dimensions_used": int(n_pcs),
        "metric": metric,
        "n_neighbors": int(neighbors),
        "min_dist": float(umap_min_dist),
        "random_state": int(random_state),
        "orientation": "principal_axis_aligned",
    }


def _embedding_frame(
    analysis_dir: Path,
    root: str,
    display_coordinates: pd.DataFrame | None = None,
    *,
    include_platform: bool = False,
) -> pd.DataFrame:
    if display_coordinates is None:
        coords = _comparison_coordinates(analysis_dir, root)
        frame = coords.iloc[:, :2].copy()
        frame.columns = ["x", "y"]
    else:
        frame = display_coordinates.copy()
    if include_platform:
        paths = _root_paths(analysis_dir, root)
        metadata = _read(paths["metadata"])
        _require(metadata, {ID, PLATFORM}, f"{root} metadata")
        metadata[ID] = metadata[ID].astype(str)
        metadata = metadata.drop_duplicates(ID, keep="first").set_index(ID)
        frame = frame.join(metadata[[PLATFORM]], how="left")
        frame[PLATFORM] = frame[PLATFORM].map(_platform)
    return frame


def _style_embedding_axis(
    axis: plt.Axes,
    *,
    x_label: str,
    y_label: str,
) -> None:
    axis.set_xlabel(x_label, labelpad=4, color=TEXT)
    axis.set_ylabel(y_label, labelpad=4, color=TEXT)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA4A8")
    axis.spines["bottom"].set_color("#9AA4A8")
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)


def _partition_frame(analysis_dir: Path, root: str) -> pd.DataFrame:
    partition = _read(_root_paths(analysis_dir, root)["partition"])
    _require(partition, {ID, "provisional_context"}, f"{root} partition")
    partition[ID] = partition[ID].astype(str)
    partition["provisional_context"] = pd.to_numeric(
        partition["provisional_context"], errors="raise"
    ).astype(int)
    return partition


def _draw_comparison_space(
    analysis_dir: Path,
    root: str,
    display_coordinates: pd.DataFrame,
    display_labels: tuple[str, str],
    maximum_points: int,
) -> plt.Figure:
    frame = _embedding_frame(
        analysis_dir,
        root,
        display_coordinates,
        include_platform=True,
    )
    sampled = _balanced_sample(frame, PLATFORM, maximum_points, 11)
    figure, axis = plt.subplots(figsize=(4.15, 3.1))
    for platform in PLATFORM_ORDER:
        subset = sampled[sampled[PLATFORM].eq(platform)]
        if subset.empty:
            continue
        axis.scatter(
            subset["x"],
            subset["y"],
            s=5.2,
            color=PLATFORM_COLORS[platform],
            alpha=0.40,
            linewidths=0,
            rasterized=True,
            label=PLATFORM_LABELS[platform],
            zorder=2,
        )
    axis.set_xlim(_robust_limits(frame["x"]))
    axis.set_ylim(_robust_limits(frame["y"]))
    axis.set_aspect("equal", adjustable="box")
    _style_embedding_axis(
        axis,
        x_label=display_labels[0],
        y_label=display_labels[1],
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handletextpad=0.30,
        columnspacing=0.75,
        markerscale=1.7,
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.25)
    return figure


def _candidate_resolution(row: pd.Series) -> object:
    for column in (
        "stage2_selected_resolution",
        "selected_resolution",
        "resolution",
    ):
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return np.nan


def _format_number(value: object, digits: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(number):
        return "?"
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.{digits}g}"


def _draw_parameter_landscape(
    candidate_summary: pd.DataFrame,
    subsampling_runs: pd.DataFrame,
    selected_parameters: pd.DataFrame,
    root: str,
) -> plt.Figure:
    _require(
        candidate_summary,
        {
            "root",
            "candidate_id",
            "n_clusters_full",
            "subsample_ari_mean",
            "subsample_ari_q10",
            "plateau_log_width",
            "n_pcs",
            "K",
            "resolution",
        },
        "subsampling candidate summary",
    )
    _require(
        subsampling_runs,
        {"root", "candidate_id", "success", "ari"},
        "subsampling runs",
    )
    data = candidate_summary[candidate_summary["root"].astype(str).eq(root)].copy()
    run_data = subsampling_runs[
        subsampling_runs["root"].astype(str).eq(root)
    ].copy()
    selected_rows = selected_parameters[
        selected_parameters["root"].astype(str).eq(root)
    ]
    if data.empty or run_data.empty or selected_rows.empty:
        raise ValueError(
            f"[{root}] candidate summary, subsampling runs, or selected rows are absent"
        )
    selected = selected_rows.iloc[0]
    for column in (
        "subsample_ari_mean",
        "subsample_ari_q10",
        "plateau_log_width",
        "n_clusters_full",
        "K",
        "n_pcs",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    selected_id = str(selected.get("candidate_id", ""))
    data["_selected"] = data["candidate_id"].astype(str).eq(selected_id)
    if not data["_selected"].any():
        raise ValueError(
            f"[{root}] selected candidate {selected_id!r} is absent from summary"
        )
    if "candidate_rank" in data.columns:
        data["candidate_rank"] = pd.to_numeric(
            data["candidate_rank"], errors="coerce"
        )
        data = data.sort_values(["candidate_rank", "candidate_id"])
    else:
        data = data.sort_values("candidate_id")
    data = data.reset_index(drop=True)
    success_mask = _as_bool(run_data["success"])
    run_data = run_data[success_mask].copy()
    run_data["ari"] = pd.to_numeric(run_data["ari"], errors="coerce")
    run_data = run_data.dropna(subset=["ari"])
    if run_data.empty:
        raise ValueError(f"[{root}] no successful subsampling ARI values")

    figure, axis = plt.subplots(figsize=(4.55, 3.0))
    figure.subplots_adjust(left=0.025, right=0.99, top=0.80, bottom=0.055)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.55, len(data) + 0.70)
    axis.axis("off")
    rng = np.random.default_rng(1701)
    selection_status = str(selected.get("selection_status", ""))
    selection_label = (
        "selected"
        if selection_status == "subsampling_selected"
        else "chosen for refinement"
    )
    ari_left = 0.50
    ari_right = 0.78
    plateau_left = 0.85
    plateau_right = 0.94
    plateau_scale = max(
        0.4,
        float(pd.to_numeric(data["plateau_log_width"], errors="coerce").max()),
    )

    header_y = len(data) + 0.38
    header_style = {
        "fontsize": 9.5,
        "color": MUTED,
        "fontweight": "bold",
        "va": "center",
    }
    axis.text(0.012, header_y + 0.14, "Candidate", ha="left", **header_style)
    axis.text(
        0.012,
        header_y - 0.20,
        "(n communities)",
        ha="left",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )
    axis.text(0.20, header_y, r"PCs $\cdot$ K $\cdot$ $\gamma$", ha="left", **header_style)
    axis.text(
        ari_left,
        header_y,
        "Subsampling ARI",
        ha="left",
        **header_style,
    )
    axis.text(
        plateau_left,
        header_y,
        "Plateau width",
        ha="left",
        **header_style,
    )

    scale_y = len(data) + 0.08
    axis.plot(
        [ari_left, ari_right],
        [scale_y, scale_y],
        color=GRID,
        linewidth=0.55,
        zorder=0,
    )
    for tick_x, tick_label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        position = ari_left + tick_x * (ari_right - ari_left)
        axis.plot(
            [position, position],
            [scale_y - 0.04, scale_y + 0.04],
            color=GRID,
            linewidth=0.55,
            zorder=0,
        )
        axis.text(
            position,
            scale_y - 0.08,
            tick_label,
            ha="center",
            va="top",
            fontsize=8.2,
            color=MUTED,
        )

    for index, (_, row) in enumerate(data.iterrows()):
        row_y = len(data) - index - 0.55
        selected_candidate = bool(row["_selected"])
        if selected_candidate:
            axis.axhspan(row_y - 0.43, row_y + 0.43, color="#EAF4F3", zorder=0)
        axis.axhline(
            row_y - 0.48,
            color=GRID,
            linewidth=0.5,
            zorder=0,
        )

        axis.plot(
            [ari_left, ari_right],
            [row_y, row_y],
            color=GRID,
            linewidth=0.55,
            zorder=0,
        )
        for tick_x in (0.0, 0.5, 1.0):
            position = ari_left + tick_x * (ari_right - ari_left)
            axis.plot(
                [position, position],
                [row_y - 0.035, row_y + 0.035],
                color=GRID,
                linewidth=0.55,
                zorder=0,
            )

        candidate = str(row["candidate_id"]).split("__")[-1]
        axis.text(
            0.012,
            row_y + (0.14 if selected_candidate else 0.0),
            f"{candidate} · {int(row['n_clusters_full'])}",
            ha="left",
            va="center",
            fontsize=10.5,
            color=TEXT,
            fontweight="bold" if selected_candidate else "normal",
        )
        axis.text(
            0.20,
            row_y,
            (
                f"{int(row['n_pcs'])} · {int(row['K'])} · "
                f"{_format_number(row['resolution']).removeprefix('0')}"
            ),
            ha="left",
            va="center",
            fontsize=10.0,
            color=TEXT,
            fontweight="bold" if selected_candidate else "normal",
        )
        if selected_candidate:
            axis.text(
                0.012,
                row_y - 0.30,
                selection_label,
                ha="left",
                va="center",
                fontsize=6.6,
                color=TEAL,
                fontweight="bold",
            )

        run_values = run_data.loc[
            run_data["candidate_id"].astype(str).eq(str(row["candidate_id"])),
            "ari",
        ]
        jitter = rng.uniform(-0.035, 0.035, len(run_values))
        run_positions = ari_left + run_values * (ari_right - ari_left)
        axis.scatter(
            run_positions,
            row_y + jitter,
            s=11,
            color=GREY,
            alpha=0.62,
            linewidth=0,
            zorder=2,
        )
        q10 = float(row["subsample_ari_q10"])
        mean = float(row["subsample_ari_mean"])
        q10_position = ari_left + q10 * (ari_right - ari_left)
        mean_position = ari_left + mean * (ari_right - ari_left)
        axis.plot(
            [q10_position, mean_position],
            [row_y, row_y],
            color=TEXT,
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            [q10_position],
            [row_y],
            marker="D",
            s=29,
            facecolor="white",
            edgecolor=TEXT,
            linewidth=0.9,
            zorder=4,
        )
        axis.scatter(
            [mean_position],
            [row_y],
            marker="o",
            s=31,
            facecolor=TEAL if selected_candidate else TEXT,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )

        width = float(row["plateau_log_width"])
        axis.barh(
            row_y,
            plateau_right - plateau_left,
            height=0.09,
            left=plateau_left,
            color="#EFF2F2",
            edgecolor="none",
            zorder=1,
        )
        if np.isfinite(width):
            displayed_width = (
                width / plateau_scale * (plateau_right - plateau_left)
            )
            axis.barh(
                row_y,
                displayed_width,
                height=0.09,
                left=plateau_left,
                color=TEAL if selected_candidate else "#A8C7C5",
                edgecolor="none",
                zorder=2,
            )
            axis.text(
                min(plateau_left + displayed_width + 0.007, 0.995),
                row_y,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontsize=8.7,
                color=TEXT,
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.2,
            markerfacecolor=GREY,
            markeredgewidth=0,
            label="repeat",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=TEXT,
            markeredgewidth=0.9,
            label="q10",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.8,
            markerfacecolor=TEXT,
            markeredgecolor="white",
            label="mean",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.905),
        frameon=False,
        ncol=3,
        fontsize=9.2,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    return figure


def _community_data(
    analysis_dir: Path,
    root: str,
    display_coordinates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = _embedding_frame(
        analysis_dir,
        root,
        display_coordinates,
    )
    partition = _partition_frame(analysis_dir, root).set_index(ID)
    frame = frame.join(partition[["provisional_context"]], how="inner")
    frame = frame[frame["provisional_context"].ge(0)].copy()
    sizes = frame["provisional_context"].astype(int).value_counts().sort_index()
    return frame, sizes


def _draw_initial_communities(
    analysis_dir: Path,
    root: str,
    display_coordinates: pd.DataFrame,
    display_labels: tuple[str, str],
    maximum_points: int,
) -> plt.Figure:
    frame, _sizes = _community_data(
        analysis_dir,
        root,
        display_coordinates,
    )
    sampled = _balanced_sample(frame, "provisional_context", maximum_points, 21)
    figure, axis = plt.subplots(figsize=DISCOVERY_PANEL_SIZE)
    for context in sorted(frame["provisional_context"].astype(int).unique()):
        subset = sampled[sampled["provisional_context"].astype(int).eq(context)]
        axis.scatter(
            subset["x"],
            subset["y"],
            s=5.2,
            color=_colour(context),
            alpha=0.42,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        full_subset = frame[frame["provisional_context"].astype(int).eq(context)]
        centroid = full_subset[["x", "y"]].median().to_numpy(float)
        label = axis.text(
            centroid[0],
            centroid[1],
            f"C{context}",
            fontsize=7.2,
            fontweight="bold",
            color=_colour(context),
            ha="center",
            va="center",
            zorder=5,
        )
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.3, foreground="white"),
                path_effects.Normal(),
            ]
        )
    axis.set_xlim(_robust_limits(frame["x"]))
    axis.set_ylim(_robust_limits(frame["y"]))
    axis.set_aspect("equal", adjustable="box")
    _style_embedding_axis(
        axis,
        x_label=display_labels[0],
        y_label=display_labels[1],
    )
    figure.subplots_adjust(left=0.10, right=0.99, top=0.99, bottom=0.11)
    return figure


def _draw_community_sizes(
    analysis_dir: Path,
    root: str,
    display_coordinates: pd.DataFrame,
) -> plt.Figure:
    _frame, sizes = _community_data(
        analysis_dir,
        root,
        display_coordinates,
    )
    data = sizes.sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=COMMUNITY_SIZE_PANEL_SIZE)
    y = np.arange(len(data))
    axis.barh(
        y,
        data.to_numpy(),
        color=[_colour(int(context)) for context in data.index],
        height=0.64,
        edgecolor="none",
        zorder=2,
    )
    offset = max(float(data.max()) * 0.025, 5)
    for position, value in zip(y, data.to_numpy(), strict=True):
        axis.text(
            float(value) + offset,
            position,
            f"{int(value):,}",
            va="center",
            fontsize=7,
            color=TEXT,
        )
    axis.set_yticks(y, [f"C{int(context)}" for context in data.index])
    axis.set_xlabel("Analyses")
    axis.set_xlim(0, float(data.max()) + 7 * offset)
    _clean(axis, grid_axis="x")
    figure.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.28)
    return figure


def _select_examples(
    decisions: pd.DataFrame,
    evidence: pd.DataFrame,
    root: str,
) -> tuple[pd.Series, pd.Series]:
    work = decisions[decisions["root"].astype(str).eq(root)].copy()
    if work.empty:
        raise ValueError(f"No pair decisions for {root}")
    for column in (
        "merge_applied",
        "structural_merge_candidate",
        "evidence_veto",
        "track1_evidence_veto",
        "track2_evidence_veto",
    ):
        work[column] = _as_bool(work[column]) if column in work.columns else False
    merged = work[work["merge_applied"]]
    if merged.empty:
        raise ValueError(f"[{root}] no merge-applied pair is available")
    merge = merged.sort_values("comerge_rate", ascending=False).iloc[0]
    veto_rows = work[work["evidence_veto"]].copy()
    if veto_rows.empty:
        raise ValueError(f"[{root}] no evidence-retained pair is available")
    supported = evidence[
        evidence["root"].astype(str).eq(root)
        & _as_bool(evidence["supports_boundary_veto"])
    ]
    support_types = supported.groupby(["context_a", "context_b"])["evidence_type"].agg(
        lambda values: set(values.astype(str))
    )
    scores: list[float] = []
    for row in veto_rows.itertuples(index=False):
        observed = support_types.get(
            (int(row.context_a), int(row.context_b)),
            set(),
        )
        scores.append(
            100.0 * int(observed == {"track1", "track2"})
            + 20.0 * int(bool(row.structural_merge_candidate))
            + 5.0 * len(observed)
            + float(row.comerge_rate)
        )
    veto_rows["_example_score"] = scores
    veto = veto_rows.sort_values("_example_score", ascending=False).iloc[0]
    return merge, veto


def _select_reference_example(
    references: pd.DataFrame,
    signatures: pd.DataFrame,
    root: str,
    reference_key: str | None = None,
) -> tuple[pd.Series, int]:
    candidates = references[
        references["reference_level"].astype(str).eq("subtype")
    ].copy()
    if reference_key is not None:
        candidates = candidates[candidates["reference_key"].astype(str).eq(reference_key)]
        if len(candidates) != 1:
            raise ValueError(f"{reference_key}: expected one released subtype")
        root = str(candidates.iloc[0]["root"])
    else:
        candidates = candidates[candidates["root"].astype(str).eq(root)]
    if candidates.empty:
        raise ValueError(f"[{root}] no released subtype reference is available")
    observed = (
        signatures[
            signatures["reference_key"]
            .astype(str)
            .isin(candidates["reference_key"].astype(str))
        ]
        .groupby(["reference_key", "track"])
        .size()
        .unstack(fill_value=0)
    )
    candidates["_track1_events"] = (
        (candidates["reference_key"].astype(str).map(observed.get("track1", {})))
        .fillna(0)
        .astype(int)
    )
    candidates["_track2_events"] = (
        (candidates["reference_key"].astype(str).map(observed.get("track2", {})))
        .fillna(0)
        .astype(int)
    )
    for track in ("track1", "track2"):
        count_column = f"n_{track}_events"
        if count_column in candidates:
            expected = pd.to_numeric(candidates[count_column], errors="raise")
            if not expected.eq(candidates[f"_{track}_events"]).all():
                raise ValueError(f"{track} counts differ between reference and signature tables")
    candidates = candidates[
        candidates["_track1_events"].gt(0) & candidates["_track2_events"].gt(0)
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"[{root}] no released community contains both Track 1 and Track 2 "
            "signature events"
        )
    candidates["_dual_supported"] = (
        candidates["evidence_class"].astype(str).eq("dual_supported")
    )
    candidates["_balanced_evidence"] = candidates[
        ["_track1_events", "_track2_events"]
    ].min(axis=1)
    candidates["_definition_members"] = pd.to_numeric(
        candidates["n_definition_members"], errors="coerce"
    ).fillna(0)
    selected = candidates.sort_values(
        [
            "_dual_supported",
            "_balanced_evidence",
            "_track2_events",
            "_definition_members",
        ],
        ascending=False,
    ).iloc[0]
    match = re.search(r"-CTX(-?\d+)$", str(selected["reference_key"]))
    if match is None:
        raise ValueError(
            "Cannot infer refined context from reference key: "
            f"{selected['reference_key']}"
        )
    return selected, int(match.group(1))


def _draw_comerge_pairs(
    decisions: pd.DataFrame,
    root: str,
    highlighted_pairs: tuple[pd.Series, ...] = (),
) -> plt.Figure:
    data = decisions[decisions["root"].astype(str).eq(root)].copy()
    if data.empty:
        raise ValueError(f"No pair decisions for {root}")
    examples = [pair for pair in highlighted_pairs if pair is not None]
    if len(examples) != 2:
        raise ValueError("Co-merge repeat panel requires merge and retained examples")
    repeats = 20
    figure, axis = plt.subplots(figsize=(4.25, 1.65))
    rows = [
        (examples[0], 1.0, GREEN, "merge"),
        (examples[1], 0.0, RED, "retain boundary"),
    ]
    for pair, y, decision_color, outcome in rows:
        rate = float(pair["comerge_rate"])
        n_comerged = int(np.clip(round(rate * repeats), 0, repeats))
        x = np.arange(1, repeats + 1)
        colors = [TEAL] * n_comerged + ["#E7ECEC"] * (repeats - n_comerged)
        edges = [TEAL] * n_comerged + ["#C7D0D2"] * (repeats - n_comerged)
        axis.scatter(
            x,
            np.full(repeats, y),
            s=29,
            color=colors,
            edgecolor=edges,
            linewidth=0.5,
            zorder=3,
        )
        axis.text(
            repeats + 1.0,
            y,
            f"{n_comerged}/{repeats}  {outcome}",
            color=decision_color,
            fontsize=6.8,
            fontweight="bold",
            va="center",
        )
    axis.set_xlim(0.25, repeats + 5.0)
    axis.set_ylim(-0.55, 1.55)
    axis.set_xticks([])
    axis.set_yticks(
        [1.0, 0.0],
        [
            f"C{int(examples[0]['context_a'])} + C{int(examples[0]['context_b'])}",
            f"C{int(examples[1]['context_a'])} + C{int(examples[1]['context_b'])}",
        ],
    )
    axis.set_xlabel("20 subsampling repeats")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="y", length=0, pad=5, colors=MUTED)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=5,
            label=label,
        )
        for face, edge, label in (
            (TEAL, TEAL, "co-clustered"),
            ("#E7ECEC", "#C7D0D2", "separate"),
        )
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.16),
        frameon=False,
        fontsize=6.4,
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    figure.subplots_adjust(left=0.19, right=0.99, top=0.83, bottom=0.24)
    return figure


def _pair_frame(
    analysis_dir: Path,
    row: pd.Series,
    display_coordinates: pd.DataFrame,
    maximum_points: int,
) -> tuple[pd.DataFrame, int, int]:
    root = str(row["root"])
    frame = _embedding_frame(
        analysis_dir,
        root,
        display_coordinates,
    )
    partition = _partition_frame(analysis_dir, root).set_index(ID)
    frame = frame.join(partition[["provisional_context"]], how="inner")
    context_a, context_b = int(row["context_a"]), int(row["context_b"])
    frame = frame[frame["provisional_context"].astype(int).isin({context_a, context_b})]
    sampled = _balanced_sample(frame, "provisional_context", maximum_points, 31)
    return sampled, context_a, context_b


def _draw_pair_example(
    analysis_dir: Path,
    row: pd.Series | None,
    display_coordinates: pd.DataFrame | None,
    display_labels: tuple[str, str],
    *,
    retain: bool,
    maximum_points: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(3.1, 3.1))
    if row is None:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No qualifying pair",
            ha="center",
            va="center",
            color=MUTED,
        )
        return figure
    if display_coordinates is None:
        raise ValueError("Pair display coordinates are required")
    frame, context_a, context_b = _pair_frame(
        analysis_dir,
        row,
        display_coordinates,
        maximum_points,
    )
    for context in (context_a, context_b):
        subset = frame[frame["provisional_context"].astype(int).eq(context)]
        axis.scatter(
            subset["x"],
            subset["y"],
            s=6,
            color=_colour(context),
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            label=f"C{context}",
            zorder=2,
        )
    centroids = frame.groupby("provisional_context")[["x", "y"]].median()
    first = centroids.loc[context_a].to_numpy(float)
    second = centroids.loc[context_b].to_numpy(float)
    if retain:
        midpoint = (first + second) / 2
        direction = second - first
        norm = np.linalg.norm(direction)
        if norm:
            perpendicular = np.array([-direction[1], direction[0]]) / norm
            x_span = frame["x"].max() - frame["x"].min()
            y_span = frame["y"].max() - frame["y"].min()
            edge = perpendicular * max(x_span, y_span) * 0.20
            axis.plot(
                [midpoint[0] - edge[0], midpoint[0] + edge[0]],
                [midpoint[1] - edge[1], midpoint[1] + edge[1]],
                color=RED,
                linewidth=1.4,
                zorder=5,
            )
    else:
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=GREEN,
            linewidth=1.25,
            zorder=5,
        )
        axis.scatter(
            [first[0], second[0]],
            [first[1], second[1]],
            s=24,
            color=[_colour(context_a), _colour(context_b)],
            edgecolor="white",
            linewidth=0.65,
            zorder=6,
        )
    axis.set_xlim(_robust_limits(display_coordinates["x"], lower=0.01))
    axis.set_ylim(_robust_limits(display_coordinates["y"], lower=0.01))
    axis.set_aspect("equal", adjustable="box")
    _style_embedding_axis(
        axis,
        x_label=display_labels[0],
        y_label=display_labels[1],
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=6.7,
        handletextpad=0.35,
        columnspacing=0.9,
        markerscale=1.4,
    )
    figure.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.22)
    return figure


def _track1_events(signatures: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    reference_key = str(reference["reference_key"])
    events = signatures[
        signatures["reference_key"].astype(str).eq(reference_key)
        & signatures["track"].astype(str).eq("track1")
    ].copy()
    if events.empty:
        raise ValueError(f"{reference_key}: no released Track 1 events")
    events["chromosome"] = events["feature"].map(_autosome_number)
    if events["chromosome"].isna().any() or not events["event"].isin(["gain", "loss"]).all():
        raise ValueError("Track 1 requires autosomal gain/loss events")
    return events


def _draw_track1(signatures: pd.DataFrame, reference: pd.Series) -> plt.Figure:
    figure, axis = plt.subplots(figsize=REFINEMENT_PANEL_SIZE)
    events = _track1_events(signatures, reference)
    released = set(zip(events["chromosome"].astype(int), events["event"]))
    # Presence of a released event is categorical; do not rescale tiny coverage values.
    for chromosome in range(1, 23):
        for event, row_y, color in (("gain", 1.12, ORANGE), ("loss", 0.12, BLUE)):
            axis.add_patch(
                Rectangle(
                    (chromosome - 0.38, row_y),
                    0.72,
                    0.62,
                    facecolor=color if (chromosome, event) in released else LIGHT,
                    edgecolor="white",
                    linewidth=0.55,
                )
            )
    axis.set_xlim(0.35, 22.65)
    axis.set_ylim(-0.02, 1.9)
    axis.set_xticks([1, 5, 9, 13, 17, 21])
    axis.set_yticks([0.43, 1.43], ["loss", "gain"])
    axis.set_xlabel("Chromosome")
    axis.tick_params(axis="y", length=0, pad=7)
    axis.tick_params(axis="x", length=2.5, color=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        handles=[Patch(facecolor=LIGHT, edgecolor=GRID, label="No released event")],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=6.4,
    )
    figure.subplots_adjust(left=0.16, right=0.99, top=0.90, bottom=0.22)
    return figure


def _autosome_number(value: object) -> int | None:
    text = str(value).strip().lower()
    if text.startswith("chr"):
        text = text[3:]
    if not text.isdigit():
        return None
    chromosome = int(text)
    return chromosome if 1 <= chromosome <= 22 else None


def _select_track2_markers(data: pd.DataFrame, maximum: int = 7) -> pd.DataFrame:
    ranked = data.sort_values(
        ["context_value", "_effect_abs"],
        ascending=False,
    )
    selected = ranked.head(maximum).copy()
    selected_events = set(selected["event"].astype(str).str.lower())
    for event in ("gain", "loss"):
        if event in selected_events:
            continue
        replacement = ranked[ranked["event"].astype(str).str.lower().eq(event)]
        if replacement.empty:
            continue
        selected = pd.concat(
            [selected.iloc[:-1], replacement.head(1)],
            ignore_index=True,
        )
    return selected.drop_duplicates(["feature", "event"], keep="first").sort_values(
        ["_chromosome", "context_value"]
    )


def _track2_display_data(
    signatures: pd.DataFrame,
    reference: pd.Series,
    gene_intervals: pd.DataFrame,
) -> pd.DataFrame:
    reference_key = str(reference["reference_key"])
    data = signatures[
        signatures["reference_key"].astype(str).eq(reference_key)
        & signatures["track"].astype(str).eq("track2")
    ].copy()
    adjusted = pd.to_numeric(data["adjusted_effect_size"], errors="coerce")
    raw = pd.to_numeric(data["effect_size"], errors="coerce")
    data["_effect"] = adjusted.fillna(raw)
    data["_effect_abs"] = data["_effect"].abs()
    data["context_value"] = pd.to_numeric(data["context_value"], errors="coerce")
    chromosome_map = (
        gene_intervals.assign(_chromosome=gene_intervals["chrom"].map(_autosome_number))
        .dropna(subset=["gene_symbol", "_chromosome"])
        .drop_duplicates("gene_symbol", keep="first")
        .set_index("gene_symbol")["_chromosome"]
    )
    data["_chromosome"] = (
        data["feature"].astype(str).map(chromosome_map).astype("Int64")
    )
    data = data.dropna(subset=["_effect", "context_value", "_chromosome"])
    if data.empty:
        raise ValueError(
            f"No chromosome-mapped Track 2 events are available for {reference_key}"
        )
    if not data["context_value"].between(0, 1).all():
        raise ValueError("Track 2 raw carrier fractions must be in [0, 1]")
    if not data["event"].isin(["gain", "loss"]).all():
        raise ValueError("Track 2 requires gain/loss events")
    data = _select_track2_markers(data).reset_index(drop=True)
    data["raw_carrier_fraction_percent"] = 100.0 * data["context_value"]
    return data


def _draw_track2(
    signatures: pd.DataFrame,
    reference: pd.Series,
    gene_intervals: pd.DataFrame,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=REFINEMENT_PANEL_SIZE)
    figure.subplots_adjust(left=0.17, right=0.95, top=0.83, bottom=0.22)
    data = _track2_display_data(signatures, reference, gene_intervals)
    positions = np.zeros(len(data), dtype=float)
    chromosome_ticks: list[float] = []
    chromosome_labels: list[str] = []
    cursor = 0.0
    for chromosome in sorted(data["_chromosome"].astype(int).unique()):
        indices = np.flatnonzero(data["_chromosome"].astype(int).eq(chromosome))
        group_positions = cursor + np.arange(len(indices), dtype=float)
        positions[indices] = group_positions
        chromosome_ticks.append(float(group_positions.mean()))
        chromosome_labels.append(str(chromosome))
        cursor = float(group_positions[-1]) + 1.8
    values = data["raw_carrier_fraction_percent"].to_numpy(float)
    events = data["event"].astype(str).str.lower().to_numpy()
    colors = np.where(events == "gain", ORANGE, BLUE)
    for position, value, color in zip(positions, values, colors, strict=True):
        axis.vlines(position, 0, value, color=color, linewidth=1.25, zorder=2)
    axis.scatter(
        positions,
        values,
        s=30,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    value_range = max(float(np.nanmax(values)), 1.0)
    label_positions: list[tuple[float, float]] = []
    label_intervals: list[tuple[float, float, float]] = []
    x_limits = (float(positions.min()) - 0.75, float(positions.max()) + 0.75)
    axis.set_xlim(*x_limits)
    renderer = figure.canvas.get_renderer()
    label_font = FontProperties(size=6.7, weight="bold")
    for rank, (position, value, feature, event) in enumerate(
        zip(
            positions,
            values,
            data["feature"].astype(str),
            events,
            strict=True,
        )
    ):
        if rank == 0:
            direction = 1.0
        elif rank == len(data) - 1:
            direction = -1.0
        else:
            direction = 1.0 if rank % 2 == 0 else -1.0
        label_x = position + 0.2 * direction
        label_y = value + 0.085 * value_range
        label_text = f"{feature}{'+' if event == 'gain' else '-'}"
        width_pixels = renderer.get_text_width_height_descent(label_text, label_font, False)[0]
        width = width_pixels / axis.get_window_extent().width * (x_limits[1] - x_limits[0])
        left = label_x if direction > 0 else label_x - width
        right = label_x + width if direction > 0 else label_x
        # Opposite text alignments can overlap even when their anchor points do not.
        while any(
            left < previous_right + 0.12 and right > previous_left - 0.12
            and abs(label_y - previous_y) < 0.16 * value_range
            for previous_left, previous_right, previous_y in label_intervals
        ):
            label_y += 0.16 * value_range
        label_positions.append((label_x, label_y))
        label_intervals.append((left, right, label_y))
        axis.annotate(
            label_text,
            xy=(position, value),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="left" if direction > 0 else "right",
            va="bottom",
            color=TEXT,
            fontsize=6.7,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": "#AAB3B7",
                "linewidth": 0.55,
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )
    axis.axhline(0, color="#C8C3B8", linewidth=1.0)
    axis.set_xlim(float(positions.min()) - 0.75, float(positions.max()) + 0.75)
    label_ceiling = max(label_y for _, label_y in label_positions)
    axis.set_ylim(
        0,
        max(value_range * 1.32, label_ceiling + 0.12 * value_range),
    )
    # Space above 100% is for labels, not additional carrier-fraction values.
    axis.set_yticks([tick for tick in axis.get_yticks() if 0 <= tick <= 100])
    axis.set_xticks(chromosome_ticks, chromosome_labels)
    axis.set_xlabel("Chromosome")
    axis.set_ylabel("Raw carrier fraction (%)")
    _clean(axis, grid_axis="y")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=5,
            label=event.capitalize(),
        )
        for event, color in (("gain", ORANGE), ("loss", BLUE))
        if event in set(events)
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    figure.subplots_adjust(left=0.17, right=0.95, top=0.83, bottom=0.22)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Defaults to <atlas-dir>/work/analysis_v1/analysis.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--root",
        default="C22_C24",
        help="Root for representation, discovery and refinement panels (default: C22_C24).",
    )
    parser.add_argument(
        "--reference-key",
        help=(
            "Released subtype for both signature panels, independent of --root. "
            "If omitted, an example is selected from --root."
        ),
    )
    parser.add_argument(
        "--signatures-only",
        action="store_true",
        help="Render only the two signature panels and their event tables; skip UMAP and discovery panels.",
    )
    parser.add_argument(
        "--display-space",
        choices=("umap", "pc"),
        default="umap",
        help=(
            "Two-dimensional display used by stages II and IV. UMAP is fitted "
            "to the selected comparison-space PCs and is visualization-only."
        ),
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.25,
        help="UMAP minimum distance for the shared II/IV display (default: 0.25).",
    )
    parser.add_argument(
        "--display-random-state",
        type=int,
        default=42,
        help="Random seed for the shared UMAP display.",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _configure()
    atlas_dir = args.atlas_dir.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir is not None
        else atlas_dir / "work" / "analysis_v1" / "analysis"
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_stems = [
        "II_comparison_space",
        "III_parameter_candidates",
        "IV_initial_communities",
        "IV_initial_community_sizes",
        "V_pairwise_comerge",
        "V_merge_applied",
        "V_boundary_retained",
        "V_track1_architecture",
        "V_track2_gene_events",
    ]
    if args.signatures_only:
        output_stems = ["V_track1_architecture", "V_track2_gene_events"]
    output_paths = [out_dir / f"{stem}.png" for stem in output_stems]
    display_table_paths = {
        "track1_events": out_dir / "V_track1_events.tsv",
        "track2_displayed_events": out_dir / "V_track2_displayed_events.tsv",
    }
    manifest_path = out_dir / (
        "figure2_signature_panels.manifest.json"
        if args.signatures_only
        else "figure2_workflow_panels.manifest.json"
    )
    existing = [
        path for path in [*output_paths, *display_table_paths.values(), manifest_path]
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {existing[0]}")
    if args.overwrite:
        for stem in output_stems:
            for extension in ("svg", "pdf"):
                legacy_path = out_dir / f"{stem}.{extension}"
                if legacy_path.exists():
                    legacy_path.unlink()

    release_tables = atlas_dir / "release_candidate" / "tables"
    references_path = release_tables / "reference_contexts.tsv"
    signatures_path = release_tables / "signature_events.tsv"
    gene_intervals_path = (
        atlas_dir / "projection_bundle_candidate" / "gene_intervals.tsv"
    )
    references = _read(references_path)
    signatures = _read(signatures_path)
    gene_intervals = _read(gene_intervals_path)
    _require(
        references,
        {
            "reference_key",
            "root",
            "reference_level",
            "n_definition_members",
            "evidence_class",
        },
        "reference contexts",
    )
    _require(
        signatures,
        {
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "context_value",
            "effect_size",
            "adjusted_effect_size",
        },
        "signature events",
    )
    _require(
        gene_intervals,
        {"gene_symbol", "chrom"},
        "projection gene intervals",
    )
    reference_example, reference_context = _select_reference_example(
        references,
        signatures,
        str(args.root),
        reference_key=args.reference_key,
    )
    inputs = {
        "reference_contexts": str(references_path),
        "signature_events": str(signatures_path),
        "gene_intervals": str(gene_intervals_path),
    }
    figures = {}
    display_metadata = None
    merge_example = veto_example = None
    if not args.signatures_only:
        parameter_dir = analysis_dir / "parameters"
        candidate_summary_path = parameter_dir / "subsampling_by_candidate.tsv"
        subsampling_runs_path = parameter_dir / "subsampling_runs.tsv"
        selected_path = parameter_dir / "selected_parameters.tsv"
        decisions_path = _find(
            analysis_dir, "pair_merge_decisions.tsv", "pair merge decisions"
        )
        evidence_path = _find(
            analysis_dir, "pair_boundary_evidence.tsv", "pair boundary evidence"
        )
        candidate_summary = _read(candidate_summary_path)
        subsampling_runs = _read(subsampling_runs_path)
        selected_parameters = _read(selected_path)
        decisions = _read(decisions_path)
        evidence = _read(evidence_path)
        _require(
            decisions,
            {"root", "context_a", "context_b", "comerge_rate"},
            "pair decisions",
        )
        _require(
            evidence,
            {
                "root",
                "context_a",
                "context_b",
                "evidence_type",
                "supports_boundary_veto",
                "pair_delta_a_minus_b",
            },
            "pair evidence",
        )
        merge_example, veto_example = _select_examples(
            decisions,
            evidence,
            str(args.root),
        )
        display_coordinates, display_metadata = _display_embedding(
            analysis_dir,
            selected_parameters,
            str(args.root),
            display_space=str(args.display_space),
            umap_min_dist=float(args.umap_min_dist),
            random_state=int(args.display_random_state),
        )
        display_labels = (
            ("UMAP1", "UMAP2") if display_metadata["method"] == "umap" else ("PC1", "PC2")
        )
        inputs.update({
            "analysis_dir": str(analysis_dir),
            "subsampling_candidate_summary": str(candidate_summary_path),
            "subsampling_runs": str(subsampling_runs_path),
            "selected_parameters": str(selected_path),
            "pair_decisions": str(decisions_path),
            "pair_boundary_evidence": str(evidence_path),
        })
        figures.update({
            "II_comparison_space": _draw_comparison_space(
                analysis_dir,
                str(args.root),
                display_coordinates,
                display_labels,
                int(args.max_points),
            ),
            "III_parameter_candidates": _draw_parameter_landscape(
                candidate_summary,
                subsampling_runs,
                selected_parameters,
                str(args.root),
            ),
            "IV_initial_communities": _draw_initial_communities(
                analysis_dir,
                str(args.root),
                display_coordinates,
                display_labels,
                int(args.max_points),
            ),
            "IV_initial_community_sizes": _draw_community_sizes(
                analysis_dir,
                str(args.root),
                display_coordinates,
            ),
            "V_pairwise_comerge": _draw_comerge_pairs(
                decisions,
                str(args.root),
                (
                    merge_example,
                    veto_example,
                ),
            ),
            "V_merge_applied": _draw_pair_example(
                analysis_dir,
                merge_example,
                display_coordinates,
                display_labels,
                retain=False,
                maximum_points=max(1200, int(args.max_points) // 2),
            ),
            "V_boundary_retained": _draw_pair_example(
                analysis_dir,
                veto_example,
                display_coordinates,
                display_labels,
                retain=True,
                maximum_points=max(1200, int(args.max_points) // 2),
            ),
        })
    figures.update({
        "V_track1_architecture": _draw_track1(
            signatures,
            reference_example,
        ),
        "V_track2_gene_events": _draw_track2(
            signatures,
            reference_example,
            gene_intervals,
        ),
    })
    written: dict[str, str] = {}
    fixed_canvas_stems = {
        "IV_initial_communities",
        "IV_initial_community_sizes",
        "V_merge_applied",
        "V_boundary_retained",
        "V_track1_architecture",
        "V_track2_gene_events",
    }
    for stem, figure in figures.items():
        path = _save(
            figure,
            out_dir,
            stem,
            int(args.dpi),
            fixed_canvas=stem in fixed_canvas_stems,
        )
        written[stem] = path.name
    track1_events = _track1_events(signatures, reference_example)
    track2_displayed = _track2_display_data(signatures, reference_example, gene_intervals)
    track1_events.to_csv(display_table_paths["track1_events"], sep="\t", index=False)
    track2_displayed.to_csv(display_table_paths["track2_displayed_events"], sep="\t", index=False)

    def _pair_record(row: pd.Series | None) -> dict[str, object] | None:
        if row is None:
            return None
        return {
            "root": str(row["root"]),
            "context_a": int(row["context_a"]),
            "context_b": int(row["context_b"]),
            "comerge_rate": float(row["comerge_rate"]),
        }

    manifest = {
        "schema": "cnvsim_figure2_workflow_panels_v3",
        "render_scope": "signatures_only" if args.signatures_only else "all",
        "description": (
            "Title-free, data-driven panels for manual Figure 2A assembly. "
            "No atlas analysis is recomputed."
        ),
        "representation_root": None if args.signatures_only else str(args.root),
        "display_space": display_metadata,
        "merge_example": _pair_record(merge_example),
        "boundary_retained_example": _pair_record(veto_example),
        "signature_example": {
            "reference_key": str(reference_example["reference_key"]),
            "root": str(reference_example["root"]),
            "refined_context": int(reference_context),
            "evidence_class": str(reference_example["evidence_class"]),
            "n_definition_members": int(reference_example["n_definition_members"]),
            "n_track1_events": int(reference_example["_track1_events"]),
            "n_track2_events": int(reference_example["_track2_events"]),
            "n_track2_displayed": int(len(track2_displayed)),
        },
        "panel_encoding": {
            "track1": (
                "Categorical presence of released chromosome gain/loss events. "
                "Grey means no released event, not absence of CNV or copy-neutrality. "
                "No per-direction coverage scaling is used."
            ),
            "track2": (
                "Released events selected using the existing raw-prevalence ranking. "
                "Height is context_value * 100, the fraction of analyses with a nonzero "
                "raw gene-event fraction, not relative enrichment or CNV amplitude. "
                "Signature selection uses within-analysis mean-centred fractions."
            ),
            "parameter_selection": (
                "ARI compares each subsampled partition with the full-data partition. "
                "The selected highlight does not imply all automatic gates passed."
            ),
            "boundary_refinement": (
                "Dots summarize co-clustering counts, not repeat order. "
                "The red boundary line is illustrative, not a fitted decision boundary."
            ),
        },
        "inputs": inputs,
        "outputs": written,
        "display_tables": {key: path.name for key, path in display_table_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    scope = "signature" if args.signatures_only else "workflow"
    print(f"[CNVSIM] Figure 2 {scope} panels complete")
    print(
        "  signature example: "
        f"{reference_example['reference_key']} "
        f"(Track 1={int(reference_example['_track1_events'])}, "
        f"Track 2={int(reference_example['_track2_events'])})"
    )
    for stem, path in written.items():
        print(f"  {stem}: {out_dir / path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
