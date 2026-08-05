#!/usr/bin/env python3
from __future__ import annotations

"""Render projection and experimental-model application panels for Figure 7."""

import argparse
import math
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from cnvsim.preprocess.gene_panel_store import load_gene_panel_subset_from_cache
from cnvsim.projection import EVENT_COLUMNS


TEXT = "#111111"
MUTED = "#68747A"
GRID = "#E3E8EA"
BLUE = "#2E5EAA"
LIGHT_BLUE = "#A8C4E5"
ORANGE = "#D9822B"
LIGHT_GRAY = "#E4E8EA"
MID_GRAY = "#AAB4B9"
WHITE = "#FFFFFF"
GAIN = "#F2BD35"
LOSS = "#1F5F93"
PROFILE_GRID = "#D8D5CC"
PAPER_TEXT_SIZE = 7.0
PAPER_SMALL_TEXT_SIZE = 6.1

ROOT_LABELS = {
    "C00_C14": "Oral/pharynx",
    "C15_C17_C21_C26": "Other digestive",
    "C16": "Stomach",
    "C18_C20": "Colon/rectum",
    "C22_C24": "Liver/biliary",
    "C25": "Pancreas",
    "C30_C39": "Respiratory",
    "C40_C41_C48": "Bone/connective",
    "C42": "Haematopoietic",
    "C44": "Skin",
    "C49": "Soft tissue",
    "C50": "Breast",
    "C51_C58": "Female genital",
    "C60_C63": "Male genital",
    "C64_C68": "Urinary tract",
    "C69_C70_C72": "Eye/CNS",
    "C71": "Brain",
    "C73_C75": "Endocrine",
    "C77": "Lymph nodes",
}

META_COLUMNS = [
    "depmap_id",
    "cell_line_display_name",
    "lineage_1",
    "lineage_2",
    "lineage_3",
    "lineage_6",
    "lineage_4",
]

FUNCTIONAL_CASE = {
    "D": {
        "slug": "melanoma",
        "root": "C44",
        "reference_key": "C44-CTX5",
        "label": "Melanoma",
        "pathway_drugs": [
            "AZ-628",
            "COBIMETINIB",
            "DABRAFENIB",
            "ENCORAFENIB",
            "SELUMETINIB",
            "TRAMETINIB",
        ],
        "dependency_anchors": ["MAPK1"],
    },
}

PROFILE_CASES = [
    {
        "root": "C44",
        "reference_key": "C44-CTX5",
        "reference_label": "Melanoma core subtype",
        "model_id": "cellosaurus:CVCL_1779",
        "model_name": "UACC-257",
    },
    {
        "root": "C18_C20",
        "reference_key": "C18_C20-CTX12",
        "reference_label": "Colorectal core subtype",
        "model_id": "cellosaurus:CVCL_1718",
        "model_name": "SW1463",
    },
]

CHROMS = [str(index) for index in range(1, 23)]
CHR_LENGTHS = {
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
}
CHR_OFFSETS: dict[str, int] = {}
_offset = 0
for _chrom in CHROMS:
    CHR_OFFSETS[_chrom] = _offset
    _offset += CHR_LENGTHS[_chrom]
GENOME_LENGTH = float(_offset)
CHR_MIDS = {
    chrom: (CHR_OFFSETS[chrom] + CHR_LENGTHS[chrom] / 2.0) / GENOME_LENGTH
    for chrom in CHROMS
}
CHR_BOUNDS = [CHR_OFFSETS[chrom] / GENOME_LENGTH for chrom in CHROMS] + [1.0]

GENERATED_OUTPUT_NAMES = {
    "A_projection_performance.png",
    "A_projection_performance.tsv",
    "B_experimental_model_coverage.png",
    "B_experimental_model_coverage.tsv",
    "C_tumour_model_concordance.png",
    "C_tumour_model_concordance.tsv",
    "D_melanoma_use_case.png",
    "D_melanoma_models.tsv",
    "D_melanoma_dependencies.tsv",
    "D_melanoma_drugs.tsv",
}

STALE_OUTPUT_NAMES = {
    "C_melanoma_use_case.png",
    "C_melanoma_models.tsv",
    "C_melanoma_dependencies.tsv",
    "C_melanoma_drugs.tsv",
    "D_colorectal_use_case.png",
    "D_colorectal_models.tsv",
    "D_colorectal_dependencies.tsv",
    "D_colorectal_drugs.tsv",
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


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )


def _style_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#91A0A6")
    axis.spines["bottom"].set_color("#91A0A6")
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _context_number(reference_key: str) -> int:
    match = re.search(r"(?:CTX|COND)(\d+)$", str(reference_key))
    return int(match.group(1)) if match else 10**6


def _read_metadata(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = [column for column in META_COLUMNS if column in header.columns]
    if "depmap_id" not in available:
        raise KeyError(f"DepMap file has no depmap_id column: {path}")
    return pd.read_csv(path, usecols=available, low_memory=False)


def _read_selected_rows(path: Path, depmap_ids: set[str]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=128, low_memory=False):
        keep = chunk["depmap_id"].astype(str).isin(depmap_ids)
        if keep.any():
            selected.append(chunk.loc[keep].copy())
    if not selected:
        raise ValueError(f"No requested DepMap models were found in {path}")
    return pd.concat(selected, ignore_index=True)


def _prepare_recommended_models(
    application_dir: Path,
    crosswalk_path: Path,
) -> pd.DataFrame:
    candidates = _read_tsv(
        application_dir / "context_model_candidates.tsv",
        {
            "model_id",
            "model_label",
            "root",
            "assigned_reference_key",
            "model_evidence_status",
            "median_distance_to_limit_ratio",
            "catalog_eligible",
            "is_recommended_model",
        },
        "experimental-model candidates",
    )
    candidates = candidates.loc[
        _as_bool(candidates["catalog_eligible"])
        & _as_bool(candidates["is_recommended_model"])
    ].copy()
    crosswalk = _read_tsv(
        crosswalk_path,
        {"model_id", "depmap_match_status"},
        "CNVSIM-DepMap crosswalk",
    )
    if "depmap_model_ids" not in crosswalk.columns:
        if "depmap_model_id" not in crosswalk.columns:
            raise KeyError(
                "CNVSIM-DepMap crosswalk requires depmap_model_ids or depmap_model_id"
            )
        crosswalk = crosswalk.rename(columns={"depmap_model_id": "depmap_model_ids"})
    merge_keys = ["model_id", "root"] if "root" in crosswalk.columns else ["model_id"]
    mapping_columns = merge_keys + ["depmap_match_status", "depmap_model_ids"]
    duplicated = crosswalk.duplicated(merge_keys, keep=False)
    if duplicated.any():
        duplicate_rows = crosswalk.loc[duplicated].copy()
        if duplicate_rows["depmap_match_status"].astype(str).eq("matched_unique").any():
            raise ValueError(f"DepMap crosswalk has duplicate unique-match keys: {merge_keys}")
        crosswalk.loc[duplicated, "depmap_model_ids"] = pd.NA
        crosswalk = crosswalk.drop_duplicates(merge_keys, keep="first")
    merged = candidates.merge(
        crosswalk[mapping_columns],
        on=merge_keys,
        how="left",
        validate="one_to_one" if "root" in merge_keys else "many_to_one",
    )
    merged["depmap_id"] = merged["depmap_model_ids"].astype("string")
    merged["depmap_eligible"] = (
        merged["depmap_match_status"].astype(str).eq("matched_unique")
        & merged["depmap_id"].notna()
    )
    return merged


def draw_projection_performance(
    projection_metrics: pd.DataFrame,
    out_path: Path,
    data_path: Path,
) -> None:
    global_row = projection_metrics.loc[
        projection_metrics["metric_scope"].astype(str).eq("global")
    ]
    root_rows = projection_metrics.loc[
        projection_metrics["metric_scope"].astype(str).eq("root")
    ].copy()
    if len(global_row) != 1 or root_rows.empty:
        raise ValueError("Projection metrics require one global and root rows")
    global_row = global_row.iloc[0]
    root_order = [root for root in ROOT_LABELS if root in set(root_rows["root"])]
    root_rows["root_order"] = pd.Categorical(
        root_rows["root"], categories=root_order, ordered=True
    )
    root_rows = root_rows.sort_values("root_order").copy()
    root_rows["root_label"] = root_rows["root"].map(ROOT_LABELS)
    root_rows["assignment_coverage_percent"] = (
        pd.to_numeric(root_rows["assignment_coverage"], errors="coerce") * 100
    )
    root_rows["accuracy_among_assigned_percent"] = (
        pd.to_numeric(root_rows["conditional_accuracy"], errors="coerce") * 100
    )
    root_rows[
        [
            "root",
            "root_label",
            "assignment_coverage_percent",
            "accuracy_among_assigned_percent",
        ]
    ].to_csv(data_path, sep="\t", index=False)

    definitions = [
        (
            "Samples assigned (%)",
            "assignment_coverage_percent",
            (80.0, 92.0),
            "All samples",
        ),
        (
            "Correct among assigned (%)",
            "accuracy_among_assigned_percent",
            (99.5, 100.05),
            "Assigned samples",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(4.25, 4.45),
        sharey=True,
        gridspec_kw={"wspace": 0.20, "width_ratios": [1.0, 1.0]},
    )
    y = np.arange(len(root_rows))[::-1]
    for axis, (label, column, limits, overall_label) in zip(axes, definitions):
        values = root_rows[column].to_numpy(dtype=float)
        overall_column = (
            "assignment_coverage"
            if column == "assignment_coverage_percent"
            else "conditional_accuracy"
        )
        overall = float(global_row[overall_column]) * 100
        axis.scatter(
            values,
            y,
            s=19,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.45,
            zorder=3,
        )
        axis.axvline(
            overall,
            color=ORANGE,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.set_xlim(*limits)
        axis.set_xlabel(label)
        axis.text(
            0.98,
            1.015,
            f"{overall_label}: {overall:.2f}%",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=ORANGE,
            fontsize=PAPER_SMALL_TEXT_SIZE,
            fontweight="bold",
        )
        axis.tick_params(axis="y", length=0)
        _style_axis(axis)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(root_rows["root_label"])
    axes[1].tick_params(labelleft=False)
    axes[0].set_xticks([80, 84, 88, 92])
    axes[1].set_xticks([99.5, 99.75, 100.0])
    fig.subplots_adjust(left=0.34, right=0.98, top=0.93, bottom=0.12)
    _save(fig, out_path)


def draw_model_coverage(
    availability: pd.DataFrame,
    out_path: Path,
    data_path: Path,
) -> None:
    work = availability.loc[
        availability["reference_level"].astype(str).isin(
            {"subtype", "conditional"}
        )
    ].copy()
    work["context_number"] = work["reference_key"].map(_context_number)
    root_order = [root for root in ROOT_LABELS if root in set(work["root"])]
    work["root_order"] = pd.Categorical(
        work["root"], categories=root_order, ordered=True
    )
    work = work.sort_values(
        ["root_order", "reference_level", "context_number", "reference_key"]
    ).copy()
    status_map = {
        "recommended_model_available": "Recommended model",
        "provisional_model_only": "No recommended model",
        "not_covered": "No recommended model",
    }
    work["coverage_plot"] = (
        work["recommended_model_coverage_status"]
        .astype(str)
        .map(status_map)
        .fillna("No recommended model")
    )
    work.to_csv(data_path, sep="\t", index=False)

    grouped = {root: group for root, group in work.groupby("root", sort=False)}
    maximum = max(len(group) for group in grouped.values())
    fig, axis = plt.subplots(figsize=(3.8, 4.45))
    colors = {
        "Recommended model": BLUE,
        "No recommended model": LIGHT_GRAY,
    }
    tile = 0.72
    for row_index, root in enumerate(root_order):
        group = grouped[root].reset_index(drop=True)
        y = len(root_order) - 1 - row_index
        for column_index, record in group.iterrows():
            edge = ORANGE if record["reference_level"] == "conditional" else WHITE
            linewidth = 1.35 if record["reference_level"] == "conditional" else 0.45
            axis.add_patch(
                Rectangle(
                    (column_index - tile / 2, y - tile / 2),
                    tile,
                    tile,
                    facecolor=colors[record["coverage_plot"]],
                    edgecolor=edge,
                    linewidth=linewidth,
                )
            )
    axis.set_xlim(-0.65, maximum - 0.15)
    axis.set_ylim(-0.75, len(root_order) - 0.25)
    axis.set_yticks(np.arange(len(root_order))[::-1])
    axis.set_yticklabels([ROOT_LABELS[root] for root in root_order])
    axis.set_xticks([0, 4, 9, 14, 18])
    axis.set_xticklabels(["1", "5", "10", "15", "19"])
    axis.set_xlabel("Reference subtypes by root")
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor=WHITE, label="Recommended"),
            Patch(
                facecolor=LIGHT_GRAY,
                edgecolor=WHITE,
                label="No recommended model",
            ),
            Patch(facecolor=WHITE, edgecolor=ORANGE, linewidth=1.4, label="Conditional"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=PAPER_SMALL_TEXT_SIZE,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.33, right=0.98, top=0.94, bottom=0.18)
    _save(fig, out_path)


def _clean_chromosome(value: object) -> str:
    return str(value).strip().replace("chr", "").replace("CHR", "")


def _gene_coordinates(cellline_profiles: pd.DataFrame) -> pd.DataFrame:
    genes = cellline_profiles[
        ["gene_symbol", "chrom", "start", "end"]
    ].dropna().drop_duplicates("gene_symbol")
    genes["chrom"] = genes["chrom"].map(_clean_chromosome)
    genes = genes.loc[genes["chrom"].isin(CHROMS)].copy()
    genes["start"] = pd.to_numeric(genes["start"], errors="coerce")
    genes["end"] = pd.to_numeric(genes["end"], errors="coerce")
    genes = genes.dropna(subset=["start", "end"])
    genes["midpoint"] = (genes["start"] + genes["end"]) / 2.0
    genes["genome_position"] = genes.apply(
        lambda row: (
            CHR_OFFSETS[str(row["chrom"])] + float(row["midpoint"])
        )
        / GENOME_LENGTH,
        axis=1,
    )
    return (
        genes.rename(columns={"gene_symbol": "gene"})[
            ["gene", "chrom", "start", "end", "genome_position"]
        ]
        .sort_values("genome_position")
        .reset_index(drop=True)
    )


def _definition_member_profile(
    reference_features_dir: Path,
    assignments: pd.DataFrame,
    genes: pd.DataFrame,
    root: str,
    reference_key: str,
    *,
    min_mean_intensity: float = 0.05,
) -> tuple[pd.DataFrame, int]:
    member_statuses = {"anchor_member", "conditional_definition_member"}
    member_ids = (
        assignments.loc[
            assignments["root"].astype(str).eq(root)
            & assignments["assigned_reference_key"].astype(str).eq(reference_key)
            & assignments["pipeline_status"].astype(str).isin(member_statuses),
            "analysis_id",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not member_ids:
        raise ValueError(f"No definition members found for {reference_key}")

    feature_path = reference_features_dir / f"{root}_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    matrix = pd.read_parquet(feature_path)
    matrix.index = matrix.index.astype(str)
    selected = matrix.reindex(member_ids).dropna(how="all")
    if selected.empty:
        raise ValueError(
            f"No {reference_key} definition members were found in {feature_path}"
        )

    profile = genes.copy()
    for event in ["gain", "loss"]:
        suffix = f"__{event}"
        columns = [column for column in selected.columns if str(column).endswith(suffix)]
        if not columns:
            raise ValueError(f"{feature_path} contains no {suffix} features")
        intensity = selected[columns].clip(lower=0.0, upper=3.0).mean(axis=0) / 3.0
        values = intensity.rename("value").rename_axis("feature").reset_index()
        values["gene"] = values["feature"].astype(str).str.removesuffix(suffix)
        values[event] = pd.to_numeric(values["value"], errors="coerce").fillna(0.0)
        values.loc[values[event] < min_mean_intensity, event] = 0.0
        profile = profile.merge(values[["gene", event]], on="gene", how="left")
    profile[["gain", "loss"]] = profile[["gain", "loss"]].fillna(0.0)
    return profile, int(len(selected))


def _model_gene_profile(
    cellline_profiles: pd.DataFrame,
    genes: pd.DataFrame,
    analysis_id: str,
) -> pd.DataFrame:
    rows = cellline_profiles.loc[
        cellline_profiles["analysis_id"].astype(str).eq(analysis_id)
    ].copy()
    if rows.empty:
        raise ValueError(f"No cell-line gene profile for {analysis_id}")
    for column in ["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    rows["gain"] = (
        (rows["dup_frac"] >= 0.10).astype(float)
        + 2.0 * (rows["hldup_frac"] >= 0.20).astype(float)
    ) / 3.0
    rows["loss"] = (
        (rows["del_frac"] >= 0.10).astype(float)
        + 2.0 * (rows["hldel_frac"] >= 0.20).astype(float)
    ) / 3.0
    values = (
        rows.groupby("gene_symbol", as_index=False)
        .agg(gain=("gain", "max"), loss=("loss", "max"))
        .rename(columns={"gene_symbol": "gene"})
    )
    return genes.merge(values, on="gene", how="left").fillna(
        {"gain": 0.0, "loss": 0.0}
    )


def _profile_bins(
    profile: pd.DataFrame,
    event: str,
    *,
    n_bins: int = 1100,
) -> np.ndarray:
    values = np.zeros(n_bins, dtype=float)
    work = profile[["genome_position", event]].copy()
    work["bin"] = (
        np.floor(work["genome_position"] * n_bins)
        .astype(int)
        .clip(0, n_bins - 1)
    )
    maxima = work.groupby("bin")[event].max()
    values[maxima.index.to_numpy(dtype=int)] = maxima.to_numpy(dtype=float)
    return values


def _blend_profile_color(color: str, strength: float) -> tuple[float, float, float]:
    neutral = np.array(to_rgb(WHITE))
    target = np.array(to_rgb(color))
    weight = 0.24 + 0.76 * float(np.clip(strength, 0.0, 1.0))
    return tuple(neutral * (1.0 - weight) + target * weight)


def _profile_image(values: np.ndarray, color: str) -> np.ndarray:
    image = np.ones((1, len(values), 3), dtype=float)
    for index, value in enumerate(values):
        if value > 0:
            image[0, index, :] = _blend_profile_color(color, float(value))
    return image


def _draw_profile_strip(
    axis: plt.Axes,
    profile: pd.DataFrame,
    *,
    y: float,
    label: str,
) -> None:
    gain = _profile_bins(profile, "gain")
    loss = _profile_bins(profile, "loss")
    axis.imshow(
        _profile_image(gain, GAIN),
        extent=(0.17, 1.0, y + 0.42, y + 0.78),
        aspect="auto",
        interpolation="nearest",
    )
    axis.imshow(
        _profile_image(loss, LOSS),
        extent=(0.17, 1.0, y, y + 0.36),
        aspect="auto",
        interpolation="nearest",
    )
    axis.add_patch(
        Rectangle(
            (0.17, y),
            0.83,
            0.78,
            fill=False,
            edgecolor="#BFB7A8",
            linewidth=0.55,
        )
    )
    axis.plot([0.17, 1.0], [y + 0.39, y + 0.39], color=PROFILE_GRID, linewidth=0.35)
    for boundary in CHR_BOUNDS:
        x = 0.17 + 0.83 * boundary
        axis.plot([x, x], [y, y + 0.78], color=PROFILE_GRID, linewidth=0.25)
    axis.text(0.155, y + 0.39, label, ha="right", va="center", fontsize=6.4)


def draw_tumour_model_concordance(
    assignments: pd.DataFrame,
    reference_features_dir: Path,
    models: pd.DataFrame,
    cellline_profiles: pd.DataFrame,
    out_path: Path,
    data_path: Path,
) -> None:
    genes = _gene_coordinates(cellline_profiles)
    records: list[dict[str, object]] = []
    figure, axes = plt.subplots(
        len(PROFILE_CASES),
        1,
        figsize=(7.1, 3.25),
    )
    axes = np.atleast_1d(axes)
    for axis, spec in zip(axes, PROFILE_CASES):
        model = _select_profile_case_model(models, spec)
        atlas_profile, n_definition_members = _definition_member_profile(
            reference_features_dir,
            assignments,
            genes,
            str(spec["root"]),
            str(spec["reference_key"]),
        )
        model_profile = _model_gene_profile(
            cellline_profiles,
            genes,
            str(model["representative_analysis_id"]),
        )
        axis.set_xlim(0, 1.0)
        axis.set_ylim(-0.12, 2.35)
        axis.axis("off")
        if str(model["reference_level"]) == "conditional":
            axis.add_patch(
                Rectangle(
                    (0.0, -0.08),
                    1.0,
                    2.35,
                    facecolor="#FFF4EC",
                    edgecolor="none",
                    zorder=-5,
                )
            )
        axis.text(
            0.0,
            2.22,
            f"{spec['reference_key']}  |  {spec['reference_label']}",
            ha="left",
            va="top",
            fontsize=7.4,
            fontweight="bold",
        )
        accession = str(model["model_id"]).replace("cellosaurus:", "")
        axis.text(
            1.0,
            2.22,
            f"{spec['model_name']} ({accession})",
            ha="right",
            va="top",
            fontsize=6.2,
            color=MUTED,
        )
        _draw_profile_strip(axis, atlas_profile, y=1.18, label="Tumour reference")
        _draw_profile_strip(axis, model_profile, y=0.20, label="Cell-line model")
        for chrom in CHROMS:
            if int(chrom) % 2 == 1 or chrom == "22":
                x = 0.17 + 0.83 * CHR_MIDS[chrom]
                axis.text(x, 0.12, chrom, ha="center", va="top", fontsize=4.7, color=MUTED)
        records.append(
            {
                "root": spec["root"],
                "reference_key": spec["reference_key"],
                "reference_label": spec["reference_label"],
                "model_id": model["model_id"],
                "model_name": spec["model_name"],
                "representative_analysis_id": model["representative_analysis_id"],
                "reference_level": model["reference_level"],
                "model_evidence_status": model["model_evidence_status"],
                "n_definition_members": n_definition_members,
                "reference_profile_source": "mean_encoded_gain_loss_intensity",
                "distance_to_calibration_limit": model[
                    "median_distance_to_limit_ratio"
                ],
            }
        )
    figure.legend(
        handles=[
            Patch(facecolor=GAIN, edgecolor="none", label="Gain"),
            Patch(facecolor=LOSS, edgecolor="none", label="Loss"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.006),
        ncol=2,
        fontsize=PAPER_SMALL_TEXT_SIZE,
        handlelength=1.2,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.12, hspace=0.10)
    pd.DataFrame(records).to_csv(data_path, sep="\t", index=False)
    _save(figure, out_path)


def _select_profile_case_model(
    models: pd.DataFrame,
    spec: dict[str, object],
) -> pd.Series:
    selected = models.loc[
        models["model_id"].astype(str).eq(str(spec["model_id"]))
        & models["assigned_reference_key"].astype(str).eq(
            str(spec["reference_key"])
        )
        & models["catalog_eligible"]
        & models["is_recommended_model"]
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one recommended model for {spec['reference_key']} / "
            f"{spec['model_id']}; found {len(selected)}"
        )
    return selected.iloc[0]


def _load_profile_case_gene_panels(
    models: pd.DataFrame,
    gene_panel_cache: Path,
) -> pd.DataFrame:
    analysis_ids = sorted(
        {
            str(_select_profile_case_model(models, spec)["representative_analysis_id"])
            for spec in PROFILE_CASES
        }
    )
    profiles = load_gene_panel_subset_from_cache(
        cache_dir=gene_panel_cache,
        analysis_ids=analysis_ids,
        columns=[
            "analysis_id",
            "gene_symbol",
            "chrom",
            "start",
            "end",
            *EVENT_COLUMNS,
        ],
    )
    observed = set(profiles.get("analysis_id", pd.Series(dtype=str)).astype(str))
    missing = sorted(set(analysis_ids) - observed)
    if missing:
        raise ValueError(
            "Current gene-panel cache is missing Figure 7 model profiles: "
            f"{missing}"
        )
    return profiles


def _functional_summary(
    matrix: pd.DataFrame,
    target_ids: set[str],
    background_ids: set[str],
    *,
    kind: str,
) -> pd.DataFrame:
    features = [column for column in matrix.columns if column not in META_COLUMNS]
    target = matrix.loc[matrix["depmap_id"].astype(str).isin(target_ids), features]
    background = matrix.loc[
        matrix["depmap_id"].astype(str).isin(background_ids), features
    ]
    target = target.apply(pd.to_numeric, errors="coerce")
    background = background.apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "feature": features,
            "target_n": target.notna().sum(axis=0).to_numpy(),
            "background_n": background.notna().sum(axis=0).to_numpy(),
            "target_median": target.median(axis=0, skipna=True).to_numpy(),
            "background_median": background.median(axis=0, skipna=True).to_numpy(),
        }
    )
    if kind == "dependency":
        summary["effect"] = (
            summary["target_median"] - summary["background_median"]
        )
        summary["target_fraction_dependency_ge_0_5"] = (
            (target >= 0.5).sum(axis=0).to_numpy()
            / target.notna().sum(axis=0).replace(0, np.nan).to_numpy()
        )
    elif kind == "drug":
        summary["effect"] = (
            summary["background_median"] - summary["target_median"]
        )
        summary["drug_name"] = summary["feature"].map(
            lambda value: str(value).split(" (BRD:", 1)[0]
        )
    else:
        raise ValueError(f"Unknown functional summary kind: {kind}")
    summary = summary.sort_values("effect", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def _case_models(
    models: pd.DataFrame,
    crispr_metadata: pd.DataFrame,
    prism_metadata: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    root = str(spec["root"])
    reference_key = str(spec["reference_key"])
    assay_ids = set(crispr_metadata["depmap_id"].astype(str)) & set(
        prism_metadata["depmap_id"].astype(str)
    )
    eligible = models.loc[
        models["root"].astype(str).eq(root)
        & models["depmap_eligible"]
        & models["depmap_id"].astype(str).isin(assay_ids)
    ].copy()
    eligible = eligible.sort_values(
        ["assigned_reference_key", "median_distance_to_limit_ratio", "model_id"]
    ).drop_duplicates("depmap_id", keep="first")
    display = crispr_metadata[["depmap_id", "cell_line_display_name"]].drop_duplicates(
        "depmap_id"
    )
    eligible = eligible.merge(display, on="depmap_id", how="left", validate="one_to_one")
    target = eligible.loc[
        eligible["assigned_reference_key"].astype(str).eq(reference_key)
    ].copy()
    background = eligible.loc[
        ~eligible["assigned_reference_key"].astype(str).eq(reference_key)
    ].copy()
    if len(target) < 3:
        raise ValueError(f"{reference_key} has fewer than three models with both assays")
    if len(background) < 3:
        raise ValueError(f"{root} has fewer than three same-root comparator models")
    return target, set(target["depmap_id"].astype(str)), set(
        background["depmap_id"].astype(str)
    )


def _draw_model_match(axis: plt.Axes, target: pd.DataFrame) -> None:
    work = target.sort_values("median_distance_to_limit_ratio", ascending=False).copy()
    y = np.arange(len(work))
    for index, (_, record) in enumerate(work.iterrows()):
        axis.scatter(
            [record["median_distance_to_limit_ratio"]],
            [index],
            marker="o",
            s=28,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        axis.plot(
            [0, record["median_distance_to_limit_ratio"]],
            [index, index],
            color=GRID,
            linewidth=1.0,
            zorder=1,
        )
    labels: list[str] = []
    for _, record in work.iterrows():
        catalog_label = str(record.get("model_label", ""))
        match = re.search(r"\[cell line ([^\]]+)\]", catalog_label, flags=re.IGNORECASE)
        if match:
            model_name = match.group(1)
        elif catalog_label and ";" not in catalog_label and catalog_label.lower() != "nan":
            model_name = catalog_label
        else:
            model_name = str(record.get("cell_line_display_name", "model"))
        accession = str(record["model_id"]).replace("cellosaurus:", "")
        labels.append(f"{model_name}\n{accession}")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=5.2, linespacing=0.88)
    axis.axvline(1.0, color=ORANGE, linestyle="--", linewidth=1.0)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Fraction of maximum accepted distance\n(lower = closer)")
    axis.set_title("CNV-subtype similarity", fontweight="bold", pad=7)
    axis.tick_params(axis="y", length=0)
    _style_axis(axis)


def _draw_dependencies(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    target_ids: set[str],
    background_ids: set[str],
    summary: pd.DataFrame,
    dependency_anchors: list[str],
) -> None:
    minimum_target = max(3, math.ceil(len(target_ids) * 0.6))
    eligible = summary.loc[
        (summary["target_n"] >= minimum_target)
        & (summary["background_n"] >= 3)
        & (summary["target_median"] >= 0.45)
        & (summary["target_fraction_dependency_ge_0_5"] >= 0.5)
        & (summary["effect"] > 0)
    ].copy()
    selected = eligible.head(5).copy()
    if selected.empty:
        selected = summary.loc[
            (summary["target_n"] >= minimum_target)
            & (summary["background_n"] >= 3)
        ].head(5)
    anchors = eligible.loc[
        eligible["feature"].astype(str).isin(dependency_anchors)
    ]
    for _, anchor in anchors.iterrows():
        if anchor["feature"] in set(selected["feature"]):
            continue
        if len(selected) >= 5:
            selected = selected.iloc[:-1]
        selected = pd.concat([selected, anchor.to_frame().T], ignore_index=True)
    features = selected["feature"].tolist()[::-1]
    y = np.arange(len(features))
    rng = np.random.default_rng(11)
    for position, feature in zip(y, features):
        target_values = pd.to_numeric(
            matrix.loc[matrix["depmap_id"].astype(str).isin(target_ids), feature],
            errors="coerce",
        ).dropna()
        background_values = pd.to_numeric(
            matrix.loc[
                matrix["depmap_id"].astype(str).isin(background_ids), feature
            ],
            errors="coerce",
        ).dropna()
        axis.scatter(
            background_values,
            np.full(len(background_values), position)
            + rng.uniform(-0.10, 0.10, len(background_values)),
            s=8,
            color=MID_GRAY,
            alpha=0.48,
            linewidth=0,
            zorder=1,
        )
        axis.scatter(
            target_values,
            np.full(len(target_values), position)
            + rng.uniform(-0.10, 0.10, len(target_values)),
            s=17,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=2,
        )
        axis.plot(
            [background_values.median(), target_values.median()],
            [position, position],
            color=MUTED,
            linewidth=1.0,
            zorder=1,
        )
    axis.set_yticks(y)
    axis.set_yticklabels(features, fontstyle="italic")
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("Dependency probability")
    axis.set_title("CRISPR gene dependencies", fontweight="bold", pad=7)
    axis.tick_params(axis="y", length=0)
    _style_axis(axis)


def _draw_drug_response(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    target_ids: set[str],
    background_ids: set[str],
    summary: pd.DataFrame,
    pathway_drugs: list[str],
) -> None:
    eligible = summary.loc[
        (summary["target_n"] >= 3) & (summary["background_n"] >= 3)
    ].copy()
    requested = {name.upper() for name in pathway_drugs}
    selected = (
        eligible.loc[eligible["drug_name"].astype(str).str.upper().isin(requested)]
        .sort_values("effect", ascending=False)
        .head(6)
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("No requested PRISM pathway compounds have sufficient coverage")

    rng = np.random.default_rng(17)
    for position, (_, record) in enumerate(selected.iterrows()):
        feature = str(record["feature"])
        target_values = pd.to_numeric(
            matrix.loc[matrix["depmap_id"].astype(str).isin(target_ids), feature],
            errors="coerce",
        ).dropna()
        background_values = pd.to_numeric(
            matrix.loc[
                matrix["depmap_id"].astype(str).isin(background_ids), feature
            ],
            errors="coerce",
        ).dropna()
        axis.scatter(
            background_values,
            np.full(len(background_values), position)
            + rng.uniform(-0.10, 0.10, len(background_values)),
            s=8,
            color=MID_GRAY,
            alpha=0.52,
            linewidth=0,
            zorder=1,
        )
        axis.scatter(
            target_values,
            np.full(len(target_values), position)
            + rng.uniform(-0.10, 0.10, len(target_values)),
            s=17,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=2,
        )
        axis.plot(
            [target_values.median(), background_values.median()],
            [position, position],
            color=MUTED,
            linewidth=1.0,
            zorder=1,
        )
    axis.set_yticks(np.arange(len(selected)))
    axis.set_yticklabels(selected["drug_name"].astype(str).str.lower())
    axis.invert_yaxis()
    axis.set_xlabel("PRISM AUC (lower = more sensitive)")
    axis.set_title("PRISM drug sensitivity", fontweight="bold", pad=7)
    axis.tick_params(axis="y", length=0)
    axis.margins(x=0.06)
    _style_axis(axis)


def draw_use_case(
    target: pd.DataFrame,
    target_ids: set[str],
    background_ids: set[str],
    dependency_matrix: pd.DataFrame,
    drug_matrix: pd.DataFrame,
    dependency_summary: pd.DataFrame,
    drug_summary: pd.DataFrame,
    spec: dict[str, object],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.1, 3.45))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.08, 1.12, 1.55],
        wspace=0.48,
    )
    model_axis = fig.add_subplot(grid[0, 0])
    dependency_axis = fig.add_subplot(grid[0, 1])
    drug_axis = fig.add_subplot(grid[0, 2])
    _draw_model_match(model_axis, target)
    _draw_dependencies(
        dependency_axis,
        dependency_matrix,
        target_ids,
        background_ids,
        dependency_summary,
        list(spec["dependency_anchors"]),
    )
    _draw_drug_response(
        drug_axis,
        drug_matrix,
        target_ids,
        background_ids,
        drug_summary,
        list(spec["pathway_drugs"]),
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=BLUE,
                markeredgecolor=WHITE,
                markersize=5,
                label=f"C44-CTX5 models (n={len(target_ids)})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=MID_GRAY,
                markeredgecolor="none",
                markersize=4,
                label=f"Other C44 models (n={len(background_ids)})",
            ),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        fontsize=PAPER_SMALL_TEXT_SIZE,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(top=0.91, bottom=0.23, left=0.10, right=0.98)
    _save(fig, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the projection and model-selection panels for Figure 7."
    )
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--application-dir", required=True, type=Path)
    parser.add_argument("--depmap-crosswalk", required=True, type=Path)
    parser.add_argument("--crispr-dependency", required=True, type=Path)
    parser.add_argument("--prism-auc", required=True, type=Path)
    parser.add_argument(
        "--gene-panel-cache",
        required=True,
        type=Path,
        help=(
            "Current partitioned gene-panel cache used by experimental-model "
            "projection. Only Figure 7 representative analyses are loaded."
        ),
    )
    parser.add_argument(
        "--reference-features-dir",
        required=True,
        type=Path,
        help="Directory containing per-root *_features.parquet matrices.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    application_dir = args.application_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and not args.overwrite:
        occupied = [out_dir / name for name in GENERATED_OUTPUT_NAMES if (out_dir / name).exists()]
        if occupied:
            raise FileExistsError(
                f"Figure 7 output exists; pass --overwrite: {occupied[0]}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in GENERATED_OUTPUT_NAMES | STALE_OUTPUT_NAMES:
            path = out_dir / name
            if path.exists():
                path.unlink()

    projection_metrics = _read_tsv(
        release_dir / "validation" / "holdout_v1" / "projection_metrics.tsv",
        {
            "metric_scope",
            "raw_top1_accuracy",
            "assignment_coverage",
            "conditional_accuracy",
            "incorrect_assignment_fraction",
        },
        "projection metrics",
    )
    availability = _read_tsv(
        application_dir / "context_model_availability.tsv",
        {
            "reference_key",
            "root",
            "reference_level",
            "recommended_model_coverage_status",
        },
        "experimental-model availability",
    )
    assignment_path = release_dir / "tables" / "analysis_assignments.parquet"
    if not assignment_path.exists():
        raise FileNotFoundError(assignment_path)
    assignments = pd.read_parquet(
        assignment_path,
        columns=[
            "analysis_id",
            "root",
            "pipeline_status",
            "assigned_reference_key",
        ],
    )
    reference_features_dir = args.reference_features_dir.expanduser().resolve()
    if not reference_features_dir.is_dir():
        raise NotADirectoryError(reference_features_dir)
    models = _prepare_recommended_models(
        application_dir,
        args.depmap_crosswalk.expanduser().resolve(),
    )
    gene_panel_cache = args.gene_panel_cache.expanduser().resolve()
    if not gene_panel_cache.is_dir():
        raise NotADirectoryError(gene_panel_cache)
    cellline_profiles = _load_profile_case_gene_panels(
        models,
        gene_panel_cache,
    )
    crispr_path = args.crispr_dependency.expanduser().resolve()
    prism_path = args.prism_auc.expanduser().resolve()
    crispr_metadata = _read_metadata(crispr_path)
    prism_metadata = _read_metadata(prism_path)

    case_data: dict[str, tuple[pd.DataFrame, set[str], set[str]]] = {}
    selected_ids: set[str] = set()
    for label, spec in FUNCTIONAL_CASE.items():
        case_data[label] = _case_models(
            models,
            crispr_metadata,
            prism_metadata,
            spec,
        )
        selected_ids |= case_data[label][1] | case_data[label][2]
    print(
        f"[CNVSIM] reading DepMap functional rows for {len(selected_ids)} models"
    )
    dependency_matrix = _read_selected_rows(crispr_path, selected_ids)
    drug_matrix = _read_selected_rows(prism_path, selected_ids)

    draw_projection_performance(
        projection_metrics,
        out_dir / "A_projection_performance.png",
        out_dir / "A_projection_performance.tsv",
    )
    draw_model_coverage(
        availability,
        out_dir / "B_experimental_model_coverage.png",
        out_dir / "B_experimental_model_coverage.tsv",
    )
    draw_tumour_model_concordance(
        assignments,
        reference_features_dir,
        models,
        cellline_profiles,
        out_dir / "C_tumour_model_concordance.png",
        out_dir / "C_tumour_model_concordance.tsv",
    )

    for panel_label, spec in FUNCTIONAL_CASE.items():
        target, target_ids, background_ids = case_data[panel_label]
        dependency_summary = _functional_summary(
            dependency_matrix,
            target_ids,
            background_ids,
            kind="dependency",
        )
        drug_summary = _functional_summary(
            drug_matrix,
            target_ids,
            background_ids,
            kind="drug",
        )
        slug = str(spec["slug"])
        target.to_csv(
            out_dir / f"{panel_label}_{slug}_models.tsv",
            sep="\t",
            index=False,
        )
        dependency_summary.to_csv(
            out_dir / f"{panel_label}_{slug}_dependencies.tsv",
            sep="\t",
            index=False,
        )
        drug_summary.to_csv(
            out_dir / f"{panel_label}_{slug}_drugs.tsv",
            sep="\t",
            index=False,
        )
        draw_use_case(
            target,
            target_ids,
            background_ids,
            dependency_matrix,
            drug_matrix,
            dependency_summary,
            drug_summary,
            spec,
            out_dir / f"{panel_label}_{slug}_use_case.png",
        )

    print("[CNVSIM] Figure 7 projection and model-selection panels complete")
    for name in sorted(GENERATED_OUTPUT_NAMES):
        print(f"  {name}: {out_dir / name}")


if __name__ == "__main__":
    main()
