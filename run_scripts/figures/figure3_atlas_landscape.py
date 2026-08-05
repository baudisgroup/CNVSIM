#!/usr/bin/env python3
from __future__ import annotations

"""Render the three final Figure 3 panels and its Mermaid workflow."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


TEXT = "#111111"
MUTED = "#6B6F72"
GRID = "#E5E5E5"
CORE = "#2E5EAA"
CONDITIONAL = "#D6872C"
NOT_SELECTED = "#C8C8C8"
WHITE = "#FFFFFF"
PAPER_COLUMN_WIDTH_IN = 3.55
PAPER_TALL_PANEL_HEIGHT_IN = 4.25
PAPER_TEXT_SIZE = 7.0
PAPER_SMALL_TEXT_SIZE = 6.2

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

GENERATED_OUTPUT_NAMES = {
    "A_context_stability_heatmap.png",
    "A_selection_workflow.mmd",
    "B_reference_coverage.png",
    "C_root_diagnostic_heterogeneity.png",
    # Legacy Figure 3 outputs removed when --overwrite is used.
    "A_context_stability.tsv",
    "A_reference_coverage.png",
    "A_reference_coverage.tsv",
    "A_selection_workflow_edges.tsv",
    "A_selection_workflow_nodes.tsv",
    "B_context_stability.png",
    "B_context_stability.tsv",
    "B_context_stability_heatmap.png",
    "B_reference_coverage.tsv",
    "B_selection_workflow.mmd",
    "B_selection_workflow_edges.tsv",
    "B_selection_workflow_nodes.tsv",
    "C_diagnostic_concentration.png",
    "C_diagnostic_concentration.tsv",
    "C_root_diagnostic_heterogeneity.tsv",
    "D_root_diagnostic_heterogeneity.png",
    "D_root_diagnostic_heterogeneity.tsv",
    "S_diagnostic_concentration.png",
    "S_diagnostic_concentration.tsv",
    "manifest.json",
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
    "C47": "C47  Peripheral nerves",
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


def _read_tsv(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")
    return frame


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype("string").str.strip().str.lower()
    invalid = normalized.notna() & ~normalized.isin(
        {"true", "false", "1", "0"}
    )
    if invalid.any():
        raise ValueError(
            "Invalid boolean values: "
            f"{sorted(normalized.loc[invalid].dropna().unique().tolist())}"
        )
    return normalized.isin({"true", "1"})


def _numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[columns].isna().all(axis=None):
        raise ValueError(f"{name} has no finite values in {columns}")


def _style_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#95A3A8")
    axis.spines["bottom"].set_color("#95A3A8")
    axis.tick_params(colors=TEXT)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _root_label(root: str) -> str:
    return ROOT_LABELS.get(str(root), str(root))


def build_reference_coverage(references: pd.DataFrame) -> pd.DataFrame:
    references = references[
        references["reference_level"].astype(str).isin(
            {"subtype", "conditional"}
        )
    ].copy()
    if references.empty:
        raise ValueError("No core or conditional references are available")
    references["root"] = references["root"].astype(str)
    references["n_definition_members"] = pd.to_numeric(
        references["n_definition_members"], errors="raise"
    ).astype(int)
    references["layer"] = references["reference_level"].map(
        {"subtype": "core", "conditional": "conditional"}
    )
    coverage = (
        references.groupby(["root", "layer"], sort=False)
        .agg(
            n_contexts=("reference_key", "nunique"),
            n_definition_members=("n_definition_members", "sum"),
        )
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [sorted(references["root"].unique()), ["core", "conditional"]],
        names=["root", "layer"],
    ).to_frame(index=False)
    coverage = complete.merge(
        coverage, on=["root", "layer"], how="left", validate="one_to_one"
    )
    coverage[["n_contexts", "n_definition_members"]] = coverage[
        ["n_contexts", "n_definition_members"]
    ].fillna(0).astype(int)
    totals = (
        coverage.groupby("root")
        .agg(
            total_contexts=("n_contexts", "sum"),
            total_definition_members=("n_definition_members", "sum"),
        )
        .reset_index()
    )
    coverage = coverage.merge(totals, on="root", validate="many_to_one")
    coverage["root_label"] = coverage["root"].map(_root_label)
    return coverage.sort_values(
        ["total_definition_members", "root", "layer"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def draw_reference_coverage(coverage: pd.DataFrame, out_path: Path) -> None:
    root_order = (
        coverage[
            [
                "root",
                "root_label",
                "total_contexts",
                "total_definition_members",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["total_definition_members", "total_contexts", "root"],
            ascending=[True, True, True],
        )
    )
    roots = root_order["root"].tolist()
    labels = root_order["root_label"].tolist()
    pivot_members = (
        coverage.pivot(
            index="root",
            columns="layer",
            values="n_definition_members",
        )
        .reindex(roots)
        .fillna(0)
    )
    pivot_contexts = (
        coverage.pivot(index="root", columns="layer", values="n_contexts")
        .reindex(roots)
        .fillna(0)
    )

    fig, axis = plt.subplots(
        figsize=(PAPER_COLUMN_WIDTH_IN, PAPER_TALL_PANEL_HEIGHT_IN)
    )
    y = np.arange(len(roots))
    core_members = pivot_members.get("core", pd.Series(0, index=roots))
    conditional_members = pivot_members.get(
        "conditional", pd.Series(0, index=roots)
    )
    core_contexts = pivot_contexts.get("core", pd.Series(0, index=roots))
    conditional_contexts = pivot_contexts.get(
        "conditional", pd.Series(0, index=roots)
    )
    axis.barh(
        y,
        core_members,
        color=CORE,
        edgecolor=WHITE,
        linewidth=0.4,
        height=0.66,
        label="Core",
    )
    axis.barh(
        y,
        conditional_members,
        left=core_members,
        color=NOT_SELECTED,
        edgecolor=WHITE,
        linewidth=0.4,
        height=0.66,
        label="Conditional",
    )
    total_members = core_members + conditional_members
    max_members = float(total_members.max())
    value_pad = max_members * 0.012
    subtype_column_x = max_members * 1.30
    for row, total in enumerate(total_members):
        axis.text(
            float(total) + value_pad,
            row,
            f"{int(total):,}",
            va="center",
            ha="left",
            fontsize=PAPER_SMALL_TEXT_SIZE,
            color=TEXT,
        )
        axis.text(
            subtype_column_x,
            row,
            (
                f"{int(core_contexts.iloc[row])} / "
                f"{int(conditional_contexts.iloc[row])}"
            ),
            va="center",
            ha="center",
            fontsize=PAPER_SMALL_TEXT_SIZE,
            color=TEXT,
        )
    axis.set_yticks(y, labels, fontsize=PAPER_SMALL_TEXT_SIZE)
    axis.set_xlabel("Number of tumour analyses in atlas")
    axis.set_xlim(0, max(max_members * 1.42, 2.0))
    axis.text(
        subtype_column_x,
        1.018,
        "Subtypes\ncore / conditional",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=PAPER_SMALL_TEXT_SIZE,
        color=TEXT,
        linespacing=1.15,
    )
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.00, 1.01),
        frameon=False,
        ncol=2,
        handlelength=1.4,
        fontsize=PAPER_TEXT_SIZE,
    )
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.xaxis.label.set_size(PAPER_TEXT_SIZE)
    axis.tick_params(axis="x", labelsize=PAPER_SMALL_TEXT_SIZE)
    _style_axis(axis)
    _save(fig, out_path)


def build_stability_map(
    stability: pd.DataFrame,
    conditional_contexts: pd.DataFrame,
) -> pd.DataFrame:
    stability = stability.copy()
    stability["root"] = stability["root"].astype(str)
    stability["final_context"] = pd.to_numeric(
        stability["final_context"], errors="raise"
    ).astype(int)
    _numeric(
        stability,
        [
            "detection_rate",
            "conditional_recall_q10",
            "conditional_precision_q10",
            "conditional_jaccard_q10",
            "n_samples",
        ],
        "context stability",
    )
    if stability.duplicated(["root", "final_context"]).any():
        raise ValueError(
            "Context stability must be unique by root/final_context"
        )
    stability["is_reference_anchor"] = _boolean_series(
        stability["is_reference_anchor"]
    )

    conditional_contexts = conditional_contexts.copy()
    conditional_contexts["root"] = conditional_contexts["root"].astype(str)
    conditional_contexts["final_context"] = pd.to_numeric(
        conditional_contexts["final_context"], errors="raise"
    ).astype(int)
    conditional_pairs = set(
        conditional_contexts[["root", "final_context"]].itertuples(
            index=False, name=None
        )
    )
    stability_pairs = set(
        stability[["root", "final_context"]].itertuples(
            index=False, name=None
        )
    )
    unknown = sorted(conditional_pairs - stability_pairs)
    if unknown:
        raise ValueError(
            f"Conditional references are absent from context stability: {unknown}"
        )
    stability["is_conditional_reference"] = [
        pair in conditional_pairs
        for pair in stability[["root", "final_context"]].itertuples(
            index=False, name=None
        )
    ]
    overlap = stability[
        stability["is_reference_anchor"]
        & stability["is_conditional_reference"]
    ]
    if not overlap.empty:
        raise ValueError("Core and conditional reference contexts overlap")
    stability["selection_class"] = np.select(
        [
            stability["is_reference_anchor"],
            stability["is_conditional_reference"],
        ],
        ["core", "conditional"],
        default="not_selected",
    )
    stability["context_key"] = [
        f"{root}-CTX{context}"
        for root, context in stability[
            ["root", "final_context"]
        ].itertuples(index=False, name=None)
    ]
    conditional_key_map = (
        conditional_contexts.set_index(["root", "final_context"])[
            "conditional_reference_key"
        ].astype(str)
        if "conditional_reference_key" in conditional_contexts.columns
        else pd.Series(dtype=str)
    )
    stability["conditional_reference_key"] = [
        conditional_key_map.get((root, context), pd.NA)
        for root, context in stability[
            ["root", "final_context"]
        ].itertuples(index=False, name=None)
    ]
    stability["conditional_jaccard_plot"] = stability[
        "conditional_jaccard_q10"
    ].fillna(0.0)
    return stability.sort_values(
        ["selection_class", "root", "final_context"]
    ).reset_index(drop=True)


def build_selection_workflow(
    stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = stability["selection_class"].value_counts()
    n_core = int(counts.get("core", 0))
    n_conditional = int(counts.get("conditional", 0))
    n_not_selected = int(counts.get("not_selected", 0))
    non_core = stability[~stability["selection_class"].eq("core")]
    recovery_gate_columns = [
        "passes_context_size",
        "passes_conditional_recall",
        "passes_conditional_precision",
        "passes_conditional_jaccard",
    ]
    if "recovery_class" in non_core.columns:
        candidate_mask = non_core["recovery_class"].astype(str).eq(
            "intermittent"
        )
    elif set(recovery_gate_columns).issubset(non_core.columns):
        candidate_mask = pd.Series(True, index=non_core.index)
        for column in recovery_gate_columns:
            candidate_mask &= _boolean_series(non_core[column])
    else:
        candidate_mask = non_core["selection_class"].eq("conditional")
    n_candidates = int(candidate_mask.sum())
    n_other_non_core = int(len(non_core) - n_candidates)
    n_candidate_failures = int(n_candidates - n_conditional)
    if n_not_selected != n_other_non_core + n_candidate_failures:
        raise ValueError(
            "Not-selected accounting does not match conditional screening: "
            f"{n_not_selected} != {n_other_non_core} + "
            f"{n_candidate_failures}"
        )

    nodes = pd.DataFrame(
        [
            {
                "node_id": "refined",
                "stage": "refined communities",
                "label": "Refined communities",
                "n_communities": int(len(stability)),
                "node_class": "neutral",
            },
            {
                "node_id": "stable",
                "stage": "discovery stability",
                "label": "Stable in discovery repeats",
                "n_communities": n_core,
                "node_class": "core",
            },
            {
                "node_id": "non_core",
                "stage": "discovery stability",
                "label": "Non-core communities",
                "n_communities": int(len(non_core)),
                "node_class": "neutral",
            },
            {
                "node_id": "conditional_candidates",
                "stage": "conditional candidate screen",
                "label": "Conditional candidates",
                "n_communities": n_candidates,
                "node_class": "conditional",
            },
            {
                "node_id": "other_non_core",
                "stage": "conditional candidate screen",
                "label": "Other non-core communities",
                "n_communities": n_other_non_core,
                "node_class": "excluded",
            },
            {
                "node_id": "core",
                "stage": "final reference role",
                "label": "Core",
                "n_communities": n_core,
                "node_class": "core",
            },
            {
                "node_id": "conditional",
                "stage": "final reference role",
                "label": "Conditional",
                "n_communities": n_conditional,
                "node_class": "conditional",
            },
            {
                "node_id": "not_selected",
                "stage": "final reference role",
                "label": "Not selected",
                "n_communities": n_not_selected,
                "node_class": "excluded",
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "refined", "target": "stable", "label": ""},
            {"source": "refined", "target": "non_core", "label": ""},
            {
                "source": "stable",
                "target": "core",
                "label": "20 discovery repeats",
            },
            {
                "source": "non_core",
                "target": "conditional_candidates",
                "label": "",
            },
            {
                "source": "non_core",
                "target": "other_non_core",
                "label": "",
            },
            {
                "source": "conditional_candidates",
                "target": "conditional",
                "label": "passed 10/10 fixed-label repeats",
            },
            {
                "source": "conditional_candidates",
                "target": "not_selected",
                "label": f"{n_candidate_failures} failed",
            },
            {
                "source": "other_non_core",
                "target": "not_selected",
                "label": str(n_other_non_core),
            },
        ]
    )
    return nodes, edges


def write_selection_workflow_mermaid(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        (
            "%%{init: {\"theme\":\"base\",\"themeVariables\":"
            "{\"fontFamily\":\"Arial, Helvetica, sans-serif\","
            "\"fontSize\":\"14px\",\"lineColor\":\"#5F6368\","
            "\"edgeLabelBackground\":\"#FFFFFF\"},"
            "\"flowchart\":{\"htmlLabels\":true,\"curve\":\"linear\","
            "\"nodeSpacing\":14,\"rankSpacing\":22}}}%%"
        ),
        "flowchart TB",
        (
            "  classDef neutral fill:#FFFFFF,stroke:#5F6368,"
            "color:#1F2528,stroke-width:1.2px;"
        ),
        (
            "  classDef core fill:#EEF3FA,stroke:#2E5EAA,"
            "color:#1F2528,stroke-width:1.4px;"
        ),
        (
            "  classDef conditional fill:#FCF3EA,stroke:#D6872C,"
            "color:#1F2528,stroke-width:1.4px;"
        ),
        (
            "  classDef excluded fill:#F2F2F2,stroke:#777777,"
            "color:#1F2528,stroke-width:1.2px;"
        ),
    ]
    for row in nodes.itertuples(index=False):
        label = str(row.label).replace(";", "<br/>")
        lines.append(
            f'  {row.node_id}["{label}<br/><b>n={row.n_communities}</b>"]'
        )
    for row in edges.itertuples(index=False):
        if str(row.label):
            lines.append(
                f'  {row.source} -->|"{row.label}"| {row.target}'
            )
        else:
            lines.append(f"  {row.source} --> {row.target}")
    for node_class, group in nodes.groupby("node_class", sort=False):
        lines.append(
            f"  class {','.join(group['node_id'].astype(str))} {node_class};"
        )
    out_path.write_text("\n".join(lines) + "\n")


def draw_stability_heatmap(
    stability: pd.DataFrame,
    out_path: Path,
) -> None:
    metrics = [
        "detection_rate",
        "conditional_recall_q10",
        "conditional_precision_q10",
        "conditional_jaccard_q10",
    ]
    metric_labels = [
        "Detection rate",
        "Recovery recall q10",
        "Recovery precision q10",
        "Recovery Jaccard q10",
    ]
    class_specs = [
        ("core", "Core"),
        ("conditional", "Conditional"),
        ("not_selected", "Not selected"),
    ]
    color_map = LinearSegmentedColormap.from_list(
        "cnvsim_stability_common",
        ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    )
    color_map.set_bad("#D9DDE1")
    ordered_groups: list[pd.DataFrame] = []
    group_ranges: list[tuple[str, int, int]] = []
    start = 0
    for selection_class, label in class_specs:
        group = stability[
            stability["selection_class"].eq(selection_class)
        ].sort_values(
            [
                "detection_rate",
                "conditional_jaccard_plot",
                "root",
                "final_context",
            ],
            ascending=[False, False, True, True],
        )
        if group.empty:
            raise ValueError(
                f"No communities are available for {selection_class}"
            )
        end = start + len(group)
        group_ranges.append((label, start, end))
        ordered_groups.append(group)
        start = end

    ordered = pd.concat(ordered_groups, ignore_index=True)
    values = np.ma.masked_invalid(
        ordered[metrics].to_numpy(dtype=float).T
    )
    fig, axis = plt.subplots(figsize=(PAPER_COLUMN_WIDTH_IN, 1.85))
    heatmap_image = axis.imshow(
        values,
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks(
        np.arange(len(metric_labels)),
        labels=metric_labels,
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    axis.tick_params(axis="y", length=0, pad=4, colors=TEXT)
    for boundary in np.arange(0.5, len(metrics) - 0.5, 1.0):
        axis.axhline(boundary, color=WHITE, linewidth=1.4, zorder=3)
    for label, group_start, group_end in group_ranges:
        center = (group_start + group_end - 1) / 2
        label_x = center
        label_alignment = "center"
        if label == "Conditional":
            label_x = group_end - 1.0
            label_alignment = "right"
        elif label == "Not selected":
            label_x = group_start + 1.0
            label_alignment = "left"
        axis.text(
            label_x,
            1.08,
            f"{label}\nn={group_end - group_start}",
            transform=axis.get_xaxis_transform(),
            ha=label_alignment,
            va="bottom",
            color="#17365D",
            fontsize=PAPER_SMALL_TEXT_SIZE,
            weight="bold",
            linespacing=0.92,
            clip_on=False,
        )
        if group_end < len(ordered):
            axis.axvline(
                group_end - 0.5,
                color="#17365D",
                linewidth=2.0,
                zorder=4,
            )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#8FA4B8")
        spine.set_linewidth(1.0)

    colorbar = fig.colorbar(
        heatmap_image,
        ax=axis,
        orientation="horizontal",
        fraction=0.11,
        pad=0.30,
        aspect=32,
    )
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_label(
        "Gate value",
        color=TEXT,
        labelpad=2,
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    colorbar.ax.tick_params(
        length=2.5,
        colors=TEXT,
        labelsize=PAPER_SMALL_TEXT_SIZE,
    )
    colorbar.outline.set_visible(False)
    _save(fig, out_path)


def build_diagnostic_contexts(diagnostics: pd.DataFrame) -> pd.DataFrame:
    diagnostics = diagnostics[
        diagnostics["reference_level"].astype(str).isin(
            {"subtype", "conditional"}
        )
    ].copy()
    if diagnostics.empty:
        raise ValueError("No core or conditional diagnostic contexts exist")
    if diagnostics["reference_key"].duplicated().any():
        duplicates = diagnostics.loc[
            diagnostics["reference_key"].duplicated(False), "reference_key"
        ].astype(str)
        raise ValueError(
            "Diagnostic contexts contain duplicate reference keys: "
            f"{sorted(duplicates.unique().tolist())}"
        )
    diagnostics["root"] = diagnostics["root"].astype(str)
    diagnostics["layer"] = diagnostics["reference_level"].map(
        {"subtype": "core", "conditional": "conditional"}
    )
    _numeric(
        diagnostics,
        [
            "icdo_morphology_top1_fraction",
            "icdo_topography_top1_fraction",
            "n_samples",
        ],
        "diagnostic contexts",
    )
    diagnostics["root_label"] = diagnostics["root"].map(_root_label)
    return diagnostics.sort_values(
        ["root", "layer", "reference_key"]
    ).reset_index(drop=True)


def build_root_diagnostic_summary(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        diagnostics.groupby("root", sort=False)
        .agg(
            n_contexts=("reference_key", "nunique"),
            n_definition_samples=("n_samples", "sum"),
            n_conditional_contexts=(
                "layer",
                lambda values: int(pd.Series(values).eq("conditional").sum()),
            ),
            median_morphology_dominance=(
                "icdo_morphology_top1_fraction",
                "median",
            ),
            median_topography_dominance=(
                "icdo_topography_top1_fraction",
                "median",
            ),
        )
        .reset_index()
    )
    summary["root_label"] = summary["root"].map(_root_label)
    summary["mean_dominance"] = summary[
        ["median_morphology_dominance", "median_topography_dominance"]
    ].mean(axis=1)
    return summary.sort_values(
        ["mean_dominance", "root"], ascending=[False, True]
    ).reset_index(drop=True)


def draw_root_diagnostic_heterogeneity(
    summary: pd.DataFrame, out_path: Path
) -> None:
    fig, axis = plt.subplots(
        figsize=(PAPER_COLUMN_WIDTH_IN, PAPER_TALL_PANEL_HEIGHT_IN)
    )
    y = np.arange(len(summary))
    morphology = summary["median_morphology_dominance"].to_numpy(dtype=float)
    topography = summary["median_topography_dominance"].to_numpy(dtype=float)
    for row, (left, right) in enumerate(zip(morphology, topography)):
        axis.plot(
            [left, right],
            [row, row],
            color=GRID,
            linewidth=1.4,
            zorder=1,
        )
    axis.scatter(
        morphology,
        y,
        s=24,
        c=CORE,
        marker="o",
        edgecolors=WHITE,
        linewidths=0.6,
        zorder=2,
        label="Morphology",
    )
    axis.scatter(
        topography,
        y,
        s=27,
        c=CONDITIONAL,
        marker="D",
        edgecolors=WHITE,
        linewidths=0.6,
        zorder=3,
        label="Fine topography",
    )
    axis.axvline(0.70, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
    axis.set_yticks(
        y,
        summary["root_label"],
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    axis.invert_yaxis()
    subtype_column_x = 1.075
    axis.set_xlim(0, 1.15)
    axis.set_xlabel("Median dominant-label fraction across subtypes")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 1.005),
        frameon=False,
        ncol=2,
        fontsize=PAPER_TEXT_SIZE,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    axis.text(
        subtype_column_x,
        1.018,
        "Subtypes",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=PAPER_SMALL_TEXT_SIZE,
        color=TEXT,
    )
    for row, record in enumerate(summary.itertuples(index=False)):
        axis.text(
            subtype_column_x,
            row,
            f"n={int(record.n_contexts)}",
            ha="center",
            va="center",
            fontsize=PAPER_SMALL_TEXT_SIZE,
            color=TEXT,
        )
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.xaxis.label.set_size(PAPER_TEXT_SIZE)
    axis.tick_params(axis="x", labelsize=PAPER_SMALL_TEXT_SIZE)
    _style_axis(axis)
    _save(fig, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 3 from exported, aggregate source-data tables."
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the aggregate Figure 3 source-data TSV files."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty; pass --overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    source_data_dir = args.source_data_dir.expanduser().resolve()
    coverage_path = source_data_dir / "B_reference_coverage.tsv"
    if not coverage_path.exists():
        coverage_path = source_data_dir / "A_reference_coverage.tsv"
    stability_path = source_data_dir / "A_context_stability.tsv"
    if not stability_path.exists():
        stability_path = source_data_dir / "B_context_stability.tsv"
    root_diagnostic_path = source_data_dir / "C_root_diagnostic_heterogeneity.tsv"
    if not root_diagnostic_path.exists():
        root_diagnostic_path = source_data_dir / "D_root_diagnostic_heterogeneity.tsv"
    paths = {
        "coverage": coverage_path,
        "stability": stability_path,
        "root_diagnostics": root_diagnostic_path,
    }
    coverage = _read_tsv(
        paths["coverage"],
        {
            "root",
            "layer",
            "n_contexts",
            "n_definition_members",
            "total_contexts",
            "total_definition_members",
            "root_label",
        },
        "Figure 3B source data",
    )
    stability_map = _read_tsv(
        paths["stability"],
        {
            "root",
            "final_context",
            "detection_rate",
            "conditional_recall_q10",
            "conditional_precision_q10",
            "conditional_jaccard_q10",
            "n_samples",
            "selection_class",
            "conditional_jaccard_plot",
        },
        "Figure 3A source data",
    )
    root_diagnostics = _read_tsv(
        paths["root_diagnostics"],
        {
            "root",
            "n_contexts",
            "n_conditional_contexts",
            "median_morphology_dominance",
            "median_topography_dominance",
            "root_label",
        },
        "Figure 3C source data",
    )

    if args.overwrite:
        for name in GENERATED_OUTPUT_NAMES:
            path = out_dir / name
            if path.is_file() or path.is_symlink():
                path.unlink()

    workflow_nodes, workflow_edges = build_selection_workflow(stability_map)
    outputs = {
        "panel_a_heatmap": out_dir / "A_context_stability_heatmap.png",
        "panel_a_workflow_mermaid": out_dir
        / "A_selection_workflow.mmd",
        "panel_b": out_dir / "B_reference_coverage.png",
        "panel_c": out_dir / "C_root_diagnostic_heterogeneity.png",
    }
    draw_stability_heatmap(stability_map, outputs["panel_a_heatmap"])
    draw_reference_coverage(coverage, outputs["panel_b"])
    draw_root_diagnostic_heterogeneity(
        root_diagnostics, outputs["panel_c"]
    )
    write_selection_workflow_mermaid(
        workflow_nodes,
        workflow_edges,
        outputs["panel_a_workflow_mermaid"],
    )

    counts = {
        "atlas_roots": int(coverage["root"].nunique()),
        "core_contexts": int(
            coverage.loc[
                coverage["layer"].astype(str).eq("core"), "n_contexts"
            ].sum()
        ),
        "conditional_contexts": int(
            coverage.loc[
                coverage["layer"].astype(str).eq("conditional"),
                "n_contexts",
            ].sum()
        ),
        "core_definition_members": int(
            coverage.loc[
                coverage["layer"].astype(str).eq("core"),
                "n_definition_members",
            ].sum()
        ),
        "conditional_definition_members": int(
            coverage.loc[
                coverage["layer"].astype(str).eq("conditional"),
                "n_definition_members",
            ].sum()
        ),
        "refined_communities": int(len(stability_map)),
        "core_stability_selected": int(
            stability_map["selection_class"].eq("core").sum()
        ),
        "conditional_stability_selected": int(
            stability_map["selection_class"].eq("conditional").sum()
        ),
        "not_selected": int(
            stability_map["selection_class"].eq("not_selected").sum()
        ),
        "diagnostic_contexts": int(
            pd.to_numeric(
                root_diagnostics["n_contexts"], errors="raise"
            ).sum()
        ),
    }

    print("[CNVSIM] Figure 3 panels complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print(f"  counts: {counts}")


if __name__ == "__main__":
    main()
