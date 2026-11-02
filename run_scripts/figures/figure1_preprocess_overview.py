#!/usr/bin/env python3
"""Render the manuscript Figure 1 from existing atlas-workspace summaries.

Inputs:
  ATLAS_ROOT/figures/preprocess_baseline/figure1/
    figure1_platform_composition_counts.tsv
    figure1B_root_cnv_shape_summary.tsv
    figure1B_platform_consistency.tsv
  RELEASE_DIR/atlas/{subtypes,subtype_annotations}.tsv

Outputs default to the existing Figure 1 directory, using the original
figure1_preprocess_overview basename (vector PDF and PNG). No
intermediate tables, copied data, QA folders or manifests are generated.
Dependencies are provided by the repository requirements.txt.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_COL = "icdot_root_grouped"
PLATFORM_ORDER = ["Arrays_HighRes", "Arrays_LowRes", "NGS_Based"]
PLATFORM_COLORS = {
    "Arrays_HighRes": "#66C2A5",
    "Arrays_LowRes": "#FC8D62",
    "NGS_Based": "#8DA0CB",
}
PLATFORM_LABELS = {
    "Arrays_HighRes": "High-resolution arrays",
    "Arrays_LowRes": "Low-resolution arrays/CGH",
    "NGS_Based": "NGS-based assays",
}
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
    "C47": "Peripheral nerves",
    "C76": "Ill-defined sites",
    "C80": "Unknown primary site",
}
SOURCE_ORDER = ["GENIE", "GEO", "cBioPortal", "TCGA", "ArrayExpress", "Other"]
DISPLAY_SOURCE_ORDER = ["GENIE", "GEO", "cBioPortal", "TCGA", "Other"]
DISPLAY_SOURCE_MAPPING = {"ArrayExpress": "Other"}
SOURCE_COLORS = {
    "GENIE": "#2E5EAA",
    "GEO": "#69A9A4",
    "cBioPortal": "#D99B63",
    "TCGA": "#9985B7",
    "ArrayExpress": "#DCC565",
    "Other": "#BDBDBD",
}
TEXT = "#111111"
BLUE = "#2E5EAA"
MUTED = "#68747A"
GRID = "#E3E8EA"


def reference_number(key: str) -> int:
    for marker in ("CTX", "COND"):
        if marker in str(key):
            return int(str(key).rsplit(marker, 1)[1])
    raise ValueError(f"Unrecognized subtype key: {key}")


def style_axis(axis, *, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#91A0A6")
    axis.spines["bottom"].set_color("#91A0A6")
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.55)
    axis.set_axisbelow(True)


def point_size(root: str, size_map: dict[str, int]) -> float:
    # Preserve the original Figure 1's square-root count scaling at journal size.
    return 8.0 + 42.0 * (float(size_map[root]) / max(size_map.values())) ** 0.5


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def root_palette(roots):
    ordered = sorted(set(str(root) for root in roots))
    colors = sns.color_palette("husl", n_colors=len(ordered))
    return {root: colors[i] for i, root in enumerate(ordered)}


def root_display_label(root):
    codes = {
        "C15_C17_C21_C26": "C15/C17/C21/C26",
        "C40_C41_C48": "C40/C41/C48",
        "C69_C70_C72": "C69/C70/C72",
    }
    return f"{codes.get(root, root.replace('_', '-'))} {ROOT_LABELS[root]}"


def draw_original_panel_b(fig, shape, platform, counts):
    roots = counts[ROOT_COL].astype(str).tolist()
    palette = root_palette(roots)
    size_map = counts.set_index(ROOT_COL).Total.to_dict()
    left = fig.add_axes([0.105, 0.565, 0.255, 0.278], label="A_burden")
    right = fig.add_axes([0.485, 0.565, 0.255, 0.278], label="A_platform")
    legend = fig.add_axes([0.765, 0.530, 0.225, 0.325], label="Anatomical legend")
    legend.axis("off")
    xvalues, yvalues = [], []
    for _, row in shape.iterrows():
        root = str(row[ROOT_COL])
        x, y = float(np.log1p(row.median_variant_count)), float(row.median_cnv_fraction)
        xvalues.append(x)
        yvalues.append(y)
        left.scatter(x, y, s=point_size(root, size_map), color=palette[root], edgecolor="white", linewidth=0.5, zorder=3)
    coefficients = np.polyfit(np.asarray(xvalues), np.asarray(yvalues), deg=1)
    xx = np.linspace(min(xvalues), max(xvalues), 100)
    left.plot(xx, np.polyval(coefficients, xx), color="0.35", linewidth=0.8, zorder=2)
    left.set(xlim=(1.5, 3.45), ylim=(-0.02, 0.255),
             xlabel="log(1 + median CNV event count\nper analysis)",
             ylabel="Median genome fraction\naltered by CNVs")
    left.set_xticks([1.5, 2.0, 2.5, 3.0])
    left.set_yticks([0, 0.05, 0.10, 0.15, 0.20])

    for _, row in platform.iterrows():
        root = str(row[ROOT_COL])
        right.scatter(float(row.avg_corr_offdiag), float(row.median_delta_freq),
                      s=point_size(root, size_map), color=palette[root], edgecolor="white", linewidth=0.5, zorder=3)
    x = platform.avg_corr_offdiag.to_numpy(float)
    y = platform.median_delta_freq.to_numpy(float)
    x25, x75 = np.nanquantile(x, [0.25, 0.75])
    y25, y75 = np.nanquantile(y, [0.25, 0.75])
    right.axvspan(x25, x75, color="0.90", zorder=0)
    right.axhline(y25, color="0.75", linewidth=0.55, linestyle="--")
    right.axhline(y75, color="0.75", linewidth=0.55, linestyle="--")
    right.set(xlim=(-0.06, 1.03), ylim=(0.018, 0.185),
              xlabel="Mean cross-family\ncorrelation", ylabel="Median frequency range\nacross platform families")
    right.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    right.set_yticks([0.04, 0.08, 0.12, 0.16])
    for axis in [left, right]:
        style_axis(axis, grid_axis="y")

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[root],
                      markeredgecolor="white", markersize=3.5, label=root_display_label(root)) for root in roots]
    legend.legend(handles=handles, loc="center left", frameon=False, fontsize=6.0,
                  labelspacing=0.28, handlelength=0.9, handletextpad=0.3, borderaxespad=0.0)
    return left, right, legend


def horizontal_root_labels(fig, axis, centres, names):
    """Place horizontal labels in the smallest available non-overlapping lane.

    Thin vertical guides link labels to their groups. Guides occupy whitespace
    only: segments stop around intervening text instead of crossing letters.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_pt = fig.get_figwidth() * 72
    height_pt = fig.get_figheight() * 72
    properties = FontProperties(family="Arial", size=6.1)
    lanes = [[], [], [], []]
    labels = []
    placements = []
    for centre, name in zip(centres, names, strict=True):
        root_x = fig.transFigure.inverted().transform(axis.transData.transform((centre, 0)))[0]
        text_width = renderer.get_text_width_height_descent(name, properties, ismath=False)[0] / fig.dpi * 72
        text_x = np.clip(root_x * width_pt, 0.035 * width_pt + text_width / 2, 0.985 * width_pt - text_width / 2)
        interval = (float(text_x - text_width / 2), float(text_x + text_width / 2))
        lane = next((i for i, occupied in enumerate(lanes) if all(interval[0] >= previous[1] + 3.0 for previous in occupied)), None)
        if lane is None:
            raise ValueError(f"Horizontal root labels need more room near {name}")
        lanes[lane].append(interval)
        y = 0.172 - lane * 0.026
        artist = fig.text(text_x / width_pt, y, name, ha="center", va="top", fontsize=6.1)
        labels.append(artist)
        placements.append({"root_label":name, "lane":lane, "root_x":float(root_x), "text_x":float(text_x / width_pt)})
    fig.canvas.draw()
    boxes = [artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) for artist in labels]
    start_y = axis.get_position().y0 * height_pt - 2.0
    for placement, own_box in zip(placements, boxes, strict=True):
        x = placement["root_x"] * width_pt
        end_y = own_box.y1 * 72 + 1.5
        segments = [(end_y, start_y)]
        for other_box in boxes:
            if other_box is own_box or not (other_box.x0 * 72 - 1.0 <= x <= other_box.x1 * 72 + 1.0):
                continue
            bottom, top = other_box.y0 * 72 - 1.0, other_box.y1 * 72 + 1.0
            next_segments = []
            for lo, hi in segments:
                if top <= lo or bottom >= hi:
                    next_segments.append((lo, hi))
                else:
                    if lo < bottom: next_segments.append((lo, bottom))
                    if top < hi: next_segments.append((top, hi))
            segments = next_segments
        for lo, hi in segments:
            if hi > lo:
                fig.add_artist(Line2D([placement["root_x"]] * 2, [lo / height_pt, hi / height_pt],
                                      transform=fig.transFigure, color="#AAB3B8", lw=0.5, zorder=0.5))
    return placements


def render(data: dict[str, pd.DataFrame], output: Path) -> None:
    shape = data["shape"]
    platform = data["platform"]
    counts = data["counts"]
    families = data["families"]
    refs = data["references"]
    font_manager.findfont("Arial", fallback_to_default=False)
    fig = plt.figure(figsize=(174 / 25.4, 190 / 25.4), dpi=300)

    fig.text(0.040, 0.985, "A", fontsize=10, weight="bold", va="top")
    fig.text(0.105, 0.957, "204,587 neoplastic CNV analyses across 22 anatomical groups", fontsize=8.2, weight="bold", va="top")
    family_axis = fig.add_axes([0.105, 0.916, 0.865, 0.014], label="Platform family composition")
    left = 0.0
    for family in PLATFORM_ORDER:
        fraction = float(families.loc[family, "fraction"])
        family_axis.barh(0.5, fraction, left=left, height=1.0, color=PLATFORM_COLORS[family], edgecolor="white", linewidth=0.6)
        family_axis.text(left + fraction / 2, 0.5, f"{100*fraction:.1f}%", ha="center", va="center", fontsize=6.0)
        left += fraction
    family_axis.set(xlim=(0, 1), ylim=(0, 1))
    family_axis.axis("off")
    platform_handles = [Patch(facecolor=PLATFORM_COLORS[f], label=PLATFORM_LABELS[f]) for f in PLATFORM_ORDER]
    fig.legend(handles=platform_handles, loc="center", bbox_to_anchor=(0.538, 0.892), ncol=3,
               fontsize=6.4, handlelength=1.1, handletextpad=0.45, columnspacing=1.5)

    ax_a, ax_b, root_legend = draw_original_panel_b(fig, shape, platform, counts)

    fig.text(0.040, 0.488, "B", fontsize=10, weight="bold", va="center")
    fig.text(0.105, 0.488, "Reference depth and source composition", fontsize=7.7, weight="bold", va="center")
    fig.text(0.970, 0.462, "112 subtypes  |  155,881 defining analyses  |  median depth 882.5", fontsize=6.5, ha="right", color=MUTED)
    depth_axis = fig.add_axes([0.105, 0.338, 0.865, 0.107], label="B_depth")
    source_axis = fig.add_axes([0.105, 0.196, 0.865, 0.108], label="B_sources")
    positions = refs.plot_position.to_numpy(float)
    xmin, xmax = -0.8, float(positions.max() + 0.8)
    root_centres, root_names = [], []
    for i, (root, group) in enumerate(refs.groupby("root", sort=False)):
        lo, hi = float(group.plot_position.min() - 0.48), float(group.plot_position.max() + 0.48)
        if i % 2 == 0:
            depth_axis.axvspan(lo, hi, color="#E6E6E6", zorder=0)
        root_centres.append(float(group.plot_position.mean()))
        root_names.append(ROOT_LABELS[root])
    median_n = float(refs.n_definition_samples.median())
    depth_axis.axhline(median_n, color="#A4ADB3", lw=0.65, linestyle=(0, (3, 2)), zorder=1)
    depth_axis.scatter(positions, refs.n_definition_samples, color=BLUE, s=5.4, linewidth=0, zorder=3)
    depth_axis.set(yscale="log", ylim=(180, 20000), xlim=(xmin, xmax), ylabel="Defining analyses\n(log scale)")
    depth_axis.yaxis.set_major_locator(FixedLocator([200, 1000, 10000]))
    depth_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    depth_axis.yaxis.set_minor_locator(NullLocator())
    depth_axis.set_xticks([])
    style_axis(depth_axis, grid_axis="y")
    depth_axis.spines["bottom"].set_visible(False)

    display = refs[["reference_key", "root", "n_definition_samples", "plot_position", *DISPLAY_SOURCE_ORDER]].copy()
    display["Other"] = refs["Other"] + refs["ArrayExpress"]
    assert display[DISPLAY_SOURCE_ORDER].sum(axis=1).eq(refs.n_definition_samples).all()
    bottom = np.zeros(len(refs), dtype=float)
    for name in DISPLAY_SOURCE_ORDER:
        values = (display[name] / display.n_definition_samples).to_numpy(float)
        source_axis.bar(positions, 100 * values, bottom=100 * bottom, width=0.94,
                        color=SOURCE_COLORS[name], linewidth=0, align="center")
        bottom += values
    assert np.allclose(bottom, 1.0)
    source_axis.set(ylim=(0, 100), xlim=(xmin, xmax), ylabel="Source share (%)")
    source_axis.set_yticks([0, 50, 100])
    source_axis.set_xticks([])
    source_axis.spines["top"].set_visible(False)
    source_axis.spines["right"].set_visible(False)
    source_axis.spines["bottom"].set_visible(False)
    source_axis.spines["left"].set_color("#91A0A6")
    horizontal_root_labels(fig, source_axis, root_centres, root_names)
    source_handles = [Patch(facecolor=SOURCE_COLORS[name], label=name) for name in DISPLAY_SOURCE_ORDER]
    fig.legend(handles=source_handles, loc="center", bbox_to_anchor=(0.538, 0.037), ncol=5,
               fontsize=6.5, handlelength=1.0, handletextpad=0.4, columnspacing=1.25)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, ax_a, ax_b, depth_axis, source_axis)
    metadata_pdf = {"Title": "CNVSIM Figure 1: anatomical heterogeneity and reference depth", "Author": "", "Creator": "", "Producer": ""}
    fig.savefig(output / "figure1_preprocess_overview.pdf", metadata=metadata_pdf)
    fig.savefig(output / "figure1_preprocess_overview.png", dpi=400, metadata={"Software": ""})
    plt.close(fig)


def require_matplotlib_panel_alignment(fig, left, right, depth, sources) -> None:
    """Check this figure's final plot areas in memory, without helper imports.

    Both background panels must share their vertical edges and plot widths;
    the two reference tracks must share their horizontal edges. All plot
    areas must be disjoint. The tolerance is 1.5 pt at the exported size.
    """
    width, height = fig.get_size_inches() * 72
    boxes = {
        name: np.array([axis.get_position().x0 * width,
                        axis.get_position().y0 * height,
                        axis.get_position().x1 * width,
                        axis.get_position().y1 * height])
        for name, axis in [("left", left), ("right", right),
                           ("depth", depth), ("sources", sources)]
    }
    differences = [
        abs(boxes["left"][1] - boxes["right"][1]),
        abs(boxes["left"][3] - boxes["right"][3]),
        abs(np.diff(boxes["left"][[0, 2]])[0]
            - np.diff(boxes["right"][[0, 2]])[0]),
        abs(boxes["depth"][0] - boxes["sources"][0]),
        abs(boxes["depth"][2] - boxes["sources"][2]),
    ]
    if max(differences) > 1.5:
        raise ValueError("Figure 1 plot areas differ by more than 1.5 pt")
    items = list(boxes.items())
    for index, (first, a) in enumerate(items):
        for second, b in items[index + 1:]:
            if (min(a[2], b[2]) > max(a[0], b[0]) + 0.01
                    and min(a[3], b[3]) > max(a[1], b[1]) + 0.01):
                raise ValueError(f"Overlapping Figure 1 plot areas: {first}, {second}")


def _read_tsv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 1 input: {path}")
    table = pd.read_csv(path, sep="\t")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return table


def load_data(summary_dir: Path, release_dir: Path) -> dict[str, pd.DataFrame]:
    """Build plot tables in memory from the existing analysis outputs."""
    counts = _read_tsv(
        summary_dir / "figure1_platform_composition_counts.tsv",
        {ROOT_COL, "Total", "Other", *PLATFORM_ORDER},
    )
    shape = _read_tsv(
        summary_dir / "figure1B_root_cnv_shape_summary.tsv",
        {ROOT_COL, "n_shape_profiles", "median_cnv_fraction", "median_variant_count"},
    )
    platform = _read_tsv(
        summary_dir / "figure1B_platform_consistency.tsv",
        {ROOT_COL, "feature_scope", "avg_corr_offdiag", "median_delta_freq"},
    )
    subtypes = _read_tsv(
        release_dir / "atlas/subtypes.tsv",
        {"reference_key", "root", "reference_layer", "n_definition_samples"},
    )
    annotations = _read_tsv(
        release_dir / "atlas/subtype_annotations.tsv",
        {"reference_key", "annotation_type", "annotation_value", "n_samples", "fraction"},
    )

    for table, key in [(counts, ROOT_COL), (shape, ROOT_COL),
                       (platform, ROOT_COL), (subtypes, "reference_key")]:
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError(f"Missing or duplicate Figure 1 identifiers: {key}")
    if (len(counts) != 22 or int(counts.Total.sum()) != 204587
            or not counts.Other.eq(0).all()
            or not counts[PLATFORM_ORDER].sum(axis=1).eq(counts.Total).all()):
        raise ValueError("Input counts do not match the approved Figure 1 cohort")
    if (set(shape[ROOT_COL]) != set(counts[ROOT_COL])
            or int(shape.n_shape_profiles.sum()) != int(counts.Total.sum())):
        raise ValueError("Burden summaries and input-analysis counts do not reconcile")
    if not shape.median_variant_count.ge(0).all():
        raise ValueError("CNV event counts must be non-negative before log1p transformation")
    if (len(platform) != 21
            or set(shape[ROOT_COL]) - set(platform[ROOT_COL]) != {"C80"}
            or not set(platform[ROOT_COL]).issubset(set(counts[ROOT_COL]))
            or not platform.feature_scope.eq("autosomes").all()):
        raise ValueError("Platform summaries must contain the 21 evaluable autosomal roots")
    for table, columns in [
        (shape, ["median_cnv_fraction", "median_variant_count"]),
        (platform, ["avg_corr_offdiag", "median_delta_freq"]),
    ]:
        if not np.isfinite(table[columns].to_numpy(float)).all():
            raise ValueError("Non-finite Figure 1 background summaries")

    source = annotations.loc[annotations.annotation_type.eq("source_group")].copy()
    if (source.duplicated(["reference_key", "annotation_value"]).any()
            or set(source.annotation_value) != set(SOURCE_ORDER)
            or not source.n_samples.ge(0).all()
            or not source.n_samples.mod(1).eq(0).all()):
        raise ValueError("Invalid or duplicate source-composition annotations")
    pivot = source.pivot(index="reference_key", columns="annotation_value",
                         values="n_samples").fillna(0).astype(int)
    if set(pivot.index) != set(subtypes.reference_key):
        raise ValueError("Source annotations do not cover exactly the released subtypes")
    refs = subtypes.merge(pivot[SOURCE_ORDER], left_on="reference_key",
                          right_index=True, validate="one_to_one")
    if (len(refs) != 112 or refs.root.nunique() != 19
            or int(refs.n_definition_samples.sum()) != 155881
            or not refs.n_definition_samples.gt(0).all()
            or not refs[SOURCE_ORDER].sum(axis=1).eq(refs.n_definition_samples).all()):
        raise ValueError("Reference depths and source annotations do not reconcile")
    source_depths = source.merge(
        subtypes[["reference_key", "n_definition_samples"]],
        on="reference_key", validate="many_to_one",
    )
    if not np.allclose(source_depths.fraction,
                       source_depths.n_samples / source_depths.n_definition_samples):
        raise ValueError("Source fractions disagree with defining-analysis counts")
    root_order = [root for root in ROOT_LABELS if root in set(refs.root)]
    refs["root_order"] = pd.Categorical(refs.root, categories=root_order, ordered=True)
    refs["layer_order"] = refs.reference_layer.map({"core": 0, "conditional": 1})
    if refs.root_order.isna().any() or refs.layer_order.isna().any():
        raise ValueError("Unrecognized reference root or layer")
    refs["reference_number"] = refs.reference_key.map(reference_number)
    refs = refs.sort_values(["root_order", "layer_order", "reference_number"]).reset_index(drop=True)
    refs["plot_position"] = 0.0
    cursor = 0.0
    for _, group in refs.groupby("root", sort=False):
        refs.loc[group.index, "plot_position"] = cursor + np.arange(len(group))
        cursor += len(group) + 0.9
    families = counts[PLATFORM_ORDER].sum().rename("n_analyses").to_frame()
    families["fraction"] = families.n_analyses / families.n_analyses.sum()
    return {"shape": shape, "platform": platform, "counts": counts,
            "families": families, "references": refs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, required=True,
                        help="Analysis workspace containing figures/preprocess_baseline/figure1")
    parser.add_argument("--release-dir", type=Path, required=True,
                        help="Public data release containing atlas/subtypes.tsv and subtype_annotations.tsv")
    parser.add_argument("--out-dir", type=Path,
                        help="Default: ATLAS_ROOT/figures/preprocess_baseline/figure1")
    args = parser.parse_args()
    summary_dir = args.atlas_root.expanduser().resolve() / "figures/preprocess_baseline/figure1"
    release_dir = args.release_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve() if args.out_dir else summary_dir
    data = load_data(summary_dir, release_dir)
    output.mkdir(parents=True, exist_ok=True)
    render(data, output)
    print("[CNVSIM] Figure 1 complete: 112 subtypes, 155,881 defining analyses")
    print(f"[CNVSIM] wrote figure1_preprocess_overview.{{pdf,png}} to {output}")


if __name__ == "__main__":
    main()
