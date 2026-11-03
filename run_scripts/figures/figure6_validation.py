#!/usr/bin/env python3
"""Render Figure 6 from frozen CNVSIM summaries and CNV profiles.

Default outputs: ATLAS_ROOT/figures/figure6/Figure6.pdf and Figure6.png.
Use --out-dir to override and --overwrite to replace existing files.
Only these two files are generated; no models are fitted.
All data are read from --release-dir; subtype detection uses the public
root-level minimum-fold summary, not an all-fold subtype intersection.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

WIDTH_MM, HEIGHT_MM = 170.0, 230.0
LAYOUT_WIDTH_MM = 183.0
BLUE, ORANGE, TEXT = "#2E5EAA", "#D9822B", "#111111"
GREY, DARK, GRID = "#DFE5E8", "#77868D", "#E8ECEE"
ROOTS = {
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
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
    }
)


def text(fig, x, y, value, **kw):
    return fig.text(x / LAYOUT_WIDTH_MM, y / HEIGHT_MM, value, **kw)


def axis(fig, x, y, w, h, label):
    return fig.add_axes(
        [x / LAYOUT_WIDTH_MM, y / HEIGHT_MM, w / LAYOUT_WIDTH_MM, h / HEIGHT_MM], label=label
    )


def root_axis(ax, first=False):
    ax.set_ylim(18.65, -0.65)
    ax.set_yticks(range(19), list(ROOTS.values()) if first else [""] * 19)
    ax.tick_params(axis="y", length=0, pad=3, labelsize=6)
    ax.tick_params(axis="x", length=2, width=0.5, pad=2, labelsize=6)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#95A2A8")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)


def require_matplotlib_panel_alignment(fig, axes, rows, columns=()):
    """Check comparable plot areas at final size without external QA imports."""
    fig.canvas.draw()
    scale = np.asarray([*fig.get_size_inches(), *fig.get_size_inches()]) * 72
    boxes = {
        a.get_label(): np.asarray(
            [a.get_position().x0, a.get_position().y0, a.get_position().x1, a.get_position().y1]
        )
        * scale
        for a in axes
    }
    for group in rows:
        b = np.asarray([boxes[name] for name in group])
        if any(np.ptp(b[:, i]) > 1.5 for i in (1, 3)):
            raise ValueError("Comparable plot areas have different vertical edges")
        if np.ptp(b[:, 2] - b[:, 0]) > 1.5:
            raise ValueError("Comparable plot areas have different widths")
        order = np.argsort(b[:, 0])
        b = b[order]
        if len(b) > 2 and np.ptp(b[1:, 0] - b[:-1, 2]) > 1.5:
            raise ValueError("Comparable plot areas have unequal gutters")
    for group in columns:
        b = np.asarray([boxes[name] for name in group])
        if any(np.ptp(b[:, i]) > 1.5 for i in (0, 2)):
            raise ValueError("Comparable plot areas have different horizontal edges")
    values = list(boxes.values())
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            if min(a[2], b[2]) - max(a[0], b[0]) > 0.1 and min(a[3], b[3]) - max(a[1], b[1]) > 0.1:
                raise ValueError("Plot areas overlap")


def pair_correspondence(crosstab):
    """Describe splitting and merging separately on commonly assigned records.

    These are descriptive record-pair fractions, not independent observations,
    diagnostic accuracies or chance-adjusted validation scores.
    """
    choose2 = lambda values: values * (values - 1) // 2
    records = []
    for root in ROOTS:
        frame = crosstab[crosstab.root.eq(root)]
        old = frame.groupby("v1_label").n.sum().astype("int64")
        rebuilt = frame.groupby("new_context").n.sum().astype("int64")
        n = int(frame.n.sum())
        total = choose2(n)
        same = int(choose2(old).sum())
        shared = int(choose2(frame.n.astype("int64")).sum())
        different = total - same
        apart = total - same - int(choose2(rebuilt).sum()) + shared
        assert 0 <= shared <= same and 0 <= apart <= different
        records.append(
            dict(
                root=root,
                common_n=n,
                released_groups=len(old),
                rebuilt_groups=len(rebuilt),
                minimum_released_n=int(old.min()) if len(old) else 0,
                same_released_pairs=same,
                same_released_and_rebuilt_pairs=shared,
                different_released_pairs=different,
                different_released_and_rebuilt_pairs=apart,
                same_subtype_kept_together=shared / same if same else np.nan,
                different_subtypes_kept_apart=apart / different if different else np.nan,
                small_released_subtype=bool(len(old) and old.min() < 10),
            )
        )
    return pd.DataFrame(records).set_index("root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, help="Default: ATLAS_ROOT/figures/figure6")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    p = args.release_dir.expanduser().resolve()
    if args.atlas_root is None and args.out_dir is None:
        parser.error("Supply --atlas-root or --out-dir")
    args.atlas_root = args.atlas_root.expanduser().resolve() if args.atlas_root else None
    out = (
        args.out_dir.expanduser().resolve() if args.out_dir else args.atlas_root / "figures/figure6"
    )
    existing = [
        out / f"Figure6.{ext}" for ext in ("pdf", "png") if (out / f"Figure6.{ext}").exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))
    out.mkdir(parents=True, exist_ok=True)
    recovery = pd.read_csv(p / "validation/signature_recovery.tsv", sep="\t")
    r = (
        recovery[recovery.validation_scope.eq("root")]
        .pivot(index="root", columns="metric", values="median")
        .reindex(ROOTS)
    )
    root_summary = recovery[recovery.validation_scope.eq("root")]
    r_min = root_summary.pivot(index="root", columns="metric", values="minimum").reindex(ROOTS)
    n = root_summary.groupby("root").n_subtypes.first().reindex(ROOTS).astype(int)
    minimum_recovered = (r_min.subtype_detection * n).round().astype(int)
    assert int(n.sum()) == 103
    assert np.allclose(minimum_recovered / n, r_min.subtype_detection, atol=1e-6, rtol=0)
    detection = pd.DataFrame({"n": n, "recovered": minimum_recovered})
    coverage = pd.read_csv(
        p / "validation/source_holdout/source_holdout_coverage_by_root.tsv", sep="\t"
    ).set_index("root")
    compatibility = pd.read_csv(
        p / "validation/source_holdout/source_holdout_compatibility_by_root.tsv", sep="\t"
    ).set_index("root")
    crosstab = pd.read_csv(p / "validation/source_holdout/source_holdout_crosstab.tsv", sep="\t")
    overall = coverage.loc["ALL"]
    assert list(
        overall[["roster_n", "assigned_n", "abstained_n", "missing_root_run_n"]].astype(int)
    ) == [10156, 7880, 2083, 193]
    assert (
        (
            coverage.loc[list(ROOTS), "assigned_n"]
            + coverage.loc[list(ROOTS), "abstained_n"]
            + coverage.loc[list(ROOTS), "missing_root_run_n"]
        )
        .eq(coverage.loc[list(ROOTS), "roster_n"])
        .all()
    )
    queries = pd.read_csv(p / "validation/projection_performance.tsv", sep="\t")
    queries = queries[
        queries.reference_level.eq("core") & queries.validation_design.eq("fixed_holdout")
    ]
    query_root = (
        queries[queries.validation_scope.eq("root")]
        .pivot(index="root", columns="metric", values="value")
        .reindex(ROOTS)
    )
    query_all = queries[queries.validation_scope.eq("global")].set_index("metric")["value"]
    assert np.isclose(query_all.assignment_coverage, 26223 / 30609)
    assert np.isclose(query_all.assigned_sample_accuracy, 26213 / 26223)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300, facecolor="white")
    axes = []
    groups = []
    columns = []
    yy = np.arange(19)
    text(fig, 3, 226, "A", fontsize=10, weight="bold", va="center")
    adefs = [
        ("Subtype", 100 * detection.recovered / detection.n, 100 * r.definition_sample_recall),
        ("Track 1", 100 * r.track1_signature_recall, 100 * r.track1_signature_similarity),
        ("Track 2", 100 * r.track2_signature_recall, 100 * r.track2_signature_similarity),
    ]
    anames = []
    for i, (name, recovered, similarity) in enumerate(adefs):
        x = 35 + 49 * i
        ax = axis(fig, x, 167, 44, 48, f"A_{i}")
        root_axis(ax, first=i == 0)
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Recovery / similarity (%)", fontsize=6, labelpad=3)
        text(fig, x + 22, 219, name, ha="center", va="center", fontsize=7, weight="bold")
        for y, a, b in zip(yy, recovered, similarity):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [y, y], color="#BCC7CC", lw=0.7, zorder=1)
            if np.isfinite(a):
                ax.scatter(a, y, s=12, c=BLUE, edgecolors="white", lw=0.35, zorder=4)
            if np.isfinite(b):
                ax.scatter(b, y, s=26, facecolors="none", edgecolors=ORANGE, lw=1, zorder=3)
            if not np.isfinite(a) and not np.isfinite(b):
                ax.text(25, y, "—", ha="center", va="center", fontsize=6, color=DARK)
        for y, row in enumerate(detection.itertuples()):
            if i == 0 and row.recovered < row.n:
                ax.annotate(
                    f"{row.recovered}/{row.n}",
                    (100 * row.recovered / row.n, y),
                    xytext=(-5, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=6,
                    color=BLUE,
                )
        axes.append(ax)
        anames.append(ax.get_label())
    groups.append(anames)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="", color=BLUE, ms=3, label="Subtype / signature recovery"
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="none",
                mec=ORANGE,
                ms=4,
                label="Analysis recovery / signature similarity",
            ),
        ],
        loc="center",
        bbox_to_anchor=(110 / LAYOUT_WIDTH_MM, 226 / HEIGHT_MM),
        ncol=2,
        fontsize=6,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    text(fig, 3, 157.5, "B", fontsize=10, weight="bold", va="center")
    cnames = []
    for i, (metric, label, limits) in enumerate(
        [
            ("assignment_coverage", "Query assignment (%)", (80, 92)),
            ("assigned_sample_accuracy", "Assigned-label agreement (%)", (99.5, 100.05)),
        ]
    ):
        x = 35 + 77 * i
        ax = axis(fig, x, 101, 65, 48, f"B_{i}")
        root_axis(ax, i == 0)
        ax.scatter(
            100 * query_root[metric], yy, s=14, c=BLUE, edgecolors="white", lw=0.35, zorder=3
        )
        ax.axvline(100 * query_all[metric], color=ORANGE, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(*limits)
        ax.set_xticks([80, 84, 88, 92] if i == 0 else [99.5, 99.75, 100])
        ax.set_xlabel(label, fontsize=6, labelpad=3)
        text(
            fig,
            x + 32.5,
            153,
            f"Overall: {100*query_all[metric]:.2f}%",
            ha="center",
            va="center",
            fontsize=6.4,
            color=ORANGE,
            weight="bold",
        )
        axes.append(ax)
        cnames.append(ax.get_label())
    groups.append(cnames)
    text(fig, 3, 86, "C", fontsize=10, weight="bold", va="center")
    pair = pair_correspondence(crosstab)
    assert int(pair.common_n.sum()) == 6981
    assert pair.common_n.eq(compatibility.reindex(ROOTS).common_n).all()
    assert pair.released_groups.eq(compatibility.reindex(ROOTS).v1_clusters_in_common).all()
    assert pair.rebuilt_groups.eq(compatibility.reindex(ROOTS).new_clusters_in_common).all()
    ax = axis(fig, 35, 11, 33, 64, "C_coverage")
    root_axis(ax, True)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("TCGA assigned (%)", fontsize=6, labelpad=3)
    for y, root in enumerate(ROOTS):
        row = coverage.loc[root]
        left = 0
        for col, color in [
            ("assigned_n", BLUE),
            ("abstained_n", GREY),
            ("missing_root_run_n", DARK),
        ]:
            value = 100 * row[col] / row.roster_n
            if value:
                ax.barh(
                    y,
                    value,
                    left=left,
                    height=0.55,
                    color=color,
                    edgecolor="white",
                    lw=0.25,
                    zorder=2,
                )
            left += value
    text(fig, 51.5, 80, "Assigned: 77.6%", ha="center", va="center", fontsize=6.6, weight="bold")
    axes.append(ax)
    caxes = [ax]
    text(fig, 75.5, 80, "Common n", ha="center", va="center", fontsize=6)
    text(fig, 174, 80, "Released\n→ rebuilt", ha="center", va="center", fontsize=6, linespacing=1.1)
    for y, row in enumerate(pair.itertuples()):
        y_mm = 11 + 64 * (18.65 - y) / 19.3
        text(fig, 75.5, y_mm, f"{row.common_n:,}", ha="center", va="center", fontsize=6)
        value = f"{row.released_groups} → {row.rebuilt_groups}" if row.common_n else "—"
        text(fig, 174, y_mm, value, ha="center", va="center", fontsize=6)
    for x, metric, title, label in [
        (83, "same_subtype_kept_together", "Same subtype", "Pairs kept together (%)"),
        (131, "different_subtypes_kept_apart", "Different subtypes", "Pairs kept apart (%)"),
    ]:
        ax = axis(fig, x, 11, 33, 64, f"C_{metric}")
        root_axis(ax, False)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(label, fontsize=6, labelpad=3)
        text(fig, x + 16.5, 80, title, ha="center", va="center", fontsize=6.6, weight="bold")
        for y, row in enumerate(pair.itertuples()):
            value = getattr(row, metric)
            if np.isfinite(value):
                ax.scatter(
                    100 * value,
                    y,
                    s=20,
                    facecolors="white" if row.small_released_subtype else BLUE,
                    edgecolors=BLUE,
                    lw=0.8,
                    zorder=3,
                )
            else:
                ax.text(25, y, "—", ha="center", va="center", fontsize=6, color=DARK)
        axes.append(ax)
        caxes.append(ax)
    groups.append([ax.get_label() for ax in caxes])
    fig.legend(
        handles=[
            Patch(fc=BLUE, label="Assigned"),
            Patch(fc=GREY, label="Unassigned"),
            Patch(fc=DARK, label="No reference"),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="white",
                mec=BLUE,
                ms=3.5,
                label="Small released subtype (<10 records)",
            ),
        ],
        loc="center",
        bbox_to_anchor=(108 / LAYOUT_WIDTH_MM, 86 / HEIGHT_MM),
        ncol=4,
        fontsize=6,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.1,
    )
    require_matplotlib_panel_alignment(fig, axes, groups, columns)
    fig.savefig(
        out / "Figure6.pdf",
        metadata={"Title": "Reference reconstruction and query projection tests"},
    )
    fig.savefig(out / "Figure6.png", dpi=400)
    plt.close(fig)
    print(f"[CNVSIM] Figure 6 complete: wrote Figure6.pdf and Figure6.png to {out}")


if __name__ == "__main__":
    main()
