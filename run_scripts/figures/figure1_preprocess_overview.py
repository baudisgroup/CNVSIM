#!/usr/bin/env python3
from __future__ import annotations

"""Build Figure 1 panels from the frozen CNVSIM source package.

Panel A shows the platform composition of every tumour analysis entering the
pipeline. Panel B summarizes root-level CNV burden and agreement of autosomal
gene-level CNV frequencies across platform families. The script also writes a
full-root version of panel A for supplementary material and the tables behind
each plot. A precomputed platform-metrics table can be supplied to avoid
rescanning the complete gene-panel cache when only the figure layout changes.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.preprocess.roots import icdot_root_grouped


ID_COL = "analysis_id"
ROOT_COL = "icdot_root_grouped"
PLATFORM_COL = "platform_family_refined"

ROOT_LABELS = {
    "C00_C14": "C00-C14 Oral/pharynx",
    "C15_C17_C21_C26": "C15/C17/C21/C26 Other digestive",
    "C16": "C16 Stomach",
    "C18_C20": "C18-C20 Colon/rectum",
    "C22_C24": "C22-C24 Liver/biliary",
    "C25": "C25 Pancreas",
    "C30_C39": "C30-C39 Respiratory",
    "C40_C41_C48": "C40/C41/C48 Bone/connective",
    "C42": "C42 Haematopoietic",
    "C44": "C44 Skin",
    "C47": "C47 Peripheral nerves",
    "C49": "C49 Soft tissue",
    "C50": "C50 Breast",
    "C51_C58": "C51-C58 Female genital",
    "C60_C63": "C60-C63 Male genital",
    "C64_C68": "C64-C68 Urinary tract",
    "C69_C70_C72": "C69/C70/C72 Eye/CNS",
    "C71": "C71 Brain",
    "C73_C75": "C73-C75 Endocrine",
    "C76": "C76 Ill-defined sites",
    "C77": "C77 Lymph nodes",
    "C80": "C80 Unknown primary site",
    "UNMAPPED": "Unmapped topography",
}

PLATFORM_ORDER = ["Arrays_HighRes", "Arrays_LowRes", "NGS_Based"]
PLATFORM_COLORS = {
    "Arrays_HighRes": "#66C2A5",
    "Arrays_LowRes": "#FC8D62",
    "NGS_Based": "#8DA0CB",
    "Other": "#BDBDBD",
}
PLATFORM_LABELS = {
    "Arrays_HighRes": "High-resolution arrays",
    "Arrays_LowRes": "Lower-resolution array/CGH",
    "NGS_Based": "NGS-based",
    "Other": "Other/unknown",
}
TEXT = "#111111"

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

PLATFORM_METRIC_COLUMNS = [
    ROOT_COL,
    "feature_scope",
    "avg_corr_offdiag",
    "median_delta_freq",
    "n_genes",
    "n_complete_genes",
    "n_platform_families",
    "n_samples",
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _normalize_platform(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Panels_NGS": "NGS_Based",
        "NGS": "NGS_Based",
        "NGS panels": "NGS_Based",
        "NGS_panel": "NGS_Based",
        "NGS_panels": "NGS_Based",
        "High-resolution arrays": "Arrays_HighRes",
        "Low-resolution arrays": "Arrays_LowRes",
    }
    text = aliases.get(text, text)
    return text if text in PLATFORM_ORDER else "Other"


def _load_metadata(source_dir: Path, exclude_roots: set[str]) -> pd.DataFrame:
    path = source_dir / "metadata" / "tumour_analysis_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing tumour metadata: {path}")
    meta = pd.read_parquet(path)
    required = {ID_COL, PLATFORM_COL}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise KeyError(f"Tumour metadata missing columns: {missing}")

    meta = meta.copy()
    meta[ID_COL] = meta[ID_COL].astype(str)
    if ROOT_COL not in meta.columns:
        if "icdo_topography_id" not in meta.columns:
            raise KeyError(f"Metadata must contain {ROOT_COL} or icdo_topography_id")
        meta[ROOT_COL] = meta["icdo_topography_id"].map(icdot_root_grouped)
    meta[ROOT_COL] = meta[ROOT_COL].fillna("UNMAPPED").astype(str)
    meta[PLATFORM_COL] = meta[PLATFORM_COL].map(_normalize_platform)
    meta = meta.drop_duplicates(ID_COL, keep="first")
    if exclude_roots:
        meta = meta[~meta[ROOT_COL].isin(exclude_roots)].copy()
    if meta.empty:
        raise ValueError("No tumour analyses remain after root filtering")
    return meta


def _root_palette(roots: list[str]) -> dict[str, tuple[float, float, float]]:
    ordered = sorted(set(str(root) for root in roots))
    try:
        import seaborn as sns

        colors = sns.color_palette("husl", n_colors=len(ordered))
    except ImportError:
        cmap = plt.get_cmap("turbo")
        colors = [cmap(i / max(1, len(ordered) - 1))[:3] for i in range(len(ordered))]
    return {root: colors[i] for i, root in enumerate(ordered)}


def _platform_counts(meta: pd.DataFrame) -> pd.DataFrame:
    counts = (
        meta.groupby([ROOT_COL, PLATFORM_COL], dropna=False)[ID_COL]
        .nunique()
        .unstack(fill_value=0)
    )
    for platform in [*PLATFORM_ORDER, "Other"]:
        if platform not in counts.columns:
            counts[platform] = 0
    counts = counts[[*PLATFORM_ORDER, "Other"]]
    counts["Total"] = counts.sum(axis=1)
    return counts.sort_values("Total", ascending=False).reset_index()


def _plot_platform_composition(
    counts: pd.DataFrame,
    *,
    out_path: Path,
    top_n: int | None,
) -> None:
    rows = counts.copy()
    if top_n is not None and len(rows) > int(top_n):
        top = rows.iloc[: int(top_n)].copy()
        rest = rows.iloc[int(top_n) :]
        other = {ROOT_COL: "OTHER_ROOTS"}
        for col in [*PLATFORM_ORDER, "Other", "Total"]:
            other[col] = int(rest[col].sum())
        rows = pd.concat([top, pd.DataFrame([other])], ignore_index=True)

    rows = rows.iloc[::-1].reset_index(drop=True)
    labels = [
        "Other remaining roots"
        if root == "OTHER_ROOTS"
        else ROOT_LABELS.get(str(root), str(root))
        for root in rows[ROOT_COL]
    ]
    platforms = [p for p in [*PLATFORM_ORDER, "Other"] if int(rows[p].sum()) > 0]

    height = 4.7 if top_n is not None else max(5.4, 0.34 * len(rows) + 1.3)
    fig, ax = plt.subplots(figsize=(13.2, height))
    left = np.zeros(len(rows), dtype=float)
    y = np.arange(len(rows))
    for platform in platforms:
        values = rows[platform].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.72,
            color=PLATFORM_COLORS[platform],
            edgecolor="white",
            linewidth=0.8,
            label=PLATFORM_LABELS[platform],
        )
        left += values

    pad = max(1.0, float(left.max()) * 0.008)
    for yy, total in zip(y, left, strict=False):
        ax.text(
            total + pad,
            yy,
            f"{int(total):,}",
            va="center",
            ha="left",
            fontsize=10.5,
            color=TEXT,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5 if top_n is not None else 9.0)
    ax.set_xlabel("Number of tumour analyses")
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(0, float(left.max()) * 1.13)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13 if top_n is not None else 1.09),
        ncol=len(platforms),
        frameon=False,
    )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _shape_summary(meta: pd.DataFrame, proxy_path: Path) -> pd.DataFrame:
    proxy = _read_table(proxy_path)
    if ID_COL not in proxy.columns:
        raise KeyError(f"Shape proxy table must contain {ID_COL}")
    required = {"cnv_fraction", "variant_count"}
    missing = sorted(required - set(proxy.columns))
    if missing:
        raise KeyError(f"Shape proxy table missing columns: {missing}")
    proxy = proxy.copy()
    proxy[ID_COL] = proxy[ID_COL].astype(str)
    for col in ["cnv_fraction", "variant_count"]:
        proxy[col] = pd.to_numeric(proxy[col], errors="coerce")
    joined = meta[[ID_COL, ROOT_COL]].merge(
        proxy[[ID_COL, "cnv_fraction", "variant_count"]],
        on=ID_COL,
        how="inner",
    )
    if joined.empty:
        raise ValueError("No overlap between tumour metadata and shape proxies")
    return (
        joined.groupby(ROOT_COL, as_index=False)
        .agg(
            n_shape_profiles=(ID_COL, "nunique"),
            median_cnv_fraction=("cnv_fraction", "median"),
            median_variant_count=("variant_count", "median"),
        )
        .sort_values("n_shape_profiles", ascending=False)
    )


def _compute_platform_metrics(
    *,
    meta: pd.DataFrame,
    cache_dir: Path,
    min_samples_per_family: int,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    files = sorted(cache_dir.glob("bucket=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No gene-panel cache parts found under {cache_dir}")

    family_counts = (
        meta.groupby([ROOT_COL, PLATFORM_COL])[ID_COL]
        .nunique()
        .rename("n_samples")
        .reset_index()
    )
    eligible = family_counts[family_counts["n_samples"] >= int(min_samples_per_family)]
    eligible_pairs = set(zip(eligible[ROOT_COL], eligible[PLATFORM_COL], strict=False))
    id_root = meta.set_index(ID_COL)[ROOT_COL].to_dict()
    id_platform = meta.set_index(ID_COL)[PLATFORM_COL].to_dict()

    partials: list[pd.DataFrame] = []
    columns = [
        ID_COL,
        "gene_symbol",
        "chrom",
        "dup_frac",
        "del_frac",
        "hldup_frac",
        "hldel_frac",
    ]
    print(
        "[CNVSIM] computing autosomal platform metrics from the gene-panel "
        "cache; this is a one-time derived-table step",
        flush=True,
    )
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=columns)
        if frame.empty:
            continue
        frame["chrom"] = (
            frame["chrom"]
            .astype(str)
            .str.replace("chr", "", regex=False)
        )
        frame = frame[frame["chrom"].isin({str(i) for i in range(1, 23)})]
        if frame.empty:
            continue
        frame[ID_COL] = frame[ID_COL].astype(str)
        frame[ROOT_COL] = frame[ID_COL].map(id_root)
        frame[PLATFORM_COL] = frame[ID_COL].map(id_platform)
        frame = frame.dropna(subset=[ROOT_COL, PLATFORM_COL, "gene_symbol"])
        if frame.empty:
            continue
        keep = pd.MultiIndex.from_frame(frame[[ROOT_COL, PLATFORM_COL]]).isin(eligible_pairs)
        frame = frame.loc[keep].copy()
        if frame.empty:
            continue
        for col in ["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        frame["any_cnv"] = (
            (frame["dup_frac"] >= float(low_threshold))
            | (frame["del_frac"] >= float(low_threshold))
            | (frame["hldup_frac"] >= float(high_threshold))
            | (frame["hldel_frac"] >= float(high_threshold))
        )
        per_sample = (
            frame.groupby([ROOT_COL, PLATFORM_COL, "gene_symbol", ID_COL], as_index=False)["any_cnv"]
            .max()
        )
        partials.append(
            per_sample.groupby([ROOT_COL, PLATFORM_COL, "gene_symbol"], as_index=False)
            .agg(n_supporting=("any_cnv", "sum"), n_evaluable=(ID_COL, "nunique"))
        )
        if index % 25 == 0 or index == len(files):
            print(f"[CNVSIM] gene-panel cache parts: {index}/{len(files)}", flush=True)

    if not partials:
        raise ValueError("No eligible gene-panel rows were found")
    support = pd.concat(partials, ignore_index=True)
    support = (
        support.groupby([ROOT_COL, PLATFORM_COL, "gene_symbol"], as_index=False)
        .agg(n_supporting=("n_supporting", "sum"), n_evaluable=("n_evaluable", "sum"))
    )
    support["frequency"] = support["n_supporting"] / support["n_evaluable"].replace(0, np.nan)

    rows: list[dict[str, object]] = []
    root_sizes = meta.groupby(ROOT_COL)[ID_COL].nunique().to_dict()
    for root, root_support in support.groupby(ROOT_COL):
        matrix = root_support.pivot(index="gene_symbol", columns=PLATFORM_COL, values="frequency")
        if matrix.shape[1] < 2:
            continue
        delta = matrix.max(axis=1, skipna=True) - matrix.min(axis=1, skipna=True)
        complete = matrix.dropna(axis=0, how="any")
        complete = complete.loc[complete.var(axis=1) > 0]
        if complete.shape[0] < 10:
            continue
        corr = complete.corr(method="pearson", min_periods=10)
        values = corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        rows.append(
            {
                ROOT_COL: str(root),
                "feature_scope": "autosomes",
                "avg_corr_offdiag": float(finite_values.mean()),
                "median_delta_freq": float(delta.median(skipna=True)),
                "n_genes": int(matrix.shape[0]),
                "n_complete_genes": int(complete.shape[0]),
                "n_platform_families": int(matrix.shape[1]),
                "n_samples": int(root_sizes.get(root, 0)),
            }
        )
    return (
        pd.DataFrame(rows, columns=PLATFORM_METRIC_COLUMNS)
        .sort_values("n_samples", ascending=False)
        .reset_index(drop=True)
    )


def _plot_panel_b(
    *,
    shape: pd.DataFrame,
    platform: pd.DataFrame,
    counts: pd.DataFrame,
    out_path: Path,
) -> None:
    roots = counts[ROOT_COL].astype(str).tolist()
    palette = _root_palette(roots)
    size_map = counts.set_index(ROOT_COL)["Total"].to_dict()
    max_n = max(size_map.values())

    fig = plt.figure(figsize=(13.6, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.62], wspace=0.25)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax_legend = fig.add_subplot(grid[0, 2])
    ax_legend.axis("off")

    def point_size(root: str) -> float:
        return 55.0 + 290.0 * np.sqrt(float(size_map.get(root, 0)) / max(1.0, float(max_n)))

    x_values: list[float] = []
    y_values: list[float] = []
    for _, row in shape.iterrows():
        root = str(row[ROOT_COL])
        x = float(np.log1p(row["median_variant_count"]))
        y = float(row["median_cnv_fraction"])
        x_values.append(x)
        y_values.append(y)
        ax_left.scatter(x, y, s=point_size(root), color=palette[root], edgecolor="white", linewidth=0.5, zorder=3)
    if len(x_values) >= 3:
        coeff = np.polyfit(np.asarray(x_values), np.asarray(y_values), deg=1)
        xx = np.linspace(min(x_values), max(x_values), 100)
        ax_left.plot(xx, np.polyval(coeff, xx), color="0.35", linewidth=1.2, zorder=2)
    ax_left.set_xlabel("log(1 + median CNV event count per analysis)")
    ax_left.set_ylabel("Median genome fraction altered by CNVs")
    ax_left.grid(alpha=0.22)
    ax_left.text(
        0.03,
        0.54,
        "High genome-wide alteration\nHigh event count",
        transform=ax_left.transAxes,
        fontsize=7.5,
        color=TEXT,
        va="center",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.7},
    )

    platform_plot = platform.copy()
    for column in ["avg_corr_offdiag", "median_delta_freq"]:
        platform_plot[column] = pd.to_numeric(platform_plot[column], errors="coerce")
    platform_plot = platform_plot.dropna(
        subset=["avg_corr_offdiag", "median_delta_freq"]
    )

    for _, row in platform_plot.iterrows():
        root = str(row[ROOT_COL])
        ax_right.scatter(
            float(row["avg_corr_offdiag"]),
            float(row["median_delta_freq"]),
            s=point_size(root),
            color=palette[root],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    if not platform_plot.empty:
        x = platform_plot["avg_corr_offdiag"].to_numpy(dtype=float)
        y = platform_plot["median_delta_freq"].to_numpy(dtype=float)
        x25, x75 = np.nanquantile(x, [0.25, 0.75])
        y25, y75 = np.nanquantile(y, [0.25, 0.75])
        ax_right.axvspan(x25, x75, color="0.95", zorder=0)
        ax_right.axhline(y25, color="0.75", linewidth=0.8, linestyle="--")
        ax_right.axhline(y75, color="0.75", linewidth=0.8, linestyle="--")
        ax_right.text(
            0.76,
            0.92,
            "Low concordance\nHigh divergence",
            transform=ax_right.transAxes,
            fontsize=7.5,
            color=TEXT,
            va="top",
        )
    ax_right.set_xlabel("Pearson correlation of gene-level CNV frequencies\nacross platform families")
    ax_right.set_ylabel("Median Delta (max-min) in gene-level CNV frequencies\nacross platform families")
    ax_right.grid(alpha=0.22)

    for axis in [ax_left, ax_right]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[root],
            markeredgecolor="white",
            markersize=5.2,
            label=ROOT_LABELS.get(root, root),
        )
        for root in roots
    ]
    ax_legend.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=6.1,
        labelspacing=0.55,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tight_crop(image: Image.Image, tolerance: int = 248, pad: int = 6) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    mask = (array < tolerance).any(axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return image.convert("RGB")
    return image.convert("RGB").crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + 1 + pad),
            min(image.height, int(ys.max()) + 1 + pad),
        )
    )


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _combine_panels(panel_a: Path, panel_b: Path, out_path: Path) -> None:
    a = _tight_crop(Image.open(panel_a))
    b = _tight_crop(Image.open(panel_b))
    canvas_width = 3300
    left_margin = 82
    right_margin = 28
    panel_width = canvas_width - left_margin - right_margin
    a = _resize_width(a, int(panel_width * 0.95))
    b = _resize_width(b, panel_width)
    top = 18
    between = 28
    bottom = 18
    canvas = Image.new("RGB", (canvas_width, top + a.height + between + b.height + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(58, bold=True)
    x_a = left_margin + (panel_width - a.width) // 2
    canvas.paste(a, (x_a, top))
    draw.text((12, top + 8), "A", fill=(25, 25, 25), font=label_font)
    y_b = top + a.height + between
    canvas.paste(b, (left_margin, y_b))
    draw.text((12, y_b + 8), "B", fill=(25, 25, 25), font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Source-data directory used to build the atlas",
    )
    parser.add_argument("--shape-proxies", required=True, type=Path, help="Per-analysis segment-derived CNV burden table")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10, help="Roots shown individually in main panel A")
    parser.add_argument("--exclude-roots", default="", help="Optional comma-separated roots to omit")
    parser.add_argument("--min-samples-per-family", type=int, default=50)
    parser.add_argument("--low-threshold", type=float, default=0.10)
    parser.add_argument("--high-threshold", type=float, default=0.20)
    parser.add_argument(
        "--platform-metrics",
        type=Path,
        help=(
            "Optional precomputed autosomal platform-consistency TSV. "
            "Supplying this avoids rescanning the gene-panel cache."
        ),
    )
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_roots = {root.strip() for root in args.exclude_roots.split(",") if root.strip()}

    meta = _load_metadata(source_dir, exclude_roots)
    counts = _platform_counts(meta)
    counts_path = out_dir / "figure1_platform_composition_counts.tsv"
    counts.to_csv(counts_path, sep="\t", index=False)

    panel_a = out_dir / "figure1A_platform_composition_top_roots.png"
    panel_a_full = out_dir / "figureS1_platform_composition_all_roots.png"
    _plot_platform_composition(counts, out_path=panel_a, top_n=int(args.top_n))
    _plot_platform_composition(counts, out_path=panel_a_full, top_n=None)

    shape = _shape_summary(meta, args.shape_proxies.expanduser().resolve())
    shape_path = out_dir / "figure1B_root_cnv_shape_summary.tsv"
    shape.to_csv(shape_path, sep="\t", index=False)

    platform_path = out_dir / "figure1B_platform_consistency.tsv"
    platform_source: str
    if args.platform_metrics is not None:
        supplied_path = args.platform_metrics.expanduser().resolve()
        if not supplied_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed platform metrics: {supplied_path}"
            )
        platform = pd.read_csv(supplied_path, sep="\t")
        platform_source = str(supplied_path)
    elif platform_path.exists():
        platform = pd.read_csv(platform_path, sep="\t")
        platform_source = str(platform_path)
    else:
        platform = _compute_platform_metrics(
            meta=meta,
            cache_dir=source_dir / "gene_panel" / "cache",
            min_samples_per_family=int(args.min_samples_per_family),
            low_threshold=float(args.low_threshold),
            high_threshold=float(args.high_threshold),
        )
        platform_source = "computed_from_gene_panel_cache"

    required_platform_columns = set(PLATFORM_METRIC_COLUMNS) - {
        "feature_scope"
    }
    missing_platform_columns = sorted(
        required_platform_columns - set(platform.columns)
    )
    if missing_platform_columns:
        raise KeyError(
            "Platform metrics missing columns: "
            f"{missing_platform_columns}"
        )
    if "feature_scope" in platform.columns:
        scopes = set(platform["feature_scope"].dropna().astype(str))
        if scopes != {"autosomes"}:
            raise ValueError(
                "Platform metrics must use feature_scope=autosomes; "
                f"found {sorted(scopes)}"
            )
    else:
        print(
            "[CNVSIM] warning: supplied platform metrics do not record "
            "feature_scope; recompute before the final release",
            flush=True,
        )
    platform = platform[
        platform[ROOT_COL].astype(str).isin(set(meta[ROOT_COL].astype(str)))
    ].copy()
    platform.to_csv(platform_path, sep="\t", index=False)

    panel_b = out_dir / "figure1B_root_cnv_platform_diagnostics.png"
    _plot_panel_b(shape=shape, platform=platform, counts=counts, out_path=panel_b)

    combined = out_dir / "figure1_preprocess_overview.png"
    _combine_panels(panel_a, panel_b, combined)

    manifest = {
        "source_dir": str(source_dir),
        "shape_proxies": str(args.shape_proxies.expanduser().resolve()),
        "cohort_definition": "all tumour analyses in metadata/tumour_analysis_metadata.parquet",
        "n_tumour_analyses": int(meta[ID_COL].nunique()),
        "n_roots": int(meta[ROOT_COL].nunique()),
        "excluded_roots": sorted(exclude_roots),
        "thresholds": {
            "min_samples_per_family": int(args.min_samples_per_family),
            "low_fraction": float(args.low_threshold),
            "high_fraction": float(args.high_threshold),
        },
        "platform_metrics_source": platform_source,
        "platform_metrics_feature_scope": "autosomes",
        "outputs": {
            "panel_a": panel_a.name,
            "panel_a_all_roots": panel_a_full.name,
            "panel_b": panel_b.name,
            "combined": combined.name,
            "platform_counts": counts_path.name,
            "shape_summary": shape_path.name,
            "platform_consistency": platform_path.name,
        },
    }
    (out_dir / "figure1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[OK] tumour analyses: {manifest['n_tumour_analyses']:,}")
    print(f"[OK] roots: {manifest['n_roots']}")
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
