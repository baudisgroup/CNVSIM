#!/usr/bin/env python3
from __future__ import annotations

"""Visualize the first-pass removal of likely technical sample groups.

For roots with removed samples, the script plots the pre-filter UMAP, platform
composition of retained and removed samples, and segment-derived CNV burden.
All panels are generated from one preprocessing run and one shape-proxy table.
"""

import argparse
import json
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
from PIL import Image, ImageDraw, ImageFont


ID_COL = "analysis_id"
PLATFORM_COL = "platform_family_refined"
PLATFORM_ORDER = ["Arrays_HighRes", "Arrays_LowRes", "NGS_Based", "Other"]
PLATFORM_COLORS = {
    "Arrays_HighRes": "#4DB6AC",
    "Arrays_LowRes": "#FF8A65",
    "NGS_Based": "#7986CB",
    "Other": "#BDBDBD",
}
PLATFORM_LABELS = {
    "Arrays_HighRes": "High-resolution arrays",
    "Arrays_LowRes": "Lower-resolution array/CGH",
    "NGS_Based": "NGS-based",
    "Other": "Other/unknown",
}
ROOT_LABELS = {
    "C00_C14": "Oral/pharynx\n(C00-C14)",
    "C15_C17_C21_C26": "Other digestive\n(C15/C17/C21/C26)",
    "C16": "Stomach\n(C16)",
    "C18_C20": "Colon/rectum\n(C18-C20)",
    "C22_C24": "Liver/biliary\n(C22-C24)",
    "C25": "Pancreas\n(C25)",
    "C30_C39": "Respiratory\n(C30-C39)",
    "C40_C41_C48": "Bone/connective\n(C40/C41/C48)",
    "C42": "Haematopoietic/reticuloendothelial\n(C42)",
    "C44": "Skin\n(C44)",
    "C47": "Peripheral nerves\n(C47)",
    "C49": "Soft tissue\n(C49)",
    "C50": "Breast\n(C50)",
    "C51_C58": "Female genital\n(C51-C58)",
    "C60_C63": "Male genital\n(C60-C63)",
    "C64_C68": "Urinary tract\n(C64-C68)",
    "C69_C70_C72": "Eye/CNS\n(C69/C70/C72)",
    "C71": "Brain\n(C71)",
    "C73_C75": "Endocrine\n(C73-C75)",
    "C76": "Ill-defined sites\n(C76)",
    "C77": "Lymph nodes\n(C77)",
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _normalize_platform(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Panels_NGS": "NGS_Based",
        "NGS": "NGS_Based",
        "NGS_panel": "NGS_Based",
        "NGS_panels": "NGS_Based",
    }
    text = aliases.get(text, text)
    return text if text in PLATFORM_ORDER[:-1] else "Other"


def _analysis_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if ID_COL not in out.columns:
        if out.index.name == ID_COL:
            out = out.reset_index()
        else:
            out = out.reset_index().rename(columns={"index": ID_COL})
    out[ID_COL] = out[ID_COL].astype(str)
    return out


def _umap_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for pair in [("UMAP1", "UMAP2"), ("umap1", "umap2"), ("x", "y")]:
        if pair[0] in frame.columns and pair[1] in frame.columns:
            return pair
    numeric = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if len(numeric) < 2:
        raise ValueError(f"Cannot infer UMAP columns from {frame.columns.tolist()}")
    return numeric[0], numeric[1]


def _removed_ids(stage1_dir: Path, root: str) -> set[str]:
    path = stage1_dir / f"{root}_removed_samples.tsv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if ID_COL not in frame.columns:
        return set()
    return set(frame[ID_COL].dropna().astype(str))


def _discover_roots(stage1_dir: Path) -> list[str]:
    roots: list[str] = []
    for path in sorted(stage1_dir.glob("*_removed_samples.tsv")):
        root = path.name.replace("_removed_samples.tsv", "")
        if _removed_ids(stage1_dir, root):
            roots.append(root)
    return roots


def _load_root_data(
    *,
    run_dir: Path,
    root: str,
    proxy: pd.DataFrame,
) -> pd.DataFrame:
    stage1_dir = run_dir / "final_space" / "stage1"
    meta_path = run_dir / "features_unfiltered" / f"{root}_meta.tsv"
    umap_path = stage1_dir / f"{root}_UMAP_firstpass.parquet"
    if not meta_path.exists() or not umap_path.exists():
        raise FileNotFoundError(f"Missing first-pass inputs for {root}")

    meta = pd.read_csv(meta_path, sep="\t", dtype=str, low_memory=False)
    meta[ID_COL] = meta[ID_COL].astype(str)
    meta[PLATFORM_COL] = meta[PLATFORM_COL].map(_normalize_platform)
    umap = _analysis_id_column(pd.read_parquet(umap_path))
    x_col, y_col = _umap_columns(umap)
    umap = umap.rename(columns={x_col: "UMAP1", y_col: "UMAP2"})

    frame = umap.merge(meta[[ID_COL, PLATFORM_COL]], on=ID_COL, how="left")
    frame["group"] = np.where(
        frame[ID_COL].isin(_removed_ids(stage1_dir, root)),
        "removed",
        "retained",
    )
    proxy_cols = [ID_COL, "cnv_fraction", "variant_count"]
    frame = frame.merge(proxy[proxy_cols], on=ID_COL, how="left")
    frame["root"] = root
    return frame


def _plot_umap(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    roots = list(data)
    fig, axes = plt.subplots(1, len(roots), figsize=(5.0 * len(roots), 4.2), squeeze=False)
    for axis, root in zip(axes[0], roots, strict=False):
        frame = data[root]
        for platform in PLATFORM_ORDER:
            subset = frame[frame[PLATFORM_COL] == platform]
            if subset.empty:
                continue
            axis.scatter(
                subset["UMAP1"],
                subset["UMAP2"],
                s=4,
                color=PLATFORM_COLORS[platform],
                alpha=0.70,
                linewidths=0,
                rasterized=True,
            )
        removed = frame[frame["group"] == "removed"]
        if not removed.empty:
            axis.scatter(
                removed["UMAP1"],
                removed["UMAP2"],
                s=28,
                color="#D32F2F",
                edgecolor="black",
                linewidth=0.7,
                alpha=0.90,
                zorder=5,
            )
        axis.set_title(ROOT_LABELS.get(root, root))
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        # UMAP coordinates are embedding-specific and have no interpretable
        # numerical scale; retain axis names but omit numeric tick labels.
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(alpha=0.10)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PLATFORM_COLORS[p], markeredgecolor="none", label=PLATFORM_LABELS[p], markersize=7)
        for p in PLATFORM_ORDER
        if any((frame[PLATFORM_COL] == p).any() for frame in data.values())
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D32F2F", markeredgecolor="black", label="Removed", markersize=7)
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, top=0.82, wspace=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_platform_composition(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    roots = list(data)
    fig, axes = plt.subplots(1, len(roots), figsize=(5.6 * len(roots), 3.5), squeeze=False)
    for axis, root in zip(axes[0], roots, strict=False):
        frame = data[root]
        counts = (
            frame.groupby(["group", PLATFORM_COL])[ID_COL]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=["retained", "removed"], columns=PLATFORM_ORDER, fill_value=0)
        )
        left = np.zeros(2, dtype=float)
        for platform in PLATFORM_ORDER:
            values = counts[platform].to_numpy(dtype=float)
            axis.barh(
                [0, 1],
                values,
                left=left,
                color=PLATFORM_COLORS[platform],
                edgecolor="white",
                height=0.65,
            )
            left += values
        max_total = float(left.max())
        label_offset = max(1.0, max_total * 0.012)
        for y, total in enumerate(left):
            axis.text(
                total + label_offset,
                y,
                f"{int(total):,}",
                va="center",
                fontsize=9,
            )
        # Keep count labels inside this axes rather than allowing them to
        # intrude into the neighbouring root panel.
        axis.set_xlim(0, max_total * 1.14 if max_total > 0 else 1.0)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["Retained", "Removed"])
        axis.invert_yaxis()
        axis.set_title(ROOT_LABELS.get(root, root))
        axis.set_xlabel("Number of analyses")
        axis.grid(axis="x", alpha=0.15)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles = [Patch(facecolor=PLATFORM_COLORS[p], edgecolor="none", label=PLATFORM_LABELS[p]) for p in PLATFORM_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.25, top=0.78, wspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_burden(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    roots = list(data)
    fig, axes = plt.subplots(1, len(roots), figsize=(5.0 * len(roots), 3.8), squeeze=False)
    for axis, root in zip(axes[0], roots, strict=False):
        frame = data[root].copy()
        for col in ["cnv_fraction", "variant_count"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        retained = frame[(frame["group"] == "retained") & (frame["variant_count"] > 0)].copy()
        removed = frame[(frame["group"] == "removed") & (frame["variant_count"] > 0)].copy()
        axis.scatter(
            retained["cnv_fraction"],
            retained["variant_count"],
            s=8,
            color="0.55",
            alpha=0.05,
            linewidths=0,
            rasterized=True,
        )

        if not retained.empty:
            edges = np.linspace(0.0, 1.0, 27)
            centres = 0.5 * (edges[:-1] + edges[1:])
            q25 = np.full(len(centres), np.nan)
            q50 = np.full(len(centres), np.nan)
            q75 = np.full(len(centres), np.nan)
            log_counts = np.log10(retained["variant_count"].to_numpy(dtype=float))
            fractions = retained["cnv_fraction"].to_numpy(dtype=float)
            for index in range(len(centres)):
                mask = (fractions >= edges[index]) & (fractions < edges[index + 1]) & np.isfinite(log_counts)
                if mask.sum() >= 30:
                    q25[index], q50[index], q75[index] = np.quantile(log_counts[mask], [0.25, 0.50, 0.75])
            valid = np.isfinite(q25) & np.isfinite(q75)
            axis.fill_between(centres[valid], 10 ** q25[valid], 10 ** q75[valid], color="0.55", alpha=0.35)
            axis.plot(centres[valid], 10 ** q50[valid], color="0.15", linewidth=2.2)

        axis.scatter(
            removed["cnv_fraction"],
            removed["variant_count"],
            s=28,
            color="#D32F2F",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=5,
        )
        axis.set_yscale("log")
        axis.set_xlabel("Genome fraction altered by CNVs")
        axis.set_ylabel("CNV event count (log scale)")
        axis.set_title(ROOT_LABELS.get(root, root))
        axis.grid(alpha=0.12)
    handles = [
        Patch(facecolor="0.55", edgecolor="none", alpha=0.35, label="Retained IQR"),
        Line2D([0], [0], color="0.15", linewidth=2.2, label="Retained median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D32F2F", markeredgecolor="black", label="Removed", markersize=7),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.25, top=0.80, wspace=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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


def _combine(paths: list[Path], labels: list[str], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    resized = [
        image if image.width == width else image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        for image in images
    ]
    margin = 72
    gap = 24
    canvas = Image.new("RGB", (width + margin, sum(image.height for image in resized) + gap * (len(resized) - 1)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(46, bold=True)
    y = 0
    for image, label in zip(resized, labels, strict=False):
        canvas.paste(image, (margin, y))
        draw.text((12, y + 12), label, fill=(25, 25, 25), font=font)
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Preprocessing work directory for one run")
    parser.add_argument("--shape-proxies", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--roots", default="", help="Comma-separated roots; default: all roots with removed samples")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = run_dir / "final_space" / "stage1"
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Missing Stage 1 directory: {stage1_dir}")
    roots = [root.strip() for root in args.roots.split(",") if root.strip()] or _discover_roots(stage1_dir)
    if not roots:
        raise ValueError("No roots with first-pass removals were found")

    proxy = _read_table(args.shape_proxies.expanduser().resolve())
    if ID_COL not in proxy.columns:
        raise KeyError(f"Shape proxy table must contain {ID_COL}")
    proxy[ID_COL] = proxy[ID_COL].astype(str)
    for col in ["cnv_fraction", "variant_count"]:
        if col not in proxy.columns:
            raise KeyError(f"Shape proxy table missing {col}")
        proxy[col] = pd.to_numeric(proxy[col], errors="coerce")

    data = {root: _load_root_data(run_dir=run_dir, root=root, proxy=proxy) for root in roots}
    umap_path = out_dir / "firstpass_platform_umap.png"
    composition_path = out_dir / "firstpass_platform_composition.png"
    burden_path = out_dir / "firstpass_cnv_burden.png"
    combined_path = out_dir / "firstpass_filter_qc.png"
    _plot_umap(data, umap_path)
    _plot_platform_composition(data, composition_path)
    _plot_burden(data, burden_path)
    _combine([umap_path, composition_path, burden_path], ["A", "B", "C"], combined_path)

    rows = []
    for root, frame in data.items():
        rows.append(
            {
                "icdot_root": root,
                "n_total": int(frame[ID_COL].nunique()),
                "n_removed": int(frame.loc[frame["group"] == "removed", ID_COL].nunique()),
                "n_retained": int(frame.loc[frame["group"] == "retained", ID_COL].nunique()),
                "n_removed_with_shape_proxy": int(
                    frame.loc[
                        (frame["group"] == "removed")
                        & frame["cnv_fraction"].notna()
                        & frame["variant_count"].notna(),
                        ID_COL,
                    ].nunique()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "firstpass_filter_qc_summary.tsv", sep="\t", index=False)
    (out_dir / "firstpass_filter_qc_manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "shape_proxies": str(args.shape_proxies.expanduser().resolve()),
                "roots": roots,
                "outputs": {
                    "umap": umap_path.name,
                    "platform_composition": composition_path.name,
                    "cnv_burden": burden_path.name,
                    "combined": combined_path.name,
                    "summary": "firstpass_filter_qc_summary.tsv",
                },
            },
            indent=2,
        )
    )
    print(f"[OK] roots: {', '.join(roots)}")
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
