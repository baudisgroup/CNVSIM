#!/usr/bin/env python3
"""Render Figure 7 from frozen CNVSIM summaries and CNV profiles.

Default outputs: ATLAS_ROOT/figures/figure7/Figure7.pdf and Figure7.png.
Use --out-dir to override and --overwrite to replace existing files.
Only these two files are generated; no models are fitted.
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

WIDTH_MM, HEIGHT_MM = 183.0, 230.0
BLUE, ORANGE, TEXT = "#2E5EAA", "#D9822B", "#111111"
OTHER_MATCHED = "#A9C4E6"
GREY, GRID, GAIN, LOSS = "#E5EAEC", "#E4E9EB", "#F2BE35", "#1F5F93"
PROVISIONAL = "#DDBB7B"
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
CHROM_LENGTHS = [
    248956422,
    242193529,
    198295559,
    190214555,
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
]
OFFSETS = np.r_[0, np.cumsum(CHROM_LENGTHS)]
TOTAL = float(OFFSETS[-1])
MIDS = (OFFSETS[:-1] + np.array(CHROM_LENGTHS) / 2) / TOTAL
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
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, value, **kw)


def axis(fig, x, y, w, h, label):
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM], label=label)


def root_axis(ax):
    ax.set_ylim(18.65, -0.65)
    ax.set_yticks(range(19), list(ROOTS.values()))
    ax.tick_params(axis="y", length=0, pad=3, labelsize=6)
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.5, pad=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#95A2A8")


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


def encoded_gene_profile(path, ids, genes, *, reference=False):
    columns = ["analysis_id", "gene_symbol", "dup_frac", "del_frac", "hldup_frac", "hldel_frac"]
    raw = pd.read_parquet(path, columns=columns, filters=[("analysis_id", "in", list(ids))])
    assert set(raw.analysis_id) == set(ids), "A selected profile is absent"
    assert not raw.duplicated(["analysis_id", "gene_symbol"]).any()
    assert raw.groupby("analysis_id").gene_symbol.nunique().eq(len(genes)).all()
    vals = raw[["dup_frac", "del_frac", "hldup_frac", "hldel_frac"]].to_numpy(float)
    assert np.isfinite(vals).all(), "Do not silently discard unknown fractions"
    raw["gain"] = (
        (raw.dup_frac >= 0.1).astype(float) + 2 * (raw.hldup_frac >= 0.2).astype(float)
    ) / 3
    raw["loss"] = (
        (raw.del_frac >= 0.1).astype(float) + 2 * (raw.hldel_frac >= 0.2).astype(float)
    ) / 3
    means = raw.groupby("gene_symbol")[["gain", "loss"]].mean()
    result = genes.merge(
        means, left_on="gene_symbol", right_index=True, how="left", validate="one_to_one"
    )
    assert result[["gain", "loss"]].notna().all().all()
    if reference:
        for event in ["gain", "loss"]:
            result.loc[result[event] < 0.05, event] = 0
    return result


def profile_image(frame):
    bins = np.minimum(1099, np.floor(frame.position.to_numpy() * 1100).astype(int))
    result = np.ones((2, 1100, 3), float)
    for row, (event, color) in enumerate([("gain", GAIN), ("loss", LOSS)]):
        values = pd.DataFrame({"bin": bins, "value": frame[event]}).groupby("bin").value.max()
        for b, value in values.items():
            if value > 0:
                weight = 0.24 + 0.76 * float(value)
                result[row, b, :] = 1 + (np.array(to_rgb(color)) - 1) * weight
    return result


def draw_profile(fig, frame, *, x, y, w, label, name, chromosomes=False):
    ax = axis(fig, x, y, w, 8, name)
    ax.imshow(
        profile_image(frame),
        extent=[0, 1, 0, 2],
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    for pos in OFFSETS / TOTAL:
        ax.axvline(pos, color="#D8D5CC", lw=0.25)
    ax.axhline(1, color="#D8D5CC", lw=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#BFB7A8")
        spine.set_linewidth(0.4)
    labelled_chromosomes = [1, 5, 9, 13, 17, 22]
    ax.set_xticks(
        [MIDS[i - 1] for i in labelled_chromosomes],
        [str(i) for i in labelled_chromosomes] if chromosomes else [""] * 6,
    )
    ax.tick_params(axis="x", length=0, labelsize=6, pad=2)
    text(fig, x, y + 9.8, label, fontsize=6, va="center")
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path)
    parser.add_argument("--out-dir", type=Path, help="Default: ATLAS_ROOT/figures/figure7")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    p = args.release_dir.expanduser().resolve()
    if args.atlas_root is None and args.out_dir is None:
        parser.error("Supply --atlas-root or --out-dir")
    args.atlas_root = args.atlas_root.expanduser().resolve() if args.atlas_root else None
    out = (
        args.out_dir.expanduser().resolve() if args.out_dir else args.atlas_root / "figures/figure7"
    )
    existing = [
        out / f"Figure7.{ext}" for ext in ("pdf", "png") if (out / f"Figure7.{ext}").exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))
    out.mkdir(parents=True, exist_ok=True)
    availability = pd.read_csv(p / "cell_line/subtype_model_availability.tsv", sep="\t")
    assert len(availability) == 112 and availability.reference_key.nunique() == 112
    assert availability.coverage_status.value_counts().to_dict() == {
        "recommended_model_available": 90,
        "not_covered": 16,
        "provisional_model_only": 6,
    }
    signatures = pd.read_csv(
        p / "cell_line/signature_evidence/subtype_signature_coverage.tsv", sep="\t"
    )
    evidence = pd.read_csv(
        p / "cell_line/signature_evidence/model_signature_evidence.tsv", sep="\t"
    )
    higher = signatures[
        signatures.track.eq("track2") & signatures.relative_to_root_background.eq("higher")
    ].copy()
    selected = higher[higher.recommended_models.gt(0)].copy()
    selected["recorded"] = (
        selected.recommended_recorded_only.fillna(0) + selected.recommended_mixed.fillna(0)
    ).gt(0)
    assert len(higher) == 2942 and len(selected) == 2210 and int(selected.recorded.sum()) == 1772
    signature_keys = [
        "reference_key",
        "root",
        "track",
        "feature",
        "event",
        "amplitude",
        "relative_to_root_background",
    ]
    assert not selected.duplicated(signature_keys).any()
    recorded_keys = evidence.loc[
        evidence.recorded_profiles.fillna(0).gt(0), signature_keys
    ].drop_duplicates()
    selected = selected.merge(
        recorded_keys.assign(any_matched_recorded=True),
        on=signature_keys,
        how="left",
        validate="one_to_one",
    )
    selected["any_matched_recorded"] = selected.any_matched_recorded.eq(True)
    assert selected.loc[selected.recorded, "any_matched_recorded"].all()
    selected["other_matched_only"] = selected.any_matched_recorded & ~selected.recorded
    selected["not_recorded"] = ~selected.any_matched_recorded
    assert int(selected.other_matched_only.sum()) == 66
    assert int(selected.not_recorded.sum()) == 372
    assert selected[["recorded", "other_matched_only", "not_recorded"]].sum(axis=1).eq(1).all()
    events_by_root = (
        selected.groupby("root")
        .agg(
            n=("recorded", "size"),
            recorded=("recorded", "sum"),
            other_matched_only=("other_matched_only", "sum"),
            not_recorded=("not_recorded", "sum"),
        )
        .reindex(ROOTS)
        .fillna(0)
        .astype(int)
    )
    models = pd.read_csv(p / "cell_line/cell_lines.tsv", sep="\t")
    assignments = pd.read_parquet(
        p / "atlas/analysis_assignments.parquet",
        columns=["analysis_id", "root", "pipeline_status", "assigned_reference_key"],
    )
    genes = pd.read_csv(p / "source_data/source_file/gene_cnv_fractions/genes.tsv", sep="\t")
    genes["chrom"] = genes.chrom.astype(str).str.replace("chr", "", regex=False).astype(int)
    assert genes.chrom.between(1, 22).all() and len(genes) == 1263
    genes["position"] = (
        genes.chrom.map({i + 1: v for i, v in enumerate(OFFSETS[:-1])})
        + (genes.start + genes.end) / 2
    ) / TOTAL
    assert genes.position.between(0, 1).all()
    genes = genes.sort_values("position")
    profiles = []
    for root, ref, model_id, label in [
        ("C44", "C44-CTX5", "cellosaurus:CVCL_1779", "UACC-257"),
        ("C18_C20", "C18_C20-CTX12", "cellosaurus:CVCL_1718", "SW1463"),
    ]:
        model = models[
            models.model_id.eq(model_id)
            & models.assigned_reference_key.eq(ref)
            & models.is_recommended
        ]
        assert len(model) == 1
        ids = (
            assignments.loc[
                assignments.root.eq(root)
                & assignments.assigned_reference_key.eq(ref)
                & assignments.pipeline_status.isin(
                    ["anchor_member", "conditional_definition_member"]
                ),
                "analysis_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        assert ids
        ref_profile = encoded_gene_profile(
            p / f"source_data/source_file/gene_cnv_fractions/tumour/{root}.parquet",
            ids,
            genes,
            reference=True,
        )
        model_profile = encoded_gene_profile(
            p / f"source_data/source_file/gene_cnv_fractions/cell_line/{root}.parquet",
            [str(model.iloc[0].representative_analysis_id)],
            genes,
        )
        profiles.append((ref, label, len(ids), ref_profile, model_profile))
    d = higher[higher.reference_key.eq("C25-CTX6")].copy()
    positions = genes[["gene_symbol", "chrom", "start"]].rename(
        columns={"gene_symbol": "feature", "chrom": "plot_chrom", "start": "plot_start"}
    )
    d = (
        d.merge(positions, on="feature", validate="one_to_one")
        .sort_values(["plot_chrom", "plot_start", "feature"])
        .reset_index(drop=True)
    )
    assert len(d) == 31 and d.feature.nunique() == 31
    assert d.groupby(["plot_chrom", "event"]).size().to_dict() == {
        (9, "gain"): 1,
        (11, "loss"): 28,
        (14, "gain"): 2,
    }
    all_model_evidence = evidence
    evidence = evidence[
        evidence.reference_key.eq("C25-CTX6")
        & evidence.track.eq("track2")
        & evidence.relative_to_root_background.eq("higher")
        & evidence.is_recommended
    ].copy()
    model_ids = ["cellosaurus:CVCL_0186", "cellosaurus:CVCL_0334", "cellosaurus:CVCL_4012"]
    model_labels = ["BxPC-3", "Hs 766T", "PancTu-I"]
    assert len(evidence) == 93 and set(evidence.model_id) == set(model_ids)
    states = evidence.pivot(index="feature", columns="model_id", values="evidence_status").reindex(
        index=d.feature, columns=model_ids
    )
    assert (
        states.notna().all().all()
        and states.isin(["recorded_only", "not_recorded_only"]).all().all()
    )
    recorded = states.eq("recorded_only")
    assert int(recorded.any(axis=1).sum()) == 2 and not recorded.loc["BIRC3"].any()
    assert np.isclose(
        d.loc[d.feature.eq("BIRC3"), "tumour_context_value"].iloc[0], 0.789954, atol=1e-6
    )
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300, facecolor="white")
    axes = []
    rows = []
    columns = []
    text(fig, 3, 225, "A", fontsize=10, weight="bold", va="center")
    ax = axis(fig, 30, 136, 66, 80, "A_availability")
    root_axis(ax)
    ax.set_xlim(0.3, 19.7)
    colours = {
        "recommended_model_available": BLUE,
        "provisional_model_only": PROVISIONAL,
        "not_covered": GREY,
    }
    tile_records = []
    for y, root in enumerate(ROOTS):
        frame = availability[availability.root.eq(root)].copy()
        frame["number"] = frame.reference_key.str.extract(r"(\d+)$").astype(int)
        frame["layer_sort"] = frame.reference_layer.eq("conditional").astype(int)
        frame = frame.sort_values(["layer_sort", "number"])
        for index, row in enumerate(frame.itertuples(index=False), 1):
            ax.add_patch(
                Rectangle(
                    (index - 0.39, y - 0.32),
                    0.78,
                    0.64,
                    fc=colours[row.coverage_status],
                    ec="white",
                    lw=0.3,
                )
            )
            tile_records.append(
                {
                    "root": root,
                    "subtype": row.reference_key,
                    "position": index,
                    "status": row.coverage_status,
                }
            )
    assert len(tile_records) == 112
    ax.set_xticks([1, 5, 10, 15, 19])
    ax.set_xlabel("Reference subtypes", fontsize=6, labelpad=3)
    axes.append(ax)
    fig.legend(
        handles=[
            Patch(fc=BLUE, label="Recommended (90)"),
            Patch(fc=PROVISIONAL, label="Matched, not recommended (6)"),
            Patch(fc=GREY, label="No match (16)"),
        ],
        loc="center",
        bbox_to_anchor=(54 / WIDTH_MM, 225 / HEIGHT_MM),
        ncol=3,
        fontsize=6,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    text(fig, 104, 225, "B", fontsize=10, weight="bold", va="center")
    profile_axes = []
    for i, (ref, label, n, ref_profile, model_profile) in enumerate(profiles):
        top = 219 - 47 * i
        text(fig, 110, top, ref, fontsize=6.8, weight="bold", va="center")
        text(fig, 178, top, label, fontsize=6.5, weight="bold", ha="right", va="center")
        aa = draw_profile(
            fig,
            ref_profile,
            x=110,
            y=top - 16,
            w=68,
            label=f"Reference (CNV n={n:,})",
            name=f"B_{i}_reference",
        )
        bb = draw_profile(
            fig,
            model_profile,
            x=110,
            y=top - 29,
            w=68,
            label="Representative model profile",
            name=f"B_{i}_model",
            chromosomes=True,
        )
        profile_axes.extend([aa, bb])
        columns.append([aa.get_label(), bb.get_label()])
    axes.extend(profile_axes)
    # Separate within-example and between-example column comparisons: the
    # intentional gap between the two cases is larger than the within-case gap.
    columns.extend(
        [
            [profile_axes[0].get_label(), profile_axes[2].get_label()],
            [profile_axes[1].get_label(), profile_axes[3].get_label()],
        ]
    )
    fig.legend(
        handles=[Patch(fc=GAIN, label="Gain"), Patch(fc=LOSS, label="Loss")],
        loc="center",
        bbox_to_anchor=(144 / WIDTH_MM, 129 / HEIGHT_MM),
        ncol=2,
        fontsize=6,
        frameon=False,
        handlelength=1,
        columnspacing=1,
    )
    text(fig, 3, 120, "C", fontsize=10, weight="bold", va="center")
    ax = axis(fig, 30, 15, 66, 83, "C_events")
    root_axis(ax)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    for y, row in enumerate(events_by_root.itertuples()):
        if not row.n:
            ax.text(50, y, "—", ha="center", va="center", fontsize=6, color="#6E7B82")
            continue
        height = 0.30 if row.n < 10 else 0.55
        offset = 0.0
        for count, colour in [
            (row.recorded, BLUE),
            (row.other_matched_only, OTHER_MATCHED),
            (row.not_recorded, ORANGE),
        ]:
            width = 100 * count / row.n
            if count:
                ax.barh(
                    y,
                    width,
                    left=offset,
                    height=height,
                    color=colour,
                    edgecolor="white",
                    lw=0.25,
                )
            offset += width
        assert np.isclose(offset, 100)
    for y, row in enumerate(events_by_root.itertuples()):
        y_mm = 15 + 83 * (18.65 - y) / 19.3
        text(fig, 102, y_mm, str(row.n), ha="center", va="center", fontsize=6)
    ax.set_xlabel("Track 2 CNV signatures (%)", fontsize=6, labelpad=3)
    text(
        fig,
        63,
        106.5,
        "Track 2 CNV signatures (n = 2,210)",
        ha="center",
        va="center",
        fontsize=6.5,
        weight="bold",
    )
    text(fig, 102, 106.5, "n", ha="center", va="center", fontsize=6.5, weight="bold")
    fig.legend(
        handles=[
            Patch(fc=BLUE, label="Recommended (1,772)"),
            Patch(fc=OTHER_MATCHED, label="Other matched only (+66)"),
            Patch(fc=ORANGE, label="Not recorded (372)"),
        ],
        loc="center",
        bbox_to_anchor=(60 / WIDTH_MM, 113 / HEIGHT_MM),
        ncol=3,
        fontsize=6,
        frameon=False,
        handlelength=1,
        columnspacing=1.2,
    )
    axes.append(ax)
    text(fig, 104, 120, "D", fontsize=10, weight="bold", va="center")
    text(fig, 112, 115, "C25-CTX6", fontsize=7, weight="bold", va="center")
    text(fig, 112, 110, "Pancreatic neuroendocrine tumour: 55.0%", fontsize=6.4, va="center")
    text(fig, 112, 105, "Models: pancreatic adenocarcinoma (n = 3)", fontsize=6.2, va="center")
    t1 = signatures[signatures.reference_key.eq("C25-CTX6") & signatures.track.eq("track1")].copy()
    t1["chrom_number"] = t1.feature.str.replace("chr", "", regex=False).astype(int)
    t1 = t1.sort_values("chrom_number")
    assert len(t1) == 8 and t1.event.eq("loss").all()
    e1 = all_model_evidence[
        all_model_evidence.reference_key.eq("C25-CTX6")
        & all_model_evidence.track.eq("track1")
        & all_model_evidence.is_recommended
    ]
    assert len(e1) == 24 and e1.unknown_profiles.eq(0).all()
    e1 = e1.pivot(index="feature", columns="model_id", values="fraction_median").reindex(
        index=t1.feature, columns=model_ids
    )
    chrom_values = np.column_stack([t1.tumour_context_value.to_numpy(), e1.to_numpy()]) * 100
    assert (
        np.isfinite(chrom_values).all()
        and (chrom_values >= 0).all()
        and (chrom_values <= 100).all()
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("coverage", ["#F4F7F9", "#1E508B"])
    mat = axis(fig, 128, 44, 50, 54, "D_chromosomes")
    mat.imshow(chrom_values, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    mat.set_yticks(range(8), t1.feature)
    mat.set_xticks(range(4), ["Tumour", *model_labels])
    mat.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        labelsize=6,
        pad=4,
    )
    mat.tick_params(axis="y", length=0, labelsize=6, pad=3)
    for spine in mat.spines.values():
        spine.set_visible(False)
    mat.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    mat.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    mat.grid(which="minor", color="white", lw=1)
    mat.tick_params(which="minor", bottom=False, left=False, top=False)
    for row in range(8):
        for col in range(4):
            value = chrom_values[row, col]
            mat.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if value >= 80 else TEXT,
            )
    chr11 = list(t1.feature).index("chr11")
    mat.add_patch(Rectangle((-0.5, chr11 - 0.5), 4, 1, fill=False, ec=ORANGE, lw=1, zorder=5))
    for label in mat.get_yticklabels():
        if label.get_text() == "chr11":
            label.set_color(ORANGE)
            label.set_weight("bold")
    mat.set_xlabel("Chromosome length with loss (%)", fontsize=6, labelpad=4)
    text(fig, 112, 29, "Track 2: BIRC3 loss (chr11)", fontsize=6.5, weight="bold", va="center")
    gene = axis(fig, 128, 12, 50, 11, "D_BIRC3")
    gene.set_xlim(-0.5, 3.5)
    gene.set_ylim(-0.5, 0.5)
    gene.set_xticks(range(4), ["Tumour", *model_labels])
    gene.set_yticks([])
    gene.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        labelsize=6,
        pad=2,
    )
    for spine in gene.spines.values():
        spine.set_visible(False)
    for col, value in enumerate(
        ["79% of\nanalyses", "Not\nrecorded", "Not\nrecorded", "Not\nrecorded"]
    ):
        gene.add_patch(
            Rectangle(
                (col - 0.47, -0.46),
                0.94,
                0.92,
                fc="#E5EEF6" if col == 0 else "#EFF2F3",
                ec="white",
                lw=0.5,
            )
        )
        gene.text(col, 0, value, ha="center", va="center", fontsize=6.2, linespacing=1.1)
    axes.extend([mat, gene])
    columns.append([mat.get_label(), gene.get_label()])
    require_matplotlib_panel_alignment(fig, axes, rows, columns)
    fig.savefig(
        out / "Figure7.pdf", metadata={"Title": "Candidate cell-line coverage and gene-event gaps"}
    )
    fig.savefig(out / "Figure7.png", dpi=400)
    plt.close(fig)
    print(f"[CNVSIM] Figure 7 complete: wrote Figure7.pdf and Figure7.png to {out}")


if __name__ == "__main__":
    main()
