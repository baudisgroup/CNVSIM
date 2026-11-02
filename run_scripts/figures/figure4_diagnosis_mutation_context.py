#!/usr/bin/env python3
"""Render Figure 4: diagnostic organization and accompanying mutation contexts.

Inputs come from the frozen CNVSIM v1.0.1 public data release. By default,
outputs are ATLAS_ROOT/figures/figure4/Figure4.pdf and Figure4.png. Pass
--out-dir to choose another destination and --overwrite to replace these files.
No intermediate tables are written and no association models are refitted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd


WIDTH_MM = 183.0
HEIGHT_MM = 230.0
TEXT = "#111111"
MUTED = "#68747A"
GRID = "#DFE4E8"
UNAVAILABLE = "#EEF1F3"
LOWER = "#2E5EAA"
HIGHER = "#C57343"
COMPOSITION_DARK = "#357596"
COMPOSITION_LIGHT = "#D7E7EF"
KEYS = ["root", "diagnosis", "target_subtype", "gene"]
MODEL_STATUS_COLUMNS = ["primary_status", "amplitude_status", "continuous_status"]
SITE_NAMES = {
    "C18_C20": "Colorectal", "C30_C39": "Lung", "C50": "Breast",
    "C60_C63": "Male genital", "C64_C68": "Urinary tract", "C71": "Brain",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def axes_mm(fig, x, y, width, height, label):
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM,
                         width / WIDTH_MM, height / HEIGHT_MM], label=label)


def text_mm(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, value, **kwargs)


def subtype_number(value):
    match = re.search(r"(?:CTX|COND)(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Unrecognized subtype identifier: {value}")
    return int(match.group(1))


def key_of(row):
    return tuple(getattr(row, name) for name in KEYS)


def load_data(release_dir):
    base = release_dir / "validation/mutation_context"
    paths = {
        "focus": base / "reader/focus_45_overview.tsv",
        "all": base / "association/conditional_associations.tsv",
        "tcga": base / "tcga/tcga_mutation_comparisons.tsv",
        "subtypes": release_dir / "atlas/subtypes.tsv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    focus = pd.read_csv(paths["focus"], sep="\t")
    all_results = pd.read_csv(paths["all"], sep="\t")
    tcga = pd.read_csv(paths["tcga"], sep="\t")
    subtypes = pd.read_csv(paths["subtypes"], sep="\t")
    if len(focus) != 45 or len(all_results) != 9582 or len(tcga) != 45:
        raise ValueError("The input tables differ from the declared comparison family")
    for table in [focus, all_results, tcga]:
        if table.duplicated(KEYS).any():
            raise ValueError("Duplicate mutation comparison keys")
    focus_keys = set(focus[KEYS].itertuples(index=False, name=None))
    selected_keys = set(all_results.loc[
        all_results.association_status.eq("robust_residual_association"), KEYS
    ].itertuples(index=False, name=None))
    if focus_keys != selected_keys:
        raise ValueError("The displayed association set does not match the declared selection")
    joined = focus.merge(all_results, on=KEYS, validate="one_to_one", suffixes=("", "_all"))
    for prefix in ["primary", "amplitude", "continuous"]:
        q_column = f"{prefix}_q_all" if prefix == "primary" else f"{prefix}_q"
        if not joined[q_column].lt(0.05).all():
            raise ValueError("A displayed association fails a local-CNV model criterion")
    if focus.target_subtype.nunique() != 17 or focus.root.nunique() != 6 or focus.gene.nunique() != 18:
        raise ValueError("Unexpected mutation association scope")
    definitions = subtypes.set_index("reference_key")
    if not definitions.loc[focus.target_subtype.unique(), "reference_layer"].eq("core").all():
        raise ValueError("Expected core subtype identifiers")
    for column in ["primary_or", "primary_ci_low", "primary_ci_high"]:
        if not np.isfinite(focus[column]).all() or not focus[column].gt(0).all():
            raise ValueError(f"Non-positive or missing odds-ratio value: {column}")
    if not (focus.primary_ci_low.le(focus.primary_or) & focus.primary_or.le(focus.primary_ci_high)).all():
        raise ValueError("GENIE confidence interval does not contain its estimate")
    focus["context_supported"] = focus.combined_q.lt(0.05)
    if int(focus.context_supported.sum()) != 34:
        raise ValueError("Unexpected combined-context support count")
    retained = focus.loc[focus.context_supported]
    if not ((retained.primary_or - 1) * (retained.combined_or - 1)).gt(0).all():
        raise ValueError("A context-supported association reverses direction")
    # Draw only estimates supplied by the corresponding TCGA table.
    tcga = tcga.loc[tcga.tcga_status.eq("estimable")].copy()
    tcga["three_model_support"] = tcga.significant_in_all_three_models.astype(str).str.lower().map(
        {"true": True, "false": False})
    if len(tcga) != 14 or tcga.three_model_support.isna().any() or tcga.three_model_support.sum() != 6:
        raise ValueError("Unexpected TCGA estimate/support counts")
    if not tcga.loc[tcga.three_model_support, "diagnosis"].eq("IDC").all():
        raise ValueError("Unexpected TCGA three-model support distribution")
    if not tcga.loc[~tcga.three_model_support, "tcga_fdr_q"].ge(0.05).all():
        raise ValueError("An open TCGA point is significant in the primary model; revise its legend")
    for column in ["matched_genie_odds_ratio", "tcga_odds_ratio", "tcga_ci_lower", "tcga_ci_upper"]:
        if not np.isfinite(tcga[column]).all() or not tcga[column].gt(0).all():
            raise ValueError(f"Missing TCGA comparison value: {column}")
    if not (tcga.tcga_ci_lower.le(tcga.tcga_odds_ratio) & tcga.tcga_odds_ratio.le(tcga.tcga_ci_upper)).all():
        raise ValueError("TCGA confidence interval does not contain its estimate")
    if not ((tcga.matched_genie_odds_ratio - 1) * (tcga.tcga_odds_ratio - 1)).gt(0).all():
        raise ValueError("Unexpected cross-cohort directional agreement")
    if not tcga.tcga_ci_lower.ge(0.01).all() or not tcga.tcga_ci_upper.le(20).all():
        raise ValueError("TCGA confidence intervals exceed the displayed axis limits")
    return focus, all_results, tcga



DIAGNOSIS_LABELS = {
    "NCIT:C3512": "Lung adenocarcinoma",
    "NCIT:C2919": "Prostate adenocarcinoma",
    "NCIT:C4349": "Colon adenocarcinoma",
    "NCIT:C4017": "Breast ductal carcinoma",
    "NCIT:C3058": "Glioblastoma",
    "NCIT:C9245": "Invasive breast carcinoma",
    "NCIT:C39851": "Bladder urothelial carcinoma",
    "NCIT:C4194": "Invasive breast carcinoma, NST",
    "NCIT:C8294": "Pancreatic adenocarcinoma",
    "NCIT:C2926": "Lung non-small cell carcinoma",
    "NCIT:C4004": "Gastric adenocarcinoma",
    "NCIT:C27753": "Acute myeloid leukemia, NOS",
    "NCIT:C6287": "Endometrioid adenocarcinoma",
    "NCIT:C3099": "Hepatocellular carcinoma",
    "NCIT:C80280": "DLBCL, NOS",
    "NCIT:C3510": "Cutaneous melanoma",
    "NCIT:C4025": "Esophageal adenoca.",
    "NCIT:C3270": "Neuroblastoma",
    "NCIT:C8181": "Oropharyngeal SCC",
    "NCIT:C4035": "Papillary thyroid carcinoma",
    "NCIT:C3868": "GIST",
    "NCIT:C9145": "Osteosarcoma",
}


def load_diagnosis_data(release_dir):
    """Aggregate complete recorded NCIt labels; never export individual IDs."""
    paths = {
        "assignments": release_dir / "atlas/analysis_assignments.parquet",
        "metadata": release_dir / "source_data/source_file/metadata/tumour_analysis_metadata.parquet",
        "subtypes": release_dir / "atlas/subtypes.tsv",
    }
    assignments = pd.read_parquet(paths["assignments"], columns=[
        "analysis_id", "root", "assigned_reference_key", "assignment_method"])
    metadata = pd.read_parquet(paths["metadata"], columns=[
        "analysis_id", "histological_diagnosis_id", "histological_diagnosis_label"])
    subtypes = pd.read_csv(paths["subtypes"], sep="\t").set_index("reference_key")
    if assignments.analysis_id.duplicated().any() or metadata.analysis_id.duplicated().any():
        raise ValueError("Non-unique analysis identifiers in diagnosis inputs")
    defining = assignments.loc[assignments.assignment_method.isin([
        "direct_anchor", "direct_conditional_definition"])].copy()
    observed = defining.groupby("assigned_reference_key").size().reindex(subtypes.index)
    if len(defining) != 155881 or len(observed) != 112 or not observed.eq(subtypes.n_definition_samples).all():
        raise ValueError("Definition-only accounting disagrees with released subtype counts")
    expected_roots = defining.assigned_reference_key.map(subtypes.root)
    if not defining.root.eq(expected_roots).all():
        raise ValueError("A diagnosis record crosses the subtype's anatomical root")
    data = defining.merge(metadata, on="analysis_id", validate="one_to_one", how="left")
    if data[["histological_diagnosis_id", "histological_diagnosis_label"]].isna().any().any():
        raise ValueError("A defining analysis lacks its recorded diagnosis")
    keys = ["root", "histological_diagnosis_id"]
    aliases = data.groupby(keys + ["histological_diagnosis_label"], as_index=False).size()
    aliases = aliases.sort_values(keys + ["size", "histological_diagnosis_label"],
                                  ascending=[True, True, False, True], kind="stable")
    canonical = aliases.drop_duplicates(keys)[keys + ["histological_diagnosis_label"]]
    grouped = data.groupby(keys + ["assigned_reference_key"],
                           as_index=False, sort=True).size().rename(columns={
                               "size": "n_analyses", "assigned_reference_key": "reference_key"})
    grouped = grouped.merge(canonical, on=keys, validate="many_to_one")
    summary = grouped.groupby(keys + ["histological_diagnosis_label"], as_index=False).agg(
        n_analyses=("n_analyses", "sum"), n_subtypes=("reference_key", "nunique"),
        largest_subtype_n=("n_analyses", "max"))
    summary = summary.sort_values(["n_analyses", "root", "histological_diagnosis_id"],
                                  ascending=[False, True, True], kind="stable")
    top = summary.drop_duplicates("root").copy().reset_index(drop=True)
    top["display_label"] = top.histological_diagnosis_id.map(DIAGNOSIS_LABELS)
    if len(top) != 19 or set(top.root) != set(subtypes.root) or top.display_label.isna().any():
        raise ValueError("Review the display labels for the objectively ranked diagnoses")
    top["display_order"] = np.arange(len(top))
    composition = grouped.merge(top[keys + ["display_order"]], on=keys, validate="many_to_one")
    composition = composition.sort_values(["display_order", "n_analyses", "reference_key"],
                                          ascending=[True, False, True], kind="stable")
    composition["subtype_rank"] = composition.groupby(keys).cumcount() + 1
    totals = composition.groupby(keys).n_analyses.transform("sum")
    composition["fraction"] = composition.n_analyses / totals
    if not np.allclose(composition.groupby(keys).fraction.sum(), 1):
        raise ValueError("A displayed diagnosis does not sum to 100 percent")
    observed_counts = composition.groupby(keys).size()
    if not observed_counts.eq(top.set_index(keys).n_subtypes.reindex(observed_counts.index)).all():
        raise ValueError("A nonzero subtype segment was lost")
    top["largest_subtype_fraction"] = top.largest_subtype_n / top.n_analyses
    return top, composition


def draw_diagnosis_composition(fig, top, composition):
    axis = axes_mm(fig, 48, 142, 44, 78, "A_diagnosis_composition")
    axis.set_xlim(0, 100)
    axis.set_ylim(len(top)-0.5, -0.5)
    axis.set_yticks([])
    axis.set_xticks([0, 50, 100])
    axis.set_xlabel("CNV analyses within diagnosis (%)", labelpad=3, fontsize=6.5)
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    dark = np.asarray(matplotlib.colors.to_rgb(COMPOSITION_DARK))
    light = np.asarray(matplotlib.colors.to_rgb(COMPOSITION_LIGHT))
    for index, row in top.iterrows():
        frame = composition.loc[composition.display_order.eq(index)]
        offset = 0.0
        for segment in frame.itertuples(index=False):
            intensity = (segment.subtype_rank-1)/max(1, len(frame)-1)
            color = tuple(dark*(1-intensity) + light*intensity)
            width = 100*segment.fraction
            axis.barh(index, width, left=offset, height=0.64,
                      color=color, edgecolor="white", linewidth=0.3)
            offset += width
        if not np.isclose(offset, 100):
            raise ValueError("A rendered diagnosis bar does not reach 100 percent")
        y = 220 - (index+0.5)*78/len(top)
        short_root = {
            "C15_C17_C21_C26": "C15/C17/C21/C26",
            "C40_C41_C48": "C40/C41/C48",
            "C69_C70_C72": "C69/C70/C72",
        }.get(row.root, row.root.replace("_", "–"))
        text_mm(fig, 3, y, f"{short_root} · {row.display_label}", fontsize=6, va="center")
        text_mm(fig, 99, y, f"{row.n_analyses:,}", fontsize=6, ha="center", va="center")
    text_mm(fig, 99, 224, "Analyses", fontsize=6, ha="center", va="center")
    return axis


def draw_cross_cohort(fig, tcga):
    axis = axes_mm(fig, 120, 150, 60, 60, "C_cross_cohort")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.01, 32)
    axis.set_ylim(0.01, 32)
    axis.add_patch(Rectangle((0.01, 0.01), 0.99, 0.99,
                            facecolor="#EEF3F9", edgecolor="none", zorder=-2))
    axis.add_patch(Rectangle((1, 1), 31, 31,
                            facecolor="#FBF1E9", edgecolor="none", zorder=-2))
    for orientation in (axis.axvline, axis.axhline):
        orientation(1, color="#A8AFB4", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ticks = [0.01, 0.1, 1, 10]
    for dimension in (axis.xaxis, axis.yaxis):
        dimension.set_major_locator(FixedLocator(ticks))
        dimension.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:g}"))
        dimension.set_minor_locator(NullLocator())
    axis.set_xlabel("GENIE matched-control OR", fontsize=6.5, labelpad=3)
    axis.set_ylabel("TCGA adjusted OR", fontsize=6.5, labelpad=3)
    axis.tick_params(labelsize=6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    # All 14 estimates remain; three direct labels identify representative points.
    label_locations = {
        ("C50-CTX0", "TP53"): (2.0, 20.0),
        ("C50-CTX4", "TP53"): (0.013, 0.018),
        ("C64_C68-CTX2", "FGFR3"): (0.015, 10.0),
    }
    for row in tcga.itertuples(index=False):
        color = HIGHER if row.tcga_odds_ratio > 1 else LOWER
        axis.errorbar(row.matched_genie_odds_ratio, row.tcga_odds_ratio,
                      yerr=[[row.tcga_odds_ratio-row.tcga_ci_lower],
                            [row.tcga_ci_upper-row.tcga_odds_ratio]],
                      fmt="none", ecolor=color, alpha=0.6, lw=0.7, capsize=1.5, zorder=1)
        axis.scatter([row.matched_genie_odds_ratio], [row.tcga_odds_ratio],
                     s=21 if row.three_model_support else 17,
                     facecolor=color if row.three_model_support else "white",
                     edgecolor=color, lw=0.9, zorder=4)
        location = label_locations.get((row.target_subtype, row.gene))
        if location:
            short = row.target_subtype.rsplit("-", 1)[1]
            label = f"{row.diagnosis} {short} {row.gene}"
            axis.annotate(label, (row.matched_genie_odds_ratio, row.tcga_odds_ratio),
                          xytext=location, textcoords="data", fontsize=6,
                          ha="left", va="center", color=TEXT,
                          arrowprops={"arrowstyle": "-", "color": "#99A3AA", "lw": 0.45,
                                      "shrinkA": 2, "shrinkB": 3}, zorder=5)
    handles = [Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=3.5,
                      label="Significant in TCGA (6)"),
               Line2D([], [], marker="o", linestyle="", color=MUTED, markerfacecolor="white",
                      markersize=3.5, label="Same direction, not significant (8)")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(147/WIDTH_MM, 220/HEIGHT_MM),
               ncol=1, fontsize=6, handlelength=0.8, handletextpad=0.4,
               labelspacing=0.6, borderaxespad=0)
    return axis


FIGURE5_SUBTYPES = {"C50-CTX0", "C50-CTX4", "C18_C20-CTX1", "C18_C20-CTX5", "C71-CTX2"}
CELL_WIDTH_MM = 12.0
CELL_HEIGHT_MM = 5.2
BLOCKS = (
    {"diagnosis": "IDC", "title": "Breast · IDC", "title_x": 3, "title_y": 118,
     "x": 20, "top": 109, "capacity_rows": 6,
     "genes": ["TP53", "GATA3", "PIK3CA", "ESR1"]},
    {"diagnosis": "BRCA", "title": "Breast · BRCA", "title_x": 92, "title_y": 118,
     "x": 109, "top": 109, "capacity_rows": 6,
     "genes": ["TP53", "GATA3", "PIK3CA"]},
    {"diagnosis": "COAD", "title": "Colorectal · COAD", "title_x": 3, "title_y": 70,
     "x": 20, "top": 61, "capacity_rows": 5,
     "genes": ["TP53", "APC", "BRAF", "RNF43", "EP300", "CREBBP", "MSH3", "KMT2D", "KMT2B", "ANKRD11"]},
    {"diagnosis": "READ", "title": "Rectal · READ", "title_x": 150, "title_y": 70,
     "x": 166, "top": 61, "capacity_rows": 5,
     "genes": ["TP53"]},
    {"diagnosis": "GBM", "title": "Brain · GBM", "title_x": 3, "title_y": 27,
     "x": 15, "top": 18, "capacity_rows": 2,
     "genes": ["TP53", "IDH1"]},
    {"diagnosis": "BLCA", "title": "Urinary · BLCA", "title_x": 44, "title_y": 27,
     "x": 56, "top": 18, "capacity_rows": 2,
     "genes": ["TP53", "FGFR3"]},
    {"diagnosis": "CCRCC", "title": "Urinary · CCRCC", "title_x": 85, "title_y": 27,
     "x": 97, "top": 18, "capacity_rows": 2,
     "genes": ["SETD2"]},
    {"diagnosis": "LUAD", "title": "Lung · LUAD", "title_x": 114, "title_y": 27,
     "x": 126, "top": 18, "capacity_rows": 2,
     "genes": ["TP53", "KRAS"]},
    {"diagnosis": "PRAD", "title": "Prostate · PRAD", "title_x": 155, "title_y": 27,
     "x": 167, "top": 18, "capacity_rows": 2,
     "genes": ["SPOP"]},
)


def ordered_subtypes(frame, all_results, genes):
    subtypes = frame.target_subtype.unique().tolist()
    if "TP53" not in genes:
        return sorted(subtypes, key=subtype_number)
    diagnosis = frame.diagnosis.iloc[0]
    root = frame.root.iloc[0]
    estimates = all_results.loc[
        all_results.root.eq(root) & all_results.diagnosis.eq(diagnosis)
        & all_results.target_subtype.isin(subtypes) & all_results.gene.eq("TP53")
    ].set_index("target_subtype").primary_or
    if len(estimates) != len(subtypes) or not np.isfinite(estimates).all():
        raise ValueError("A TP53 row-order estimate is unavailable")
    return sorted(subtypes, key=lambda subtype: (-float(estimates.loc[subtype]), subtype_number(subtype)))


def cell_colour(higher):
    base = np.asarray(matplotlib.colors.to_rgb(HIGHER if higher else LOWER))
    return tuple(base*0.38 + np.ones(3)*0.62)


def draw_diagnosis_block(fig, focus, all_results, spec):
    frame = focus.loc[focus.diagnosis.eq(spec["diagnosis"])].copy()
    if frame.empty or frame.root.nunique() != 1:
        raise ValueError("A diagnosis block is absent or crosses anatomical groups")
    genes = spec["genes"]
    if set(genes) != set(frame.gene):
        raise ValueError("A selected gene would be omitted from the block")
    subtypes = ordered_subtypes(frame, all_results, genes)
    root = frame.root.iloc[0]
    ncols = len(genes)
    nrows = spec["capacity_rows"]
    if nrows < len(subtypes):
        raise ValueError("The block cannot fit all selected subtypes")
    axis = axes_mm(fig, spec["x"], spec["top"]-nrows*CELL_HEIGHT_MM,
                   ncols*CELL_WIDTH_MM, nrows*CELL_HEIGHT_MM, "B_"+spec["diagnosis"])
    axis.set_xlim(0, ncols)
    axis.set_ylim(nrows, 0)
    axis.set_axis_off()
    text_mm(fig, spec["title_x"], spec["title_y"], spec["title"],
            fontsize=6.6 if spec["capacity_rows"] == 2 else 7.2, weight="bold", va="center")
    for column, gene in enumerate(genes):
        text_mm(fig, spec["x"]+(column+0.5)*CELL_WIDTH_MM, spec["top"]+3.4,
                gene, fontsize=6.1, fontstyle="italic", ha="center", va="center")
    for row, subtype in enumerate(subtypes):
        mark = "†" if subtype in FIGURE5_SUBTYPES else ""
        text_mm(fig, spec["x"]-2.2, spec["top"]-(row+0.5)*CELL_HEIGHT_MM,
                subtype.rsplit("-", 1)[1]+mark, fontsize=6.1, ha="right", va="center")
    lookup = {key_of(row): row for row in all_results.itertuples(index=False)}
    selected = {key_of(row): row for row in frame.itertuples(index=False)}
    records = []
    for row, subtype in enumerate(subtypes):
        for column, gene in enumerate(genes):
            key = (root, spec["diagnosis"], subtype, gene)
            original = lookup.get(key)
            if original is None or not all(getattr(original, status) == "estimable" for status in MODEL_STATUS_COLUMNS):
                raise ValueError("A displayed possible cell is not fully estimable")
            record = {"root": root, "diagnosis": spec["diagnosis"], "subtype": subtype,
                      "gene": gene, "selected": key in selected}
            if key not in selected:
                axis.add_patch(Rectangle((column+0.025, row+0.065), 0.95, 0.87,
                                         facecolor="white", edgecolor="#E5EAEC", lw=0.3, zorder=0))
                records.append(record)
                continue
            result = selected[key]
            higher = result.primary_or > 1
            border = HIGHER if higher else LOWER
            supported = bool(result.context_supported)
            if not supported:
                border = tuple(np.asarray(matplotlib.colors.to_rgb(border))*0.40+np.ones(3)*0.60)
            axis.add_patch(Rectangle((column+0.025, row+0.065), 0.95, 0.87,
                                     facecolor=cell_colour(higher) if supported else "white",
                                     edgecolor=border, lw=0.55 if supported else 0.35, zorder=1))
            target = 100*result.target_mutated/result.target_n
            control = 100*result.control_mutated/result.control_n
            rate = f"{target:.1f}/{control:.1f}"
            axis.text(column+0.5, row+0.5, rate, fontsize=6.0,
                      color=TEXT,
                      ha="center", va="center", zorder=3)
            record.update({"primary_or": float(result.primary_or),
                           "primary_ci": [float(result.primary_ci_low), float(result.primary_ci_high)],
                           "target_n": int(result.target_n), "target_mutated": int(result.target_mutated),
                           "control_n": int(result.control_n), "control_mutated": int(result.control_mutated),
                           "raw_rate_pct": [target, control], "displayed_rate": rate,
                           "context_supported": supported, "higher_adjusted_odds": higher})
            records.append(record)
    return axis, records, {"diagnosis": spec["diagnosis"], "subtypes": subtypes,
                           "genes": genes, "possible_cells": len(subtypes)*len(genes),
                           "selected_cells": len(frame),
                           "cell_size_mm": [CELL_WIDTH_MM, CELL_HEIGHT_MM]}


def draw_map_legend(fig):
    handles = [Patch(facecolor=cell_colour(True), edgecolor=HIGHER, lw=0.55, label="Higher adjusted odds"),
               Patch(facecolor=cell_colour(False), edgecolor=LOWER, lw=0.55, label="Lower adjusted odds"),
               Patch(facecolor="#CBD2D7", edgecolor=MUTED, lw=0.55,
                     label="Density/type adjusted: q < 0.05 (34)"),
               Patch(facecolor="white", edgecolor="#B7C2CA", lw=0.35,
                     label="Density/type adjusted: q ≥ 0.05 (11)")]
    fig.legend(handles=handles, ncol=4, loc="center", bbox_to_anchor=(99/WIDTH_MM, 126/HEIGHT_MM),
               fontsize=6, handlelength=1, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)


def require_matplotlib_panel_alignment(fig, axes):
    fig.canvas.draw()
    by_name = {axis.get_label(): axis for axis in axes}
    row_groups = [["B_IDC", "B_BRCA"], ["B_COAD", "B_READ"],
                  ["B_GBM", "B_BLCA", "B_CCRCC", "B_LUAD", "B_PRAD"]]
    column_groups = []
    w, h = fig.get_size_inches()*72
    for groups, coordinates in [(row_groups, [1, 3]), (column_groups, [0, 2])]:
        for group in groups:
            positions = [by_name[name].get_position() for name in group]
            boxes = [[p.x0*w, p.y0*h, p.x1*w, p.y1*h] for p in positions]
            if any(max(b[k] for b in boxes)-min(b[k] for b in boxes)>1.5 for k in coordinates):
                raise ValueError("Diagnosis-block layout is misaligned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path,
                        help="Analysis workspace; default output is ATLAS_ROOT/figures/figure4")
    parser.add_argument("--release-dir", type=Path, required=True,
                        help="Frozen CNVSIM_v1.0.1_public data release")
    parser.add_argument("--out-dir", type=Path,
                        help="Override the default Figure 4 output directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing Figure4.pdf and Figure4.png")
    args = parser.parse_args()
    if args.out_dir is None and args.atlas_root is None:
        parser.error("Pass --atlas-root for the default destination, or --out-dir")
    out = (args.out_dir.expanduser().resolve() if args.out_dir is not None
           else args.atlas_root.expanduser().resolve() / "figures/figure4")
    existing = [out / name for name in ("Figure4.pdf", "Figure4.png")
                if (out / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Figure 4 outputs already exist; pass --overwrite: "
                              + ", ".join(str(path) for path in existing))
    release_dir = args.release_dir.expanduser().resolve()
    focus, all_results, tcga = load_data(release_dir)
    top, composition = load_diagnosis_data(release_dir)
    evaluable = all_results.loc[all_results[MODEL_STATUS_COLUMNS].eq("estimable").all(axis=1)]
    if len(evaluable) != 3951 or evaluable.target_subtype.nunique() != 81:
        raise ValueError("Unexpected complete three-model evaluation scope")
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)
    text_mm(fig, 3, 227, "A", fontsize=10, weight="bold", va="center")
    text_mm(fig, 106, 227, "C", fontsize=10, weight="bold", va="center")
    text_mm(fig, 3, 126, "B", fontsize=10, weight="bold", va="center")
    axes = [draw_diagnosis_composition(fig, top, composition), draw_cross_cohort(fig, tcga)]
    records = []
    for block in BLOCKS:
        axis, rr, _ = draw_diagnosis_block(fig, focus, all_results, block)
        axes.append(axis)
        records.extend(rr)
    draw_map_legend(fig)
    require_matplotlib_panel_alignment(fig, axes)
    selected = [r for r in records if r["selected"]]
    if len(records) != 100 or len(selected) != 45:
        raise ValueError("The association map does not retain all expected cells")
    if sum(r["context_supported"] for r in selected) != 34:
        raise ValueError("The association map changes the further-adjustment support states")
    expected = set(focus[KEYS].itertuples(index=False, name=None))
    displayed = {(r["root"], r["diagnosis"], r["subtype"], r["gene"]) for r in selected}
    if displayed != expected:
        raise ValueError("The displayed association keys do not match the selected set")
    raw_sign = focus.target_mutated/focus.target_n-focus.control_mutated/focus.control_n
    discordant = focus.loc[raw_sign*(focus.primary_or-1)<0, KEYS]
    if len(discordant) != 4:
        raise ValueError("Unexpected raw/adjusted direction differences")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"Figure4.pdf", metadata={"Title": "Diagnostic organization and mutation contexts of CNV subtypes"})
    fig.savefig(out/"Figure4.png", dpi=400)
    plt.close(fig)
    print("[CNVSIM] Figure 4 complete: 19 roots; 45 associations; 14 TCGA comparisons")
    print(f"[CNVSIM] wrote Figure4.{{pdf,png}} to {out}")


if __name__ == "__main__":
    main()
