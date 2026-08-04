#!/usr/bin/env python3
from __future__ import annotations

"""Plot root-level platform signal across Stage 2A principal components."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _read_eta2(stage2a_dir: Path, root: str) -> pd.DataFrame:
    path = stage2a_dir / f"{root}_platform_eta2.tsv"
    if path.exists():
        frame = pd.read_csv(path, sep="\t")
    else:
        pcs_path = stage2a_dir / f"{root}_PCs.parquet"
        meta_path = stage2a_dir.parent / "stage1" / f"{root}_meta_stage1.tsv"
        if not pcs_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Cannot load or reconstruct platform eta2 for {root}")
        from cnvsim.preprocess.stage2a import compute_per_pc_platform_eta2

        pcs = pd.read_parquet(pcs_path)
        if "analysis_id" in pcs.columns:
            pcs = pcs.set_index("analysis_id")
        pcs.index = pcs.index.astype(str)
        meta = pd.read_csv(meta_path, sep="\t", dtype=str, low_memory=False)
        frame = compute_per_pc_platform_eta2(pcs, meta)
    required = {"pc_index", "platform_eta2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    return frame


def _load_table(stage2a_dir: Path, pc_max: int) -> tuple[pd.DataFrame, list[str]]:
    qc_path = stage2a_dir / "qc_stage2A.tsv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path, sep="\t")
        if "icdot_root" in qc.columns and "root" not in qc.columns:
            qc = qc.rename(columns={"icdot_root": "root"})
        roots = qc["root"].astype(str).tolist()
    else:
        qc = pd.DataFrame()
        roots = sorted(path.name.replace("_PCs.parquet", "") for path in stage2a_dir.glob("*_PCs.parquet"))
    if not roots:
        raise ValueError(f"No Stage 2A roots found under {stage2a_dir}")

    rows: list[dict[str, object]] = []
    for root in roots:
        eta2 = _read_eta2(stage2a_dir, root)
        row: dict[str, object] = {"root": root}
        for _, record in eta2.iterrows():
            row[f"PC{int(record['pc_index'])}"] = float(record["platform_eta2"])
        values = pd.to_numeric(eta2["platform_eta2"], errors="coerce")
        row["platform_eta2_max"] = float(values.max())
        row["platform_eta2_median"] = float(values.median())
        rows.append(row)
    table = pd.DataFrame(rows)

    if not qc.empty:
        keep = [col for col in ["root", "n_samples", "platform_balacc_after_stage2A_L2"] if col in qc.columns]
        table = table.merge(qc[keep], on="root", how="left")
    pc_columns = sorted(
        [col for col in table.columns if col.startswith("PC")],
        key=lambda value: int(value[2:]),
    )
    if int(pc_max) > 0:
        pc_columns = pc_columns[: int(pc_max)]
    return table, pc_columns


def _plot(table: pd.DataFrame, pc_columns: list[str], out_path: Path, sort_by: str) -> None:
    if sort_by == "sample" and "n_samples" in table.columns:
        ordered = table.sort_values(["n_samples", "platform_eta2_max"], ascending=[False, False]).reset_index(drop=True)
    else:
        ordered = table.sort_values(["platform_eta2_max", "platform_eta2_median"], ascending=False).reset_index(drop=True)

    heat = ordered[pc_columns].to_numpy(dtype=float)
    summary = ordered[["platform_eta2_max", "platform_eta2_median"]].to_numpy(dtype=float)
    values = np.concatenate([heat[np.isfinite(heat)], summary[np.isfinite(summary)]])
    vmax = max(0.10, float(np.nanpercentile(values, 99)) if len(values) else 0.10)
    cmap = LinearSegmentedColormap.from_list(
        "platform_eta2",
        ["#f7fbff", "#d6e6f4", "#91bad6", "#3f7fb5", "#084081"],
    )

    n_roots = len(ordered)
    fig_width = max(7.5, 5.1 + 0.16 * len(pc_columns))
    fig_height = max(4.8, 0.30 * n_roots + 1.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.18, 0.05], wspace=0.18)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1], sharey=ax_heat)
    color_axis = fig.add_subplot(grid[0, 2])

    image = ax_heat.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    labels = []
    for _, row in ordered.iterrows():
        if "n_samples" in ordered.columns and pd.notna(row.get("n_samples")):
            labels.append(f"{row['root']}  (n={int(row['n_samples']):,})")
        else:
            labels.append(str(row["root"]))
    ax_heat.set_yticks(np.arange(n_roots))
    ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_xticks(np.arange(len(pc_columns)))
    ax_heat.set_xticklabels(pc_columns, rotation=90, fontsize=6)
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel("Stage 2A principal components")

    ax_summary.imshow(summary, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    ax_summary.set_xticks([0, 1])
    ax_summary.set_xticklabels(["max", "median"], rotation=90, fontsize=7)
    ax_summary.tick_params(axis="y", left=False, labelleft=False)
    ax_summary.tick_params(axis="x", length=0)
    ax_summary.set_xlabel("Summary")

    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label(r"Platform $\eta^2$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    for axis in [ax_heat, ax_summary]:
        for spine in axis.spines.values():
            spine.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ordered.to_csv(out_path.with_suffix(".tsv"), sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Preprocessing work directory for one run")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pc-max", type=int, default=20, help="Leading PCs shown; use 0 for all PCs")
    parser.add_argument("--sort-by", choices=["eta2", "sample"], default="eta2")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    stage2a_dir = run_dir / "final_space" / "stage2A"
    if not stage2a_dir.exists():
        raise FileNotFoundError(f"Missing Stage 2A directory: {stage2a_dir}")
    table, pc_columns = _load_table(stage2a_dir, int(args.pc_max))
    _plot(table, pc_columns, args.out.expanduser().resolve(), str(args.sort_by))
    print(f"[OK] roots: {len(table)}")
    print(f"[OK] wrote {args.out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
