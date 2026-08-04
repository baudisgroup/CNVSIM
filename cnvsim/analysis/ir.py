from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.analysis.config import IRBuildParams

"""
Build analysis IR (coords+meta) from a preprocessing reference space.

Inputs (required):
    <preprocessing-release>/reference_space/final_space/*_PCs_final.parquet
    <preprocessing-release>/reference_space/final_space/meta_final/*_meta_final.tsv

Outputs:
    <output-base>/analysis/ir/perroot/<root>/{coords.parquet, meta.parquet, manifest.json}
    + index TSV + top-level manifest.
"""

def _pc_sort_key(col: str) -> int:
    import re

    m = re.match(r"^PC(\d+)$", str(col))
    return int(m.group(1)) if m else 10**9


def dimension_columns(frame: pd.DataFrame) -> list[str]:
    """Return analysis-coordinate columns in numeric dimension order."""

    return sorted(
        (str(column) for column in frame.columns if str(column).startswith("D")),
        key=lambda value: int(value[1:]) if value[1:].isdigit() else 10**9,
    )


def read_analysis_coordinates(path: Path) -> pd.DataFrame:
    """Read an analysis-coordinate table with a unique string index."""

    frame = pd.read_parquet(path)
    if "analysis_id" in frame.columns:
        frame["analysis_id"] = frame["analysis_id"].astype(str)
        frame = frame.set_index("analysis_id")
    frame.index = frame.index.astype(str)
    frame.index.name = "analysis_id"
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate analysis IDs in {path}")
    return frame

def build_analysis_ir_from_release(
    *,
    release_dir: Path,
    out_subdir: str = "analysis/ir",
    params: IRBuildParams = IRBuildParams(),
    roots: list[str] | None = None,
    excluded_roots: tuple[str, ...] = (),
    overwrite: bool = False,
    out_base_dir: Path | None = None,
) -> pd.DataFrame:

    release_dir = Path(release_dir)

    space_dir = release_dir / "reference_space" / "final_space"
    if not space_dir.exists():
        raise FileNotFoundError(f"Missing preprocessing comparison space: {space_dir}")

    meta_dir = Path(space_dir) / "meta_final"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing: {meta_dir}")

    base = Path(out_base_dir) if out_base_dir is not None else release_dir
    out_dir = base / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if roots is None:
        roots = sorted([p.name.replace("_PCs_final.parquet", "") for p in space_dir.glob("*_PCs_final.parquet")])
    excluded = set(str(root) for root in excluded_roots)
    roots = [root for root in roots if str(root) not in excluded]
    if not roots:
        raise ValueError("No analysis roots remain after applying excluded_roots.")

    ir_rows: list[dict] = []

    for root in roots:
        pcs_path = space_dir / f"{root}_PCs_final.parquet"
        meta_path = meta_dir / f"{root}_meta_final.tsv"
        if not pcs_path.exists():
            raise FileNotFoundError(f"[{root}] missing comparison-space coordinates: {pcs_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"[{root}] missing comparison-space metadata: {meta_path}")

        root_out = out_dir / "perroot" / root
        if root_out.exists() and (not overwrite):
            raise FileExistsError(f"[{root}] analysis IR already exists; pass --overwrite: {root_out}")
        root_out.mkdir(parents=True, exist_ok=True)

        pcs = pd.read_parquet(pcs_path).copy()

        # Ensure we have correct analysis_id index.
        if "analysis_id" in pcs.columns:
            pcs["analysis_id"] = pcs["analysis_id"].astype(str)
            pcs = pcs.set_index("analysis_id")
        else:
            # Parquet may have lost the index on write (index=False). In that case,
            # use meta_final ordering as the source of truth.
            pcs.index = pcs.index.astype(str)
            if pcs.index.name != "analysis_id":
                meta_tmp = pd.read_csv(
                    meta_path,
                    sep="\t",
                    dtype={"analysis_id": str},
                    low_memory=False,
                )
                if meta_tmp.shape[0] == pcs.shape[0] and "analysis_id" in meta_tmp.columns:
                    pcs.index = meta_tmp["analysis_id"].astype(str).to_numpy()
                    pcs.index.name = "analysis_id"

        pcs.index = pcs.index.astype(str)

        pc_cols = [c for c in pcs.columns if str(c).startswith("PC")]
        pc_cols = sorted(pc_cols, key=_pc_sort_key)
        if not pc_cols:
            raise ValueError(f"[{root}] no PC columns found in {pcs_path.name}")

        used_pc = pc_cols[: min(int(params.n_dims), len(pc_cols))]
        arr = pcs[used_pc].to_numpy(dtype=np.float32)
        coord_cols = [f"D{i+1}" for i in range(arr.shape[1])]
        coords_df = pd.DataFrame(arr, index=pcs.index, columns=coord_cols)
        coords_path = root_out / "coords.parquet"
        coords_df.to_parquet(coords_path)

        meta = pd.read_csv(
            meta_path,
            sep="\t",
            dtype={"analysis_id": str},
            low_memory=False,
        ).copy()
        meta["analysis_id"] = meta["analysis_id"].astype(str)
        meta = (
            meta.drop_duplicates("analysis_id", keep="first")
            .set_index("analysis_id")
            .reindex(coords_df.index)
            .reset_index()
        )

        if meta.isna().all(axis=1).any():
            n = int(meta.isna().all(axis=1).sum())
            bad = meta.loc[meta.isna().all(axis=1), "analysis_id"].head(10).tolist()
            raise ValueError(f"[{root}] meta missing for {n} samples. Examples: {bad}")

        meta_out_path = root_out / "meta.parquet"
        meta.to_parquet(meta_out_path, index=False)

        (root_out / "manifest.json").write_text(json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "icdot_root": root,
            "inputs": {"pcs_final": str(pcs_path), "meta_final": str(meta_path)},
            "n_samples": int(coords_df.shape[0]),
            "n_dims": int(coords_df.shape[1]),
            "dims": used_pc,
            "coord_columns": coord_cols,
        }, indent=2))

        ir_rows.append({
            "icdot_root": root,
            "n_samples": int(coords_df.shape[0]),
            "n_dims": int(coords_df.shape[1]),
            "coords_path": str(coords_path),
            "meta_path": str(meta_out_path),
        })

    ir_df = pd.DataFrame(ir_rows).sort_values("icdot_root").reset_index(drop=True)
    index_path = out_dir / "ir_index.tsv"
    ir_df.to_csv(index_path, sep="\t", index=False)

    (out_dir / "ir_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "release_dir": str(release_dir),
        "out_base_dir": str(base),
        "space_dir": str(space_dir),
        "out_dir": str(out_dir),
        "params": {"n_dims": int(params.n_dims)},
        "excluded_roots": sorted(excluded),
        "roots": sorted(ir_df["icdot_root"].astype(str).tolist()),
        "outputs": {"index": str(index_path)},
    }, indent=2))

    return ir_df
