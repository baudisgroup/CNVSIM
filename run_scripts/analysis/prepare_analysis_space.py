from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.ir import build_analysis_ir_from_release
from cnvsim.analysis.config import IRBuildParams, load_analysis_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build root-specific analysis coordinates and metadata from a preprocessing reference space."
    )
    ap.add_argument(
        "--release-dir",
        required=True,
        help="Preprocessing release containing reference_space/final_space.",
    )

    # Preferred layout: write into a workdir intermediate folder, not into the release.
    ap.add_argument("--out-base-dir", default=None, help="Optional base dir to write IR under. If set, output goes to <out-base-dir>/<out-subdir>.")
    ap.add_argument("--out-subdir", default="analysis/ir", help="Subdirectory name for analysis IR output.")

    ap.add_argument("--analysis-config", required=True, help="Canonical analysis config YAML.")
    ap.add_argument("--n-dims", type=int, default=None, help="Number of leading PCs to use as analysis coordinates")
    ap.add_argument("--roots", default=None, help="Optional comma-separated roots to include")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    roots = None
    if args.roots:
        roots = [r.strip() for r in str(args.roots).split(",") if r.strip()]
    
    acfg = load_analysis_config(args.analysis_config)
    n_dims = int(args.n_dims) if args.n_dims is not None else int(acfg.ir_build.n_dims)

    out_base_dir = Path(args.out_base_dir) if args.out_base_dir else None

    df = build_analysis_ir_from_release(
        release_dir=Path(args.release_dir),
        out_subdir=str(args.out_subdir),
        params=IRBuildParams(n_dims=int(n_dims)),
        roots=roots,
        excluded_roots=tuple(acfg.excluded_roots),
        overwrite=bool(args.overwrite),
        out_base_dir=out_base_dir,
    )

    out_path = (out_base_dir / str(args.out_subdir)) if out_base_dir else (Path(args.release_dir) / str(args.out_subdir))
    print(f"[DONE] analysis IR -> {out_path}")
    print(f"[DONE] roots: {df.shape[0]}")


if __name__ == "__main__":
    main()
