from __future__ import annotations

"""Project non-anchor atlas samples into calibrated stable contexts."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.analysis.config import load_analysis_config
from cnvsim.analysis.reference_projection import project_reference_contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate anchor projection by leave-one-out recovery, project "
            "non-anchor samples, and route configured root-only references."
        )
    )
    parser.add_argument("--reference-contexts-dir", required=True)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--selected-parameters", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing projection result in --out-dir.",
    )
    args = parser.parse_args()

    config = load_analysis_config(args.analysis_config)
    outputs = project_reference_contexts(
        reference_contexts_dir=Path(args.reference_contexts_dir).expanduser().resolve(),
        ir_dir=Path(args.ir_dir).expanduser().resolve(),
        selected_parameters_path=(
            Path(args.selected_parameters).expanduser().resolve()
        ),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        params=config.projection,
        overwrite=bool(args.overwrite),
    )
    print("[CNVSIM] calibrated reference projection complete")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
