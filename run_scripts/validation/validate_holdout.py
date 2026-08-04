from __future__ import annotations

"""Run leakage-controlled fixed-reference hold-out projection."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Deterministic native execution also avoids libc++ thread-pool failures on macOS.
_native_threads = os.environ.get("CNVSIM_NATIVE_THREADS", "1")
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_variable] = _native_threads
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import pandas as pd

from cnvsim.analysis.config import load_analysis_config
from cnvsim.config import load_preprocess_config
from cnvsim.validation.config import (
    HoldoutValidationParams,
    load_validation_config,
)
from cnvsim.validation.holdout import (
    HoldoutWorkspace,
    audit_holdout_leakage,
    prepare_holdout_workspace,
    project_and_evaluate_fixed_labels,
    transform_holdout_queries,
    write_fixed_label_evaluation,
)
from cnvsim.validation.release import freeze_holdout_validation


def _configure_arrow_threads() -> None:
    try:
        import pyarrow as pa

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except (ImportError, AttributeError):
        pass


def _run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("\n[CNVSIM holdout] " + " ".join(command), flush=True)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing
        else str(REPO_ROOT) + os.pathsep + existing
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def _fit_train_reference_space(
    *,
    workspace: HoldoutWorkspace,
    preprocess_config: Path,
    analysis_config: Path,
    overwrite: bool,
) -> Path:
    source_dir = workspace.atlas_dir / "source_file"
    train_metadata = workspace.train_run_dir / "meta_cleaned.tsv"
    if not train_metadata.exists():
        raise FileNotFoundError(train_metadata)

    preprocess_command = [
        str(REPO_ROOT / "run_scripts" / "preprocess.py"),
        "--config",
        str(preprocess_config),
        "--source-dir",
        str(source_dir),
        "--work-dir",
        str(workspace.train_work_base),
        "--release-dir",
        str(workspace.train_preprocess_release),
        "--run-name",
        workspace.train_run_name,
    ]
    if overwrite:
        preprocess_command.append("--overwrite")
    _run(preprocess_command)

    prepare_ir_command = [
        str(REPO_ROOT / "run_scripts" / "analysis" / "prepare_analysis_space.py"),
        "--release-dir",
        str(workspace.train_preprocess_release),
        "--out-base-dir",
        str(workspace.train_analysis_base),
        "--analysis-config",
        str(analysis_config),
    ]
    if overwrite:
        prepare_ir_command.append("--overwrite")
    _run(prepare_ir_command)

    leakage = audit_holdout_leakage(workspace=workspace)
    leakage_path = workspace.validation_dir / "post_fit_leakage_audit.tsv"
    leakage.to_csv(leakage_path, sep="\t", index=False)
    return leakage_path


def _evaluate_fixed_reference_projection(
    *,
    workspace: HoldoutWorkspace,
    preprocess_config: Path,
    analysis_config: Path,
    holdout_params: HoldoutValidationParams,
    overwrite: bool,
) -> dict[str, Path]:
    output_manifest = (
        workspace.validation_dir / "fixed_label_projection" / "manifest.json"
    )
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(
            "Fixed-reference evaluation exists; pass --overwrite: "
            f"{output_manifest}"
        )
    pcfg = load_preprocess_config(preprocess_config)
    acfg = load_analysis_config(analysis_config)
    leakage = audit_holdout_leakage(workspace=workspace)
    query_path = workspace.validation_dir / "heldout_query_coords.parquet"
    if query_path.exists() and not overwrite:
        query_coords = pd.read_parquet(query_path)
    else:
        print("[CNVSIM holdout] transforming held-out profiles", flush=True)
        query_coords = transform_holdout_queries(
            workspace=workspace,
            gene_panel_cache=(
                workspace.atlas_dir / "source_file" / "gene_panel" / "cache"
            ),
            min_frac_low=float(pcfg.encoding.low_frac_threshold),
            min_frac_high=float(pcfg.encoding.high_frac_threshold),
            n_dims=int(acfg.ir_build.n_dims),
        )
        query_coords.to_parquet(query_path, index=False)
    full_parameters_path = (
        workspace.atlas_dir
        / "work"
        / str(acfg.run_name)
        / "analysis"
        / "parameters"
        / "selected_parameters.tsv"
    )
    print("[CNVSIM holdout] evaluating fixed-reference placement", flush=True)
    tables = project_and_evaluate_fixed_labels(
        workspace=workspace,
        query_coords=query_coords,
        full_parameters_path=full_parameters_path,
        projection_params=acfg.projection,
    )
    outputs = write_fixed_label_evaluation(
        workspace=workspace,
        tables=tables,
        leakage_audit=leakage,
        params=holdout_params,
    )
    outputs["query_coords"] = query_path
    print("\n[CNVSIM holdout] fixed-reference projection metrics")
    print(tables["metrics_global"].to_string(index=False))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and evaluate independent fixed-reference hold-out "
            "projection with train-only representation and calibration."
        )
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument(
        "--preprocess-config", default="configs/preprocess_baseline.yaml"
    )
    parser.add_argument("--analysis-config", default="configs/analysis_v1.yaml")
    parser.add_argument(
        "--validation-config",
        default="configs/validation_v1.yaml",
    )
    parser.add_argument("--validation-dir")
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "fit-space",
            "fixed-reference",
            "freeze",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--freeze-dir")
    args = parser.parse_args()

    _configure_arrow_threads()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    workspace = HoldoutWorkspace.create(
        atlas_dir=atlas_dir,
        validation_dir=(
            Path(args.validation_dir) if args.validation_dir else None
        ),
    )
    preprocess_config = Path(args.preprocess_config).expanduser().resolve()
    analysis_config = Path(args.analysis_config).expanduser().resolve()
    validation_config = Path(args.validation_config).expanduser().resolve()
    pcfg = load_preprocess_config(preprocess_config)
    load_analysis_config(analysis_config)
    vcfg = load_validation_config(validation_config)

    if args.stage in {"prepare", "all"}:
        outputs = prepare_holdout_workspace(
            workspace=workspace,
            params=vcfg.holdout,
            baseline_run_name=pcfg.run_name,
            overwrite=bool(args.overwrite),
        )
        print("[CNVSIM holdout] prepared")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    if args.stage in {"fit-space", "all"}:
        leakage_path = _fit_train_reference_space(
            workspace=workspace,
            preprocess_config=preprocess_config,
            analysis_config=analysis_config,
            overwrite=bool(args.overwrite),
        )
        print(f"[CNVSIM holdout] train-only feature space: {leakage_path}")
    if args.stage in {"fixed-reference", "all"}:
        outputs = _evaluate_fixed_reference_projection(
            workspace=workspace,
            preprocess_config=preprocess_config,
            analysis_config=analysis_config,
            holdout_params=vcfg.holdout,
            overwrite=bool(args.overwrite),
        )
        print("[CNVSIM holdout] fixed-reference projection complete")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    if args.stage in {"freeze", "all"}:
        outputs = freeze_holdout_validation(
            workspace=workspace,
            out_dir=(Path(args.freeze_dir) if args.freeze_dir else None),
            overwrite=bool(args.overwrite),
        )
        print("[CNVSIM holdout] frozen fixed-reference validation")
        for name, path in outputs.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
