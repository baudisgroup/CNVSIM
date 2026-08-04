from __future__ import annotations

"""Freeze publication-facing fixed-reference hold-out artifacts."""

from datetime import datetime, timezone
import hashlib
import json
from numbers import Number
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cnvsim.validation.holdout import HoldoutWorkspace


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, *, atlas_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(atlas_dir.resolve()))
    except ValueError:
        return path.name


def _fixed_reference_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if len(metrics) != 1:
        raise ValueError("Fixed-reference global metrics must contain one row")
    row = metrics.iloc[0]
    n_test = int(row["n_test"])
    records: list[dict[str, object]] = []
    for metric, value in row.items():
        if (
            metric in {"scope", "n_test"}
            or isinstance(value, (bool, np.bool_))
            or not isinstance(value, Number)
            or pd.isna(value)
        ):
            continue
        records.append(
            {
                "endpoint": "fixed_reference_projection",
                "metric": str(metric),
                "value": float(value),
                "n_test": n_test,
                "primary_endpoint": True,
            }
        )
    return pd.DataFrame(records)


def _combine_projection_metrics(fixed_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for scope in ("global", "root", "context", "platform", "source"):
        frame = _read_tsv(fixed_dir / f"metrics_{scope}.tsv")
        frame.insert(0, "metric_scope", scope)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False)


def _assert_no_leakage(path: Path) -> None:
    frame = _read_tsv(path)
    leak_columns = [column for column in frame if "leak" in column.lower()]
    for column in leak_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.gt(0).any():
            raise ValueError(f"Held-out leakage detected in {path}: {column}")


def freeze_holdout_validation(
    *,
    workspace: HoldoutWorkspace,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Freeze only the independent fixed-reference projection endpoint."""

    atlas_dir = workspace.atlas_dir
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else atlas_dir / "validation" / "holdout_v1"
    )
    manifest_path = output_dir / "validation_manifest.json"
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Frozen hold-out validation exists; pass --overwrite: {output_dir}"
            )
        shutil.rmtree(output_dir)

    fixed_dir = workspace.validation_dir / "fixed_label_projection"
    fixed_metrics = _read_tsv(fixed_dir / "metrics_global.tsv")
    predictions_path = fixed_dir / "predictions.parquet"
    leakage_path = fixed_dir / "leakage_audit.tsv"
    fixed_manifest_path = fixed_dir / "manifest.json"
    for path in (
        predictions_path,
        leakage_path,
        fixed_manifest_path,
        workspace.split_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    _assert_no_leakage(leakage_path)

    split = pd.read_parquet(workspace.split_path)
    predictions = pd.read_parquet(predictions_path)
    n_test = int(split["holdout_role"].eq("test").sum())
    if len(predictions) != n_test:
        raise ValueError(
            "Fixed-reference predictions do not match the frozen hold-out split"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_dir / "holdout_validation_summary.tsv",
        "projection_metrics": output_dir / "projection_metrics.tsv",
        "predictions": output_dir / "holdout_predictions.parquet",
    }
    _fixed_reference_summary(fixed_metrics).to_csv(
        outputs["summary"], sep="\t", index=False
    )
    _combine_projection_metrics(fixed_dir).to_csv(
        outputs["projection_metrics"], sep="\t", index=False
    )
    predictions.to_parquet(outputs["predictions"], index=False)

    checksums = {name: _sha256(path) for name, path in outputs.items()}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "frozen",
        "validation_design": {
            "split": "deterministic five-fold context/platform/source-aware",
            "train_fraction": 0.8,
            "test_fraction": 0.2,
            "primary_endpoint": "fixed_reference_projection",
            "reference_labels": "frozen full-atlas labels on train anchors",
            "representation": "train-only preprocessing and PCA",
            "calibration": "train-only leave-one-out",
        },
        "counts": {
            "reference_contexts": int(split["full_reference_key"].nunique()),
            "train_anchor_samples": int(split["holdout_role"].eq("train").sum()),
            "test_samples": n_test,
        },
        "inputs": {
            "split": _relative(workspace.split_path, atlas_dir=atlas_dir),
            "fixed_reference_manifest": _relative(
                fixed_manifest_path, atlas_dir=atlas_dir
            ),
        },
        "outputs": {
            name: {
                "path": _relative(path, atlas_dir=atlas_dir),
                "sha256": checksums[name],
            }
            for name, path in outputs.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs["manifest"] = manifest_path
    checksums_path = output_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in outputs.values()
        )
    )
    outputs["checksums"] = checksums_path
    return outputs
