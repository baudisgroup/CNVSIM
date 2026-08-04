from __future__ import annotations

"""Freeze the publication-facing signature-recovery validation bundle."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


SCHEMA_VERSION = "cnvsim_signature_recovery_v2"
FROZEN_FILES = {
    "fold_context_signature_metrics": "fold_context_signature_metrics.parquet",
    "fold_root_signature_metrics": "fold_root_signature_metrics.tsv",
    "fold_global_signature_metrics": "fold_global_signature_metrics.tsv",
    "heldout_track1_target_effects": "heldout_track1_target_effects.parquet",
    "heldout_track2_target_effects": "heldout_track2_target_effects.parquet",
    "context_signature_recovery": "context_signature_recovery.tsv",
    "root_signature_recovery": "root_signature_recovery.tsv",
    "signature_recovery_summary": "signature_recovery_summary.tsv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing signature-recovery artifacts: " + ", ".join(missing)
        )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def _validate_source_bundle(
    source_dir: Path,
    manifest: dict[str, object],
) -> None:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise KeyError("Signature-recovery manifest lacks counts")
    expected_counts = {
        "roots": int(counts["roots"]),
        "core_contexts": int(counts["core_contexts"]),
        "folds": int(counts["folds"]),
        "track1_target_events": int(counts["track1_target_events"]),
        "track2_target_events": int(counts["track2_target_events"]),
    }
    declared_outputs = manifest.get("outputs")
    if not isinstance(declared_outputs, dict):
        raise KeyError("Signature-recovery manifest lacks outputs")

    paths = {
        key: source_dir / filename
        for key, filename in FROZEN_FILES.items()
    }
    _require_paths(list(paths.values()))
    for key, path in paths.items():
        declaration = declared_outputs.get(key)
        if Path(str(declaration or "")).name != path.name:
            raise ValueError(
                f"Unexpected signature-recovery output for {key}: {declaration}"
            )

    fold_context = _read_table(paths["fold_context_signature_metrics"])
    fold_root = _read_table(paths["fold_root_signature_metrics"])
    fold_global = _read_table(paths["fold_global_signature_metrics"])
    heldout_track1 = _read_table(paths["heldout_track1_target_effects"])
    heldout_track2 = _read_table(paths["heldout_track2_target_effects"])
    context = _read_table(paths["context_signature_recovery"])
    root = _read_table(paths["root_signature_recovery"])
    summary = _read_table(paths["signature_recovery_summary"])

    roots = expected_counts["roots"]
    contexts = expected_counts["core_contexts"]
    folds = expected_counts["folds"]
    expected_rows = {
        "fold_context_signature_metrics": contexts * folds,
        "fold_root_signature_metrics": roots * folds,
        "fold_global_signature_metrics": folds,
        "heldout_track1_target_effects": (
            expected_counts["track1_target_events"] * folds
        ),
        "heldout_track2_target_effects": (
            expected_counts["track2_target_events"] * folds
        ),
        "context_signature_recovery": contexts,
        "root_signature_recovery": roots,
        "signature_recovery_summary": 1,
    }
    observed_rows = {
        "fold_context_signature_metrics": len(fold_context),
        "fold_root_signature_metrics": len(fold_root),
        "fold_global_signature_metrics": len(fold_global),
        "heldout_track1_target_effects": len(heldout_track1),
        "heldout_track2_target_effects": len(heldout_track2),
        "context_signature_recovery": len(context),
        "root_signature_recovery": len(root),
        "signature_recovery_summary": len(summary),
    }
    mismatches = {
        key: (observed_rows[key], expected)
        for key, expected in expected_rows.items()
        if observed_rows[key] != expected
    }
    if mismatches:
        raise ValueError(
            "Signature-recovery row counts disagree with the manifest: "
            f"{mismatches}"
        )

    uniqueness_checks = [
        (fold_context, ["root", "fold", "reference_key"]),
        (fold_root, ["root", "fold"]),
        (fold_global, ["fold"]),
        (context, ["root", "reference_key"]),
        (root, ["root"]),
    ]
    for frame, columns in uniqueness_checks:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise KeyError(
                "Signature-recovery table is missing key columns: "
                f"{missing}"
            )
        if frame.duplicated(columns).any():
            raise ValueError(
                "Signature-recovery table contains duplicate keys: "
                f"{columns}"
            )


def freeze_signature_recovery(
    *,
    atlas_dir: Path,
    source_dir: Path,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Validate and freeze the completed five-fold recovery results."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else atlas_dir / "validation" / "signature_recovery_v1"
    )
    source_manifest_path = source_dir / "manifest.json"
    _require_paths([source_manifest_path])
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected signature-recovery schema")
    if source_manifest.get("validation_role") != (
        "post_selection_signature_robustness"
    ):
        raise ValueError("Unexpected signature-recovery validation role")
    if bool(source_manifest.get("mapping_uses_signature_evidence", True)):
        raise ValueError("Context mapping must not use signature evidence")
    _validate_source_bundle(source_dir, source_manifest)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "Frozen signature recovery exists; pass --overwrite: "
                f"{output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    outputs = {
        key: output_dir / filename
        for key, filename in FROZEN_FILES.items()
    }
    for key, destination in outputs.items():
        shutil.copy2(source_dir / FROZEN_FILES[key], destination)

    manifest = {
        **source_manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "frozen",
        "source_work_manifest_sha256": _sha256(source_manifest_path),
        "outputs": {
            key: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for key, path in outputs.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksums_path = output_dir / "checksums.sha256"
    checksum_files = [*outputs.values(), manifest_path]
    checksums_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_files)
    )
    return {
        **outputs,
        "manifest": manifest_path,
        "checksums": checksums_path,
    }
