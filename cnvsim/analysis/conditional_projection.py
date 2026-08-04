from __future__ import annotations

"""Materialize a calibrated second-stage conditional projection reference."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cnvsim.analysis.config import ReferenceProjectionParams
from cnvsim.analysis.ir import dimension_columns, read_analysis_coordinates
from cnvsim.analysis.reference_projection import calibrate_anchor_projection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame, columns: set[str], *, label: str
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def build_conditional_projection_reference(
    *,
    analysis_work_dir: Path,
    validation_dir: Path,
    out_dir: Path,
    params: ReferenceProjectionParams,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Fit the final conditional reference after repeated hold-out validation."""

    analysis_dir = Path(analysis_work_dir).expanduser().resolve()
    validation_dir = Path(validation_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Conditional projection reference exists: {out_dir}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    input_paths = {
        "validated_contexts": validation_dir / "conditional_contexts.tsv",
        "validation_manifest": validation_dir / "manifest.json",
        "sample_status": analysis_dir
        / "reference_contexts"
        / "sample_reference_status.parquet",
        "selected_parameters": analysis_dir
        / "parameters"
        / "selected_parameters.tsv",
        "ir_dir": analysis_dir / "ir" / "perroot",
    }
    missing = [
        str(path) for path in input_paths.values() if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing conditional reference inputs: " + ", ".join(missing)
        )

    validation_manifest = json.loads(
        input_paths["validation_manifest"].read_text()
    )
    contexts = pd.read_csv(input_paths["validated_contexts"], sep="\t")
    status = pd.read_parquet(input_paths["sample_status"])
    parameters = pd.read_csv(input_paths["selected_parameters"], sep="\t")
    _require_columns(
        contexts,
        {
            "root",
            "final_context",
            "conditional_reference_key",
            "n_full_members",
            "conditional_layer_eligible",
        },
        label="validated conditional contexts",
    )
    _require_columns(
        status,
        {"analysis_id", "root", "source_final_context", "assignment_status"},
        label="sample reference status",
    )
    _require_columns(
        parameters,
        {"root", "metric", "n_pcs", "K"},
        label="selected parameters",
    )
    if contexts.empty:
        raise ValueError("No validated conditional contexts were supplied")
    if not contexts["conditional_layer_eligible"].fillna(False).astype(bool).all():
        raise ValueError(
            "conditional_contexts.tsv contains contexts that did not pass validation"
        )
    if contexts.duplicated(["root", "final_context"]).any():
        raise ValueError("Validated conditional contexts are not unique")

    contexts["root"] = contexts["root"].astype(str)
    contexts["final_context"] = pd.to_numeric(
        contexts["final_context"], errors="raise"
    ).astype(int)
    contexts["reference_key"] = contexts[
        "conditional_reference_key"
    ].astype(str)
    contexts["reference_level"] = "conditional"
    contexts["discovery_stability"] = "intermittent_coherent"
    contexts["projection_validation"] = "passed_repeated_fixed_label_holdout"
    contexts["assignment_method"] = "primary_first_conditional_knn"

    status["analysis_id"] = status["analysis_id"].astype(str)
    status["root"] = status["root"].astype(str)
    status["source_final_context"] = pd.to_numeric(
        status["source_final_context"], errors="coerce"
    ).astype("Int64")
    nonanchors = status[status["assignment_status"].eq("non_anchor_query")].copy()
    members = nonanchors.merge(
        contexts[
            ["root", "final_context", "conditional_reference_key"]
        ],
        left_on=["root", "source_final_context"],
        right_on=["root", "final_context"],
        how="inner",
        validate="many_to_one",
    )
    if members["analysis_id"].duplicated().any():
        raise ValueError("A sample belongs to multiple conditional contexts")
    observed = members.groupby("conditional_reference_key").size()
    expected = contexts.set_index("conditional_reference_key")[
        "n_full_members"
    ].astype(int)
    if not observed.reindex(expected.index).fillna(0).astype(int).equals(expected):
        raise ValueError(
            "Conditional definition membership disagrees with validation counts"
        )
    members = members[
        [
            "analysis_id",
            "root",
            "source_final_context",
            "conditional_reference_key",
        ]
    ].rename(columns={"source_final_context": "context"})
    members["context"] = members["context"].astype(int)
    members["definition_role"] = "conditional_reference_member"
    members["definition_source"] = "full_atlas_non_reference_community"

    parameters["root"] = parameters["root"].astype(str)
    if parameters.duplicated("root").any():
        raise ValueError("Selected parameters contain duplicate roots")
    parameter_index = parameters.set_index("root")
    coordinate_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    calibration_sample_parts: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    key_lookup = contexts.set_index(["root", "final_context"])[
        "conditional_reference_key"
    ].to_dict()

    for root in sorted(contexts["root"].unique()):
        if root not in parameter_index.index:
            raise KeyError(f"[{root}] selected parameters are missing")
        parameter = parameter_index.loc[root]
        metric = str(parameter["metric"]).lower()
        n_pcs = int(parameter["n_pcs"])
        k = int(parameter["K"])
        coords_path = input_paths["ir_dir"] / root / "coords.parquet"
        if not coords_path.exists():
            raise FileNotFoundError(coords_path)
        coords = read_analysis_coordinates(coords_path)
        dimensions = dimension_columns(coords)[:n_pcs]
        if len(dimensions) != n_pcs:
            raise ValueError(
                f"[{root}] requested {n_pcs} dimensions, found {len(dimensions)}"
            )
        root_members = members[members["root"].eq(root)].copy()
        member_ids = root_members["analysis_id"].astype(str).tolist()
        missing_coords = set(member_ids) - set(coords.index)
        if missing_coords:
            raise ValueError(
                f"[{root}] {len(missing_coords)} definition samples lack IR coordinates"
            )
        labels = root_members["context"].astype(int).to_numpy()
        values = coords.loc[member_ids, dimensions].to_numpy(np.float32)
        thresholds, calibration_samples = calibrate_anchor_projection(
            analysis_ids=member_ids,
            values=values,
            contexts=labels,
            metric=metric,
            n_neighbors=k,
            params=params,
        )
        thresholds.insert(0, "root", root)
        thresholds["conditional_reference_key"] = [
            key_lookup[(root, int(context))]
            for context in thresholds["context"]
        ]
        thresholds["metric"] = metric
        thresholds["n_pcs"] = n_pcs
        thresholds["K"] = k
        calibration_parts.append(thresholds)

        calibration_samples.insert(1, "root", root)
        calibration_samples["true_conditional_reference_key"] = [
            key_lookup[(root, int(context))]
            for context in calibration_samples["true_context"]
        ]
        calibration_samples["predicted_conditional_reference_key"] = [
            key_lookup[(root, int(context))]
            for context in calibration_samples["predicted_context"]
        ]
        calibration_sample_parts.append(calibration_samples)

        coordinate_output = coords.loc[member_ids, dimensions].reset_index()
        coordinate_output.insert(1, "root", root)
        coordinate_output.insert(
            2,
            "context",
            root_members.set_index("analysis_id")
            .loc[member_ids, "context"]
            .to_numpy(),
        )
        coordinate_output.insert(
            3,
            "conditional_reference_key",
            root_members.set_index("analysis_id")
            .loc[member_ids, "conditional_reference_key"]
            .to_numpy(),
        )
        coordinate_parts.append(coordinate_output)
        model_rows.append(
            {
                "root": root,
                "projection_mode": "conditional_knn",
                "routing": "primary_ambiguous_or_unresolved_only",
                "metric": metric,
                "n_pcs": n_pcs,
                "K": k,
                "n_definition_samples": int(len(root_members)),
                "n_contexts": int(root_members["context"].nunique()),
            }
        )

    calibration = pd.concat(calibration_parts, ignore_index=True, sort=False)
    unavailable = calibration[~calibration["calibration_available"].astype(bool)]
    if not unavailable.empty:
        failed = unavailable["conditional_reference_key"].astype(str).tolist()
        raise ValueError(
            "Final conditional calibration is unavailable for contexts: "
            f"{failed}"
        )
    calibration_samples = pd.concat(
        calibration_sample_parts, ignore_index=True, sort=False
    )
    coordinates = pd.concat(coordinate_parts, ignore_index=True, sort=False)
    models = pd.DataFrame(model_rows).sort_values("root").reset_index(drop=True)
    contexts = contexts.sort_values(["root", "final_context"]).reset_index(
        drop=True
    )
    members = members.sort_values(
        ["root", "context", "analysis_id"]
    ).reset_index(drop=True)

    outputs = {
        "contexts": out_dir / "conditional_contexts.tsv",
        "members": out_dir / "conditional_definition_members.parquet",
        "coordinates": out_dir / "conditional_reference_coordinates.parquet",
        "calibration": out_dir / "projection_calibration.tsv",
        "calibration_samples": out_dir
        / "projection_calibration_samples.parquet",
        "models": out_dir / "model_index.tsv",
    }
    contexts.to_csv(outputs["contexts"], sep="\t", index=False)
    members.to_parquet(outputs["members"], index=False)
    coordinates.to_parquet(outputs["coordinates"], index=False)
    calibration.to_csv(outputs["calibration"], sep="\t", index=False)
    calibration_samples.to_parquet(outputs["calibration_samples"], index=False)
    models.to_csv(outputs["models"], sep="\t", index=False)

    checksums = {
        path.name: _sha256(path) for path in outputs.values()
    }
    manifest = {
        "schema_version": "cnvsim_conditional_projection_reference_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_level": "conditional",
        "discovery_stability": "intermittent_coherent",
        "projection_validation": "passed_repeated_fixed_label_holdout",
        "routing": "primary_ambiguous_or_unresolved_only",
        "final_fit": "all_validated_definition_members",
        "performance_source": "repeated_holdout_validation_not_final_refit",
        "projection": asdict(params),
        "counts": {
            "conditional_contexts": int(len(contexts)),
            "definition_samples": int(len(members)),
            "roots": int(contexts["root"].nunique()),
            "calibrated_contexts": int(
                calibration["calibration_available"].sum()
            ),
        },
        "inputs": {
            name: str(path) for name, path in input_paths.items()
        },
        "validation_manifest_schema": validation_manifest.get("schema_version"),
        "files": {name: path.name for name, path in outputs.items()},
        "sha256": checksums,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksums_path = out_dir / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{checksum}  {name}\n"
            for name, checksum in sorted(checksums.items())
        )
    )
    return {
        **outputs,
        "manifest": manifest_path,
        "checksums": checksums_path,
    }
