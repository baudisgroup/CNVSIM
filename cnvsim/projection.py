from __future__ import annotations

"""Build and apply the frozen CNVSIM projection reference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from cnvsim.analysis.reference_projection import (
    classify_reference_queries,
    score_reference_knn_with_neighbors,
)
from cnvsim.preprocess.roots import icdot_root_grouped


AUTOSOMES = {str(value) for value in range(1, 23)}
STATE_TO_EVENT = {
    "gain": "dup_frac",
    "duplication": "dup_frac",
    "low_gain": "dup_frac",
    "loss": "del_frac",
    "deletion": "del_frac",
    "low_loss": "del_frac",
    "high_gain": "hldup_frac",
    "amplification": "hldup_frac",
    "focal_genome_amplification": "hldup_frac",
    "high_level_gain": "hldup_frac",
    "high_level_duplication": "hldup_frac",
    "high_loss": "hldel_frac",
    "deep_loss": "hldel_frac",
    "complete_genomic_deletion": "hldel_frac",
    "homozygous_deletion": "hldel_frac",
    "high_level_loss": "hldel_frac",
}
EVENT_COLUMNS = ("dup_frac", "del_frac", "hldup_frac", "hldel_frac")
NEAREST_REFERENCE_COLUMNS = (
    "sample_id",
    "root",
    "projection_layer",
    "neighbor_rank",
    "reference_analysis_id",
    "reference_key",
    "reference_context",
    "distance",
    "distance_metric",
    "layer_assignment_status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_manifest_path(bundle_dir: Path) -> Path:
    """Locate a public bundle manifest or its legacy internal equivalent."""
    for name in ("manifest.json", "projection_manifest.json"):
        path = bundle_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(bundle_dir / "manifest.json")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator, low_memory=False)


def _concat_preserving_all_na_columns(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate heterogeneous frames without pandas all-NA dtype warnings."""

    columns = list(
        dict.fromkeys(column for frame in frames for column in frame.columns)
    )
    populated = [frame.dropna(axis=1, how="all") for frame in frames]
    result = pd.concat(populated, ignore_index=True, sort=False)
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result.reindex(columns=columns)


def _normalize_chromosome(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace("chr", "", regex=False)
        .str.replace("CHR", "", regex=False)
    )


def _context_number(reference_key: pd.Series) -> pd.Series:
    extracted = reference_key.astype(str).str.extract(r"-CTX(\d+)$", expand=False)
    if extracted.isna().any():
        bad = reference_key[extracted.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Subtype reference keys lack CTX labels: {bad}")
    return extracted.astype(int)


def _portable_transform(
    *,
    scaler_path: Path,
    pca_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    if bool(getattr(pca, "whiten", False)):
        raise ValueError("Whitened PCA is not supported by the portable bundle")
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
    pca_mean = np.asarray(pca.mean_, dtype=np.float64)
    components = np.asarray(pca.components_, dtype=np.float64)
    if scaler_mean.shape != scaler_scale.shape:
        raise ValueError(f"Scaler shape mismatch: {scaler_path}")
    if components.shape[1] != scaler_mean.shape[0]:
        raise ValueError(f"PCA/scaler shape mismatch: {pca_path}")
    np.savez_compressed(
        output_path,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_mean=pca_mean,
        pca_components=components,
    )
    return int(components.shape[1]), int(components.shape[0])


def _model_source_path(value: object, model_source: Path) -> Path:
    declared = Path(str(value))
    if declared.exists():
        return declared
    fallback = model_source / declared.name
    return fallback


def _copy_conditional_reference(
    source_dir: Path,
    out_dir: Path,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Validate and copy the optional second-stage projection reference."""

    source_dir = Path(source_dir).expanduser().resolve()
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "cnvsim_conditional_projection_reference_v1":
        raise ValueError("Unsupported conditional projection reference schema")
    for relative, expected in manifest.get("sha256", {}).items():
        path = source_dir / relative
        if not path.exists() or _sha256(path) != str(expected):
            raise ValueError(f"Conditional reference checksum mismatch: {path}")

    declared = manifest.get("files", {})
    required = {"contexts", "members", "coordinates", "calibration", "models"}
    missing = sorted(required - set(declared))
    if missing:
        raise KeyError(f"Conditional reference manifest lacks files: {missing}")
    conditional_dir = out_dir / "conditional"
    conditional_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name in sorted(required):
        source = source_dir / str(declared[name])
        if not source.exists():
            raise FileNotFoundError(source)
        destination = conditional_dir / source.name
        shutil.copy2(source, destination)
        outputs[f"conditional_{name}"] = destination

    contexts = pd.read_csv(outputs["conditional_contexts"], sep="\t")
    members = pd.read_parquet(outputs["conditional_members"])
    coordinates = pd.read_parquet(outputs["conditional_coordinates"])
    calibration = pd.read_csv(outputs["conditional_calibration"], sep="\t")
    models = pd.read_csv(outputs["conditional_models"], sep="\t")
    required_context_columns = {
        "root",
        "final_context",
        "conditional_reference_key",
    }
    if missing_columns := sorted(required_context_columns - set(contexts.columns)):
        raise KeyError(f"Conditional contexts lack columns: {missing_columns}")
    if contexts.duplicated(["root", "final_context"]).any():
        raise ValueError("Conditional reference contains duplicate contexts")
    if members["analysis_id"].astype(str).duplicated().any():
        raise ValueError("Conditional reference contains duplicate definition samples")
    required_coordinate_columns = {
        "analysis_id",
        "root",
        "context",
        "conditional_reference_key",
    }
    if missing_columns := sorted(
        required_coordinate_columns - set(coordinates.columns)
    ):
        raise KeyError(f"Conditional coordinates lack columns: {missing_columns}")
    if coordinates["analysis_id"].astype(str).duplicated().any():
        raise ValueError("Conditional coordinates contain duplicate samples")
    if not calibration["calibration_available"].astype(bool).all():
        raise ValueError("Conditional projection contains unavailable calibration")
    context_keys = set(contexts["conditional_reference_key"].astype(str))
    calibrated_keys = set(calibration["conditional_reference_key"].astype(str))
    if calibrated_keys != context_keys:
        raise ValueError("Conditional calibration/context keys disagree")
    if set(models["root"].astype(str)) != set(contexts["root"].astype(str)):
        raise ValueError("Conditional model/context roots disagree")

    return outputs, {
        "conditional_contexts": int(len(contexts)),
        "conditional_definition_samples": int(len(members)),
        "conditional_roots": int(contexts["root"].nunique()),
    }


def build_projection_bundle(
    *,
    atlas_dir: Path,
    preprocess_release_dir: Path,
    analysis_run_dir: Path,
    release_candidate_dir: Path,
    out_dir: Path,
    low_fraction_threshold: float,
    high_fraction_threshold: float,
    top_candidates: int,
    conditional_reference_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Assemble a portable fixed-atlas projection reference bundle."""

    atlas_dir = Path(atlas_dir).expanduser().resolve()
    preprocess_release_dir = Path(preprocess_release_dir).expanduser().resolve()
    analysis_run_dir = Path(analysis_run_dir).expanduser().resolve()
    release_candidate_dir = Path(release_candidate_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Projection bundle exists: {out_dir}")
        shutil.rmtree(out_dir)

    model_source = preprocess_release_dir / "reference_space" / "models_stage2A"
    model_manifest_path = model_source / "models_manifest.tsv"
    reference_path = release_candidate_dir / "tables" / "reference_contexts.tsv"
    assignments_path = (
        release_candidate_dir / "tables" / "analysis_assignments.parquet"
    )
    selected_path = analysis_run_dir / "parameters" / "selected_parameters.tsv"
    calibration_path = (
        analysis_run_dir
        / "projection"
        / "non_anchor_recovery"
        / "projection_calibration.tsv"
    )
    ir_dir = analysis_run_dir / "ir" / "perroot"
    gene_intervals_path = (
        atlas_dir
        / "source_file"
        / "gene_panel"
        / "cancer_gene_list_geneid_p14.tsv"
    )
    required = (
        model_manifest_path,
        reference_path,
        assignments_path,
        selected_path,
        calibration_path,
        ir_dir,
        gene_intervals_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing projection bundle inputs: " + ", ".join(missing))

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    references = pd.read_csv(reference_path, sep="\t")
    assignments = pd.read_parquet(assignments_path)
    parameters = pd.read_csv(selected_path, sep="\t")
    calibration = pd.read_csv(calibration_path, sep="\t")
    model_manifest = pd.read_csv(model_manifest_path, sep="\t")
    for frame in (references, assignments, parameters, calibration, model_manifest):
        if "root" in frame.columns:
            frame["root"] = frame["root"].astype(str)
        elif "icdot_root" in frame.columns:
            frame.rename(columns={"icdot_root": "root"}, inplace=True)
            frame["root"] = frame["root"].astype(str)

    subtype_references = references[
        references["reference_level"].astype(str).eq("subtype")
    ].copy()
    root_references = references[
        references["reference_level"].astype(str).eq("root")
    ].copy()
    conditional_references = references[
        references["reference_level"].astype(str).eq("conditional")
    ].copy()
    primary_catalog = pd.concat(
        [subtype_references, root_references],
        ignore_index=True,
        sort=False,
    )
    if primary_catalog["reference_key"].astype(str).duplicated().any():
        raise ValueError("Primary projection reference keys must be unique")
    if (
        set(primary_catalog["reference_key"].astype(str))
        & set(conditional_references["reference_key"].astype(str))
    ):
        raise ValueError("Primary and conditional release reference keys overlap")
    if not conditional_references.empty and conditional_reference_dir is None:
        raise ValueError(
            "Release contains conditional references but no conditional "
            "projection reference was supplied"
        )
    parameter_index = parameters.set_index("root")
    model_index = model_manifest.set_index("root")
    model_rows: list[dict[str, object]] = []
    anchor_parts: list[pd.DataFrame] = []

    for root in sorted(subtype_references["root"].unique()):
        if root not in parameter_index.index or root not in model_index.index:
            raise KeyError(f"[{root}] missing selected parameters or Stage 2A model")
        parameter = parameter_index.loc[root]
        model = model_index.loc[root]
        feature_source = _model_source_path(
            model["feature_columns_path"], model_source
        )
        scaler_source = _model_source_path(model["scaler_path"], model_source)
        pca_source = _model_source_path(model["pca_path"], model_source)
        for path in (feature_source, scaler_source, pca_source):
            if not path.exists():
                raise FileNotFoundError(path)

        features = json.loads(feature_source.read_text())
        feature_output = models_dir / f"{root}_features.json"
        feature_output.write_text(json.dumps(features) + "\n")
        transform_output = models_dir / f"{root}_transform.npz"
        n_features, n_components = _portable_transform(
            scaler_path=scaler_source,
            pca_path=pca_source,
            output_path=transform_output,
        )
        if len(features) != n_features:
            raise ValueError(f"[{root}] feature list/model dimension mismatch")

        anchor_labels = assignments[
            assignments["root"].astype(str).eq(root)
            & assignments["pipeline_status"].astype(str).eq("anchor_member")
        ][["analysis_id", "assigned_reference_key"]].copy()
        if anchor_labels.empty:
            raise ValueError(f"[{root}] has no released anchor members")
        anchor_labels["analysis_id"] = anchor_labels["analysis_id"].astype(str)
        anchor_labels["context"] = _context_number(
            anchor_labels["assigned_reference_key"]
        )
        coords_path = ir_dir / root / "coords.parquet"
        coords = pd.read_parquet(coords_path)
        if "analysis_id" not in coords.columns:
            coords = coords.reset_index()
        coords["analysis_id"] = coords["analysis_id"].astype(str)
        n_pcs = int(parameter["n_pcs"])
        dimensions = [f"D{index}" for index in range(1, n_pcs + 1)]
        missing_dimensions = sorted(set(dimensions) - set(coords.columns))
        if missing_dimensions:
            raise KeyError(f"[{root}] missing IR dimensions: {missing_dimensions}")
        anchors = anchor_labels.merge(
            coords[["analysis_id", *dimensions]],
            on="analysis_id",
            how="left",
            validate="one_to_one",
        )
        if anchors[dimensions].isna().any(axis=None):
            raise ValueError(f"[{root}] released anchors lack coordinates")
        anchors.insert(1, "root", root)
        anchor_parts.append(anchors)
        model_rows.append(
            {
                "root": root,
                "projection_mode": "subtype_knn",
                "metric": str(parameter["metric"]).lower(),
                "n_pcs": n_pcs,
                "K": int(parameter["K"]),
                "n_features": n_features,
                "n_components": n_components,
                "n_anchor_samples": int(len(anchors)),
                "n_contexts": int(anchor_labels["context"].nunique()),
                "feature_file": f"models/{feature_output.name}",
                "transform_file": f"models/{transform_output.name}",
            }
        )

    for row in root_references.itertuples(index=False):
        model_rows.append(
            {
                "root": str(row.root),
                "projection_mode": "root_only",
                "reference_key": str(row.reference_key),
                "metric": "",
                "n_pcs": 0,
                "K": 0,
                "n_features": 0,
                "n_components": 0,
                "n_anchor_samples": int(row.n_definition_members),
                "n_contexts": 1,
                "feature_file": "",
                "transform_file": "",
            }
        )

    for row in model_rows:
        if row["projection_mode"] == "subtype_knn":
            row["reference_key"] = ""

    anchors = pd.concat(anchor_parts, ignore_index=True, sort=False)
    model_table = pd.DataFrame(model_rows).sort_values("root").reset_index(drop=True)
    valid_contexts = set(subtype_references["reference_key"].astype(str))
    calibration["reference_key"] = [
        f"{root}-CTX{int(context)}"
        for root, context in zip(calibration["root"], calibration["context"])
    ]
    calibration = calibration[
        calibration["reference_key"].isin(valid_contexts)
    ].copy()
    if set(calibration["reference_key"]) != valid_contexts:
        missing_contexts = sorted(valid_contexts - set(calibration["reference_key"]))
        raise ValueError(f"Projection calibration lacks contexts: {missing_contexts}")

    outputs = {
        "models": out_dir / "model_index.tsv",
        "anchors": out_dir / "anchor_coordinates.parquet",
        "calibration": out_dir / "projection_calibration.tsv",
        "references": out_dir / "reference_contexts.tsv",
        "genes": out_dir / "gene_intervals.tsv",
    }
    model_table.to_csv(outputs["models"], sep="\t", index=False)
    anchors.to_parquet(outputs["anchors"], index=False)
    calibration.to_csv(outputs["calibration"], sep="\t", index=False)
    primary_catalog.to_csv(outputs["references"], sep="\t", index=False)
    shutil.copy2(gene_intervals_path, outputs["genes"])

    conditional_counts = {
        "conditional_contexts": 0,
        "conditional_definition_samples": 0,
        "conditional_roots": 0,
    }
    if conditional_reference_dir is not None:
        conditional_outputs, conditional_counts = _copy_conditional_reference(
            conditional_reference_dir,
            out_dir,
        )
        outputs.update(conditional_outputs)
        copied_contexts = pd.read_csv(
            conditional_outputs["conditional_contexts"],
            sep="\t",
        )
        copied_key_column = (
            "reference_key"
            if "reference_key" in copied_contexts.columns
            else "conditional_reference_key"
        )
        copied_keys = set(copied_contexts[copied_key_column].astype(str))
        release_keys = set(conditional_references["reference_key"].astype(str))
        if release_keys and copied_keys != release_keys:
            raise ValueError(
                "Conditional projection references disagree with the release: "
                f"release_only={sorted(release_keys - copied_keys)}, "
                f"projection_only={sorted(copied_keys - release_keys)}"
            )

    checksum_paths = sorted(path for path in out_dir.rglob("*") if path.is_file())
    checksums = {
        path.relative_to(out_dir).as_posix(): _sha256(path) for path in checksum_paths
    }
    manifest = {
        "schema_version": "cnvsim_projection_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "genome_build": "GRCh38",
        "coordinate_system": "0-based-half-open",
        "reference_ids_frozen": True,
        "encoding": {
            "autosomes_only": True,
            "low_fraction_threshold": float(low_fraction_threshold),
            "high_fraction_threshold": float(high_fraction_threshold),
            "feature_value": "low_event_indicator + 2 * high_event_indicator",
        },
        "top_candidates": int(top_candidates),
        "conditional_reference": {
            "enabled": conditional_reference_dir is not None,
            "routing": "primary_ambiguous_or_unresolved_only",
        },
        "counts": {
            "subtype_roots": int(
                model_table["projection_mode"].eq("subtype_knn").sum()
            ),
            "root_only_roots": int(
                model_table["projection_mode"].eq("root_only").sum()
            ),
            "reference_contexts": int(len(primary_catalog)),
            "all_reference_contexts": int(
                len(primary_catalog) + conditional_counts["conditional_contexts"]
            ),
            "anchor_samples": int(len(anchors)),
            **conditional_counts,
        },
        "files": {key: path.relative_to(out_dir).as_posix() for key, path in outputs.items()},
        "sha256": checksums,
    }
    manifest_path = out_dir / "projection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksum_path = out_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{checksum}  {path}\n" for path, checksum in sorted(checksums.items()))
    )
    return {**outputs, "manifest": manifest_path, "checksums": checksum_path}


def _canonical_samples(frame: pd.DataFrame) -> pd.DataFrame:
    samples = frame.copy()
    if "sample_id" not in samples.columns:
        if "analysis_id" not in samples.columns:
            raise KeyError("samples must contain sample_id or analysis_id")
        samples = samples.rename(columns={"analysis_id": "sample_id"})
    samples["sample_id"] = samples["sample_id"].astype(str)
    if samples["sample_id"].duplicated().any():
        raise ValueError("samples contains duplicate sample_id values")
    if "genome_build" in samples.columns:
        builds = set(samples["genome_build"].dropna().astype(str))
        accepted = {"GRCh38", "hg38"}
        invalid = sorted(builds - accepted)
        if invalid:
            raise ValueError(f"Only GRCh38/hg38 inputs are supported: {invalid}")
    if "root" in samples.columns:
        samples["root"] = samples["root"].astype(str)
    elif "icdot_root_grouped" in samples.columns:
        samples["root"] = samples["icdot_root_grouped"].astype(str)
    elif "icdo_topography_id" in samples.columns:
        samples["root"] = samples["icdo_topography_id"].map(icdot_root_grouped)
    else:
        raise KeyError(
            "samples must contain root, icdot_root_grouped, or icdo_topography_id"
        )
    return samples


def _canonical_segments(frame: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    segments = frame.copy()
    if "sample_id" not in segments.columns:
        if "analysis_id" not in segments.columns:
            raise KeyError("segments must contain sample_id or analysis_id")
        segments = segments.rename(columns={"analysis_id": "sample_id"})
    required = {"sample_id", "chromosome", "start", "end", "cnv_state"}
    missing = sorted(required - set(segments.columns))
    if missing:
        raise KeyError(f"segments is missing columns: {missing}")
    segments["sample_id"] = segments["sample_id"].astype(str)
    unknown = sorted(set(segments["sample_id"]) - sample_ids)
    if unknown:
        raise ValueError(f"segments contains unknown sample IDs: {unknown[:5]}")
    segments["chromosome"] = _normalize_chromosome(segments["chromosome"])
    segments["start"] = pd.to_numeric(segments["start"], errors="raise").astype(np.int64)
    segments["end"] = pd.to_numeric(segments["end"], errors="raise").astype(np.int64)
    if (segments["start"] < 0).any() or (segments["end"] <= segments["start"]).any():
        raise ValueError("segments requires 0 <= start < end")
    state = (
        segments["cnv_state"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    invalid_states = sorted(set(state) - set(STATE_TO_EVENT))
    if invalid_states:
        raise ValueError(f"Unsupported cnv_state values: {invalid_states}")
    segments["event_column"] = state.map(STATE_TO_EVENT)
    return segments[segments["chromosome"].isin(AUTOSOMES)].copy()


def _merged_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def segment_gene_fractions(
    *,
    segments: pd.DataFrame,
    genes: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate union overlap fractions for each sample, gene and event class."""

    required_genes = {"gene_symbol", "chrom", "start", "end"}
    missing = sorted(required_genes - set(genes.columns))
    if missing:
        raise KeyError(f"gene intervals are missing columns: {missing}")
    genes = genes.copy()
    genes["chrom"] = _normalize_chromosome(genes["chrom"])
    genes["start"] = pd.to_numeric(genes["start"], errors="raise").astype(np.int64)
    genes["end"] = pd.to_numeric(genes["end"], errors="raise").astype(np.int64)
    genes = genes[genes["chrom"].isin(AUTOSOMES)].drop_duplicates(
        ["gene_symbol", "chrom", "start", "end"]
    )
    duplicated_symbols = genes["gene_symbol"].astype(str).duplicated(keep=False)
    if duplicated_symbols.any():
        examples = (
            genes.loc[duplicated_symbols, "gene_symbol"]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "gene intervals contain symbols mapped to multiple intervals: "
            f"{examples}"
        )
    overlaps: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    gene_lengths: dict[str, int] = {}
    by_chrom = {chrom: frame for chrom, frame in genes.groupby("chrom", sort=False)}
    for row in segments.itertuples(index=False):
        chromosome = str(row.chromosome)
        chromosome_genes = by_chrom.get(chromosome)
        if chromosome_genes is None:
            continue
        starts = chromosome_genes["start"].to_numpy()
        ends = chromosome_genes["end"].to_numpy()
        mask = (starts < int(row.end)) & (ends > int(row.start))
        for gene in chromosome_genes.loc[mask].itertuples(index=False):
            symbol = str(gene.gene_symbol)
            start = max(int(row.start), int(gene.start))
            end = min(int(row.end), int(gene.end))
            overlaps.setdefault(
                (str(row.sample_id), symbol, str(row.event_column)), []
            ).append((start, end))
            gene_lengths[symbol] = int(gene.end) - int(gene.start)

    rows: dict[tuple[str, str], dict[str, object]] = {}
    for (sample_id, symbol, event), intervals in overlaps.items():
        key = (sample_id, symbol)
        target = rows.setdefault(
            key,
            {
                "sample_id": sample_id,
                "gene_symbol": symbol,
                **{column: 0.0 for column in EVENT_COLUMNS},
            },
        )
        target[event] = min(
            1.0, _merged_length(intervals) / float(gene_lengths[symbol])
        )
    return pd.DataFrame(rows.values())


def _encode_features(
    *,
    sample_ids: list[str],
    gene_fractions: pd.DataFrame,
    feature_columns: list[str],
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    values = pd.DataFrame(0.0, index=sample_ids, columns=feature_columns, dtype=np.float32)
    if gene_fractions.empty:
        values.index.name = "sample_id"
        return values
    fractions = gene_fractions.set_index(["sample_id", "gene_symbol"])
    for (sample_id, gene), row in fractions.iterrows():
        sample_id = str(sample_id)
        if sample_id not in values.index:
            continue
        gain = float(row["dup_frac"] >= low_threshold) + 2.0 * float(
            row["hldup_frac"] >= high_threshold
        )
        loss = float(row["del_frac"] >= low_threshold) + 2.0 * float(
            row["hldel_frac"] >= high_threshold
        )
        gain_column = f"{gene}__gain"
        loss_column = f"{gene}__loss"
        if gain_column in values.columns:
            values.loc[sample_id, gain_column] = gain
        if loss_column in values.columns:
            values.loc[sample_id, loss_column] = loss
    values.index.name = "sample_id"
    return values


def _transform_features(
    features: pd.DataFrame,
    transform_path: Path,
    n_pcs: int,
) -> pd.DataFrame:
    transform = np.load(transform_path)
    values = features.to_numpy(dtype=np.float64)
    norms = np.sqrt(np.sum(values * values, axis=1, keepdims=True))
    normalized = values / np.maximum(norms, 1e-8)
    scaled = (normalized - transform["scaler_mean"]) / transform["scaler_scale"]
    coordinates = (scaled - transform["pca_mean"]) @ transform["pca_components"].T
    coordinates = coordinates[:, : int(n_pcs)].astype(np.float32)
    return pd.DataFrame(
        coordinates,
        index=features.index,
        columns=[f"D{index}" for index in range(1, int(n_pcs) + 1)],
    )


def _canonical_gene_fractions(
    frame: pd.DataFrame,
    sample_ids: set[str],
) -> pd.DataFrame:
    fractions = frame.copy()
    if "sample_id" not in fractions.columns:
        if "analysis_id" not in fractions.columns:
            raise KeyError("gene profiles must contain sample_id or analysis_id")
        fractions = fractions.rename(columns={"analysis_id": "sample_id"})
    required = {"sample_id", "gene_symbol", *EVENT_COLUMNS}
    missing = sorted(required - set(fractions.columns))
    if missing:
        raise KeyError(f"gene profiles are missing columns: {missing}")
    fractions["sample_id"] = fractions["sample_id"].astype(str)
    fractions["gene_symbol"] = fractions["gene_symbol"].astype(str)
    unknown = sorted(set(fractions["sample_id"]) - sample_ids)
    if unknown:
        raise ValueError(f"gene profiles contain unknown sample IDs: {unknown[:5]}")
    for column in EVENT_COLUMNS:
        fractions[column] = pd.to_numeric(
            fractions[column], errors="raise"
        ).fillna(0.0)
    if fractions.duplicated(["sample_id", "gene_symbol"]).any():
        fractions = (
            fractions.groupby(["sample_id", "gene_symbol"], as_index=False)[
                list(EVENT_COLUMNS)
            ]
            .max()
        )
    return fractions[["sample_id", "gene_symbol", *EVENT_COLUMNS]]


def _signal_sample_ids(
    fractions: pd.DataFrame,
    *,
    low_threshold: float,
    high_threshold: float,
) -> set[str]:
    if fractions.empty:
        return set()
    signal = (
        fractions["dup_frac"].ge(low_threshold)
        | fractions["del_frac"].ge(low_threshold)
        | fractions["hldup_frac"].ge(high_threshold)
        | fractions["hldel_frac"].ge(high_threshold)
    )
    return set(fractions.loc[signal, "sample_id"].astype(str))


def _load_conditional_reference(
    bundle_dir: Path,
    manifest: dict[str, object],
) -> dict[str, pd.DataFrame] | None:
    config = manifest.get("conditional_reference") or {}
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return None
    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        raise TypeError("Projection manifest files must be a mapping")
    required = {
        "contexts": "conditional_contexts",
        "coordinates": "conditional_coordinates",
        "calibration": "conditional_calibration",
        "models": "conditional_models",
    }
    missing = sorted(value for value in required.values() if value not in files)
    if missing:
        raise KeyError(f"Projection bundle lacks conditional files: {missing}")
    return {
        "contexts": pd.read_csv(
            bundle_dir / str(files[required["contexts"]]), sep="\t"
        ),
        "coordinates": pd.read_parquet(
            bundle_dir / str(files[required["coordinates"]])
        ),
        "calibration": pd.read_csv(
            bundle_dir / str(files[required["calibration"]]), sep="\t"
        ),
        "models": pd.read_csv(
            bundle_dir / str(files[required["models"]]), sep="\t"
        ),
    }


def _apply_conditional_routing(
    *,
    root: str,
    coordinates: pd.DataFrame,
    primary: pd.DataFrame,
    conditional: dict[str, pd.DataFrame] | None,
    top_candidates: int,
    nearest_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Route only uncertain primary results into the conditional reference."""

    result = primary.copy()
    result["primary_assignment_status"] = result["assignment_status"]
    result["primary_assigned_reference_key"] = result[
        "assigned_reference_key"
    ]
    result["conditional_assignment_status"] = np.where(
        result["assignment_status"].eq("projected_assignment"),
        "not_routed_primary_assignment",
        "not_available_for_root",
    )
    result["conditional_assigned_reference_key"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["assignment_layer"] = np.where(
        result["assignment_status"].eq("projected_assignment"),
        "primary",
        "none",
    )
    empty_candidates = pd.DataFrame(
        columns=[
            "sample_id",
            "root",
            "projection_layer",
            "candidate_rank",
            "context",
            "candidate_reference_key",
        ]
    )
    empty_neighbors = pd.DataFrame(columns=NEAREST_REFERENCE_COLUMNS)
    if conditional is None:
        return result, empty_candidates, empty_neighbors
    models = conditional["models"].copy()
    models["root"] = models["root"].astype(str)
    root_models = models[models["root"].eq(root)]
    eligible = result["assignment_status"].isin(["ambiguous", "unresolved"])
    if root_models.empty or not eligible.any():
        return result, empty_candidates, empty_neighbors
    if len(root_models) != 1:
        raise ValueError(f"[{root}] conditional model index is not unique")
    model = root_models.iloc[0]
    n_pcs = int(model["n_pcs"])
    dimensions = [f"D{index}" for index in range(1, n_pcs + 1)]
    missing_dimensions = sorted(set(dimensions) - set(coordinates.columns))
    if missing_dimensions:
        raise KeyError(
            f"[{root}] query coordinates lack conditional dimensions: "
            f"{missing_dimensions}"
        )
    references = conditional["coordinates"].copy()
    references["root"] = references["root"].astype(str)
    references = references[references["root"].eq(root)]
    if references.empty:
        raise ValueError(f"[{root}] conditional reference coordinates are empty")
    missing_dimensions = sorted(set(dimensions) - set(references.columns))
    if missing_dimensions:
        raise KeyError(
            f"[{root}] conditional reference lacks dimensions: {missing_dimensions}"
        )
    contexts = conditional["contexts"].copy()
    contexts["root"] = contexts["root"].astype(str)
    contexts = contexts[contexts["root"].eq(root)]
    key_map = {
        int(row.final_context): str(row.conditional_reference_key)
        for row in contexts.itertuples(index=False)
    }
    if set(references["context"].astype(int)) != set(key_map):
        raise ValueError(f"[{root}] conditional coordinate/context labels disagree")

    query_ids = result.loc[eligible, "sample_id"].astype(str).tolist()
    query_values = coordinates.loc[query_ids, dimensions].to_numpy(np.float32)
    candidates, scores, neighbors = score_reference_knn_with_neighbors(
        query_ids=query_ids,
        query_values=query_values,
        reference_ids=references["analysis_id"].astype(str).tolist(),
        reference_values=references[dimensions].to_numpy(np.float32),
        reference_labels=references["context"].astype(int).to_numpy(),
        metric=str(model["metric"]),
        n_neighbors=int(model["K"]),
        top_n=int(top_candidates),
        exclude_self=False,
        nearest_n=int(nearest_samples),
    )
    thresholds = conditional["calibration"].copy()
    thresholds["root"] = thresholds["root"].astype(str)
    thresholds = thresholds[thresholds["root"].eq(root)]
    classified = classify_reference_queries(
        scores,
        thresholds,
        root=root,
        reference_key_map=key_map,
    ).rename(
        columns={
            "analysis_id": "sample_id",
            "assignment_status": "conditional_assignment_status",
            "assigned_reference_id": "conditional_assigned_reference_key",
            "predicted_context": "conditional_predicted_context",
            "top_vote_fraction": "conditional_top_vote_fraction",
            "second_vote_fraction": "conditional_second_vote_fraction",
            "vote_margin": "conditional_vote_margin",
            "mean_neighbor_distance": "conditional_mean_neighbor_distance",
            "min_neighbor_distance": "conditional_min_neighbor_distance",
            "n_neighbors_used": "conditional_n_neighbors_used",
            "max_mean_neighbor_distance": (
                "conditional_max_mean_neighbor_distance"
            ),
        }
    )
    update_columns = [
        "sample_id",
        "conditional_assignment_status",
        "conditional_assigned_reference_key",
        "conditional_predicted_context",
        "conditional_top_vote_fraction",
        "conditional_second_vote_fraction",
        "conditional_vote_margin",
        "conditional_mean_neighbor_distance",
        "conditional_min_neighbor_distance",
        "conditional_n_neighbors_used",
        "conditional_max_mean_neighbor_distance",
    ]
    result = result.merge(
        classified[update_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_conditional_update"),
    )
    for column in (
        "conditional_assignment_status",
        "conditional_assigned_reference_key",
    ):
        update = f"{column}_conditional_update"
        result[column] = result[update].combine_first(result[column])
        result.drop(columns=update, inplace=True)
    accepted = result["conditional_assignment_status"].eq(
        "projected_assignment"
    )
    routed = result["conditional_assignment_status"].isin(
        ["projected_assignment", "ambiguous", "unresolved"]
    )
    result.loc[routed, "assignment_status"] = result.loc[
        routed, "conditional_assignment_status"
    ]
    result.loc[routed, "assigned_reference_key"] = pd.NA
    result.loc[routed, "assigned_context"] = pd.NA
    result.loc[accepted, "assigned_reference_key"] = result.loc[
        accepted, "conditional_assigned_reference_key"
    ]
    result.loc[accepted, "assigned_context"] = result.loc[
        accepted, "conditional_predicted_context"
    ].astype("Int64")
    result.loc[accepted, "assignment_layer"] = "conditional"
    result.loc[routed & ~accepted, "assignment_layer"] = "none"
    result.loc[accepted, "status_reason"] = "passes_conditional_calibrated_limits"
    result.loc[
        routed & result["assignment_status"].eq("ambiguous"), "status_reason"
    ] = "weak_conditional_vote_or_margin"
    result.loc[
        routed & result["assignment_status"].eq("unresolved"), "status_reason"
    ] = "outside_conditional_calibrated_distance"

    candidates = candidates.rename(columns={"analysis_id": "sample_id"})
    candidates.insert(1, "root", root)
    candidates.insert(2, "projection_layer", "conditional")
    candidates["candidate_reference_key"] = candidates["context"].map(key_map)
    neighbors = neighbors.rename(columns={"analysis_id": "sample_id"})
    neighbors.insert(1, "root", root)
    neighbors.insert(2, "projection_layer", "conditional")
    neighbors.insert(
        5,
        "reference_key",
        neighbors["reference_context"].map(key_map),
    )
    neighbors["distance_metric"] = str(model["metric"])
    status_by_sample = classified.set_index("sample_id")[
        "conditional_assignment_status"
    ]
    neighbors["layer_assignment_status"] = neighbors["sample_id"].map(
        status_by_sample
    )
    neighbors = neighbors[list(NEAREST_REFERENCE_COLUMNS)]
    return result, candidates, neighbors


def _project_gene_fractions_impl(
    *,
    bundle_dir: Path,
    samples: pd.DataFrame,
    fractions: pd.DataFrame,
    out_dir: Path,
    input_kind: str,
    input_provenance: dict[str, str],
    nearest_samples: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Project canonical sample-gene fractions into the frozen reference."""

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if int(nearest_samples) < 1:
        raise ValueError("nearest_samples must be at least 1")
    manifest_path = _bundle_manifest_path(bundle_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Projection output exists: {out_dir}")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "cnvsim_projection_bundle_v1":
        raise ValueError("Unsupported projection bundle schema")
    for relative, expected in manifest.get("sha256", {}).items():
        path = bundle_dir / relative
        if not path.exists() or _sha256(path) != str(expected):
            raise ValueError(f"Projection bundle checksum mismatch: {path}")

    model_table = pd.read_csv(bundle_dir / manifest["files"]["models"], sep="\t")
    anchors = pd.read_parquet(bundle_dir / manifest["files"]["anchors"])
    calibration = pd.read_csv(
        bundle_dir / manifest["files"]["calibration"], sep="\t"
    )
    conditional_reference = _load_conditional_reference(bundle_dir, manifest)
    encoding = manifest["encoding"]
    low_threshold = float(encoding["low_fraction_threshold"])
    high_threshold = float(encoding["high_fraction_threshold"])
    signal_ids = _signal_sample_ids(
        fractions,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    model_index = model_table.set_index("root")

    assignment_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    neighbor_parts: list[pd.DataFrame] = []
    coordinate_parts: list[pd.DataFrame] = []
    for root, root_samples in samples.groupby("root", dropna=False, sort=True):
        root = str(root)
        sample_ids = root_samples["sample_id"].astype(str).tolist()
        if root not in model_index.index:
            assignment_parts.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_ids,
                        "root": root,
                        "assignment_status": "out_of_scope_root",
                        "assigned_reference_key": pd.NA,
                        "status_reason": "root_not_in_frozen_reference",
                    }
                )
            )
            continue
        model = model_index.loc[root]
        mode = str(model["projection_mode"])
        root_fractions = (
            fractions[fractions["sample_id"].astype(str).isin(sample_ids)]
            if not fractions.empty
            else fractions
        )
        no_signal = [
            sample_id for sample_id in sample_ids if sample_id not in signal_ids
        ]
        if no_signal:
            assignment_parts.append(
                pd.DataFrame(
                    {
                        "sample_id": no_signal,
                        "root": root,
                        "assignment_status": "no_autosomal_gene_cnv_signal",
                        "assigned_reference_key": pd.NA,
                        "status_reason": "no_segment_overlap_with_reference_genes",
                    }
                )
            )
        evaluable_ids = [
            sample_id for sample_id in sample_ids if sample_id in signal_ids
        ]
        if not evaluable_ids:
            continue
        if mode == "root_only":
            reference_key = str(model["reference_key"])
            assignment_parts.append(
                pd.DataFrame(
                    {
                        "sample_id": evaluable_ids,
                        "root": root,
                        "assignment_status": "root_only_reference",
                        "assigned_reference_key": reference_key,
                        "status_reason": "root_has_no_stable_subtype_partition",
                    }
                )
            )
            continue

        features = json.loads((bundle_dir / str(model["feature_file"])).read_text())
        encoded = _encode_features(
            sample_ids=evaluable_ids,
            gene_fractions=root_fractions,
            feature_columns=features,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        zero_encoded = encoded.abs().sum(axis=1).eq(0)
        if zero_encoded.any():
            zero_ids = encoded.index[zero_encoded].astype(str).tolist()
            assignment_parts.append(
                pd.DataFrame(
                    {
                        "sample_id": zero_ids,
                        "root": root,
                        "assignment_status": "no_autosomal_gene_cnv_signal",
                        "assigned_reference_key": pd.NA,
                        "status_reason": "no_signal_in_root_specific_feature_set",
                    }
                )
            )
            encoded = encoded.loc[~zero_encoded]
        if encoded.empty:
            continue

        n_pcs = int(model["n_pcs"])
        coordinates = _transform_features(
            encoded,
            bundle_dir / str(model["transform_file"]),
            n_pcs,
        )
        coordinate_output = coordinates.reset_index()
        coordinate_output.insert(1, "root", root)
        coordinate_parts.append(coordinate_output)
        dimensions = list(coordinates.columns)
        root_anchors = anchors[anchors["root"].astype(str).eq(root)].copy()
        candidates, scores, primary_neighbors = (
            score_reference_knn_with_neighbors(
                query_ids=coordinates.index.astype(str).tolist(),
                query_values=coordinates.to_numpy(np.float32),
                reference_ids=root_anchors["analysis_id"].astype(str).tolist(),
                reference_values=root_anchors[dimensions].to_numpy(np.float32),
                reference_labels=root_anchors["context"].astype(int).to_numpy(),
                metric=str(model["metric"]),
                n_neighbors=int(model["K"]),
                top_n=int(manifest["top_candidates"]),
                exclude_self=False,
                nearest_n=int(nearest_samples),
            )
        )
        thresholds = calibration[calibration["root"].astype(str).eq(root)].copy()
        classified = classify_reference_queries(
            scores, thresholds, root=root
        ).rename(
            columns={
                "analysis_id": "sample_id",
                "assigned_reference_id": "assigned_reference_key",
            }
        )
        if "root" in classified.columns:
            classified["root"] = root
        else:
            classified.insert(1, "root", root)
        classified["status_reason"] = np.select(
            [
                classified["assignment_status"].eq("projected_assignment"),
                classified["assignment_status"].eq("ambiguous"),
            ],
            ["passes_calibrated_limits", "weak_vote_or_margin"],
            default="outside_calibrated_distance_or_missing_calibration",
        )
        primary_status = classified.set_index("sample_id")["assignment_status"]
        primary_neighbors = primary_neighbors.rename(
            columns={"analysis_id": "sample_id"}
        )
        primary_neighbors.insert(1, "root", root)
        primary_neighbors.insert(2, "projection_layer", "primary")
        primary_key_map = (
            root_anchors[["context", "assigned_reference_key"]]
            .drop_duplicates()
            .set_index("context")["assigned_reference_key"]
        )
        primary_neighbors.insert(
            5,
            "reference_key",
            primary_neighbors["reference_context"].map(primary_key_map),
        )
        primary_neighbors["distance_metric"] = str(model["metric"])
        primary_neighbors["layer_assignment_status"] = primary_neighbors[
            "sample_id"
        ].map(primary_status)
        primary_neighbors = primary_neighbors[
            list(NEAREST_REFERENCE_COLUMNS)
        ]
        classified, conditional_candidates, conditional_neighbors = (
            _apply_conditional_routing(
                root=root,
                coordinates=coordinates,
                primary=classified,
                conditional=conditional_reference,
                top_candidates=int(manifest["top_candidates"]),
                nearest_samples=int(nearest_samples),
            )
        )
        assignment_parts.append(classified)
        candidates = candidates.rename(columns={"analysis_id": "sample_id"})
        candidates.insert(1, "root", root)
        candidates.insert(2, "projection_layer", "primary")
        candidates["candidate_reference_key"] = [
            f"{root}-CTX{int(context)}" for context in candidates["context"]
        ]
        candidate_parts.append(candidates)
        neighbor_parts.append(primary_neighbors)
        if not conditional_candidates.empty:
            candidate_parts.append(conditional_candidates)
        if not conditional_neighbors.empty:
            neighbor_parts.append(conditional_neighbors)

    assignments = _concat_preserving_all_na_columns(assignment_parts)
    if assignments["sample_id"].astype(str).duplicated().any():
        duplicates = assignments[
            assignments["sample_id"].astype(str).duplicated(keep=False)
        ]
        raise ValueError(
            "Projection produced duplicate sample assignments: "
            f"{duplicates[['sample_id', 'root', 'assignment_status', 'status_reason']].to_dict(orient='records')[:10]}"
        )
    assignments = samples[["sample_id", "root"]].merge(
        assignments.drop(columns="root"),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if assignments["assignment_status"].isna().any():
        raise ValueError("Some query samples were not accounted for")
    if "assignment_layer" not in assignments:
        assignments["assignment_layer"] = pd.NA
    fallback_layer = pd.Series(
        np.where(
            assignments["assignment_status"].eq("root_only_reference"),
            "root",
            "none",
        ),
        index=assignments.index,
    )
    assignments["assignment_layer"] = assignments[
        "assignment_layer"
    ].combine_first(fallback_layer)
    for column in (
        "primary_assignment_status",
        "conditional_assignment_status",
    ):
        if column not in assignments:
            assignments[column] = pd.NA
        assignments[column] = assignments[column].fillna("not_applicable")
    candidates = (
        pd.concat(candidate_parts, ignore_index=True, sort=False)
        if candidate_parts
        else pd.DataFrame(
            columns=[
                "sample_id",
                "root",
                "projection_layer",
                "candidate_rank",
                "context",
                "candidate_reference_key",
            ]
        )
    )
    nearest_references = (
        pd.concat(neighbor_parts, ignore_index=True, sort=False)
        if neighbor_parts
        else pd.DataFrame(columns=NEAREST_REFERENCE_COLUMNS)
    )
    nearest_references = nearest_references.merge(
        assignments[["sample_id", "assignment_layer"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    nearest_references["is_final_assignment_layer"] = (
        nearest_references["projection_layer"].eq(
            nearest_references["assignment_layer"]
        )
        & nearest_references["assignment_layer"].isin(["primary", "conditional"])
    )
    nearest_references.drop(columns="assignment_layer", inplace=True)
    coordinates = (
        pd.concat(coordinate_parts, ignore_index=True, sort=False)
        if coordinate_parts
        else pd.DataFrame(columns=["sample_id", "root"])
    )
    outputs = {
        "assignments": out_dir / "projection_assignments.tsv",
        "candidates": out_dir / "projection_candidates.tsv",
        "nearest_reference_samples": (
            out_dir / "nearest_reference_samples.tsv"
        ),
        "coordinates": out_dir / "query_coordinates.parquet",
    }
    assignments.to_csv(outputs["assignments"], sep="\t", index=False)
    candidates.to_csv(outputs["candidates"], sep="\t", index=False)
    nearest_references.to_csv(
        outputs["nearest_reference_samples"], sep="\t", index=False
    )
    coordinates.to_parquet(outputs["coordinates"], index=False)
    output_manifest = {
        "schema_version": "cnvsim_projection_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": _sha256(manifest_path),
        "input_kind": str(input_kind),
        "inputs": dict(input_provenance),
        "counts": {
            "samples": int(len(assignments)),
            "nearest_reference_rows": int(len(nearest_references)),
            "samples_with_nearest_references": int(
                nearest_references["sample_id"].nunique()
            ),
            "assignment_status": {
                str(status): int(count)
                for status, count in assignments[
                    "assignment_status"
                ].value_counts().items()
            },
            "assignment_layer": {
                str(layer): int(count)
                for layer, count in assignments[
                    "assignment_layer"
                ].value_counts().items()
            },
        },
        "nearest_reference_samples": {
            "maximum_per_query_layer": int(nearest_samples),
            "ranking": "ascending_distance",
        },
        "outputs": {key: path.name for key, path in outputs.items()},
    }
    result_manifest_path = out_dir / "projection_manifest.json"
    result_manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    return {**outputs, "manifest": result_manifest_path}


def project_samples(
    *,
    bundle_dir: Path,
    samples_path: Path,
    segments_path: Path,
    out_dir: Path,
    nearest_samples: int = 5,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Project external segment profiles into the frozen CNVSIM reference."""

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    samples_path = Path(samples_path).expanduser().resolve()
    segments_path = Path(segments_path).expanduser().resolve()
    manifest_path = _bundle_manifest_path(bundle_dir)
    manifest = json.loads(manifest_path.read_text())
    genes = pd.read_csv(bundle_dir / manifest["files"]["genes"], sep="\t")
    samples = _canonical_samples(_read_table(samples_path))
    segments = _canonical_segments(
        _read_table(segments_path), set(samples["sample_id"])
    )
    fractions = segment_gene_fractions(segments=segments, genes=genes)
    return _project_gene_fractions_impl(
        bundle_dir=bundle_dir,
        samples=samples,
        fractions=fractions,
        out_dir=out_dir,
        input_kind="segments",
        input_provenance={
            "samples": str(samples_path),
            "segments": str(segments_path),
        },
        nearest_samples=int(nearest_samples),
        overwrite=overwrite,
    )


def project_gene_profiles(
    *,
    bundle_dir: Path,
    samples: pd.DataFrame,
    gene_fractions: pd.DataFrame,
    out_dir: Path,
    input_provenance: dict[str, str] | None = None,
    nearest_samples: int = 5,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Project precomputed sample-gene CNV fractions into the frozen atlas."""

    canonical_samples = _canonical_samples(samples)
    canonical_fractions = _canonical_gene_fractions(
        gene_fractions,
        set(canonical_samples["sample_id"]),
    )
    return _project_gene_fractions_impl(
        bundle_dir=bundle_dir,
        samples=canonical_samples,
        fractions=canonical_fractions,
        out_dir=out_dir,
        input_kind="gene_fractions",
        input_provenance=input_provenance or {},
        nearest_samples=int(nearest_samples),
        overwrite=overwrite,
    )
