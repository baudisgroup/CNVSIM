from __future__ import annotations

"""Leakage-controlled fixed-reference hold-out projection."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib

import joblib
import numpy as np
import pandas as pd

from cnvsim.analysis.config import ReferenceProjectionParams
from cnvsim.analysis.ir import dimension_columns, read_analysis_coordinates
from cnvsim.validation.config import HoldoutValidationParams
from cnvsim.analysis.reference_projection import (
    classify_reference_queries,
    score_reference_knn,
    calibrate_anchor_projection,
)
from cnvsim.preprocess.encode import add_cnv_event_flags, compute_feature_burden
from cnvsim.preprocess.gene_panel_store import load_gene_panel_subset_from_cache
from cnvsim.preprocess.scaling import l2_normalize_rows


@dataclass(frozen=True)
class HoldoutWorkspace:
    """Canonical paths for one fixed-reference hold-out evaluation."""

    atlas_dir: Path
    validation_dir: Path
    split_path: Path
    train_work_base: Path
    train_run_name: str
    train_run_dir: Path
    train_preprocess_release: Path
    train_analysis_base: Path
    train_analysis_dir: Path

    @classmethod
    def create(
        cls,
        *,
        atlas_dir: Path,
        validation_dir: Path | None = None,
        train_run_name: str = "preprocess_holdout_v1_train",
    ) -> "HoldoutWorkspace":
        atlas = Path(atlas_dir).expanduser().resolve()
        validation = (
            Path(validation_dir).expanduser().resolve()
            if validation_dir is not None
            else atlas / "work" / "validation" / "holdout_v1"
        )
        train_work_base = validation / "train_work"
        train_analysis_base = validation / "train_analysis"
        return cls(
            atlas_dir=atlas,
            validation_dir=validation,
            split_path=validation / "split.parquet",
            train_work_base=train_work_base,
            train_run_name=str(train_run_name),
            train_run_dir=train_work_base / str(train_run_name),
            train_preprocess_release=validation / "train_preprocessing",
            train_analysis_base=train_analysis_base,
            train_analysis_dir=train_analysis_base / "analysis",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_ids(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.dropna().astype(str).unique())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    random_seed: int,
    context_token: str,
) -> pd.Series:
    """Balance platform/source strata while keeping context fold sizes close."""

    assigned = pd.Series(-1, index=frame.index, dtype=int)
    context_counts = np.zeros(int(n_folds), dtype=int)
    grouped = list(frame.groupby(["_platform", "_source"], sort=True))
    grouped.sort(key=lambda item: (-len(item[1]), str(item[0])))
    for stratum, members in grouped:
        token = zlib.crc32(f"{context_token}:{stratum}".encode("utf-8"))
        rng = np.random.default_rng(int(random_seed) + int(token))
        indices = rng.permutation(members.index.to_numpy())
        stratum_counts = np.zeros(int(n_folds), dtype=int)
        priority = rng.permutation(int(n_folds))
        rank = {int(fold): pos for pos, fold in enumerate(priority)}
        for index in indices:
            fold = min(
                range(int(n_folds)),
                key=lambda value: (
                    int(stratum_counts[value]),
                    int(context_counts[value]),
                    int(rank[value]),
                ),
            )
            assigned.loc[index] = int(fold)
            stratum_counts[fold] += 1
            context_counts[fold] += 1
    if assigned.lt(0).any():
        raise ValueError(f"[{context_token}] some samples lack a hold-out fold")
    return assigned


def build_holdout_split(
    *,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    params: HoldoutValidationParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign stable-anchor members to deterministic platform/source-aware folds."""

    required_assignments = {
        "analysis_id",
        "root",
        "pipeline_status",
        "assigned_reference_key",
    }
    missing = sorted(required_assignments - set(assignments.columns))
    if missing:
        raise KeyError(f"Release assignments are missing columns: {missing}")
    required_metadata = {"analysis_id"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise KeyError(f"Source metadata are missing columns: {missing}")

    anchors = assignments[
        assignments["pipeline_status"].astype(str).eq("anchor_member")
    ][
        ["analysis_id", "root", "assigned_reference_key"]
    ].copy()
    anchors["analysis_id"] = anchors["analysis_id"].astype(str)
    anchors["root"] = anchors["root"].astype(str)
    anchors = anchors.rename(
        columns={"assigned_reference_key": "full_reference_key"}
    )
    if anchors.empty or anchors["full_reference_key"].isna().any():
        raise ValueError("No fully labelled stable-anchor members were found")
    if anchors["analysis_id"].duplicated().any():
        raise ValueError("Stable-anchor assignments contain duplicate analyses")

    meta = metadata.copy()
    meta["analysis_id"] = meta["analysis_id"].astype(str)
    keep = [
        "analysis_id",
        "platform_family_refined",
        "source_group_refined",
    ]
    meta = meta[[column for column in keep if column in meta.columns]].drop_duplicates(
        "analysis_id", keep="first"
    )
    anchors = anchors.merge(meta, on="analysis_id", how="left", validate="one_to_one")
    anchors["_platform"] = anchors.get(
        "platform_family_refined", pd.Series("Unknown", index=anchors.index)
    ).fillna("Unknown").astype(str)
    anchors["_source"] = anchors.get(
        "source_group_refined", pd.Series("Unknown", index=anchors.index)
    ).fillna("Unknown").astype(str)

    folds = pd.Series(-1, index=anchors.index, dtype=int)
    for (root, reference), indices in anchors.groupby(
        ["root", "full_reference_key"], sort=True
    ).groups.items():
        part = anchors.loc[list(indices)]
        if len(part) < int(params.n_folds):
            raise ValueError(
                f"[{reference}] {len(part)} anchors cannot support "
                f"{params.n_folds} folds"
            )
        folds.loc[part.index] = _balanced_folds(
            part,
            n_folds=int(params.n_folds),
            random_seed=int(params.random_seed),
            context_token=f"{root}:{reference}",
        )
    anchors["fold"] = folds
    anchors["holdout_role"] = np.where(
        anchors["fold"].eq(int(params.test_fold)), "test", "train"
    )

    context_summary = (
        anchors.groupby(["root", "full_reference_key"], as_index=False)
        .agg(
            n_total=("analysis_id", "size"),
            n_train=("holdout_role", lambda values: int((values == "train").sum())),
            n_test=("holdout_role", lambda values: int((values == "test").sum())),
            n_platforms=("_platform", "nunique"),
            n_sources=("_source", "nunique"),
        )
        .sort_values(["root", "full_reference_key"])
    )
    if context_summary["n_train"].lt(int(params.min_train_per_context)).any():
        bad = context_summary[
            context_summary["n_train"].lt(int(params.min_train_per_context))
        ][["full_reference_key", "n_train"]]
        raise ValueError(
            "Some contexts have too few train anchors: "
            f"{bad.to_dict(orient='records')}"
        )
    if context_summary["n_test"].lt(1).any():
        raise ValueError("Every stable context must contribute held-out samples")

    root_summary = (
        anchors.groupby("root", as_index=False)
        .agg(
            n_total=("analysis_id", "size"),
            n_train=("holdout_role", lambda values: int((values == "train").sum())),
            n_test=("holdout_role", lambda values: int((values == "test").sum())),
            n_contexts=("full_reference_key", "nunique"),
        )
        .sort_values("root")
    )
    output_columns = [
        "analysis_id",
        "root",
        "full_reference_key",
        "platform_family_refined",
        "source_group_refined",
        "fold",
        "holdout_role",
    ]
    for column in output_columns:
        if column not in anchors.columns:
            anchors[column] = pd.NA
    return (
        anchors[output_columns].sort_values(["root", "full_reference_key", "analysis_id"]),
        context_summary.reset_index(drop=True),
        root_summary.reset_index(drop=True),
    )


def prepare_holdout_workspace(
    *,
    workspace: HoldoutWorkspace,
    params: HoldoutValidationParams,
    baseline_run_name: str = "preprocess_v1_baseline",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create the split and filtered train-only metadata adapter output."""

    release_dir = workspace.atlas_dir / "release_candidate" / "tables"
    assignments_path = release_dir / "analysis_assignments.parquet"
    source_metadata_path = (
        workspace.atlas_dir
        / "source_file"
        / "metadata"
        / "tumour_analysis_metadata.parquet"
    )
    baseline_meta_path = (
        workspace.atlas_dir / "work" / str(baseline_run_name) / "meta_cleaned.tsv"
    )
    for path in (assignments_path, source_metadata_path, baseline_meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if workspace.split_path.exists() and not overwrite:
        raise FileExistsError(
            f"Hold-out split exists; pass --overwrite to replace it: {workspace.split_path}"
        )

    assignments = pd.read_parquet(assignments_path)
    source_metadata = pd.read_parquet(source_metadata_path)
    split, context_summary, root_summary = build_holdout_split(
        assignments=assignments,
        metadata=source_metadata,
        params=params,
    )
    test_ids = set(
        split.loc[split["holdout_role"].eq("test"), "analysis_id"].astype(str)
    )
    baseline_meta = pd.read_csv(
        baseline_meta_path,
        sep="\t",
        dtype={"analysis_id": str},
        low_memory=False,
    )
    baseline_meta["analysis_id"] = baseline_meta["analysis_id"].astype(str)
    missing_test = test_ids - set(baseline_meta["analysis_id"])
    if missing_test:
        raise ValueError(f"{len(missing_test)} held-out anchors lack baseline metadata")
    train_meta = baseline_meta[~baseline_meta["analysis_id"].isin(test_ids)].copy()
    if set(train_meta["analysis_id"]) & test_ids:
        raise AssertionError("Held-out analyses remain in train metadata")

    workspace.validation_dir.mkdir(parents=True, exist_ok=True)
    workspace.train_run_dir.mkdir(parents=True, exist_ok=True)
    context_summary_path = workspace.validation_dir / "split_by_context.tsv"
    root_summary_path = workspace.validation_dir / "split_by_root.tsv"
    train_meta_path = workspace.train_run_dir / "meta_cleaned.tsv"
    manifest_path = workspace.validation_dir / "prepare_manifest.json"
    split.to_parquet(workspace.split_path, index=False)
    context_summary.to_csv(context_summary_path, sep="\t", index=False)
    root_summary.to_csv(root_summary_path, sep="\t", index=False)
    train_meta.to_csv(train_meta_path, sep="\t", index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": _now(),
                "validation_scope": "fixed_reference_projection_holdout",
                "params": {
                    "n_folds": int(params.n_folds),
                    "test_fold": int(params.test_fold),
                    "random_seed": int(params.random_seed),
                    "min_train_per_context": int(params.min_train_per_context),
                    "baseline_run_name": str(baseline_run_name),
                },
                "counts": {
                    "anchors": int(len(split)),
                    "contexts": int(split["full_reference_key"].nunique()),
                    "roots": int(split["root"].nunique()),
                    "train_anchors": int(split["holdout_role"].eq("train").sum()),
                    "test_anchors": int(split["holdout_role"].eq("test").sum()),
                    "train_metadata_rows": int(len(train_meta)),
                },
                "test_id_sha256": _sha256_ids(
                    split.loc[split["holdout_role"].eq("test"), "analysis_id"]
                ),
                "inputs": {
                    "release_assignments": str(assignments_path),
                    "source_metadata": str(source_metadata_path),
                    "baseline_metadata": str(baseline_meta_path),
                },
                "outputs": {
                    "split": str(workspace.split_path),
                    "context_summary": str(context_summary_path),
                    "root_summary": str(root_summary_path),
                    "train_metadata": str(train_meta_path),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "split": workspace.split_path,
        "context_summary": context_summary_path,
        "root_summary": root_summary_path,
        "train_metadata": train_meta_path,
        "manifest": manifest_path,
    }


def audit_holdout_leakage(
    *,
    workspace: HoldoutWorkspace,
) -> pd.DataFrame:
    """Fail if any held-out analysis appears in a fitted train-only artifact."""

    split = pd.read_parquet(workspace.split_path)
    test_ids = set(
        split.loc[split["holdout_role"].eq("test"), "analysis_id"].astype(str)
    )
    checks: list[tuple[str, set[str]]] = []
    train_meta_path = workspace.train_run_dir / "meta_cleaned.tsv"
    if train_meta_path.exists():
        frame = pd.read_csv(
            train_meta_path,
            sep="\t",
            usecols=["analysis_id"],
            dtype={"analysis_id": str},
        )
        checks.append(("train_metadata", set(frame["analysis_id"].astype(str))))
    for path in sorted((workspace.train_run_dir / "features_unfiltered").glob("*_features.parquet")):
        frame = pd.read_parquet(path)
        identifiers = (
            set(frame["analysis_id"].astype(str))
            if "analysis_id" in frame.columns
            else set(frame.index.astype(str))
        )
        checks.append((f"encoded_features:{path.stem}", identifiers))
    ir_dir = workspace.train_analysis_dir / "ir" / "perroot"
    for path in sorted(ir_dir.glob("*/coords.parquet")):
        frame = read_analysis_coordinates(path)
        checks.append((f"analysis_ir:{path.parent.name}", set(frame.index.astype(str))))
    status_path = workspace.train_analysis_dir / "reference_contexts" / "sample_reference_status.parquet"
    if status_path.exists():
        frame = pd.read_parquet(status_path)
        checks.append(("reference_contexts", set(frame["analysis_id"].astype(str))))

    rows = []
    for artifact, identifiers in checks:
        overlap = test_ids & identifiers
        rows.append(
            {
                "artifact": artifact,
                "n_artifact_samples": int(len(identifiers)),
                "n_heldout_leaks": int(len(overlap)),
                "example_heldout_leaks": ";".join(sorted(overlap)[:5]),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and result["n_heldout_leaks"].gt(0).any():
        raise ValueError(
            "Held-out analyses leaked into train-only artifacts:\n"
            + result[result["n_heldout_leaks"].gt(0)].to_string(index=False)
        )
    return result


def _resolve_model_path(models_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else models_dir / path.name


def _encode_fixed_features(
    *,
    analysis_ids: list[str],
    feature_columns: list[str],
    gene_panel: pd.DataFrame,
    min_frac_low: float,
    min_frac_high: float,
) -> pd.DataFrame:
    genes = sorted(
        {
            str(column).rsplit("__", 1)[0]
            for column in feature_columns
            if str(column).endswith(("__gain", "__loss"))
        }
    )
    if gene_panel.empty:
        return pd.DataFrame(
            0.0,
            index=pd.Index(analysis_ids, name="analysis_id"),
            columns=feature_columns,
            dtype="float32",
        )
    required = {
        "analysis_id",
        "gene_symbol",
        "dup_frac",
        "del_frac",
        "hldup_frac",
        "hldel_frac",
    }
    missing = sorted(required - set(gene_panel.columns))
    if missing:
        raise KeyError(f"Gene-panel query is missing columns: {missing}")
    panel = gene_panel[gene_panel["gene_symbol"].astype(str).isin(genes)].copy()
    if panel.empty:
        return pd.DataFrame(
            0.0,
            index=pd.Index(analysis_ids, name="analysis_id"),
            columns=feature_columns,
            dtype="float32",
        )
    flags = add_cnv_event_flags(
        panel,
        min_frac_low=float(min_frac_low),
        min_frac_high=float(min_frac_high),
    )
    flags["gain_score"] = flags["low_gain"].astype(float) + 2.0 * flags[
        "hl_gain"
    ].astype(float)
    flags["loss_score"] = flags["low_loss"].astype(float) + 2.0 * flags[
        "hl_loss"
    ].astype(float)
    aggregated = flags.groupby(["analysis_id", "gene_symbol"], as_index=False)[
        ["gain_score", "loss_score"]
    ].max()
    gain = aggregated.pivot(
        index="analysis_id", columns="gene_symbol", values="gain_score"
    ).add_suffix("__gain")
    loss = aggregated.pivot(
        index="analysis_id", columns="gene_symbol", values="loss_score"
    ).add_suffix("__loss")
    encoded = pd.concat([gain, loss], axis=1)
    encoded.index = encoded.index.astype(str)
    return (
        encoded.reindex(index=analysis_ids, columns=feature_columns, fill_value=0.0)
        .fillna(0.0)
        .astype("float32")
    )


def transform_holdout_queries(
    *,
    workspace: HoldoutWorkspace,
    gene_panel_cache: Path,
    min_frac_low: float,
    min_frac_high: float,
    n_dims: int,
) -> pd.DataFrame:
    """Encode held-out profiles and apply only train-fitted Stage 2A models."""

    split = pd.read_parquet(workspace.split_path)
    test = split[split["holdout_role"].eq("test")].copy()
    models_dir = workspace.train_preprocess_release / "reference_space" / "models_stage2A"
    manifest_path = models_dir / "models_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    models = pd.read_csv(manifest_path, sep="\t")
    root_column = "icdot_root" if "icdot_root" in models.columns else "root"
    models[root_column] = models[root_column].astype(str)
    models = models.set_index(root_column)

    parts: list[pd.DataFrame] = []
    grouped_test = list(test.groupby("root", sort=True))
    for root_index, (root, root_test) in enumerate(grouped_test, start=1):
        print(
            f"[holdout transform] [{root_index}/{len(grouped_test)}] "
            f"root={root} queries={len(root_test):,}",
            flush=True,
        )
        if root not in models.index:
            raise ValueError(f"[{root}] train-only PCA model is unavailable")
        model = models.loc[root]
        feature_path = _resolve_model_path(models_dir, model["feature_columns_path"])
        scaler_path = _resolve_model_path(models_dir, model["scaler_path"])
        pca_path = _resolve_model_path(models_dir, model["pca_path"])
        for path in (feature_path, scaler_path, pca_path):
            if not path.exists():
                raise FileNotFoundError(path)
        feature_columns = list(json.loads(feature_path.read_text()))
        gene_symbols = sorted(
            {str(column).rsplit("__", 1)[0] for column in feature_columns}
        )
        identifiers = root_test["analysis_id"].astype(str).tolist()
        panel = load_gene_panel_subset_from_cache(
            cache_dir=Path(gene_panel_cache),
            analysis_ids=identifiers,
            gene_symbols=gene_symbols,
            columns=[
                "analysis_id",
                "gene_symbol",
                "dup_frac",
                "del_frac",
                "hldup_frac",
                "hldel_frac",
            ],
        )
        encoded = _encode_fixed_features(
            analysis_ids=identifiers,
            feature_columns=feature_columns,
            gene_panel=panel,
            min_frac_low=float(min_frac_low),
            min_frac_high=float(min_frac_high),
        )
        burden = compute_feature_burden(encoded)
        normalized = l2_normalize_rows(encoded)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        values = pca.transform(scaler.transform(normalized.to_numpy(np.float32)))
        dimensions = min(int(n_dims), int(values.shape[1]))
        coords = pd.DataFrame(
            values[:, :dimensions],
            index=encoded.index,
            columns=[f"D{index + 1}" for index in range(dimensions)],
        )
        coords.index.name = "analysis_id"
        coords = coords.reset_index().merge(
            root_test.drop(columns=["holdout_role"]),
            on="analysis_id",
            how="left",
            validate="one_to_one",
        )
        coords = coords.merge(
            burden.reset_index(),
            on="analysis_id",
            how="left",
            validate="one_to_one",
        )
        parts.append(coords)
    if not parts:
        raise ValueError("No held-out query coordinates were produced")
    return pd.concat(parts, ignore_index=True, sort=False)






def _metric_row(frame: pd.DataFrame, **labels: object) -> dict[str, object]:
    n = int(len(frame))
    projected = frame["assignment_status"].eq("projected_assignment")
    correct = frame["correct_assignment"].fillna(False).astype(bool)
    raw_scored = frame["predicted_full_reference_key"].notna()
    truth_recoverable = frame["truth_context_recovered"].fillna(False).astype(bool)
    return {
        **labels,
        "n_test": n,
        "evaluable_fraction": float(frame["query_evaluable"].mean()) if n else np.nan,
        "truth_context_recovered_fraction": float(truth_recoverable.mean()) if n else np.nan,
        "raw_top1_accuracy": (
            float(frame.loc[raw_scored, "raw_top1_correct"].mean())
            if raw_scored.any()
            else np.nan
        ),
        "assignment_coverage": float(projected.mean()) if n else np.nan,
        "conditional_accuracy": (
            float(correct[projected].mean()) if projected.any() else np.nan
        ),
        "high_confidence_recall": float(correct.sum() / n) if n else np.nan,
        "recoverable_high_confidence_recall": (
            float(correct[truth_recoverable].mean())
            if truth_recoverable.any()
            else np.nan
        ),
        "ambiguous_fraction": float(frame["assignment_status"].eq("ambiguous").mean()) if n else np.nan,
        "unresolved_fraction": float(frame["assignment_status"].eq("unresolved").mean()) if n else np.nan,
        "incorrect_assignment_fraction": float((projected & ~correct).mean()) if n else np.nan,
    }


def summarize_holdout_predictions(
    predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Summarize abstention-aware projection performance at four levels."""

    tables: dict[str, pd.DataFrame] = {
        "global": pd.DataFrame([_metric_row(predictions, scope="all")])
    }
    specifications = {
        "root": ["root"],
        "context": ["root", "full_reference_key"],
        "platform": ["platform_family_refined"],
        "source": ["source_group_refined"],
    }
    for name, columns in specifications.items():
        rows = []
        for keys, frame in predictions.groupby(columns, dropna=False, sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            labels = dict(zip(columns, key_values))
            rows.append(_metric_row(frame, **labels))
        tables[name] = pd.DataFrame(rows)
    return tables


def summarize_candidate_retrieval(candidates: pd.DataFrame) -> pd.DataFrame:
    """Report whether the top-k train references contain the frozen truth."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "top_k",
                "n_scored_queries",
                "truth_in_top_k_fraction",
            ]
        )
    rows = []
    for top_k in (1, 3, 5):
        subset = candidates[candidates["candidate_rank"].le(top_k)]
        truth_hit = subset.groupby("analysis_id")["candidate_maps_to_truth"].max()
        rows.append(
            {
                "top_k": top_k,
                "n_scored_queries": int(truth_hit.size),
                "truth_in_top_k_fraction": float(truth_hit.mean()),
            }
        )
    return pd.DataFrame(rows)






def _holdout_error_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions[
            predictions["assignment_status"].eq("projected_assignment")
            & predictions["incorrect_assignment"]
        ]
        .groupby(
            ["root", "full_reference_key", "predicted_full_reference_key"],
            dropna=False,
            as_index=False,
        )
        .agg(n_test=("analysis_id", "size"))
        .sort_values(["n_test", "root"], ascending=[False, True])
        .reset_index(drop=True)
    )




def project_and_evaluate_fixed_labels(
    *,
    workspace: HoldoutWorkspace,
    query_coords: pd.DataFrame,
    full_parameters_path: Path,
    projection_params: ReferenceProjectionParams,
) -> dict[str, pd.DataFrame]:
    """Project hold-outs into train-only references with fixed atlas labels."""

    ir_dir = workspace.train_analysis_dir / "ir"
    for path in (workspace.split_path, full_parameters_path, ir_dir):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    split = pd.read_parquet(workspace.split_path)
    split["analysis_id"] = split["analysis_id"].astype(str)
    split["root"] = split["root"].astype(str)
    train_labels = split[split["holdout_role"].eq("train")].copy()
    parameters = pd.read_csv(full_parameters_path, sep="\t")
    parameters["root"] = parameters["root"].astype(str)
    if parameters["root"].duplicated().any():
        raise ValueError("Full-atlas parameters contain duplicate roots")
    parameters = parameters.set_index("root")

    prediction_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    root_rows: list[dict[str, object]] = []
    grouped_queries = list(query_coords.groupby("root", sort=True))
    for root_index, (root, root_queries) in enumerate(grouped_queries, start=1):
        print(
            f"[fixed-label projection] [{root_index}/{len(grouped_queries)}] "
            f"root={root} queries={len(root_queries):,}",
            flush=True,
        )
        base_columns = [
            "analysis_id",
            "root",
            "full_reference_key",
            "platform_family_refined",
            "source_group_refined",
            "n_genes_altered",
            "architecture_informative",
        ]
        base = root_queries[
            [column for column in base_columns if column in root_queries.columns]
        ].copy()
        base["query_evaluable"] = base["architecture_informative"].fillna(
            False
        ).astype(bool)
        root_labels = train_labels[train_labels["root"].eq(root)].copy()
        coords_path = ir_dir / "perroot" / root / "coords.parquet"
        if root not in parameters.index or not coords_path.exists():
            base["assignment_status"] = "unresolved"
            base["projection_reason"] = "train_root_reference_unavailable"
            prediction_parts.append(base)
            root_rows.append(
                {
                    "root": root,
                    "n_train_labels": int(len(root_labels)),
                    "n_train_coordinates": 0,
                    "n_train_missing_coordinates": int(len(root_labels)),
                    "n_fixed_contexts": int(
                        root_labels["full_reference_key"].nunique()
                    ),
                    "n_calibrated_contexts": 0,
                }
            )
            continue

        parameter = parameters.loc[root]
        metric = str(parameter["metric"]).lower()
        n_pcs = int(parameter["n_pcs"])
        k = int(parameter["K"])
        train_coords = read_analysis_coordinates(coords_path)
        dimensions = dimension_columns(train_coords)[:n_pcs]
        query_dimensions = [f"D{index + 1}" for index in range(n_pcs)]
        if len(dimensions) != n_pcs or not set(query_dimensions).issubset(
            root_queries.columns
        ):
            raise ValueError(
                f"[{root}] train/query coordinate dimensions are incomplete"
            )
        available = root_labels["analysis_id"].isin(train_coords.index)
        available_labels = root_labels[available].copy()
        reference_keys = sorted(
            available_labels["full_reference_key"].astype(str).unique()
        )
        key_to_context = {
            reference_key: context
            for context, reference_key in enumerate(reference_keys)
        }
        context_to_key = {
            context: reference_key
            for reference_key, context in key_to_context.items()
        }
        if len(available_labels) < 2 or not reference_keys:
            base["assignment_status"] = "unresolved"
            base["projection_reason"] = "train_root_reference_unavailable"
            prediction_parts.append(base)
            root_rows.append(
                {
                    "root": root,
                    "n_train_labels": int(len(root_labels)),
                    "n_train_coordinates": int(len(available_labels)),
                    "n_train_missing_coordinates": int((~available).sum()),
                    "n_fixed_contexts": int(len(reference_keys)),
                    "n_calibrated_contexts": 0,
                }
            )
            continue

        anchor_ids = available_labels["analysis_id"].astype(str).tolist()
        anchor_contexts = (
            available_labels["full_reference_key"]
            .astype(str)
            .map(key_to_context)
            .to_numpy(int)
        )
        anchor_values = train_coords.loc[anchor_ids, dimensions].to_numpy(
            np.float32
        )
        thresholds, calibration = calibrate_anchor_projection(
            analysis_ids=anchor_ids,
            values=anchor_values,
            contexts=anchor_contexts,
            metric=metric,
            n_neighbors=k,
            params=projection_params,
        )
        threshold_output = thresholds.copy()
        threshold_output.insert(0, "root", root)
        threshold_output["full_reference_key"] = threshold_output["context"].map(
            context_to_key
        )
        threshold_output["metric"] = metric
        threshold_output["n_pcs"] = n_pcs
        threshold_output["K"] = k
        threshold_parts.append(threshold_output)
        calibration.insert(1, "root", root)
        calibration["true_full_reference_key"] = calibration[
            "true_context"
        ].map(context_to_key)
        calibration["predicted_full_reference_key"] = calibration[
            "predicted_context"
        ].map(context_to_key)
        calibration_parts.append(calibration)

        eligible_ids = base.loc[
            base["query_evaluable"], "analysis_id"
        ].astype(str).tolist()
        scored = pd.DataFrame({"analysis_id": pd.Series(dtype="string")})
        if eligible_ids:
            query_values = (
                root_queries.set_index("analysis_id")
                .loc[eligible_ids, query_dimensions]
                .to_numpy(np.float32)
            )
            candidates, scores = score_reference_knn(
                query_ids=eligible_ids,
                query_values=query_values,
                reference_ids=anchor_ids,
                reference_values=anchor_values,
                reference_labels=anchor_contexts,
                metric=metric,
                n_neighbors=k,
                top_n=projection_params.top_candidates,
                exclude_self=False,
            )
            scored = classify_reference_queries(scores, thresholds, root=root)
            scored["predicted_full_reference_key"] = scored[
                "predicted_context"
            ].map(context_to_key)
            accepted = scored["assignment_status"].eq("projected_assignment")
            scored["assigned_reference_id"] = pd.Series(
                pd.NA, index=scored.index, dtype="string"
            )
            scored.loc[accepted, "assigned_reference_id"] = scored.loc[
                accepted, "predicted_full_reference_key"
            ].astype(str)
            scored["projection_reason"] = np.select(
                [
                    scored["assignment_status"].eq("projected_assignment"),
                    scored["assignment_status"].eq("ambiguous"),
                ],
                ["passes_train_calibration", "weak_vote_or_margin"],
                default="outside_train_calibrated_distance_or_no_calibration",
            )
            candidates.insert(1, "root", root)
            candidates["candidate_full_reference_key"] = candidates[
                "context"
            ].map(context_to_key)
            candidate_parts.append(candidates)
        base = base.merge(scored, on="analysis_id", how="left", validate="one_to_one")
        not_evaluable = ~base["query_evaluable"]
        base.loc[not_evaluable, "assignment_status"] = "unresolved"
        base.loc[
            not_evaluable, "projection_reason"
        ] = "no_signal_in_train_feature_set"
        calibrated_lookup = threshold_output.set_index("full_reference_key")[
            "calibration_available"
        ]
        base["truth_reference_calibrated"] = (
            base["full_reference_key"]
            .map(calibrated_lookup)
            .fillna(False)
            .astype(bool)
        )
        prediction_parts.append(base)
        root_rows.append(
            {
                "root": root,
                "n_train_labels": int(len(root_labels)),
                "n_train_coordinates": int(len(available_labels)),
                "n_train_missing_coordinates": int((~available).sum()),
                "n_fixed_contexts": int(len(reference_keys)),
                "n_calibrated_contexts": int(
                    threshold_output["calibration_available"].sum()
                ),
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    expected = set(query_coords["analysis_id"].astype(str))
    observed = set(predictions["analysis_id"].astype(str))
    if predictions["analysis_id"].duplicated().any() or observed != expected:
        raise ValueError("Fixed-label hold-out prediction accounting failed")
    for column in ("predicted_context", "assigned_context"):
        if column not in predictions.columns:
            predictions[column] = pd.Series(
                pd.NA, index=predictions.index, dtype="Int64"
            )
    for column in ("assigned_reference_id", "predicted_full_reference_key"):
        if column not in predictions.columns:
            predictions[column] = pd.Series(
                pd.NA, index=predictions.index, dtype="string"
            )
    if "truth_reference_calibrated" not in predictions.columns:
        predictions["truth_reference_calibrated"] = False
    predictions["truth_reference_calibrated"] = predictions[
        "truth_reference_calibrated"
    ].fillna(False).astype(bool)
    predictions["raw_top1_correct"] = predictions[
        "predicted_full_reference_key"
    ].eq(predictions["full_reference_key"]).fillna(False)
    predictions["correct_assignment"] = (
        predictions["assignment_status"].eq("projected_assignment")
        & predictions["raw_top1_correct"]
    )
    predictions["incorrect_assignment"] = (
        predictions["assignment_status"].eq("projected_assignment")
        & ~predictions["raw_top1_correct"]
    )

    candidates = (
        pd.concat(candidate_parts, ignore_index=True, sort=False)
        if candidate_parts
        else pd.DataFrame()
    )
    if not candidates.empty:
        candidates = candidates.merge(
            query_coords[["analysis_id", "full_reference_key"]],
            on="analysis_id",
            how="left",
            validate="many_to_one",
        )
        candidates["candidate_maps_to_truth"] = candidates[
            "candidate_full_reference_key"
        ].eq(candidates["full_reference_key"]).fillna(False)

    metric_input = predictions.copy()
    metric_input["truth_context_recovered"] = metric_input[
        "truth_reference_calibrated"
    ]
    metrics = summarize_holdout_predictions(metric_input)
    renamed_metrics: dict[str, pd.DataFrame] = {}
    for name, frame in metrics.items():
        renamed_metrics[name] = frame.rename(
            columns={
                "truth_context_recovered_fraction": (
                    "truth_reference_calibrated_fraction"
                ),
                "recoverable_high_confidence_recall": (
                    "calibrated_truth_high_confidence_recall"
                ),
            }
        )
    return {
        "predictions": predictions,
        "candidates": candidates,
        "calibration_samples": (
            pd.concat(calibration_parts, ignore_index=True, sort=False)
            if calibration_parts
            else pd.DataFrame()
        ),
        "projection_thresholds": (
            pd.concat(threshold_parts, ignore_index=True, sort=False)
            if threshold_parts
            else pd.DataFrame()
        ),
        "reference_accounting": pd.DataFrame(root_rows),
        "candidate_retrieval": summarize_candidate_retrieval(candidates),
        "error_pairs": _holdout_error_pairs(predictions),
        **{f"metrics_{name}": frame for name, frame in renamed_metrics.items()},
    }


def write_fixed_label_evaluation(
    *,
    workspace: HoldoutWorkspace,
    tables: dict[str, pd.DataFrame],
    leakage_audit: pd.DataFrame,
    params: HoldoutValidationParams,
) -> dict[str, Path]:
    """Write the fixed-reference placement endpoint."""

    out_dir = workspace.validation_dir / "fixed_label_projection"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    parquet_names = {"predictions", "candidates", "calibration_samples"}
    for name, frame in tables.items():
        if name in parquet_names:
            path = out_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
        else:
            path = out_dir / f"{name}.tsv"
            frame.to_csv(path, sep="\t", index=False)
        outputs[name] = path
    leakage_path = out_dir / "leakage_audit.tsv"
    leakage_audit.to_csv(leakage_path, sep="\t", index=False)
    outputs["leakage_audit"] = leakage_path
    predictions = tables["predictions"]
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": _now(),
                "validation_scope": "fixed_label_train_only_projection",
                "independent_test_profiles": True,
                "reference_labels": "full_atlas_context_labels_on_train_anchors",
                "representation": "train_only_preprocessing_and_pca",
                "calibration": "train_only_leave_one_out",
                "neighbor_configuration": "full_atlas_selected_parameters",
                "params": {
                    "n_folds": int(params.n_folds),
                    "test_fold": int(params.test_fold),
                    "random_seed": int(params.random_seed),
                },
                "counts": {
                    "test_profiles": int(len(predictions)),
                    "projected_assignments": int(
                        predictions["assignment_status"]
                        .eq("projected_assignment")
                        .sum()
                    ),
                    "ambiguous": int(
                        predictions["assignment_status"].eq("ambiguous").sum()
                    ),
                    "unresolved": int(
                        predictions["assignment_status"].eq("unresolved").sum()
                    ),
                    "calibrated_truth_profiles": int(
                        predictions["truth_reference_calibrated"].sum()
                    ),
                },
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    outputs["manifest"] = manifest_path
    return outputs
