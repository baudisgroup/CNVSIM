from __future__ import annotations

"""Build a compact subtype diagnostic summary from a frozen CNVSIM release."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


ANNOTATION_FIELDS = {
    "ncit_diagnosis": (
        "histological_diagnosis_id",
        "histological_diagnosis_label",
    ),
    "icdo_morphology": ("icdo_morphology_id", "icdo_morphology_label"),
    "icdo_topography": ("icdo_topography_id", "icdo_topography_label"),
}

DEFINITION_STATUS_BY_LAYER = {
    "core": "anchor_member",
    "conditional": "conditional_definition_member",
}

SUPPLEMENTARY_COLUMNS = [
    "reference_key",
    "root",
    "reference_layer",
    "n_definition_samples",
    "n_ncit_diagnosis_annotated",
    "ncit_diagnosis_annotation_coverage",
    "dominant_ncit_diagnosis_id",
    "dominant_ncit_diagnosis_label",
    "dominant_ncit_diagnosis_n",
    "dominant_ncit_diagnosis_fraction_of_definition_samples",
    "ncit_diagnosis_top_labels",
    "dominant_icdo_morphology_id",
    "dominant_icdo_morphology_label",
    "dominant_icdo_morphology_fraction_of_definition_samples",
    "dominant_icdo_topography_id",
    "dominant_icdo_topography_label",
    "dominant_icdo_topography_fraction_of_definition_samples",
    "n_track1_signatures",
    "track1_signature_summary",
    "n_track2_signatures",
    "track2_representative_events",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize NCIt and ICD-O annotations among the frozen definition "
            "samples of all released CNVSIM subtypes."
        )
    )
    parser.add_argument(
        "--release-dir",
        required=True,
        help="CNVSIM public data-release directory.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--top-n-labels",
        type=int,
        default=3,
        help="Number of labels included in compact composition strings (default: 3).",
    )
    parser.add_argument(
        "--top-n-track2",
        type=int,
        default=5,
        help="Number of Track 2 events included in compact summaries (default: 5).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_inputs(release_dir: Path) -> tuple[pd.DataFrame, ...]:
    subtype_path = _require_file(release_dir / "atlas" / "subtypes.tsv")
    assignment_path = _require_file(
        release_dir / "atlas" / "analysis_assignments.parquet"
    )
    signature_path = _require_file(release_dir / "atlas" / "signature_events.tsv")
    metadata_path = _require_file(
        release_dir
        / "source_data"
        / "source_file"
        / "metadata"
        / "tumour_analysis_metadata.parquet"
    )

    subtypes = pd.read_csv(subtype_path, sep="\t")
    assignments = pd.read_parquet(assignment_path)
    signatures = pd.read_csv(signature_path, sep="\t")
    metadata = pd.read_parquet(metadata_path)

    _require_columns(
        subtypes,
        [
            "reference_key",
            "root",
            "reference_layer",
            "n_definition_samples",
            "n_track1_signatures",
            "n_track2_signatures",
        ],
        label="atlas/subtypes.tsv",
    )
    _require_columns(
        assignments,
        ["analysis_id", "root", "pipeline_status", "assigned_reference_key"],
        label="atlas/analysis_assignments.parquet",
    )
    _require_columns(
        signatures,
        [
            "reference_key",
            "track",
            "feature",
            "event",
            "amplitude",
            "effect_size",
            "adjusted_effect_size",
        ],
        label="atlas/signature_events.tsv",
    )
    metadata_columns = ["analysis_id"]
    for id_column, label_column in ANNOTATION_FIELDS.values():
        metadata_columns.extend([id_column, label_column])
    _require_columns(
        metadata,
        metadata_columns,
        label="tumour_analysis_metadata.parquet",
    )

    if subtypes["reference_key"].duplicated().any():
        raise ValueError("atlas/subtypes.tsv contains duplicate reference keys")
    if assignments["analysis_id"].duplicated().any():
        raise ValueError("analysis assignments contain duplicate analysis IDs")
    if metadata["analysis_id"].duplicated().any():
        raise ValueError("tumour metadata contain duplicate analysis IDs")

    return subtypes, assignments, signatures, metadata


def _build_definition_members(
    subtypes: pd.DataFrame,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    statuses = set(DEFINITION_STATUS_BY_LAYER.values())
    members = assignments.loc[
        assignments["pipeline_status"].isin(statuses),
        ["analysis_id", "pipeline_status", "assigned_reference_key"],
    ].copy()
    members = members.rename(columns={"assigned_reference_key": "reference_key"})
    if members["reference_key"].isna().any():
        raise ValueError("Definition members include missing reference keys")

    catalog = subtypes[
        ["reference_key", "root", "reference_layer", "n_definition_samples"]
    ].copy()
    catalog["subtype_order"] = range(len(catalog))
    members = members.merge(
        catalog,
        on="reference_key",
        how="left",
        validate="many_to_one",
    )
    if members["reference_layer"].isna().any():
        unknown = sorted(members.loc[members["reference_layer"].isna(), "reference_key"].unique())
        raise ValueError(f"Definition members reference unknown subtypes: {unknown}")

    expected_status = members["reference_layer"].map(DEFINITION_STATUS_BY_LAYER)
    invalid = members["pipeline_status"].ne(expected_status)
    if invalid.any():
        examples = members.loc[
            invalid,
            ["analysis_id", "reference_key", "reference_layer", "pipeline_status"],
        ].head(10)
        raise ValueError(
            "Definition status does not match reference layer:\n"
            + examples.to_string(index=False)
        )

    observed = members.groupby("reference_key").size().rename("observed_members")
    accounting = catalog.merge(observed, on="reference_key", how="left")
    accounting["observed_members"] = accounting["observed_members"].fillna(0).astype(int)
    mismatch = accounting["n_definition_samples"].ne(accounting["observed_members"])
    if mismatch.any():
        raise ValueError(
            "Definition-member counts do not match atlas/subtypes.tsv:\n"
            + accounting.loc[
                mismatch,
                ["reference_key", "n_definition_samples", "observed_members"],
            ].to_string(index=False)
        )

    metadata_columns = ["analysis_id"]
    for id_column, label_column in ANNOTATION_FIELDS.values():
        metadata_columns.extend([id_column, label_column])
    members = members.merge(
        metadata[metadata_columns],
        on="analysis_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if members["_merge"].ne("both").any():
        missing = members.loc[members["_merge"].ne("both"), "analysis_id"].head(10)
        raise ValueError(f"Definition members missing metadata: {missing.tolist()}")
    return members.drop(columns="_merge")


def _build_composition(members: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    definition_counts = members.groupby("reference_key").size().rename("n_definition_samples")

    for annotation_type, (id_column, label_column) in ANNOTATION_FIELDS.items():
        work = members[
            [
                "subtype_order",
                "reference_key",
                "root",
                "reference_layer",
                "analysis_id",
                id_column,
                label_column,
            ]
        ].copy()
        work["annotation_id"] = _clean_text(work[id_column])
        work["annotation_label"] = _clean_text(work[label_column])
        work = work[
            work["annotation_id"].ne("") | work["annotation_label"].ne("")
        ].copy()
        if work.empty:
            continue

        grouped = (
            work.groupby(
                [
                    "subtype_order",
                    "reference_key",
                    "root",
                    "reference_layer",
                    "annotation_id",
                    "annotation_label",
                ],
                dropna=False,
            )
            .size()
            .rename("n_samples")
            .reset_index()
        )
        annotated_counts = (
            work.groupby("reference_key")["analysis_id"]
            .nunique()
            .rename("n_annotated_samples")
        )
        grouped = grouped.merge(definition_counts, on="reference_key", how="left")
        grouped = grouped.merge(annotated_counts, on="reference_key", how="left")
        grouped["fraction_of_definition_samples"] = (
            grouped["n_samples"] / grouped["n_definition_samples"]
        )
        grouped["fraction_of_annotated_samples"] = (
            grouped["n_samples"] / grouped["n_annotated_samples"]
        )
        grouped["annotation_coverage"] = (
            grouped["n_annotated_samples"] / grouped["n_definition_samples"]
        )
        grouped["annotation_type"] = annotation_type
        grouped = grouped.sort_values(
            [
                "subtype_order",
                "n_samples",
                "annotation_label",
                "annotation_id",
            ],
            ascending=[True, False, True, True],
            kind="stable",
        )
        grouped["rank"] = grouped.groupby("reference_key").cumcount() + 1
        rows.append(grouped)

    if not rows:
        raise ValueError("No diagnostic annotations were available")
    composition = pd.concat(rows, ignore_index=True)
    return composition[
        [
            "reference_key",
            "root",
            "reference_layer",
            "annotation_type",
            "annotation_id",
            "annotation_label",
            "n_samples",
            "n_annotated_samples",
            "n_definition_samples",
            "annotation_coverage",
            "fraction_of_definition_samples",
            "fraction_of_annotated_samples",
            "rank",
        ]
    ]


def _format_composition(group: pd.DataFrame, top_n: int) -> str:
    parts: list[str] = []
    for row in group.sort_values("rank").head(top_n).itertuples(index=False):
        label = row.annotation_label or row.annotation_id
        identifier = (
            f" [{row.annotation_id}]"
            if row.annotation_id and row.annotation_id != label
            else ""
        )
        parts.append(
            f"{label}{identifier}: n={int(row.n_samples)} "
            f"({100 * row.fraction_of_annotated_samples:.1f}% of annotated)"
        )
    return "; ".join(parts)


def _format_signature_rows(group: pd.DataFrame, track: str, top_n_track2: int) -> str:
    if group.empty:
        return "None released"
    work = group.copy()
    adjusted = pd.to_numeric(work["adjusted_effect_size"], errors="coerce")
    unadjusted = pd.to_numeric(work["effect_size"], errors="coerce")
    work["reporting_score"] = adjusted.fillna(unadjusted)
    work = work.sort_values(
        ["reporting_score", "feature", "event"],
        ascending=[False, True, True],
        kind="stable",
    )
    if track == "track2":
        work = work.head(top_n_track2)

    labels: list[str] = []
    for row in work.itertuples(index=False):
        if track == "track1":
            labels.append(f"{row.feature} {row.event}")
        else:
            amplitude = str(row.amplitude).strip()
            event = str(row.event).strip()
            state = " ".join(part for part in [amplitude, event] if part)
            labels.append(f"{row.feature} {state}".strip())
    return "; ".join(labels)


def _build_signature_summary(
    subtypes: pd.DataFrame,
    signatures: pd.DataFrame,
    *,
    top_n_track2: int,
) -> pd.DataFrame:
    observed = (
        signatures.groupby(["reference_key", "track"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(columns=None)
    )
    for track in ["track1", "track2"]:
        if track not in observed:
            observed[track] = 0
    accounting = subtypes[
        ["reference_key", "n_track1_signatures", "n_track2_signatures"]
    ].merge(observed[["track1", "track2"]], on="reference_key", how="left")
    accounting[["track1", "track2"]] = accounting[["track1", "track2"]].fillna(0).astype(int)
    mismatch = accounting[
        accounting["n_track1_signatures"].ne(accounting["track1"])
        | accounting["n_track2_signatures"].ne(accounting["track2"])
    ]
    if not mismatch.empty:
        raise ValueError(
            "Signature counts do not match atlas/subtypes.tsv:\n"
            + mismatch.to_string(index=False)
        )

    rows: list[dict[str, object]] = []
    grouped = {key: frame for key, frame in signatures.groupby("reference_key")}
    for row in subtypes.itertuples(index=False):
        group = grouped.get(row.reference_key, signatures.iloc[0:0])
        track1 = group[group["track"].eq("track1")]
        track2 = group[group["track"].eq("track2")]
        rows.append(
            {
                "reference_key": row.reference_key,
                "track1_signature_summary": _format_signature_rows(
                    track1, "track1", top_n_track2
                ),
                "track2_representative_events": _format_signature_rows(
                    track2, "track2", top_n_track2
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_subtype_summary(
    subtypes: pd.DataFrame,
    composition: pd.DataFrame,
    signature_summary: pd.DataFrame,
    *,
    top_n_labels: int,
) -> pd.DataFrame:
    output_columns = [
        "reference_key",
        "root",
        "reference_layer",
        "n_definition_samples",
        "n_track1_signatures",
        "n_track2_signatures",
    ]
    summary = subtypes[output_columns].copy()

    for annotation_type in ANNOTATION_FIELDS:
        subset = composition[composition["annotation_type"].eq(annotation_type)]
        top = subset[subset["rank"].eq(1)].copy()
        prefix = f"dominant_{annotation_type}"
        top = top.rename(
            columns={
                "annotation_id": f"{prefix}_id",
                "annotation_label": f"{prefix}_label",
                "n_samples": f"{prefix}_n",
                "fraction_of_definition_samples": (
                    f"{prefix}_fraction_of_definition_samples"
                ),
                "fraction_of_annotated_samples": (
                    f"{prefix}_fraction_of_annotated_samples"
                ),
            }
        )
        top_columns = [
            "reference_key",
            "n_annotated_samples",
            "annotation_coverage",
            f"{prefix}_id",
            f"{prefix}_label",
            f"{prefix}_n",
            f"{prefix}_fraction_of_definition_samples",
            f"{prefix}_fraction_of_annotated_samples",
        ]
        top = top[top_columns].rename(
            columns={
                "n_annotated_samples": f"n_{annotation_type}_annotated",
                "annotation_coverage": f"{annotation_type}_annotation_coverage",
            }
        )
        compact = pd.DataFrame(
            [
                {
                    "reference_key": reference_key,
                    f"{annotation_type}_top_labels": _format_composition(
                        frame, top_n_labels
                    ),
                }
                for reference_key, frame in subset.groupby(
                    "reference_key", sort=False
                )
            ]
        )
        summary = summary.merge(top, on="reference_key", how="left", validate="one_to_one")
        summary = summary.merge(
            compact,
            on="reference_key",
            how="left",
            validate="one_to_one",
        )

    summary = summary.merge(
        signature_summary,
        on="reference_key",
        how="left",
        validate="one_to_one",
    )
    text_columns = summary.select_dtypes(include="object").columns
    summary[text_columns] = summary[text_columns].fillna("")
    return summary


def _prepare_output_directory(out_dir: Path, *, overwrite: bool) -> None:
    outputs = [out_dir / "subtype_diagnostic_summary.tsv"]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    if args.top_n_labels < 1 or args.top_n_track2 < 1:
        raise ValueError("--top-n-labels and --top-n-track2 must be positive")

    release_dir = Path(args.release_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_output_directory(out_dir, overwrite=args.overwrite)

    subtypes, assignments, signatures, metadata = _load_inputs(release_dir)
    members = _build_definition_members(subtypes, assignments, metadata)
    composition = _build_composition(members)
    signature_summary = _build_signature_summary(
        subtypes,
        signatures,
        top_n_track2=args.top_n_track2,
    )
    summary = _build_subtype_summary(
        subtypes,
        composition,
        signature_summary,
        top_n_labels=args.top_n_labels,
    )

    summary_path = out_dir / "subtype_diagnostic_summary.tsv"
    summary[SUPPLEMENTARY_COLUMNS].to_csv(summary_path, sep="\t", index=False)

    print("[CNVSIM] subtype diagnostic summary complete")
    print(f"  subtypes: {len(summary)}")
    print(f"  definition samples: {len(members)}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
