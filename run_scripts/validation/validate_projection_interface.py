from __future__ import annotations

"""Smoke-test the public projection interface with frozen atlas samples."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pyarrow.dataset as ds

from cnvsim.projection import STATE_TO_EVENT, project_samples


def _normalized_state(value: object) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


def _resolve_segments(atlas_dir: Path, declared: str | None) -> Path:
    if declared:
        path = Path(declared).expanduser().resolve()
    else:
        path = atlas_dir / "source_derived" / "tumour_autosomal_segments.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing projection smoke-test segments: {path}. Pass --segments "
            "to declare another whole-genome segment table."
        )
    return path


def _head_per_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    n: int,
) -> pd.DataFrame:
    return (
        frame.sort_values([group_column, "analysis_id"])
        .groupby(group_column, sort=True, as_index=False, group_keys=False)
        .head(int(n))
        .copy()
    )


def _select_queries(
    assignments: pd.DataFrame,
    references: pd.DataFrame,
    *,
    samples_per_context: int,
    route_samples: int,
    conditional_members: pd.DataFrame | None = None,
    conditional_contexts: pd.DataFrame | None = None,
    samples_per_conditional_context: int = 5,
) -> pd.DataFrame:
    required = {
        "analysis_id",
        "root",
        "pipeline_status",
        "assigned_reference_key",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise KeyError(f"Release assignments are missing columns: {missing}")
    assignments = assignments.copy()
    assignments["analysis_id"] = assignments["analysis_id"].astype(str)
    assignments["root"] = assignments["root"].astype(str)

    anchors = assignments[
        assignments["pipeline_status"].astype(str).eq("anchor_member")
    ].copy()
    anchors = _head_per_group(
        anchors,
        group_column="assigned_reference_key",
        n=samples_per_context,
    )
    anchors["smoke_role"] = "anchor"
    anchors["expected_status"] = "projected_assignment"

    expected_subtypes = set(
        references.loc[
            references["reference_level"].astype(str).eq("subtype"),
            "reference_key",
        ].astype(str)
    )
    observed_subtypes = set(anchors["assigned_reference_key"].astype(str))
    if observed_subtypes != expected_subtypes:
        missing_references = sorted(expected_subtypes - observed_subtypes)
        raise ValueError(
            "Smoke-test selection does not cover every subtype reference: "
            f"{missing_references}"
        )

    route_parts: list[pd.DataFrame] = []
    route_specs = (
        ("root_only_reference", "root_only", "root_only_reference", "root"),
        ("out_of_scope_root", "out_of_scope", "out_of_scope_root", "root"),
        (
            "no_autosomal_gene_cnv_signal",
            "no_signal",
            "no_autosomal_gene_cnv_signal",
            "pipeline_status",
        ),
    )
    for pipeline_status, role, expected_status, group_column in route_specs:
        pool = assignments[
            assignments["pipeline_status"].astype(str).eq(pipeline_status)
        ].copy()
        if pool.empty:
            continue
        selected = _head_per_group(
            pool,
            group_column=group_column,
            n=route_samples,
        )
        selected["smoke_role"] = role
        selected["expected_status"] = expected_status
        route_parts.append(selected)

    conditional_parts: list[pd.DataFrame] = []
    if conditional_members is not None or conditional_contexts is not None:
        if conditional_members is None or conditional_contexts is None:
            raise ValueError("Conditional smoke-test inputs must be supplied together")
        required_members = {
            "analysis_id",
            "root",
            "conditional_reference_key",
        }
        missing_members = sorted(required_members - set(conditional_members.columns))
        if missing_members:
            raise KeyError(
                "Conditional definition members are missing columns: "
                f"{missing_members}"
            )
        key_column = (
            "conditional_reference_key"
            if "conditional_reference_key" in conditional_contexts.columns
            else "reference_key"
        )
        if key_column not in conditional_contexts.columns:
            raise KeyError("Conditional contexts lack a reference-key column")
        eligible = conditional_members.copy()
        eligible["analysis_id"] = eligible["analysis_id"].astype(str)
        eligible["root"] = eligible["root"].astype(str)
        eligible["conditional_reference_key"] = eligible[
            "conditional_reference_key"
        ].astype(str)
        eligible = eligible.merge(
            assignments[["analysis_id", "pipeline_status"]],
            on="analysis_id",
            how="left",
            validate="one_to_one",
        )
        eligible = eligible[
            eligible["pipeline_status"].isin(["ambiguous", "unresolved"])
        ].copy()
        conditional = _head_per_group(
            eligible,
            group_column="conditional_reference_key",
            n=samples_per_conditional_context,
        )
        expected_conditional = set(conditional_contexts[key_column].astype(str))
        observed_conditional = set(
            conditional["conditional_reference_key"].astype(str)
        )
        if observed_conditional != expected_conditional:
            missing_references = sorted(
                expected_conditional - observed_conditional
            )
            raise ValueError(
                "Smoke-test selection lacks primary-uncertain samples for "
                f"conditional contexts: {missing_references}"
            )
        conditional["smoke_role"] = "conditional"
        conditional["expected_status"] = "projected_assignment"
        conditional = conditional.rename(
            columns={"conditional_reference_key": "assigned_reference_key"}
        )
        conditional_parts.append(conditional)

    selected = pd.concat(
        [anchors, *conditional_parts, *route_parts],
        ignore_index=True,
        sort=False,
    )
    if selected["analysis_id"].duplicated().any():
        raise ValueError("Smoke-test query selection contains duplicate analyses")
    return selected[
        [
            "analysis_id",
            "root",
            "smoke_role",
            "expected_status",
            "assigned_reference_key",
        ]
    ].rename(columns={"assigned_reference_key": "expected_reference_key"})


def _read_selected_segments(path: Path, analysis_ids: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    names = set(dataset.schema.names)
    id_column = "analysis_id" if "analysis_id" in names else "sample_id"
    required = {id_column, "chromosome", "start", "end"}
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"Segment source is missing columns: {missing}")
    state_columns = [
        column
        for column in ("cnv_state", "state_label", "copy_change", "event_group")
        if column in names
    ]
    if not state_columns:
        raise KeyError("Segment source has no CNV state column")
    columns = [*required, *state_columns]
    table = dataset.to_table(
        columns=columns,
        filter=ds.field(id_column).isin(sorted(analysis_ids)),
    )
    raw = table.to_pandas()
    if id_column != "analysis_id":
        raw = raw.rename(columns={id_column: "analysis_id"})

    def choose_state(row: pd.Series) -> str | None:
        for column in state_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            state = _normalized_state(value)
            if state in STATE_TO_EVENT:
                return state
        return None

    raw["cnv_state"] = raw.apply(choose_state, axis=1)
    raw = raw[raw["cnv_state"].notna()].copy()
    raw["analysis_id"] = raw["analysis_id"].astype(str)
    return raw[
        ["analysis_id", "chromosome", "start", "end", "cnv_state"]
    ].rename(columns={"analysis_id": "sample_id"})


def _summarize(
    comparison: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    anchors = comparison[comparison["smoke_role"].eq("anchor")].copy()
    top = candidates[candidates["candidate_rank"].eq(1)].copy()
    if "projection_layer" in top.columns:
        top = top[top["projection_layer"].astype(str).eq("primary")]
    top = top[
        ["sample_id", "candidate_reference_key"]
    ].copy()
    if top["sample_id"].duplicated().any():
        raise ValueError("Primary projection has duplicate rank-1 candidates")
    anchors = anchors.merge(top, on="sample_id", how="left", validate="one_to_one")
    conditional_rows = comparison[comparison["smoke_role"].eq("conditional")].copy()
    conditional_top = candidates[candidates["candidate_rank"].eq(1)].copy()
    if "projection_layer" in conditional_top.columns:
        conditional_top = conditional_top[
            conditional_top["projection_layer"].astype(str).eq("conditional")
        ]
    conditional_top = conditional_top[
        ["sample_id", "candidate_reference_key"]
    ].rename(columns={"candidate_reference_key": "conditional_top_reference_key"})
    if conditional_top["sample_id"].duplicated().any():
        raise ValueError("Conditional projection has duplicate rank-1 candidates")
    conditional_rows = conditional_rows.merge(
        conditional_top,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    status_column = (
        "primary_assignment_status"
        if "primary_assignment_status" in anchors.columns
        else "assignment_status"
    )
    reference_column = (
        "primary_assigned_reference_key"
        if "primary_assigned_reference_key" in anchors.columns
        else "assigned_reference_key"
    )
    projected = anchors[status_column].eq("projected_assignment")
    conditional_projected = (
        conditional_rows["assignment_status"].eq("projected_assignment")
        & conditional_rows["assignment_layer"].eq("conditional")
        if not conditional_rows.empty
        else pd.Series(dtype=bool)
    )
    route_rows = comparison[
        ~comparison["smoke_role"].isin(["anchor", "conditional"])
    ]
    rows = [
        {
            "metric": "selected_samples",
            "value": float(len(comparison)),
        },
        {
            "metric": "subtype_contexts_covered",
            "value": float(anchors["expected_reference_key"].nunique()),
        },
        {
            "metric": "anchor_raw_top1_accuracy",
            "value": float(
                anchors["candidate_reference_key"]
                .eq(anchors["expected_reference_key"])
                .mean()
            ),
        },
        {
            "metric": "anchor_assignment_coverage",
            "value": float(projected.mean()),
        },
        {
            "metric": "anchor_conditional_accuracy",
            "value": float(
                anchors.loc[projected, reference_column]
                .eq(anchors.loc[projected, "expected_reference_key"])
                .mean()
            )
            if projected.any()
            else float("nan"),
        },
        {
            "metric": "conditional_contexts_covered",
            "value": float(conditional_rows["expected_reference_key"].nunique()),
        },
        {
            "metric": "conditional_raw_top1_accuracy",
            "value": float(
                conditional_rows["conditional_top_reference_key"]
                .eq(conditional_rows["expected_reference_key"])
                .mean()
            )
            if not conditional_rows.empty
            else float("nan"),
        },
        {
            "metric": "conditional_assignment_coverage",
            "value": float(conditional_projected.mean())
            if not conditional_rows.empty
            else float("nan"),
        },
        {
            "metric": "conditional_assignment_accuracy",
            "value": float(
                conditional_rows.loc[
                    conditional_projected, "assigned_reference_key"
                ]
                .eq(
                    conditional_rows.loc[
                        conditional_projected, "expected_reference_key"
                    ]
                )
                .mean()
            )
            if conditional_projected.any()
            else float("nan"),
        },
        {
            "metric": "route_status_accuracy",
            "value": float(
                route_rows["assignment_status"]
                .eq(route_rows["expected_status"])
                .mean()
            )
            if not route_rows.empty
            else float("nan"),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the public projection interface on frozen atlas samples."
    )
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument("--segments")
    parser.add_argument("--bundle-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--samples-per-context", type=int, default=5)
    parser.add_argument("--samples-per-conditional-context", type=int, default=5)
    parser.add_argument("--route-samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir).expanduser().resolve()
    bundle_dir = (
        Path(args.bundle_dir).expanduser().resolve()
        if args.bundle_dir
        else atlas_dir / "projection_bundle_candidate"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else atlas_dir / "work" / "validation" / "projection_interface_smoke_v1"
    )
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Smoke-test output exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    release_tables = atlas_dir / "release_candidate" / "tables"
    assignments = pd.read_parquet(release_tables / "analysis_assignments.parquet")
    references = pd.read_csv(
        release_tables / "reference_contexts.tsv", sep="\t"
    )
    conditional_dir = bundle_dir / "conditional"
    conditional_contexts_path = conditional_dir / "conditional_contexts.tsv"
    conditional_members_path = (
        conditional_dir / "conditional_definition_members.parquet"
    )
    conditional_contexts = (
        pd.read_csv(conditional_contexts_path, sep="\t")
        if conditional_contexts_path.exists()
        else None
    )
    conditional_members = (
        pd.read_parquet(conditional_members_path)
        if conditional_members_path.exists()
        else None
    )
    selected = _select_queries(
        assignments,
        references,
        samples_per_context=int(args.samples_per_context),
        route_samples=int(args.route_samples),
        conditional_members=conditional_members,
        conditional_contexts=conditional_contexts,
        samples_per_conditional_context=int(
            args.samples_per_conditional_context
        ),
    )
    segments_path = _resolve_segments(atlas_dir, args.segments)
    segments = _read_selected_segments(
        segments_path,
        set(selected["analysis_id"].astype(str)),
    )

    samples_path = out_dir / "samples.tsv"
    segments_input_path = out_dir / "segments.tsv"
    expected_path = out_dir / "expected_assignments.tsv"
    selected.rename(columns={"analysis_id": "sample_id"})[
        ["sample_id", "root"]
    ].assign(genome_build="GRCh38").to_csv(samples_path, sep="\t", index=False)
    segments.to_csv(segments_input_path, sep="\t", index=False)
    selected.rename(columns={"analysis_id": "sample_id"}).to_csv(
        expected_path, sep="\t", index=False
    )

    projection_dir = out_dir / "projection"
    outputs = project_samples(
        bundle_dir=bundle_dir,
        samples_path=samples_path,
        segments_path=segments_input_path,
        out_dir=projection_dir,
    )
    observed = pd.read_csv(outputs["assignments"], sep="\t")
    candidates = pd.read_csv(outputs["candidates"], sep="\t")
    comparison = selected.rename(columns={"analysis_id": "sample_id"}).merge(
        observed,
        on=["sample_id", "root"],
        how="left",
        validate="one_to_one",
    )
    summary = _summarize(comparison, candidates)
    comparison_path = out_dir / "projection_comparison.tsv"
    summary_path = out_dir / "summary.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cnvsim_projection_interface_smoke_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "segments": str(segments_path),
                "bundle": str(bundle_dir),
                "counts": {
                    "selected_samples": int(len(selected)),
                    "input_segments": int(len(segments)),
                    "subtype_contexts": int(
                        selected.loc[
                            selected["smoke_role"].eq("anchor"),
                            "expected_reference_key",
                        ].nunique()
                    ),
                    "conditional_contexts": int(
                        selected.loc[
                            selected["smoke_role"].eq("conditional"),
                            "expected_reference_key",
                        ].nunique()
                    ),
                },
                "summary": {
                    str(row.metric): float(row.value)
                    for row in summary.itertuples(index=False)
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("[CNVSIM] projection interface smoke test complete")
    print(summary.to_string(index=False))
    print(f"  comparison: {comparison_path}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
