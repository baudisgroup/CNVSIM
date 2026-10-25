from __future__ import annotations

"""Small bycon client shared by the retrieval example and its smoke test."""

import json
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_BASE_URL = "https://progenetix.org"
CNV_OPERATION_ID = "EDAM:operation_3961"
RESOURCE_DATASET = {
    "tumour": None,
    "cell-line": "cellz",
}
SELECTED_VALUES = {"1", "true", "t", "yes", "y", "x", "selected"}

COPY_CHANGE_TO_STATE = {
    "gain": "gain",
    "copy number gain": "gain",
    "low-level gain": "gain",
    "high-level gain": "high_gain",
    "focal genome amplification": "high_gain",
    "loss": "loss",
    "copy number loss": "loss",
    "low-level loss": "loss",
    "high-level loss": "high_loss",
    "complete genomic deletion": "high_loss",
}
VARIANT_STATE_ID_TO_STATE = {
    "EFO:0030070": "gain",
    "EFO:0030071": "gain",
    "EFO:0030072": "high_gain",
    "EFO:0030073": "high_gain",
    "EFO:0030067": "loss",
    "EFO:0030068": "loss",
    "EFO:0020073": "high_loss",
    "EFO:0030069": "high_loss",
}
GRCH38_AUTOSOME_REFSEQ = {
    "1": "refseq:NC_000001.11",
    "2": "refseq:NC_000002.12",
    "3": "refseq:NC_000003.12",
    "4": "refseq:NC_000004.12",
    "5": "refseq:NC_000005.10",
    "6": "refseq:NC_000006.12",
    "7": "refseq:NC_000007.14",
    "8": "refseq:NC_000008.11",
    "9": "refseq:NC_000009.12",
    "10": "refseq:NC_000010.11",
    "11": "refseq:NC_000011.10",
    "12": "refseq:NC_000012.12",
    "13": "refseq:NC_000013.11",
    "14": "refseq:NC_000014.9",
    "15": "refseq:NC_000015.10",
    "16": "refseq:NC_000016.10",
    "17": "refseq:NC_000017.11",
    "18": "refseq:NC_000018.10",
    "19": "refseq:NC_000019.10",
    "20": "refseq:NC_000020.11",
    "21": "refseq:NC_000021.9",
    "22": "refseq:NC_000022.11",
}


def request_json(url: str, attempts: int = 3) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CNVSIM/1.0 bycon retrieval example",
        },
    )
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == attempts:
                raise RuntimeError(f"bycon request failed: {url}") from error
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def beacon_results(payload: dict[str, Any], url: str) -> list[dict[str, Any]]:
    if payload.get("error"):
        raise RuntimeError(f"bycon returned an error for {url}: {payload['error']}")
    response = payload.get("response", {}) or {}
    results: list[dict[str, Any]] = []
    for result_set in response.get("resultSets", []) or []:
        results.extend(result_set.get("results", []) or [])
    results.extend(response.get("results", []) or [])
    return results


def beacon_query(
    path: str,
    *,
    resource: str,
    filters: Iterable[str] = (),
    include_descendants: bool = False,
    filter_logic: str = "AND",
    limit: int = 0,
    skip: int = 0,
    extra: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    if resource not in RESOURCE_DATASET:
        raise ValueError(f"Unsupported resource: {resource}")
    params: dict[str, object] = {"limit": int(limit)}
    cleaned_filters = [str(value).strip() for value in filters if str(value).strip()]
    if cleaned_filters:
        params["filters"] = ",".join(cleaned_filters)
        params["filterLogic"] = str(filter_logic).upper()
        params["includeDescendantTerms"] = str(bool(include_descendants)).lower()
    if skip:
        params["skip"] = int(skip)
    dataset = RESOURCE_DATASET[resource]
    if dataset:
        params["datasetIds"] = dataset
    if extra:
        params.update(extra)
    url = f"{API_BASE_URL}/beacon/{path.lstrip('/')}?{urlencode(params)}"
    return beacon_results(request_json(url), url)


def is_cnv_analysis(analysis: dict[str, Any]) -> bool:
    operation = analysis.get("operation", {}) or {}
    return str(operation.get("id", "")) == CNV_OPERATION_ID


def exact_biosample(resource: str, biosample_id: str) -> dict[str, Any] | None:
    rows = beacon_query(
        "biosamples",
        resource=resource,
        limit=2,
        extra={"biosampleIds": biosample_id},
    )
    matches = [row for row in rows if str(row.get("id", "")) == biosample_id]
    if len(matches) > 1:
        raise ValueError(f"Duplicate biosample returned by bycon: {biosample_id}")
    return matches[0] if matches else None


def analysis_variants(resource: str, analysis_id: str) -> list[dict[str, Any]]:
    """Return the variants of one analysis from the analysis-scoped endpoint.

    The endpoint scopes results to the requested analysis, so variants are
    attributed without relying on ``caseLevelData.analysisId``, which current
    bycon responses may omit.
    """

    return beacon_query(
        f"analyses/{analysis_id}/g_variants",
        resource=resource,
        limit=0,
    )


def variant_analysis_ids(variant: dict[str, Any]) -> set[str]:
    return {
        str(case.get("analysisId") or case.get("analysis_id") or "").strip()
        for case in variant.get("caseLevelData", []) or []
    } - {""}


def location_sequence_id(location: dict[str, Any]) -> str:
    return str(location.get("sequenceId") or location.get("sequence_id") or "")


def cnv_state(variation: dict[str, Any]) -> str | None:
    variant_state = variation.get("variantState", {}) or {}
    state_id = str(variant_state.get("id", "")).strip()
    if state_id in VARIANT_STATE_ID_TO_STATE:
        return VARIANT_STATE_ID_TO_STATE[state_id]
    label = str(variant_state.get("label", "")).strip().lower()
    if label in COPY_CHANGE_TO_STATE:
        return COPY_CHANGE_TO_STATE[label]
    copy_change = str(variation.get("copyChange", "")).strip().lower()
    return COPY_CHANGE_TO_STATE.get(copy_change)


def selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "selected" not in frame.columns:
        raise KeyError("Selection table must contain a 'selected' column")
    mask = frame["selected"].fillna("").astype(str).str.strip().str.lower()
    selected = frame.loc[mask.isin(SELECTED_VALUES)].copy()
    if selected.empty:
        raise ValueError("No analyses are marked as selected")
    required = {"resource", "biosample_id", "analysis_id", "icdo_topography_id"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise KeyError(f"Selection table is missing columns: {missing}")
    selected["analysis_id"] = selected["analysis_id"].astype(str)
    if selected["analysis_id"].duplicated().any():
        duplicated = selected.loc[
            selected["analysis_id"].duplicated(False), "analysis_id"
        ].tolist()
        raise ValueError(f"Selection contains duplicate analyses: {duplicated[:5]}")
    invalid_resources = sorted(set(selected["resource"]) - set(RESOURCE_DATASET))
    if invalid_resources:
        raise ValueError(f"Selection contains unsupported resources: {invalid_resources}")
    return selected


def load_release_overlap(release_dir: Path | None) -> tuple[pd.DataFrame, bool]:
    if release_dir is None:
        return pd.DataFrame(), False
    path = Path(release_dir).expanduser().resolve() / "atlas" / "analysis_assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["analysis_id", "biosample_id", "pipeline_status", "assigned_reference_key"]
    frame = pd.read_parquet(path, columns=columns)
    frame["analysis_id"] = frame["analysis_id"].astype(str)
    frame["biosample_id"] = frame["biosample_id"].astype(str)
    if frame["analysis_id"].duplicated().any():
        raise ValueError(f"Public release contains duplicate analysis IDs: {path}")
    return frame, True


def overlap_annotation(
    analysis_id: str,
    biosample_id: str,
    release_assignments: pd.DataFrame,
    checked: bool,
) -> tuple[str, str]:
    if not checked:
        return "not_checked", ""
    indexed = release_assignments.set_index("analysis_id", drop=False)
    if analysis_id in indexed.index:
        row = indexed.loc[analysis_id]
        status = str(row["pipeline_status"])
        if status == "anchor_member":
            overlap = "core_definition"
        elif status == "conditional_definition_member":
            overlap = "conditional_definition"
        else:
            overlap = "source_non_definition"
        reference = row["assigned_reference_key"]
        return overlap, "" if pd.isna(reference) else str(reference)
    if biosample_id in set(release_assignments["biosample_id"]):
        return "same_biosample_other_analysis", ""
    return "not_in_release", ""


def prepare_projection_inputs(
    *,
    selection: pd.DataFrame,
    out_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    selected = selected_rows(selection)
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for (resource, biosample_id), group in selected.groupby(
        ["resource", "biosample_id"], sort=True
    ):
        selected_ids = sorted(set(group["analysis_id"].astype(str)))
        counts = {analysis_id: 0 for analysis_id in selected_ids}
        invalid_coordinate_counts = {
            analysis_id: 0 for analysis_id in selected_ids
        }
        for analysis_id in selected_ids:
            for variant in analysis_variants(str(resource), analysis_id):
                variation = variant.get("variation", {}) or {}
                if variation.get("type") != "CopyNumberChange":
                    continue
                state = cnv_state(variation)
                if state is None:
                    continue
                reported_ids = variant_analysis_ids(variant)
                if reported_ids and analysis_id not in reported_ids:
                    raise ValueError(
                        f"Variant returned for {analysis_id} reports other "
                        f"analyses: {sorted(reported_ids)}"
                    )
                location = variation.get("location", {}) or {}
                chromosome = str(location.get("chromosome", "")).removeprefix("chr")
                if chromosome not in GRCH38_AUTOSOME_REFSEQ:
                    continue
                sequence_id = location_sequence_id(location)
                if sequence_id != GRCH38_AUTOSOME_REFSEQ[chromosome]:
                    raise ValueError(
                        f"Analysis {analysis_id} under {biosample_id} is not "
                        f"consistently GRCh38: chromosome {chromosome}, "
                        f"sequence {sequence_id or 'missing'}"
                    )
                start = int(location["start"])
                end = int(location["end"])
                if start < 0 or end <= start:
                    invalid_coordinate_counts[analysis_id] += 1
                    continue
                segment_rows.append(
                    {
                        "sample_id": analysis_id,
                        "chromosome": chromosome,
                        "start": start,
                        "end": end,
                        "cnv_state": state,
                    }
                )
                counts[analysis_id] += 1

        for row in group.to_dict("records"):
            analysis_id = str(row["analysis_id"])
            topography = str(row["icdo_topography_id"])
            if not topography or topography.lower() == "nan":
                raise ValueError(f"Analysis lacks ICD-O topography: {analysis_id}")
            sample_rows.append(
                {
                    "sample_id": analysis_id,
                    "icdo_topography_id": topography,
                    "genome_build": "GRCh38",
                }
            )
            audit_rows.append(
                {
                    "resource": resource,
                    "biosample_id": biosample_id,
                    "analysis_id": analysis_id,
                    "n_autosomal_cnv_segments": counts[analysis_id],
                    "n_invalid_coordinate_segments_skipped": (
                        invalid_coordinate_counts[analysis_id]
                    ),
                    "cnvsim_overlap_status": row.get(
                        "cnvsim_overlap_status", "not_checked"
                    ),
                    "cnvsim_reference_key": row.get("cnvsim_reference_key", ""),
                }
            )

    missing = [
        row["analysis_id"]
        for row in audit_rows
        if int(row["n_autosomal_cnv_segments"]) == 0
    ]
    if missing:
        details = [
            {
                "analysis_id": row["analysis_id"],
                "invalid_coordinate_segments_skipped": int(
                    row["n_invalid_coordinate_segments_skipped"]
                ),
            }
            for row in audit_rows
            if row["analysis_id"] in set(missing)
        ]
        raise ValueError(
            "Selected analyses have no projectable CNVs after filtering "
            f"non-positive intervals: {details}"
        )

    samples = pd.DataFrame(sample_rows).drop_duplicates("sample_id")
    segments = pd.DataFrame(segment_rows).drop_duplicates(
        ["sample_id", "chromosome", "start", "end", "cnv_state"]
    )
    segments["chromosome_order"] = segments["chromosome"].astype(int)
    segments = segments.sort_values(
        ["sample_id", "chromosome_order", "start", "end", "cnv_state"]
    ).drop(columns="chromosome_order")
    audit = pd.DataFrame(audit_rows).sort_values("analysis_id")

    outputs = {
        "samples": out_dir / "samples.tsv",
        "segments": out_dir / "segments.tsv",
        "audit": out_dir / "retrieval_audit.tsv",
        "manifest": out_dir / "manifest.json",
    }
    samples.to_csv(outputs["samples"], sep="\t", index=False)
    segments.to_csv(outputs["segments"], sep="\t", index=False)
    audit.to_csv(outputs["audit"], sep="\t", index=False)
    manifest = {
        "schema_version": "cnvsim_bycon_projection_input_v1",
        "counts": {
            "selected_analyses": int(len(samples)),
            "autosomal_cnv_segments": int(len(segments)),
            "invalid_coordinate_segment_links_skipped": int(
                audit["n_invalid_coordinate_segments_skipped"].sum()
            ),
        },
        "files": {name: path.name for name, path in outputs.items() if name != "manifest"},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs
