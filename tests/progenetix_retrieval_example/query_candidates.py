from __future__ import annotations

"""Query tumour or cell-line CNV analyses from the public bycon API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bycon import (
    beacon_query,
    exact_biosample,
    is_cnv_analysis,
    load_release_overlap,
    overlap_annotation,
)


def _term(record: dict[str, object], key: str) -> tuple[str, str]:
    value = record.get(key, {}) or {}
    if not isinstance(value, dict):
        return "", ""
    return str(value.get("id", "")), str(value.get("label", ""))


def _candidate_row(
    *,
    resource: str,
    analysis: dict[str, object],
    biosample: dict[str, object] | None,
    release_assignments: pd.DataFrame,
    overlap_checked: bool,
) -> dict[str, object]:
    metadata_available = biosample is not None
    biosample = biosample or {}
    analysis_id = str(analysis.get("id", ""))
    biosample_id = str(analysis.get("biosampleId", ""))
    topography_id, topography_label = _term(biosample, "icdoTopography")
    _, diagnosis_label = _term(biosample, "histologicalDiagnosis")
    platform_id, platform_label = _term(analysis, "platformModel")
    cellline = biosample.get("celllineInfo", {}) or {}
    if not isinstance(cellline, dict):
        cellline = {}
    overlap, reference = overlap_annotation(
        analysis_id,
        biosample_id,
        release_assignments,
        overlap_checked,
    )
    return {
        "selected": "",
        "retrieval_status": "ready" if metadata_available else "metadata_unavailable",
        "resource": resource,
        "biosample_id": biosample_id,
        "analysis_id": analysis_id,
        "biosample_label": str(
            biosample.get("label") or biosample.get("name") or analysis.get("label") or ""
        ),
        "diagnosis": diagnosis_label,
        "icdo_topography_id": topography_id,
        "icdo_topography_label": topography_label,
        "cellosaurus_id": str(cellline.get("id", "")),
        "platform": platform_label or platform_id,
        "cnvsim_overlap_status": overlap,
        "cnvsim_reference_key": reference,
    }


def query_candidates(
    *,
    resource: str,
    filters: list[str],
    biosample_ids: list[str],
    include_descendants: bool,
    filter_logic: str,
    limit: int,
    skip: int,
    release_dir: Path | None,
) -> pd.DataFrame:
    release_assignments, overlap_checked = load_release_overlap(release_dir)
    biosamples: dict[str, dict[str, object]] = {}
    analyses: list[dict[str, object]] = []

    if biosample_ids:
        for biosample_id in biosample_ids:
            biosample = exact_biosample(resource, biosample_id)
            if biosample is None:
                raise ValueError(f"Biosample not found in {resource}: {biosample_id}")
            biosamples[biosample_id] = biosample
            analyses.extend(
                beacon_query(
                    f"biosamples/{biosample_id}/analyses",
                    resource=resource,
                    limit=0,
                )
            )
    else:
        if not filters:
            raise ValueError("Provide at least one --filters or --biosample-id")
        # Start from currently retrievable biosamples. The global analyses endpoint
        # can include historical records whose linked biosamples no longer exist.
        scan_limit = min(max(limit * 2, limit), 100)
        biosample_rows = beacon_query(
            "biosamples",
            resource=resource,
            filters=filters,
            include_descendants=include_descendants,
            filter_logic=filter_logic,
            limit=scan_limit,
            skip=skip,
        )
        biosamples = {str(row.get("id", "")): row for row in biosample_rows}
        biosample_order = [
            str(row.get("id", "")) for row in biosample_rows if row.get("id")
        ]

        def fetch_analyses(biosample_id: str) -> list[dict[str, object]]:
            return beacon_query(
                f"biosamples/{biosample_id}/analyses",
                resource=resource,
                limit=0,
            )

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(biosample_order)))) as pool:
            fetched = dict(zip(biosample_order, pool.map(fetch_analyses, biosample_order)))
        for biosample_id in biosample_order:
            analyses.extend(fetched[biosample_id])

    analyses = [analysis for analysis in analyses if is_cnv_analysis(analysis)]
    analyses = sorted(
        analyses,
        key=lambda row: (str(row.get("biosampleId", "")), str(row.get("id", ""))),
    )[:limit]
    if not analyses:
        raise ValueError("The query returned no CNV analyses")

    rows: list[dict[str, object]] = []
    for analysis in analyses:
        biosample_id = str(analysis.get("biosampleId", ""))
        if not biosample_id:
            raise ValueError(f"Analysis lacks biosampleId: {analysis.get('id')}")
        biosample = biosamples.get(biosample_id)
        if biosample is None:
            biosample = exact_biosample(resource, biosample_id)
        rows.append(
            _candidate_row(
                resource=resource,
                analysis=analysis,
                biosample=biosample,
                release_assignments=release_assignments,
                overlap_checked=overlap_checked,
            )
        )
    candidates = pd.DataFrame(rows).sort_values(["biosample_id", "analysis_id"])
    if candidates["analysis_id"].duplicated().any():
        raise ValueError("bycon returned duplicate analysis IDs")
    return candidates


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected 'true' or 'false'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List projectable CNV analyses from Progenetix tumour data or the "
            "cancercelllines dataset. Mark selected=yes in the output before "
            "running prepare_projection_inputs.py."
        )
    )
    parser.add_argument(
        "--resource", "-resource", choices=["tumour", "cell-line"], required=True
    )
    parser.add_argument(
        "--filters", "-filters", "--filter", dest="filter", action="append", default=[],
        help="Filter ID; repeat this option to provide multiple filters.",
    )
    parser.add_argument("--biosample-id", "-biosample-id", action="append", default=[])
    parser.add_argument(
        "--includeDescendantTerms", "-includeDescendantTerms",
        dest="include_descendants", type=parse_bool,
        default=True, metavar="{true,false}",
        help="Include descendant terms for all filters (default: true).",
    )
    parser.add_argument(
        "--include-descendants",
        dest="include_descendants", action="store_true",
        default=argparse.SUPPRESS, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--filterLogic", "-filterLogic", "--filter-logic", "-filter-logic",
        dest="filter_logic",
        choices=["AND", "OR"], default="AND",
    )
    parser.add_argument("--limit", "-limit", type=int, default=50)
    parser.add_argument("--skip", "-skip", type=int, default=0)
    parser.add_argument("--release-dir", "-release-dir")
    parser.add_argument("--out", "-out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    output = Path(args.out).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = query_candidates(
        resource=args.resource,
        filters=args.filter,
        biosample_ids=args.biosample_id,
        include_descendants=bool(args.include_descendants),
        filter_logic=args.filter_logic,
        limit=int(args.limit),
        skip=int(args.skip),
        release_dir=Path(args.release_dir) if args.release_dir else None,
    )
    candidates.to_csv(output, sep="\t", index=False)
    print("[CNVSIM] bycon candidate query complete")
    print(f"  resource: {args.resource}")
    print(f"  CNV analyses: {len(candidates):,}")
    print(
        "  ready for selection: "
        f"{int(candidates['retrieval_status'].eq('ready').sum()):,}"
    )
    print(
        "  metadata unavailable: "
        f"{int(candidates['retrieval_status'].eq('metadata_unavailable').sum()):,}"
    )
    print(f"  candidates: {output}")
    print("  Mark selected=yes for every analysis to retrieve and project.")


if __name__ == "__main__":
    main()
