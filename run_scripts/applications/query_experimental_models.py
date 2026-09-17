from __future__ import annotations

"""Query the public CNVSIM cell-line resource."""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.experimental_model_query import (
    query_context_for_model,
    query_models_for_context,
)


def _write_tsv(frame: pd.DataFrame, output: str | None, label: str) -> None:
    if not output:
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    print(f"{label}: {path}")


def _display_model_id(value: object) -> str:
    text = str(value)
    prefix = "cellosaurus:"
    return text[len(prefix) :] if text.lower().startswith(prefix) else text


def _short_text(value: object, width: int = 36) -> str:
    if pd.isna(value):
        return "-"
    text = " ".join(str(value).split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _display_model_label(value: object) -> str:
    if pd.isna(value):
        return "-"
    text = " ".join(str(value).split())
    matched = re.search(r"\[cell line ([^\]]+)\]", text, flags=re.IGNORECASE)
    return _short_text(matched.group(1) if matched else text)


def _percent(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{100 * float(numeric):.1f}%"


def _decimal(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def _print_context_summary(
    frame: pd.DataFrame,
    reference_key: str,
) -> None:
    print("\nCNVSIM cell-line recommendations")
    print(f"Reference: {reference_key}")
    if frame.empty:
        print("No models satisfy the requested evidence filter.")
        return

    level = str(frame.iloc[0]["reference_level"])
    root = str(frame.iloc[0]["root"])
    print(f"Layer: {level} | Root: {root} | Models returned: {len(frame)}")
    display = pd.DataFrame(
        {
            "rank": frame["model_rank"].astype(int),
            "model": frame["model_label"].map(_display_model_label),
            "accession": frame["model_id"].map(_display_model_id),
            "tier": frame["recommendation_tier"],
            "profiles": frame["n_profiles"].astype(int),
            "consistency": frame["assignment_consistency"].map(_percent),
            "coverage": frame["assignment_coverage"].map(_percent),
            "distance/limit": frame["calibrated_distance_ratio"].map(
                _decimal
            ),
        }
    )
    print(display.to_string(index=False))
    print("Lower distance/limit indicates a closer calibrated match.")


def _print_model_summary(frame: pd.DataFrame) -> None:
    print("\nCNVSIM cell-line lookup")
    if frame.empty:
        print("No matching model-root group.")
        return
    first = frame.iloc[0]
    display_label = _display_model_label(first["model_label"])
    print(
        f"Model: {display_label} ({_display_model_id(first['model_id'])})"
    )
    full_label = _short_text(first["model_label"], width=80)
    if display_label != full_label:
        print(f"Label: {full_label}")
    if bool(first.get("multi_root_model", False)):
        print(f"Model-root groups: {len(frame)}")

    for row in frame.itertuples(index=False):
        reference = (
            str(row.assigned_reference_key)
            if pd.notna(row.assigned_reference_key)
            else "-"
        )
        level = str(row.reference_level) if pd.notna(row.reference_level) else "-"
        print(f"\nRoot: {row.root}")
        print(
            f"Match: {reference} [{level}] | "
            f"Status: {row.model_assignment_status}"
        )
        print(
            f"Evidence: {row.recommendation_tier} | {row.evidence_label}"
        )
        print(
            f"Profiles: {int(row.n_profiles)} | "
            f"assigned: {int(row.n_reference_assigned)} | "
            f"consensus support: {int(row.n_consensus_support)}"
        )
        print(
            f"Consistency: {_percent(row.assignment_consistency)} | "
            f"coverage: {_percent(row.assignment_coverage)} | "
            f"distance/limit: {_decimal(row.calibrated_distance_ratio)}"
        )
        print(f"Interpretation: {row.status_explanation}")


def _print_analysis_summary(frame: pd.DataFrame) -> None:
    print("\nAnalysis-level audit")
    if frame.empty:
        print("No analysis rows.")
        return
    for row in frame.itertuples(index=False):
        assigned = (
            str(row.assigned_reference_key)
            if pd.notna(row.assigned_reference_key)
            else "-"
        )
        candidate = (
            str(row.descriptive_candidate_reference_key)
            if pd.notna(row.descriptive_candidate_reference_key)
            else "-"
        )
        print(
            f"- {row.analysis_id} | {row.platform_family_refined} | "
            f"{row.assignment_status} | assigned: {assigned}"
        )
        if row.candidate_interpretation == "descriptive_only_not_assignment":
            print(
                f"  descriptive candidate: {candidate} "
                "(not a calibrated assignment)"
            )
        print(f"  reason: {row.status_reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query ranked cell-line models for a CNVSIM reference, or look up "
            "the CNVSIM context of a frozen Progenetix/Cellosaurus model."
        )
    )
    parser.add_argument(
        "-cell-line-dir",
        "--cell-line-dir",
        "-application-dir",
        "--application-dir",
        dest="application_dir",
        required=True,
        help="Public CNVSIM cell_line resource directory.",
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("-reference-key", "--reference-key")
    query.add_argument("-model-id", "--model-id")
    parser.add_argument(
        "-root",
        "--root",
        help="Restrict a multi-root model query to one CNVSIM root.",
    )
    parser.add_argument("-top-n", "--top-n", type=int, default=5)
    parser.add_argument(
        "--include-nonrecommended",
        action="store_true",
        help="Include provisional and discordant context-model candidates.",
    )
    parser.add_argument(
        "--include-analysis-details",
        action="store_true",
        help="Show the analysis-level audit rows for a model query.",
    )
    parser.add_argument("-out", "--out", help="Optional TSV output for the main result.")
    parser.add_argument(
        "-analysis-out",
        "--analysis-out",
        help="Optional TSV output for model analysis-level details.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.reference_key:
        if args.root:
            parser.error("-root is only valid with -model-id")
        if args.analysis_out or args.include_analysis_details:
            parser.error(
                "analysis detail options are only valid with -model-id"
            )
        result = query_models_for_context(
            args.application_dir,
            args.reference_key,
            top_n=args.top_n,
            recommended_only=not args.include_nonrecommended,
        )
        _print_context_summary(result, args.reference_key)
        _write_tsv(result, args.out, "Full result")
        return

    summary, analyses = query_context_for_model(
        args.application_dir,
        args.model_id,
        root=args.root,
    )
    _print_model_summary(summary)
    _write_tsv(summary, args.out, "Full model result")
    if args.include_analysis_details or args.analysis_out:
        _print_analysis_summary(analyses)
        _write_tsv(analyses, args.analysis_out, "Full analysis audit")


if __name__ == "__main__":
    main()
