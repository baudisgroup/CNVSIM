#!/usr/bin/env python3
"""Summarize held-out TCGA projection coverage and compatibility with v1 assignments.

Outputs (written to --out-dir):
  SOURCE_HOLDOUT_COVERAGE_BY_ROOT.tsv       Roster counts, assigned/abstained counts, and v1 statuses.
  SOURCE_HOLDOUT_STATUS_BY_ROOT.tsv         Projection status and reason breakdown by root.
  SOURCE_HOLDOUT_COMPATIBILITY_BY_ROOT.tsv  Per-root ARI and AMI on commonly assigned samples.
  SOURCE_HOLDOUT_CROSSTAB.tsv               Contingency counts between rebuilt contexts and v1 labels.
  SOURCE_HOLDOUT_SUMMARY.json               Summary totals and input/output SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

PROJECTION_FILE = "projection_assignments.tsv"
ASSIGNED = "projected_assignment"
V1_CORE_STATUSES = ("anchor_member", "projected_assignment")
SMALL_N = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_projections(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames, hashes = [], {}
    for path in paths:
        path = Path(path)
        file = path / PROJECTION_FILE if path.is_dir() else path
        if not file.exists():
            raise FileNotFoundError(f"projection output not found: {file}")
        frame = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
        missing = {"sample_id", "root", "assignment_status", "assigned_reference_key"} - set(frame.columns)
        if missing:
            raise KeyError(f"{file} lacks columns {sorted(missing)}")
        if "status_reason" not in frame.columns:
            frame["status_reason"] = pd.NA
        frame["source_file"] = str(file)
        frames.append(frame)
        hashes[str(file)] = sha256_file(file)
    proj = pd.concat(frames, ignore_index=True)
    if proj["sample_id"].duplicated().any():
        dup = proj.loc[proj["sample_id"].duplicated(keep=False), "sample_id"].unique()[:5].tolist()
        raise ValueError(f"a held-out sample appears in more than one projection row: {dup}")
    return proj, hashes


def v1_core_label(v1: pd.DataFrame) -> pd.Series:
    core = v1["v1_status"].isin(V1_CORE_STATUSES) & v1["v1_key"].fillna("").str.contains("-CTX")
    return v1["v1_key"].where(core)


def compatibility(common: pd.DataFrame) -> dict:
    n = int(len(common))
    k_new = int(common["new_context"].nunique())
    k_v1 = int(common["v1_label"].nunique())
    row = {"common_n": n, "new_clusters_in_common": k_new, "v1_clusters_in_common": k_v1,
           "small_n": n < SMALL_N, "ari": None, "ami": None}
    if n < 2 or k_new < 2 or k_v1 < 2:
        reasons = [r for r, bad in (("n<2", n < 2), ("one_new_cluster", k_new < 2), ("one_v1_cluster", k_v1 < 2)) if bad]
        row["estimability"] = "not_estimable:" + ",".join(reasons)
        return row
    row["estimability"] = "estimable"
    row["ari"] = round(float(adjusted_rand_score(common["v1_label"], common["new_context"])), 6)
    row["ami"] = round(float(adjusted_mutual_info_score(common["v1_label"], common["new_context"])), 6)
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--roster", required=True, type=Path)
    p.add_argument("--v1-assignments", required=True, type=Path)
    p.add_argument("--projection", required=True, type=Path, action="append")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args(argv)
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")

    roster = pd.read_csv(args.roster, sep="\t", dtype=str)
    if roster["analysis_id"].duplicated().any() or roster.duplicated(["root", "participant"]).any():
        raise ValueError("roster must have one row per analysis and per root x participant")
    v1 = (pd.read_parquet(args.v1_assignments, columns=["analysis_id", "pipeline_status", "assigned_reference_key"])
          .rename(columns={"pipeline_status": "v1_status", "assigned_reference_key": "v1_key"}))
    proj, proj_hashes = read_projections(args.projection)

    extra = set(proj["sample_id"]) - set(roster["analysis_id"])
    if extra:
        raise ValueError(f"{len(extra)} projected samples are not on the roster")
    x = roster.merge(v1, on="analysis_id", how="left", validate="one_to_one")
    if x["v1_status"].isna().any():
        raise ValueError("some roster analyses are absent from the v1 assignments")
    x["v1_label"] = v1_core_label(x)
    x = x.merge(proj.rename(columns={"sample_id": "analysis_id", "root": "proj_root"}),
                on="analysis_id", how="left", validate="one_to_one")
    clash = x["proj_root"].notna() & x["proj_root"].ne(x["root"])
    if clash.any():
        raise ValueError(f"{int(clash.sum())} projected rows carry a root different from the roster")

    covered_roots = set(proj["root"])
    x["outcome"] = "abstained"
    x.loc[x["assignment_status"].isna() & ~x["root"].isin(covered_roots), "outcome"] = "missing_root_run"
    x.loc[x["assignment_status"].isna() & x["root"].isin(covered_roots), "outcome"] = "missing_row"
    assigned = x["assignment_status"].eq(ASSIGNED) & x["assigned_reference_key"].notna()
    x.loc[assigned, "outcome"] = "assigned"
    bad = x["assignment_status"].eq(ASSIGNED) & x["assigned_reference_key"].isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows are projected_assignment without a reference key")
    x["new_context"] = x["assigned_reference_key"].where(assigned)

    cov_rows, comp_rows, cross_parts = [], [], []
    for root in sorted(x["root"].unique()) + ["ALL"]:
        s = x if root == "ALL" else x[x["root"].eq(root)]
        n = len(s)
        row = {"root": root, "roster_n": n, "projection_input": root == "ALL" or root in covered_roots}
        for outcome in ["assigned", "abstained", "missing_root_run", "missing_row"]:
            row[f"{outcome}_n"] = int(s["outcome"].eq(outcome).sum())
        row["assigned_share_of_roster"] = round(row["assigned_n"] / n, 6) if n else None
        for status in ["anchor_member", "projected_assignment", "conditional_definition_member", "unresolved",
                       "ambiguous", "graph_outlier", "root_only_reference"]:
            row[f"v1_{status}_n"] = int(s["v1_status"].eq(status).sum())
        row["v1_core_label_n"] = int(s["v1_label"].notna().sum())
        cov_rows.append(row)
        if root == "ALL":
            continue
        common = s[s["outcome"].eq("assigned") & s["v1_label"].notna()]
        comp = {"root": root, "roster_n": n, **compatibility(common)}
        comp["common_share_of_roster"] = round(comp["common_n"] / n, 6) if n else None
        comp["common_v1_anchor_n"] = int(common["v1_status"].eq("anchor_member").sum())
        comp["common_v1_projected_n"] = int(common["v1_status"].eq("projected_assignment").sum())
        comp_rows.append(comp)
        if len(common):
            cross_parts.append(common.groupby(["root", "new_context", "v1_label"]).size().rename("n").reset_index())

    status = (x.assign(new_status=x["assignment_status"].fillna("(" + x["outcome"] + ")"),
                       reason=x["status_reason"].fillna(""))
               .groupby(["root", "outcome", "new_status", "reason"]).size().rename("n").reset_index())
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "SOURCE_HOLDOUT_COVERAGE_BY_ROOT.tsv": pd.DataFrame(cov_rows),
        "SOURCE_HOLDOUT_STATUS_BY_ROOT.tsv": status,
        "SOURCE_HOLDOUT_COMPATIBILITY_BY_ROOT.tsv": pd.DataFrame(comp_rows),
        "SOURCE_HOLDOUT_CROSSTAB.tsv": (pd.concat(cross_parts, ignore_index=True) if cross_parts
                            else pd.DataFrame(columns=["root", "new_context", "v1_label", "n"])),
    }
    for name, frame in files.items():
        frame.to_csv(out / name, sep="\t", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "summarize_source_holdout.py",
        "reading": ("Coverage uses the fixed roster as denominator. ARI/AMI describe compatibility with the "
                    "released v1 grouping, which was built with TCGA; they are computed per root on the "
                    "common labelled set only and are not pooled across roots. Not for tuning or root choice."),
        "definitions": {"assigned_status": ASSIGNED, "v1_core_statuses": list(V1_CORE_STATUSES),
                        "small_n_threshold": SMALL_N, "ami_average_method": "arithmetic"},
        "totals": {k: v for k, v in cov_rows[-1].items() if k != "root"},
        "inputs": {"roster": sha256_file(args.roster), "v1_assignments": sha256_file(args.v1_assignments),
                   "projections": proj_hashes},
        "outputs": {name: sha256_file(out / name) for name in files},
    }
    (out / "SOURCE_HOLDOUT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
