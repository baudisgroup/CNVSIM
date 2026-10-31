"""Export aggregate release candidates and reader tables without fitting models."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import manifest as run_manifest

KEY = ["root", "ONCOTREE_CODE", "gene", "subtype"]
CHECKS = ["matched_baseline", "density_only", "type_only", "combined", "amplitude", "quadratic", "restricted"]
STATUS_ORDER = ["supported_across_context_checks", "supported_with_unestimable_sensitivity",
                "supported_but_not_all_sensitivities", "no_detected_adjusted_association"]
AGGREGATES = [
    "association/conditional_associations.parquet", "association/cohort_reconstruction_audit.tsv",
    "center_consistency/CENTER_RESULTS.parquet", "center_consistency/FOCUS_45_CENTER_COMPARISON.parquet", "center_consistency/center_sample_counts.tsv",
    "sensitivity/CONTEXT_RESULTS.parquet", "sensitivity/sample_type_coverage.tsv",
    "variant_description/COHORT_COUNTS.tsv", "variant_description/CATEGORY_COUNTS.tsv", "variant_description/DOMAIN_RELATION_COUNTS.tsv",
    "variant_description/POSITION_COUNTS.tsv", "variant_description/PROTEIN_CHANGE_COUNTS.tsv",
]



def ensure_aggregate(frame):
    forbidden = {"analysis_id", "sample_id", "patient_id", "tumor_sample_barcode", "normal_sample_barcode", "external_sample_id"}
    if forbidden.intersection(str(c).lower() for c in frame.columns):
        raise ValueError("Individual-level identifier column in export input")
    pattern = re.compile(r"GENIE[-_]|TCGA-[A-Za-z0-9]{2}-|pgxbs-|pgxcs-|/Users/|/home/")
    for col in frame.select_dtypes(include=["object", "string"]):
        if frame[col].dropna().astype(str).str.contains(pattern, regex=True).any():
            raise ValueError(f"Identifier or local path detected in {col}")


def keyed(frame, expected):
    if len(frame) != expected or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError(f"Expected {expected} unique, nonmissing comparison keys")
    return frame.set_index(KEY).sort_index()


def add_center_directions(centers):
    """Compare two centres' log-OR signs; this is not a significance test.

    same/opposite require two estimable, finite, positive ORs with nonzero
    log-OR signs. Missing or neutral directions are not_estimable. Agreement
    with the pooled result remains in the existing center_consistency_status.
    """
    out = centers.copy()
    for name in ("primary", "amplitude", "continuous"):
        a, b = out[f"MSK_{name}_or"], out[f"non_MSK_{name}_or"]
        valid = out[f"MSK_{name}_status"].eq("estimable") & out[f"non_MSK_{name}_status"].eq("estimable")
        valid &= np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0) & a.ne(1) & b.ne(1)
        direction = pd.Series("not_estimable", index=out.index)
        direction.loc[valid] = np.where(np.sign(np.log(a.loc[valid])).eq(np.sign(np.log(b.loc[valid]))), "same", "opposite")
        out[f"{name}_center_direction"] = direction
    return out


def derive_overview(local, sensitivity, centers):
    """Join the existing effects and statuses, without fitting or FDR recalculation."""
    li, si = keyed(local, 9582), keyed(sensitivity, 9582)
    if not li.index.equals(si.index):
        raise ValueError("Association and sensitivity comparison families differ")
    robust = local[local.residual_status.eq("robust_residual_association")].copy()
    focus = sensitivity[sensitivity.pooled_robust].copy()
    ri, fi, zi = keyed(robust, 45), keyed(focus, 45), keyed(centers, 45)
    if not ri.index.equals(fi.index) or not ri.index.equals(zi.index):
        raise ValueError("The three 45-comparison families differ")
    for other, name in [(fi.old_primary_or, "sensitivity"), (zi.pooled_primary_or, "center")]:
        if not np.allclose(ri.primary_or, other, rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError(f"Pooled effects disagree in {name}")
    robust["previous_evidence"] = np.select([
        robust.fga_stratification_status.eq("retained_after_fga_stratification"),
        robust.assay_exact_score_q_declared.lt(.05)],
        ["old_FGA_retained", "old_panel_only"], default="new_after_conditioning")
    robust["direction"] = np.where(robust.primary_or > 1, "higher", np.where(robust.primary_or < 1, "lower", "equal"))
    columns = ["root", "ONCOTREE_CODE", "subtype", "gene", "previous_evidence", "direction", "subtype_n",
               "primary_target_informative", "primary_control_informative", "primary_or", "primary_ci_low",
               "primary_ci_high", "primary_q_declared", "amplitude_q_declared", "continuous_or", "continuous_q_declared",
               "subtype_pos", "other_subtypes_n", "other_subtypes_pos"]
    overview = robust[columns].merge(focus[KEY + ["combined_status", "combined_or", "combined_ci_low", "combined_ci_high", "combined_q_declared", "combined_significant", "context_status"]], on=KEY, validate="one_to_one")
    overview = overview.merge(centers[KEY + ["center_consistency_status", "MSK_primary_status", "MSK_primary_or", "non_MSK_primary_status", "non_MSK_primary_or", "heterogeneity_status", "heterogeneity_q_45"]], on=KEY, validate="one_to_one")
    return overview.sort_values(KEY, kind="stable")


def export_tables(run_dir, out_dir):
    run, out = Path(run_dir).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    if run == out or run in out.parents or out in run.parents:
        raise ValueError("Export directory must be separate from the analysis run")
    if out.exists():
        raise FileExistsError("Use a new export directory; existing exports are preserved")
    run_record = run_manifest.read(run)
    for stage in ["association-prepare", "association-analyze", "center-consistency", "sensitivity"]:
        record = run_record["stages"].get(stage, {})
        if record.get("status") != "COMPLETE":
            raise ValueError(f"Stage {stage} is not complete")
    variant_status = run_record["stages"].get("variant-description", {}).get("status")
    if variant_status not in (None, "COMPLETE"):
        raise ValueError("Variant description was attempted but did not complete")
    aggregates = [rel for rel in AGGREGATES if not rel.startswith("variant_description/") or variant_status == "COMPLETE"]
    frames = {}
    for rel in aggregates:
        p = run / rel
        frames[rel] = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep="\t", float_precision="round_trip")
        ensure_aggregate(frames[rel])
    center_all = frames["center_consistency/CENTER_RESULTS.parquet"]
    if len(center_all) != 19164 or center_all.duplicated(KEY + ["center_group"]).any():
        raise ValueError("Expected the complete two-center comparison family")
    overview = derive_overview(frames["association/conditional_associations.parquet"],
                                   frames["sensitivity/CONTEXT_RESULTS.parquet"],
                                   frames["center_consistency/FOCUS_45_CENTER_COMPARISON.parquet"])
    ensure_aggregate(overview)
    frames["center_consistency/FOCUS_45_CENTER_COMPARISON.parquet"] = add_center_directions(frames["center_consistency/FOCUS_45_CENTER_COMPARISON.parquet"])
    out.mkdir(parents=True)
    inventory = []
    for rel, frame in list(frames.items()) + [("reader/FOCUS_45_OVERVIEW.tsv", overview)]:
        dest = out / Path(rel).with_suffix(".tsv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, sep="\t", index=False)
        inventory.append({"path": dest.relative_to(out).as_posix(), "rows": len(frame),
                          "columns": {c: str(frame[c].dtype) for c in frame},
                          "source": rel if rel in frames else "Derived from frozen aggregate columns; see exporter and README.md",
                          "release_status": "candidate_pending_release_review"})
    (out / "README.md").write_text(
        "Mutation-context aggregate release candidate\n\n"
        "Start with reader/FOCUS_45_OVERVIEW.tsv. The same tables can supply the versioned data release and manuscript supplement; do not recompute results for each.\n"
        "The unit is one root/diagnosis/gene/subtype comparison, not one patient. subtype_pos/subtype_n and other_subtypes_pos/other_subtypes_n are unadjusted counts.\n"
        "primary_* are local-CNV-conditioned results; combined_* additionally adjust for reported mutation density and sample type. OR and CI belong to their respective models. *_q_declared retain the frozen comparison family.\n"
        "context_status is copied from the analysis: all checks supported, supported with an unestimable check, supported but not all checks significant, or no detected combined-model association. Missing effects are not zero.\n"
        "Center status measures consistency inside the pooled discovery dataset, not independent replication. FOCUS_45_OVERVIEW joins association, sensitivity and centre results using the four comparison keys.\n"
        "The three *_center_direction columns compare MSK and non-MSK log-OR signs: same, opposite or not_estimable. They do not imply statistical significance or agreement with the pooled estimate; use center_consistency_status for the pooled comparison.\n"
        "Variant description is optional: 8 aggregate tables without it, 13 when completed.\n"
        "Variants: frozen_evaluable and combined_informative are distinct populations; carriers and variant_observations are different denominators. Category carrier counts can overlap. No variant tests are fitted.\n"
        "PROTEIN_CHANGE_COUNTS and POSITION_COUNTS contain rare aggregate cells, including counts of one. All files are publication candidates subject to the same release review, whether supplied as a data package or journal supplement.\n"
        "No raw MAF, sample linkage, individual records, job cache or local-path configuration is included. selected_local_state_descriptive_counts and the local TCGA follow-up are not exported.\n"
        "Methods accompany the versioned data release; operational rules are in the matching analysis code. The export inventory is recorded in the run-level mutation_manifest.json. Exporting does not authorize redistribution or create a public release.\n")
    manifest = {"status": "AGGREGATE_EXPORT_COMPLETE_NOT_PUBLISHED", "statistics_refitted": False,
                "variants_included": variant_status == "COMPLETE", "files": inventory}
    data = run_manifest.read(run)
    data["exports"] = {"output_dir": str(out), **manifest}
    run_manifest.write(run, data)
    print(json.dumps({"status": manifest["status"], "tables": len(inventory), "output_dir": str(out)}, indent=2))
