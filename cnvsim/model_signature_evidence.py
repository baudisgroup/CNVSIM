"""Summarize subtype CNV signatures across cell-line profiles."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from cnvsim.analysis.chromosome_architecture import (
    CHR_LENGTH_BP_GRCH38,
    build_chr44_profiles,
    normalize_chromosome,
    normalize_segments,
)

PRIVATE_ID_PATTERN = re.compile(r"pgx(?:cs|bs|ind|an)-", re.IGNORECASE)
SIGNATURE_KEYS = ["reference_key", "track", "feature", "event", "amplitude"]
REASON_ORDER = (
    "no_segment_records",
    "not_level_labelled",
    "no_gene_row",
    "non_finite_value",
)
TRACK2_CHANNEL_COL = {
    ("gain", "all"): "dup_frac",
    ("loss", "all"): "del_frac",
    ("gain", "high"): "hldup_frac",
    ("loss", "high"): "hldel_frac",
}

SUBTYPE_COVERAGE_COLUMNS = [
    "reference_key",
    "root",
    "track",
    "feature",
    "event",
    "amplitude",
    "relative_to_root_background",
    "tumour_context_n",
    "tumour_context_value",
    "tumour_value_kind",
    "all_matched_models",
    "recommended_models",
    "recommended_evaluable_models",
    "recommended_recorded_only",
    "recommended_not_recorded_only",
    "recommended_mixed",
    "recommended_unknown_only",
    "recommended_with_unknown",
    "model_fraction_q25",
    "model_fraction_median",
    "model_fraction_q75",
]

MODEL_EVIDENCE_COLUMNS = [
    "reference_key",
    "root",
    "track",
    "feature",
    "event",
    "amplitude",
    "relative_to_root_background",
    "model_id",
    "model_label",
    "recommendation_tier",
    "is_recommended",
    "n_profiles",
    "evaluable_profiles",
    "recorded_profiles",
    "not_recorded_profiles",
    "unknown_profiles",
    "evidence_status",
    "fraction_min",
    "fraction_median",
    "fraction_max",
    "unknown_reasons",
]


def _direction_label(value: float) -> str:
    sign = int(np.sign(float(value)))
    return {1: "higher", -1: "lower", 0: "equal"}[sign]


def _counter_text(reasons: list[str]) -> str | None:
    counts = Counter(r for v in reasons if isinstance(v, str) and v for r in v.split(";") if r)
    if not counts:
        return None
    return ";".join(f"{k}={counts[k]}" for k in REASON_ORDER if k in counts)



def _validate_inputs(signatures, subtypes, models, profiles):
    for frame, columns, name in (
        (signatures, [*SIGNATURE_KEYS, "n_context", "context_value", "effect_size"], "signatures"),
        (subtypes, ["reference_key", "root"], "subtypes"),
        (models, ["model_id", "model_label", "root", "n_profiles", "assigned_reference_key",
                  "recommendation_tier", "is_recommended", "model_rank"], "models"),
        (profiles, ["analysis_id", "model_id", "root", "platform_family"], "profiles"),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if signatures[SIGNATURE_KEYS].isna().any().any() or signatures.duplicated(SIGNATURE_KEYS).any():
        raise ValueError("Signature keys must be complete and unique")
    if not signatures["track"].isin(["track1", "track2"]).all():
        raise ValueError("Unsupported signature track")
    if not signatures["event"].isin(["gain", "loss"]).all():
        raise ValueError("Unsupported signature event")
    t1 = signatures["track"].eq("track1")
    if (not signatures.loc[t1, "amplitude"].eq("chromosome_scale").all()
            or not signatures.loc[~t1, "amplitude"].isin(["all", "high"]).all()
            or not signatures.loc[t1, "feature"].isin([f"chr{x}" for x in CHR_LENGTH_BP_GRCH38]).all()):
        raise ValueError("Unsupported signature feature or amplitude")
    if not np.isfinite(signatures[["effect_size", "context_value", "n_context"]].to_numpy(float)).all():
        raise ValueError("Tumour signature statistics must be finite")
    if models.duplicated(["model_id", "root"]).any() or profiles.duplicated(["analysis_id", "root"]).any():
        raise ValueError("Model/root and analysis/root keys must be unique")
    if not models["is_recommended"].isin([True, False]).all():
        raise ValueError("is_recommended must contain booleans")
    root_map = subtypes.set_index("reference_key", verify_integrity=True)["root"]
    matched = models[models["assigned_reference_key"].notna()]
    if not matched["assigned_reference_key"].map(root_map).eq(matched["root"]).all():
        raise ValueError("Model consensus subtype/root disagrees with the atlas")
    for column in ("n_profiles", "model_rank"):
        values = matched[column].to_numpy(float)
        if not (np.isfinite(values) & (values >= 1) & (values == np.floor(values))).all():
            raise ValueError(f"Matched model {column} must contain positive integers")


def build_cell_line_signature_evidence(release_dir: Path, out_dir: Path) -> dict[str, object]:
    release_dir = Path(release_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError("Output directory must be new")
    if out_dir.is_relative_to(release_dir) or release_dir.is_relative_to(out_dir):
        raise ValueError("Output directory must not overlap the input release")

    signatures = pd.read_csv(release_dir / "atlas/signature_events.tsv", sep="\t")
    subtypes = pd.read_csv(release_dir / "atlas/subtypes.tsv", sep="\t")
    cell_lines = pd.read_csv(release_dir / "cell_line/cell_lines.tsv", sep="\t")
    profiles = pd.read_parquet(release_dir / "cell_line/model_projection_profiles.parquet")
    segments = pd.read_parquet(
        release_dir / "source_data/source_derived/cell_line/cell_line_autosomal_segments.parquet"
    )
    gene_dir = release_dir / "source_data/source_file/gene_cnv_fractions/cell_line"

    _validate_inputs(signatures, subtypes, cell_lines, profiles)

    # Keep the released signature order and raw effect direction.
    root_map = subtypes.set_index("reference_key", verify_integrity=True)["root"]
    sig = signatures[
        ["reference_key", "track", "feature", "event", "amplitude", "n_context", "context_value", "effect_size"]
    ].copy()
    sig["root"] = sig["reference_key"].map(root_map)
    if sig["root"].isna().any():
        raise ValueError("Some signature_events rows could not be mapped to a root in atlas/subtypes.tsv")
    sig["relative_to_root_background"] = sig["effect_size"].map(_direction_label)
    sig["tumour_context_n"] = sig["n_context"].astype(int)
    sig["tumour_context_value"] = sig["context_value"].astype(float)
    sig["tumour_value_kind"] = np.where(
        sig["track"].eq("track1"), "median_chromosome_fraction", "raw_event_prevalence"
    )
    sig["sig_order"] = np.arange(len(sig), dtype=int)

    # 2. Consensus-matched models and their catalogue profiles within each (model_id, root)
    matched_models = cell_lines[cell_lines["assigned_reference_key"].notna()][
        [
            "model_id",
            "model_label",
            "root",
            "n_profiles",
            "assigned_reference_key",
            "recommendation_tier",
            "is_recommended",
            "model_rank",
        ]
    ].copy()
    matched_models["public_model_label"] = matched_models["model_label"].where(
        ~matched_models["model_label"].fillna("").astype(str).str.contains(PRIVATE_ID_PATTERN),
        None,
    )
    matched_models["is_recommended"] = matched_models["is_recommended"].astype(bool)

    matched_profiles = profiles[["analysis_id", "model_id", "root", "platform_family"]].merge(
        matched_models[
            [
                "model_id",
                "public_model_label",
                "root",
                "n_profiles",
                "assigned_reference_key",
                "recommendation_tier",
                "is_recommended",
                "model_rank",
            ]
        ],
        on=["model_id", "root"],
        how="inner",
        validate="many_to_one",
    )
    prof_counts = matched_profiles.groupby(["model_id", "root"]).size().rename("actual_n_profiles").reset_index()
    chk = matched_models.merge(prof_counts, on=["model_id", "root"], how="left", validate="one_to_one")
    if not (chk["n_profiles"] == chk["actual_n_profiles"]).all():
        raise ValueError("Mismatch between cell_lines.tsv n_profiles and model_projection_profiles count")

    # 3. Segment normalization and profile-level observability flags
    all_matched_aids = sorted(matched_profiles["analysis_id"].astype(str).unique())
    raw_matched_seg = segments[segments["analysis_id"].astype(str).isin(set(all_matched_aids))].copy()
    raw_matched_seg["chr_norm"] = raw_matched_seg["chromosome"].map(normalize_chromosome)
    raw_matched_seg["chr_len"] = raw_matched_seg["chr_norm"].map(CHR_LENGTH_BP_GRCH38).astype(float)
    n_out_of_bounds = int(
        ((raw_matched_seg["start"] < 0) | (raw_matched_seg["end"] > raw_matched_seg["chr_len"])).sum()
    )
    n_nonpositive = int((raw_matched_seg["end"] <= raw_matched_seg["start"]).sum())
    if n_out_of_bounds != 0:
        raise ValueError(f"Found {n_out_of_bounds} out-of-bounds segment records")

    norm_seg, seg_qc = normalize_segments(raw_matched_seg)
    analyses_with_segments = set(norm_seg["analysis_id"].astype(str).unique())
    analyses_level_labelled = set(
        norm_seg.loc[
            norm_seg["copy_change"].fillna("").astype(str).str.contains("level", regex=False),
            "analysis_id",
        ]
        .astype(str)
        .unique()
    )

    chr44_wide = build_chr44_profiles(norm_seg, analysis_ids=all_matched_aids, prefix="CHR").set_index("analysis_id")

    # 4. Evaluate profile x signature_event across all roots with matched models
    profile_rows: list[pd.DataFrame] = []
    for root, sig_root in sig.groupby("root", sort=False):
        prof_root = matched_profiles[matched_profiles["root"].eq(root)].copy()
        if prof_root.empty:
            continue
        # Join profiles with signatures assigned to the same reference_key
        joined = prof_root.merge(
            sig_root,
            left_on=["assigned_reference_key", "root"],
            right_on=["reference_key", "root"],
            how="inner",
        )
        if joined.empty:
            continue

        aids = joined["analysis_id"].astype(str)
        has_seg = aids.isin(analyses_with_segments).to_numpy(bool)
        has_lvl = aids.isin(analyses_level_labelled).to_numpy(bool)
        is_t1 = joined["track"].eq("track1").to_numpy(bool)
        is_high = joined["amplitude"].eq("high").to_numpy(bool)

        values = np.full(len(joined), np.nan, dtype=float)
        present = np.zeros(len(joined), dtype=bool)

        # Track 1 values from chr44_wide
        if is_t1.any():
            t1_idx = np.flatnonzero(is_t1)
            t1_aids = joined.iloc[t1_idx]["analysis_id"].astype(str).to_numpy()
            t1_cols = (
                "CHR|"
                + joined.iloc[t1_idx]["feature"].astype(str).to_numpy()
                + "_"
                + joined.iloc[t1_idx]["event"].astype(str).to_numpy()
            )
            for col in np.unique(t1_cols):
                mask = t1_cols == col
                rows_idx = t1_idx[mask]
                values[rows_idx] = chr44_wide.loc[t1_aids[mask], col].to_numpy(float)
                present[rows_idx] = True

        # Track 2 values from root gene parquet
        if (~is_t1).any():
            t2_idx = np.flatnonzero(~is_t1)
            t2_genes = joined.iloc[t2_idx]["feature"].astype(str).unique().tolist()
            root_aids = prof_root["analysis_id"].astype(str).unique().tolist()
            gf = pd.read_parquet(gene_dir / f"{root}.parquet")
            gf = gf[
                gf["analysis_id"].astype(str).isin(set(root_aids))
                & gf["gene_symbol"].astype(str).isin(set(t2_genes))
            ].copy()
            gf_idx = gf.set_index(["analysis_id", "gene_symbol"], verify_integrity=True)

            t2_sub = joined.iloc[t2_idx]
            keys = list(zip(t2_sub["analysis_id"].astype(str), t2_sub["feature"].astype(str)))
            in_gf = pd.Index(keys).isin(gf_idx.index)
            present[t2_idx] = in_gf
            for (ev_dir, amp), col_name in TRACK2_CHANNEL_COL.items():
                ch_mask = (
                    (t2_sub["event"].astype(str).to_numpy() == ev_dir)
                    & (t2_sub["amplitude"].astype(str).to_numpy() == amp)
                    & in_gf
                )
                if ch_mask.any():
                    ch_rows = t2_idx[ch_mask]
                    ch_keys = [keys[i] for i in np.flatnonzero(ch_mask)]
                    values[ch_rows] = gf_idx.loc[ch_keys, col_name].to_numpy(float)

        finite = np.isfinite(values)
        if (values[finite] < 0).any() or (values[finite & is_t1] > 1).any():
            raise ValueError("Negative gene fractions or chromosome coverage outside [0, 1]")
        r_no_seg = ~has_seg
        r_no_lvl = has_seg & is_high & (~has_lvl)
        r_no_gene = (~is_t1) & has_seg & (~present)
        r_non_fin = has_seg & present & (~finite)

        reasons: list[str] = []
        for ns, nl, ng, nf in zip(r_no_seg, r_no_lvl, r_no_gene, r_non_fin):
            parts = []
            if ns:
                parts.append("no_segment_records")
            if nl:
                parts.append("not_level_labelled")
            if ng:
                parts.append("no_gene_row")
            if nf:
                parts.append("non_finite_value")
            reasons.append(";".join(parts))

        unknown = np.array([bool(r) for r in reasons], dtype=bool)
        # Clear values on unknown rows so unknown profiles never contribute to fraction summaries
        clean_values = np.where(unknown, np.nan, values)

        state = np.empty(len(joined), dtype=object)
        state[unknown] = "unknown"
        eval_t1 = (~unknown) & is_t1
        state[eval_t1] = "evaluable_continuous"
        eval_t2 = (~unknown) & (~is_t1)
        with np.errstate(invalid="ignore"):
            rec_t2 = eval_t2 & (clean_values > 0.0)
        state[rec_t2] = "recorded"
        state[eval_t2 & (~rec_t2)] = "not_recorded"

        joined["profile_value"] = clean_values
        joined["profile_state"] = state
        joined["unknown_reason"] = reasons
        profile_rows.append(joined)

    prof_ev = (pd.concat(profile_rows, ignore_index=True) if profile_rows else
               pd.DataFrame(columns=list(dict.fromkeys([
                   *matched_profiles.columns, *sig.columns,
                   "profile_value", "profile_state", "unknown_reason"]))))

    # Aggregate profiles within each model and root.
    group_cols = [
        "sig_order",
        "reference_key",
        "root",
        "track",
        "feature",
        "event",
        "amplitude",
        "relative_to_root_background",
        "model_rank",
        "model_id",
        "public_model_label",
        "recommendation_tier",
        "is_recommended",
        "n_profiles",
    ]
    model_records: list[dict[str, object]] = []
    for key, grp in prof_ev.groupby(group_cols, sort=False, dropna=False):
        (
            sig_order,
            ref_key,
            root,
            track,
            feature,
            event,
            amplitude,
            rel_dir,
            model_rank,
            model_id,
            model_label,
            rec_tier,
            is_rec,
            n_prof,
        ) = key
        states = grp["profile_state"].to_numpy()
        vals = grp["profile_value"].to_numpy(float)
        eval_vals = vals[np.isfinite(vals)]

        n_unk = int((states == "unknown").sum())
        n_eval = int(len(states) - n_unk)
        if n_eval + n_unk != int(n_prof):
            raise ValueError(f"Profile count mismatch for {model_id} {ref_key} {feature}")
        if len(eval_vals) != n_eval:
            raise ValueError(f"Finite value count mismatch for {model_id} {ref_key} {feature}")

        if track == "track1":
            n_rec = pd.NA
            n_not_rec = pd.NA
            ev_status = "unknown_only" if n_eval == 0 else "continuous_evaluable"
        else:
            n_rec = int((states == "recorded").sum())
            n_not_rec = int((states == "not_recorded").sum())
            if n_rec + n_not_rec + n_unk != int(n_prof):
                raise ValueError(f"Track 2 profile partition mismatch for {model_id} {ref_key} {feature}")
            if n_eval == 0:
                ev_status = "unknown_only"
            elif n_rec == n_eval:
                ev_status = "recorded_only"
            elif n_not_rec == n_eval:
                ev_status = "not_recorded_only"
            else:
                ev_status = "mixed"

        if n_eval > 0:
            f_min = float(np.min(eval_vals))
            f_med = float(np.median(eval_vals))
            f_max = float(np.max(eval_vals))
        else:
            f_min = np.nan
            f_med = np.nan
            f_max = np.nan

        model_records.append(
            {
                "sig_order": int(sig_order),
                "model_rank": int(model_rank),
                "reference_key": ref_key,
                "root": root,
                "track": track,
                "feature": feature,
                "event": event,
                "amplitude": amplitude,
                "relative_to_root_background": rel_dir,
                "model_id": model_id,
                "model_label": model_label if pd.notna(model_label) else None,
                "recommendation_tier": rec_tier,
                "is_recommended": bool(is_rec),
                "n_profiles": int(n_prof),
                "evaluable_profiles": int(n_eval),
                "recorded_profiles": n_rec,
                "not_recorded_profiles": n_not_rec,
                "unknown_profiles": int(n_unk),
                "evidence_status": ev_status,
                "fraction_min": f_min,
                "fraction_median": f_med,
                "fraction_max": f_max,
                "unknown_reasons": _counter_text(grp["unknown_reason"].tolist()),
            }
        )

    model_df = (
        pd.DataFrame(model_records, columns=["sig_order", "model_rank", *MODEL_EVIDENCE_COLUMNS])
        .sort_values(["sig_order", "model_rank", "model_id"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    for col in ("recorded_profiles", "not_recorded_profiles"):
        model_df[col] = model_df[col].astype("Int64")

    # Each recommended evaluable model contributes one median.
    model_by_sig = {k: g for k, g in model_df.groupby("sig_order", sort=False)}
    subtype_records: list[dict[str, object]] = []
    for row in sig.sort_values("sig_order").itertuples(index=False):
        m_all = model_by_sig.get(int(row.sig_order))
        if m_all is None or m_all.empty:
            n_all = 0
            n_rec_models = 0
            n_rec_eval = 0
            n_rec_only = pd.NA if row.track == "track1" else 0
            n_not_rec_only = pd.NA if row.track == "track1" else 0
            n_mixed = pd.NA if row.track == "track1" else 0
            n_unk_only = 0
            n_with_unk = 0
            q25, q50, q75 = np.nan, np.nan, np.nan
        else:
            n_all = int(len(m_all))
            m_rec = m_all[m_all["is_recommended"]].copy()
            n_rec_models = int(len(m_rec))
            n_rec_eval = int((m_rec["evaluable_profiles"] > 0).sum())
            n_unk_only = int((m_rec["evidence_status"] == "unknown_only").sum())
            n_with_unk = int((m_rec["unknown_profiles"] > 0).sum())
            if n_rec_eval + n_unk_only != n_rec_models:
                raise ValueError(f"Recommended evaluable + unknown_only != recommended_models for {row.reference_key}")

            if row.track == "track1":
                n_rec_only = pd.NA
                n_not_rec_only = pd.NA
                n_mixed = pd.NA
            else:
                n_rec_only = int((m_rec["evidence_status"] == "recorded_only").sum())
                n_not_rec_only = int((m_rec["evidence_status"] == "not_recorded_only").sum())
                n_mixed = int((m_rec["evidence_status"] == "mixed").sum())
                if n_rec_only + n_not_rec_only + n_mixed + n_unk_only != n_rec_models:
                    raise ValueError(f"Track 2 model category partition failed for {row.reference_key} {row.feature}")

            eval_medians = m_rec.loc[m_rec["evaluable_profiles"] > 0, "fraction_median"].to_numpy(float)
            if len(eval_medians) > 0:
                q25, q50, q75 = [float(x) for x in np.quantile(eval_medians, [0.25, 0.50, 0.75])]
            else:
                q25, q50, q75 = np.nan, np.nan, np.nan

        subtype_records.append(
            {
                "reference_key": row.reference_key,
                "root": row.root,
                "track": row.track,
                "feature": row.feature,
                "event": row.event,
                "amplitude": row.amplitude,
                "relative_to_root_background": row.relative_to_root_background,
                "tumour_context_n": int(row.tumour_context_n),
                "tumour_context_value": float(row.tumour_context_value),
                "tumour_value_kind": row.tumour_value_kind,
                "all_matched_models": int(n_all),
                "recommended_models": int(n_rec_models),
                "recommended_evaluable_models": int(n_rec_eval),
                "recommended_recorded_only": n_rec_only,
                "recommended_not_recorded_only": n_not_rec_only,
                "recommended_mixed": n_mixed,
                "recommended_unknown_only": int(n_unk_only),
                "recommended_with_unknown": int(n_with_unk),
                "model_fraction_q25": q25,
                "model_fraction_median": q50,
                "model_fraction_q75": q75,
            }
        )

    subtype_df = pd.DataFrame(subtype_records, columns=SUBTYPE_COVERAGE_COLUMNS)
    for col in ("recommended_recorded_only", "recommended_not_recorded_only", "recommended_mixed"):
        subtype_df[col] = subtype_df[col].astype("Int64")

    model_out = model_df[MODEL_EVIDENCE_COLUMNS].copy()
    subtype_out = subtype_df[SUBTYPE_COVERAGE_COLUMNS].copy()

    # 7. Verify all invariants before writing
    expected_model_rows = int(sig["reference_key"].map(
        matched_models.groupby("assigned_reference_key").size()).fillna(0).sum())
    _verify_outputs(sig, subtype_out, model_out, expected_model_rows)
    public_texts = [table.to_csv(sep="\t", index=False) for table in (subtype_out, model_out)]
    for text in public_texts:
        if PRIVATE_ID_PATTERN.search(text) or re.search(r"/(?:Users|home|private|tmp)/", text):
            raise ValueError("Public tables contain an individual identifier or local path")

    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    local_dir = out_dir / "LOCAL_ONLY"
    local_dir.mkdir(mode=0o700)
    local_dir.chmod(0o700)

    subtype_path = out_dir / "SUBTYPE_SIGNATURE_COVERAGE.tsv"
    model_path = out_dir / "MODEL_SIGNATURE_EVIDENCE.tsv"
    subtype_path.write_text(public_texts[0], encoding="utf-8")
    model_path.write_text(public_texts[1], encoding="utf-8")
    (out_dir / "README.md").write_text(TABLE_GUIDE, encoding="utf-8")

    local_prof_path = local_dir / "profile_signature_evidence.parquet"
    with local_prof_path.open("xb") as handle:
        local_prof_path.chmod(0o600)
        prof_ev.drop(columns=["sig_order", "model_rank"]).to_parquet(handle, index=False)

    summary = {
        "signature_rows": int(len(subtype_out)),
        "track1_signature_rows": int(subtype_out["track"].eq("track1").sum()),
        "track2_signature_rows": int(subtype_out["track"].eq("track2").sum()),
        "track2_higher_rows": int(
            (subtype_out["track"].eq("track2") & subtype_out["relative_to_root_background"].eq("higher")).sum()
        ),
        "track2_lower_rows": int(
            (subtype_out["track"].eq("track2") & subtype_out["relative_to_root_background"].eq("lower")).sum()
        ),
        "subtypes_total": int(subtype_out["reference_key"].nunique()),
        "subtypes_with_matched_models": int(
            subtype_out.loc[subtype_out["all_matched_models"].gt(0), "reference_key"].nunique()
        ),
        "subtypes_without_matched_models": int(
            subtype_out.loc[subtype_out["all_matched_models"].eq(0), "reference_key"].nunique()
        ),
        "matched_model_root_pairs": int(len(matched_models)),
        "matched_recommended_model_root_pairs": int(matched_models["is_recommended"].sum()),
        "matched_profiles": int(len(matched_profiles)),
        "profiles_with_normalized_segments": int(len(analyses_with_segments)),
        "profiles_without_segments": int(len(all_matched_aids) - len(analyses_with_segments)),
        "profiles_level_labelled": int(len(analyses_level_labelled)),
        "profiles_generic_gain_loss_only": int(len(analyses_with_segments) - len(analyses_level_labelled)),
        "segment_qc": {
            **seg_qc,
            "zero_or_negative_length_segments_discarded": n_nonpositive,
            "out_of_bounds_segments": n_out_of_bounds,
        },
        "model_signature_evidence_rows": int(len(model_out)),
        "model_signature_recommended_rows": int(model_out["is_recommended"].sum()),
    }
    manifest_path = local_dir / "run_manifest.json"
    manifest = {"schema_version": 1, "inputs": {"release_dir": str(release_dir)},
                "status": "COMPLETE", "summary": summary}
    with manifest_path.open("x", encoding="utf-8") as handle:
        manifest_path.chmod(0o600)
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return summary


def _verify_outputs(sig: pd.DataFrame, subtype_out: pd.DataFrame, model_out: pd.DataFrame,
                    expected_model_rows: int) -> None:
    if list(subtype_out.columns) != SUBTYPE_COVERAGE_COLUMNS:
        raise ValueError("SUBTYPE_SIGNATURE_COVERAGE columns do not match specification")
    if list(model_out.columns) != MODEL_EVIDENCE_COLUMNS:
        raise ValueError("MODEL_SIGNATURE_EVIDENCE columns do not match specification")
    if len(subtype_out) != len(sig):
        raise ValueError("Summary row count differs from input signatures")
    if len(model_out) != expected_model_rows:
        raise ValueError("Model evidence row count differs from signature/model matches")

    key_cols = ["reference_key", "track", "feature", "event", "amplitude"]
    if not subtype_out[key_cols].equals(sig[key_cols].reset_index(drop=True)):
        raise ValueError("Signature key ordering or values differ from atlas/signature_events.tsv")

    # Denominator identities in SUBTYPE_SIGNATURE_COVERAGE
    if not (
        subtype_out["recommended_evaluable_models"] + subtype_out["recommended_unknown_only"]
        == subtype_out["recommended_models"]
    ).all():
        raise ValueError("recommended_evaluable_models + recommended_unknown_only != recommended_models")

    t1_sub = subtype_out[subtype_out["track"].eq("track1")]
    if not (
        t1_sub[["recommended_recorded_only", "recommended_not_recorded_only", "recommended_mixed"]].isna().all().all()
    ):
        raise ValueError("Track 1 binary model count columns must be blank in SUBTYPE_SIGNATURE_COVERAGE")

    t2_sub = subtype_out[subtype_out["track"].eq("track2")]
    t2_sum = (
        t2_sub["recommended_recorded_only"].astype(int)
        + t2_sub["recommended_not_recorded_only"].astype(int)
        + t2_sub["recommended_mixed"].astype(int)
        + t2_sub["recommended_unknown_only"].astype(int)
    )
    if not (t2_sum == t2_sub["recommended_models"]).all():
        raise ValueError("Track 2 four-way recommended model partition does not sum to recommended_models")

    no_eval_sub = subtype_out[subtype_out["recommended_evaluable_models"].eq(0)]
    if not no_eval_sub[["model_fraction_q25", "model_fraction_median", "model_fraction_q75"]].isna().all().all():
        raise ValueError("Quartiles must be blank when recommended_evaluable_models == 0")

    # Denominator identities in MODEL_SIGNATURE_EVIDENCE
    if not (model_out["evaluable_profiles"] + model_out["unknown_profiles"] == model_out["n_profiles"]).all():
        raise ValueError("evaluable_profiles + unknown_profiles != n_profiles in MODEL_SIGNATURE_EVIDENCE")

    t1_mod = model_out[model_out["track"].eq("track1")]
    if not t1_mod[["recorded_profiles", "not_recorded_profiles"]].isna().all().all():
        raise ValueError("Track 1 recorded_profiles and not_recorded_profiles must be blank")

    t2_mod = model_out[model_out["track"].eq("track2")]
    if not (
        t2_mod["recorded_profiles"].astype(int)
        + t2_mod["not_recorded_profiles"].astype(int)
        + t2_mod["unknown_profiles"].astype(int)
        == t2_mod["n_profiles"]
    ).all():
        raise ValueError("Track 2 recorded + not_recorded + unknown != n_profiles")

    unk_mod = model_out[model_out["evaluable_profiles"].eq(0)]
    if not unk_mod[["fraction_min", "fraction_median", "fraction_max"]].isna().all().all():
        raise ValueError("fraction_min/median/max must be blank when evaluable_profiles == 0")



TABLE_GUIDE = """# Cell-Line Subtype Signature Evidence

This directory summarizes multi-profile copy-number variation (CNV) evidence across consensus-matched Cellosaurus cell-line models (`cell_line/cell_lines.tsv` and `cell_line/model_projection_profiles.parquet`) for every released subtype signature event (`atlas/signature_events.tsv`).

## Measurement Rules

1. **Event universe**: Every `(reference_key, track, feature, event, amplitude)` row in `atlas/signature_events.tsv` is preserved in its released order. `relative_to_root_background` (`higher` or `lower`) records the sign of the released raw `effect_size` (`context_value - background_value`). This is the raw direction, not the adjusted effect or a measure of model suitability. For `lower` signatures, recording an event does not indicate agreement with the subtype direction.
2. **Track 1 chromosome events (`track1`, `amplitude = chromosome_scale`)**: Tumour `tumour_context_value` is the median fraction of the GRCh38 chromosome covered by same-direction segments (`gain` or `loss`) among subtype definition samples. Cell-line profiles with autosomal segment records are evaluated using the same GRCh38 chromosome lengths and union-coverage calculation (`cnvsim.analysis.chromosome_architecture.normalize_segments` and `build_chr44_profiles`), reporting continuous chromosome coverage fractions (`fraction_min`, `fraction_median`, `fraction_max` per model; `model_fraction_q25`, `model_fraction_median`, `model_fraction_q75` across recommended evaluable models' medians). Binary carrier columns (`recorded_only`, `not_recorded_only`, `mixed`, `recorded_profiles`, `not_recorded_profiles`) are left blank for Track 1 rows.
3. **Track 2 gene events (`track2`, `amplitude = all` or `high`)**: Tumour `tumour_context_value` is the raw subtype prevalence (`dup_frac > 0` or `del_frac > 0` for `all`; `hldup_frac > 0` or `hldel_frac > 0` for `high`). In evaluable cell-line profiles, an event is counted as `recorded` when the corresponding gene fraction is `> 0` and `not_recorded` when `== 0`, while the released gene fraction values are summarized alongside the profile counts. Gene fractions are taken from the released `gene_cnv_fractions` and can exceed 1 when overlapping same-direction source segments are summed; presence (>0) is unaffected. These values are not bounded coverage percentages. No clipping or recomputation is applied.
4. **Observability (`unknown`)**:
   - Profiles with no normalized autosomal segment records (`no_segment_records`) are marked `unknown` across all channels (`track1`, `track2 all`, and `track2 high`).
   - For high-level gene channels (`track2`, `amplitude = high`), a profile is evaluable only when its normalized segments carry amplitude-level state annotations (`low-level` or `high-level` in `copy_change`); profiles carrying only generic `gain`/`loss` annotations are marked `unknown` (`not_level_labelled`). Any amplitude-labelled segment makes a profile level-labelled, even if other segments use generic labels. This is an annotation proxy, not proof of per-gene callability; a level-labelled profile with high-channel value zero is `not_recorded`.
   - Missing gene rows (`no_gene_row`) or non-finite values (`non_finite_value`), if any, are marked `unknown` and never zero-filled.

---

## Files

### 1. `SUBTYPE_SIGNATURE_COVERAGE.tsv`

One row per released signature event (`atlas/signature_events.tsv`). Subtypes without consensus-matched models (`all_matched_models = 0`) or without recommended evaluable models (`recommended_evaluable_models = 0`) retain their tumour signature statistics and have blank model fraction quartiles.

| Column | Description |
|---|---|
| `reference_key` | Released CNVSIM reference subtype identifier (`CTX` or `COND`). |
| `root` | Anatomical root grouping (`atlas/subtypes.tsv`). |
| `track` | Signature track (`track1` chromosome-scale or `track2` gene-level). |
| `feature` | Chromosome (`chr1`–`chr22`) for `track1` or gene symbol for `track2`. |
| `event` | CNV direction (`gain` or `loss`). |
| `amplitude` | Amplitude channel (`chromosome_scale` for `track1`; `all` or `high` for `track2`). |
| `relative_to_root_background` | Direction of raw `effect_size` relative to the same-root background (`higher`, `lower`, or `equal`). |
| `tumour_context_n` | Number of tumour definition analyses in `reference_key` (`n_context` in `atlas/signature_events.tsv`). |
| `tumour_context_value` | Released `context_value` in tumour definition analyses. |
| `tumour_value_kind` | Meaning of `tumour_context_value` (`median_chromosome_fraction` for `track1`; `raw_event_prevalence` for `track2`). |
| `all_matched_models` | Total cell-line models consensus-assigned to `reference_key` in `cell_line/cell_lines.tsv`. |
| `recommended_models` | Consensus-matched models with `is_recommended = True` (`Tier A` or `Tier B`). |
| `recommended_evaluable_models` | Recommended models with at least one evaluable profile for this event (`recommended_evaluable_models + recommended_unknown_only = recommended_models`). |
| `recommended_recorded_only` | Recommended models whose evaluable profiles all record the event (`recorded_profiles = evaluable_profiles > 0`). Blank for `track1`. |
| `recommended_not_recorded_only` | Recommended models whose evaluable profiles all have `0` coverage (`not_recorded_profiles = evaluable_profiles > 0`). Blank for `track1`. |
| `recommended_mixed` | Recommended models with both `recorded` and `not_recorded` profiles (`recorded_profiles > 0` and `not_recorded_profiles > 0`). Blank for `track1`. |
| `recommended_unknown_only` | Recommended models with zero evaluable profiles (`unknown_profiles = n_profiles`). For `track2`, `recommended_recorded_only + recommended_not_recorded_only + recommended_mixed + recommended_unknown_only = recommended_models`. |
| `recommended_with_unknown` | Overlapping count of recommended models that have at least one `unknown` profile (`unknown_profiles > 0`). |
| `model_fraction_q25` | 25th percentile of `fraction_median` across recommended evaluable models (one median per model). Blank when `recommended_evaluable_models = 0`. |
| `model_fraction_median` | Median of `fraction_median` across recommended evaluable models. Blank when `recommended_evaluable_models = 0`. |
| `model_fraction_q75` | 75th percentile of `fraction_median` across recommended evaluable models. Blank when `recommended_evaluable_models = 0`. |

### 2. `MODEL_SIGNATURE_EVIDENCE.tsv`

One row per `(consensus-matched model, signature event)` pair using all profiles of that model within its root, including profiles assigned to other subtypes. Models without a signature-bearing consensus subtype have no event rows.

| Column | Description |
|---|---|
| `reference_key` | Consensus-assigned CNVSIM reference subtype of the model in `root`. |
| `root` | Anatomical root grouping. |
| `track` | Signature track (`track1` or `track2`). |
| `feature` | Chromosome (`chr1`–`chr22`) or gene symbol. |
| `event` | CNV direction (`gain` or `loss`). |
| `amplitude` | Amplitude channel (`chromosome_scale`, `all`, or `high`). |
| `relative_to_root_background` | Direction of raw `effect_size` in tumours (`higher`, `lower`, or `equal`). |
| `model_id` | Standardized Cellosaurus model identifier (`cellosaurus:CVCL_...`). |
| `model_label` | Public cell-line name (withheld/blank when the source label is an internal biosample identifier). |
| `recommendation_tier` | Released model recommendation tier (`Tier A`, `Tier B`, `Provisional`, or `Not recommended`). |
| `is_recommended` | Boolean indicator (`True` for `Tier A` and `Tier B`). |
| `n_profiles` | Total catalogue CNV profiles for `(model_id, root)` (`evaluable_profiles + unknown_profiles = n_profiles`). |
| `evaluable_profiles` | Number of profiles with a valid measurement for this event and amplitude channel. |
| `recorded_profiles` | Number of evaluable profiles with gene coverage fraction `> 0`. Blank for `track1`. |
| `not_recorded_profiles` | Number of evaluable profiles with gene coverage fraction `== 0`. Blank for `track1`. |
| `unknown_profiles` | Number of profiles that cannot be evaluated for this event/channel. |
| `evidence_status` | Model-level summary across evaluable profiles (`recorded_only`, `not_recorded_only`, `mixed`, or `unknown_only` for `track2`; `continuous_evaluable` or `unknown_only` for `track1`). |
| `fraction_min` | Minimum chromosome union-coverage fraction or released gene fraction value across evaluable profiles. Blank when `evaluable_profiles = 0`. |
| `fraction_median` | Median chromosome union-coverage fraction or released gene fraction value across evaluable profiles. Blank when `evaluable_profiles = 0`. |
| `fraction_max` | Maximum chromosome union-coverage fraction or released gene fraction value across evaluable profiles. Blank when `evaluable_profiles = 0`. |
| `unknown_reasons` | Semicolon-delimited counts of `unknown` profile reasons (`no_segment_records`, `not_level_labelled`, `no_gene_row`, `non_finite_value`). Reason counts can overlap and need not sum to `unknown_profiles`. Blank when `unknown_profiles = 0`. |

A model with recorded and unknown profiles, but no evaluable zero profiles, is `recorded_only` and also contributes to `recommended_with_unknown`. Not recorded does not establish biological absence. No cell-line mutation measurements or functional validation are implied.

Only the two TSVs and this guide are aggregate outputs. `LOCAL_ONLY/` contains profile-level evidence and local run information and must not be distributed with the aggregate tables.
"""
