#!/usr/bin/env python3
"""Run source-held-out reference validation (non-TCGA rebuild and TCGA projection).

Stages (all outputs written under --out-dir):
  hashes    Record or compare SHA-256 checksums of all required inputs.
  split     Build the non-TCGA training pool and held-out TCGA test roster from metadata.
  prepare   Subset pre-encoded v1 features of one root to the training records.
  discover  Refit preprocessing, subtype discovery, and calibration on training records.
  project   Project held-out TCGA records onto the rebuilt root reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SNP6_IDS = {"geo:GPL6801", "arrayexpress:A-AFFY-142", "geo:GPL8226", "Genome_Wide_SNP_6"}
SNP6_LABEL = re.compile(r"GenomeWideSNP_6|Genome-Wide Human SNP (?:Array )?6\.0|SNP Array 6\.0", re.IGNORECASE)
TCGA_TAGS = {"pgx:cohort-TCGA", "pgx:cohort-TCGAcancers"}
NOT_ELIGIBLE = {"out_of_scope_root", "no_autosomal_gene_cnv_signal"}
MIN_FAMILY = 50  # configs/preprocess_baseline.yaml encoding.min_samples_per_family
RULE_ORDER = ["T", "L", "G", "B2", "B3"]
GLASS_TAG = "cbioportal:difg_glass"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------- inputs

def inputs(args) -> dict[str, Path]:
    atlas = args.atlas_root
    return {
        "tumour_metadata": atlas / "source_file/metadata/tumour_analysis_metadata.parquet",
        "source_release_manifest": atlas / "source_file/release_manifest.json",
        "gene_cache_manifest": atlas / "source_file/gene_panel/cache/manifest.json",
        "gene_cache_index": atlas / "source_file/gene_panel/cache/analysis_index.parquet",
        "gene_list": atlas / "source_file/gene_panel/cancer_gene_list_geneid_p14.tsv",
        "chr44_profiles": atlas / "source_derived/sample_chr44_profiles.parquet",
        "v1_meta_cleaned": atlas / "work/preprocess_v1_baseline/meta_cleaned.tsv",
        "v1_encode_manifest": atlas / "work/preprocess_v1_baseline/features_unfiltered/encode_manifest.tsv",
        "v1_assignments": args.v1_assignments,
        "preprocess_config": REPO / "configs/preprocess_baseline.yaml",
        "analysis_config": REPO / "configs/analysis_v1.yaml",
        "code_validate_source_holdout": Path(__file__).resolve(),
        "code_summarize_source_holdout": Path(__file__).resolve().with_name("summarize_source_holdout.py"),
    }


def input_hashes(args) -> dict[str, str]:
    files = dict(inputs(args))
    feat = args.atlas_root / "work/preprocess_v1_baseline/features_unfiltered"
    for path in sorted(feat.glob("*")):
        if path.is_file() and not path.name.startswith("."):
            files[f"v1_features/{path.name}"] = path
    return {name: sha256(path) for name, path in sorted(files.items())}


def cmd_hashes(args) -> None:
    current = input_hashes(args)
    if args.compare:
        other = json.loads(Path(args.compare).read_text())["sha256"]
        diff = sorted(k for k in set(current) | set(other) if current.get(k) != other.get(k))
        print(json.dumps({"identical": not diff, "differing_inputs": diff}, indent=2))
        if diff:
            sys.exit(1)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "INPUT_HASHES.json", {"sha256": current})
    print(f"{len(current)} inputs hashed -> {args.out_dir / 'INPUT_HASHES.json'}")


def require_hashes(args) -> dict[str, str]:
    """Stop unless the inputs equal the hashes recorded by the hashes stage."""
    recorded = json.loads((args.out_dir / "INPUT_HASHES.json").read_text())["sha256"]
    current = input_hashes(args)
    if current != recorded:
        diff = sorted(k for k in set(current) | set(recorded) if current.get(k) != recorded.get(k))
        raise ValueError(f"inputs differ from INPUT_HASHES.json: {diff[:10]}")
    return current


# --------------------------------------------------------------------------- split

def build_split(args) -> pd.DataFrame:
    paths = inputs(args)
    meta = pd.read_parquet(paths["tumour_metadata"], columns=[
        "analysis_id", "individual_id", "biosample_id", "icdot_root_grouped", "source_group_refined",
        "platform_id", "platform_label", "platform_family_refined", "cohorts"])
    status = pd.read_parquet(paths["v1_assignments"], columns=["analysis_id", "root", "pipeline_status"])
    x = meta.merge(status, on="analysis_id", how="inner", validate="one_to_one")
    if len(x) != len(meta) or len(x) != len(status):
        raise ValueError("metadata and v1 status do not cover the same records")
    if not x["root"].astype(str).eq(x["icdot_root_grouped"].astype(str)).all():
        raise ValueError("v1 root differs from metadata root")
    for col in ("individual_id", "biosample_id"):
        x[col] = x[col].replace("", np.nan)
    tags = x["cohorts"].fillna("").str.split(";")

    is_tcga = x["source_group_refined"].eq("TCGA")
    rule = pd.DataFrame(index=x.index)
    rule["T"] = is_tcga
    ind = set(x.loc[is_tcga, "individual_id"].dropna())
    bio = set(x.loc[is_tcga, "biosample_id"].dropna())
    rule["L"] = ~is_tcga & (x["individual_id"].isin(ind) | x["biosample_id"].isin(bio))
    rule["G"] = ~is_tcga & tags.map(lambda t: GLASS_TAG in t)
    snp6 = x["platform_id"].isin(SNP6_IDS) | x["platform_label"].fillna("").str.contains(SNP6_LABEL)
    rule["B2"] = ~is_tcga & snp6
    rule["B3"] = ~is_tcga & tags.map(lambda t: bool(TCGA_TAGS & set(t)))

    x["tier"] = np.select([rule[["T", "L", "G"]].any(axis=1), rule[["B2", "B3"]].any(axis=1)],
                          ["a", "b"], default="c")
    x["first_rule"] = "none"
    for name in reversed(RULE_ORDER):
        x.loc[rule[name], "first_rule"] = name
    x["is_eligible"] = ~x["pipeline_status"].isin(NOT_ELIGIBLE)
    # Person key for TCGA records: individual_id; a missing id stays its own unit.
    x["participant"] = np.where(is_tcga, x["individual_id"].fillna("record:" + x["analysis_id"]), None)

    pool = x[is_tcga & x["is_eligible"]].sort_values(["root", "participant", "analysis_id"], kind="stable")
    test_ids = set(pool.loc[~pool.duplicated(["root", "participant"]), "analysis_id"])
    x["role"] = np.select(
        [x["analysis_id"].isin(test_ids), x["analysis_id"].isin(set(pool["analysis_id"])), x["tier"].eq("a"),
         x["tier"].eq("b") & (args.pool == "conservative")],
        ["test", "test_pool_not_representative", "excluded_tier_a", "excluded_tier_b"], default="train")

    train, test = x[x["role"].eq("train")], x[x["role"].eq("test")]
    tier_a = x[x["tier"].eq("a")]
    checks = {
        "train_tcga_source_records": int(train["source_group_refined"].eq("TCGA").sum()),
        "train_individual_shared_with_tcga": int(train["individual_id"].isin(ind).sum()),
        "train_biosample_shared_with_tcga": int(train["biosample_id"].isin(bio).sum()),
        "train_glass_records": int(train["cohorts"].fillna("").str.split(";").map(lambda t: GLASS_TAG in t).sum()),
        "test_participant_repeated_within_root": int(test.duplicated(["root", "participant"]).sum()),
        "tier_a_records_in_train": int(train.index.isin(tier_a.index).sum()),
    }
    if any(checks.values()):
        raise ValueError(f"split identity check failed: {checks}")
    x.attrs["checks"] = checks
    x.attrs["rule_counts"] = {
        name: {"all": int(rule[name].sum()), "eligible": int((rule[name] & x["is_eligible"]).sum())}
        for name in RULE_ORDER}
    return x


def cmd_split(args) -> None:
    hashes = require_hashes(args)
    x = build_split(args)
    out = args.out_dir / "split"
    out.mkdir(parents=True, exist_ok=False)
    cols = ["analysis_id", "root", "role", "tier", "first_rule", "is_eligible", "platform_family_refined",
            "participant"]
    x[cols].sort_values(["role", "root", "analysis_id"]).to_csv(
        out / "SPLIT.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    x.loc[x["role"].eq("test"), ["root", "participant", "analysis_id"]].sort_values(
        ["root", "participant"]).to_csv(out / "TEST_ROSTER.tsv", sep="\t", index=False)
    elig = x[x["is_eligible"]]
    summary = {
        "pool": args.pool,
        "records": int(len(x)),
        "roles": x["role"].value_counts().sort_index().to_dict(),
        "eligible_by_role": elig["role"].value_counts().sort_index().to_dict(),
        "rule_counts_any_order": x.attrs["rule_counts"],
        "first_rule_eligible": elig["first_rule"].value_counts().sort_index().to_dict(),
        "test_participants": int(x["role"].eq("test").sum()),
        "identity_checks": x.attrs["checks"],
        "input_sha256": hashes,
        "split_sha256": sha256(out / "SPLIT.tsv.gz"),
        "access": "LOCAL_ONLY: contains record ids",
    }
    write_json(out / "SPLIT_SUMMARY.json", summary)
    print(json.dumps({k: summary[k] for k in ("roles", "eligible_by_role", "test_participants")}, indent=2))


def load_split(args) -> pd.DataFrame:
    path = args.out_dir / "split" / "SPLIT.tsv.gz"
    summary = json.loads((args.out_dir / "split" / "SPLIT_SUMMARY.json").read_text())
    if sha256(path) != summary["split_sha256"]:
        raise ValueError("split file changed after the split stage")
    split = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    split["is_eligible"] = split["is_eligible"].eq("True")
    return split


# --------------------------------------------------------------------------- prepare

def cmd_prepare(args) -> None:
    require_hashes(args)
    split = load_split(args)
    root = args.root
    run = args.out_dir / "runs" / root
    pre = run / "preprocess"
    feat_out = pre / "features_unfiltered"
    feat_out.mkdir(parents=True, exist_ok=False)
    v1 = args.atlas_root / "work/preprocess_v1_baseline"
    s = split[split["root"].eq(root)]
    train = s[s["role"].eq("train")]
    # Family sizes as the encoder counts them: all metadata rows of the root.
    fam = train["platform_family_refined"].fillna("Unknown").value_counts()
    dropped = fam[fam < MIN_FAMILY]
    keep_ids = set(train.loc[~train["platform_family_refined"].fillna("Unknown").isin(dropped.index), "analysis_id"])

    X = pd.read_parquet(v1 / "features_unfiltered" / f"{root}_features.parquet")
    X.index = X.index.astype(str)
    meta = pd.read_csv(v1 / "features_unfiltered" / f"{root}_meta.tsv", sep="\t", dtype=str, low_memory=False)
    burden = pd.read_csv(v1 / "features_unfiltered" / f"{root}_sample_burden.tsv", sep="\t", dtype={"analysis_id": str})
    if not meta["analysis_id"].tolist() == X.index.tolist():
        raise ValueError("v1 features and metadata rows are not aligned")
    rows = X.index.isin(keep_ids)
    eligible_train = set(train.loc[train["is_eligible"], "analysis_id"])
    if set(X.index[X.index.isin(eligible_train)]) != eligible_train:
        raise ValueError("some eligible training records are missing from the v1 features")
    X.loc[rows].to_parquet(feat_out / f"{root}_features.parquet")
    meta.loc[rows].to_csv(feat_out / f"{root}_meta.tsv", sep="\t", index=False)
    burden[burden["analysis_id"].isin(keep_ids)].to_csv(feat_out / f"{root}_sample_burden.tsv", sep="\t", index=False)
    v1_info = json.loads((v1 / "features_unfiltered" / f"{root}_info.json").read_text())
    info = {"n_samples": int(rows.sum()), "n_features": int(X.shape[1]),
            "v1_encoding_info": v1_info,
            "source_holdout": {"reused_from": "v1 features_unfiltered", "v1_rows": int(len(X)),
                               "removed_held_out_or_excluded": int((~X.index.isin(set(train["analysis_id"]))).sum()),
                               "families_below_minimum_all_rows": {k: int(v) for k, v in dropped.items()},
                               "removed_by_family_rule": int((X.index.isin(set(train["analysis_id"])) & ~rows).sum())}}
    write_json(feat_out / f"{root}_info.json", info)
    pd.DataFrame([{"icdot_root": root, "n_samples": int(rows.sum()), "n_features": int(X.shape[1]),
                   "features_file": f"{root}_features.parquet", "meta_file": f"{root}_meta.tsv",
                   "sample_burden_file": f"{root}_sample_burden.tsv", "info_file": f"{root}_info.json"}]
                 ).to_csv(feat_out / "encode_manifest.tsv", sep="\t", index=False)
    print(json.dumps(info["source_holdout"] | {"rows_kept": int(rows.sum())}, indent=2))


# --------------------------------------------------------------------------- discover

def plan(args) -> list[tuple[str, list[str]]]:
    run = args.out_dir / "runs" / args.root
    release, base, cfg = run / "reference_space", run / "discovery", run / "analysis.yaml"
    ana = base / "analysis"
    common = ["--analysis-config", cfg, "--roots", args.root]

    def cli(module, *values):
        return [sys.executable, "-m", "run_scripts." + module] + [str(v) for v in values]
    return [
        ("coordinates", cli("analysis.prepare_analysis_space", "--release-dir", release, "--out-base-dir", base, *common)),
        ("select", cli("analysis.select_subtype_parameters", "--ir-base-dir", base,
                       "--out-dir", ana / "parameters", "--stage", "all", *common)),
        ("assess", cli("analysis.select_subtype_parameters", "--ir-base-dir", base,
                       "--out-dir", ana / "parameters", "--stage", "assess", *common)),
        ("signatures", cli("analysis.extract_subtype_signatures", "--source-dir", args.atlas_root / "source_file",
                           "--release-dir", release, "--assignments-dir", ana / "parameters" / "assessment_partitions",
                           "--chr44-profiles", args.atlas_root / "source_derived/sample_chr44_profiles.parquet",
                           "--out-dir", ana / "signatures" / "provisional", *common)),
        ("refine", cli("analysis.refine_subtype_contexts", "--parameter-dir", ana / "parameters",
                       "--provisional-partitions-dir", ana / "parameters" / "assessment_partitions",
                       "--provisional-signatures-dir", ana / "signatures" / "provisional",
                       "--ir-dir", ana / "ir", "--out-dir", ana / "refinement", *common)),
        ("anchors", cli("analysis.build_reference_contexts", "--refinement-dir", ana / "refinement",
                        "--out-dir", ana / "reference_contexts", *common)),
        ("calibrate", cli("analysis.project_reference_contexts", "--reference-contexts-dir", ana / "reference_contexts",
                          "--ir-dir", ana / "ir", "--selected-parameters", ana / "parameters" / "selected_parameters.tsv",
                          "--analysis-config", cfg, "--out-dir", ana / "projection" / "non_anchor_recovery")),
    ]


def held_out_ids(split: pd.DataFrame) -> set[str]:
    return set(split.loc[~split["role"].eq("train"), "analysis_id"])


def check_no_held_out(run: Path, held_out: set[str]) -> None:
    paths = list((run / "discovery/analysis/parameters/assessment_partitions").glob("*.parquet"))
    paths += [run / "discovery/analysis/reference_contexts/sample_reference_status.parquet",
              run / "projection_bundle/anchor_coordinates.parquet"]
    for path in paths:
        if path.exists() and set(pd.read_parquet(path, columns=["analysis_id"])["analysis_id"].astype(str)) & held_out:
            raise ValueError(f"held-out record found in a fitted output: {path}")


def build_bundle(args, run: Path) -> None:
    from cnvsim.analysis.config import load_analysis_config
    from cnvsim.config import load_preprocess_config
    from cnvsim.projection import build_projection_bundle
    status = pd.read_parquet(run / "discovery/analysis/reference_contexts/sample_reference_status.parquet")
    anchors = status[status["assignment_status"].eq("anchor_member")]
    if anchors.empty:
        raise ValueError("no stable anchors; no projection bundle is built for this root")
    tables = run / "bundle_inputs" / "tables"
    tables.mkdir(parents=True)
    anchors.rename(columns={"assignment_status": "pipeline_status", "reference_id": "assigned_reference_key"}
                   ).to_parquet(tables / "analysis_assignments.parquet", index=False)
    refs = (anchors.groupby(["root", "reference_id"]).size().rename("n_definition_members").reset_index()
            .rename(columns={"reference_id": "reference_key"}))
    refs["reference_level"] = "subtype"
    refs.to_csv(tables / "reference_contexts.tsv", sep="\t", index=False)
    # The bundle builder reads the fixed gene-interval file below <atlas_dir>/source_file.
    (run / "bundle_inputs" / "source_file").symlink_to(args.atlas_root / "source_file", target_is_directory=True)
    p = load_preprocess_config(run / "preprocess.yaml")
    a = load_analysis_config(run / "analysis.yaml")
    build_projection_bundle(atlas_dir=run / "bundle_inputs", preprocess_release_dir=run / "reference_space",
                            analysis_run_dir=run / "discovery/analysis", release_candidate_dir=tables.parent,
                            out_dir=run / "projection_bundle", low_fraction_threshold=p.encoding.low_frac_threshold,
                            high_fraction_threshold=p.encoding.high_frac_threshold,
                            top_candidates=a.projection.top_candidates, conditional_reference_dir=None)


def cmd_discover(args) -> None:
    import yaml
    from cnvsim.config import load_preprocess_config
    from cnvsim.preprocess.export_reference_space import export_reference_space
    from cnvsim.preprocess.pipeline import (materialize_final_space_from_stage2a, run_firstpass_stage1,
                                            run_stage2a_all)
    hashes = require_hashes(args)
    split = load_split(args)
    run = args.out_dir / "runs" / args.root
    if not (run / "preprocess/features_unfiltered/encode_manifest.tsv").exists():
        raise FileNotFoundError("run the prepare stage first")
    if (run / "RUN.json").exists():
        raise FileExistsError("this root was already discovered; use a new --out-dir")
    shutil.copyfile(REPO / "configs/preprocess_baseline.yaml", run / "preprocess.yaml")
    config = yaml.safe_load((REPO / "configs/analysis_v1.yaml").read_text())
    config.setdefault("reference_contexts", {})["root_only_fallback_roots"] = []  # C47-only full-atlas exception
    (run / "analysis.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    record = {"status": "RUNNING", "root": args.root, "input_sha256": hashes,
              "run_config_sha256": {n: sha256(run / n) for n in ("preprocess.yaml", "analysis.yaml")},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
              "config_change": "analysis root_only_fallback_roots emptied (C47 only); core layer only",
              "reused": "v1 encoded features of training records; first-pass filter and PCA are refitted"}
    write_json(run / "RUN.json", record)
    held_out = held_out_ids(split)
    try:
        cfg = load_preprocess_config(run / "preprocess.yaml")
        roots = [args.root]
        run_firstpass_stage1(cfg=cfg, run_dir=run / "preprocess", roots=roots)
        run_stage2a_all(cfg=cfg, run_dir=run / "preprocess", roots=roots)
        materialize_final_space_from_stage2a(cfg=cfg, run_dir=run / "preprocess", roots=roots)
        export_reference_space(cfg=cfg, run_dir=run / "preprocess", release_dir=run / "reference_space",
                               roots=roots, include_final_space=True)
        for stage, argv in plan(args):
            if stage == "signatures":
                check_no_held_out(run, held_out)
            with (run / f"{stage}.log").open("w") as log:
                subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        build_bundle(args, run)
        check_no_held_out(run, held_out)
        if input_hashes(args) != hashes:
            raise ValueError("inputs changed during discovery")
        record["bundle_sha256"] = {str(p.relative_to(run)): sha256(p)
                                   for p in sorted((run / "projection_bundle").rglob("*")) if p.is_file()}
        record["status"] = "REFERENCE_BUILT"
    except Exception as exc:
        record["status"], record["error"] = "FAILED", str(exc)
        raise
    finally:
        write_json(run / "RUN.json", record)


# --------------------------------------------------------------------------- project

def cmd_project(args) -> None:
    from cnvsim.analysis.reference_projection import classify_reference_queries, score_reference_knn_with_neighbors
    from cnvsim.projection import _transform_features
    hashes = require_hashes(args)
    split = load_split(args)
    run = args.out_dir / "runs" / args.root
    record = json.loads((run / "RUN.json").read_text())
    if record["status"] != "REFERENCE_BUILT":
        raise ValueError("discovery did not finish for this root")
    if record["input_sha256"] != hashes:
        raise ValueError("inputs differ from those used to build this root's reference")
    if record["run_config_sha256"] != {n: sha256(run / n) for n in record["run_config_sha256"]}:
        raise ValueError("run configuration changed after discovery")
    bundle = run / "projection_bundle"
    for rel, digest in record["bundle_sha256"].items():
        if sha256(run / rel) != digest:
            raise ValueError(f"bundle changed after discovery: {rel}")
    manifest = json.loads((bundle / "projection_manifest.json").read_text()) \
        if (bundle / "projection_manifest.json").exists() else json.loads((bundle / "manifest.json").read_text())
    models = pd.read_csv(bundle / manifest["files"]["models"], sep="\t").set_index("root")
    model = models.loc[args.root]
    if str(model["projection_mode"]) != "subtype_knn":
        raise ValueError(f"root projection mode is {model['projection_mode']}; nothing to score")
    roster = split[split["role"].eq("test") & split["root"].eq(args.root)]
    X = pd.read_parquet(args.atlas_root / f"work/preprocess_v1_baseline/features_unfiltered/{args.root}_features.parquet")
    X.index = X.index.astype(str)
    columns = json.loads((bundle / str(model["feature_file"])).read_text())
    missing_cols = [c for c in columns if c not in X.columns]
    missing_rows = sorted(set(roster["analysis_id"]) - set(X.index))
    if missing_cols or missing_rows:
        raise ValueError(f"missing feature columns {missing_cols[:5]} or roster rows {missing_rows[:5]}")
    coords = _transform_features(X.loc[roster["analysis_id"], columns], bundle / str(model["transform_file"]),
                                 int(model["n_pcs"]))
    anchors = pd.read_parquet(bundle / manifest["files"]["anchors"])
    anchors = anchors[anchors["root"].astype(str).eq(args.root)]
    if set(anchors["analysis_id"].astype(str)) & held_out_ids(split):
        raise ValueError("held-out record among bundle anchors")
    dims = list(coords.columns)
    _, scores, _ = score_reference_knn_with_neighbors(
        query_ids=coords.index.astype(str).tolist(), query_values=coords.to_numpy(np.float32),
        reference_ids=anchors["analysis_id"].astype(str).tolist(),
        reference_values=anchors[dims].to_numpy(np.float32),
        reference_labels=anchors["context"].astype(int).to_numpy(), metric=str(model["metric"]),
        n_neighbors=int(model["K"]), top_n=int(manifest["top_candidates"]), exclude_self=False, nearest_n=0)
    calibration = pd.read_csv(bundle / manifest["files"]["calibration"], sep="\t")
    out = classify_reference_queries(scores, calibration[calibration["root"].astype(str).eq(args.root)],
                                     root=args.root).rename(columns={"analysis_id": "sample_id",
                                                                     "assigned_reference_id": "assigned_reference_key"})
    out["root"] = args.root
    out["status_reason"] = np.select(
        [out["assignment_status"].eq("projected_assignment"), out["assignment_status"].eq("ambiguous")],
        ["passes_calibrated_limits", "weak_vote_or_margin"],
        default="outside_calibrated_distance_or_missing_calibration")
    if set(out["sample_id"].astype(str)) != set(roster["analysis_id"]):
        raise ValueError("projection output does not cover the roster exactly")
    target = run / "heldout_projection"
    target.mkdir(exist_ok=False)
    out.to_csv(target / "projection_assignments.tsv", sep="\t", index=False)
    print(out["assignment_status"].value_counts().to_dict())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["hashes", "split", "prepare", "discover", "project"])
    p.add_argument("--atlas-root", type=Path, required=True, help="CNVSIM_atlas_v1 directory")
    p.add_argument("--v1-assignments", type=Path, required=True, help="released atlas/analysis_assignments.parquet")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pool", choices=["conservative", "strict"], default="conservative")
    p.add_argument("--root")
    p.add_argument("--compare", help="hashes stage: INPUT_HASHES.json written on the other machine")
    args = p.parse_args()
    for name in ("atlas_root", "v1_assignments", "out_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.stage in {"prepare", "discover", "project"}:
        if not args.root or args.root in {"C47", "C76", "C80"}:
            raise ValueError("give one analysed root (C47, C76 and C80 are outside this check)")
    sys.path.insert(0, str(REPO))
    {"hashes": cmd_hashes, "split": cmd_split, "prepare": cmd_prepare,
     "discover": cmd_discover, "project": cmd_project}[args.stage](args)


if __name__ == "__main__":
    main()
