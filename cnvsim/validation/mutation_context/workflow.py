"""CLI input preparation and staged analysis with one local manifest."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

from . import association, center_consistency, sensitivity, variant_description, manifest


def configure(files):
    files = {key: Path(value).expanduser().resolve() for key, value in files.items()}
    out = files["output_dir"]
    for key, path in files.items():
        if key != "output_dir" and (out == path or out in path.parents or path in out.parents):
            raise ValueError(f"Output overlaps input {key}")
    for attr, key in {"RAW": "genie_maf", "SAMPLES": "genie_samples", "COVERAGE": "assay_coverage", "ANALYSIS_CONFIG": "analysis_config", "CONTRASTS": "contrasts", "SHAPE": "shape", "FEATURES": "features_dir"}.items():
        setattr(association, attr, files[key])
    for name, mod in [("association", association), ("center_consistency", center_consistency), ("sensitivity", sensitivity), ("variant_description", variant_description)]:
        mod.HERE = out / name
    center_consistency.PRIOR = sensitivity.PRIOR = association.HERE
    variant_description.PRIOR = sensitivity.HERE
    return files


def check_inputs(files):
    for key, path in files.items():
        if key != "output_dir" and not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
    required = {
        "genie_samples": {"analysis_id", "sample_id", "SAMPLE_ID", "selected_patient_root", "reference_layer", "root", "assigned_reference_key", "SEQ_ASSAY_ID", "ONCOTREE_CODE", "has_any_raw_mutation_record", "CENTER", "SAMPLE_TYPE", "SAMPLE_TYPE_DETAILED"},
        "assay_coverage": {"SEQ_ASSAY_ID", "Hugo_Symbol", "atlas_eligible", "assay_exonic_mb"},
        "shape": {"analysis_id", "cnv_fraction"},
    }
    for key, columns in required.items():
        missing = columns - set(pq.read_schema(files[key]).names)
        if missing:
            raise ValueError(f"{key} missing columns: {sorted(missing)}")
    contrasts = pd.read_parquet(files["contrasts"])
    keys = ["root", "ONCOTREE_CODE", "gene", "subtype"]
    if len(contrasts) != 9582 or contrasts.duplicated(keys).any():
        raise ValueError("Expected the complete frozen 9,582-comparison family")
    samples = pd.read_parquet(files["genie_samples"], columns=["selected_patient_root", "has_any_raw_mutation_record"])
    coverage = pd.read_parquet(files["assay_coverage"], columns=["atlas_eligible"])
    for key, series in [(c, samples[c]) for c in samples] + [("atlas_eligible", coverage.atlas_eligible)]:
        if not pd.api.types.is_bool_dtype(series) or series.isna().any():
            raise ValueError(f"{key} must be nonmissing boolean")
    for root in sorted(contrasts.root.unique()):
        p = files["features_dir"] / f"{root}_features.parquet"
        if not p.is_file():
            raise FileNotFoundError(p)
    return {"family_n": len(contrasts), "roots": sorted(contrasts.root.unique()), "status": "SCHEMA_CHECK_PASSED; source-specific reconstruction checks run during preparation"}



def _environment():
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "scipy", "statsmodels", "PyYAML"]}}


def initialize(files):
    files = configure(files)
    check = check_inputs(files)
    out = files["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Initialization needs an empty output directory")
    with manifest.exclusive_run(out):
        if any(p.name != ".mutation.lock" for p in out.iterdir()):
            raise FileExistsError("Run was initialized by another process")
        data = {"schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {k: str(v) for k, v in files.items()},
                "environment": _environment(), "stages": {"check-inputs": {"status": "COMPLETE", "check": check}},
                "records": {}, "exports": {}}
        manifest.write(out, data)
    print(json.dumps({**check, "manifest": str(out / manifest.NAME)}, indent=2))


def execute(stage, run_dir, workers=1, export_dir=None, protein_fasta_dir=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    run = Path(run_dir).expanduser().resolve()
    with manifest.exclusive_run(run):
        data = manifest.read(run)
        if any(x["status"] == "RUNNING" for x in data["stages"].values()):
            raise RuntimeError("An earlier stage is still marked RUNNING; inspect it before proceeding")
        if stage in data["stages"]:
            raise FileExistsError("Stage already attempted; preserve its record and use a fresh run")
        dependencies = {"association-prepare": ["check-inputs"], "association-analyze": ["association-prepare"],
                        "center-consistency": ["association-analyze"], "sensitivity": ["association-analyze"], "variant-description": ["sensitivity"],
                        "export-tables": ["association-analyze", "center-consistency", "sensitivity"]}
        for required in dependencies[stage]:
            if data["stages"].get(required, {}).get("status") != "COMPLETE":
                raise ValueError(f"Complete {required} first")
        files = configure(data["inputs"])
        if files["output_dir"] != run:
            raise ValueError("Run directory differs from its manifest")
        if stage == "variant-description":
            if protein_fasta_dir is None:
                raise ValueError("variant_description requires --protein-fasta-dir")
            variant_description.FASTA_DIR = Path(protein_fasta_dir).expanduser().resolve()
            if run == variant_description.FASTA_DIR or run in variant_description.FASTA_DIR.parents or variant_description.FASTA_DIR in run.parents:
                raise ValueError("Protein inputs overlap the run directory")
            for accession, *_ in variant_description.REF.values():
                path = variant_description.FASTA_DIR / (accession + ".fasta")
                if not path.is_file():
                    raise FileNotFoundError(path)
        elif protein_fasta_dir is not None:
            raise ValueError("Protein FASTA is used only by variant_description")
        if stage != "export-tables":
            folder = run / ("association" if stage.startswith("association-") else stage.replace("-", "_"))
            if stage != "association-analyze":
                folder.mkdir(exist_ok=False)
        manifest.set_stage(run, stage, "RUNNING")
        try:
            if stage == "association-prepare":
                association.prepare()
            elif stage == "association-analyze":
                association.analyze(workers)
            elif stage == "center-consistency":
                center_consistency.run(workers)
            elif stage == "sensitivity":
                sensitivity.run(workers)
            elif stage == "variant-description":
                variant_description.main(refresh=True)
            elif stage == "export-tables":
                from .export import export_tables
                export_tables(run, export_dir)
        except Exception as exc:
            manifest.set_stage(run, stage, "FAILED", error=str(exc))
            raise
        manifest.set_stage(run, stage, "COMPLETE")
