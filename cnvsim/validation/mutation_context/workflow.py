"""CLI input preparation and staged analysis with one local manifest."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

from . import local_cnv, center, context, variants, manifest


def configure(files):
    files = {key: Path(value).expanduser().resolve() for key, value in files.items()}
    out = files["output_dir"]
    for key, path in files.items():
        if key != "output_dir" and (out == path or out in path.parents or path in out.parents):
            raise ValueError(f"Output overlaps input {key}")
    for attr, key in {"RAW": "genie_maf", "SAMPLES": "genie_samples", "COVERAGE": "assay_coverage", "ANALYSIS_CONFIG": "analysis_config", "CONTRASTS": "contrasts", "SHAPE": "shape", "FEATURES": "features_dir"}.items():
        setattr(local_cnv, attr, files[key])
    for name, mod in [("local_cnv", local_cnv), ("center", center), ("context", context), ("variants", variants)]:
        mod.HERE = out / name
    center.PRIOR = context.PRIOR = local_cnv.HERE
    variants.PRIOR = context.HERE
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
        data = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
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
        dependencies = {"local-prepare": ["check-inputs"], "local-analyze": ["local-prepare"],
                        "center": ["local-analyze"], "context": ["local-analyze"], "variants": ["context"],
                        "export-tables": ["local-analyze", "center", "context"]}
        for required in dependencies[stage]:
            if data["stages"].get(required, {}).get("status") != "COMPLETE":
                raise ValueError(f"Complete {required} first")
        files = configure(data["inputs"])
        if files["output_dir"] != run:
            raise ValueError("Run directory differs from its manifest")
        if stage == "variants":
            if protein_fasta_dir is None:
                raise ValueError("variants requires --protein-fasta-dir")
            variants.FASTA_DIR = Path(protein_fasta_dir).expanduser().resolve()
            if run == variants.FASTA_DIR or run in variants.FASTA_DIR.parents or variants.FASTA_DIR in run.parents:
                raise ValueError("Protein inputs overlap the run directory")
            for accession, *_ in variants.REF.values():
                path = variants.FASTA_DIR / (accession + ".fasta")
                if not path.is_file():
                    raise FileNotFoundError(path)
        elif protein_fasta_dir is not None:
            raise ValueError("Protein FASTA is used only by variants")
        if stage != "export-tables":
            folder = run / ("local_cnv" if stage.startswith("local-") else stage)
            if stage != "local-analyze":
                folder.mkdir(exist_ok=False)
        manifest.set_stage(run, stage, "RUNNING")
        try:
            if stage == "local-prepare":
                local_cnv.prepare()
            elif stage == "local-analyze":
                local_cnv.analyze(workers)
            elif stage == "center":
                center.run(workers)
            elif stage == "context":
                context.run(workers)
            elif stage == "variants":
                variants.main(refresh=True)
            elif stage == "export-tables":
                from .export import export_tables
                export_tables(run, export_dir)
        except Exception as exc:
            manifest.set_stage(run, stage, "FAILED", error=str(exc))
            raise
        manifest.set_stage(run, stage, "COMPLETE")
