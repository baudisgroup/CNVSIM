"""Explicit inputs for the frozen 9,582-comparison GENIE reproduction."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import local_cnv, center, context, variants

INPUT_KEYS = {"genie_maf", "genie_samples", "assay_coverage", "analysis_config", "contrasts", "shape", "features_dir", "protein_fasta_dir", "output_dir"}
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = ROOT / "configs/mutation_context_protocols"


def configure(path):
    """Resolve paths relative to the JSON file, never to a developer workspace."""
    path = Path(path).expanduser().resolve()
    values = json.loads(path.read_text())
    if set(values) != INPUT_KEYS:
        raise ValueError(f"Input keys: missing={sorted(INPUT_KEYS-set(values))}, unexpected={sorted(set(values)-INPUT_KEYS)}")
    files = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected a nonempty path for {key}")
        p = Path(value).expanduser()
        files[key] = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    out = files["output_dir"]
    for key, p in files.items():
        if key == "output_dir":
            continue
        if out == p or out in p.parents or p in out.parents:
            raise ValueError(f"Output must be separate from input {key}")
    local_cnv.HERE = out / "local_cnv"
    for attr, key in {"RAW": "genie_maf", "SAMPLES": "genie_samples", "COVERAGE": "assay_coverage", "ANALYSIS_CONFIG": "analysis_config", "CONTRASTS": "contrasts", "SHAPE": "shape", "FEATURES": "features_dir"}.items():
        setattr(local_cnv, attr, files[key])
    for name, mod in [("local_cnv", local_cnv), ("center", center), ("context", context), ("variants", variants)]:
        mod.HERE = out / name
        mod.PROTOCOL = PROTOCOL_DIR / (name + ".txt")
    center.PRIOR = context.PRIOR = local_cnv.HERE
    variants.PRIOR = context.HERE
    variants.FASTA_DIR = files["protein_fasta_dir"]
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
    for gene, (accession, *_) in variants.REF.items():
        p = files["protein_fasta_dir"] / (accession + ".fasta")
        if not p.is_file():
            raise FileNotFoundError(f"{gene} reference FASTA: {p}")
    return {"family_n": len(contrasts), "roots": sorted(contrasts.root.unique()), "status": "SCHEMA_CHECK_PASSED; source-specific reconstruction checks run during preparation"}


def _code_hashes():
    return {p.name: local_cnv.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))} | {
        "protocol/" + p.name: local_cnv.digest(p) for p in sorted(PROTOCOL_DIR.glob("*.txt"))
    }


def execute(stage, inputs, workers=2):
    if workers < 1:
        raise ValueError("workers must be positive")
    files = configure(inputs)
    check = check_inputs(files)
    if stage == "check-inputs":
        print(json.dumps(check, indent=2))
        return
    out = files["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    config_hash = local_cnv.digest(inputs)
    code_hashes = _code_hashes()
    lock = out / "LOCAL_ONLY_REPRODUCTION_LOCK.json"
    binding = {"input_config_sha256": config_hash, "code_sha256": code_hashes}
    if lock.exists():
        if json.loads(lock.read_text()) != binding:
            raise ValueError("Code/config changed since this reproduction started; use a new output directory")
    else:
        if any(out.iterdir()):
            raise FileExistsError("Output directory must be empty for a new reproduction")
        with lock.open("x") as f:
            json.dump(binding, f, indent=2)
        lock.chmod(0o600)
    stage_dir = out / ("local_cnv" if stage.startswith("local-") else stage)
    if stage == "local-analyze":
        if not (stage_dir / "PREPARED.json").exists():
            raise FileNotFoundError("Run local-prepare first")
    else:
        stage_dir.mkdir(exist_ok=False)
    status_path = stage_dir / (stage + "_STATUS.json")
    if status_path.exists():
        raise FileExistsError("Stage has already been attempted; preserve its audit and use a fresh run")
    local_cnv.save_json(status_path, {"status": "RUNNING", "stage": stage})
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
        else:
            raise ValueError(stage)
        if _code_hashes() != code_hashes or local_cnv.digest(inputs) != config_hash:
            raise RuntimeError("Code or input configuration changed during the stage")
    except Exception as exc:
        local_cnv.save_json(status_path, {"status": "FAILED", "stage": stage, "error": str(exc)})
        raise
    local_cnv.save_json(status_path, {"status": "COMPLETE", "stage": stage, "utc": datetime.now(timezone.utc).isoformat()})
