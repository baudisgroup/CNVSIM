"""Local intermediate jobs as JSON + Parquet, without executable serialization."""
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _encode(value):
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__nonfinite_float__"}:
            if value["__nonfinite_float__"] not in {"nan", "inf", "-inf"}:
                raise ValueError("Invalid non-finite value in local cache")
            return float(value["__nonfinite_float__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def write_jobs(path, jobs):
    """Store specimen-level data locally; callers bind the whole file's SHA256."""
    path = Path(path)
    if not path.name.startswith("LOCAL_ONLY_"):
        raise ValueError("Specimen-level caches require a LOCAL_ONLY_ filename")
    if not jobs:
        raise ValueError("No jobs to cache")
    metadata = []
    for job in jobs:
        metadata.append({
            "job": _encode({k: v for k, v in job.items() if k != "frame"}),
            "dtypes": {c: str(d) for c, d in job["frame"].dtypes.items()},
            "index_names": job["frame"].index.names,
            "columns": list(job["frame"]),
            "n": len(job["frame"]),
        })
    frames = pd.concat([j["frame"] for j in jobs], keys=range(len(jobs)), names=["__mutation_job__"])
    buf = BytesIO()
    frames.to_parquet(buf, index=True)
    # Exclusive creation protects a prior run; the stage retains failed artifacts.
    with path.open("xb") as handle:
        path.chmod(0o600)
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("metadata.json", json.dumps({"format": 1, "jobs": metadata}, allow_nan=False))
            archive.writestr("frames.parquet", buf.getvalue())


def read_jobs(path):
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["frames.parquet", "metadata.json"]:
            raise ValueError("Unexpected local cache members")
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("format") != 1:
            raise ValueError("Unsupported local cache format")
        frames = pd.read_parquet(BytesIO(archive.read("frames.parquet")))
    jobs = []
    for i, record in enumerate(metadata["jobs"]):
        if record["n"]:
            frame = frames.xs(i, level=0).loc[:, record["columns"]].copy()
        else:
            frame = frames.iloc[:0].droplevel(0).loc[:, record["columns"]].copy()
        frame = frame.astype(record["dtypes"])
        frame.index.names = record["index_names"]
        if len(frame) != record["n"]:
            raise ValueError("Local cache row count mismatch")
        jobs.append(_decode(record["job"]) | {"frame": frame})
    if sum(len(j["frame"]) for j in jobs) != len(frames):
        raise ValueError("Unmapped rows in local cache")
    return jobs
