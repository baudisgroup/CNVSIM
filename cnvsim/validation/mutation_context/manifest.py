"""One local JSON record for an entire mutation-analysis run."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

NAME = "mutation_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(run):
    data = json.loads((Path(run) / NAME).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported mutation manifest version")
    return data


def write(run, data):
    run = Path(run)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = run / ".mutation_manifest.tmp"
    with temporary.open("x") as f:
        temporary.chmod(0o600)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, run / NAME)


@contextmanager
def exclusive_run(run):
    lock = Path(run) / ".mutation.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Another stage may be active; preserve the run and inspect its status") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def save_record(stage_dir, name, value):
    stage_dir = Path(stage_dir)
    data = read(stage_dir.parent)
    data["records"].setdefault(stage_dir.name, {})[name] = value
    write(stage_dir.parent, data)


def load_record(stage_dir, name):
    stage_dir = Path(stage_dir)
    return read(stage_dir.parent)["records"][stage_dir.name][name]


def has_record(stage_dir, name):
    stage_dir = Path(stage_dir)
    return name in read(stage_dir.parent)["records"].get(stage_dir.name, {})


def set_stage(run, stage, status, **extra):
    data = read(run)
    data["stages"][stage] = {"status": status, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    write(run, data)


def verify_outputs(run):
    run = Path(run)
    for rel, expected in read(run)["output_sha256"].items():
        if digest(run / rel) != expected:
            raise ValueError(f"Output changed since completion: {rel}")


def record_outputs(run, stage_dir):
    run, stage_dir = Path(run), Path(stage_dir)
    data = read(run)
    for path in sorted(stage_dir.rglob("*")):
        if path.is_file():
            data["output_sha256"][path.relative_to(run).as_posix()] = digest(path)
    write(run, data)
