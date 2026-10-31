"""Generate the v1 encoded gene events using the existing public encoder."""
import argparse
import hashlib
import json
from pathlib import Path

from cnvsim.config import load_preprocess_config
from cnvsim.preprocess.gene_panel_store import resolve_gene_fraction_store
from cnvsim.preprocess.pipeline import run_encode
from cnvsim.progenetix_reference.preprocess_reference import preprocess_progenetix_reference
from run_scripts.preprocess import _resolve_source_dir, _required_source_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="New LOCAL output directory for metadata and encoded specimen-level features")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/preprocess_baseline.yaml")
    p.add_argument("--roots", help="Comma-separated roots; omit to encode all configured roots")
    args = p.parse_args()
    source = _resolve_source_dir(args.source_dir)
    for role, path in _required_source_paths(source).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}. Use a complete source release; do not fabricate the manifest.")
    out = args.out_dir.expanduser().resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Output overlaps source data")
    cfg = load_preprocess_config(args.config)
    store = resolve_gene_fraction_store(source)
    out.mkdir(parents=True, exist_ok=False)
    roots = [x.strip() for x in args.roots.split(",") if x.strip()] if args.roots else None
    preprocess_progenetix_reference(cfg=cfg, source_dir=source, run_dir=out)
    run_encode(cfg=cfg, run_dir=out, gene_fraction_store=store, roots=roots)
    hashes = {}
    for path in sorted((out / "features_unfiltered").glob("*_features.parquet")):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        hashes[path.name] = h.hexdigest()
    (out / "LOCAL_ONLY_FEATURE_MANIFEST.json").write_text(json.dumps({"role": "v1 encoded event scores, not raw fractions or absolute copy number", "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "features_sha256": hashes}, indent=2) + "\n")
    print(f"Encoded features: {out / 'features_unfiltered'}")


if __name__ == "__main__":
    main()
