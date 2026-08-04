from __future__ import annotations

"""Create and query an on-disk cache of the large sample-gene CNV table.

Rows are partitioned into 256 deterministic buckets based on ``analysis_id``.
Root-level encoding can then read only buckets containing the requested
samples instead of scanning the full TSV once per root. The MD5 value is used
only for even partitioning, not for data integrity or security.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


COMPACT_STORE_SCHEMA = "cnvsim_gene_cnv_fractions_v1"


def bucket_of(analysis_id: str) -> str:
    """Return the two-hex-character partition key for an analysis ID."""

    h = hashlib.md5(str(analysis_id).encode("utf-8")).hexdigest()
    return h[:2]


def compute_buckets(analysis_ids: Iterable[str]) -> set[str]:
    """Return the cache partitions needed for a collection of analyses."""

    buckets: set[str] = set()
    for a in analysis_ids:
        if a is None:
            continue
        buckets.add(bucket_of(str(a)))
    return buckets


@dataclass(frozen=True)
class GenePanelCacheSpec:
    """Descriptive defaults for the partitioned Parquet cache layout."""

    partition_col: str = "bucket"
    partition_bins: int = 256
    compression: str = "zstd"
    version: str = "v1"


def build_gene_panel_cache(
    *,
    in_tsv: Path,
    out_dir: Path,
    chunksize: int = 2_000_000,
    compression: str = "zstd",
    schema_cols: list[str] | None = None,
) -> dict:
    """Convert a large tab-separated sample-gene table to partitioned Parquet.

    Each input chunk is split by the deterministic analysis-ID bucket and
    written as a separate Parquet part. The returned manifest records the
    source, schema, row count, and partitions written.
    """

    in_tsv = Path(in_tsv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if schema_cols is None:
        schema_cols = [
            "analysis_id",
            "gene_symbol",
            "gene_id",
            "chrom",
            "start",
            "end",
            "dup_frac",
            "del_frac",
            "hldup_frac",
            "hldel_frac",
        ]

    dtype = {
        "analysis_id": str,
        "gene_symbol": str,
        "gene_id": str,
        "chrom": str,
    }

    n_rows = 0
    n_chunks = 0
    written_parts: set[str] = set()

    for ch in pd.read_csv(
        in_tsv,
        sep="\t",
        usecols=schema_cols,
        dtype=dtype,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        n_chunks += 1

        ch["analysis_id"] = ch["analysis_id"].astype(str)
        ch["gene_symbol"] = ch["gene_symbol"].astype(str)

        # Partitioning keeps all rows for an analysis in the same directory.
        ch["bucket"] = ch["analysis_id"].map(bucket_of).astype(str)

        # New chunk files are appended without rewriting existing partitions.
        for part, dfp in ch.groupby("bucket", sort=False):
            part = str(part)
            part_dir = out_dir / f"bucket={part}"
            part_dir.mkdir(parents=True, exist_ok=True)

            out_path = part_dir / f"part-{n_chunks:05d}.parquet"
            dfp.to_parquet(out_path, index=False, compression=str(compression))
            written_parts.add(part)

        n_rows += int(len(ch))

    manifest = {
        "version": "gene_panel_cache_v1",
        "source_tsv": str(in_tsv),
        "out_dir": str(out_dir),
        "n_rows": int(n_rows),
        "n_chunks": int(n_chunks),
        "partition": {"col": "bucket", "method": "md5(analysis_id)[:2]", "n_bins": 256},
        "compression": str(compression),
        "columns": list(schema_cols) + ["bucket"],
        "partitions_written": sorted(written_parts),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_gene_panel_subset_from_cache(
    *,
    cache_dir: Path,
    analysis_ids: Iterable[str],
    gene_symbols: Iterable[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read selected analyses and genes from the partitioned cache.

    Only buckets associated with ``analysis_ids`` are scanned. Rows are then
    filtered to the exact analysis IDs and optional gene symbols in memory.
    """

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")

    ids = pd.Index([str(x) for x in analysis_ids if x is not None], dtype="object")
    if len(ids) == 0:
        return pd.DataFrame()

    buckets = compute_buckets(ids)
    if not buckets:
        return pd.DataFrame()

    keep_genes = None
    if gene_symbols is not None:
        keep_genes = set(str(g) for g in gene_symbols if g is not None)

    dfs: list[pd.DataFrame] = []

    for part in sorted(buckets):
        part_dir = cache_dir / f"bucket={part}"
        if not part_dir.exists():
            continue
        files = sorted(part_dir.glob("*.parquet"))
        for fp in files:
            df = pd.read_parquet(fp, columns=columns)
            if df is None or df.empty:
                continue
            if "analysis_id" in df.columns:
                df["analysis_id"] = df["analysis_id"].astype(str)
            if "gene_symbol" in df.columns:
                df["gene_symbol"] = df["gene_symbol"].astype(str)

            df = df[df["analysis_id"].isin(ids)]
            if keep_genes is not None and "gene_symbol" in df.columns:
                df = df[df["gene_symbol"].isin(keep_genes)]
            if not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    # Drop partition col unless explicitly requested
    if (columns is not None) and ("bucket" not in columns) and ("bucket" in out.columns):
        out = out.drop(columns=["bucket"])
    return out


def _classify_gene_fraction_store(path: Path) -> str | None:
    """Return the supported store type for an exact directory path."""

    manifest_path = path / "manifest.json"
    index_path = path / "analysis_index.parquet"
    if not manifest_path.exists() or not index_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if manifest.get("schema_version") == COMPACT_STORE_SCHEMA:
        return "compact"
    if str(manifest.get("version", "")).startswith("gene_panel_cache_"):
        return "legacy_cache"
    return None


def resolve_gene_fraction_store(source: Path) -> Path:
    """Locate a compact public store or the legacy internal cache.

    ``source`` may point directly to a store, to ``source_file``, to
    ``source_data``, or to the root of an unpacked CNVSIM data release.
    Compact public fractions are preferred when both layouts are present.
    """

    source = Path(source).expanduser().resolve()
    candidates = [
        source,
        source / "gene_cnv_fractions",
        source / "source_file" / "gene_cnv_fractions",
        source / "source_data" / "source_file" / "gene_cnv_fractions",
        source / "gene_panel" / "cache",
        source / "source_file" / "gene_panel" / "cache",
        source / "source_data" / "source_file" / "gene_panel" / "cache",
    ]
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if _classify_gene_fraction_store(candidate) is not None:
            return candidate
    checked = "\n  ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        "No supported gene-CNV fraction store was found. Checked:\n  " + checked
    )


def load_gene_panel_subset_from_compact_store(
    *,
    store_dir: Path,
    analysis_ids: Iterable[str],
    gene_symbols: Iterable[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read selected rows from the public root-partitioned Parquet store."""

    store_dir = Path(store_dir).expanduser().resolve()
    if _classify_gene_fraction_store(store_dir) != "compact":
        raise ValueError(f"Not a compact public gene-CNV store: {store_dir}")

    ids = pd.Index(
        [str(value) for value in analysis_ids if value is not None],
        dtype="object",
    ).drop_duplicates()
    if ids.empty:
        return pd.DataFrame(columns=columns)
    keep_ids = set(ids)
    keep_genes = (
        None
        if gene_symbols is None
        else {str(value) for value in gene_symbols if value is not None}
    )

    manifest = json.loads((store_dir / "manifest.json").read_text())
    available = list(manifest.get("columns", []))
    if not available:
        raise ValueError(f"Compact store manifest lacks columns: {store_dir}")
    requested = available if columns is None else list(columns)
    missing_columns = sorted(set(requested) - set(available))
    if missing_columns:
        raise KeyError(
            "Compact gene-CNV store does not contain requested columns: "
            f"{missing_columns}"
        )

    index = pd.read_parquet(
        store_dir / "analysis_index.parquet",
        columns=["analysis_id", "relative_path"],
    )
    index["analysis_id"] = index["analysis_id"].astype(str)
    if index["analysis_id"].duplicated().any():
        raise ValueError("Compact gene-CNV analysis index contains duplicate IDs")
    selected = index[index["analysis_id"].isin(keep_ids)].copy()
    missing_ids = sorted(keep_ids - set(selected["analysis_id"]))
    if missing_ids:
        raise KeyError(
            "Requested analyses are absent from the compact gene-CNV store: "
            f"{missing_ids[:5]}"
        )

    read_columns = list(dict.fromkeys([*requested, "analysis_id", "gene_symbol"]))
    parts: list[pd.DataFrame] = []
    store_root = store_dir.resolve()
    for relative, group in selected.groupby("relative_path", sort=True):
        path = (store_dir / str(relative)).resolve()
        if store_root not in path.parents or not path.exists():
            raise FileNotFoundError(f"Invalid compact-store data path: {relative}")

        selected_for_file = group["analysis_id"].astype(str).tolist()
        all_for_file = index.loc[
            index["relative_path"].eq(relative), "analysis_id"
        ].astype(str)
        filters: list[tuple[str, str, object]] = []
        if len(selected_for_file) != len(all_for_file):
            filters.append(("analysis_id", "in", selected_for_file))
        if keep_genes is not None:
            filters.append(("gene_symbol", "in", sorted(keep_genes)))
        frame = pd.read_parquet(
            path,
            columns=read_columns,
            filters=filters or None,
        )
        frame["analysis_id"] = frame["analysis_id"].astype(str)
        frame["gene_symbol"] = frame["gene_symbol"].astype(str)
        frame = frame[frame["analysis_id"].isin(selected_for_file)]
        if keep_genes is not None:
            frame = frame[frame["gene_symbol"].isin(keep_genes)]
        if not frame.empty:
            parts.append(frame[requested])

    if not parts:
        return pd.DataFrame(columns=requested)
    return pd.concat(parts, ignore_index=True)


def load_gene_panel_subset_from_store(
    *,
    store_dir: Path,
    analysis_ids: Iterable[str],
    gene_symbols: Iterable[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read gene-CNV rows from either supported on-disk layout."""

    resolved = resolve_gene_fraction_store(store_dir)
    store_type = _classify_gene_fraction_store(resolved)
    if store_type == "compact":
        return load_gene_panel_subset_from_compact_store(
            store_dir=resolved,
            analysis_ids=analysis_ids,
            gene_symbols=gene_symbols,
            columns=columns,
        )
    if store_type == "legacy_cache":
        return load_gene_panel_subset_from_cache(
            cache_dir=resolved,
            analysis_ids=analysis_ids,
            gene_symbols=gene_symbols,
            columns=columns,
        )
    raise ValueError(f"Unsupported gene-CNV fraction store: {resolved}")


def load_gene_panel_subset(
    *,
    gene_panel_tsv: Path,
    cache_dir: Path | None,
    analysis_ids: Iterable[str],
    gene_symbols: Iterable[str] | None = None,
    columns: list[str] | None = None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Load selected sample-gene rows from cache or by streaming the source TSV.

    ``cache_dir`` is preferred for repeated root-level reads. If it is ``None``,
    the source TSV is scanned in chunks and only matching rows are retained.
    """

    if cache_dir is not None:
        return load_gene_panel_subset_from_store(
            store_dir=Path(cache_dir),
            analysis_ids=analysis_ids,
            gene_symbols=gene_symbols,
            columns=columns,
        )

    keep_ids = set(str(x) for x in analysis_ids if x is not None)
    keep_genes = None if gene_symbols is None else set(str(g) for g in gene_symbols if g is not None)

    if not keep_ids:
        return pd.DataFrame()

    usecols = columns

    chunks = []
    for ch in pd.read_csv(
        gene_panel_tsv,
        sep="\t",
        dtype={"analysis_id": str, "gene_symbol": str},
        usecols=usecols,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        ch["analysis_id"] = ch["analysis_id"].astype(str)
        if "gene_symbol" in ch.columns:
            ch["gene_symbol"] = ch["gene_symbol"].astype(str)

        sub = ch[ch["analysis_id"].isin(keep_ids)]
        if keep_genes is not None and "gene_symbol" in sub.columns:
            sub = sub[sub["gene_symbol"].isin(keep_genes)]

        if len(sub):
            chunks.append(sub.copy())

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
