"""Run frozen-family mutation analyses with one mutation_manifest.json."""
import argparse
from pathlib import Path

# GENIE release 17.0 data are obtained by each user under AACR terms:
# https://www.aacr.org/wp-content/uploads/2025/02/17.0-data_guide.pdf
# Do not redistribute GENIE data without express written permission from AACR.
# This repository distributes analysis code, not MAFs, clinical/sample-linkage
# tables or specimen-level caches. LOCAL_ONLY outputs remain local; aggregate
# outputs are publication candidates, not automatically cleared for release.
# Cite the AACR Project GENIE Consortium (Cancer Discovery, 2017) and release 17.0.


def main():
    if not __debug__:
        raise SystemExit("Do not run mutation analyses with -O, -OO or PYTHONOPTIMIZE; scientific consistency checks require assertions.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check-inputs", "local-prepare", "local-analyze", "center", "context", "variants", "export-tables"])
    parser.add_argument("--run-dir", type=Path, help="Initialized run directory (all stages after check-inputs)")
    parser.add_argument("--out-dir", type=Path, help="New run directory for check-inputs, or new export directory for export-tables")
    parser.add_argument("--genie-maf", type=Path)
    parser.add_argument("--prepared-dir", type=Path, help="Contains genie_samples.parquet, genie_assay_gene_coverage.parquet and analysis_config.json")
    parser.add_argument("--contrasts", type=Path)
    parser.add_argument("--shape", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--protein-fasta-dir", type=Path, help="Required only for optional variants description; not an association-analysis input")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    supplied = [args.genie_maf, args.prepared_dir, args.contrasts, args.shape, args.features_dir]
    if args.stage == "check-inputs":
        if not all(x is not None for x in supplied) or args.out_dir is None or args.run_dir is not None or args.protein_fasta_dir is not None:
            parser.error("check-inputs requires all five input options and --out-dir; omit --run-dir and --protein-fasta-dir")
        from cnvsim.validation.mutation_context.workflow import initialize
        initialize({"genie_maf": args.genie_maf, "genie_samples": args.prepared_dir / "genie_samples.parquet",
                    "assay_coverage": args.prepared_dir / "genie_assay_gene_coverage.parquet",
                    "analysis_config": args.prepared_dir / "analysis_config.json", "contrasts": args.contrasts,
                    "shape": args.shape, "features_dir": args.features_dir,
                    "output_dir": args.out_dir})
        return
    if any(x is not None for x in supplied) or args.run_dir is None:
        parser.error("After initialization, supply --run-dir instead of individual input paths")
    if (args.stage == "export-tables") != (args.out_dir is not None):
        parser.error("--out-dir is required only for check-inputs or export-tables")
    if (args.stage == "variants") != (args.protein_fasta_dir is not None):
        parser.error("--protein-fasta-dir is required only for variants")
    from cnvsim.validation.mutation_context.workflow import execute
    execute(args.stage, args.run_dir, args.workers, args.out_dir, args.protein_fasta_dir)


if __name__ == "__main__":
    main()
