"""Run GENIE and TCGA mutation-context analyses and export aggregate tables."""
import argparse
from pathlib import Path


STAGE_ALIASES = {'local-prepare': 'association-prepare', 'local-analyze': 'association-analyze', 'center': 'center-consistency', 'context': 'sensitivity', 'variants': 'variant-description'}


def main():
    if not __debug__:
        raise SystemExit("Do not run mutation analyses with -O, -OO or PYTHONOPTIMIZE; scientific consistency checks require assertions.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check-inputs", "association-prepare", "association-analyze", "center-consistency", "sensitivity", "variant-description", "export-tables", "tcga-prepare", "tcga-analyze", *STAGE_ALIASES])
    parser.add_argument("--run-dir", type=Path, help="Existing run directory")
    parser.add_argument("--out-dir", type=Path, help="New directory for check-inputs or export-tables")
    parser.add_argument("--genie-maf", type=Path)
    parser.add_argument("--prepared-dir", type=Path, help="Directory with genie_samples.parquet, genie_assay_gene_coverage.parquet, and analysis_config.json")
    parser.add_argument("--contrasts", type=Path)
    parser.add_argument("--shape", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--protein-fasta-dir", type=Path, help="UniProt FASTA directory for variant-description")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--atlas-dir", type=Path, help="CNVSIM atlas directory for tcga-prepare")
    parser.add_argument("--mc3-maf", type=Path, help="TCGA MC3 v0.2.8 PUBLIC MAF.gz")
    parser.add_argument("--tcga-barcode-map", type=Path, help="TSV with analysis_id, sample_id (TCGA barcode), and study_id")
    args = parser.parse_args()
    args.stage = STAGE_ALIASES.get(args.stage, args.stage)
    tcga_inputs = [args.atlas_dir, args.mc3_maf, args.tcga_barcode_map]
    if args.stage.startswith('tcga-'):
        if any(x is not None for x in [args.genie_maf, args.prepared_dir, args.contrasts, args.shape, args.protein_fasta_dir]):
            parser.error('GENIE input options are not used by TCGA stages')
        from cnvsim.validation.mutation_context import tcga
        if args.stage == 'tcga-prepare':
            if not all(x is not None for x in tcga_inputs) or args.run_dir is None or args.out_dir is not None:
                parser.error('tcga-prepare requires --run-dir, --atlas-dir, --mc3-maf and --tcga-barcode-map; omit --out-dir')
            tcga.prepare(args.run_dir, *tcga_inputs, args.features_dir)
        else:
            if args.run_dir is None or args.out_dir is not None or any(x is not None for x in tcga_inputs + [args.features_dir]):
                parser.error('tcga-analyze requires only --run-dir; use export-tables for aggregate output')
            tcga.analyze(args.run_dir)
        return
    if any(x is not None for x in tcga_inputs):
        parser.error('TCGA input options apply only to tcga-prepare')
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
    if (args.stage == "variant-description") != (args.protein_fasta_dir is not None):
        parser.error("--protein-fasta-dir is required only for variant-description")
    from cnvsim.validation.mutation_context.workflow import execute
    execute(args.stage, args.run_dir, args.workers, args.out_dir, args.protein_fasta_dir)


if __name__ == "__main__":
    main()
