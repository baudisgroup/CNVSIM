"""Reproduce frozen GENIE local-CNV, centre, context and variant-description analyses."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check-inputs", "local-prepare", "local-analyze", "center", "context", "variants", "export-tables"])
    parser.add_argument("--inputs", help="JSON paths; relative paths resolve beside this file")
    parser.add_argument("--run-dir", help="Completed analysis directory; export-tables only")
    parser.add_argument("--out-dir", help="New aggregate export directory; export-tables only")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "export-tables":
        if not args.run_dir or not args.out_dir or args.inputs:
            parser.error("export-tables requires --run-dir and --out-dir, without --inputs")
        from cnvsim.validation.mutation_context.export import export_tables
        export_tables(args.run_dir, args.out_dir)
        return
    if not args.inputs or args.run_dir or args.out_dir:
        parser.error("analysis stages require --inputs; --run-dir/--out-dir are export-only")
    from cnvsim.validation.mutation_context.workflow import execute
    execute(args.stage, args.inputs, args.workers)


if __name__ == "__main__":
    main()
