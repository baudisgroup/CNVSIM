"""Project CNV samples and report subtype, model and tumour mutation context."""
from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.reference_query import run_reference_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True,
                        help="samples.tsv; optional oncotree_code filters tumour associations exactly.")
    parser.add_argument("--segments", type=Path, required=True, help="GRCh38, 0-based half-open segments.tsv.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for private query reports.")
    parser.add_argument("--model-signature-dir", type=Path,
                        help="Default: release cell_line/signature_evidence.")
    parser.add_argument("--mutation-dir", type=Path,
                        help="Default: release validation/mutation_context; tumour data, not cell-line mutation calls.")
    parser.add_argument("--nearest", type=int, default=5)
    parser.add_argument("--top-models", type=int, default=5)
    args = parser.parse_args()
    reports = run_reference_query(
        release_dir=args.release_dir, samples_path=args.samples, segments_path=args.segments,
        out_dir=args.out_dir, model_signature_dir=args.model_signature_dir, mutation_dir=args.mutation_dir,
        nearest=args.nearest, top_models=args.top_models)
    for report in reports:
        print(f"{report['sample_id']}: {report['assignment_status']} | "
              f"{report['assigned_reference_key'] or '-'} | {report['report']}")


if __name__ == "__main__":
    main()
