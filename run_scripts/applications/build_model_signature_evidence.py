"""Build aggregate cell-line evidence for released subtype CNV signatures."""
from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnvsim.model_signature_evidence import build_cell_line_signature_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory.")
    args = parser.parse_args()
    result = build_cell_line_signature_evidence(args.release_dir, args.out_dir)
    print(f"Complete: {result['signature_rows']} signatures, "
          f"{result['model_signature_evidence_rows']} model-event rows")


if __name__ == "__main__":
    main()
