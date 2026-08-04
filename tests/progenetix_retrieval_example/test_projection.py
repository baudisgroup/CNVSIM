from __future__ import annotations

"""Live bycon-to-CNVSIM projection smoke test."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _bycon import prepare_projection_inputs
from cnvsim.projection import project_samples


RUN_NETWORK_TESTS = os.environ.get("CNVSIM_RUN_NETWORK_TESTS") == "1"
CASES = [
    {
        "selected": "yes",
        "resource": "tumour",
        "biosample_id": "pgxbs-kftviyx8",
        "analysis_id": "pgxcs-kftwd47k",
        "icdo_topography_id": "pgx:icdot-C18.9",
        "cnvsim_overlap_status": "core_definition",
        "cnvsim_reference_key": "C18_C20-CTX3",
        "assignment_status": "projected_assignment",
        "assignment_layer": "primary",
        "assigned_reference_key": "C18_C20-CTX3",
    },
    {
        "selected": "yes",
        "resource": "tumour",
        "biosample_id": "pgxbs-kftvk2uk",
        "analysis_id": "pgxcs-kftwohnj",
        "icdo_topography_id": "pgx:icdot-C16.9",
        "cnvsim_overlap_status": "conditional_definition",
        "cnvsim_reference_key": "C16-COND4",
        "assignment_status": "projected_assignment",
        "assignment_layer": "conditional",
        "assigned_reference_key": "C16-COND4",
    },
    {
        "selected": "yes",
        "resource": "tumour",
        "biosample_id": "pgxbs-kftva6gr",
        "analysis_id": "pgxcs-kftvlnr5",
        "icdo_topography_id": "pgx:icdot-C44.9",
        "cnvsim_overlap_status": "source_non_definition",
        "cnvsim_reference_key": "",
        "assignment_status": "unresolved",
        "assignment_layer": "none",
        "assigned_reference_key": None,
    },
]


@unittest.skipUnless(
    RUN_NETWORK_TESTS,
    "set CNVSIM_RUN_NETWORK_TESTS=1 to run public API integration tests",
)
class ProgenetixProjectionTest(unittest.TestCase):
    def test_selected_analyses_project_with_expected_routes(self) -> None:
        data_dir_value = os.environ.get("CNVSIM_DATA")
        if not data_dir_value:
            self.fail("CNVSIM_DATA must point to the public CNVSIM data package")
        bundle_dir = Path(data_dir_value).expanduser().resolve() / "projection_bundle"
        self.assertTrue(bundle_dir.is_dir(), f"Missing projection bundle: {bundle_dir}")

        selection = pd.DataFrame(CASES)
        expected_ids = set(selection["analysis_id"])
        with tempfile.TemporaryDirectory(prefix="cnvsim_bycon_smoke_") as tmp:
            tmp_dir = Path(tmp)
            inputs = prepare_projection_inputs(
                selection=selection,
                out_dir=tmp_dir / "inputs",
                overwrite=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                outputs = project_samples(
                    bundle_dir=bundle_dir,
                    samples_path=inputs["samples"],
                    segments_path=inputs["segments"],
                    out_dir=tmp_dir / "projection",
                    overwrite=True,
                )

            observed = pd.read_csv(outputs["assignments"], sep="\t").set_index(
                "sample_id"
            )
            self.assertEqual(set(observed.index), expected_ids)
            for expected in CASES:
                row = observed.loc[expected["analysis_id"]]
                self.assertEqual(row["assignment_status"], expected["assignment_status"])
                self.assertEqual(row["assignment_layer"], expected["assignment_layer"])
                reference = row["assigned_reference_key"]
                reference = None if pd.isna(reference) else str(reference)
                self.assertEqual(reference, expected["assigned_reference_key"])

            nearest = pd.read_csv(outputs["nearest_reference_samples"], sep="\t")
            self.assertEqual(set(nearest["sample_id"]), expected_ids)
            audit = pd.read_csv(inputs["audit"], sep="\t")
            self.assertEqual(set(audit["analysis_id"]), expected_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
