from __future__ import annotations

"""Live bycon-to-CNVSIM projection smoke test."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

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


class ProjectionInputSanitizationTest(unittest.TestCase):
    @patch("_bycon.analysis_variants")
    def test_zero_length_segments_are_skipped_and_audited(
        self, variants_mock
    ) -> None:
        analysis_id = "pgxcs-test"
        base = {
            "type": "CopyNumberChange",
            "copyChange": "low-level gain",
        }
        variants_mock.return_value = [
            {
                "variation": {
                    **base,
                    "location": {
                        "chromosome": "1",
                        "sequence_id": "refseq:NC_000001.11",
                        "start": 100,
                        "end": 100,
                    },
                },
                "caseLevelData": [{"analysisId": analysis_id}],
            },
            {
                "variation": {
                    **base,
                    "location": {
                        "chromosome": "1",
                        "sequence_id": "refseq:NC_000001.11",
                        "start": 200,
                        "end": 300,
                    },
                },
                "caseLevelData": [{"analysisId": analysis_id}],
            },
        ]
        selection = pd.DataFrame(
            [
                {
                    "selected": "yes",
                    "resource": "tumour",
                    "biosample_id": "pgxbs-test",
                    "analysis_id": analysis_id,
                    "icdo_topography_id": "C18.9",
                }
            ]
        )
        with tempfile.TemporaryDirectory(prefix="cnvsim-input-") as tmp:
            outputs = prepare_projection_inputs(
                selection=selection,
                out_dir=Path(tmp) / "inputs",
                overwrite=True,
            )
            segments = pd.read_csv(outputs["segments"], sep="\t")
            audit = pd.read_csv(outputs["audit"], sep="\t")
            manifest = json.loads(outputs["manifest"].read_text())

        self.assertEqual(len(segments), 1)
        self.assertEqual(int(segments.iloc[0]["start"]), 200)
        self.assertEqual(int(segments.iloc[0]["end"]), 300)
        self.assertEqual(
            int(audit.iloc[0]["n_invalid_coordinate_segments_skipped"]), 1
        )
        self.assertEqual(
            manifest["counts"]["invalid_coordinate_segment_links_skipped"],
            1,
        )

    @staticmethod
    def _variant(start, end, *, copy_change="low-level gain", case=None):
        variant = {
            "variation": {
                "type": "CopyNumberChange",
                "copyChange": copy_change,
                "location": {
                    "chromosome": "1",
                    "sequenceId": "refseq:NC_000001.11",
                    "start": start,
                    "end": end,
                },
            },
            "caseLevelData": [case if case is not None else {"variantId": "v"}],
        }
        return variant

    @staticmethod
    def _selection(*analysis_ids):
        return pd.DataFrame(
            [
                {
                    "selected": "yes",
                    "resource": "tumour",
                    "biosample_id": "pgxbs-test",
                    "analysis_id": analysis_id,
                    "icdo_topography_id": "C18.9",
                }
                for analysis_id in analysis_ids
            ]
        )

    @patch("_bycon.analysis_variants")
    def test_current_response_without_case_analysis_id(self, variants_mock) -> None:
        variants_mock.return_value = [self._variant(200, 300)]
        with tempfile.TemporaryDirectory(prefix="cnvsim-input-") as tmp:
            outputs = prepare_projection_inputs(
                selection=self._selection("pgxcs-a"),
                out_dir=Path(tmp) / "inputs",
                overwrite=True,
            )
            segments = pd.read_csv(outputs["segments"], sep="\t")
        variants_mock.assert_called_once_with("tumour", "pgxcs-a")
        self.assertEqual(segments["sample_id"].tolist(), ["pgxcs-a"])

    @patch("_bycon.analysis_variants")
    def test_analyses_of_one_biosample_are_not_mixed(self, variants_mock) -> None:
        responses = {
            "pgxcs-a": [self._variant(200, 300)],
            "pgxcs-b": [
                self._variant(400, 500, copy_change="high-level loss"),
                self._variant(600, 700),
            ],
        }
        variants_mock.side_effect = lambda resource, analysis_id: responses[
            analysis_id
        ]
        with tempfile.TemporaryDirectory(prefix="cnvsim-input-") as tmp:
            outputs = prepare_projection_inputs(
                selection=self._selection("pgxcs-a", "pgxcs-b"),
                out_dir=Path(tmp) / "inputs",
                overwrite=True,
            )
            segments = pd.read_csv(outputs["segments"], sep="\t")
        counts = segments.groupby("sample_id").size().to_dict()
        self.assertEqual(counts, {"pgxcs-a": 1, "pgxcs-b": 2})
        b_states = set(segments.loc[segments["sample_id"] == "pgxcs-b", "cnv_state"])
        self.assertEqual(b_states, {"high_loss", "gain"})

    @patch("_bycon.analysis_variants")
    def test_conflicting_case_analysis_id_is_rejected(self, variants_mock) -> None:
        variants_mock.return_value = [
            self._variant(200, 300, case={"analysisId": "pgxcs-other"})
        ]
        with tempfile.TemporaryDirectory(prefix="cnvsim-input-") as tmp:
            with self.assertRaises(ValueError):
                prepare_projection_inputs(
                    selection=self._selection("pgxcs-a"),
                    out_dir=Path(tmp) / "inputs",
                    overwrite=True,
                )


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
