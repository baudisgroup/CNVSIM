from __future__ import annotations

"""Offline contract tests for the four public-facing CNVSIM CLIs."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from run_scripts import project_samples as projection_cli
from run_scripts.applications import query_experimental_models as model_cli
from tests.progenetix_retrieval_example import prepare_projection_inputs as prepare_cli
from tests.progenetix_retrieval_example import query_candidates as candidates_cli


class UserCliTest(unittest.TestCase):
    def test_preferred_option_style(self) -> None:
        for module in [candidates_cli, prepare_cli, projection_cli, model_cli]:
            parser = module.build_parser()
            for action in parser._actions:
                if action.dest == "help" or action.help == argparse.SUPPRESS:
                    continue
                with self.subTest(module=module.__name__, dest=action.dest):
                    preferred = action.option_strings[0]
                    if action.nargs == 0:
                        self.assertTrue(preferred.startswith("--"))
                    else:
                        self.assertTrue(preferred.startswith("-"))
                        self.assertFalse(preferred.startswith("--"))

    def test_projection_aliases_preserve_argument_names(self) -> None:
        parsed = []
        for prefix in ["-", "--"]:
            args = projection_cli.build_parser().parse_args(
                [
                    prefix + "bundle-dir", "bundle",
                    prefix + "samples", "samples.tsv",
                    prefix + "segments", "segments.tsv",
                    prefix + "out-dir", "results",
                    "--overwrite",
                ]
            )
            parsed.append(vars(args))
        self.assertEqual(parsed[0], parsed[1])
        self.assertEqual(
            parsed[0],
            {
                "bundle_dir": "bundle",
                "samples": "samples.tsv",
                "segments": "segments.tsv",
                "out_dir": "results",
                "overwrite": True,
            },
        )

    def test_preparation_aliases_preserve_argument_names(self) -> None:
        for prefix in ["-", "--"]:
            with self.subTest(prefix=prefix):
                args = prepare_cli.build_parser().parse_args(
                    [prefix + "selection", "candidates.tsv", prefix + "out-dir", "inputs"]
                )
                self.assertEqual(args.selection, "candidates.tsv")
                self.assertEqual(args.out_dir, "inputs")
                self.assertFalse(args.overwrite)

    def test_model_lookup_aliases_preserve_argument_names(self) -> None:
        for option in [
            "-cell-line-dir", "--cell-line-dir",
            "-application-dir", "--application-dir",
        ]:
            for prefix in ["-", "--"]:
                with self.subTest(directory_option=option, prefix=prefix):
                    args = model_cli.build_parser().parse_args(
                        [
                            option, "cell_line",
                            prefix + "model-id", "CVCL_7042",
                            prefix + "root", "C16",
                            prefix + "out", "models.tsv",
                            prefix + "analysis-out", "analyses.tsv",
                            "--include-analysis-details",
                        ]
                    )
                    self.assertEqual(args.application_dir, "cell_line")
                    self.assertEqual(args.model_id, "CVCL_7042")
                    self.assertEqual(args.root, "C16")
                    self.assertEqual(args.out, "models.tsv")
                    self.assertEqual(args.analysis_out, "analyses.tsv")
                    self.assertTrue(args.include_analysis_details)
                    self.assertIsNone(args.reference_key)

    def test_subtype_lookup_aliases_preserve_defaults_and_flags(self) -> None:
        for prefix in ["-", "--"]:
            with self.subTest(prefix=prefix):
                parser = model_cli.build_parser()
                required = [
                    prefix + "cell-line-dir", "cell_line",
                    prefix + "reference-key", "C16-COND4",
                ]
                defaults = parser.parse_args(required)
                self.assertEqual(defaults.top_n, 5)
                self.assertFalse(defaults.include_nonrecommended)
                args = parser.parse_args(
                    required + [prefix + "top-n", "3", "--include-nonrecommended"]
                )
                self.assertEqual(args.reference_key, "C16-COND4")
                self.assertEqual(args.top_n, 3)
                self.assertTrue(args.include_nonrecommended)

    def test_model_query_modes_remain_required_and_mutually_exclusive(self) -> None:
        for query in [
            [],
            ["-reference-key", "C16-COND4", "--model-id", "CVCL_7042"],
        ]:
            with self.subTest(query=query), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    model_cli.build_parser().parse_args(
                        ["-cell-line-dir", "cell_line", *query]
                    )

    def test_boolean_flags_do_not_consume_values(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_cli.build_parser().parse_args(
                ["-selection", "candidates.tsv", "-out-dir", "inputs", "--overwrite", "true"]
            )

    def test_projection_main_forwards_paths_and_flag(self) -> None:
        for prefix in ["-", "--"]:
            with (
                self.subTest(prefix=prefix),
                patch.object(
                    sys, "argv",
                    [
                        "project_samples.py",
                        prefix + "bundle-dir", "bundle",
                        prefix + "samples", "samples.tsv",
                        prefix + "segments", "segments.tsv",
                        prefix + "out-dir", "results",
                        "--overwrite",
                    ],
                ),
                patch.object(projection_cli, "project_samples", return_value={}) as project,
                redirect_stdout(io.StringIO()),
            ):
                projection_cli.main()
                project.assert_called_once_with(
                    bundle_dir=Path("bundle"),
                    samples_path=Path("samples.tsv"),
                    segments_path=Path("segments.tsv"),
                    out_dir=Path("results"),
                    overwrite=True,
                )

    def test_preparation_main_forwards_selection_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            selection_path = directory / "candidates.tsv"
            selection = pd.DataFrame([{"selected": "yes", "analysis_id": "pgxcs-test"}])
            selection.to_csv(selection_path, sep="\t", index=False)
            output_frames = {
                "samples": pd.DataFrame([{"sample_id": "pgxcs-test"}]),
                "segments": pd.DataFrame([{"sample_id": "pgxcs-test", "start": 10, "end": 20}]),
                "audit": pd.DataFrame([{"n_invalid_coordinate_segments_skipped": 0}]),
            }
            outputs = {}
            for name, frame in output_frames.items():
                outputs[name] = directory / (name + ".tsv")
                frame.to_csv(outputs[name], sep="\t", index=False)
            with (
                patch.object(
                    sys, "argv",
                    [
                        "prepare_projection_inputs.py",
                        "-selection", str(selection_path),
                        "-out-dir", str(directory / "inputs"),
                        "--overwrite",
                    ],
                ),
                patch.object(
                    prepare_cli, "prepare_projection_inputs", return_value=outputs
                ) as prepare,
                redirect_stdout(io.StringIO()),
            ):
                prepare_cli.main()
            kwargs = prepare.call_args.kwargs
            pd.testing.assert_frame_equal(kwargs["selection"], selection)
            self.assertEqual(kwargs["out_dir"], directory / "inputs")
            self.assertIs(kwargs["overwrite"], True)

    def test_subtype_lookup_main_forwards_rank_and_evidence_filter(self) -> None:
        with (
            patch.object(
                sys, "argv",
                [
                    "query_experimental_models.py",
                    "-cell-line-dir", "cell_line",
                    "-reference-key", "C16-COND4",
                    "-top-n", "3",
                    "--include-nonrecommended",
                ],
            ),
            patch.object(
                model_cli, "query_models_for_context", return_value=pd.DataFrame()
            ) as query,
            redirect_stdout(io.StringIO()),
        ):
            model_cli.main()
        query.assert_called_once_with(
            "cell_line", "C16-COND4", top_n=3, recommended_only=False
        )

    def test_model_lookup_main_forwards_root_and_prints_details(self) -> None:
        with (
            patch.object(
                sys, "argv",
                [
                    "query_experimental_models.py",
                    "-cell-line-dir", "cell_line",
                    "-model-id", "CVCL_7042",
                    "-root", "C16",
                    "--include-analysis-details",
                ],
            ),
            patch.object(
                model_cli, "query_context_for_model",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ) as query,
            redirect_stdout(io.StringIO()) as output,
        ):
            model_cli.main()
        query.assert_called_once_with("cell_line", "CVCL_7042", root="C16")
        self.assertIn("Analysis-level audit", output.getvalue())

    def test_invalid_subtype_only_options_fail_before_query(self) -> None:
        for option in [["-root", "C16"], ["-analysis-out", "analyses.tsv"]]:
            with (
                self.subTest(option=option),
                patch.object(
                    sys, "argv",
                    [
                        "query_experimental_models.py",
                        "-cell-line-dir", "cell_line",
                        "-reference-key", "C16-COND4",
                        *option,
                    ],
                ),
                patch.object(model_cli, "query_models_for_context") as query,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                model_cli.main()
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
