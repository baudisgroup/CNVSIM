from __future__ import annotations

"""Offline tests for the retrieval CLI and Beacon API parameter mapping."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _bycon as bycon_client
import query_candidates as candidate_cli


class QueryCandidatesCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = candidate_cli.build_parser()
        self.required = ["--resource", "tumour", "--out", "candidates.tsv"]

    def test_long_options_preserve_internal_names(self) -> None:
        args = self.parser.parse_args(
            self.required
            + [
                "--filters", "pgx:icdot-C16",
                "--filters", "NCIT:C7541",
                "--biosample-id", "pgxbs-test",
                "--filterLogic", "OR",
                "--includeDescendantTerms", "false",
                "--limit", "20",
                "--skip", "5",
                "--release-dir", "/tmp/release",
                "--overwrite",
            ]
        )
        self.assertEqual(args.filter, ["pgx:icdot-C16", "NCIT:C7541"])
        self.assertEqual(args.biosample_id, ["pgxbs-test"])
        self.assertEqual(args.filter_logic, "OR")
        self.assertIs(args.include_descendants, False)
        self.assertEqual(args.limit, 20)
        self.assertEqual(args.skip, 5)
        self.assertEqual(args.release_dir, "/tmp/release")
        self.assertTrue(args.overwrite)

    def test_descendants_default_to_true(self) -> None:
        args = self.parser.parse_args(self.required)
        self.assertIs(args.include_descendants, True)

    def test_explicit_descendant_values(self) -> None:
        for value, expected in [("true", True), ("false", False), ("FALSE", False)]:
            with self.subTest(value=value):
                args = self.parser.parse_args(
                    self.required + ["--includeDescendantTerms", value]
                )
                self.assertIs(args.include_descendants, expected)

    def test_invalid_descendant_values_are_rejected(self) -> None:
        for value in ["yes", "0", ""]:
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parser.parse_args(
                        self.required + ["--includeDescendantTerms", value]
                    )
                self.assertEqual(error.exception.code, 2)

    def test_descendant_value_is_required(self) -> None:
        for option in ["--includeDescendantTerms", "-includeDescendantTerms"]:
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(self.required + [option])

    def test_single_dash_aliases_preserve_argument_values(self) -> None:
        parsed = []
        for prefix in ["--", "-"]:
            parsed.append(vars(self.parser.parse_args([
                prefix + "resource", "cell-line",
                prefix + "filters", "NCIT:C7541",
                prefix + "filters", "pgx:icdot-C16",
                prefix + "biosample-id", "pgxbs-test",
                prefix + "includeDescendantTerms", "false",
                prefix + "filterLogic", "OR",
                prefix + "limit", "20",
                prefix + "skip", "5",
                prefix + "release-dir", "/tmp/release",
                prefix + "out", "candidates.tsv",
                "--overwrite",
            ])))
        self.assertEqual(parsed[0], parsed[1])
        self.assertIs(parsed[0]["include_descendants"], False)

    def test_long_options_accept_equals_syntax(self) -> None:
        args = self.parser.parse_args(self.required + [
            "--filters=pgx:icdot-C16",
            "--filters=NCIT:C7541",
            "--includeDescendantTerms=false",
            "--filterLogic=OR",
            "--limit=20",
        ])
        self.assertEqual(args.filter, ["pgx:icdot-C16", "NCIT:C7541"])
        self.assertIs(args.include_descendants, False)
        self.assertEqual(args.filter_logic, "OR")
        self.assertEqual(args.limit, 20)

    def test_legacy_option_aliases_remain_accepted(self) -> None:
        for flags in [[], ["--include-descendants"]]:
            with self.subTest(flags=flags):
                args = self.parser.parse_args(
                    [
                        "--resource", "cell-line",
                        "--filter", "NCIT:C7541",
                        "--biosample-id", "pgxbs-test",
                        *flags,
                        "--filter-logic", "OR",
                        "--limit", "10",
                        "--skip", "2",
                        "--release-dir", "/tmp/release",
                        "--out", "candidates.tsv",
                        "--overwrite",
                    ]
                )
                self.assertEqual(args.resource, "cell-line")
                self.assertEqual(args.filter, ["NCIT:C7541"])
                self.assertEqual(args.biosample_id, ["pgxbs-test"])
                self.assertIs(args.include_descendants, True)
                self.assertEqual(args.filter_logic, "OR")
                self.assertEqual(args.limit, 10)
                self.assertEqual(args.skip, 2)
                self.assertEqual(args.release_dir, "/tmp/release")
                self.assertEqual(args.out, "candidates.tsv")
                self.assertTrue(args.overwrite)

    def test_hyphenated_single_dash_filter_logic_alias(self) -> None:
        args = self.parser.parse_args(self.required + ["-filter-logic", "OR"])
        self.assertEqual(args.filter_logic, "OR")

    def test_main_passes_boolean_to_query_and_writes_candidates(self) -> None:
        rows = pd.DataFrame(
            [{"analysis_id": "pgxcs-test", "retrieval_status": "ready"}]
        )
        cases = [
            ([], True),
            (["--includeDescendantTerms", "true"], True),
            (["--includeDescendantTerms", "false"], False),
        ]
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "candidates.tsv"
                argv = [
                    "query_candidates.py",
                    "--resource", "tumour",
                    "--filters", "pgx:icdot-C16",
                    "--filterLogic", "OR",
                    "--out", str(output),
                    *options,
                ]
                with (
                    patch.object(candidate_cli, "query_candidates", return_value=rows)
                    as query_mock,
                    patch.object(sys, "argv", argv),
                    redirect_stdout(io.StringIO()),
                ):
                    candidate_cli.main()
                kwargs = query_mock.call_args.kwargs
                self.assertIs(kwargs["include_descendants"], expected)
                self.assertEqual(kwargs["filters"], ["pgx:icdot-C16"])
                self.assertEqual(kwargs["filter_logic"], "OR")
                self.assertEqual(
                    pd.read_csv(output, sep="\t")["analysis_id"].tolist(),
                    ["pgxcs-test"],
                )

    def test_nonpositive_limit_fails_before_query(self) -> None:
        with (
            patch.object(sys, "argv", ["query_candidates.py", *self.required, "--limit", "0"]),
            patch.object(candidate_cli, "query_candidates") as query_mock,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            candidate_cli.main()
        query_mock.assert_not_called()

    def test_api_parameter_mapping_for_descendant_values(self) -> None:
        for value in ["true", "false"]:
            with self.subTest(value=value):
                args = self.parser.parse_args(
                    self.required
                    + [
                        "--filters", "pgx:icdot-C16",
                        "--filters", "NCIT:C7541",
                        "--includeDescendantTerms", value,
                        "--filterLogic", "OR",
                    ]
                )
                with patch.object(bycon_client, "request_json", return_value={}) as request:
                    bycon_client.beacon_query(
                        "biosamples",
                        resource=args.resource,
                        filters=args.filter,
                        include_descendants=args.include_descendants,
                        filter_logic=args.filter_logic,
                    )
                params = parse_qs(urlparse(request.call_args.args[0]).query)
                self.assertEqual(params["filters"], ["pgx:icdot-C16,NCIT:C7541"])
                self.assertEqual(params["includeDescendantTerms"], [value])
                self.assertEqual(params["filterLogic"], ["OR"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
