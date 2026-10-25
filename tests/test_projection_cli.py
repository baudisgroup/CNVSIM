"""Projection CLI, inclusive encoding and frozen-release regression tests."""
from __future__ import annotations

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
from tests.progenetix_retrieval_example import prepare_projection_inputs as prepare_cli
from tests.progenetix_retrieval_example import query_candidates as candidates_cli
from cnvsim.projection import (
    EVENT_COLUMNS,
    _canonical_segments,
    _encode_features,
    segment_gene_fractions,
)
import json
import os
import subprocess
import pyarrow.dataset as ds
from cnvsim.projection import (
    _canonical_segments,
    _encode_features,
    project_gene_profiles,
    segment_gene_fractions,
)


class ProjectionCliTest(unittest.TestCase):
    def test_preferred_option_style(self) -> None:
        for module in [candidates_cli, prepare_cli, projection_cli]:
            parser = module.build_parser()
            for action in parser._actions:
                if action.dest == "help" or action.help == argparse.SUPPRESS:
                    continue
                with self.subTest(module=module.__name__, dest=action.dest):
                    preferred = action.option_strings[0]
                    self.assertTrue(preferred.startswith("--"))

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




    def test_boolean_flags_do_not_consume_values(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_cli.build_parser().parse_args(
                ["--selection", "candidates.tsv", "--out-dir", "inputs", "--overwrite", "true"]
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
                        "--selection", str(selection_path),
                        "--out-dir", str(directory / "inputs"),
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


class SegmentFractionsTest(unittest.TestCase):
    def setUp(self):
        self.genes = pd.DataFrame(
            [("G1", "1", 1000, 2000), ("G2", "2", 1000, 2000)],
            columns=["gene_symbol", "chrom", "start", "end"],
        )

    def fractions(self, rows):
        segments = pd.DataFrame(
            rows, columns=["sample_id", "chromosome", "start", "end", "cnv_state"]
        )
        canonical = _canonical_segments(segments, set(segments.sample_id))
        return segment_gene_fractions(segments=canonical, genes=self.genes)

    def test_high_events_are_also_total_events(self):
        result = self.fractions([
            ("gain", "1", 1000, 2000, "high_gain"),
            ("loss", "1", 1000, 2000, "high_loss"),
        ]).set_index(["sample_id", "gene_symbol"])
        self.assertEqual(result.loc[("gain", "G1"), list(EVENT_COLUMNS)].to_dict(),
                         {"dup_frac": 1.0, "del_frac": 0.0,
                          "hldup_frac": 1.0, "hldel_frac": 0.0})
        self.assertEqual(result.loc[("loss", "G1"), list(EVENT_COLUMNS)].to_dict(),
                         {"dup_frac": 0.0, "del_frac": 1.0,
                          "hldup_frac": 0.0, "hldel_frac": 1.0})

    def test_low_events_do_not_become_high_events(self):
        result = self.fractions([
            ("gain", "1", 1000, 1200, "gain"),
            ("loss", "1", 1000, 1200, "loss"),
        ]).set_index(["sample_id", "gene_symbol"])
        self.assertEqual(result.loc[("gain", "G1"), "dup_frac"], 0.2)
        self.assertEqual(result.loc[("loss", "G1"), "del_frac"], 0.2)
        self.assertEqual(result[["hldup_frac", "hldel_frac"]].to_numpy().sum(), 0.0)

    def test_total_coverage_is_a_union_not_a_sum(self):
        for low, high, total_column, high_column in [
            ("gain", "high_gain", "dup_frac", "hldup_frac"),
            ("loss", "high_loss", "del_frac", "hldel_frac"),
        ]:
            with self.subTest(event=high):
                result = self.fractions([
                    ("S", "1", 1000, 1400, low),
                    ("S", "1", 1200, 1600, high),
                    ("S", "1", 1200, 1600, high),
                    ("S", "1", 1600, 1700, low),
                ]).iloc[0]
                self.assertEqual(result[total_column], 0.7)
                self.assertEqual(result[high_column], 0.4)

    def test_threshold_boundaries_match_inclusive_encoding(self):
        for state, direction in [("high_gain", "gain"), ("high_loss", "loss")]:
            for length, expected in [(99, 0), (100, 1), (199, 1), (200, 3)]:
                with self.subTest(state=state, length=length):
                    result = self.fractions([("S", "1", 1000, 1000 + length, state)])
                    encoded = _encode_features(
                        sample_ids=["S"], gene_fractions=result,
                        feature_columns=["G1__gain", "G1__loss"],
                        low_threshold=0.1, high_threshold=0.2,
                    )
                    self.assertEqual(encoded.loc["S", f"G1__{direction}"], expected)
                    other = "loss" if direction == "gain" else "gain"
                    self.assertEqual(encoded.loc["S", f"G1__{other}"], 0)

    def test_clipping_and_half_open_boundaries(self):
        result = self.fractions([
            ("S", "1", 900, 1200, "high_gain"),
            ("S", "1", 2000, 2200, "high_loss"),
            ("S", "1", 800, 1000, "loss"),
        ]).iloc[0]
        self.assertEqual(result["dup_frac"], 0.2)
        self.assertEqual(result["hldup_frac"], 0.2)
        self.assertEqual(result["del_frac"], 0)
        self.assertEqual(result["hldel_frac"], 0)

    def test_samples_and_chromosomes_remain_separate(self):
        result = self.fractions([
            ("A", "1", 1000, 1200, "high_gain"),
            ("B", "2", 1000, 1300, "high_loss"),
        ]).set_index(["sample_id", "gene_symbol"])
        self.assertEqual(set(result.index), {("A", "G1"), ("B", "G2")})
        self.assertEqual(result.loc[("A", "G1"), "del_frac"], 0)
        self.assertEqual(result.loc[("B", "G2"), "dup_frac"], 0)
        self.assertEqual(result.loc[("B", "G2"), "del_frac"], 0.3)


CELL_IDS = [
    "pgxcs-kftw5l33", "pgxcs-kftw5l3l", "pgxcs-kftw5l92", "pgxcs-kftw5l9j",
    "pgxcs-kftw5laj", "pgxcs-kftw5lfy", "pgxcs-kftw5x7x", "pgxcs-kftw5xfb",
    "pgxcs-kftw6i0z", "pgxcs-kftw8be3",
]


KEYS = ["assignment_status", "assignment_layer", "assigned_reference_key"]


def run_release_regression(release: Path, out: Path) -> dict:
    release = release.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(release / "cell_line/model_projection_profiles.parquet")
    cells = cells.set_index("analysis_id").loc[CELL_IDS].reset_index()
    tumour = pd.DataFrame([
        ["pgxcs-kftwd47k", "C18_C20", "projected_assignment", "primary", "C18_C20-CTX3"],
        ["pgxcs-kftwohnj", "C16", "projected_assignment", "conditional", "C16-COND4"],
        ["pgxcs-kftvlnr5", "C44", "unresolved", "none", None],
    ], columns=["analysis_id", "root", *KEYS])
    expected = pd.concat([tumour, cells[["analysis_id", "root", *KEYS]]], ignore_index=True)
    samples = expected[["analysis_id", "root"]].rename(columns={"analysis_id": "sample_id"})
    samples["genome_build"] = "GRCh38"
    samples.to_csv(out / "samples.tsv", sep="\t", index=False)

    state_map = {
        "gain": "gain", "loss": "loss",
        "low-level gain": "gain", "low-level loss": "loss",
        "high-level gain": "high_gain", "high-level loss": "high_loss",
        "focal genome amplification": "high_gain",
        "complete genomic deletion": "high_loss",
    }
    parts = []
    for path, ids in [
        ("source_data/source_derived/tumour_autosomal_segments.parquet", tumour.analysis_id),
        ("source_data/source_derived/cell_line/cell_line_autosomal_segments.parquet", cells.analysis_id),
    ]:
        frame = ds.dataset(release / path).to_table(
            filter=ds.field("analysis_id").isin(ids.tolist()),
            columns=["analysis_id", "chromosome", "start", "end", "copy_change"],
        ).to_pandas()
        unknown = set(frame.copy_change) - set(state_map)
        if unknown:
            raise ValueError(f"Unexpected segment states: {unknown}")
        frame["cnv_state"] = frame.copy_change.map(state_map)
        parts.append(frame.drop(columns="copy_change").rename(columns={"analysis_id": "sample_id"}))
    segments = pd.concat(parts, ignore_index=True)
    segments.to_csv(out / "segments.tsv", sep="\t", index=False)
    command = [
        sys.executable, "-m", "run_scripts.project_samples",
        "--bundle-dir", str(release / "projection_bundle"),
        "--samples", str(out / "samples.tsv"), "--segments", str(out / "segments.tsv"),
        "--out-dir", str(out / "segments_projection"), "--overwrite",
    ]
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True)
    (out / "cli.log").write_text(process.stdout + process.stderr)
    process.check_returncode()
    observed = pd.read_csv(out / "segments_projection/projection_assignments.tsv", sep="\t")
    comparisons = expected.set_index("analysis_id")[KEYS].fillna("").eq(
        observed.set_index("sample_id")[KEYS].fillna("")
    ).all(axis=1)

    cached_parts = []
    for category, cases in [("tumour", tumour), ("cell_line", cells)]:
        for root, rows in cases.groupby("root"):
            cached_parts.append(ds.dataset(
                release / f"source_data/source_file/gene_cnv_fractions/{category}/{root}.parquet"
            ).to_table(filter=ds.field("analysis_id").isin(rows.analysis_id.tolist())).to_pandas())
    cached = pd.concat(cached_parts, ignore_index=True).rename(columns={"analysis_id": "sample_id"})
    genes = pd.read_csv(release / "projection_bundle/gene_intervals.tsv", sep="\t")
    reconstructed = segment_gene_fractions(
        segments=_canonical_segments(segments, set(samples.sample_id)), genes=genes,
    )
    features = json.loads((release / "projection_bundle/models/features.json").read_text())
    encoded = [_encode_features(
        sample_ids=samples.sample_id.tolist(), gene_fractions=frame,
        feature_columns=features, low_threshold=0.1, high_threshold=0.2,
    ) for frame in [cached, reconstructed]]
    feature_differences = (encoded[0] != encoded[1]).sum(axis=1)
    cached_outputs = project_gene_profiles(
        bundle_dir=release / "projection_bundle", samples=samples,
        gene_fractions=cached, out_dir=out / "cached_projection", overwrite=True,
    )
    cached_observed = pd.read_csv(cached_outputs["assignments"], sep="\t")
    cached_comparisons = expected.set_index("analysis_id")[KEYS].fillna("").eq(
        cached_observed.set_index("sample_id")[KEYS].fillna("")
    ).all(axis=1)
    coordinate_matches = True
    try:
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "segments_projection/query_coordinates.parquet"),
            pd.read_parquet(out / "cached_projection/query_coordinates.parquet"),
            check_exact=True,
        )
    except AssertionError:
        coordinate_matches = False
    report = {
        "release": str(release), "command": command,
        "samples": len(samples), "segments": len(segments),
        "segments_match_expected_assignments": comparisons.to_dict(),
        "cached_match_expected_assignments": cached_comparisons.to_dict(),
        "encoded_feature_differences": feature_differences.astype(int).to_dict(),
        "segment_and_cached_coordinates_identical": coordinate_matches,
        "passed": bool(comparisons.all() and cached_comparisons.all()
                       and feature_differences.eq(0).all() and coordinate_matches),
    }
    (out / "regression_results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class FrozenProjectionTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CNVSIM_RELEASE_DIR"),
                         "Set CNVSIM_RELEASE_DIR to the unpacked frozen data release")
    def test_segment_and_cached_routes_agree_with_frozen_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = run_release_regression(Path(os.environ["CNVSIM_RELEASE_DIR"]), Path(temporary))
            self.assertTrue(report["passed"], json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
