"""Read aggregate cell-line signature evidence and tumour mutation context."""
from pathlib import Path

import numpy as np
import pandas as pd

from cnvsim.model_signature_evidence import (
    MODEL_EVIDENCE_COLUMNS, SIGNATURE_KEYS, SUBTYPE_COVERAGE_COLUMNS,
)

MUTATION_KEYS = ["root", "diagnosis", "target_subtype", "gene"]
MUTATION_FILES = {
    "genie_mutation_context": "reader/focus_45_overview.tsv",
    "tcga_mutation_context": "tcga/tcga_mutation_comparisons.tsv",
}
COUNT_COLUMNS = {
    "tumour_context_n", "all_matched_models", "recommended_models", "recommended_evaluable_models",
    "recommended_recorded_only", "recommended_not_recorded_only", "recommended_mixed",
    "recommended_unknown_only", "recommended_with_unknown", "n_profiles", "evaluable_profiles",
    "recorded_profiles", "not_recorded_profiles", "unknown_profiles", "target_n", "target_mutated",
    "control_n", "control_mutated", "primary_informative_target_n", "primary_informative_control_n",
    "tcga_target_participants", "tcga_target_mutation_carriers", "tcga_control_participants",
    "tcga_control_mutation_carriers",
}


def resolve_table(directory, filename):
    """Prefer current lowercase names and accept legacy uppercase table names."""
    path = Path(directory) / filename
    if path.is_file():
        return path
    legacy = path.with_name(path.stem.upper() + path.suffix)
    return legacy if legacy.is_file() else path


def _read_table(path, required, keys):
    table = pd.read_csv(path, sep="\t", float_precision="round_trip")
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError(f"{path.name} requires complete, unique keys")
    for column in COUNT_COLUMNS.intersection(table.columns):
        table[column] = table[column].astype("Int64")
        if table[column].lt(0).any():
            raise ValueError(f"{path.name} contains a negative count")
    return table


class ModelSignatureContext:
    """Catalogue-wide coverage and model evidence for an assigned subtype."""

    def __init__(self, directory, signatures, models):
        directory = Path(directory)
        self.coverage = _read_table(resolve_table(directory, "subtype_signature_coverage.tsv"),
                                    SUBTYPE_COVERAGE_COLUMNS, SIGNATURE_KEYS)
        self.evidence = _read_table(resolve_table(directory, "model_signature_evidence.tsv"),
                                    MODEL_EVIDENCE_COLUMNS, [*SIGNATURE_KEYS, "model_id"])
        expected = signatures.set_index(SIGNATURE_KEYS)
        observed = self.coverage.set_index(SIGNATURE_KEYS)
        if set(expected.index) != set(observed.index):
            raise ValueError("Model coverage signature keys differ from the atlas")
        observed = observed.reindex(expected.index)
        if (not np.allclose(expected.context_value, observed.tumour_context_value,
                            rtol=1e-14, atol=1e-15)
                or not expected.n_context.eq(observed.tumour_context_n).all()):
            raise ValueError("Model coverage tumour values differ from the atlas")
        if not self.evidence.is_recommended.isin([True, False]).all():
            raise ValueError("Model evidence is_recommended must contain booleans")
        checks = self.evidence.merge(
            models[["model_id", "root", "assigned_reference_key", "n_profiles", "is_recommended"]],
            on=["model_id", "root"], how="left", validate="many_to_one", suffixes=("", "_catalogue"))
        if (not checks.reference_key.eq(checks.assigned_reference_key).all()
                or not checks.n_profiles.eq(checks.n_profiles_catalogue).all()
                or not checks.is_recommended.eq(checks.is_recommended_catalogue).all()):
            raise ValueError("Model evidence disagrees with the cell-line catalogue")
        if not pd.MultiIndex.from_frame(self.evidence[SIGNATURE_KEYS]).isin(expected.index).all():
            raise ValueError("Model evidence contains unknown signature keys")
        counts = self.evidence.groupby(SIGNATURE_KEYS).size().reindex(observed.index, fill_value=0)
        if not counts.eq(observed.all_matched_models).all():
            raise ValueError("Model evidence rows disagree with coverage counts")
        self.evidence = self.evidence[MODEL_EVIDENCE_COLUMNS]

    def lookup(self, reference_key, eligible_model_ids):
        coverage = self.coverage[self.coverage.reference_key.eq(reference_key)].copy()
        evidence = self.evidence[self.evidence.reference_key.eq(reference_key)]
        eligible = evidence[evidence.model_id.isin(eligible_model_ids)].copy()
        return {
            "status": "available" if len(coverage) else "no_released_signatures",
            "coverage_scope": "All catalogue matches; counts precede query self-exclusion.",
            "model_evidence_scope": "Eligible catalogue models after query self-exclusion; recommendation flags retained.",
            "excluded_self_model_event_rows": len(evidence) - len(eligible),
            "gene_value_note": "Released gene fractions may exceed 1; they are not coverage percentages.",
            "interpretation": "CNV records, not functional validation or cell-line mutation measurements.",
        }, {"subtype_signature_coverage": coverage, "model_signature_evidence": eligible}


class MutationContext:
    """Diagnosis-specific tumour associations; never cell-line mutation calls."""

    def __init__(self, directory, subtypes):
        directory = Path(directory)
        self.tables = {}
        required = {
            "genie_mutation_context": ["target_n", "target_mutated", "control_n", "control_mutated",
                                       "primary_or", "primary_ci_low", "primary_ci_high", "primary_q"],
            "tcga_mutation_context": ["tcga_status", "shared_control_subtypes", "comparator_membership",
                                      "matched_genie_odds_ratio", "tcga_odds_ratio", "tcga_ci_lower",
                                      "tcga_ci_upper", "tcga_fdr_q", "significant_in_all_three_models"],
        }
        roots = subtypes.set_index("reference_key", verify_integrity=True).root
        for name, filename in MUTATION_FILES.items():
            frame = _read_table(resolve_table(directory, filename), [*MUTATION_KEYS, *required[name]], MUTATION_KEYS)
            if not frame.target_subtype.map(roots).eq(frame.root).all():
                raise ValueError("Mutation subtype/root keys disagree with the atlas")
            self.tables[name] = frame
        keys = [set(map(tuple, table[MUTATION_KEYS].to_numpy())) for table in self.tables.values()]
        if keys[0] != keys[1]:
            raise ValueError("GENIE and TCGA candidate keys differ")

    def lookup(self, reference_key, sample):
        diagnosis = str(sample.get("oncotree_code") or "").strip()
        tables = {name: frame[frame.target_subtype.eq(reference_key)].copy()
                  for name, frame in self.tables.items()}
        available = sorted(tables["genie_mutation_context"].diagnosis.unique().tolist())
        if diagnosis:
            tables = {name: frame[frame.diagnosis.eq(diagnosis)].copy() for name, frame in tables.items()}
        return {
            "status": "available" if any(len(frame) for frame in tables.values()) else "no_applicable_candidates",
            "diagnosis_scope": "exact_supplied_oncotree_code" if diagnosis else "available_diagnoses_not_matched_to_query",
            "supplied_oncotree_code": diagnosis or None,
            "available_diagnoses": available,
            "cell_line_mutation_status": "not_assessed",
            "query_mutation_status": "not_assessed",
            "genie_note": "Within-diagnosis tumour associations; primary q uses the GENIE discovery family.",
            "tcga_note": "Atlas-internal cross-cohort check, with 45-candidate BH adjustment; not independent validation.",
            "comparator_note": "TCGA uses shared control subtypes. matched_genie_odds_ratio uses those controls; GENIE primary_or uses its full controls.",
            "missing_note": "No row or an unavailable estimate does not mean wild type, zero prevalence or no association.",
        }, tables


def optional_context(directory, filenames, factory, *args):
    """Missing optional data is explicit; malformed supplied data raises."""
    directory = Path(directory)
    missing = [name for name in filenames if not resolve_table(directory, name).is_file()]
    if missing:
        return None, {"status": "unavailable", "missing_files": missing}
    return factory(directory, *args), {"status": "not_queried_assignment_not_accepted"}
