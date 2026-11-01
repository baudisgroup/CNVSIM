"""Export the explicit current-analysis table schema without fitting models."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from . import manifest as run_manifest
from .table_schema import KEY, TABLE_COLUMNS, OVERVIEW_COLUMNS, MODEL_DESCRIPTIONS, describe, select_columns, add_center_directions

PARQUET_TABLES = {'association/conditional_associations.tsv', 'center_consistency/CENTER_RESULTS.tsv',
                  'center_consistency/FOCUS_45_CENTER_COMPARISON.tsv', 'sensitivity/CONTEXT_RESULTS.tsv'}


def ensure_aggregate(frame):
    forbidden = {'analysis_id', 'sample_id', 'patient_id', 'tumor_sample_barcode', 'normal_sample_barcode', 'external_sample_id'}
    if forbidden.intersection(str(c).lower() for c in frame.columns):
        raise ValueError('Individual-level identifier column in export input')
    pattern = re.compile(r'GENIE[-_]|TCGA-[A-Za-z0-9]{2}-|pgxbs-|pgxcs-|/Users/|/home/')
    for col in frame.select_dtypes(include=['object', 'string']):
        if frame[col].dropna().astype(str).str.contains(pattern, regex=True).any():
            raise ValueError(f'Identifier or local path detected in {col}')


def keyed(frame, expected):
    if len(frame) != expected or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError(f'Expected {expected} unique, nonmissing comparison keys')
    return frame.set_index(KEY).sort_index()


def derive_overview(main, sensitivity, centers):
    mi, si = keyed(main, 9582), keyed(sensitivity, 9582)
    if not mi.index.equals(si.index):
        raise ValueError('Association and sensitivity comparison families differ')
    focus = main[main.association_status.eq('robust_residual_association')].copy()
    adjusted = sensitivity[sensitivity.in_focus_45]
    fi, ai, ci = keyed(focus, 45), keyed(adjusted, 45), keyed(centers, 45)
    if not fi.index.equals(ai.index) or not fi.index.equals(ci.index):
        raise ValueError('The three focus-comparison families differ')
    if not np.allclose(fi.primary_or, ci.main_primary_or, rtol=1e-12, atol=0, equal_nan=True):
        raise ValueError('Center comparison does not match the current main analysis')
    focus['adjusted_association'] = np.where(focus.primary_or > 1, 'enriched', np.where(focus.primary_or < 1, 'depleted', 'neutral'))
    columns = [c for c in OVERVIEW_COLUMNS if c in focus]
    result = focus[columns].merge(adjusted[KEY + ['combined_or', 'combined_ci_low', 'combined_ci_high', 'combined_q', 'sensitivity_status']], on=KEY, validate='one_to_one')
    result = result.merge(centers[KEY + ['center_consistency_status', 'MSK_primary_or', 'non_MSK_primary_or', 'heterogeneity_q_45']], on=KEY, validate='one_to_one')
    return select_columns(result, OVERVIEW_COLUMNS).sort_values(KEY, kind='stable')


def readme(frames):
    lines = [
        '# Mutation association results', '',
        'Start with reader/FOCUS_45_OVERVIEW.tsv. Each association row compares mutation odds in one target CNV subtype with other eligible core subtypes within the same root and diagnosis.', '',
        'The pipeline reads only four comparison keys and four cohort counts from the supplied comparison plan. Counts are reconstructed from metadata, panel coverage and qualifying reported MAF calls. Upstream test statistics are neither required nor copied into results.', '',
        'Generation order: association-prepare reconstructs cohorts; association-analyze fits three main models; center-consistency and sensitivity separately follow that analysis; optional variant-description uses the sensitivity sample sets; export-tables joins current results without refitting or recalculating FDR.', '',
        'Target/control counts are unadjusted evaluable counts. Informative counts identify the subset contributing to a model. CENTER_RESULTS counts refer to that center group. Cohort-audit, center-count and sample-type tables have their own grouped counting units.', '',
        'adjusted_association sits next to primary_or: enriched means adjusted OR > 1, depleted means adjusted OR < 1, and neutral means OR = 1. These labels describe mutation odds after conditioning on assay, FGA bin and local gene CNV. They can disagree with the crude proportions in target_mutated/target_n and control_mutated/control_n. They do not describe CNV gain/loss or significance of the combined model. An odds ratio is not a mutation-rate ratio.', '',
        'status=not_estimable means the model did not meet the analysis criteria. Its OR, P and confidence limits are blank. Support counts and failure reasons remain available. A blank is not zero.', '',
        'For the stratified exact-test models, estimable denotes the prespecified support criteria, not a guarantee of a finite CMH effect estimate. Zero cells can give an unbounded Mantel-Haenszel odds ratio (inf) and unavailable confidence limits while the exact P value is still computed. Such boundary estimates are preserved rather than changing the method after seeing results. In this comparison family they occur only in non-significant comparisons (q>=0.05), outside the main 45.', '',
        'Model *_q values use BH FDR across the full 9,582 comparisons, separately for each model and each center group where applicable. Non-estimable comparisons enter with p=1, so their q=1 is bookkeeping, not evidence of no association. heterogeneity_q_45 uses a separate family of 45 interaction tests.', '',
        'The main 45 associations have three estimable models with q<0.05 and matching directions. sensitivity_status distinguishes supported_across_context_checks, supported_with_unestimable_sensitivity, supported_but_not_all_sensitivities, no_detected_adjusted_association, not_estimable, direction_not_estimable and direction_inconsistent. A supported adjusted result must agree with the current main-analysis direction.', '',
        'Center consistency uses subdivisions of the same discovery cohort. non_MSK pools all non-MSK contributors. Direction agreement is not a significance test or independent replication; lack of detected heterogeneity is not proof that effects are equal.', '',
        'Variant descriptions concern selected ESR1, GATA3 and EP300 follow-up cases, not new association tests. main_analysis_samples and adjusted_model_samples are different populations. Carriers count samples with reported variants; variant_observations count sample-allele observations. Carrier counts can overlap between categories.', '',
        'ESR1 categories distinguish Y537/D538 missense variants, other ligand-binding-domain missense variants (311-547), and variants outside that domain. GATA3 categories distinguish splice calls, zinc-finger-2 frameshifts (317-341), predicted truncation/extension and other classes. EP300 categories distinguish truncating/splice calls and missense variants inside/outside the HAT domain (1287-1663). Annotation-unknown records lack verified protein interpretation.', '',
        'Eight tables are exported without variant description, eleven with it. POSITION_COUNTS and PROTEIN_CHANGE_COUNTS remain in the local run only. Raw MAF, sample linkage, sample-level caches and local paths are not exported. These are release candidates; data-package and supplementary use require the same release review.', '',
        '## Models', '', '| Prefix | Meaning |', '|---|---|',
    ]
    lines.extend(f'| {name} | {meaning} |' for name, meaning in MODEL_DESCRIPTIONS.items())
    lines += [
        '', '## Reading the classifications', '',
        '| Field/value | Meaning |', '|---|---|',
        '| association_status: robust_residual_association | All three main models are estimable, have q<0.05, and have matching OR directions. |',
        '| association_status: no_detected_residual_association | The primary model is estimable but does not have q<0.05. |',
        '| association_status: partial_support_unestimable | The primary model has q<0.05, but at least one of its two sensitivity models cannot be estimated. |',
        '| association_status: primary_only_not_robust | The primary model has q<0.05; both sensitivity models are estimable, but at least one does not have q<0.05. |',
        '| association_status: direction_inconsistent | The three main models pass their significance checks but disagree in effect direction. |',
        '| association_status: not_estimable | The primary model does not meet the analysis criteria. |',
        '| sensitivity_status: supported_across_context_checks | Combined, amplitude, quadratic and restricted models are estimable, have q<0.05, and agree in direction with each other and the main analysis. |',
        '| sensitivity_status: supported_with_unestimable_sensitivity | The combined model has q<0.05 and agrees with the main direction; at least one extra check cannot be estimated. This label does not assert that every other check passes. |',
        '| sensitivity_status: supported_but_not_all_sensitivities | The combined model has q<0.05 and agrees with the main direction; the extra checks are estimable, but at least one has q>=0.05. |',
        '| sensitivity_status: no_detected_adjusted_association | The combined model is estimable but has q>=0.05. |',
        '| sensitivity_status: not_estimable | The combined model cannot be estimated. |',
        '| sensitivity_status: direction_not_estimable | An effect required for direction comparison is missing, non-finite or non-positive. |',
        '| sensitivity_status: direction_inconsistent | The combined effect disagrees with the main direction, or all significant extra checks do not agree in direction. |',
        '| center_consistency_status: same_direction_as_pooled | Both center groups have finite positive primary ORs and their directions match the main estimate. |',
        '| center_consistency_status: MSK_only_estimable / non_MSK_only_estimable | Only the named group has an estimable finite positive primary OR for this comparison. |',
        '| center_consistency_status: not_comparable_both_centers | Neither group provides an estimable finite positive primary OR. |',
        '| center_consistency_status: center_direction_discordant / both_opposite_to_pooled | The two center effects disagree with each other / both oppose the main direction. |',
        '| *_center_direction: same / opposite / not_estimable | Compare the two finite, positive, non-neutral OR directions only. Missing, unsupported or neutral directions are not_estimable; this does not test significance. |',
    ]
    lines += ['', '## Tables and fields', '']
    for name, frame in frames.items():
        lines += [f'### {name}', '', f'{len(frame):,} rows; {len(frame.columns)} columns.', '', '| Field | Definition |', '|---|---|']
        lines.extend(f'| {column} | {describe(column)} |' for column in frame)
        lines.append('')
    return '\n'.join(lines)


def export_tables(run_dir, out_dir):
    run, out = Path(run_dir).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    if run == out or run in out.parents or out in run.parents:
        raise ValueError('Export directory must be separate from the analysis run')
    if out.exists():
        raise FileExistsError('Use a new export directory; existing exports are preserved')
    record = run_manifest.read(run)
    for stage in ['association-prepare', 'association-analyze', 'center-consistency', 'sensitivity']:
        if record['stages'].get(stage, {}).get('status') != 'COMPLETE':
            raise ValueError(f'Stage {stage} is not complete')
    variant_status = record['stages'].get('variant-description', {}).get('status')
    if variant_status not in (None, 'COMPLETE'):
        raise ValueError('Variant description was attempted but did not complete')
    frames = {}
    for rel, columns in TABLE_COLUMNS.items():
        if rel.startswith('reader/') or (rel.startswith('variant_description/') and variant_status != 'COMPLETE'):
            continue
        path = (run / rel).with_suffix('.parquet') if rel in PARQUET_TABLES else run / rel
        frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, sep='\t', float_precision='round_trip')
        frames[rel] = select_columns(frame, columns)
        ensure_aggregate(frames[rel])
    centers = frames['center_consistency/CENTER_RESULTS.tsv']
    if len(centers) != 19164 or centers.duplicated(KEY + ['center_group']).any():
        raise ValueError('Expected the complete two-center comparison family')
    frames['reader/FOCUS_45_OVERVIEW.tsv'] = derive_overview(frames['association/conditional_associations.tsv'], frames['sensitivity/CONTEXT_RESULTS.tsv'], frames['center_consistency/FOCUS_45_CENTER_COMPARISON.tsv'])
    ensure_aggregate(frames['reader/FOCUS_45_OVERVIEW.tsv'])
    documentation = readme(frames)
    tcga = record.get('tcga')
    if tcga is not None:
        if tcga['stages'].get('analyze') != 'COMPLETE':
            raise ValueError('Complete the attempted TCGA analysis before exporting')
        from .tcga_export import comparison_table, readme as tcga_readme
        path = run / 'tcga/LOCAL_ONLY_results.parquet'
        frames['tcga/TCGA_MUTATION_COMPARISONS.tsv'] = comparison_table(pd.read_parquet(path))
        documentation += '\n\n' + tcga_readme()
    # Retain historical run filenames above; normalize only public export names.
    public_frames = {}
    for rel, frame in frames.items():
        public_rel = Path(rel).with_name(Path(rel).name.lower()).as_posix()
        documentation = documentation.replace(rel, public_rel)
        public_frames[public_rel] = frame
    out.mkdir(parents=True)
    inventory = []
    for rel, frame in public_frames.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep='\t', index=False)
        inventory.append({'path': rel, 'rows': len(frame), 'columns': {c: str(frame[c].dtype) for c in frame}})
    (out / 'README.md').write_text(documentation)
    data = run_manifest.read(run)
    data['exports'] = {'output_dir': str(out), 'status': 'AGGREGATE_EXPORT_COMPLETE_NOT_PUBLISHED',
                       'statistics_refitted': False, 'variants_included': variant_status == 'COMPLETE', 'files': inventory}
    run_manifest.write(run, data)
    print(json.dumps({'status': data['exports']['status'], 'tables': len(inventory), 'output_dir': str(out)}, indent=2))
