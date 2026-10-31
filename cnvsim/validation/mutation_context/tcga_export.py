"""TCGA table and field definitions for the shared mutation-table exporter."""
import numpy as np
import pandas as pd

from .export import ensure_aggregate
from .table_schema import KEY
from .tcga_inputs import MODES


# Explicit whitelist: do not export rank, likelihood ratios, retention or sample identifiers.
FIELDS = {
    'root': ('root', 'Anatomical cancer root.'),
    'diagnosis': ('diagnosis', 'OncoTree diagnosis used to define the comparison.'),
    'target_subtype': ('target_subtype', 'Target core CNV subtype.'),
    'gene': ('gene', 'Tested gene.'),
    'export_status': ('tcga_status', 'cohort_unavailable, not_estimable or estimable under the diagnosis, pairing, size and conditional-support rules.'),
    'export_reason': ('reason', 'Reason the cohort is unavailable or its adjusted effect cannot be estimated; blank when estimable.'),
    'control_subtypes': ('shared_control_subtypes', 'Control subtype labels used in TCGA and the matched GENIE comparison.'),
    'comparator_membership': ('comparator_membership', 'complete or reduced relative to the original GENIE control subtype set.'),
    'target_n': ('tcga_target_participants', 'Evaluable target participants under the gene-specific PASS endpoint.'),
    'target_pos': ('tcga_target_mutation_carriers', 'Target participants with a qualifying mutation.'),
    'control_n': ('tcga_control_participants', 'Evaluable control participants under the same endpoint.'),
    'control_pos': ('tcga_control_mutation_carriers', 'Control participants with a qualifying mutation.'),
    'genie_common_primary_or': ('matched_genie_odds_ratio', 'GENIE adjusted OR restricted to the shared control subtype labels; original assay and FGA bins are preserved.'),
    'primary_or': ('tcga_odds_ratio', 'TCGA Mantel-Haenszel OR within project, FGA-quintile and binary local gene CNV strata.'),
    'primary_ci_low': ('tcga_ci_lower', 'Lower 95% confidence limit for the TCGA OR.'),
    'primary_ci_high': ('tcga_ci_upper', 'Upper 95% confidence limit for the TCGA OR.'),
    'primary_q_candidate45': ('tcga_fdr_q', 'BH q value across the fixed 45 candidates for the PASS endpoint and binary local-CNV analysis.'),
    'direction_status': ('direction_comparison', 'Direction of estimable TCGA and matched-control GENIE ORs; agreement is separate from significance.'),
    'amplitude_q_candidate45': ('cnv_amplitude_fdr_q', 'BH q across the same 45 candidates for the amplitude analysis.'),
    'continuous_q_candidate45': ('continuous_fga_fdr_q', 'BH q across the same 45 candidates for the continuous-FGA analysis.'),
}


def comparison_table(results):
    primary = results[results['mode'].eq(MODES[0])].copy()
    if len(primary) != 45 or primary.duplicated(KEY).any():
        raise ValueError('The export must retain all 45 candidates')
    unavailable = primary['cohort_status'].eq('unavailable')
    primary['export_status'] = primary['primary_status'].where(~unavailable, 'cohort_unavailable')
    primary['export_reason'] = primary['primary_reason'].where(~unavailable, primary['reason'])
    primary['direction_status'] = primary['direction_status'].replace(
        {'unavailable_original_tcga_cohort': 'tcga_cohort_unavailable'})
    for model in ('primary', 'amplitude', 'continuous', 'genie_common_primary'):
        if model + '_status' not in primary:
            primary[model + '_status'] = 'not_estimable'
        missing = ~primary[model + '_status'].eq('estimable')
        for field in ('or', 'ci_low', 'ci_high', 'p', 'q_candidate45'):
            c = model + '_' + field
            if c in primary:
                primary.loc[missing, c] = np.nan
    table = primary.reindex(columns=FIELDS).rename(columns={k: v[0] for k, v in FIELDS.items()})
    for column in ('tcga_target_participants', 'tcga_target_mutation_carriers',
                   'tcga_control_participants', 'tcga_control_mutation_carriers'):
        table[column] = table[column].astype('Int64')
    table['significant_in_all_three_models'] = primary[[n + '_q_candidate45' for n in ('primary', 'amplitude', 'continuous')]].lt(.05).all(axis=1)
    table = table.sort_values(KEY, kind='stable').reset_index(drop=True)
    ensure_aggregate(table)
    return table


def summarize(results):
    table = comparison_table(results)
    summary = {'candidates': len(table), 'cohort_eligible': int(table.tcga_status.ne('cohort_unavailable').sum()),
        'estimable': int(table.tcga_status.eq('estimable').sum()),
        'same_direction': int(table.direction_comparison.eq('same_direction').sum()),
        'significant_all_three': int(table.significant_in_all_three_models.sum()),
        'columns': len(table.columns)}
    wga = results[results['mode'].eq(MODES[1])].set_index(KEY).sort_index()
    main = results[results['mode'].eq(MODES[0])].set_index(KEY).sort_index()
    summary['pass_or_wga_checks'] = {name: {
        'estimable': int(wga[name + '_status'].eq('estimable').sum()),
        'significant': int(wga[name + '_q_candidate45'].lt(.05).sum()),
        'significance_flags_match_pass': bool(wga[name + '_q_candidate45'].lt(.05).equals(main[name + '_q_candidate45'].lt(.05)))
        } for name in ('primary', 'amplitude', 'continuous')}
    return summary


def readme():
    text = [
        '## TCGA mutation comparisons', '',
        '`tcga/TCGA_MUTATION_COMPARISONS.tsv` retains all 45 GENIE candidates, including unavailable and non-estimable comparisons. It contains aggregate statistics only.', '',
        'Participants are deduplicated within each cancer root before MC3 matching. The primary endpoint requires an unambiguous same-vial match and qualifying PASS mutations. Filter-uncertain calls and unavailable specimens are unknown, not negative. Absence of a qualifying reported call does not establish clinical wild-type status or gene-specific exome callability.', '',
        'TCGA controls are the eligible subtypes with at least 20 evaluable participants. GENIE is restricted to those same subtype labels, preserving its original assay and FGA bins. TCGA uses project, rank-first FGA quintiles and local gene CNV. These analyses do not include the later GENIE mutation-density and sample-type adjustment.', '',
        'Each TCGA model uses a fixed 45-candidate BH family. Unavailable tests contribute p=1 internally and have blank displayed effects and q values. Directional agreement and statistical significance are distinct. TCGA profiles contributed to the CNV atlas; this is a cross-cohort comparison, not an independent test of atlas construction.', '',
        'The table reports the PASS endpoint with CNV-amplitude and continuous-FGA checks. All three models use predefined support and numerical criteria. PASS-or-wga checks are summarized in the run manifest.', '',
        '| Column | Meaning |', '|---|---|']
    text.extend(f'| {name} | {meaning} |' for name, meaning in FIELDS.values())
    text += ['| significant_in_all_three_models | All three PASS-endpoint models have q < 0.05. Direction comparison is reported separately. |', '']
    return '\n'.join(text)
