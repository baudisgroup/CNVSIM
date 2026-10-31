"""Explicit result columns and their English definitions; no statistical fitting."""
import numpy as np
import pandas as pd

KEY = ['root', 'diagnosis', 'target_subtype', 'gene']
COUNTS = ['target_n', 'target_mutated', 'control_n', 'control_mutated']
MODELS = ['primary', 'amplitude', 'continuous']
SENSITIVITY_MODELS = ['matched_baseline', 'density_only', 'type_only', 'combined', 'amplitude', 'quadratic', 'restricted']
INPUT_RENAME = {'ONCOTREE_CODE': 'diagnosis', 'subtype': 'target_subtype',
                'subtype_n': 'target_n', 'subtype_pos': 'target_mutated',
                'other_subtypes_n': 'control_n', 'other_subtypes_pos': 'control_mutated'}
INPUT_COLUMNS = ['root', 'ONCOTREE_CODE', 'subtype', 'gene', 'subtype_n', 'subtype_pos', 'other_subtypes_n', 'other_subtypes_pos']
EFFECT_FIELDS = ['status', 'reason', 'or', 'ci_low', 'ci_high', 'p', 'q']


def model_columns(names, regression=False):
    return [name + '_' + field for name in names
            for field in EFFECT_FIELDS + (['model_n'] if regression else [])]


SUPPORT_COLUMNS = [name + '_' + field for name in ['primary', 'amplitude']
                   for field in ['informative_strata', 'informative_target_n', 'informative_control_n']]
ASSOCIATION_COLUMNS = KEY + COUNTS + ['local_cnv_missing_n'] + model_columns(MODELS) + SUPPORT_COLUMNS + ['continuous_model_n', 'association_status']
CENTER_COLUMNS = KEY + COUNTS + ['center_group', 'in_focus_45', 'local_cnv_missing_n'] + model_columns(MODELS) + SUPPORT_COLUMNS + ['continuous_model_n']
SENSITIVITY_COLUMNS = KEY + COUNTS + ['in_focus_45'] + [group + '_' + field for group in ['complete', 'restricted'] for field in COUNTS] + ['local_cnv_missing_n'] + model_columns(SENSITIVITY_MODELS, regression=True) + ['sensitivity_status']
CENTER_FOCUS_COLUMNS = KEY + ['main_primary_or'] + [center + '_' + field for center in ['MSK', 'non_MSK'] for field in COUNTS + ['primary_status', 'primary_reason', 'primary_or', 'primary_ci_low', 'primary_ci_high', 'primary_q']] + [name + '_center_direction' for name in MODELS] + ['center_consistency_status', 'heterogeneity_status', 'heterogeneity_reason', 'heterogeneity_p', 'heterogeneity_interaction_or', 'heterogeneity_q_45']
OVERVIEW_COLUMNS = KEY + COUNTS + ['primary_informative_target_n', 'primary_informative_control_n', 'primary_or', 'adjusted_association', 'primary_ci_low', 'primary_ci_high', 'primary_q', 'combined_or', 'combined_ci_low', 'combined_ci_high', 'combined_q', 'sensitivity_status', 'center_consistency_status', 'MSK_primary_or', 'non_MSK_primary_or', 'heterogeneity_q_45']
VARIANT_KEY = KEY + ['population', 'group']

# These are the only result tables exposed by export-tables. Detailed protein
# positions/changes remain local run outputs until a separate release decision.
TABLE_COLUMNS = {
    'association/cohort_reconstruction_audit.tsv': ['root', 'diagnosis', 'gene', 'samples', 'comparisons', 'local_gene_available', 'missing_local_cnv'],
    'association/conditional_associations.tsv': ASSOCIATION_COLUMNS,
    'center_consistency/CENTER_RESULTS.tsv': CENTER_COLUMNS,
    'center_consistency/FOCUS_45_CENTER_COMPARISON.tsv': CENTER_FOCUS_COLUMNS,
    'center_consistency/center_sample_counts.tsv': ['root', 'CENTER', 'center_group', 'unique_evaluable_analyses'],
    'sensitivity/CONTEXT_RESULTS.tsv': SENSITIVITY_COLUMNS,
    'sensitivity/sample_type_coverage.tsv': ['SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED', 'unique_analyses'],
    'variant_description/COHORT_COUNTS.tsv': VARIANT_KEY + ['evaluable_samples', 'mutation_carriers', 'variant_observations', 'distinct_genomic_alleles', 'mutation_rate', 'primary_samples', 'metastasis_samples', 'coordinate_unknown_variants'],
    'variant_description/CATEGORY_COUNTS.tsv': VARIANT_KEY + ['category', 'variant_observations', 'carriers', 'total_mutation_carriers', 'total_variant_observations', 'evaluable_samples', 'carrier_fraction_of_mutated', 'carrier_fraction_of_evaluable', 'variant_fraction'],
    'variant_description/DOMAIN_RELATION_COUNTS.tsv': VARIANT_KEY + ['category', 'domain_relation', 'variant_observations', 'carriers'],
    'reader/FOCUS_45_OVERVIEW.tsv': OVERVIEW_COLUMNS,
}


def select_columns(frame, columns):
    """Fail on a missing result column instead of silently inventing an output."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError('Missing result columns: ' + ', '.join(sorted(missing)))
    return frame.loc[:, columns].copy()


def normalize_model_columns(frame, names):
    """Name current-model fields consistently and hide unsupported estimates.

    Only p/OR/CI on non-estimable rows are cleared. The q values keep the full
    declared family (non-estimable rows enter BH with p=1). Support counts stay.
    """
    rename = {}
    for name in names:
        rename.update({name + '_' + a: name + '_' + b for a, b in {
            'strata': 'informative_strata', 'target_informative': 'informative_target_n',
            'control_informative': 'informative_control_n', 'n': 'model_n'}.items()})
    out = frame.rename(columns=rename).copy()
    for name in names:
        bad = out[name + '_status'].ne('estimable')
        for field in ['p', 'or', 'ci_low', 'ci_high']:
            column = name + '_' + field
            if column not in out:
                out[column] = np.nan
            out.loc[bad, column] = np.nan
    return out


DESCRIPTIONS = {
    'root': 'Anatomical cancer group in the atlas.',
    'diagnosis': 'OncoTree diagnosis; target and control are compared within this diagnosis.',
    'target_subtype': 'CNV subtype being compared with the other eligible core subtypes.',
    'gene': 'Gene whose qualifying reported somatic mutations are evaluated.',
    'target_n': 'Number of evaluable target-subtype samples before model-specific support filtering.',
    'target_mutated': 'Number of those target samples with a qualifying reported mutation in this gene.',
    'control_n': 'Number of evaluable samples in the other eligible core subtypes of the same diagnosis.',
    'control_mutated': 'Number of those control samples with a qualifying reported mutation in this gene.',
    'local_cnv_missing_n': 'Number of evaluable samples lacking a gain or loss encoding for this gene.',
    'in_focus_45': 'Whether this comparison belongs to the 45 robust associations identified in the current main analysis.',
    'association_status': 'Joint classification of the primary, amplitude and continuous main-analysis models.',
    'sensitivity_status': 'Joint sensitivity classification, including consistency with the current main-analysis direction.',
    'center_group': 'MSK or the pooled group of all non-MSK contributing centers.',
    'center_consistency_status': 'Whether estimable center-specific primary effects agree with the main-analysis direction.',
    'main_primary_or': 'Primary adjusted odds ratio from the current combined-center main analysis.',
    'adjusted_association': 'Direction of the adjusted primary-model odds ratio: enriched means OR > 1, depleted means OR < 1, and neutral means OR = 1. It can differ from the crude target and control mutation proportions in the count columns, because the model compares samples within the same assay, FGA bin and local CNV state. This label does not describe CNV gain/loss or significance of the combined model.',
    'heterogeneity_status': 'Whether the center-by-target interaction test is estimable.',
    'heterogeneity_reason': 'Reason that the interaction test is not estimable; blank when estimable.',
    'heterogeneity_p': 'Likelihood-ratio P value for the center-by-target interaction.',
    'heterogeneity_interaction_or': 'Exponentiated center-by-target interaction coefficient; it is not a subgroup mutation rate.',
    'heterogeneity_q_45': 'BH-adjusted interaction P value across the 45 main associations, using p=1 for non-estimable tests.',
    'samples': 'Number of evaluable samples in this root/diagnosis/gene cohort.',
    'comparisons': 'Number of target-subtype comparisons constructed from this cohort.',
    'local_gene_available': 'Whether this gene has gain and loss columns in the local CNV encoding.',
    'missing_local_cnv': 'Number of cohort samples with missing local CNV encoding.',
    'CENTER': 'Submitting center recorded in GENIE metadata.',
    'unique_evaluable_analyses': 'Number of distinct evaluable CNV analyses in this root and center.',
    'SAMPLE_TYPE': 'Reported primary, metastatic or other sample-type category.',
    'SAMPLE_TYPE_DETAILED': 'Detailed sample-type annotation supplied in the metadata.',
    'unique_analyses': 'Number of distinct evaluable CNV analyses with these sample-type annotations.',
    'population': 'main_analysis_samples uses the evaluable association cohort; adjusted_model_samples uses the combined-model informative subset.',
    'group': 'Target subtype or other eligible core subtypes (control) within the same diagnosis.',
    'evaluable_samples': 'Number of samples in this population and group.',
    'mutation_carriers': 'Number of samples carrying a qualifying reported mutation in this gene.',
    'variant_observations': 'Number of distinct sample-allele observations; one carrier may contribute several.',
    'distinct_genomic_alleles': 'Number of distinct genomic alleles after removing the sample identifier.',
    'mutation_rate': 'Mutation carriers divided by evaluable samples.',
    'primary_samples': 'Number of samples annotated as Primary.',
    'metastasis_samples': 'Number of samples annotated as Metastasis.',
    'coordinate_unknown_variants': 'Number of variant observations without a usable checked protein coordinate.',
    'category': 'Gene-specific descriptive variant class; definitions are given below.',
    'carriers': 'Number of distinct mutation carriers in this category or category/domain combination.',
    'total_mutation_carriers': 'All mutation carriers in this gene, population and group.',
    'total_variant_observations': 'All variant observations in this gene, population and group.',
    'carrier_fraction_of_mutated': 'Category carriers divided by all mutation carriers in this gene and group.',
    'carrier_fraction_of_evaluable': 'Category carriers divided by all evaluable samples in this group.',
    'variant_fraction': 'Category variant observations divided by all variant observations in this gene and group.',
    'domain_relation': 'Position relative to the annotated domain: before, within, after, overlap or unknown.',
}
MODEL_DESCRIPTIONS = {
    'primary': 'Assay, FGA quintile and binary local-CNV conditioned exact test with a CMH odds ratio.',
    'amplitude': 'Local-CNV amplitude sensitivity model; exact stratification in association/center tables and GLM adjustment in the sensitivity table.',
    'continuous': 'Assay/local-CNV adjusted logistic model with a continuous FGA spline.',
    'matched_baseline': 'Continuous-FGA baseline on the same complete-case informative sample set used for adjusted sensitivity checks.',
    'density_only': 'Matched baseline plus log1p leave-gene-out reported mutation density.',
    'type_only': 'Matched baseline plus primary/metastatic sample type.',
    'combined': 'Matched baseline plus both leave-gene-out mutation density and sample type.',
    'quadratic': 'Combined model plus a squared log1p mutation-density term.',
    'restricted': 'Combined model after excluding density >=20 mutations/Mb or other MMR/polymerase calls.',
}
FIELD_DESCRIPTIONS = {
    'status': 'Model estimability: estimable or not_estimable.',
    'reason': 'Reason the model is not estimable; blank when estimable.',
    'or': 'Adjusted odds ratio for mutation in the target subtype versus control.',
    'ci_low': 'Lower limit of the 95% confidence interval for the adjusted odds ratio.',
    'ci_high': 'Upper limit of the 95% confidence interval for the adjusted odds ratio.',
    'p': 'Unadjusted model P value; blank for a non-estimable comparison.',
    'q': 'BH FDR across all 9,582 comparisons for this model (separately per center group where applicable); non-estimable tests enter with p=1.',
    'model_n': 'Number of informative samples supplied to this regression fit; blank if support screening prevented fitting.',
    'informative_strata': 'Number of strata containing both comparison groups and both mutation outcomes.',
    'informative_target_n': 'Target samples in the informative strata for this model.',
    'informative_control_n': 'Control samples in the informative strata for this model.',
    'center_direction': 'Center-to-center effect direction: same, opposite or not_estimable; this is not a significance test.',
}


def describe(column):
    if column in DESCRIPTIONS:
        return DESCRIPTIONS[column]
    for prefix, label in [('non_MSK_', 'Non-MSK center group: '), ('MSK_', 'MSK center group: '),
                          ('complete_', 'After primary/metastatic and covariate completeness filtering: '),
                          ('restricted_', 'After the additional mutation-density/MMR restriction: ')]:
        if column.startswith(prefix) and column[len(prefix):] in COUNTS:
            return label + describe(column[len(prefix):])
        if prefix in ['MSK_', 'non_MSK_'] and column.startswith(prefix):
            return label + describe(column[len(prefix):])
    for model in sorted(MODEL_DESCRIPTIONS, key=len, reverse=True):
        prefix = model + '_'
        if column.startswith(prefix) and column[len(prefix):] in FIELD_DESCRIPTIONS:
            return model + ': ' + FIELD_DESCRIPTIONS[column[len(prefix):]]
    raise KeyError('Missing field definition: ' + column)


def add_center_directions(centers):
    """Compare two centres' log-OR signs; this is not a significance test.

    same/opposite require two estimable, finite, positive ORs with nonzero
    log-OR signs. Missing or neutral directions are not_estimable. Agreement
    with the pooled result remains in the existing center_consistency_status.
    """
    out = centers.copy()
    for name in ("primary", "amplitude", "continuous"):
        a, b = out[f"MSK_{name}_or"], out[f"non_MSK_{name}_or"]
        valid = out[f"MSK_{name}_status"].eq("estimable") & out[f"non_MSK_{name}_status"].eq("estimable")
        valid &= np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0) & a.ne(1) & b.ne(1)
        direction = pd.Series("not_estimable", index=out.index)
        direction.loc[valid] = np.where(np.sign(np.log(a.loc[valid])).eq(np.sign(np.log(b.loc[valid]))), "same", "opposite")
        out[f"{name}_center_direction"] = direction
    return out
