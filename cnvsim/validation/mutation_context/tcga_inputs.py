"""TCGA participant selection, strict diagnosis mapping and MC3 outcome states."""
from collections import defaultdict
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ('primary_vial16_pass', 'sensitivity_vial16_pass_or_wga')
POS, NEG, UNKNOWN = 'positive', 'no_qualifying_call', 'unknown'


# Diagnosis rules for the fixed 45 GENIE candidates.
DIAGNOSIS_MAPPING = {
    ('C18_C20', 'COAD'): {
        'root': 'C18_C20', 'diagnosis': 'COAD',
        'status': 'mapped', 'mapping_class': 'exact_morphology_and_ncit_within_project',
        'projects': ['TCGA.COAD'], 'morphologies': ['pgx:icdom-81403'],
        'topography_prefixes': ['pgx:icdot-C18'], 'histology_labels': ['Colon Adenocarcinoma'],
        'ncit_filter': ['NCIT:C4349'], 'topography_include': [], 'topography_exclude': [],
        'note': 'Rectosigmoid (NCIT:C43584) and mucinous 8480/3 excluded.',
    },
    ('C18_C20', 'READ'): {
        'root': 'C18_C20', 'diagnosis': 'READ',
        'status': 'mapped', 'mapping_class': 'exact_morphology_and_ncit_within_project',
        'projects': ['TCGA.READ'], 'morphologies': ['pgx:icdom-81403'],
        'topography_prefixes': ['pgx:icdot-C20'], 'histology_labels': ['Rectal Adenocarcinoma'],
        'ncit_filter': ['NCIT:C9383'], 'topography_include': [], 'topography_exclude': [],
        'note': 'Rectosigmoid (NCIT:C43584) excluded.',
    },
    ('C30_C39', 'LUAD'): {
        'root': 'C30_C39', 'diagnosis': 'LUAD',
        'status': 'mapped', 'mapping_class': 'exact_morphology_within_project',
        'projects': ['TCGA.LUAD'], 'morphologies': ['pgx:icdom-81403'],
        'topography_prefixes': ['pgx:icdot-C34'], 'histology_labels': ['Lung Adenocarcinoma'],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': '',
    },
    ('C50', 'BRCA'): {
        'root': 'C50', 'diagnosis': 'BRCA',
        'status': 'not_mapped', 'mapping_class': 'parent_or_unspecified_code',
        'projects': [], 'morphologies': [],
        'topography_prefixes': [], 'histology_labels': [],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': "GENIE 'Invasive Breast Carcinoma' is the parent of IDC/ILC etc.",
    },
    ('C50', 'IDC'): {
        'root': 'C50', 'diagnosis': 'IDC',
        'status': 'mapped', 'mapping_class': 'exact_morphology_within_project',
        'projects': ['TCGA.BRCA'], 'morphologies': ['pgx:icdom-85003'],
        'topography_prefixes': ['pgx:icdot-C50'], 'histology_labels': ['Ductal Breast Carcinoma'],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': '',
    },
    ('C60_C63', 'PRAD'): {
        'root': 'C60_C63', 'diagnosis': 'PRAD',
        'status': 'mapped', 'mapping_class': 'exact_morphology_within_project',
        'projects': ['TCGA.PRAD'], 'morphologies': ['pgx:icdom-81403'],
        'topography_prefixes': ['pgx:icdot-C61'], 'histology_labels': ['Prostate Adenocarcinoma'],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': 'Acinar 8550/3 not included.',
    },
    ('C64_C68', 'BLCA'): {
        'root': 'C64_C68', 'diagnosis': 'BLCA',
        'status': 'mapped', 'mapping_class': 'exact_morphology_within_project',
        'projects': ['TCGA.BLCA'], 'morphologies': ['pgx:icdom-81203'],
        'topography_prefixes': ['pgx:icdot-C67'], 'histology_labels': ['Bladder Urothelial Carcinoma'],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': '',
    },
    ('C64_C68', 'CCRCC'): {
        'root': 'C64_C68', 'diagnosis': 'CCRCC',
        'status': 'mapped', 'mapping_class': 'exact_morphology_within_project',
        'projects': ['TCGA.KIRC'], 'morphologies': ['pgx:icdom-83103'],
        'topography_prefixes': ['pgx:icdot-C64'], 'histology_labels': ['Clear Cell Renal Cell Carcinoma'],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': '',
    },
    ('C71', 'GBM'): {
        'root': 'C71', 'diagnosis': 'GBM',
        'status': 'pending_decision', 'mapping_class': 'competing_codes_one_to_many',
        'projects': ['TCGA.GBM'], 'morphologies': ['pgx:icdom-94403'],
        'topography_prefixes': [], 'histology_labels': [],
        'ncit_filter': [], 'topography_include': [], 'topography_exclude': [],
        'note': 'Competing GBM/astrocytoma codes; mapping not decided.',
    },
}


def unique(frame, key, source):
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f'{source}: missing or duplicate {key}')


def select_participants(assignments, metadata, shape, barcode_map):
    """Choose representatives before reading mutation outcomes, including conditional profiles."""
    links = pd.read_csv(barcode_map, sep='\t', dtype=str,
                        usecols=['analysis_id', 'sample_id', 'study_id'])
    unique(links, 'analysis_id', 'TCGA barcode map')
    meta = pd.read_parquet(metadata)
    meta = meta[meta.source_group.eq('TCGA') & meta.platform_id.eq('geo:GPL6801')].copy()
    assign = pd.read_parquet(assignments, columns=['analysis_id', 'root',
                             'assigned_reference_key', 'assignment_method'])
    burden = pd.read_parquet(shape, columns=['analysis_id', 'cnv_fraction'])
    for f, name in [(meta, 'TCGA metadata'), (assign, 'assignments'), (burden, 'FGA')]:
        unique(f, 'analysis_id', name)
    z = meta.merge(assign, on='analysis_id', validate='one_to_one')
    z = z.merge(links, on='analysis_id', how='left', validate='one_to_one')
    z = z.merge(burden, on='analysis_id', how='left', validate='one_to_one')
    if z[['sample_id', 'study_id', 'cnv_fraction']].isna().any().any():
        raise ValueError('TCGA profiles lack barcode, project or FGA')
    # Preserve source barcode suffixes: the primary pairing rule uses the first
    # 16 characters, including a few source records with nonstandard suffixes.
    if not z.sample_id.str.match(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]').all():
        raise ValueError('TCGA source barcodes must contain a valid 16-character vial prefix')
    z['reference_layer'] = z.assignment_method.map(
        {'direct_anchor': 'core', 'direct_conditional_definition': 'conditional'})
    z = z[z.reference_layer.notna()].copy()
    z['participant'] = z.sample_id.str[:12]
    z['vial'] = z.sample_id.str[:16]
    z['primary_priority'] = np.where(z.sample_id.str[13:15].eq('01'), 0, 1)
    z['vial_priority'] = np.where(z.sample_id.str[15].eq('A'), 0, 1)
    z = z.sort_values(['root', 'participant', 'primary_priority', 'vial_priority',
                      'sample_id', 'analysis_id'], kind='stable')
    z = z.drop_duplicates(['root', 'participant'], keep='first')
    # A conditional representative is not replaced by a second core profile.
    z = z[z.reference_layer.eq('core')].copy().reset_index(drop=True)
    z['subtype'] = z.assigned_reference_key
    return z[['analysis_id', 'root', 'participant', 'sample_id', 'vial', 'study_id',
              'subtype', 'cnv_fraction', 'icdo_morphology_id', 'icdo_topography_id',
              'histological_diagnosis_id', 'histological_diagnosis_label']]


def scan_mc3(path, genes, qualifying):
    """Stream MC3 once; retain only barcode documentation and candidate-gene counts."""
    use = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position',
           'Reference_Allele', 'Tumor_Seq_Allele2', 'Variant_Classification',
           'HGVSp_Short', 'Tumor_Sample_Barcode', 'FILTER', 'NCBI_Build']
    documentation, counts, seen, vocabulary = {}, defaultdict(lambda: [0, 0, 0]), set(), set()
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = 0
    with opener(path, 'rt') as handle:
        header = None
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if header is None:
                header = fields
                ix = [header.index(c) for c in use]
                ig, ib, iff = (header.index(c) for c in ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'FILTER'])
                continue
            rows += 1
            gene, barcode, flt = fields[ig], fields[ib], fields[iff]
            vocabulary.add(gene)
            doc = documentation.setdefault(barcode, [False, False])
            doc[0] |= flt == 'PASS'
            doc[1] |= flt in ('PASS', 'wga')
            if gene not in genes:
                continue
            values = tuple(fields[i] for i in ix)
            if values[-1] not in ('GRCh37', '37'):
                raise ValueError('Candidate-gene MC3 call is not GRCh37')
            if values in seen:
                continue
            seen.add(values)
            if values[6] not in qualifying:
                continue
            counts[(barcode, gene)][0 if flt == 'PASS' else 1 if flt == 'wga' else 2] += 1
    if header is None:
        raise ValueError('Empty MC3 input')
    return documentation, dict(counts), vocabulary, rows


def pair_barcodes(parts, documentation):
    """Require one MC3 aliquot per vial; different vials never rescue a primary match."""
    by_vial = defaultdict(list)
    for barcode in documentation:
        by_vial[barcode[:16]].append(barcode)
    if any(len(x) != 1 for x in by_vial.values()):
        raise ValueError('Multiple MC3 aliquots per vial; the primary pairing rule is not applicable')
    z = parts.copy()
    z['matched_mc3_barcode'] = z.vial.map({k: v[0] for k, v in by_vial.items()}).fillna('')
    z['paired'] = z.matched_mc3_barcode.ne('')
    z['has_pass'] = z.matched_mc3_barcode.map(lambda b: documentation.get(b, [False, False])[0])
    z['has_pass_or_wga'] = z.matched_mc3_barcode.map(lambda b: documentation.get(b, [False, False])[1])
    z['mc3_available'] = z.paired & z.has_pass
    return z


def endpoint_state(paired, documented, accepted, uncertain):
    if not paired:
        return UNKNOWN
    if accepted:
        return POS
    if uncertain or not documented:
        return UNKNOWN
    return NEG


def build_states(parts, genes, counts):
    rows = []
    for r in parts.itertuples(index=False):
        for gene in sorted(genes):
            p, w, other = counts.get((r.matched_mc3_barcode, gene), [0, 0, 0])
            rows.append((r.analysis_id, gene,
                         endpoint_state(r.paired, r.has_pass, p, w + other),
                         endpoint_state(r.paired, r.has_pass_or_wga, p + w, other)))
    return pd.DataFrame(rows, columns=['analysis_id', 'gene', *MODES])


def diagnosis_members(parts, rule):
    if rule['status'] != 'mapped':
        return parts.iloc[:0].copy()
    m = (parts.root.eq(rule['root']) & parts.study_id.isin(rule['projects'])
         & parts.icdo_morphology_id.isin(rule['morphologies']))
    if rule.get('ncit_filter'):
        m &= parts.histological_diagnosis_id.isin(rule['ncit_filter'])
    topo = parts.icdo_topography_id.fillna('')
    for field in ('topography_prefixes', 'topography_include'):
        if rule.get(field):
            m &= topo.str.startswith(tuple(rule[field]))
    if rule.get('topography_exclude'):
        m &= ~topo.str.startswith(tuple(rule['topography_exclude']))
    if rule['histology_labels']:
        m &= parts.histological_diagnosis_label.isin(rule['histology_labels'])
    return parts.loc[m].copy()


def select_comparison(parts, states, rule, gene, target, genie_members, minimum=20):
    """Freeze eligible control labels from the primary endpoint only."""
    if rule['status'] != 'mapped':
        reason = 'mapping_pending' if rule['status'] == 'pending_decision' else 'parent_or_unspecified_diagnosis'
        return {'cohort_status': 'unavailable', 'reason': reason}, None
    z = diagnosis_members(parts, rule)
    selected = z.subtype.value_counts()
    available = z.loc[z.mc3_available, 'subtype'].value_counts()
    rest = sorted(set(genie_members) - {target})
    kept = [s for s in rest if available.get(s, 0) >= minimum]
    selected_n, available_n = int(selected.get(target, 0)), int(available.get(target, 0))
    record = {'target_selected_n': selected_n, 'target_mc3_available_n': available_n}
    if available_n < minimum or not kept:
        if selected_n == 0 or not any(selected.get(s, 0) for s in rest):
            reason = 'subtype_or_control_absent'
        elif selected_n >= minimum and any(selected.get(s, 0) >= minimum for s in rest):
            reason = 'mc3_pairing_or_coverage'
        else:
            reason = 'subtype_sample_size_below_20'
        return record | {'cohort_status': 'unavailable', 'reason': reason}, None
    z = z.sort_values('analysis_id', kind='stable').merge(
        states[states.gene.eq(gene)].drop(columns='gene'), on='analysis_id', validate='one_to_one')
    z = z[z.subtype.isin([target, *kept]) & z[MODES[0]].ne(UNKNOWN)].copy()
    counts = z.subtype.value_counts()
    final_rest = [s for s in kept if counts.get(s, 0) >= minimum]
    if counts.get(target, 0) < minimum or not final_rest:
        return record | {'cohort_status': 'unavailable', 'reason': 'filtered_calls_reduce_cohort'}, None
    z = z[z.subtype.isin([target, *final_rest])].copy().reset_index(drop=True)
    if z[MODES[1]].eq(UNKNOWN).any():
        raise ValueError('PASS-or-wga must preserve primary evaluable participants')
    z['positive'] = z[MODES[0]].eq(POS)
    z['positive_wga'] = z[MODES[1]].eq(POS)
    z['assigned_reference_key'] = z.subtype
    z['SEQ_ASSAY_ID'] = z.study_id
    order = z.sort_values(['cnv_fraction', 'analysis_id'], kind='stable')
    ranks = pd.Series(np.arange(1, len(order) + 1), index=order.index, dtype=float)
    z['fga_bin'] = pd.qcut(ranks, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).astype(str).reindex(z.index)
    record.update(cohort_status='eligible', reason='', control_subtypes=';'.join(final_rest),
                  comparator_membership='complete' if final_rest == rest else 'reduced')
    return record, z
