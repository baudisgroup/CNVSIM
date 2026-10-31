"""Two-stage TCGA follow-up of the fixed 45 GENIE mutation associations."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests

from . import association, manifest
from .cache import read_jobs, write_jobs
from .tcga_inputs import (DIAGNOSIS_MAPPING, MODES, select_participants, scan_mc3, pair_barcodes,
                          build_states, select_comparison)
from .table_schema import KEY


QUALIFYING = {'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
              'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site',
              'Nonstop_Mutation', 'Translation_Start_Site'}


def record(run, data):
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    combined = manifest.read(run.parent)
    combined['tcga'] = data
    manifest.write(run.parent, combined)


def private_parquet(frame, path):
    with path.open('xb') as handle:
        path.chmod(0o600)
        frame.to_parquet(handle, index=False)


def load_candidates(genie_run):
    meta = manifest.read(genie_run)
    for stage in ('association-prepare', 'association-analyze'):
        if meta['stages'].get(stage, {}).get('status') != 'COMPLETE':
            raise ValueError(f'GENIE {stage} must be complete')
    main = pd.read_parquet(genie_run / 'association/conditional_associations.parquet')
    candidates = main[main.association_status.eq('robust_residual_association')].copy()
    if len(candidates) != 45 or candidates[KEY].isna().any().any() or candidates.duplicated(KEY).any():
        raise ValueError('Expected the fixed 45 unique GENIE candidates')
    return candidates.sort_values(KEY, kind='stable'), meta


def prepare(genie_run, atlas_dir, mc3_maf, barcode_map, features_dir=None):
    genie = Path(genie_run).expanduser().resolve()
    run = genie / 'tcga'
    atlas = Path(atlas_dir).expanduser().resolve()
    focus, gm = load_candidates(genie)
    features = Path(features_dir or gm['inputs']['features_dir']).expanduser().resolve()
    files = {'genie_results': genie / 'association/conditional_associations.parquet',
             'genie_cache': genie / 'association/LOCAL_ONLY_jobs.zip',
             'assignments': atlas / 'atlas/analysis_assignments.parquet',
             'metadata': atlas / 'source_data/source_file/metadata/tumour_analysis_metadata.parquet',
             'shape': atlas / 'source_data/source_derived/analysis_shape_proxies.parquet',
             'mc3_maf': Path(mc3_maf).expanduser().resolve(),
             'barcode_map': Path(barcode_map).expanduser().resolve()}
    for root in sorted(focus.root.unique()):
        files['features_' + root] = features / f'{root}_features.parquet'
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if run == path or run in path.parents or path in run.parents:
            raise ValueError('TCGA run directory overlaps an input')
    mapping = DIAGNOSIS_MAPPING
    if not set(zip(focus.root, focus.diagnosis)) <= set(mapping):
        raise ValueError('A candidate diagnosis has no explicit mapping decision')
    with manifest.exclusive_run(genie):
        current = manifest.read(genie)
        if 'tcga' in current or run.exists():
            raise FileExistsError('TCGA was already attempted; preserve the run')
        if any(s.get('status') == 'RUNNING' for s in current['stages'].values()):
            raise RuntimeError('A GENIE stage is marked RUNNING')
        run.mkdir()
        data = {'schema_version': 1, 'inputs': {k: str(v) for k, v in files.items()},
                'stages': {'prepare': 'RUNNING'}}
        record(run, data)
        try:
            parts = select_participants(files['assignments'], files['metadata'], files['shape'], files['barcode_map'])
            documentation, counts, vocabulary, raw_rows = scan_mc3(files['mc3_maf'], set(focus.gene), QUALIFYING)
            if not set(focus.gene) <= vocabulary:
                raise ValueError('A candidate gene is absent from MC3; do not infer negative states')
            parts = pair_barcodes(parts, documentation)
            states = build_states(parts, set(focus.gene), counts)
            private_parquet(parts, run / 'LOCAL_ONLY_participants.parquet')
            private_parquet(states, run / 'LOCAL_ONLY_mutation_states.parquet')
            jobs = {(j['root'], j['diagnosis'], j['gene']): j for j in read_jobs(files['genie_cache'])}
            features_by_root = {}
            for root, sub in focus.groupby('root'):
                path = files['features_' + root]
                names = set(pq.read_schema(path).names)
                wanted = sorted({g + '__' + d for g in sub.gene for d in ('gain', 'loss')} & names)
                features_by_root[root] = pd.read_parquet(path, columns=wanted)
                if not features_by_root[root].index.is_unique:
                    raise ValueError('Duplicate feature analysis_id')
            cases, accounting = [], []
            for row in focus.to_dict('records'):
                k = (row['root'], row['diagnosis'], row['gene'])
                target = row['target_subtype']
                gj = jobs[k]['frame']
                left = gj.assigned_reference_key.eq(target)
                reconstructed = (int(left.sum()), int(gj.loc[left, 'positive'].sum()),
                                 int((~left).sum()), int(gj.loc[~left, 'positive'].sum()))
                if reconstructed != tuple(int(row[c]) for c in ('target_n', 'target_mutated', 'control_n', 'control_mutated')):
                    raise ValueError('GENIE cache and association counts differ')
                meta, tcga = select_comparison(parts, states, mapping[k[:2]], row['gene'], target,
                                               set(gj.assigned_reference_key), 20)
                entry = {c: row[c] for c in KEY} | meta | {'original_genie_or': row['primary_or']}
                accounting.append(entry)
                if tcga is None:
                    cases.append({'record': entry, 'cohort': 'unavailable', 'frame': pd.DataFrame()})
                    continue
                for direction in ('gain', 'loss'):
                    col = row['gene'] + '__' + direction
                    x = features_by_root[row['root']]
                    tcga[direction] = tcga.analysis_id.map(x[col]) if col in x else np.nan
                    if not tcga[direction].dropna().isin([0, 1, 2, 3]).all():
                        raise ValueError('Invalid local CNV encoding')
                labels = [target, *entry['control_subtypes'].split(';')]
                # Preserve GENIE's original assay and FGA bins after restricting labels.
                common = gj[gj.assigned_reference_key.isin(labels)].copy().reset_index(drop=True)
                cases.extend([{'record': entry, 'cohort': 'TCGA', 'frame': tcga},
                              {'record': entry, 'cohort': 'GENIE', 'frame': common}])
            write_jobs(run / 'LOCAL_ONLY_cases.zip', cases)
            data['stages']['prepare'] = 'COMPLETE'
            data['summary'] = {'candidates': len(accounting), 'mc3_rows_read': raw_rows,
                'cohort_eligible': sum(r['cohort_status'] == 'eligible' for r in accounting),
                'unavailable_reasons': pd.Series([r['reason'] for r in accounting if r['cohort_status'] != 'eligible']).value_counts().to_dict()}
            record(run, data)
            print(json.dumps(data['summary'], indent=2))
        except Exception as exc:
            data['stages']['prepare'] = 'FAILED'
            data['error'] = str(exc)
            record(run, data)
            raise


def analyze_frame(z, target):
    left = z.assigned_reference_key.eq(target)
    target_n, control_n = int(left.sum()), int((~left).sum())
    out = dict(target_n=target_n, control_n=control_n, target_pos=int(z.loc[left, 'positive'].sum()),
               control_pos=int(z.loc[~left, 'positive'].sum()), local_cnv_missing_n=int(z[['gain', 'loss']].isna().any(axis=1).sum()))
    for name in ('primary', 'amplitude', 'continuous'):
        out.update({name + '_status': 'not_estimable', name + '_reason': 'local_cnv_missing',
                    **{name + '_' + field: np.nan for field in ('p', 'or', 'ci_low', 'ci_high')}})
    complete = z.dropna(subset=['gain', 'loss']).copy()
    if complete.empty:
        return out
    for name, mode in [('primary', 'binary'), ('amplitude', 'amplitude')]:
        result, info, _ = association.exact_result(complete, target, mode)
        good = association.support(result, target_n)
        out.update({name + '_' + k: v for k, v in result.items()})
        out.update({name + '_status': 'estimable' if good else 'not_estimable',
                    name + '_reason': '' if good else 'insufficient_common_support',
                    name + '_target_retention': result['target_informative'] / target_n,
                    name + '_control_retention': result['control_informative'] / control_n})
        if name == 'primary':
            primary_info = info
    if out['primary_status'] == 'estimable':
        knots = np.quantile(z.cnv_fraction, [.05, .35, .65, .95])
        result = association.continuous(complete.iloc[np.flatnonzero(primary_info)], target, knots)
        out.update({'continuous_' + k: v for k, v in result.items()})
    else:
        out['continuous_reason'] = 'primary_common_support_insufficient'
    return out


def direction(value):
    return int(np.sign(np.log(value))) if np.isfinite(value) and value > 0 else None


def score(cases):
    grouped = {}
    for case in cases:
        key = tuple(case['record'][k] for k in KEY)
        grouped.setdefault(key, {})[case['cohort']] = case
    rows = []
    for _, cohorts in sorted(grouped.items()):
        entry = next(iter(cohorts.values()))['record']
        common = dict(entry)
        gs = analyze_frame(cohorts['GENIE']['frame'], entry['target_subtype']) if 'GENIE' in cohorts else {}
        common.update({'genie_common_' + k: v for k, v in gs.items() if not k.endswith('_p')})
        for mode in MODES:
            row = common | {'mode': mode}
            if 'TCGA' not in cohorts:
                for name in ('primary', 'amplitude', 'continuous'):
                    row.update({name + '_status': 'not_estimable', name + '_reason': entry['reason'],
                                name + '_p': np.nan, name + '_or': np.nan})
                row['direction_status'] = 'unavailable_original_tcga_cohort'
            else:
                z = cohorts['TCGA']['frame'].copy()
                if mode != MODES[0]:
                    z['positive'] = z.positive_wga.astype(bool)
                row.update(analyze_frame(z, entry['target_subtype']))
                td, gd = direction(row['primary_or']), direction(gs['primary_or'])
                if row['primary_status'] != 'estimable':
                    ds = 'tcga_insufficient_common_support'
                elif gs['primary_status'] != 'estimable':
                    ds = 'genie_common_members_insufficient_support'
                elif td in (None, 0) or gd in (None, 0):
                    ds = 'boundary_or_uninformative_direction'
                else:
                    ds = 'same_direction' if td == gd else 'opposite_direction'
                row['direction_status'] = ds
            rows.append(row)
    out = pd.DataFrame(rows)
    for mode in MODES:
        ix = out.index[out['mode'].eq(mode)]
        if len(ix) != 45:
            raise ValueError('BH denominator must contain all 45 candidates')
        for model in ('primary', 'amplitude', 'continuous'):
            valid = out.loc[ix, model + '_status'].eq('estimable') & np.isfinite(out.loc[ix, model + '_p'])
            q = multipletests(out.loc[ix, model + '_p'].where(valid, 1.), method='fdr_bh')[1]
            out.loc[ix, model + '_q_candidate45'] = np.where(valid, q, np.nan)
    return out


def analyze(run_dir):
    from .tcga_export import summarize
    genie = Path(run_dir).expanduser().resolve()
    run = genie / 'tcga'
    with manifest.exclusive_run(genie):
        combined = manifest.read(genie)
        if any(s.get('status') == 'RUNNING' for s in combined['stages'].values()):
            raise RuntimeError('A GENIE stage is marked RUNNING')
        data = combined.get('tcga', {})
        if data.get('schema_version') != 1 or data['stages'].get('prepare') != 'COMPLETE' or 'analyze' in data['stages']:
            raise ValueError('Preparation must be complete and analysis not yet attempted')
        data['stages']['analyze'] = 'RUNNING'; record(run, data)
        try:
            result = score(read_jobs(run / 'LOCAL_ONLY_cases.zip'))
            private_parquet(result, run / 'LOCAL_ONLY_results.parquet')
            summary = summarize(result)
            data['stages']['analyze'] = 'COMPLETE'
            data['summary'].update(summary)
            record(run, data); print(json.dumps(summary, indent=2))
        except Exception as exc:
            data['stages']['analyze'] = 'FAILED'; data['error'] = str(exc); record(run, data)
            raise
