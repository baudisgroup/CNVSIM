"""Frozen-family, local-CNV conditional association audit. Frozen-family reproduction; user-supplied local inputs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import qr
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from . import statistics as stats
from .cache import read_jobs, write_jobs
from .manifest import save_record, has_record

exact = stats.fast_two_by_two_exact_and_cmh
HERE = FEATURES = RAW = SAMPLES = COVERAGE = ANALYSIS_CONFIG = CONTRASTS = SHAPE = None



def inputs():
    files = {
        'samples': SAMPLES,
        'coverage': COVERAGE,
        'config': ANALYSIS_CONFIG,
        'contrasts': CONTRASTS,
        'shape': SHAPE,
        'raw_maf': RAW,
    }
    for root in sorted(pd.read_parquet(files['contrasts'], columns=['root']).root.unique()):
        files['features_' + root] = FEATURES / f'{root}_features.parquet'
    return files


def prepare():
    if has_record(HERE, 'PREPARED'):
        raise FileExistsError('prepared inputs already exist')
    paths = inputs()
    # Our upstream sequence_variant_v1.yaml is saved as analysis_config.json.
    # It contains analysis rules, not a GENIE-supplied data file.
    cfg = json.loads(paths['config'].read_text())
    w = pd.read_parquet(paths['contrasts'])
    assert len(w) == 9582
    assert not w.duplicated(['root', 'ONCOTREE_CODE', 'gene', 'subtype']).any()
    gs = pd.read_parquet(paths['samples'])
    gs = gs[gs.selected_patient_root & gs.reference_layer.eq('core')].copy()
    assert gs.analysis_id.is_unique and gs.sample_id.is_unique
    assert gs.sample_id.eq(gs.SAMPLE_ID).all()
    shape = pd.read_parquet(paths['shape'], columns=['analysis_id', 'cnv_fraction'])
    gs = gs.merge(shape, on='analysis_id', how='left', validate='one_to_one', sort=False)
    assert gs.cnv_fraction.notna().all()
    coverage = pd.read_parquet(paths['coverage'])
    genes = set(w.gene)
    keep_samples = set(gs.sample_id)
    positive = {g: set() for g in genes}
    uncertain = {g: set() for g in genes}
    cols = ['Tumor_Sample_Barcode', 'Hugo_Symbol', 'NCBI_Build', 'Variant_Classification',
            'Mutation_Status', 'FILTER', 'Annotation_Status']
    raw_rows = 0
    for chunk in pd.read_csv(RAW, sep='\t', dtype=str, usecols=cols, chunksize=250000):
        raw_rows += len(chunk)
        m = chunk[chunk.Tumor_Sample_Barcode.isin(keep_samples) & chunk.Hugo_Symbol.isin(genes)
                  & chunk.FILTER.eq('PASS') & chunk.Annotation_Status.eq('SUCCESS')
                  & chunk.NCBI_Build.eq('GRCh37') & chunk.Variant_Classification.isin(cfg['qualifying_classes'])]
        som = m.Mutation_Status.fillna('').str.lower().eq('somatic')
        for gene, g in m[som].groupby('Hugo_Symbol'):
            positive[gene].update(g.Tumor_Sample_Barcode)
        for gene, g in m[~som].groupby('Hugo_Symbol'):
            uncertain[gene].update(g.Tumor_Sample_Barcode)
    jobs = []
    audit = []
    for root, wr in w.groupby('root', sort=True):
        names = set(pq.read_schema(paths['features_' + root]).names)
        wanted = sorted({g + '__' + d for g in wr.gene.unique() for d in ('gain', 'loss')} & names)
        X = pd.read_parquet(paths['features_' + root], columns=wanted)
        assert X.index.is_unique
        gr = gs[gs.root.eq(root)]
        for (dx, gene), wg in wr.groupby(['ONCOTREE_CODE', 'gene'], sort=True):
            assays = set(coverage.loc[coverage.Hugo_Symbol.eq(gene) & coverage.atlas_eligible, 'SEQ_ASSAY_ID'])
            # Primary positive and negative rules are copied as logical definitions,
            # not approximated by "no mutation row means negative".
            # Negative here means no qualifying reported call under these rules,
            # not independently verified clinical wild-type status.
            z = gr[gr.ONCOTREE_CODE.eq(dx) & gr.SEQ_ASSAY_ID.isin(assays)].copy()
            pos = z.sample_id.isin(positive[gene])
            neg = ~pos & ~z.sample_id.isin(uncertain[gene]) & z.has_any_raw_mutation_record
            z = z[pos | neg].copy()
            z['positive'] = z.sample_id.isin(positive[gene])
            counts = z.assigned_reference_key.value_counts()
            z = z[z.assigned_reference_key.isin(counts[counts.ge(cfg['eligibility']['within_diagnosis_min_evaluable'])].index)].copy()
            z['fga_bin'] = pd.qcut(z.cnv_fraction.rank(method='first'), q=5, labels=False).astype(int)
            assert set(z.assigned_reference_key) == set(wg.subtype)
            for row in wg.itertuples():
                t = z.assigned_reference_key.eq(row.subtype)
                found = (int(t.sum()), int(z.loc[t, 'positive'].sum()), int((~t).sum()), int(z.loc[~t, 'positive'].sum()))
                expected = (int(row.subtype_n), int(row.subtype_pos), int(row.other_subtypes_n), int(row.other_subtypes_pos))
                if found != expected:
                    raise ValueError((root, dx, gene, row.subtype, found, expected))
            gain, loss = gene + '__gain', gene + '__loss'
            available = gain in X and loss in X
            if available:
                z['gain'] = z.analysis_id.map(X[gain])
                z['loss'] = z.analysis_id.map(X[loss])
                for c in ('gain', 'loss'):
                    assert z[c].dropna().isin([0, 1, 2, 3]).all()
            else:
                z['gain'] = np.nan
                z['loss'] = np.nan
            audit.append({'root': root, 'diagnosis': dx, 'gene': gene, 'samples': len(z),
                          'comparisons': len(wg), 'local_gene_available': available,
                          'missing_local_cnv': int(z[['gain', 'loss']].isna().any(axis=1).sum())})
            jobs.append({'root': root, 'diagnosis': dx, 'gene': gene, 'frame': z[[
                'analysis_id', 'sample_id', 'assigned_reference_key', 'SEQ_ASSAY_ID',
                'positive', 'cnv_fraction', 'fga_bin', 'gain', 'loss']].reset_index(drop=True),
                'rows': wg.to_dict('records'), 'available': available})
        print(f'PREPARED {root}: {len(wr)} comparisons', flush=True)
    assert sum(len(j['rows']) for j in jobs) == len(w)
    write_jobs(HERE / 'LOCAL_ONLY_jobs.zip', jobs)
    pd.DataFrame(audit).to_csv(HERE / 'cohort_reconstruction_audit.tsv', sep='\t', index=False)
    save_record(HERE, 'PREPARED', {'utc': datetime.now(timezone.utc).isoformat(),
              'groups': len(jobs),
              'contrasts_reproduced': len(w), 'raw_maf_rows_read': raw_rows,
              'statistics_refitted_so_far': False})


def stratify(frame, local=None):
    key = frame.SEQ_ASSAY_ID.astype(str) + '|' + frame.fga_bin.astype(str)
    if local == 'binary':
        key += '|' + (frame.gain.gt(0).astype(int) + 2 * frame.loss.gt(0).astype(int)).astype(str)
    elif local == 'amplitude':
        key += '|' + (frame.gain.astype(int) + 4 * frame.loss.astype(int)).astype(str)
    idx, levels = pd.factorize(key, sort=True)
    return idx, len(levels)


def exact_result(frame, target, mode):
    idx, count = stratify(frame, mode)
    left = frame.assigned_reference_key.eq(target).to_numpy()
    y = frame.positive.to_numpy(dtype=bool)
    N = np.bincount(idx, minlength=count)
    M = np.bincount(idx, weights=y, minlength=count)
    L = np.bincount(idx, weights=left, minlength=count)
    valid = (N > 1) & (M > 0) & (M < N) & (L > 0) & (L < N)
    informative = valid[idx]
    res = exact(left, y, idx, count)
    return {'p': float(res[1]), 'or': float(res[2]), 'ci_low': float(res[3]), 'ci_high': float(res[4]),
            'strata': int(valid.sum()), 'target_informative': int((left & informative).sum()),
            'control_informative': int((~left & informative).sum())}, informative, idx


def support(result, target_n):
    return (result['strata'] >= 2 and result['target_informative'] >= 20
            and result['target_informative'] / target_n >= .5
            and result['control_informative'] >= 20 and np.isfinite(result['p']))


def spline(values, knots):
    a, b = knots[-2:]
    width = knots[-1] - knots[0]
    out = [values]
    for k in knots[:-2]:
        out.append((np.maximum(values-k, 0)**3 - (b-k)/(b-a)*np.maximum(values-a, 0)**3
                    + (a-k)/(b-a)*np.maximum(values-b, 0)**3) / width**2)
    return np.column_stack(out)


def continuous(frame, target, knots):
    result = {'status': 'not_estimable', 'reason': '', 'p': np.nan, 'or': np.nan,
              'ci_low': np.nan, 'ci_high': np.nan, 'n': len(frame), 'rank': 0}
    if not np.all(np.diff(knots) > 0):
        result['reason'] = 'repeated_fga_knots'
        return result
    y = frame.positive.to_numpy(dtype=float)
    if min(y.sum(), len(y)-y.sum()) < 5:
        result['reason'] = 'fewer_than_five_positive_or_negative'
        return result
    state = frame.gain.gt(0).astype(int) + 2 * frame.loss.gt(0).astype(int)
    blocks = frame.SEQ_ASSAY_ID.astype(str) + '|' + state.astype(str)
    categorical = pd.get_dummies(blocks, dtype=float).to_numpy()
    sp = spline(frame.cnv_fraction.to_numpy(), knots)
    sd = sp.std(axis=0)
    sp = (sp[:, sd > 1e-12] - sp[:, sd > 1e-12].mean(axis=0)) / sd[sd > 1e-12]
    x = np.column_stack([categorical, sp])
    _, R, piv = qr(x, mode='economic', pivoting=True)
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-10 * max(1, np.abs(R).max())))
    x = x[:, piv[:rank]]
    z = frame.assigned_reference_key.eq(target).to_numpy(dtype=float)
    full_x = np.column_stack([x, z])
    result['rank'] = rank
    if np.linalg.matrix_rank(full_x) != rank + 1:
        result['reason'] = 'target_collinear_with_baseline'
        return result
    if len(y) - rank - 1 < 20:
        result['reason'] = 'insufficient_residual_df'
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            null = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=150, tol=1e-9)
            full = sm.GLM(y, full_x, family=sm.families.Binomial()).fit(
                start_params=np.r_[null.params, 0.], maxiter=150, tol=1e-9)
        warning_text = ' '.join(str(c.message).lower() for c in caught)
        if any(t in warning_text for t in ('separation', 'overflow', 'invalid value', 'divide by zero')):
            result['reason'] = 'separation_or_numerical_warning'
            return result
        if not null.converged or not full.converged:
            result['reason'] = 'nonconvergence'
            return result
        for fit in (null, full):
            if (not np.isfinite(fit.llf) or not np.isfinite(fit.params).all()
                    or not np.isfinite(fit.bse).all() or np.abs(fit.params).max() > 25
                    or fit.bse.max() > 10):
                result['reason'] = 'unstable_or_separated_fit'
                return result
        lr = 2 * (full.llf - null.llf)
        if lr < -1e-6:
            result['reason'] = 'invalid_nested_likelihood'
            return result
        beta, se = float(full.params[-1]), float(full.bse[-1])
        result.update(status='estimable', reason='', p=float(chi2.sf(max(0, lr), 1)),
                      or_=float(np.exp(beta)), ci_low=float(np.exp(beta-1.96*se)),
                      ci_high=float(np.exp(beta+1.96*se)), lr=float(max(0, lr)))
        result['or'] = result.pop('or_')
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result['reason'] = type(exc).__name__ + ':' + str(exc)[:120]
    return result


def process_job(job):
    z = job['frame']
    output = []
    for original in job['rows']:
        row = dict(original)
        target = original['subtype']
        base, _, _ = exact_result(z, target, None)
        p_old = original['assay_fga_exact_score_p']
        if not np.isclose(base['p'], p_old, rtol=1e-6, atol=1e-300, equal_nan=True):
            raise ValueError(('frozen_p_not_reproduced', job['root'], job['diagnosis'], job['gene'], target, base['p'], p_old))
        assert base['strata'] == original['n_informative_assay_fga_strata']
        assert base['target_informative'] + base['control_informative'] == original['n_in_assay_fga_strata']
        row['frozen_p_reproduced'] = True
        for name in ('primary', 'amplitude', 'continuous'):
            row.update({name+'_status': 'not_estimable', name+'_reason': 'local_gene_not_encoded',
                        name+'_p': np.nan, name+'_or': np.nan})
        row['local_cnv_missing_n'] = int(z[['gain', 'loss']].isna().any(axis=1).sum())
        if not job['available']:
            output.append(row)
            continue
        complete = z.dropna(subset=['gain', 'loss'])
        if complete.empty:
            output.append(row)
            continue
        pri, info, st_idx = exact_result(complete, target, 'binary')
        amp, _, _ = exact_result(complete, target, 'amplitude')
        for name, res in [('primary', pri), ('amplitude', amp)]:
            good = support(res, original['subtype_n'])
            row.update({name+'_'+k: v for k, v in res.items()})
            row[name+'_status'] = 'estimable' if good else 'not_estimable'
            row[name+'_reason'] = '' if good else 'insufficient_common_support'
            row[name+'_target_retention'] = res['target_informative'] / original['subtype_n']
            row[name+'_control_retention'] = res['control_informative'] / original['other_subtypes_n']
        diffs, weights = [], []
        for st in np.unique(st_idx[info]):
            g = complete.iloc[np.where((st_idx == st) & info)[0]]
            lm = g.assigned_reference_key.eq(target)
            diffs.append(abs(g.loc[lm, 'cnv_fraction'].mean()-g.loc[~lm, 'cnv_fraction'].mean()))
            weights.append(lm.sum()*(~lm).sum()/len(g))
        row['within_stratum_weighted_abs_fga_mean_gap'] = float(np.average(diffs, weights=weights)) if weights else np.nan
        if row['primary_status'] == 'estimable':
            knots = np.quantile(z.cnv_fraction, [.05, .35, .65, .95])
            lr = continuous(complete.iloc[np.flatnonzero(info)], target, knots)
            row.update({'continuous_'+k: v for k, v in lr.items()})
            row['continuous_fga_knots'] = ','.join(map(str, knots))
        else:
            row['continuous_reason'] = 'primary_common_support_insufficient'
        output.append(row)
    return output


def summarize(frame):
    out = frame.copy()
    for name in ('primary', 'amplitude', 'continuous'):
        valid = out[name+'_status'].eq('estimable') & np.isfinite(out[name+'_p'])
        out[name+'_q_declared'] = multipletests(out[name+'_p'].where(valid, 1.), method='fdr_bh')[1]
        out[name+'_q_estimable'] = np.nan
        if valid.any():
            out.loc[valid, name+'_q_estimable'] = multipletests(out.loc[valid, name+'_p'], method='fdr_bh')[1]
        out[name+'_significant'] = valid & out[name+'_q_declared'].lt(.05)
    labels = []
    for row in out.itertuples():
        if row.primary_status != 'estimable':
            label = 'not_estimable'
        elif not row.primary_significant:
            label = 'no_detected_residual_association'
        elif row.amplitude_status != 'estimable' or row.continuous_status != 'estimable':
            label = 'partial_support_unestimable'
        elif not row.amplitude_significant or not row.continuous_significant:
            label = 'primary_only_not_robust'
        elif len({np.sign(np.log(x)) for x in (row.primary_or, row.amplitude_or, row.continuous_or)}) > 1:
            label = 'direction_inconsistent'
        else:
            label = 'robust_residual_association'
        labels.append(label)
    out['residual_status'] = labels
    out['primary_direction_agrees_with_old_fga'] = np.sign(np.log(out.primary_or)) == np.sign(np.log(out.assay_fga_cmh_or))
    out.to_parquet(HERE / 'conditional_associations.parquet', index=False)
    out.to_csv(HERE / 'conditional_associations.tsv', sep='\t', index=False)
    def counts(g):
        return {'comparisons': len(g), 'status_counts': g.residual_status.value_counts().to_dict(),
                'checks': {n: {'estimable': int(g[n+'_status'].eq('estimable').sum()),
                              'significant': int(g[n+'_significant'].sum()),
                              'failure_reasons': g.loc[g[n+'_status'].ne('estimable'), n+'_reason'].value_counts().to_dict()}
                           for n in ('primary', 'amplitude', 'continuous')},
                'primary_supported_subtypes': int(g.loc[g.primary_significant, 'subtype'].nunique()),
                'robust_subtypes': int(g.loc[g.residual_status.eq('robust_residual_association'), 'subtype'].nunique()),
                'robust_roots': int(g.loc[g.residual_status.eq('robust_residual_association'), 'root'].nunique())}
    summary = {'all': counts(out),
               'old_panel_significant_512': counts(out[out.assay_exact_score_q_declared.lt(.05)]),
               'old_fga_retained_71': counts(out[out.fga_stratification_status.eq('retained_after_fga_stratification')]),
               'by_old_status': {k: counts(g) for k, g in out.groupby('fga_stratification_status')},
               'interpretation': 'Exploratory conditional association, not causal, predictive or independent source validation.'}
    save_record(HERE, 'SUMMARY', summary)
    return summary


def analyze(workers):
    if not has_record(HERE, 'PREPARED'):
        raise ValueError('Run local-prepare first')
    if (HERE / 'conditional_associations.parquet').exists():
        raise FileExistsError('completed results already exist')
    jobs = read_jobs(HERE / 'LOCAL_ONLY_jobs.zip')
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(process_job, jobs, chunksize=1), 1):
            records.extend(rows)
            if i % 100 == 0 or i == len(jobs):
                print(f'PROGRESS {i}/{len(jobs)} groups, {len(records)}/9582 comparisons', flush=True)
    df = pd.DataFrame(records)
    assert len(df) == 9582 and df.frozen_p_reproduced.all()
    summary = summarize(df)
    save_record(HERE, 'COMPLETE', {'utc': datetime.now(timezone.utc).isoformat(),
              'reproduced_all_frozen_counts_and_p_values': True})
    print(json.dumps({k: summary[k] for k in ('all', 'old_fga_retained_71')}, indent=2), flush=True)
