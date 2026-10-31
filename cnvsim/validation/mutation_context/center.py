"""Same-cohort center consistency; aggregate exports only."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from . import local_cnv as h
from .cache import read_jobs, write_jobs
HERE = PRIOR = None
KEY = ['root', 'ONCOTREE_CODE', 'gene', 'subtype']
CHECKS = ['primary', 'amplitude', 'continuous']


def heterogeneity(frame, target, knots):
    out = dict(status='not_estimable', reason='', p=np.nan, interaction_or=np.nan)
    if not np.all(np.diff(knots) > 0):
        out['reason'] = 'repeated_fga_knots'; return out
    y = frame.positive.to_numpy(dtype=float)
    if min(y.sum(), len(y)-y.sum()) < 5:
        out['reason'] = 'fewer_than_five_positive_or_negative'; return out
    state = frame.gain.gt(0).astype(int) + 2 * frame.loss.gt(0).astype(int)
    cat = pd.get_dummies(frame.SEQ_ASSAY_ID.astype(str)+'|'+state.astype(str), dtype=float).to_numpy()
    sp = h.spline(frame.cnv_fraction.to_numpy(), knots)
    m = frame.center_group.eq('MSK').to_numpy(dtype=float)
    sp = np.column_stack([sp*m[:, None], sp*(1-m[:, None])])
    sd = sp.std(axis=0)
    sp = (sp[:, sd > 1e-12]-sp[:, sd > 1e-12].mean(axis=0))/sd[sd > 1e-12]
    x = np.column_stack([cat, sp])
    _, R, piv = qr(x, mode='economic', pivoting=True)
    rank = int((np.abs(np.diag(R)) > 1e-10*max(1, np.abs(R).max())).sum())
    x = x[:, piv[:rank]]
    z = frame.assigned_reference_key.eq(target).to_numpy(dtype=float)
    null_x = np.column_stack([x, z])
    full_x = np.column_stack([null_x, z*m])
    if np.linalg.matrix_rank(null_x) != rank+1 or np.linalg.matrix_rank(full_x) != rank+2:
        out['reason'] = 'target_or_interaction_collinear'; return out
    if len(y)-rank-2 < 20:
        out['reason'] = 'insufficient_residual_df'; return out
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            null = sm.GLM(y, null_x, family=sm.families.Binomial()).fit(maxiter=150, tol=1e-9)
            full = sm.GLM(y, full_x, family=sm.families.Binomial()).fit(start_params=np.r_[null.params, 0.], maxiter=150, tol=1e-9)
        wt = ' '.join(str(w.message).lower() for w in caught)
        if any(s in wt for s in ['separation', 'overflow', 'invalid value', 'divide by zero']):
            out['reason'] = 'separation_or_numerical_warning'; return out
        if not null.converged or not full.converged:
            out['reason'] = 'nonconvergence'; return out
        for f in (null, full):
            if not np.isfinite(f.llf) or not np.isfinite(f.params).all() or not np.isfinite(f.bse).all() or abs(f.params).max()>25 or f.bse.max()>10:
                out['reason'] = 'unstable_or_separated_fit'; return out
        lr = 2*(full.llf-null.llf)
        if lr < -1e-6:
            out['reason'] = 'invalid_nested_likelihood'; return out
        out.update(status='estimable', p=float(chi2.sf(max(0, lr), 1)), interaction_or=float(np.exp(full.params[-1])), lr=float(max(0, lr)))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        out['reason'] = type(exc).__name__+':'+str(exc)[:120]
    return out


def process(job):
    pooled = job['frame']
    knots = np.quantile(pooled.cnv_fraction, [.05, .35, .65, .95])
    output, interactions = [], []
    for old in job['rows']:
        target = old['subtype']
        info_frames = []
        for center in ('MSK', 'non_MSK'):
            z = pooled[pooled.center_group.eq(center)]
            left = z.assigned_reference_key.eq(target)
            nt, nc = int(left.sum()), int((~left).sum())
            row = {k: old[k] for k in KEY}
            row.update(center_group=center, pooled_robust=old['pooled_robust'],
                       center_target_n=nt, center_control_n=nc,
                       center_target_pos=int(z.loc[left, 'positive'].sum()),
                       center_control_pos=int(z.loc[~left, 'positive'].sum()),
                       pooled_target_n=int(old['subtype_n']), pooled_control_n=int(old['other_subtypes_n']),
                       local_cnv_missing_n=int(z[['gain', 'loss']].isna().any(axis=1).sum()))
            reason = 'local_gene_not_encoded' if not job['available'] else 'no_target_in_center' if nt==0 else 'no_control_in_center' if nc==0 else 'insufficient_common_support'
            for check in CHECKS:
                row.update({check+'_status':'not_estimable', check+'_reason':reason, check+'_p':np.nan, check+'_or':np.nan})
            if job['available'] and nt and nc:
                complete = z.dropna(subset=['gain','loss'])
                if len(complete):
                    pri, info, _ = h.exact_result(complete, target, 'binary')
                    amp, _, _ = h.exact_result(complete, target, 'amplitude')
                    for name, result in [('primary', pri), ('amplitude', amp)]:
                        good = h.support(result, nt)
                        row.update({name+'_'+k:v for k,v in result.items()})
                        row[name+'_status'] = 'estimable' if good else 'not_estimable'
                        row[name+'_reason'] = '' if good else 'insufficient_common_support'
                        row[name+'_target_retention'] = result['target_informative']/nt
                        row[name+'_control_retention'] = result['control_informative']/nc
                    if row['primary_status']=='estimable':
                        active = complete.iloc[np.flatnonzero(info)]
                        info_frames.append(active)
                        row.update({'continuous_'+k:v for k,v in h.continuous(active, target, knots).items()})
                    else:
                        row['continuous_reason'] = 'primary_common_support_insufficient'
            output.append(row)
        if old['pooled_robust']:
            out = {k:old[k] for k in KEY}
            result = heterogeneity(pd.concat(info_frames), target, knots) if len(info_frames)==2 else dict(status='not_estimable', reason='both_centers_not_supported', p=np.nan, interaction_or=np.nan)
            out.update({'heterogeneity_'+k:v for k,v in result.items()})
            interactions.append(out)
    return output, interactions


def run(workers):
    if (HERE/'CENTER_RESULTS.parquet').exists():
        raise FileExistsError('completed center results exist')
    paths = {'jobs':PRIOR/'LOCAL_ONLY_jobs.zip', 'pooled_results':PRIOR/'conditional_associations.parquet',
             'samples':h.SAMPLES}
    gs = pd.read_parquet(paths['samples'])
    gs = gs[gs.selected_patient_root & gs.reference_layer.eq('core')].copy()
    assert gs.analysis_id.is_unique and gs.CENTER.notna().all()
    assert gs.groupby('SEQ_ASSAY_ID').CENTER.nunique().eq(1).all()
    centers = gs.set_index('analysis_id').CENTER
    pooled = pd.read_parquet(paths['pooled_results'])
    focus = pooled[pooled.residual_status.eq('robust_residual_association')].copy()
    assert len(focus)==45
    focus_keys = set(map(tuple, focus[KEY].to_numpy()))
    jobs = read_jobs(paths['jobs'])
    unique = []
    for job in jobs:
        z=job['frame']
        z['CENTER']=z.analysis_id.map(centers)
        assert z.CENTER.notna().all()
        z['center_group']=np.where(z.CENTER.eq('MSK'),'MSK','non_MSK')
        for old in job['rows']:
            old['pooled_robust']=tuple(old[k] for k in KEY) in focus_keys
        unique.append(z[['analysis_id','sample_id','CENTER','center_group']].assign(root=job['root']))
    unique=pd.concat(unique).drop_duplicates('analysis_id')
    unique.groupby(['root','CENTER','center_group']).size().rename('unique_evaluable_analyses').reset_index().to_csv(HERE/'center_sample_counts.tsv', sep='\t', index=False)
    assert sum(len(j['rows']) for j in jobs)==9582
    out, interactions=[], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(rows,hets) in enumerate(pool.map(process,jobs,chunksize=1),1):
            out.extend(rows); interactions.extend(hets)
            if i%100==0 or i==len(jobs):
                print(f'PROGRESS {i}/{len(jobs)} groups; {len(out)}/19164 center comparisons',flush=True)
    df=pd.DataFrame(out)
    assert len(df)==19164
    for center,idx in df.groupby('center_group').groups.items():
        assert len(idx)==9582
        for name in CHECKS:
            good=df.loc[idx,name+'_status'].eq('estimable') & np.isfinite(df.loc[idx,name+'_p'])
            df.loc[idx,name+'_q_declared']=multipletests(df.loc[idx,name+'_p'].where(good,1),method='fdr_bh')[1]
    df.to_parquet(HERE/'CENTER_RESULTS.parquet',index=False)
    df.to_csv(HERE/'CENTER_RESULTS.tsv',sep='\t',index=False)
    comparison=focus[KEY+['primary_or','residual_status','assay_exact_score_q_declared','fga_stratification_status']].rename(columns={'primary_or':'pooled_primary_or'})
    for center in ['MSK','non_MSK']:
        part=df[df.center_group.eq(center)].drop(columns=['center_group','pooled_robust'])
        comparison=comparison.merge(part.rename(columns={c:center+'_'+c for c in part if c not in KEY}),on=KEY,validate='one_to_one')
    het=pd.DataFrame(interactions)
    assert len(het)==45
    valid=het.heterogeneity_status.eq('estimable') & np.isfinite(het.heterogeneity_p)
    het['heterogeneity_q_45']=multipletests(het.heterogeneity_p.where(valid,1),method='fdr_bh')[1]
    comparison=comparison.merge(het,on=KEY,validate='one_to_one')
    categories=[]
    for row in comparison.to_dict('records'):
        available=[row[c+'_primary_status']=='estimable' and np.isfinite(row[c+'_primary_or']) and row[c+'_primary_or']>0 for c in ['MSK','non_MSK']]
        if not all(available):
            label='not_comparable_both_centers' if not any(available) else 'MSK_only_estimable' if available[0] else 'non_MSK_only_estimable'
        else:
            signs=[np.sign(np.log(row[c+'_primary_or'])) for c in ['MSK','non_MSK']]
            label='same_direction_as_pooled' if signs[0]==signs[1]==np.sign(np.log(row['pooled_primary_or'])) else 'center_direction_discordant' if signs[0]!=signs[1] else 'both_opposite_to_pooled'
        categories.append(label)
    comparison['center_consistency_status']=categories
    comparison.to_csv(HERE/'FOCUS_45_CENTER_COMPARISON.tsv',sep='\t',index=False)
    comparison.to_parquet(HERE/'FOCUS_45_CENTER_COMPARISON.parquet',index=False)
    def counts(g):
        return {c:{n:{'estimable':int(g.loc[g.center_group.eq(c),n+'_status'].eq('estimable').sum()),
                      'q_under_005':int(g.loc[g.center_group.eq(c),n+'_q_declared'].lt(.05).sum()),
                      'failure_reasons':g.loc[g.center_group.eq(c)&g[n+'_status'].ne('estimable'),n+'_reason'].value_counts().to_dict()} for n in CHECKS} for c in ['MSK','non_MSK']}
    summary={'unique_analyses_by_center':unique.CENTER.value_counts().to_dict(), 'all':counts(df),
             'focus_45':counts(df[df.pooled_robust]),
             'focus_consistency':comparison.center_consistency_status.value_counts().to_dict(),
             'focus_old71_consistency':comparison[comparison.fga_stratification_status.eq('retained_after_fga_stratification')].center_consistency_status.value_counts().to_dict(),
             'focus_new6_consistency':comparison[comparison.assay_exact_score_q_declared.ge(.05)].center_consistency_status.value_counts().to_dict(),
             'heterogeneity_estimable':int(valid.sum()),'heterogeneity_significant':int(het.heterogeneity_q_45.lt(.05).sum()),
             'heterogeneity_failures':het.loc[~valid,'heterogeneity_reason'].value_counts().to_dict()}
    h.save_record(HERE, 'SUMMARY',summary)
    h.save_record(HERE, 'COMPLETE',{'utc':datetime.now(timezone.utc).isoformat(),'comparisons':len(df)})
    print(json.dumps(summary,indent=2),flush=True)
