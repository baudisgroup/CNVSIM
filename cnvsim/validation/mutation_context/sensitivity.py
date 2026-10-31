"""Prespecified exploratory density and sampling-type sensitivity."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import gc
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from . import association as h
from .cache import read_jobs, write_jobs
HERE = PRIOR = None
KEY=['root','ONCOTREE_CODE','gene','subtype']
NAMES=['matched_baseline','density_only','type_only','combined','amplitude','quadratic','restricted']


def fit(frame,target,knots,name):
    out=dict(status='not_estimable',reason='',p=np.nan,or_=np.nan,ci_low=np.nan,ci_high=np.nan,n=len(frame))
    if not np.all(np.diff(knots)>0):
        out['reason']='repeated_fga_knots';return out
    y=frame.positive.to_numpy(dtype=float)
    if min(y.sum(),len(y)-y.sum())<5:
        out['reason']='fewer_than_five_positive_or_negative';return out
    state=(frame.gain.astype(int)+4*frame.loss.astype(int)) if name=='amplitude' else (frame.gain.gt(0).astype(int)+2*frame.loss.gt(0).astype(int))
    blocks=frame.SEQ_ASSAY_ID.astype(str)+'|'+state.astype(str)
    cat=pd.get_dummies(blocks,dtype=float).to_numpy()
    numeric=[h.spline(frame.cnv_fraction.to_numpy(),knots)]
    density=np.log1p(frame.leave_gene_out_density.to_numpy())
    if name not in ['matched_baseline','type_only']:
        numeric.append(density[:,None])
    if name not in ['matched_baseline','density_only']:
        numeric.append(frame.SAMPLE_TYPE.eq('Metastasis').to_numpy(dtype=float)[:,None])
    if name=='quadratic':numeric.append((density**2)[:,None])
    num=np.column_stack(numeric);sd=num.std(axis=0)
    num=(num[:,sd>1e-12]-num[:,sd>1e-12].mean(axis=0))/sd[sd>1e-12]
    x=np.column_stack([cat,num]);_,R,piv=qr(x,mode='economic',pivoting=True)
    rank=int((abs(np.diag(R))>1e-10*max(1,abs(R).max())).sum());x=x[:,piv[:rank]]
    z=frame.assigned_reference_key.eq(target).to_numpy(dtype=float);full_x=np.column_stack([x,z])
    out['rank']=rank
    if np.linalg.matrix_rank(full_x)!=rank+1:
        out['reason']='target_collinear_with_baseline';return out
    if len(y)-rank-1<20:
        out['reason']='insufficient_residual_df';return out
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            null=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=150,tol=1e-9)
            full=sm.GLM(y,full_x,family=sm.families.Binomial()).fit(start_params=np.r_[null.params,0.],maxiter=150,tol=1e-9)
        wt=' '.join(str(w.message).lower() for w in caught)
        if any(s in wt for s in ['separation','overflow','invalid value','divide by zero']):
            out['reason']='separation_or_numerical_warning';return out
        if not null.converged or not full.converged:
            out['reason']='nonconvergence';return out
        for f in [null,full]:
            if not np.isfinite(f.llf) or not np.isfinite(f.params).all() or not np.isfinite(f.bse).all() or abs(f.params).max()>25 or f.bse.max()>10:
                out['reason']='unstable_or_separated_fit';return out
        lr=2*(full.llf-null.llf)
        if lr < -1e-6:
            out['reason']='invalid_nested_likelihood';return out
        beta,se=float(full.params[-1]),float(full.bse[-1])
        out.update(status='estimable',p=float(chi2.sf(max(lr,0),1)),or_=float(np.exp(beta)),ci_low=float(np.exp(beta-1.96*se)),ci_high=float(np.exp(beta+1.96*se)),lr=float(max(lr,0)))
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        out['reason']=type(exc).__name__+':'+str(exc)[:120]
    return out


def process(job):
    z=job['frame'];knots=np.quantile(z.cnv_fraction,[.05,.35,.65,.95]);output=[]
    complete=z[z.SAMPLE_TYPE.isin(['Primary','Metastasis'])].dropna(subset=['gain','loss','leave_gene_out_density'])
    restricted=complete[complete.leave_gene_out_density.lt(20)&~complete.other_mmr_polymerase_call]
    for old in job['rows']:
        row={k:old[k] for k in KEY};target=old['subtype'];nt=int(old['subtype_n']);nc=int(old['other_subtypes_n'])
        row.update(pooled_robust=old['pooled_robust'],old_fga_status=old['fga_stratification_status'],old_panel_q=old['assay_exact_score_q_declared'],old_primary_or=old['old_primary_or'],original_target_n=nt,original_control_n=nc)
        for name in NAMES:
            row.update({name+'_status':'not_estimable',name+'_reason':'local_gene_not_encoded' if not job['available'] else 'insufficient_common_support',name+'_p':np.nan,name+'_or':np.nan})
        for label,f in [('all',z),('complete',complete),('restricted',restricted)]:
            target_mask=f.assigned_reference_key.eq(target)
            row[label+'_target_n']=int(target_mask.sum());row[label+'_control_n']=int((~target_mask).sum())
            row[label+'_target_pos']=int(f.loc[target_mask,'positive'].sum());row[label+'_control_pos']=int(f.loc[~target_mask,'positive'].sum())
        row['complete_target_retention']=row['complete_target_n']/nt
        row['complete_control_retention']=row['complete_control_n']/nc
        row['sample_type_excluded_target_n']=int((z.assigned_reference_key.eq(target)&~z.SAMPLE_TYPE.isin(['Primary','Metastasis'])).sum())
        row['sample_type_excluded_control_n']=int((z.assigned_reference_key.ne(target)&~z.SAMPLE_TYPE.isin(['Primary','Metastasis'])).sum())
        row['local_cnv_missing_n']=int(z[['gain','loss']].isna().any(axis=1).sum())
        if job['available'] and len(complete):
            base,info,_=h.exact_result(complete,target,'binary')
            row.update({'support_'+k:v for k,v in base.items()})
            if h.support(base,nt):
                active=complete.iloc[np.flatnonzero(info)]
                for name in NAMES[:-1]:
                    if name=='amplitude':
                        amp,_,_=h.exact_result(active,target,'amplitude')
                        row.update({'amplitude_support_'+k:v for k,v in amp.items()})
                        if not h.support(amp,nt):
                            row['amplitude_reason']='amplitude_common_support_insufficient';continue
                    result=fit(active,target,knots,name)
                    row.update({name+'_'+('or' if k=='or_' else k):v for k,v in result.items()})
            if len(restricted):
                restr,inf,_=h.exact_result(restricted,target,'binary')
                row.update({'restricted_support_'+k:v for k,v in restr.items()})
                if h.support(restr,nt):
                    row.update({'restricted_'+('or' if k=='or_' else k):v for k,v in fit(restricted.iloc[np.flatnonzero(inf)],target,knots,'restricted').items()})
        output.append(row)
    return output


def prepare():
    paths={'jobs':PRIOR/'LOCAL_ONLY_jobs.zip','pooled':PRIOR/'conditional_associations.parquet',
           'samples':h.SAMPLES,'coverage':h.COVERAGE,'config':h.ANALYSIS_CONFIG,'raw_maf':h.RAW}
    cfg=json.loads(paths['config'].read_text());gs=pd.read_parquet(paths['samples'], columns=['selected_patient_root','reference_layer','analysis_id','sample_id','SAMPLE_TYPE','SAMPLE_TYPE_DETAILED'])
    gs=gs[gs.selected_patient_root&gs.reference_layer.eq('core')];assert gs.analysis_id.is_unique and gs.sample_id.is_unique
    meta=gs.set_index('analysis_id');keep=set(gs.sample_id)
    coverage=pd.read_parquet(paths['coverage'])
    assert coverage.groupby('SEQ_ASSAY_ID').assay_exonic_mb.nunique().eq(1).all()
    mb=coverage.groupby('SEQ_ASSAY_ID').assay_exonic_mb.first()
    dedup=['Tumor_Sample_Barcode','Hugo_Symbol','Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2','Mutation_Status','Variant_Classification','HGVSp_Short']
    cols=dedup+['FILTER','Annotation_Status','NCBI_Build'];parts=[];raw_n=0
    for chunk in pd.read_csv(h.RAW,sep='\t',dtype=str,usecols=cols,chunksize=250000):
        raw_n+=len(chunk)
        mask=chunk.Tumor_Sample_Barcode.isin(keep)&chunk.FILTER.eq('PASS')&chunk.Annotation_Status.eq('SUCCESS')&chunk.NCBI_Build.eq('GRCh37')&chunk.Variant_Classification.isin(cfg['qualifying_classes'])&chunk.Mutation_Status.str.lower().eq('somatic')
        parts.append(chunk.loc[mask,dedup])
    v=pd.concat(parts).drop_duplicates(dedup);total=v.groupby('Tumor_Sample_Barcode').size();by_gene=v.groupby(['Hugo_Symbol','Tumor_Sample_Barcode']).size()
    mmr=v[v.Hugo_Symbol.isin(cfg['mmr_polymerase_genes'])][['Tumor_Sample_Barcode','Hugo_Symbol']].drop_duplicates()
    mmr_count=mmr.groupby('Tumor_Sample_Barcode').size()
    qualifying_count = len(v)
    # Counts above are sufficient; release raw MAF rows before loading sample caches.
    del v, parts, chunk, mask
    gc.collect()
    old=pd.read_parquet(paths['pooled']).set_index(KEY);jobs=read_jobs(paths['jobs']);unique=[]
    for job in jobs:
        z=job['frame'];gene=job['gene'];own=by_gene.loc[gene] if gene in by_gene.index.get_level_values(0) else pd.Series(dtype=float)
        all_count=z.sample_id.map(total).fillna(0).astype(int);own_count=z.sample_id.map(own).fillna(0).astype(int)
        assert (all_count>=own_count).all();assert np.array_equal(own_count.gt(0),z.positive)
        z['all_qualifying_records']=all_count;z['target_gene_records']=own_count
        z['assay_exonic_mb']=z.SEQ_ASSAY_ID.map(mb);assert z.assay_exonic_mb.notna().all() and z.assay_exonic_mb.gt(0).all()
        z['leave_gene_out_density']=(all_count-own_count)/z.assay_exonic_mb
        z['full_reported_density']=all_count/z.assay_exonic_mb
        nmmr=z.sample_id.map(mmr_count).fillna(0).astype(int)
        if gene in cfg['mmr_polymerase_genes']:nmmr=nmmr-own_count.gt(0).astype(int)
        assert nmmr.ge(0).all();z['other_mmr_polymerase_call']=nmmr.gt(0)
        z['SAMPLE_TYPE']=z.analysis_id.map(meta.SAMPLE_TYPE)
        z['SAMPLE_TYPE_DETAILED']=z.analysis_id.map(meta.SAMPLE_TYPE_DETAILED)
        for row in job['rows']:
            orig=old.loc[tuple(row[k] for k in KEY)];row['pooled_robust']=orig.residual_status=='robust_residual_association';row['old_primary_or']=orig.primary_or
        unique.append(z[['analysis_id','SAMPLE_TYPE','SAMPLE_TYPE_DETAILED','full_reported_density','assay_exonic_mb']])
    u=pd.concat(unique).drop_duplicates('analysis_id')
    u.groupby(['SAMPLE_TYPE','SAMPLE_TYPE_DETAILED'],dropna=False).size().rename('unique_analyses').reset_index().to_csv(HERE/'sample_type_coverage.tsv',sep='\t',index=False)
    h.save_record(HERE, 'PREPARED',{'raw_rows_read':raw_n,'qualifying_deduplicated_records_selected_core':qualifying_count,'unique_evaluable_analyses':len(u),'sample_type_counts':u.SAMPLE_TYPE.fillna('missing').value_counts().to_dict(),'all_gene_positivity_reproduced':True,'inference_not_started':True})
    write_jobs(HERE/'LOCAL_ONLY_jobs.zip', jobs)
    print('PREPARED all9582; target-gene positivity reproduced',flush=True)
    return jobs


def summarize(df):
    for name in NAMES:
        valid=df[name+'_status'].eq('estimable')&np.isfinite(df[name+'_p'])
        df[name+'_q_declared']=multipletests(df[name+'_p'].where(valid,1),method='fdr_bh')[1]
        df[name+'_significant']=valid&df[name+'_q_declared'].lt(.05)
    labels=[];checks=['combined','amplitude','quadratic','restricted']
    for r in df.to_dict('records'):
        if r['combined_status']!='estimable':label='not_estimable'
        elif not r['combined_significant']:label='no_detected_adjusted_association'
        elif any(r[n+'_status']!='estimable' for n in checks[1:]):label='supported_with_unestimable_sensitivity'
        elif not all(r[n+'_significant'] for n in checks):label='supported_but_not_all_sensitivities'
        elif not all(np.isfinite(r[n+'_or']) and r[n+'_or']>0 for n in checks):label='direction_not_estimable'
        elif len(set(np.sign(np.log(r[n+'_or'])) for n in checks))>1:label='direction_inconsistent'
        else:label='supported_across_context_checks'
        labels.append(label)
    df['context_status']=labels
    df.to_parquet(HERE/'CONTEXT_RESULTS.parquet',index=False);df.to_csv(HERE/'CONTEXT_RESULTS.tsv',sep='\t',index=False)
    focus=df[df.pooled_robust]
    def counts(g):
        return {'comparisons':len(g),'context_status':g.context_status.value_counts().to_dict(),
                'checks':{n:{'estimable':int(g[n+'_status'].eq('estimable').sum()),'significant':int(g[n+'_significant'].sum()),'failure_reasons':g.loc[g[n+'_status'].ne('estimable'),n+'_reason'].value_counts().to_dict()} for n in NAMES}}
    summary={'all':counts(df),'prior45':counts(focus),'prior71':counts(df[df.old_fga_status.eq('retained_after_fga_stratification')]),'prior512':counts(df[df.old_panel_q.lt(.05)]),'other_than_prior45':counts(df[~df.pooled_robust])}
    for name,g in [('prior45',focus),('all_supported',df[df.context_status.eq('supported_across_context_checks')]),('prior45_supported',focus[focus.context_status.eq('supported_across_context_checks')])]:
        summary[name+'_pan_cancer']={'comparisons':len(g),'subtypes':int(g.subtype.nunique()),'roots':int(g.root.nunique()),'genes':int(g.gene.nunique()),'root_counts':g.root.value_counts().to_dict(),'TP53_comparisons':int(g.gene.eq('TP53').sum())}
    h.save_record(HERE, 'SUMMARY',summary)
    return summary


def run(workers):
    if (HERE/'CONTEXT_RESULTS.parquet').exists():raise FileExistsError('completed results exist')
    jobs=prepare();rows=[]
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        for i,records in enumerate((pool.map(process, jobs, chunksize=1) if pool is not None else map(process, jobs)),1):
            rows.extend(records)
            if i%100==0 or i==len(jobs):print(f'PROGRESS {i}/{len(jobs)}; {len(rows)}/9582',flush=True)
    df=pd.DataFrame(rows);assert len(df)==9582 and int(df.pooled_robust.sum())==45
    summary=summarize(df)
    h.save_record(HERE, 'COMPLETE',{'utc':datetime.now(timezone.utc).isoformat(),'comparisons':len(df)})
    print(json.dumps({'prior45':summary['prior45'],'prior45_supported_pan_cancer':summary['prior45_supported_pan_cancer']},indent=2),flush=True)
