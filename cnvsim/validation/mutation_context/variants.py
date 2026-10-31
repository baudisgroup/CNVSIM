"""Frozen-cohort descriptive variant audit. No new statistical tests."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

from . import context as c
h = c.h
from .cache import read_jobs, write_jobs
HERE = PRIOR = None
FASTA_DIR = None
CASES=[('C50','IDC','C50-CTX4','ESR1'),('C50','IDC','C50-CTX4','GATA3'),('C18_C20','COAD','C18_C20-CTX10','EP300')]
REF={'ESR1':('P03372','ENST00000206249',595,311,547),'GATA3':('P23771','ENST00000346208',443,317,341),'EP300':('Q09472','ENST00000263253',2414,1287,1663)}
ALLELE=['Tumor_Sample_Barcode','Hugo_Symbol','NCBI_Build','Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2']
DEDUP=['Tumor_Sample_Barcode','Hugo_Symbol','Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2','Mutation_Status','Variant_Classification','HGVSp_Short']
COLS=list(dict.fromkeys(ALLELE+DEDUP+['Consequence','Variant_Type','HGVSc','HGVSp','Transcript_ID','RefSeq','Protein_position','Exon_Number','FILTER','SWISSPROT','Annotation_Status','mutationInCis_Flag']))
AA=dict(zip(['Ala','Arg','Asn','Asp','Cys','Gln','Glu','Gly','His','Ile','Leu','Lys','Met','Phe','Pro','Ser','Thr','Trp','Tyr','Val','Ter'],list('ARNDCQEGHILKMFPSTWYV')+['*']))
CATEGORIES={
 'ESR1':['Y537_D538_missense','other_LBD_missense','outside_LBD_missense','other_qualifying','annotation_unknown'],
 'GATA3':['splice_site','ZnF2_frameshift','predicted_truncation','predicted_extension','missense','in_frame_indel','frameshift_length_unknown','frameshift_length_neutral','other_qualifying','annotation_unknown'],
 'EP300':['nonsense','frameshift','splice_site','HAT_missense','outside_HAT_missense','other_qualifying','annotation_unknown']}

def norm(x):
    if pd.isna(x):return ''
    x=str(x).split(':')[-1].replace('(','').replace(')','')
    for a,b in AA.items():x=x.replace(a,b)
    return x

def classify(r,seq):
    gene=r['Hugo_Symbol'];acc,tr,L,lo,hi=REF[gene]
    short=norm(r.get('HGVSp_Short'));full=norm(r.get('HGVSp'))
    hg=short or full; vc=r['Variant_Classification']; start=None;end=None; length=None
    reason=''; region='unknown';ref_status='unparsed'
    conflict=bool(r.get('annotation_conflict',False)) or bool(short and full and short!=full)
    transcript_ok=str(r.get('Transcript_ID','')).split('.')[0]==tr
    # Splice placeholders are deliberately never interpreted as amino acid changes.
    m=re.match(r'^p\.([A-Z*])(\d+)(?:_([A-Z*])(\d+))?',hg)
    if m and vc!='Splice_Site' and '_splice' not in hg:
        start=int(m[2]);end=int(m[4]) if m[4] else start
        def check(aa,pos):return (aa=='*' and pos==L+1) or (1<=pos<=L and seq[pos-1]==aa)
        valid=check(m[1],start) and (not m[3] or check(m[3],end))
        ref_status='match' if valid else 'mismatch'
    else:valid=False
    coord_ok=valid and transcript_ok and not conflict
    if coord_ok:
        region='before' if end<lo else ('after' if start>hi else ('within' if start>=lo and end<=hi else 'overlap'))
        fs=re.search(r'fs\*(\d+)$',hg)
        if vc.startswith('Frame_Shift') and fs:length=start+int(fs[1])-2
        elif vc=='Nonsense_Mutation' and hg.endswith('*'):length=start-1
    if conflict:reason='conflicting_annotations'
    elif not transcript_ok:reason='unexpected_or_missing_transcript'
    elif vc=='Splice_Site':reason='splice_no_protein_coordinate'
    elif not coord_ok:reason='missing_or_mismatching_protein_coordinate'
    if conflict or not transcript_ok:cat='annotation_unknown'
    elif gene=='GATA3':
        if vc=='Splice_Site':cat='splice_site'
        elif not coord_ok:cat='annotation_unknown'
        elif vc.startswith('Frame_Shift') and lo<=start<=hi:cat='ZnF2_frameshift'
        elif vc in ['Nonsense_Mutation','Frame_Shift_Del','Frame_Shift_Ins']:
            if length is None:cat='frameshift_length_unknown' if vc.startswith('Frame_Shift') else 'annotation_unknown'
            else:cat='predicted_truncation' if length<L else ('predicted_extension' if length>L else 'frameshift_length_neutral')
        elif vc=='Nonstop_Mutation':cat='predicted_extension'
        elif vc=='Missense_Mutation':cat='missense'
        elif vc in ['In_Frame_Del','In_Frame_Ins']:cat='in_frame_indel'
        else:cat='other_qualifying'
    elif gene=='ESR1':
        if vc!='Missense_Mutation':cat='other_qualifying'
        elif not coord_ok:cat='annotation_unknown'
        elif start in [537,538]:cat='Y537_D538_missense'
        elif lo<=start<=hi:cat='other_LBD_missense'
        else:cat='outside_LBD_missense'
    else:
        if vc=='Nonsense_Mutation':cat='nonsense'
        elif vc.startswith('Frame_Shift'):cat='frameshift'
        elif vc=='Splice_Site':cat='splice_site'
        elif vc=='Missense_Mutation':cat=('HAT_missense' if lo<=start<=hi else 'outside_HAT_missense') if coord_ok else 'annotation_unknown'
        else:cat='other_qualifying'
    return dict(protein_change=hg or 'unknown',main_category=cat,first_changed_aa=start,last_changed_aa=end,
                predicted_protein_length=length,domain_relation=region,coordinate_usable=bool(coord_ok),
                reference_residue_status=ref_status,annotation_reason=reason,annotation_conflict=conflict)

def parser_checks(seqs):
    def test(g,vc,hg,cat,length=None):
        r=dict(Hugo_Symbol=g,Variant_Classification=vc,HGVSp_Short=hg,HGVSp='',Transcript_ID=REF[g][1])
        out=classify(r,seqs[g]);assert out['main_category']==cat,(hg,out)
        if length is not None:assert out['predicted_protein_length']==length
    test('GATA3','Frame_Shift_Del','p.R330Vfs*24','ZnF2_frameshift',352)
    test('GATA3','Frame_Shift_Del','p.S404Efs*103','predicted_extension',505)
    test('GATA3','Frame_Shift_Del','p.E359Afs*44','predicted_truncation',401)
    test('GATA3','Frame_Shift_Del','p.E359fs','frameshift_length_unknown')
    test('GATA3','Splice_Site','p.X308_splice','splice_site')
    test('ESR1','Missense_Mutation','p.Y537S','Y537_D538_missense')
    test('ESR1','Missense_Mutation','p.A537S','annotation_unknown')
    test('EP300','Missense_Mutation','p.D1399N','HAT_missense')

def main(refresh=False):
    # Classification rules are versioned in this module; methods accompany the data release.
    # This reproduction does not claim a new prospective protocol freeze.
    paths={'jobs':PRIOR/'LOCAL_ONLY_jobs.zip','context_results':PRIOR/'CONTEXT_RESULTS.parquet',
           'prior_code':Path(c.__file__),'local_cnv_code':Path(h.__file__),
           'raw_maf':h.RAW,'config':h.ANALYSIS_CONFIG,'script':Path(__file__)}
    paths.update({g+'_fasta':FASTA_DIR/f'{v[0]}.fasta' for g,v in REF.items()})
    hashes={k:h.digest(p) for k,p in paths.items()}
    old=h.load_record(PRIOR, 'INPUT_HASHES')['sha256']
    prepared=h.load_record(PRIOR, 'PREPARED')
    assert hashes['jobs']==prepared['jobs_sha256']
    for k in ['raw_maf','config']:assert hashes[k]==old[k],k
    assert hashes['prior_code']==old['code']
    assert hashes['local_cnv_code']==old['prior_code']
    h.save_record(HERE, 'INPUT_HASHES',{'sha256':hashes,'utc':datetime.now(timezone.utc).isoformat()})
    seqs={g:''.join(paths[g+'_fasta'].read_text().splitlines()[1:]) for g in REF}
    for g,seq in seqs.items():assert len(seq)==REF[g][2]
    parser_checks(seqs)
    jobs=read_jobs(paths['jobs']); chosen=[]; keep=set()
    for root,dx,target,g in CASES:
        j=next(j for j in jobs if (j['root'],j['diagnosis'],j['gene'])==(root,dx,g))
        chosen.append((root,dx,target,g,j));keep.update(j['frame'].sample_id)
    raw_path=HERE/'LOCAL_ONLY_raw_three_genes.parquet'
    if refresh or not raw_path.exists():
        parts=[]
        for chunk in pd.read_csv(h.RAW,sep='\t',dtype=str,usecols=COLS,chunksize=250000):
            parts.append(chunk[chunk.Tumor_Sample_Barcode.isin(keep)&chunk.Hugo_Symbol.isin(REF)])
        pd.concat(parts,ignore_index=True).to_parquet(raw_path,index=False)
    raw=pd.read_parquet(raw_path);cfg=json.loads(paths['config'].read_text())
    q=raw[raw.FILTER.eq('PASS')&raw.Annotation_Status.eq('SUCCESS')&raw.NCBI_Build.eq('GRCh37')&raw.Mutation_Status.str.lower().eq('somatic')&raw.Variant_Classification.isin(cfg['qualifying_classes'])].copy()
    qualified=q.drop_duplicates(DEDUP)
    oldr=pd.read_parquet(paths['context_results']).set_index(c.KEY)
    cohort_rows=[];count_rows=[];protein_rows=[];local_frames=[];sample_frames=[];audit=[];position_rows=[];aux_rows=[]
    for root,dx,target,g,job in chosen:
        frame=job['frame'].copy(); frozen=next(x for x in job['rows'] if x['subtype']==target)
        r=oldr.loc[(root,dx,g,target)];mask=frame.assigned_reference_key.eq(target)
        observed=[int(mask.sum()),int(frame.loc[mask,'positive'].sum()),int((~mask).sum()),int(frame.loc[~mask,'positive'].sum())]
        assert observed==[int(frozen[k]) for k in ['subtype_n','subtype_pos','other_subtypes_n','other_subtypes_pos']]
        allv=qualified[qualified.Hugo_Symbol.eq(g)&qualified.Tumor_Sample_Barcode.isin(frame.sample_id)].copy()
        counts=allv.groupby('Tumor_Sample_Barcode').size()
        assert frame.sample_id.map(counts).fillna(0).astype(int).eq(frame.target_gene_records).all()
        assert frame.sample_id.isin(counts.index).eq(frame.positive).all()
        assert frame.sample_id.is_unique and frame.analysis_id.is_unique
        complete=frame[frame.SAMPLE_TYPE.isin(['Primary','Metastasis'])].dropna(subset=['gain','loss','leave_gene_out_density'])
        base,info,_=h.exact_result(complete,target,'binary');active=complete.iloc[np.flatnonzero(info)]
        assert h.support(base,int(frozen['subtype_n']))
        for key in ['target_informative','control_informative','strata']:assert base[key]==r['support_'+key]
        assert len(active)==r.combined_n
        unique=[]
        for _,group in allv.groupby(ALLELE,dropna=False,sort=True):
            row=group.iloc[0].to_dict()
            row['annotation_conflict']=any(group[k].fillna('').nunique()>1 for k in ['Transcript_ID','Variant_Classification','HGVSp_Short','HGVSp'])
            row['qualifying_record_count']=len(group);row.update(classify(row,seqs[g]));unique.append(row)
        v=pd.DataFrame(unique)
        audit.append(dict(gene=g,root=root,diagnosis=dx,target=target,qualifying_records=len(allv),unique_sample_variants=len(v),
                          annotation_conflicts=int(v.annotation_conflict.sum()),coordinate_usable=int(v.coordinate_usable.sum()),
                          reference_mismatches=int(v.reference_residue_status.eq('mismatch').sum()),
                          distinct_transcripts=sorted(v.Transcript_ID.dropna().unique().tolist()),
                          full_short_HGVS_disagreements=int((v.HGVSp.map(norm).ne(v.HGVSp_Short.map(norm))&v.HGVSp.notna()&v.HGVSp_Short.notna()).sum())))
        for pop,popframe in [('frozen_evaluable',frame),('combined_informative',active)]:
            for side,f in [('target',popframe[popframe.assigned_reference_key.eq(target)]),('control',popframe[popframe.assigned_reference_key.ne(target)])]:
                w=v[v.Tumor_Sample_Barcode.isin(f.sample_id)].copy();carriers=int(f.positive.sum());n=len(f)
                assert w.Tumor_Sample_Barcode.nunique()==carriers
                keys=dict(root=root,diagnosis=dx,subtype=target,gene=g,population=pop,group=side)
                cr=keys|dict(evaluable_samples=n,mutation_carriers=carriers,variant_observations=len(w),
                      distinct_genomic_alleles=len(w.drop_duplicates(ALLELE[1:])),mutation_rate=carriers/n,
                      retained_from_frozen=n/(int(frozen['subtype_n']) if side=='target' else int(frozen['other_subtypes_n'])),
                      primary_samples=int(f.SAMPLE_TYPE.eq('Primary').sum()),metastasis_samples=int(f.SAMPLE_TYPE.eq('Metastasis').sum()),
                      coordinate_unknown_variants=int((~w.coordinate_usable).sum()))
                cohort_rows.append(cr)
                for cat in CATEGORIES[g]:
                    sub=w[w.main_category.eq(cat)];nc=sub.Tumor_Sample_Barcode.nunique()
                    count_rows.append(keys|dict(category=cat,variant_observations=len(sub),carriers=nc,total_mutation_carriers=carriers,
                           total_variant_observations=len(w),evaluable_samples=n,carrier_fraction_of_mutated=nc/carriers if carriers else None,
                           carrier_fraction_of_evaluable=nc/n,variant_fraction=len(sub)/len(w) if len(w) else None))
                for category,sub in w.groupby(['protein_change','Variant_Classification','main_category'],dropna=False):
                    protein_rows.append(keys|dict(protein_change=category[0],maf_classification=category[1],category=category[2],variant_observations=len(sub),carriers=sub.Tumor_Sample_Barcode.nunique()))
                for (pos,cat),sub in w[w.coordinate_usable].groupby(['first_changed_aa','main_category']):
                    position_rows.append(keys|dict(protein_position=int(pos),category=cat,variant_observations=len(sub),carriers=sub.Tumor_Sample_Barcode.nunique()))
                for (cat,region),sub in w.groupby(['main_category','domain_relation']):
                    aux_rows.append(keys|dict(category=cat,domain_relation=region,variant_observations=len(sub),carriers=sub.Tumor_Sample_Barcode.nunique()))
                for k,val in keys.items():w[k]=val
                local_frames.append(w)
                sf=f.copy()
                for k,val in keys.items():sf[k]=val
                sample_frames.append(sf)
    outputs={'COHORT_COUNTS':cohort_rows,'CATEGORY_COUNTS':count_rows,'PROTEIN_CHANGE_COUNTS':protein_rows,'POSITION_COUNTS':position_rows,'DOMAIN_RELATION_COUNTS':aux_rows}
    for name,rows in outputs.items():
        table=pd.DataFrame(rows);table.to_csv(HERE/(name+'.tsv'),sep='\t',index=False)
        assert not any(col in table for col in ['sample_id','analysis_id','Tumor_Sample_Barcode'])
    pd.concat(local_frames,ignore_index=True).to_parquet(HERE/'LOCAL_ONLY_annotated_variants.parquet',index=False)
    pd.concat(sample_frames,ignore_index=True).to_parquet(HERE/'LOCAL_ONLY_cohort_members.parquet',index=False)
    h.save_record(HERE, 'AUDIT',{'created_utc':datetime.now(timezone.utc).isoformat(),'cases':audit,
          'all_frozen_denominators_and_positives_reproduced':True,'combined_information_masks_reproduced':True,
          'new_statistical_tests':0,'raw_extract_sha256':h.digest(raw_path),
          'source_raw_maf_sha256':hashes['raw_maf'],'sources_refreshed_from_raw_maf':refresh})
    print(pd.DataFrame(cohort_rows).query("population=='combined_informative'")[['gene','group','evaluable_samples','mutation_carriers','variant_observations']].to_string(index=False))
    print('COMPLETE: frozen counts and conditional information sets reproduced; no new tests')
