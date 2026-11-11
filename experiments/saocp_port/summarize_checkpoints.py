import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,zipfile,io
ROOT=Path(__file__).resolve().parent
files=sorted((ROOT/'checkpoint').glob('*.json'))
assert len(files)==108,len(files)
raw=[json.loads(p.read_text()) for p in files]
rows=pd.DataFrame([x['row'] for x in raw])
assert len(rows)==108 and rows['case'].nunique()==108
assert all(rows['n_eval']>0) and rows[['coverage','width','winkler']].notna().all().all()
assert set(rows['kind'])=={'none','level','variance'}
assert set(rows['dataset'])=={'ETTh1','ETTh2','ETTm1','ETTm2','exchange','illness'}
rows.to_csv(ROOT/'SAOCP_RESULTS_108.csv',index=False)
events=pd.DataFrame([e for x in raw for e in x['events']]);events.to_csv(ROOT/'SAOCP_EVENT_MISSES.csv',index=False)
CORRECTED=Path('/mnt/data/BRCP_Code_Audit_Fix_2026-11-11/corrected_results_h500/all_method_results_QUANTILE_FIXED_DEV_MATCHED.csv')
b=pd.read_csv(CORRECTED)
paired=rows.copy()
for m in ('BRCP','ACI(0.05)','BRCP-safe'):
 v=b.loc[b.method==m,['dataset','col_idx','kind','coverage','width','winkler']].copy()
 v=v.rename(columns={c:f'{c}_{m}' for c in ['coverage','width','winkler']})
 paired=paired.merge(v,on=['dataset','col_idx','kind'],how='left',validate='one_to_one')
assert len(paired)==108 and not paired.isna().any().any()
paired['ratio_Winkler_SAOCP_BRCP']=paired.winkler/paired.winkler_BRCP
paired['ratio_Winkler_SAOCP_ACI']=paired.winkler/paired['winkler_ACI(0.05)']
paired['coverage_SAOCP_minus_BRCP']=paired.coverage-paired.coverage_BRCP
paired.to_csv(ROOT/'SAOCP_PAIRED_108.csv',index=False)
summary=paired.groupby('kind').agg(n_series=('case','size'),coverage_sao=('coverage','mean'),coverage_brcp=('coverage_BRCP','mean'),coverage_aci=('coverage_ACI(0.05)','mean'),ratio_winkler_sao_brcp=('ratio_Winkler_SAOCP_BRCP','mean'),ratio_winkler_sao_aci=('ratio_Winkler_SAOCP_ACI','mean'))
summary.to_csv(ROOT/'SAOCP_SUMMARY.csv')
print('108-case SAOCP summary:');print(summary.to_string())
# Check paired event early miscoverage using the archived corrected trace windows.
P=Path('/mnt/data/BRCP_GPU_AUDIT_FIXED_2026-11-11-20261111T043547Z-1-001.zip')
other=[]
with zipfile.ZipFile(P) as z:
 for a in raw:
  row=a['row'];case=row['case']
  if row['kind']=='none':continue
  k='BRCP_GPU_AUDIT_FIXED_2026-11-11/by_series/'+case+'_events.npz'
  with np.load(io.BytesIO(z.read(k)),allow_pickle=False) as d:
   # [100 pre, 150 post]; first 100 post, avoids offset-120 return break.
   rr={m:d['event_'+m][:,100:200] for m in ('BRCP','ACI(0.05)')}
  assert all(len(v)==len(a['events']) for v in rr.values()),(case,{m:v.shape for m,v in rr.items()},len(a['events']))
  for j,e in enumerate(a['events']):
   other.append({'case':case,'dataset':row['dataset'],'kind':row['kind'],'event_idx':j,'SAOCP_mean_miss100':e['miscoverage100'],
                 'BRCP_mean_miss100':float(rr['BRCP'][j].mean()),'ACI_mean_miss100':float(rr['ACI(0.05)'][j].mean())})
ee=pd.DataFrame(other);ee.to_csv(ROOT/'PAIRED_EARLY_100_MISSRATE.csv',index=False)
print('Matched early event counts:',len(ee));print(ee.groupby('kind')[['SAOCP_mean_miss100','BRCP_mean_miss100','ACI_mean_miss100']].mean().to_string())
# Cluster per dataset, retain all six channels and ten events (where present).
rng=np.random.default_rng(811110);result=[]
for kind in ('none','level','variance'):
 p=paired[paired.kind==kind]
 ds=p.groupby('dataset')[['ratio_Winkler_SAOCP_BRCP','ratio_Winkler_SAOCP_ACI','coverage_SAOCP_minus_BRCP']].mean()
 assert len(ds)==6
 sample=ds.to_numpy()[rng.integers(6,size=(4000,6))].mean(axis=1)
 for j,metric in enumerate(ds.columns):
  result.append(dict(kind=kind,metric=metric,mean=float(ds.iloc[:,j].mean()),lower=float(np.quantile(sample[:,j],.025)),upper=float(np.quantile(sample[:,j],.975)),n_dataset_clusters=6))
 if kind!='none':
  p=ee[ee.kind==kind].groupby('dataset')[['SAOCP_mean_miss100','BRCP_mean_miss100','ACI_mean_miss100']].mean()
  p['SAOCP_minus_BRCP_early_miss']=p.SAOCP_mean_miss100-p.BRCP_mean_miss100
  sample=p['SAOCP_minus_BRCP_early_miss'].to_numpy()[rng.integers(6,size=(4000,6))].mean(axis=1)
  result.append(dict(kind=kind,metric='SAOCP_minus_BRCP_early_miss',mean=float(p['SAOCP_minus_BRCP_early_miss'].mean()),lower=float(np.quantile(sample,.025)),upper=float(np.quantile(sample,.975)),n_dataset_clusters=6))
pd.DataFrame(result).to_csv(ROOT/'DATASET_CLUSTER_BOOTSTRAP.csv',index=False)
manifest={'n_cases':108,'n_events':len(ee),'source_zip_sha256':hashlib.sha256(Path('/mnt/data/BRCP_Chronos_Bolt_GPU_Confirm_2026-11-11-20261111T035612Z-1-001.zip').read_bytes()).hexdigest(),
'baseline_sha256':hashlib.sha256(CORRECTED.read_bytes()).hexdigest(),
'code_sha256':hashlib.sha256((ROOT/'saocp_scoreport.py').read_bytes()).hexdigest(),
'checkpoint_count':len(files),'algorithm':'Salesforce SAOCP score-only source-derived port, not direct official package'}
(ROOT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('SUCCESS: 108/108 rows, full paired early results available')
