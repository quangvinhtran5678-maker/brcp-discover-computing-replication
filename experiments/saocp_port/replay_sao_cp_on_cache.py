"""Source-derived SAOCP score port, paired Chronos-Bolt replay on cached scores.
NO new GPU forward pass. Avoids testing on future samples during initialization.
"""
import io,json,zipfile,hashlib,concurrent.futures,os,time
from pathlib import Path
import numpy as np,pandas as pd
from saocp_scoreport import replay

ROOT=Path(__file__).resolve().parent
CACHE_ZIP=Path('/mnt/data/BRCP_Chronos_Bolt_GPU_Confirm_2026-11-11-20261111T035612Z-1-001.zip')
CORRECTED=Path('/mnt/data/BRCP_Code_Audit_Fix_2026-11-11/corrected_results_h500/all_method_results_QUANTILE_FIXED_DEV_MATCHED.csv')

def case_names():
 with zipfile.ZipFile(CACHE_ZIP) as z:
  return sorted([f for f in z.namelist() if '/forecast_cache/' in f and f.endswith('.npz') and 'col000' not in f and not f.endswith('.partial.npz')])

def calc_one(f):
 with zipfile.ZipFile(CACHE_ZIP) as z:
  with np.load(io.BytesIO(z.read(f)),allow_pickle=False) as d:
   scores=d['scores'];taus=d['taus'];origins=d['origins'];meta=json.loads(str(d['meta']))
 r=replay(scores)
 event_traces=[]
 for tau in taus:
  j=int(np.searchsorted(origins,tau))-300
  if j>=0 and j+100<=len(r['err']):event_traces.append(r['err'][j:j+100])
 events=np.array(event_traces,dtype=np.uint8).reshape((-1,100))
 return dict(case=Path(f).stem,dataset=meta['dataset'],col_idx=meta['col_idx'],kind=meta['kind'],method='SAOCP_score_port',n_eval=r['n_eval'],coverage=r['coverage'],width=r['width'],winkler=r['winkler'],n_events=len(events)),events

def main():
 names=case_names();assert len(names)==108,len(names)
 t=time.monotonic()
 results=[];evtrows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  for i,(row,events) in enumerate(pool.map(calc_one,names),1):
   results.append(row)
   for j,seq in enumerate(events):evtrows.append({**{k:row[k] for k in ('case','dataset','kind')},'event_idx':j,'miscoverage100':float(seq.mean())})
   if i%18==0:print('[SAOCP]',i,'/',len(names),'elapsed',round(time.monotonic()-t,1),flush=True)
 d=pd.DataFrame(results); d.to_csv(ROOT/'SAOCP_108.csv',index=False)
 pd.DataFrame(evtrows).to_csv(ROOT/'SAOCP_EVENTS.csv',index=False)
 b=pd.read_csv(CORRECTED)
 for m in ('BRCP','ACI(0.05)','BRCP-safe'):
  v=b.loc[b.method==m,['dataset','col_idx','kind','coverage','width','winkler']].copy()
  v=v.rename(columns={c:f'{c}_{m}' for c in ['coverage','width','winkler']})
  d=d.merge(v,on=['dataset','col_idx','kind'],how='left',validate='one_to_one')
 assert len(d)==108 and not d.isna().any().any()
 d['ratio_vs_BRCP']=d.winkler/d.winkler_BRCP
 d['ratio_vs_ACI']=d.winkler/d['winkler_ACI(0.05)']
 d.to_csv(ROOT/'SAOCP_PAIRED_108.csv',index=False)
 a=d.groupby('kind').agg(n_series=('case','size'),coverage_sao=('coverage','mean'),coverage_brcp=('coverage_BRCP','mean'),coverage_aci=('coverage_ACI(0.05)','mean'),ratio_winkler_sao_brcp=('ratio_vs_BRCP','mean'),ratio_winkler_sao_aci=('ratio_vs_ACI','mean'))
 a.to_csv(ROOT/'SAOCP_SUMMARY.csv');print(a.to_string(),flush=True)
 # Resample 6 dataset clusters, all channels/scenarios within dataset retained.
 rng=np.random.default_rng(20261111);res=[]
 for kind in ('none','level','variance'):
  sub=d[d.kind==kind];g=sub.groupby('dataset')[['ratio_vs_BRCP','ratio_vs_ACI']].mean().to_numpy()
  ix=rng.integers(6,size=(4000,6));boot=g[ix].mean(1)
  for i,l in enumerate(('SAOCP/BRCP_Winkler','SAOCP/ACI_Winkler')):
   res.append(dict(kind=kind,contrast=l,estimate=float(g[:,i].mean()),ci_lower=float(np.quantile(boot[:,i],.025)),ci_upper=float(np.quantile(boot[:,i],.975)),n_clusters=6))
 pd.DataFrame(res).to_csv(ROOT/'CLUSTER_BOOTSTRAP.csv',index=False)
 (ROOT/'RUN_MANIFEST.json').write_text(json.dumps({'n_cases':len(names),'elapsed_s':round(time.monotonic()-t,1),'input_cache_sha256':hashlib.sha256(CACHE_ZIP.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'SAOCP score-only PORT (not direct official package)'},indent=2))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
