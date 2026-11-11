"""EXPLORATORY follow-on: enforce positive KI, SELECT ON SAME SIX DEV streams.
This was added AFTER observing an unrestricted development selection KI=0.
Do not present as independent prospective confirmation or hide the KI=0 result.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from pid_ar1_replay import extract_cases,run_pid,events_100,conformal_warm_quantile,WARM,CORRECTED,SEED
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'FORCED_POSITIVE_KI';OUT.mkdir(exist_ok=True)
DEV=[];TEST=[]
for name,d,meta in extract_cases():
 (DEV if 'col000' in name else TEST).append((name,d,meta))
assert len(DEV)==6 and len(TEST)==108
# Restrict pre-existing DEV_TUNING table to positive-KI configurations. No test metric used for selection.
dev=pd.read_csv(ROOT/'DEV_TUNING.csv'); sub=dev[dev.KI>0].sort_values(['dev_mean_normalized_winkler','eta','KI'])
sub.to_csv(OUT/'DEV_FORCED_POSITIVE_KI.csv',index=False)
eta,ki=map(float,[sub.iloc[0].eta,sub.iloc[0].KI]);print('FORCED positive integral selected on dev only:',eta,ki,flush=True)
res=[];events=[]
for i,(name,d,meta) in enumerate(TEST):
 r=run_pid(d['scores'],eta,ki)
 key=Path(name).stem
 ev=events_100(r['err'],d['origins'],d['taus'])
 res.append(dict(case=key,dataset=meta['dataset'],col_idx=meta['col_idx'],kind=meta['kind'],method='PID-AR1-positive-integral',coverage=r['coverage'],width=r['width'],winkler=r['winkler'],n_eval=r['n_eval'],n_events=len(ev),zero_radius=r['zero_radius'],tangent_caps=r['capcount']))
 for j,row in enumerate(ev):events.append(dict(case=key,kind=meta['kind'],event_idx=j,miscoverage100=float(row.mean())))
 if (i+1)%36==0: print('DONE',i+1,flush=True)
pd.DataFrame(res).to_csv(OUT/'PID_AR1_POSITIVE_KI_108.csv',index=False)
pd.DataFrame(events).to_csv(OUT/'PID_AR1_POSITIVE_KI_EVENTS.csv',index=False)
b=pd.read_csv(CORRECTED)
t=pd.DataFrame(res)
for m in ['BRCP','ACI(0.05)']:
 v=b.loc[b.method==m,['dataset','col_idx','kind','coverage','width','winkler']].copy()
 v=v.rename(columns={c:f'{c}_{m}' for c in ['coverage','width','winkler']})
 t=t.merge(v,on=['dataset','col_idx','kind'],how='left',validate='one_to_one')
assert len(t)==108 and not t.isna().any().any()
t['ratio_to_aci']=t.winkler/t['winkler_ACI(0.05)'];t['ratio_to_brcp']=t.winkler/t.winkler_BRCP
t.to_csv(OUT/'PID_POSITIVE_KI_PAIRED.csv',index=False)
summary=t.groupby('kind').agg(n_series=('case','size'),coverage_pid=('coverage','mean'),winkler_ratio_pid_aci=('ratio_to_aci','mean'),winkler_ratio_pid_brcp=('ratio_to_brcp','mean'),zero_radius=('zero_radius','sum'),tangent_caps=('tangent_caps','sum'))
summary.to_csv(OUT/'SUMMARY.csv');print(summary,flush=True)
json.dump({'selected_on_dev':{'eta':eta,'KI':ki},'note':'Exploratory post-hoc positive integral; original unrestricted dev choice KI=0 retained in parent folder'},open(OUT/'STATUS.json','w'),indent=2)
