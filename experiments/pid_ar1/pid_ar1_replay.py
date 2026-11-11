#!/usr/bin/env python3
"""Exploratory causal PID with 1-step AR(1) scorecaster on existing Chronos cache.

Cites: Angelopoulos/Candes/Tibshirani (2023) Conformal PID, and official
https://github.com/aangelopoulos/conformal-time-series/core/methods.py
(quantile_integrator_log_scorecaster). The referenced code supplies a master
P+log-tangent I+external scorecast interface. THIS implementation is not claimed
source-identical: AR(1) scorecaster in place of their example ThetaModel, with
pre-evaluation warm calibration, finite nonnegative radius and fixed-time seed.
"""
import io, zipfile, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CACHE_ZIP=Path('/mnt/data/BRCP_Chronos_Bolt_GPU_Confirm_2026-11-11-20261111T035612Z-1-001.zip')
CORRECTED=Path('/mnt/data/BRCP_Code_Audit_Fix_2026-11-11/corrected_results_h500/all_method_results_QUANTILE_FIXED_DEV_MATCHED.csv')
PROTO=json.loads((HERE/'PROTOCOL.json').read_text())
SEED=20261111
ALPHA=float(PROTO['alpha']); WARM=int(PROTO['warm']); L=int(PROTO['ar1_window']);CSAT=float(PROTO['Csat'])


def causal_ar1_forecasts(scores, window=L):
    """At index t predicts score[t] from scores[:t] only; vectorized O(T)."""
    s=np.asarray(scores,dtype=np.float64)
    n=len(s)
    idx=np.arange(n)
    out=np.zeros(n,dtype=np.float64)
    if n<=3:return out
    xs=s[:-1];ys=s[1:]
    def pref(a):return np.concatenate(([0.0],np.cumsum(a,dtype=np.float64)))
    p_x,p_y,p_x2,p_xy=map(pref,(xs,ys,xs**2,xs*ys))
    e=np.maximum(idx-1,0)   # n pairs (x=s[j],y=s[j+1]) available before score[t]
    b=np.maximum(e-int(window),0)
    m=e-b
    use=m>=5
    xx=(p_x[e]-p_x[b]) / np.maximum(m,1)
    yy=(p_y[e]-p_y[b]) / np.maximum(m,1)
    v=(p_x2[e]-p_x2[b])/np.maximum(m,1)-xx*xx
    c=(p_xy[e]-p_xy[b])/np.maximum(m,1)-xx*yy
    phi=np.clip(np.divide(c,v,out=np.zeros_like(c),where=v>1e-12),-.95,.95)
    last=s[np.maximum(idx-1,0)]
    pred=np.maximum(0.0,yy+phi*(last-xx))
    out[use]=pred[use]
    out[~use]=np.maximum(0.0,yy[~use])
    out[0]=np.nan
    return out


def conformal_warm_quantile(s,alpha=ALPHA):
    n=len(s)
    k=int(np.ceil((n+1)*(1-alpha)))
    return float(np.partition(s,k-1)[k-1]) if 1<=k<=n else float('inf')


def run_pid(scores, eta,ki, ar=None, warm=WARM, alpha=ALPHA,csat=CSAT):
    """Issue q_t first; observe score_t; then update controller for q_{t+1}."""
    x=np.asarray(scores,float)
    scale=conformal_warm_quantile(x[:warm],alpha)
    if not np.isfinite(scale) or scale<=0: raise ValueError('bad warm scale')
    s=x/scale
    casts=causal_ar1_forecasts(s) if ar is None else ar
    n=len(s);tvals=np.arange(warm,n)
    # Initialize P at warm-up so first q matches warm quantile (no lookahead).
    p=conformal_warm_quantile(s[:warm],alpha)-casts[warm]
    excess=0.0
    errs=np.zeros(len(tvals),np.uint8); widths=np.zeros(len(tvals)); wink=np.zeros(len(tvals))
    capcount=0; nonpositive=0
    for i,t in enumerate(tvals):
        angle=excess*np.log(t-warm+2)/(csat*(t-warm+2))
        if abs(angle)>=np.pi/2:
            capcount+=1
            angle=np.clip(angle,-(np.pi/2-1e-5),np.pi/2-1e-5)
        integ=ki*np.tan(angle)
        q_unclipped=casts[t]+p+integ
        if q_unclipped<0:nonpositive+=1
        q=max(0.0,q_unclipped)*scale
        miss=x[t]>q
        errs[i]=miss
        widths[i]=2*q
        wink[i]=2*q+(2/alpha)*max(0.,x[t]-q)
        # P and I update: no s[t] until after q_t has been produced.
        p+=eta*(float(miss)-alpha)
        excess+=float(miss)-alpha
    return dict(coverage=1-errs.mean(),width=widths.mean(),winkler=wink.mean(),
                n_eval=len(errs),capcount=capcount,zero_radius=nonpositive,err=errs)


def extract_cases():
    with zipfile.ZipFile(CACHE_ZIP) as z:
        cases=sorted([name for name in z.namelist() if '/forecast_cache/' in name and name.endswith('.npz') and not name.endswith('.partial.npz')])
        for name in cases:
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as data:
                d={k:data[k].copy() for k in ('scores','origins','taus')}
                meta=json.loads(str(data['meta']))
            yield name,d,meta


def events_100(miss,origins,taus,warm=WARM):
    # Index j in full `origins` translates into evaluation miss index j-warm.
    arr=[]
    for tau in taus:
        j=int(np.searchsorted(origins,tau))-warm
        if j>=0 and j+100<=len(miss):
            arr.append(miss[j:j+100])
    return np.array(arr,dtype=np.uint8).reshape((-1,100))


def run():
    if not CACHE_ZIP.exists():raise FileNotFoundError(CACHE_ZIP)
    if not CORRECTED.exists():raise FileNotFoundError(CORRECTED)
    dev=[];test=[]
    for name,d,meta in extract_cases():
        row=(name,d,meta)
        if 'col000' in name:dev.append(row)
        else:test.append(row)
    assert len(dev)==6 and len(test)==108,(len(dev),len(test))
    print('Split:',len(dev),'dev and',len(test),'evaluation streams',flush=True)
    # Tuning is stream-macro: each (dataset, break-kind) receives one equal vote.
    tune=[]
    for eta in PROTO['candidate_eta']:
      for ki in PROTO['candidate_KI']:
        vals=[]
        for _,d,_ in dev:
            scores=d['scores']
            r=run_pid(scores,eta,ki)
            scale=conformal_warm_quantile(scores[:WARM]);vals.append(r['winkler']/scale)
        tune.append(dict(eta=eta,KI=ki,dev_mean_normalized_winkler=np.mean(vals),dev_sd_normalized_winkler=np.std(vals)))
    tab=pd.DataFrame(tune).sort_values(['dev_mean_normalized_winkler','eta','KI']);tab.to_csv(HERE/'DEV_TUNING.csv',index=False)
    choice=tab.iloc[0]; eta,ki=float(choice['eta']),float(choice['KI'])
    print('Development-only chosen eta,KI:',eta,ki,flush=True)
    selected={"eta":eta,"KI":ki,"chosen_by":"minimum mean normalized Winkler across 6 unperturbed dev streams","review":"posthoc_exploratory"}
    (HERE/'SELECTED_PARAMETERS.json').write_text(json.dumps(selected,indent=2))
    results=[];events=[]; event_scores=[]
    for j,(name,d,meta) in enumerate(test):
        r=run_pid(d['scores'],eta,ki)
        ev=events_100(r['err'],d['origins'],d['taus']);family=meta['dataset'];kind=meta['kind']
        base=dict(case=Path(name).stem,dataset=family,col_idx=meta['col_idx'],kind=kind,
                  method='PID-AR1-causal',coverage=r['coverage'],width=r['width'],winkler=r['winkler'],
                  n_eval=r['n_eval'],n_events=len(ev),zero_radius=r['zero_radius'],tangent_caps=r['capcount'])
        results.append(base)
        if len(ev):
            # per-event observed miss rate; does not estimate population CD of the definition.
            for ei,a in enumerate(ev):events.append(dict(case=base['case'],dataset=family,kind=kind,event_idx=ei,miss100=float(a.mean())))
            # Per-series aggregate mean of observed excess violations within first 100 positions.
            # Need ensembles across 6 channels for population CD; below grouped separately.
            event_scores.append(dict(case=base['case'],dataset=family,kind=kind,mean_excess_misses=float(100*np.maximum(ev.mean(axis=1)-ALPHA,0).mean())))
        if (j+1)%18==0:print('Evaluation completed',j+1,'/',len(test),flush=True)
    pd.DataFrame(results).to_csv(HERE/'PID_AR1_RESULTS_108.csv',index=False)
    pd.DataFrame(events).to_csv(HERE/'PID_AR1_EVENT_MISSES.csv',index=False)
    pd.DataFrame(event_scores).to_csv(HERE/'PID_AR1_EVENT_EXCESS_PATHWISE.csv',index=False)
    t=pd.DataFrame(results)
    b=pd.read_csv(CORRECTED)
    # Corrected results include name preserved in source archives; combine on exact dataset / channel / kind.
    for m in ['BRCP','ACI(0.05)','BRCP-safe']:
        filt=b.loc[b.method==m,['dataset','col_idx','kind','coverage','width','winkler']].copy()
        filt=filt.rename(columns={c:f'{c}_{m}' for c in ['coverage','width','winkler']})
        t=t.merge(filt,on=['dataset','col_idx','kind'],how='left',validate='one_to_one')
    assert t.notna().all().all(), 'Missing baseline join'
    t['ratio_to_aci']=t['winkler']/t['winkler_ACI(0.05)']
    t['ratio_to_brcp']=t['winkler']/t['winkler_BRCP']
    t['brcp_minus_pid_coverage']=t['coverage_BRCP']-t['coverage']
    t.to_csv(HERE/'PID_AR1_PAIRED_108.csv',index=False)
    agg=t.groupby('kind').agg(n_series=('case','size'),coverage_pid=('coverage','mean'),coverage_brcp=('coverage_BRCP','mean'),
                              coverage_aci=('coverage_ACI(0.05)','mean'),
                              winkler_ratio_pid_aci=('ratio_to_aci','mean'),
                              winkler_ratio_pid_brcp=('ratio_to_brcp','mean'),
                              zero_radius=('zero_radius','sum'),tangent_caps=('tangent_caps','sum'))
    agg.to_csv(HERE/'SUMMARY_BY_KIND.csv')
    print(agg.to_string(),flush=True)
    # Equal dataset-cluster weighting, descriptive CIs: resample six DATASET identifiers.
    rng=np.random.default_rng(SEED)
    bs=[]
    for kind in ('none','level','variance'):
        subset=t[t.kind==kind]
        ds=np.array(sorted(subset.dataset.unique()))
        mat=subset.groupby('dataset')[['ratio_to_brcp','ratio_to_aci','brcp_minus_pid_coverage']].mean().loc[ds].to_numpy()
        idx=rng.integers(len(ds),size=(4000,len(ds)))
        out=mat[idx].mean(axis=1)
        for k,label in enumerate(('PID/BRCP_Winkler','PID/ACI_Winkler','BRCP_minus_PID_coverage')):
            bs.append(dict(kind=kind,estimand=label,estimate=float(mat[:,k].mean()),lower=float(np.quantile(out[:,k],.025)),upper=float(np.quantile(out[:,k],.975)),n_dataset_clusters=len(ds)))
    pd.DataFrame(bs).to_csv(HERE/'DATASET_CLUSTER_BOOTSTRAP.csv',index=False)
    manifest={'source_zip_sha256':hashlib.sha256(CACHE_ZIP.read_bytes()).hexdigest(),'baseline_sha256':hashlib.sha256(CORRECTED.read_bytes()).hexdigest(),
              'n_dev_streams':len(dev),'n_test_streams':len(test),'selected_params':selected,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scientific_restrictions':'Exploratory posthoc; compares one AR1 PID scorecaster, not exact authors Theta baseline; no global performance superiority.'}
    (HERE/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    return agg

if __name__=='__main__':run()
