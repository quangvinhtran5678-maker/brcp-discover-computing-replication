"""BR-CP post-hoc code correction audit (CPU-only, no new Chronos inference).

Input is an existing GPU-run directory with RUN_MANIFEST.json and forecast_cache/*.npz.
This program corrects one-off conformal order statistic and expands detector
threshold development grid. It outputs rerun results, matched EMPIRICAL alarm
checks, per-series normalized ratios, exploratory CD100 (pulse length=120),
and a source-provenance audit. This is an exploratory correction, not an
independent preregistered confirmation and not a true-null false-positive test.

Run:
 python audit_cached.py --input-root /path/to/e6247c5f3e92 --output-root /path/to/corrected --workers 2
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from runner import make_layers, replay, empirical_alarm_rate
from brcp.metrics import coverage_debt

PERIOD={'ETTh1':24,'ETTh2':24,'ETTm1':96,'ETTm2':96,'exchange':7,'illness':52}
BASE_METHODS=['SplitCP','RollingCP','ACI(0.005)','ACI(0.05)','AgACI','ConformalPID','BRCP','BRCP-safe','BRCP-data']


def source_sha(root):
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*.py')):
        if p.parent.name=='__pycache__':continue
        h.update(p.relative_to(Path(__file__).parent).as_posix().encode()+b'\0');h.update(p.read_bytes())
    return h.hexdigest()


def calibrate(root,cfg,paths, out):
    grids=[]
    for space,hs in [('score',[4,6,8,10,12,16,20]),('data',[4,6,8,10,12,16,20,30,40,60,80,100,120,160,200,240,320,400,500,640,800,1000])]:
        for h in hs:
            alarm_sum=step_sum=0
            for file in paths:
                with np.load(file,allow_pickle=False) as z:s=z['scores'] if space=='score' else z['targets']
                count,steps=empirical_alarm_rate(s,space,cfg['k'],h,cfg['warm']);alarm_sum+=count;step_sum+=steps
            grids.append(dict(space=space,h=h,alarms=alarm_sum,steps=step_sum,alarms_per800=800*alarm_sum/step_sum))
    grid=pd.DataFrame(grids);grid.to_csv(out/'development_threshold_grid.csv',index=False)
    selected={}
    for sp in ['score','data']:
        choices=grid[grid.space==sp].copy();choices['distance']=(choices.alarms_per800-cfg['target_alarms_per800']).abs()
        top=choices.sort_values(['distance','h'],ascending=[True,False]).iloc[0]
        selected[sp]=int(top.h)
    return selected


def calc(args):
    path,out_path,cfg,chosen,sourcehash,model_revision=args
    path=Path(path);out=Path(out_path);z=np.load(path,allow_pickle=False)
    with z:
        scores,targets,preds,origins,taus=[z[x] for x in ('scores','targets','preds','origins','taus')]
        meta=json.loads(str(z['meta']))
    if not (len(scores)==len(targets)==len(preds)==len(origins)):
        raise AssertionError(f'Invalid cache dimensions in {path}')
    if not np.allclose(scores,abs(targets-preds),rtol=0,atol=1e-6):
        raise AssertionError(f'Cached scores != abs(y-yhat) in {path}')
    ds,col,kind=meta['dataset'],meta['column'],meta['kind']
    if model_revision and meta['revision'] != model_revision:
        raise AssertionError(f'Model revision mismatch {path}')
    if meta['input_sha256']!=sourcehash[ds]:
        raise AssertionError(f'Input hash mismatch {path}')
    name=path.stem;row_path=out/f'{name}_methods.csv';trace_path=out/f'{name}_events.npz'
    if row_path.exists() and trace_path.exists():return name
    methods=make_layers(cfg['alpha'],PERIOD[ds],cfg['window'],cfg['k'],chosen['score'],chosen['data'])
    rows=[];traces={};P=cfg['pre'];H=cfg['horizon_cd']
    for method,f in methods.items():
        err,width,wink,alarms,clips=replay(f(),scores,targets,preds,cfg['warm'],cfg['alpha'])
        if not (np.isfinite(width).all() and np.isfinite(wink).all()):raise AssertionError('Nonfinite metrics')
        event_windows=[]
        for tau in taus:
            index=int(np.searchsorted(origins,tau))-cfg['warm']
            if index-P>=0 and index+H<=len(err):event_windows.append(err[index-P:index+H])
        traces['event_'+method]=np.asarray(event_windows,dtype=np.uint8).reshape((-1,P+H))
        rows.append(dict(dataset=ds,col=col,col_idx=meta['col_idx'],kind=kind,method=method,
            n_eval=len(err),coverage=float(1-err.mean()),width=float(width.mean()),winkler=float(wink.mean()),
            n_alarms=len(alarms),clips=clips,n_events=len(event_windows),source_sha256=meta['input_sha256'],
            model_revision=meta['revision']))
    tmp=row_path.with_suffix('.tmp.csv');pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,row_path)
    tmp=trace_path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**traces);os.replace(tmp,trace_path)
    return name


def postprocess(out,cfg):
    df=pd.concat([pd.read_csv(p) for p in sorted((out/'by_series').glob('*_methods.csv'))],ignore_index=True)
    assert len(df)==972,(len(df), 'expected 972 method rows')
    df.to_csv(out/'all_method_results_CORRECTED.csv',index=False)
    df.groupby(['kind','method'],as_index=False).agg(n_series=('col','size'),avg_coverage=('coverage','mean'),avg_width=('width','mean'),avg_winkler=('winkler','mean'),mean_alarms=('n_alarms','mean')).to_csv(out/'summary_CORRECTED.csv',index=False)
    n0=df[df.kind=='none'];rates={}
    for method in ['BRCP','BRCP-data']:
        a=n0[n0.method==method];rates[method]=float(a.n_alarms.sum())/float(a.n_eval.sum())*800
    tolerance=float(cfg['test_rate_tolerance_per800'])
    rates.update(diff_per800=abs(rates['BRCP']-rates['BRCP-data']),tolerance=tolerance,
       status='matched empirical alarm rates within tolerance; not a true FPR guarantee' if abs(rates['BRCP']-rates['BRCP-data'])<=tolerance else 'UNMATCHED')
    piv=df.pivot(index=['dataset','col','col_idx','kind'],columns='method',values=['coverage','width','winkler'])
    norm=[]
    for (ds,col,idx,kind),row in piv.iterrows():
        norm.append(dict(dataset=ds,col=col,col_idx=idx,kind=kind,
          brcp_vs_aci_winkler_ratio=float(row['winkler']['BRCP']/row['winkler']['ACI(0.05)']),
          brcp_vs_aci_width_ratio=float(row['width']['BRCP']/row['width']['ACI(0.05)']),
          brcp_minus_aci_coverage=float(row['coverage']['BRCP']-row['coverage']['ACI(0.05)'])))
    pd.DataFrame(norm).to_csv(out/'within_series_ratios_CORRECTED.csv',index=False)
    H=min(100,int(cfg['segment']))
    if H >= cfg['segment']:raise ValueError('CD horizon must end before second intervention boundary')
    P=int(cfg['pre']);al=float(cfg['alpha'])
    event_series={}
    for file in sorted((out/'by_series').glob('*_events.npz')):
        parts=file.stem.split('_');ds,idx,kind=parts[0],int(parts[1][3:]),parts[2]
        if kind=='none':continue
        with np.load(file,allow_pickle=False) as z:
            for method in BASE_METHODS:
                data=z['event_'+method]
                if len(data):event_series[(ds,idx,kind,method)]=data[:,P:P+H].mean(axis=0)
    cd_rows=[];curve_rows=[]
    for kind in ['level','variance']:
        for method in BASE_METHODS:
            keys=[k for k in event_series if k[2]==kind and k[3]==method]
            m=np.stack([event_series[k] for k in keys]).mean(axis=0)
            cd_rows.append(dict(kind=kind,method=method,cd100=coverage_debt(m,al,0,H),
                n_channels=len(keys),post100_mean_miss=float(m.mean()),note='exploratory; 100 offsets before pulse termination at 120'))
            curve_rows += [dict(kind=kind,method=method,offset=j,mean_miscoverage=float(val)) for j,val in enumerate(m)]
    pd.DataFrame(cd_rows).to_csv(out/'cd100_descriptive_CORRECTED.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(out/'cd100_curves_CORRECTED.csv',index=False)
    return rates,df


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args();root=a.input_root;out=a.output_root;out.mkdir(parents=True,exist_ok=True);(out/'by_series').mkdir(exist_ok=True)
    manifest=json.loads((root/'RUN_MANIFEST.json').read_text());cfg=manifest['config']
    dev=[root/'forecast_cache'/f'{ds}_col000_none.npz' for ds in cfg['datasets']]
    for p in dev:
        if not p.exists():raise FileNotFoundError(p)
    chosen=calibrate(root,cfg,dev,out)
    tasks=sorted(p for p in (root/'forecast_cache').glob('*.npz') if 'col000' not in p.stem and not p.stem.endswith('.partial'))
    if len(tasks)!=108:raise RuntimeError(f'Expected 108 TEST cache files, got {len(tasks)}')
    t=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures=[ex.submit(calc,(p,out/'by_series',cfg,chosen,manifest['input_source_hashes'],manifest['model_revision'])) for p in tasks]
        for idx,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result()
            if idx%18==0:print('CPU REPLAY',idx,'/',len(tasks),'seconds',round(time.perf_counter()-t,2),flush=True)
    rates,df=postprocess(out,cfg)
    (out/'SCIENTIFIC_STATUS.json').write_text(json.dumps(dict(input_model_revision=manifest['model_revision'],input_gpu=manifest['gpu'],
       input_source_hashes=manifest['input_source_hashes'],development_thresholds=chosen,test_empirical_alarm_rates=rates,
       n_method_rows=len(df),n_series_scenarios=108,source_digest=source_sha(root),
       warning='Post-hoc methodological/code repair; no fresh hold-out; not a final confirmatory study'),indent=2))
    print('Selected dev thresholds',chosen,'TEST',rates,'ROWS',len(df),'SECONDS',round(time.perf_counter()-t,1),flush=True)
if __name__=='__main__':main()
