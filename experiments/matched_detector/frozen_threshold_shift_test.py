#!/usr/bin/env python3
"""Additional exploratory third-stage shift test with frozen thresholds from SECOND-stage
1200-dev-seed calibration. Designed after the previous two audits; NOT preregistered.
Independent new seeds; no threshold optimization or selection on this test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fresh_detector_null_audit import get_paths,P,alarms
HERE=Path(__file__).resolve().parent
PROTO=dict(stage='exploratory third-stage after two inspected tests',
           known_null='stationary Gaussian AR1 phi=0.6',
           dev_threshold_origin='1200-seed development only from SECOND-stage stability analysis',
           score_h=5.0,data_h=13.0,seed_base=820210001,n_test=1000,
           shift_tau=650,first_120_steps_for_power=True,
           hypotheses='descriptive comparisons, no unconditional superiority claim',
           no_reselection='fix thresholds before new-seed test; publish all outcomes')
if __name__=='__main__':
    path=HERE/'FROZEN_THIRD_STAGE_PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==PROTO
    else:path.write_text(json.dumps(PROTO,indent=2)+'\n')
    rows=[]
    for seed in range(PROTO['seed_base'],PROTO['seed_base']+PROTO['n_test']):
        for label,kind in [('null',0),('level',1),('variance',2)]:
            obs,score=get_paths(seed,kind)
            for space,stream,h in [('data',obs,PROTO['data_h']),('score',score,PROTO['score_h'])]:
                a,b,hit,delay=alarms(stream,h)
                rows.append(dict(seed=seed,scenario=label,space=space,threshold=h,
                                 total_alarms=a,pre_alarm=b,
                                 post_alarm=hit,post_hit=int(hit>0),delay=delay))
    df=pd.DataFrame(rows)
    df.to_csv(HERE/'FROZEN_THIRD_STAGE_PER_SEED.csv',index=False)
    summary=df.groupby(['scenario','space']).agg(n=('seed','size'),alarms_per800=('total_alarms','mean'),
               post_hit_prob=('post_hit','mean'),median_delay=('delay',lambda x:float(np.median(x[x>=0])) if (x>=0).any() else float('nan'))).reset_index()
    summary.to_csv(HERE/'FROZEN_THIRD_STAGE_SUMMARY.csv',index=False)
    bootstrap=[]; rng=np.random.default_rng(820210999)
    for label in ['null','level','variance']:
        w=df[df.scenario==label].pivot(index='seed',columns='space',values=['total_alarms','post_hit'])
        for measure in ['total_alarms','post_hit']:
            diff=(w[(measure,'score')]-w[(measure,'data')]).to_numpy(dtype=float)
            ids=rng.integers(0,len(diff),(4000,len(diff)))
            lo,hi=np.quantile(diff[ids].mean(axis=1),[.025,.975])
            bootstrap.append(dict(scenario=label,measure=measure,mean_diff=float(diff.mean()),lo=float(lo),hi=float(hi)))
    pd.DataFrame(bootstrap).to_csv(HERE/'FROZEN_THIRD_STAGE_BOOTSTRAP.csv',index=False)
    null=summary[summary.scenario=='null'].set_index('space')
    gap=float(abs(null.loc['score','alarms_per800']-null.loc['data','alarms_per800']))
    status=dict(n=PROTO['n_test'],null_alarm_gap=gap,null_matched_under_second_stage_tol_0p15=bool(gap<=0.15),
                note='third-stage exploratory follow-up after seeing first and second results, not preregistered overall')
    (HERE/'FROZEN_THIRD_STAGE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(summary.to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))
    print('status',status)
