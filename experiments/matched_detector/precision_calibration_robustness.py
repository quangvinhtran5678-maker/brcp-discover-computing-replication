#!/usr/bin/env python3
"""Second independent-sample calibration robustness check, motivated by failed first test.
EXPLORATORY FOLLOW-UP, not the first frozen test, and NOT Chronos/BRCP prediction intervals.
No adjustment is permitted based on newly generated second evaluation outcomes.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import fresh_detector_null_audit as first
from fresh_detector_null_audit import series_from_noise,count_alarms
ROOT=Path(__file__).resolve().parent
Q=dict(status='SECOND-STAGE EXPLORATORY ROBUSTNESS AFTER 90-SEED GATE FAILURE',
    null='stationary Gaussian AR(1)', phi=0.6, n=1100, warm=300, tau=650,
    anchor_size=250,min_anchor=60,k=0.8,winsor=4.0,
    grid=[round(x,2) for x in np.arange(3.0,31.01,0.5)]+[32,36,40,48,56,64,80,100],
    target=0.5,tolerance=0.15,n_dev=1200,n_test=900,
    dev_seed_base=520210001,test_seed_base=620210001,
    kind='NULL only: no break, detector alarm rate matching; follow-up designed after first failure',
    selection='closest dev null alarm rate to 0.5; tie -> larger threshold',
    release_all_results='including failures; do not retune on second-stage test')

def gen(seed):
    rng=np.random.default_rng(int(seed));noise=rng.normal(size=Q['n']); initial=rng.normal(0,1/np.sqrt(1-Q['phi']**2))
    return series_from_noise(noise,initial,Q['phi'],Q['tau'],0,0.,1.)

def alarms(stream,h):
    return count_alarms(stream,h,Q['warm'],Q['anchor_size'],Q['min_anchor'],Q['k'],Q['winsor'],Q['tau'],120)[0]

if __name__=='__main__':
    path=ROOT/'FROZEN_PROTOCOL_PRECISION_CALIBRATION.json'
    txt=json.dumps(Q,sort_keys=True,indent=2)
    if path.exists():assert json.loads(path.read_text())==Q,'Frozen protocol modified'
    else:path.write_text(txt+'\n')
    print('protocol_sha256',hashlib.sha256(txt.encode()).hexdigest(),flush=True)
    dev=[gen(seed) for seed in range(Q['dev_seed_base'],Q['dev_seed_base']+Q['n_dev'])]
    rows=[]; sel={}
    for ix,space in [(0,'data'),(1,'score')]:
        for h in Q['grid']:
            counts=[alarms(streams[ix],float(h)) for streams in dev]
            rows.append(dict(space=space,h=h,alarms=sum(counts),n=Q['n_dev'],alarms_per800=np.mean(counts)))
        cand=[r for r in rows if r['space']==space]
        best=sorted(cand,key=lambda r:(abs(r['alarms_per800']-Q['target']),-r['h']))[0]
        sel[space]=best['h']
    pd.DataFrame(rows).to_csv(ROOT/'PRECISION_DEVELOPMENT_GRID.csv',index=False)
    (ROOT/'FROZEN_PRECISION_THRESHOLDS.json').write_text(json.dumps(sel,indent=2)+'\n')
    print('selected',sel,flush=True)
    outcome=[]
    for seed in range(Q['test_seed_base'],Q['test_seed_base']+Q['n_test']):
        obs,score=gen(seed)
        outcome.append(dict(seed=seed,data_alarms=alarms(obs,sel['data']),score_alarms=alarms(score,sel['score'])))
    df=pd.DataFrame(outcome)
    df.to_csv(ROOT/'PRECISION_FRESH_TEST_PER_SEED.csv',index=False)
    d=(df.score_alarms-df.data_alarms).to_numpy(dtype=float)
    rng=np.random.default_rng(730210001)
    idx=rng.integers(0,len(d),(5000,len(d)))
    interval=np.quantile(d[idx].mean(axis=1),[.025,.975])
    result=dict(n_dev=Q['n_dev'],n_test=Q['n_test'],selected_h=sel,
                data_per800=float(df.data_alarms.mean()),score_per800=float(df.score_alarms.mean()),
                gap_score_minus_data=float(d.mean()),abs_gap=float(abs(d.mean())),
                ci95=[float(x) for x in interval],
                test_pass=bool(abs(d.mean())<=Q['tolerance']),
                label='SECOND exploratory calibration stability audit; known stationary synthetic null; not TSFM or full baseline')
    (ROOT/'PRECISION_TEST_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
