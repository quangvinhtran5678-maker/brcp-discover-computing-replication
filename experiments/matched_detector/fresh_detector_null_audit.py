#!/usr/bin/env python3
"""Prospectively frozen, independent new synthetic null/shift detector audit.
Does not use previously observed Chronos validation series, does NOT evaluate
BRCP intervals, and does not establish pretraining corpus disjointness.
Quantile / controller implementations are not used in this detector-only test.

Protocol is written before any development or evaluation streams are drawn.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE=Path(__file__).resolve().parent
P = dict(date='2026-11-11', study='independent frozen synthetic detector-only test',
         distinction='new IID seeds for stationary AR(1) null and persistent changes; NO Chronos checkpoint',
         phi=0.6, n=1100, warm=300, post_tau=650, post_window=120,
         k=0.8, anchor_size=250, min_anchor=60, winsor=4.0,
         target_alarms_per_800=0.5, target_tolerance=0.2,
         h_grid=[2,3,4,5,6,8,10,12,16,20,30,40,60,80,100,150,200,300,500],
         dev_seed_base=320210001, n_dev=90,
         test_seed_base=420210001, n_test=300,
         level_delta_innovation_units=2.0, variance_sigma_multiplier=3.0,
         target_name='mean alarms in exact synthetic AR1 null over 800 evaluated observations',
         weighting='one independent generated series per seed',
         notes='Power evaluated on separate persistent level/variance interventions; no tuning on test.')

@njit(cache=True)
def count_alarms(stream,h,warm,anchor_size,min_anchor,k,winsor,tau,post_window):
    n=len(stream)
    mu=0.0
    for j in range(warm-anchor_size,warm): mu+=stream[j]
    mu/=anchor_size
    sd=0.0
    for j in range(warm-anchor_size,warm):
        d=stream[j]-mu; sd+=d*d
    sd=(sd/anchor_size)**0.5+1e-9
    su=sdn=0.0
    armed=True;buf=np.empty(min_anchor,dtype=np.float64); nb=0
    all_alarms=0;before=0;in_window=0;first=-1
    for i in range(warm,n):
        if armed:
            z=(stream[i]-mu)/sd
            if z> winsor:z=winsor
            if z<-winsor:z=-winsor
            su=max(0.0,su+z-k)
            sdn=max(0.0,sdn-z-k)
            if su>=h or sdn>=h:
                all_alarms+=1
                if i<tau: before+=1
                if tau<=i<tau+post_window:
                    in_window+=1
                    if first<0:first=i-tau
                armed=False;nb=0;su=sdn=0.0
        else:
            buf[nb]=stream[i];nb+=1
            if nb==min_anchor:
                mu=0.0
                for b in range(min_anchor):mu+=buf[b]
                mu/=min_anchor
                sd=0.0
                for b in range(min_anchor):
                    d=buf[b]-mu;sd+=d*d
                sd=(sd/min_anchor)**0.5+1e-9
                armed=True;su=sdn=0.0;nb=0
    return all_alarms,before,in_window,first

@njit(cache=True)
def series_from_noise(noise,initial,phi,tau,kind,delta,mult):
    n=len(noise)
    y=np.empty(n,dtype=np.float64)
    scores=np.empty(n,dtype=np.float64)
    prev=initial
    for t in range(n):
        e=noise[t]
        if t>=tau:
            if kind==1:e=e+(1.0-phi)*delta
            if kind==2:e=e*mult
        y[t]=phi*prev+e
        scores[t]=abs(y[t]-phi*prev)
        prev=y[t]
    return y,scores

def get_paths(seed,kind):
    rng=np.random.default_rng(int(seed))
    noise=rng.standard_normal(P['n'])
    initial=rng.normal(0,1/np.sqrt(1-P['phi']**2))
    return series_from_noise(noise,initial,P['phi'],P['post_tau'],kind,
                             P['level_delta_innovation_units'],P['variance_sigma_multiplier'])

def alarms(stream, h):
    return count_alarms(stream,float(h),P['warm'],P['anchor_size'],P['min_anchor'],
                        P['k'],P['winsor'],P['post_tau'],P['post_window'])

def run_grid():
    rows=[]
    dev_seeds=np.arange(P['dev_seed_base'],P['dev_seed_base']+P['n_dev'])
    seq=[get_paths(int(i),0) for i in dev_seeds]
    for space_idx,space in enumerate(['data','score']):
        for h in P['h_grid']:
            alarms_list=[alarms(paths[space_idx],h)[0] for paths in seq]
            mean=float(np.mean(alarms_list))
            rows.append(dict(space=space,h=h,n=P['n_dev'],total_alarms=int(sum(alarms_list)),
                             mean_alarms_per800=mean))
    df=pd.DataFrame(rows)
    selections={}
    for space in ['data','score']:
        x=df[df.space==space].copy()
        x['distance']=(x.mean_alarms_per800-P['target_alarms_per_800']).abs()
        # deterministic selection: closest to target, tie favors larger threshold
        x=x.sort_values(['distance','h'],ascending=[True,False])
        selections[space]=float(x.iloc[0].h)
    return df,selections

def evaluate(selections):
    rows=[]
    for j in range(P['n_test']):
        seed=int(P['test_seed_base']+j)
        for kind_name,kind in [('null',0),('level',1),('variance',2)]:
            y,s=get_paths(seed,kind)
            for space,stream in [('data',y),('score',s)]:
                a,b,post,first=alarms(stream,selections[space])
                rows.append(dict(seed=seed,scenario=kind_name,detector=space,
                                 h=selections[space],all_alarms=a,
                                 pre_alarm_count=b,post_alarm_count=post,
                                 post_hit=int(post>0),first_detection_delay=first))
    return pd.DataFrame(rows)

def summarize(df):
    out=[]
    for (scenario,detector),g in df.groupby(['scenario','detector']):
        out.append(dict(scenario=scenario,detector=detector,h=float(g.h.iloc[0]),
                        n_seeds=len(g), alarms_per800=float(g.all_alarms.mean()),
                        mean_pre_alarm_count=float(g.pre_alarm_count.mean()),
                        probability_post_alarm=float(g.post_hit.mean()),
                        median_delay_conditional=float(g.loc[g.first_detection_delay>=0,'first_detection_delay'].median())
                                 if (g.first_detection_delay>=0).any() else None))
    return pd.DataFrame(out)

def pair_bootstrap(df,n=5000):
    rng=np.random.default_rng(451234)
    results=[]
    for scenario in ['null','level','variance']:
        x=df[df.scenario==scenario].pivot(index='seed',columns='detector',values=['all_alarms','post_hit'])
        a=x[('all_alarms','score')].to_numpy(dtype=float)-x[('all_alarms','data')].to_numpy(dtype=float)
        b=x[('post_hit','score')].to_numpy(dtype=float)-x[('post_hit','data')].to_numpy(dtype=float)
        ids=rng.integers(0,len(a),(n,len(a)))
        for measure,v in [('alarms_per800_diff_score_minus_data',a),('post_detection_probability_diff_score_minus_data',b)]:
            bs=v[ids].mean(axis=1)
            results.append(dict(scenario=scenario,measure=measure,mean=float(v.mean()),
                                ci95lo=float(np.quantile(bs,.025)),ci95hi=float(np.quantile(bs,.975)),n_seeds=len(v)))
    return pd.DataFrame(results)

if __name__=='__main__':
    path=HERE/'FROZEN_PROTOCOL_SYNTHETIC_NULL.json'
    protocol=json.dumps(P,sort_keys=True,indent=2)
    if path.exists():
        assert json.loads(path.read_text())==P,'Frozen protocol changed: stop.'
    else:path.write_text(protocol+'\n')
    print('Protocol SHA256:',hashlib.sha256(protocol.encode()).hexdigest(),flush=True)
    grid,selected=run_grid()
    grid.to_csv(HERE/'DEVELOPMENT_THRESHOLD_GRID.csv',index=False)
    (HERE/'FROZEN_SELECTED_THRESHOLDS.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('Frozen selected:',selected,flush=True)
    ev=evaluate(selected)
    ev.to_csv(HERE/'FRESH_HELDOUT_PER_SEED.csv',index=False)
    summ=summarize(ev)
    summ.to_csv(HERE/'FRESH_HELDOUT_SUMMARY.csv',index=False)
    paired=pair_bootstrap(ev)
    paired.to_csv(HERE/'FRESH_PAIRED_BOOTSTRAP.csv',index=False)
    print(summ.to_string(index=False),flush=True)
    print(paired.to_string(index=False),flush=True)
    r=summ[summ.scenario=='null'];diff=abs(r.iloc[0].alarms_per800-r.iloc[1].alarms_per800)
    (HERE/'FRESH_STUDY_STATUS.json').write_text(json.dumps(dict(frozen_selected=selected,test_null_rate_gap=diff,
                         test_null_matched_within_tolerance=bool(diff<=P['target_tolerance']),
                         qualification='standalone synthetic AR1 detector-only new holdout; not Chronos model or coverage'),indent=2)+'\n')
