#!/usr/bin/env python3
"""Independent numeric consistency check for the 2026-11-11 post-hoc addendum.
Requires pandas/numpy. Does not rerun Chronos GPU or interpret source-code equivalence.
"""
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent.parent
S=ROOT/'experiments'/'saocp_port'; P=ROOT/'experiments'/'pid_ar1'
s=pd.read_csv(S/'SAOCP_PAIRED_108.csv'); p=pd.read_csv(P/'FORCED_POSITIVE_KI'/'PID_POSITIVE_KI_PAIRED.csv')
assert len(s)==108 and len(p)==108
for z in (s,p):
    assert z[['dataset','col_idx','kind']].duplicated().sum()==0
    assert set(z.kind)=={'none','level','variance'}
    assert all((z.groupby('kind').size()==36).values)
    assert z[['coverage','winkler','winkler_BRCP','coverage_BRCP']].notna().all().all()
    assert all(z.coverage.between(0,1))
assert set(map(tuple,s[['dataset','col_idx','kind']].values))==set(map(tuple,p[['dataset','col_idx','kind']].values))
e=pd.read_csv(S/'SAOCP_EVENT_MISSES.csv');assert len(e)==564
assert e.groupby('kind').size().to_dict()=={'level':282,'variance':282}
check={
    'none':(0.9208264272,0.9836075495,0.8963619105,0.9575335925),
    'level':(0.9203599279,0.9878886675,0.8962988013,0.9626921758),
    'variance':(0.9155345708,0.9572138831,0.8927478,1.050480),
}
for kind,values in check.items():
    ss=s[s.kind==kind]; pp=p[p.kind==kind]
    derived=(ss.coverage.mean(),ss.ratio_Winkler_SAOCP_BRCP.mean(),pp.coverage.mean(),pp.ratio_to_brcp.mean())
    assert all(abs(float(a)-float(b))<0.0001 for a,b in zip(derived,values)),(kind,derived,values)
    print(f'{kind:8} SAOCP coverage={derived[0]:.6f}, SAOCP/BRCP Winkler={derived[1]:.6f}; PID+I coverage={derived[2]:.6f}, PID+I/BRCP Winkler={derived[3]:.6f}')
ci=pd.read_csv(S/'DATASET_CLUSTER_BOOTSTRAP.csv')
row=ci[(ci.kind=='variance') & (ci.metric=='ratio_Winkler_SAOCP_BRCP')].iloc[0]
assert abs(row.lower-0.933258)<1e-6 and abs(row.upper-0.975265)<1e-6
print('VERIFY PASS: 108 paired scenarios per method, 564 event windows, six-cluster bootstrap CI agrees.')
