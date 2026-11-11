"""Independent audit of already-saved BR-CP real benchmark error curves.
Preserves the 35 series as resampling units, keeping windows within a series intact.
No new raw-data experiments are claimed by this script.
"""
import numpy as np,pandas as pd
from pathlib import Path
base=Path(__file__).resolve().parents[2] / 'results'
out=Path(__file__).resolve().parents[1] / 'results'
rows=pd.read_csv(base/'real_rows_main_variance.csv')
z=np.load(base/'real_curves_main_variance.npz')
ref=rows[rows.method=='BRCP'][['dataset','series','n_breaks']].reset_index(drop=True)
assert len(ref)==35 and ref.n_breaks.sum()==322
alpha=.1; N=322; eps_nom=2*np.sqrt(alpha*(1-alpha)/N); H=200; PRE=150
methods=['BRCP','BRCP-safe','ACI(0.05)','ACI(0.01)','AgACI','ConformalPID']
cs={}
for method in methods:
    arr=z[f'variance|3.0|{method}'][:,PRE:PRE+H]
    assert arr.shape==(322,200), arr.shape
    cs[method]=np.stack([chunk.sum(axis=0) for chunk in np.split(arr,np.cumsum(ref.n_breaks.to_numpy())[:-1])])

def debt(m,eps): return np.maximum(0, m-alpha-eps).sum(axis=-1)
def recovery(m,eps,period=24):
    # Published TTR-C marg: first 24-step moving average meeting tolerance,
    # no persistence after that point; censored set to 201.
    sums=np.cumsum(np.pad(m,[(0,0),(1,0)]),axis=1)
    rolls=(sums[:,period:]-sums[:,:-period])/period
    ok=rolls<=(alpha+eps)
    first=ok.argmax(axis=1)
    return np.where(ok.any(axis=1),first,H+1)

# Resample 35 entire series with replacement; publication-style equal-window
# estimand preserves each series' window count as its weight.
rng=np.random.default_rng(20261106);B=2500; n=len(ref)
ix=rng.integers(0,n,size=(B,n),dtype=np.int16)
weights=np.take(ref.n_breaks.to_numpy(),ix).sum(axis=1)
vals={}
for method in methods:
    m=cs[method][ix].sum(axis=1)/weights[:,None]
    vals[method]=dict(cd=debt(m,eps_nom), cd0=debt(m,0), ttr=recovery(m,eps_nom),m=m)

summary=[]
for method in methods:
   v=vals[method];
   full=cs[method].sum(axis=0)/N
   for metric,observed in [('CD',debt(full,eps_nom)),('CD_eps0',debt(full,0)),('TTR_marg',recovery(full[None,:],eps_nom)[0])]:
       samp=v[{'CD':'cd','CD_eps0':'cd0','TTR_marg':'ttr'}[metric]]
       summary.append({'method':method,'metric':metric,'estimate':float(observed),
                       'ci_low':float(np.quantile(samp,.025)), 'ci_high':float(np.quantile(samp,.975)),
                       'bootstrap_median':float(np.median(samp)),
                       'failure_fraction':float(np.mean(samp>H)) if metric=='TTR_marg' else ''})
for baseline in ['ACI(0.05)','ACI(0.01)','AgACI','ConformalPID']:
   diffs=vals[baseline]['cd']-vals['BRCP']['cd']
   summary.append({'method':f'{baseline} minus BRCP','metric':'paired_CD_diff',
                   'estimate':float(debt(cs[baseline].sum(axis=0)/N,eps_nom)-debt(cs['BRCP'].sum(axis=0)/N,eps_nom)),
                   'ci_low':float(np.quantile(diffs,.025)),'ci_high':float(np.quantile(diffs,.975)),
                   'bootstrap_median':float(np.median(diffs)),'failure_fraction':''})
pd.DataFrame(summary).to_csv(out/'real_cluster_bootstrap_summary.csv',index=False)

sensitivity=[]
for eps in [0,.01,.02,float(eps_nom),.042,.05]:
   for method in methods:
      full=cs[method].sum(axis=0)/N
      sensitivity.append({'epsilon':eps,'method':method,'CD':float(debt(full,eps)),
                          'TTR_marg':float(recovery(full[None,:],eps)[0])})
pd.DataFrame(sensitivity).to_csv(out/'real_epsilon_sensitivity.csv',index=False)

# series-balanced variant (each of 35 series contributes 1/35)
weighted=[]
for method in methods:
   full=np.mean(cs[method]/ref.n_breaks.to_numpy()[:,None],axis=0)
   weighted.append({'method':method, 'series_equal_CD':float(debt(full,eps_nom)),
                    'window_equal_CD':float(debt(cs[method].sum(axis=0)/N,eps_nom))})
pd.DataFrame(weighted).to_csv(out/'real_series_vs_window_weighting.csv',index=False)

print('epsilon main:',eps_nom)
print('---cluster summary---')
print(pd.DataFrame(summary).to_string(index=False))
print('---epsilon sensitivity---')
print(pd.DataFrame(sensitivity).pivot(index='epsilon',columns='method',values='CD').round(3))
print('---weights---')
print(pd.DataFrame(weighted))
