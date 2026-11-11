import pandas as pd,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parents[2] / 'results'
z=np.load(p/'curves.npz')
meta=pd.read_csv(p/'table_main.csv')
# all 21 non-null configurations / exactly paired method streams
names=sorted(set(k.rsplit('|',1)[-1] for k in z.files))
families=['level','variance','trend','season','ar','predictability','slow_drift']
EPS=[0.,.02,2*np.sqrt(.1*.9/200),.05,.07]
rows=[]
for family in families:
 for mag in [2.,3.,4.]:
  for method in ['BRCP','ACI(0.05)','ACI(0.005)','AgACI','ConformalPID']:
   key=f'{family}|{mag}|{method}'
   if key not in z.files:continue
   m=z[key][:,150:350].mean(axis=0)
   for eps in EPS:
    rows.append({'break_type':family,'mag':mag,'method':method,'epsilon':eps,'CD':np.maximum(0,m-.1-eps).sum()})
df=pd.DataFrame(rows)
pivot=df.pivot_table(index=['epsilon','method'],values='CD',aggfunc='mean').reset_index();pivot.to_csv(str(Path(__file__).resolve().parents[1] / 'results' / 'simulation_epsilon_sensitivity.csv'),index=False)
print('Mean CD over 21 cells:');print(pivot.pivot(index='epsilon',columns='method',values='CD').round(3))
byfam=meta[(meta.mag==3)&(meta.method.isin(['BRCP','ACI(0.05)']))][['break_type','method','CD']].pivot(index='break_type',columns='method',values='CD')
print('Magnitude3 family diffs:');print(byfam)
