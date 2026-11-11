import sys,json,time,os
from pathlib import Path
from replay_sao_cp_on_cache import case_names,calc_one
ROOT=Path(__file__).resolve().parent;CHECK=ROOT/'checkpoint';CHECK.mkdir(exist_ok=True)
names=case_names();s=int(sys.argv[1]);e=int(sys.argv[2]);start=time.monotonic()
for j,f in enumerate(names[s:e],s+1):
 p=CHECK/f'{j:03d}.json'
 if p.exists():continue
 row,ev=calc_one(f)
 payload={'row':row,'events':[{'case':row['case'],'dataset':row['dataset'],'kind':row['kind'],'event_idx':i,'miscoverage100':float(x.mean())} for i,x in enumerate(ev)]}
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(payload));os.replace(tmp,p)
 if j%6==0:print('[SAOCP]',j,'/',len(names),'elapsed',round(time.monotonic()-start,1),flush=True)
