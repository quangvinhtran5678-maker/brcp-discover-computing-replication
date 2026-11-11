"""Paired detector-plus-reset-only ablation using existing DGP and frozen forecaster protocol.
This does not reproduce the published Calleo algorithm; it isolates contributions
from BRCP graded step-size, post-alarm boost and small-buffer tolerance.
"""
import sys,time
import pandas as pd,numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import brcp.experiment as E
from brcp.conformal import BRCP
from brcp.detectors import ScoreCUSUM
orig=E.build_methods

def built(cfg,subset=None,tau_idx=10**9):
    methods=orig(cfg,subset=None,tau_idx=tau_idx)
    def detector_reset():
        det=ScoreCUSUM(anchor_size=250,k=.65,h=6.,min_anchor=60,two_sided=True,stat='std')
        return BRCP(alpha=cfg.alpha,window=cfg.window,detector=det,
                    detector_space='score',period=cfg.period, use_reset=True,
                    use_inflation=False,use_boost=False,gamma_mode='constant')
    methods['DetectorResetOnly']=detector_reset
    return {k:v for k,v in methods.items() if subset is None or k in subset}
E.build_methods=built
if __name__=='__main__':
    from pathlib import Path
    out=Path(__file__).resolve().parents[1] / 'results'
    t0=time.time();cfg=E.Config()
    for fam in ['level','variance','trend','season','ar','predictability','slow_drift']:
        st=time.time()
        rows,curves=E.run_grid(cfg,[fam],[2.0,3.0,4.0],range(200),method_names=['DetectorResetOnly'],verbose=False)
        pd.DataFrame(rows).to_csv(out/f'reset_only_rows_{fam}.csv',index=False)
        np.savez_compressed(out/f'reset_only_curves_{fam}.npz',**{f'{a}|{b}|{c}':v for (a,b,c),v in curves.items()})
        for (bt,mag,name),x in curves.items():
            m=x[:,150:350].mean(axis=0);eps=2*np.sqrt(.1*.9/len(x))
            print(f'{bt} {mag} CD={np.maximum(0,m-.1-eps).sum():.3f}',flush=True)
        print('done',fam,'seconds',round(time.time()-st,1),flush=True)
    print('DONE ALL',round(time.time()-t0,1),flush=True)
