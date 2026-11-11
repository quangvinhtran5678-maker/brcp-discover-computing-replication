"""Reproducible exact-equivalence test of accelerated AR1 detector to original Page-CUSUM code."""
from fresh_detector_null_audit import get_paths,P,alarms
from detectors_reference import ScoreCUSUM

def reference(stream,h):
    d=ScoreCUSUM(anchor_size=P['anchor_size'],min_anchor=P['min_anchor'],
                 two_sided=True,stat='std',k=P['k'],h=h,winsor=P['winsor'])
    d.set_anchor(stream[P['warm']-P['anchor_size']:P['warm']])
    refill=[];cnt=before=post=0;first=-1
    for i in range(P['warm'],len(stream)):
        if d.armed:
            if d.update(stream[i]):
                cnt+=1
                if i<P['post_tau']:before+=1
                if P['post_tau']<=i<P['post_tau']+P['post_window']:
                    post+=1
                    if first<0:first=i-P['post_tau']
                d.disarm();refill=[]
        else:
            refill.append(stream[i])
            if len(refill)>=P['min_anchor']:
                d.set_anchor(refill);refill=[]
    return cnt,before,post,first

def main():
    n=0
    for seed in [320210001,420210001,420210007,420210101,420210299]:
        for kind in [0,1,2]:
            obs,score=get_paths(seed,kind)
            for h in [4,5,6,10,16]:
                for stream in [obs,score]:
                    got=alarms(stream,h)
                    expect=reference(stream,h)
                    assert got==expect,(seed,kind,h,got,expect)
                    n+=1
    print(f'PASS {n} exact reference-equivalence checks.')

if __name__=='__main__':main()
