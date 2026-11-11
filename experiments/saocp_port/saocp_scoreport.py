"""SAOCP score-only port adapted from Salesforce/online_conformal (Apache-2.0).
Reference https://github.com/salesforce/online_conformal/blob/main/online_conformal/saocp.py
and utilities pinball_loss & pinball_loss_grad from online_conformal/utils.py.

Original SAOCP's Merlion BasePredictor and residuals container are bypassed for a
one-step, scores-only feed. Core expert expiry, coin-betting weights and
scale-free OGD updates follow the published source. This is a PORT and not a
byte-identical execution of the reference package. Used with alpha=0.10.
"""
from __future__ import annotations
import math

def pinball_loss(y,yhat,q):return max(q*(y-yhat),(1-q)*(yhat-y))
def pinball_grad(y,yhat,q):return -q*(y>yhat)+(1-q)*(y<yhat)

class OGDExpert:
    __slots__=('t','scale','base_lr','alpha','yhat','grad_norm','lifetime','z','wz','s_t')
    def __init__(self,t,scale,alpha,yhat_0,g=8):
        self.t=t;self.scale=scale;self.base_lr=scale/math.sqrt(3)
        self.alpha=alpha;self.yhat=yhat_0;self.grad_norm=0.
        u=0;tmp=t
        while tmp%2==0:
            tmp//=2;u+=1
        self.lifetime=g*2**u
        self.z=0.;self.wz=0.;self.s_t=0
    def weight(self):return 0.0 if self.s_t==0 else self.z/self.s_t*(1+self.wz)
    def update(self,y,meta_loss):
        w=self.weight()
        g=min(1.0,max(-float(w>0),(meta_loss-pinball_loss(y,self.yhat,1-self.alpha))/self.scale/max(self.alpha,1-self.alpha)))
        self.z+=g;self.wz+=g*w;self.s_t+=1
        grad=pinball_grad(y,self.yhat,1-self.alpha)
        self.grad_norm+=grad**2
        if self.grad_norm:
            self.yhat=max(0.,self.yhat-self.base_lr/math.sqrt(self.grad_norm)*grad)
    @property
    def expired(self):return self.s_t>self.lifetime

class SAOCPScore:
    def __init__(self,alpha,max_scale,lifetime=8):
        if max_scale<=0:raise ValueError('max_scale must be positive')
        self.alpha=alpha;self.scale=max_scale;self.lifetime=lifetime;self.t=1
        self.experts={}
    def predict(self):
        if not self.experts:return 0.
        prior={key:1/(key**2*(1+math.floor(math.log2(key)))) for key in self.experts}
        total_prior=sum(prior.values());prior={k:v/total_prior for k,v in prior.items()}
        p={k:prior[k]*max(0.,self.experts[k].weight()) for k in self.experts}
        s=sum(p.values());p={k:v/s for k,v in p.items()} if s>0 else prior
        return sum(p[k]*self.experts[k].yhat for k in self.experts)
    def update(self,y):
        prev=self.predict()
        # Remove expired experts and initiate an expert at this time t.
        self.experts={k:v for k,v in self.experts.items() if not v.expired}
        self.experts[self.t]=OGDExpert(self.t,self.scale,self.alpha,prev,self.lifetime)
        meta_loss=pinball_loss(y,self.predict(),1-self.alpha)
        for e in self.experts.values():e.update(y,meta_loss)
        self.t+=1

def replay(scores,warm=300,alpha=0.10,lifetime=8):
    import numpy as np
    arr=np.asarray(scores,float)
    if len(arr)<warm+1:raise ValueError('short stream')
    scale=max(np.max(np.abs(arr[:warm]))*np.sqrt(3),1e-9)
    model=SAOCPScore(alpha,max_scale=scale,lifetime=lifetime)
    for s in arr[:warm]:model.update(float(s))
    n=len(arr)-warm
    err=np.empty(n,dtype=np.uint8);width=np.empty(n);wink=np.empty(n)
    for i,s in enumerate(arr[warm:]):
        rad=model.predict()
        err[i]=s>rad
        width[i]=2*rad
        wink[i]=2*rad+2/alpha*max(0.,s-rad)
        model.update(float(s))
    return dict(coverage=float(1-err.mean()),width=float(width.mean()),winkler=float(wink.mean()),n_eval=n,err=err)
