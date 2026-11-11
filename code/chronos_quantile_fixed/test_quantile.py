"""Regression tests for the conformal quantile error discovered 2026-11-11."""
import numpy as np
from brcp.conformal import conformal_quantile
from brcp.fastbuf import order_stat


def run_tests():
    rng=np.random.default_rng(11)
    ntest=0
    for n in [10,19,20,30,60,100,300,1200]:
        for alpha in [.01,.05,.1,.15,.2,.5,.9]:
            x=np.arange(1,n+1,dtype=float)
            k=int(np.ceil((n+1)*(1-alpha)))
            # Original experiment uses a finite-max convention when k>n.
            expected=float(x[min(k,n)-1])
            obtained=conformal_quantile(x,alpha)
            assert obtained==expected,(n,alpha,k,obtained,expected)
            y=rng.normal(size=n)
            ys=np.sort(y)
            result=conformal_quantile(y,alpha)
            assert result==ys[min(k,n)-1]
            ntest+=2
    assert conformal_quantile(np.arange(1,21),.1)==19.
    assert conformal_quantile(np.arange(1,301),.1)==271.
    print(f'PASS {ntest+2} quantile regression assertions')
if __name__=='__main__':run_tests()
