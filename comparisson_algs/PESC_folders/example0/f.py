import numpy as np
from scipy.stats import multivariate_normal

def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    rnd = np.random.RandomState(42)
    K = 42
    Mu = (rnd.rand(K,2)-.5)*2
    w = rnd.rand(K)+1
    Sigma = (((0.011)*rnd.rand(K)[:,None,None]+.041)*np.identity(2)[None]) + 0*rnd.randn(K, 2, 2) * 0.025
    
    f = -np.sum([w[i]*multivariate_normal.pdf(xx, mean=Mu[i], cov=Sigma[i])
                for i in range(len(Mu))],axis=0)
    if np.shape(f) == ():
        f = f[None]
    
    return {'f':np.array([f[0]])} #Fixed output format...