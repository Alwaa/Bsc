import numpy as np
from scipy.stats import multivariate_normal

#--------------------------------#
# Bounds and such
#--------------------------------#

boundtype = "square"
bounds = (-1,1,-1,1)

#################################
# Example problem
#################################
# x is 2d (or more) w and Mu samde d as x, Sigma square with ds same as x
def costf(x,w,Mu,Sigma):
    
    f = -np.sum([w[i]*multivariate_normal.pdf(x, mean=Mu[i], cov=Sigma[i])
                for i in range(len(Mu))],axis=0)
    if np.shape(f) == ():
        return f[None]
    return f

def constraintf(x,phi,r):
    xx = x[None] if x.ndim == 1 else x # In case of single value input

    phie = np.arctan2(xx[...,1],xx[...,0])
    re = np.sqrt(np.sum(xx**2, axis=-1))
    return re - np.interp(phie,phi,r) #<= 0

rnd = np.random.RandomState(42)
K = 42
Mu = (rnd.rand(K,2)-.5)*2
w = rnd.rand(K)+1
Sigma = (((0.011)*rnd.rand(K)[:,None,None]+.041)*np.identity(2)[None]) + 0*rnd.randn(K, 2, 2) * 0.025

# xin = np.linspace(-1, 1, 301) #uniform 1d input
# xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T #2d input
phi = np.linspace(-np.pi, np.pi, 100)
r = (1 + np.sin(6*phi-.1))*0.3+0.2

# Stored functions
example0 = {"Bound Type" : boundtype,
            "Bounds" : bounds,
            "Cost Function (x)":  lambda x: costf(x,w,Mu,Sigma),
            "Constraint Functions (z)": [lambda z: constraintf(z,phi,r)]} #[lambda z: 1-constraintf(z,phi,r)]}

#################################
# XXXXXX problem
#################################

if __name__ == "__main__":
    print(costf(np.array([1.,0.]),w,Mu,Sigma))
    print(constraintf(np.array([1.,0.]),phi,r)[None]) 