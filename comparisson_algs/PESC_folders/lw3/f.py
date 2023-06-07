import numpy as np


def main(job_id, params):
    x = np.array([[params['x0'],params['x1'],params['x2'],params['x3']]])
    
    f = 0.5*np.sum(x**4 - 16*(x**2) + 5*x,axis = 1)
    
    return {'f':f[0]}