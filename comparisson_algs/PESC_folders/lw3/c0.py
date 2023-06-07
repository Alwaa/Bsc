import numpy as np


def main(job_id, params):
    x = np.array([[params['x0'],params['x1'],params['x2'],params['x3']]])
    
    c0 = -0.5 + np.sin(x[:,0] + 2*x[:,1]) -np.cos(x[:,2]*np.cos(x[:,3]))
    
    return {'c0' : np.array([float(c0[0][0] <= 0)])}