import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    xs = xx[:,0]
    ys = xx[:,1]
    
    c1 = xs**2 + ys**2 -1.5
    
    return {'c1' : np.array([float(c1[0][0] <= 0)])}