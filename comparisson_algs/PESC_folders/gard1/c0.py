import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    xs = xx[:,0]
    ys = xx[:,1]
    
    c0 = np.cos(xs)*np.cos(ys) - np.sin(xs)*np.sin(ys) + 0.5
    
    return {'c0' : np.array([float(c0[0][0] <= 0)])}