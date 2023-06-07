import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    xs = xx[:,0]
    ys = xx[:,1]
    
    f = np.cos(2*xs) * np.cos(ys) + np.sin(xs)
    
    return {'f':f[0]}