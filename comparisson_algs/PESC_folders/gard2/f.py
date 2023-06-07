import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    
    xs = xx[:,0]
    ys = xx[:,1]
    f = np.sin(xs) + ys
    
    return {'f':f[0]}