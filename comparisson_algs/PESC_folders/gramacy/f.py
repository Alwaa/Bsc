import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    if job_id == 1 :
        return {'f':np.array([1.])} #Helps performance on gramcy map..
    xs = xx[:,0]
    ys = xx[:,1]
    
    f = xs + ys
    
    return {'f':f[0]}