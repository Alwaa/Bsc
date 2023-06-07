import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    xs = xx[:,0]
    ys = xx[:,1]
    
    c0 = 0.5*np.sin(2*np.pi*(2*ys - xs**2)) - xs - 2*ys + 1.5
    
    return {'c0' : np.array([float(c0[0][0] <= 0)])}