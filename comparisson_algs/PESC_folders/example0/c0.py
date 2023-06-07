import numpy as np


def main(job_id, params):
    xx = np.array([[params['x0'],params['x1']]])
    phi = np.linspace(-np.pi, np.pi, 100)
    r = (1 + np.sin(6*phi-.1))*0.3+0.2
    phie = np.arctan2(xx[:,1],xx[:,0])
    re = np.sqrt(np.sum(xx**2, axis=-1))
    c0 = - re + np.interp(phie,phi,r)
    
    return {'c0' : np.array([float(c0[0][0] <= 0)])}