import numpy as np

def monte_carlo_sampling(problem, num = 1, seed = 42): #Return list of random ints, not applicable in most algs, as onlt one start point is allowed...
    bounds = problem["Bounds"]
    dim_num = len(bounds)//2
    
    x0s = np.zeros((num, dim_num))
    
    rnd = np.random.RandomState(seed)
    for i in range(dim_num):
        x0s[:,i] = rnd.uniform(bounds[2*i], bounds[2*i + 1], num)

    return x0s

def grid_sampling(problem, num_per_dim = 3):
    bounds = problem["Bounds"]
    dim_num = len(bounds)//2
    
    arrays = [np.linspace(bounds[2*i],bounds[2*i +1],num_per_dim + 2)[1:-1] for i in range(dim_num)]
    
    return cartesian(arrays)

# Taken from https://stackoverflow.com/a/1235363/21573210
def cartesian(arrays, out=None):     #arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
            
    return out
    
    