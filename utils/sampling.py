import numpy as np

def monte_carlo_sampling(problem, num = 1): #Return list of random ints, not applicable in most algs, as onlt one start point is allowed...
    bounds = problem["Bounds"]
    dim_num = len(bounds)//2
    
    x0s = np.zeros((num, dim_num))
    
    rnd = np.random.RandomState(42 + 1)
    for i in range(dim_num):
        x0s[:,i] = rnd.uniform(bounds[2*i], bounds[2*i + 1], num)

    return x0s