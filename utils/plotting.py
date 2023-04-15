import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

NUM_OF_OBJECTIVE_FUNCTIONS = 1 #Multi-objective optimization is not covered. One should conbine and weigh the multiple objectives into one in thoscases

def vizualize_toy(xs: NDArray[np.float64],
                  objs: NDArray[np.float64], #objective values
                  eval_type: NDArray[np.bool8] | bool,
                  problem,
                  decoupled: bool = False):
    num_samples = 301
    ## Fetching problem info ##
    bounds = problem["Bounds"]
    if problem["Bound Type"] == "square":
        xin = np.linspace(bounds[0], bounds[1], num_samples)
        xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
    else:
        raise Exception("Not yet Implemented non-square bounds")
    
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"] #TODO: rewrite to multiple (s)
    ## ---------------------- ##
    cc = np.all(
        objs[:,1:],
        axis=1)#axis 1 since it is per point not per constraint as is it below 
    
    #print(objs[:,1:],cc)
    #########'
    ######### All or any???? Check for ADMMBO part also
    #########

    extent_tuple = (bounds[0],bounds[1],bounds[0],bounds[1])

    
    func = costf(xy)
    con = np.all(
            np.array([constraintf[i](xy) <= 0  for i in range(len(constraintf))]),
            axis=0) #Boolean constraints
    ff = con*func

    #TODO: Merge both decoupled and not into same flow by creating a full eval_type mask?
    if len(eval_type) == 0:
        decoupled = False
    else:
        decoupled = True

    print("DECOUPLED = ", decoupled)
    
    if decoupled:
        assert len(eval_type) != 0, "Need mask of evaluation type"
        assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
        assert NUM_OF_OBJECTIVE_FUNCTIONS == 1, "Not implemented"

        o1mask = eval_type[:,0]
        class_mask = np.any(eval_type[:,1:],axis=1)

        xs_obj = xs[o1mask]
        xs_class = xs[class_mask]
        cc = cc[class_mask]
        o1_class = [o1mask]
        o1_vals = objs[:,0][o1mask]
    else:
        xs_obj = xs
        xs_class = xs
        o1_vals = objs[:,0]
        
        
    def check_validity(xx):
        cons = np.array([constraintf[i](xx) <= 0 for i in range(len(constraintf))])#Boolean constraint #Upheld if over 0 ??
        return np.all(cons,axis=0)
    
    cc = np.logical_not(check_validity(xs_class)) #TODO: check output of obj_matrix

    
    ### Main Plot ###
    plt.figure()
    plt.imshow(ff.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar()
    plt.plot(xs_obj[:,0],xs_obj[:,1],'kx') #WWhere objective function was evaluated
    for i in range(xs_obj.shape[0]):
        plt.text(xs_obj[i,0],xs_obj[i,1],f'{i}')

    plt.plot(xs_class[:,0][cc>0],xs_class[:,1][cc>0],'r+') #Where classification was attempted but is incorrect
    plt.plot(xs_class[:,0][cc<=0],xs_class[:,1][cc<=0],'ro') #Where it is correct
    ### --------- ###

    #Objfunction plot
    plt.figure()
    sampl_it = np.arange(len(o1_vals))
    valid_it, o1_valid = sampl_it[check_validity(xs_obj)], o1_vals[check_validity(xs_obj)]
    plt.plot(sampl_it, o1_vals,"kx")
    plt.plot(valid_it, o1_valid,"mo")
    if len(o1_valid) > 0:
        curr_min = o1_valid[0]
        rolling_min = np.ones(len(o1_valid))
        for i in range(len(o1_valid)):
            if curr_min > o1_valid[i]:
                curr_min = o1_valid[i]
            rolling_min[i] = curr_min
        plt.plot(valid_it,rolling_min,"r--") #TODO: Fix indexing of valid line to drop off abruplty (per it.)


    
    # Needed for VScode
    plt.show()

def vizualize_toy_problem(problem, points_per_dim = 300):
    xin = np.linspace(bounds[0], bounds[1], points_per_dim)
    if problem["Bound Type"] == "square":
        xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
    else:
        raise Exception("Not yet Implemented non-square bounds")
    
    bounds = problem["Bounds"]
    ubs = bounds[1::2]
    lbs = bounds[0::2]
    
    xins = [np.linspace(lbs[i],ubs[i],points_per_dim) for i in range(len(ubs))] #TODO: Make into per 2 axis plots
    
    ##insert original code-ish here
    
    #TODO: Add global and (maybe) local (maybe) optima to plot. Both constrained and unconstrained

    
    pass
