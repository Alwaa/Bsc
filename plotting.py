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
    if len(constraintf) != 1:
        raise NotImplementedError("NO PLOTTING SUPPORT FOR MULTIPLE CONSTRAINTS")
    cc = objs[:,1].flatten()

    extent_tuple = (bounds[0],bounds[1],bounds[0],bounds[1])

    check_validity = lambda xx: constraintf[0](xx) <= 0 #Boolean constraint #Upheld if over 0 ??
    func = costf(xy)
    con = ( constraintf[0](xy) <= 0 )
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
        class_mask = eval_type[:,1] # TODO: Should be a union of all clasifiers

        xs_obj = xs[o1mask]
        xs_class = xs[class_mask]
        cc = cc[class_mask]
        o1_class = [o1mask]
        o1_vals = objs[:,0][o1mask]
    else:
        xs_obj = xs
        xs_class = xs
        o1_vals = objs[:,0]
    
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



