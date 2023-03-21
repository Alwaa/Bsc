import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

NUM_OF_OBJECTIVE_FUNCTIONS = 1 #Multi-objective optimization is not covered. One should conbine and weigh the multiple objectives into one in thoscases

def vizualize_toy(xs: NDArray[np.float64],
                  objs: NDArray[np.float64], #objective values
                  problem,
                  eval_type: NDArray[np.bool8] = [],
                  decoupled: bool = True):
    num_samples = 301
    ## Fetching problem info ##
    bounds = problem["Bounds"]
    if problem["Bound Type"] == "square":
        xin = np.linspace(bounds[0], bounds[1], num_samples)
        xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
    else:
        raise Exception("Not yet Implemented non-square bounds")
    
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Function (z)"] #TODO: rewrite to multiple (s)
    ## ---------------------- ##
    if len(constraintf) != 1:
        raise NotImplementedError("NO PLOTTING SUPPORT FOR MULTIPLE CONSTRAINTS")
    cc = objs[:,1].flatten()

    extent_tuple = (bounds[0],bounds[1],bounds[0],bounds[1])

    func = costf(xy)
    con = ( constraintf[0](xy) <= 0 )#Boolean constraint
    ff = con*func


    if decoupled:
        assert len(eval_type) != 0, "Need mask of evaluation type"
        assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
        assert NUM_OF_OBJECTIVE_FUNCTIONS == 1, "Not implemented"

        o1mask = eval_type[:,0]
        class_mask = eval_type[:,1] # TODO: Should be a union of all clasifiers

        xs_obj = xs[o1mask]
        xs_class = xs[class_mask]
        cc = cc[class_mask]
        o1_vals = objs[:,0][o1mask]
    else:
        xs_obj = xs
        xs_class = xs
        o1_vals = objs[:,0]
    
    ### Main Plot ###
    plt.figure()
    plt.imshow(ff.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar()
    plt.plot(xs_obj[:,0],xs_obj[:,1],'kx')
    for i in range(xs_obj.shape[0]):
        plt.text(xs_obj[i,0],xs_obj[i,1],f'{i}')

    plt.plot(xs_class[:,0][cc>0],xs_class[:,1][cc>0],'r+') #Where classification was attempted but is incorrect
    plt.plot(xs_class[:,0][cc<=0],xs_class[:,1][cc<=0],'ro') #Where it is correct
    ### --------- ###

    #Objfunction plot
    plt.figure()
    plt.plot(o1_vals,"kx")
    curr_min = o1_vals[0]
    rolling_min = np.ones(len(o1_vals))
    for i in range(len(o1_vals)):
        if curr_min > o1_vals[i]:
            curr_min = o1_vals[i]
        rolling_min[i] = curr_min
    plt.plot(rolling_min,"r--")


    
    # Needed for VScode
    plt.show()



