import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import NDArray
import copy

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
    decoupled = not len(eval_type) == 0

    print("DECOUPLED = ", decoupled)
    
    if decoupled:
        assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
        assert NUM_OF_OBJECTIVE_FUNCTIONS == 1, "Not implemented"

        o1mask = eval_type[:,0]
        class_mask = np.any(eval_type[:,1:],axis=1)

        xs_obj = xs[o1mask]
        xs_class = xs[class_mask]
        cc = cc[class_mask]
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


def expretiment_plot(   exps,
                        problem,
                        single = False,
                        title = "Comparisson plot"):
    ## Fetching problem info ##
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"] #TODO: rewrite to multiple (s)
    ## ---------------------- ##
    def check_validity(xx):
        cons_list = [constraintf[i](xx) <= 0 for i in range(len(constraintf))]
        cons = np.array(cons_list)#Boolean constraint #Upheld if over 0 ??
        return np.all(cons,axis=0)
    

    
    if single:
        comps = {"Algorithm": exp_list}
    else:
        comps = exps
    colors = list(mcolors.TABLEAU_COLORS)
    
    plot_num = 0
    for alg_name, exp_list in comps.items():
        num_exp = len(exp_list)
        decoupled = not len(exp_list[0][2]) == 0
        roll_min_list = []
        max_roll_min_len = 0
        plot_iters = []
        for xs, objs, eval_type in exp_list:
            if decoupled: 
                assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
                assert NUM_OF_OBJECTIVE_FUNCTIONS == 1, "Not implemented"

                o1mask = eval_type[:,0]
                class_mask = np.any(eval_type[:,1:],axis=1)
                xs_class = xs[class_mask]
            else:
                o1mask = np.full(len(xs),True)
                xs_class = xs

            xs_obj = xs[o1mask]
            o1_vals = objs[:,0][o1mask]

            #Objfunction rolling min
            sampl_it = np.arange(len(o1_vals))
            total_it = np.arange(len(xs))
            o1_valid =  o1_vals[check_validity(xs_obj)]
            valid_it = sampl_it[check_validity(xs_obj)]
            tot_valid_it = total_it[
                np.logical_and(check_validity(xs), o1mask)
                                    ]
            tot_roll_min = np.full(len(total_it), np.nan)
            smpl_roll_min = np.full(len(sampl_it), np.nan)
            # print(tot_valid_it,exp_list[0][2])

            for it in range(len(o1_valid)):
                tot_roll_min[tot_valid_it[it]] = o1_valid[it]

            #Rolling min
            roll_min_len = len(tot_roll_min)
            for i in range(1,roll_min_len):
                if tot_roll_min[i-1] < tot_roll_min[i] or np.isnan(tot_roll_min[i]):
                    tot_roll_min[i] = tot_roll_min[i-1]

            roll_min_list.append(tot_roll_min)
            if max_roll_min_len < roll_min_len:
                max_roll_min_len = roll_min_len
                plot_iters = total_it
        
        #Prob overkill and not needed
        for i in range(len(roll_min_list)):
            if len(roll_min_list[i]) < max_roll_min_len:
                roll_min_list[i] = np.concatenate(
                    (roll_min_list[i],np.full(max_roll_min_len - len(roll_min_list[i]) ,np.nan))
                )
            elif len(roll_min_list[i] > max_roll_min_len): 
                roll_min_list[i] = copy.deepcopy(roll_min_list[i][:max_roll_min_len])
        
        #TODO: Investigate ficing last part of plot ticking up...
        
        if not decoupled:
            plot_iters *= (1 + len(constraintf))
        
        arr_roll_mins = np.array(roll_min_list)
        stds = np.nanstd(arr_roll_mins, axis=0)
        means = np.nanmean(arr_roll_mins, axis=0)
        #plt.errorbar(plot_iters, means,yerr=stds)
        plt.plot(plot_iters,means, color = colors[plot_num], label = alg_name)
        plt.fill_between(plot_iters, means - stds, means + stds,
                    color=colors[plot_num], alpha=0.2)
    
        plot_num += 1
    
    plt.title(title)
    plt.legend()